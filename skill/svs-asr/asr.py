#!/usr/bin/env python3
"""Zero-dependency CLI for the self-hosted SenseVoice ASR service.

    python asr.py health
    python asr.py transcribe audio.mp3                      # -> plain text
    python asr.py transcribe video.mp4 --format srt --out video.srt
    python asr.py transcribe a.wav --json                   # raw response + headers
    python asr.py models

Config: config.json next to this file (base_url / api_key / model).
Env overrides: ASR_BASE_URL, ASR_API_KEY, ASR_MODEL, ASR_TIMEOUT.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.json"


def load_config() -> dict:
    cfg = {"base_url": "http://127.0.0.1:8001", "api_key": "", "model": "sensevoice"}
    if CONFIG_FILE.is_file():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: cannot parse {CONFIG_FILE}: {exc}", file=sys.stderr)
    cfg["base_url"] = os.getenv("ASR_BASE_URL", cfg.get("base_url", "")).rstrip("/")
    cfg["api_key"] = os.getenv("ASR_API_KEY", cfg.get("api_key", ""))
    cfg["model"] = os.getenv("ASR_MODEL", cfg.get("model", "sensevoice"))
    cfg["timeout"] = float(os.getenv("ASR_TIMEOUT", "900"))
    if not cfg["api_key"]:
        die("no api_key: set it in config.json or export ASR_API_KEY")
    if not cfg["base_url"]:
        die("no base_url: set it in config.json or export ASR_BASE_URL")
    return cfg


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def request(cfg: dict, path: str, *, body: bytes | None = None,
            headers: dict | None = None, method: str | None = None):
    url = cfg["base_url"] + path
    hdrs = {"Authorization": f"Bearer {cfg['api_key']}", "User-Agent": "asr-skill/1.0"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method or ("POST" if body else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
            raw = resp.read()
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:  # noqa: BLE001
            pass
        hint = {401: "check api_key in config.json", 503: "server busy, retry later",
                413: "audio too large/long"}.get(exc.code, "")
        die(f"HTTP {exc.code}: {detail}" + (f" ({hint})" if hint else ""))
    except urllib.error.URLError as exc:
        die(f"cannot reach {cfg['base_url']} ({exc.reason}) — is the service running? "
            f"try: docker compose -f asr/compose.yml up -d")
    except TimeoutError:
        die(f"timed out after {cfg['timeout']:.0f}s (raise ASR_TIMEOUT for very long audio)")


def multipart(fields: dict, file_path: Path) -> tuple[bytes, str]:
    boundary = "----asr" + uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        if value in (None, ""):
            continue
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
    ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{file_path.name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
    out += file_path.read_bytes()
    out += f"\r\n--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def cmd_health(cfg: dict, args) -> int:
    status, headers, raw = request(cfg, "/health")
    data = json.loads(raw or b"{}")
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(f"{data.get('status')} · model={data.get('model')} loaded={data.get('model_loaded')} "
          f"rss={data.get('rss_mb')}MB inflight={data.get('inflight')}/{data.get('max_concurrency')} "
          f"uptime={data.get('uptime_sec')}s")
    return 0


def cmd_models(cfg: dict, args) -> int:
    _, _, raw = request(cfg, "/v1/models")
    data = json.loads(raw or b"{}")
    for m in data.get("data", []):
        print(m.get("id"))
    return 0


def cmd_transcribe(cfg: dict, args) -> int:
    path = Path(args.audio).expanduser()
    if not path.is_file():
        die(f"no such file: {path}")
    size_mb = path.stat().st_size / 1e6
    if size_mb > 300:
        print(f"warning: {size_mb:.0f} MB upload; pre-convert with "
              f"`ffmpeg -i in -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav`", file=sys.stderr)

    fields = {"model": args.model or cfg["model"], "response_format": args.format,
              "language": args.language}
    if args.format == "verbose_json" and args.words:
        fields["timestamp_granularities"] = "word"
    body, ctype = multipart(fields, path)
    t0 = time.time()
    status, headers, raw = request(cfg, "/v1/audio/transcriptions", body=body,
                                   headers={"Content-Type": ctype})
    elapsed = time.time() - t0
    text = raw.decode("utf-8", "replace")

    if args.json:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"text": text}
        payload["_meta"] = {"request_id": headers.get("x-request-id"),
                            "audio_sec": headers.get("x-audio-seconds"),
                            "inference_sec": headers.get("x-inference-seconds"),
                            "rtf": headers.get("x-rtf"), "wall_sec": round(elapsed, 3)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.out:
        out = Path(args.out).expanduser()
        if args.format in ("json", "verbose_json"):
            out.write_text(text, encoding="utf-8")
        else:
            out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        print(f"saved {out} ({out.stat().st_size} bytes)", file=sys.stderr)
        return 0

    if args.format in ("json", "verbose_json"):
        try:
            print(json.loads(text).get("text", ""))
        except json.JSONDecodeError:
            print(text)
    else:
        print(text.rstrip("\n"))
    if not args.quiet:
        print(f"[{headers.get('x-audio-seconds', '?')}s audio, {headers.get('x-inference-seconds', '?')}s "
              f"inference, rtf {headers.get('x-rtf', '?')}, wall {elapsed:.2f}s]",
              file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="asr.py", description="SenseVoice ASR client")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("health", help="service + model status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("models", help="list model ids")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("transcribe", help="transcribe an audio/video file")
    p.add_argument("audio")
    p.add_argument("--format", default="json",
                   choices=["json", "text", "verbose_json", "srt", "vtt"])
    p.add_argument("--language", default="auto", choices=["auto", "zh", "en", "ja", "ko", "yue"])
    p.add_argument("--model", default="")
    p.add_argument("--out", default="", help="write result to file (srt/vtt/text/json)")
    p.add_argument("--words", action="store_true", help="verbose_json: include word timestamps")
    p.add_argument("--json", action="store_true", help="print raw response + _meta")
    p.add_argument("--quiet", action="store_true", help="no timing line on stderr")
    p.set_defaults(func=cmd_transcribe)

    args = ap.parse_args()
    cfg = load_config()
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
