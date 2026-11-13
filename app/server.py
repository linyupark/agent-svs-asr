#!/usr/bin/env python3
"""OpenAI-compatible ASR server: SenseVoice-Small (int8 ONNX) + Silero VAD on sherpa-onnx.

  GET  /health                    -> liveness + model info        (no auth)
  GET  /v1/models                 -> OpenAI-style model list      (auth)
  POST /v1/audio/transcriptions   -> transcription (multipart)    (auth)

Auth:   `Authorization: Bearer <key>` or `X-API-Key: <key>`  (ASR_API_KEY, comma separated)
Limits: ASR_MAX_CONCURRENCY (default 3) concurrent transcriptions, queue with timeout,
        ASR_IDLE_UNLOAD_SEC (default 900) releases the model after that much idle time.
No torch / no onnxruntime dependency: roughly 400-600 MB RSS while loaded.
"""
from __future__ import annotations

import ctypes
import gc
from contextlib import asynccontextmanager
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from typing import Iterable, List, Optional

import numpy as np
import sherpa_onnx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
# uvicorn only knows a fixed set of level names (no 'warn' / 'fatal'), and an
# unknown value there raises KeyError *before* the server binds the port, which
# looks like a silent crash-loop in `docker compose ps`. Normalise defensively.
_LOG_ALIASES = {"warn": "warning", "err": "error", "fatal": "critical",
                "panic": "critical", "none": "critical"}
_UVICORN_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")
_raw_log_level = (os.getenv("ASR_LOG_LEVEL") or "info").strip().lower()
LOG_LEVEL = _LOG_ALIASES.get(_raw_log_level, _raw_log_level)
_BAD_LOG_LEVEL = "" if LOG_LEVEL in _UVICORN_LEVELS else _raw_log_level
if _BAD_LOG_LEVEL:
    LOG_LEVEL = "info"

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("asr")
if _BAD_LOG_LEVEL:
    log.warning("unknown ASR_LOG_LEVEL=%r, falling back to 'info' "
                "(accepted: %s)", _BAD_LOG_LEVEL, ", ".join(_UVICORN_LEVELS))

MODEL_DIR = os.getenv("ASR_MODEL_DIR", "/models")
MODEL_FILE = os.getenv("ASR_MODEL_FILE", "model.int8.onnx")
TOKENS_FILE = os.getenv("ASR_TOKENS_FILE", "tokens.txt")
VAD_FILE = os.getenv("ASR_VAD_FILE", os.path.join(MODEL_DIR, "silero_vad.onnx"))

NUM_THREADS = int(os.getenv("ASR_NUM_THREADS", "4"))
MAX_CONCURRENCY = max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "3")))
QUEUE_TIMEOUT = float(os.getenv("ASR_QUEUE_TIMEOUT", "120"))
IDLE_UNLOAD_SEC = float(os.getenv("ASR_IDLE_UNLOAD_SEC", "900"))   # 0 disables unloading
API_KEYS = [k.strip() for k in os.getenv("ASR_API_KEY", "").split(",") if k.strip()]
MAX_UPLOAD_MB = int(os.getenv("ASR_MAX_UPLOAD_MB", "300"))
MAX_AUDIO_SEC = float(os.getenv("ASR_MAX_AUDIO_SEC", "21600"))     # 6 h
WHOLE_MAX_SEC = float(os.getenv("ASR_WHOLE_MAX_SEC", "30"))        # single pass below this
DEFAULT_LANGUAGE = os.getenv("ASR_LANGUAGE", "auto")
DEFAULT_ITN = os.getenv("ASR_USE_ITN", "1").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "sensevoice")

VAD_SILENCE = float(os.getenv("ASR_VAD_MIN_SILENCE", "0.5"))
VAD_MIN_SPEECH = float(os.getenv("ASR_VAD_MIN_SPEECH", "0.25"))
VAD_MAX_SPEECH = float(os.getenv("ASR_VAD_MAX_SPEECH", "20.0"))
SEG_PAD_SEC = float(os.getenv("ASR_SEG_PAD", "0.15"))              # context padding per segment

CUE_MAX_CHARS = int(os.getenv("ASR_CUE_MAX_CHARS", "34"))
CUE_MAX_SEC = float(os.getenv("ASR_CUE_MAX_SEC", "9"))
CUE_MERGE_GAP = float(os.getenv("ASR_CUE_MERGE_GAP", "0.5"))

SUPPORTED_LANGS = {"auto", "zh", "en", "ja", "ko", "yue", ""}
SUPPORTED_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}

if not API_KEYS:
    log.error("ASR_API_KEY is empty -> refusing to start without authentication")
    sys.exit(2)

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
TOKENS_PATH = os.path.join(MODEL_DIR, TOKENS_FILE)
for _p in (MODEL_PATH, TOKENS_PATH, VAD_FILE):
    if not os.path.isfile(_p):
        log.error("missing model file %s (run scripts/fetch-models.sh)", _p)
        sys.exit(2)

# --------------------------------------------------------------------------- #
# engine: single model instance, bounded concurrency, idle unloading
# --------------------------------------------------------------------------- #
def _trim_heap() -> None:
    """Give freed pages back to the OS (glibc keeps them otherwise)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


class Engine:
    """Holds the recognizer, caps parallel work and releases the model when idle."""

    def __init__(self) -> None:
        self._rec: Optional["sherpa_onnx.OfflineRecognizer"] = None
        self._lock = threading.Lock()          # guards build/unload
        self._slots = threading.Semaphore(MAX_CONCURRENCY)
        self.inflight = 0
        self.last_used = time.time()
        self.loads = 0
        self.unloads = 0

    # -- model lifecycle -------------------------------------------------- #
    def _build_locked(self) -> "sherpa_onnx.OfflineRecognizer":
        t0 = time.time()
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=MODEL_PATH,
            tokens=TOKENS_PATH,
            num_threads=NUM_THREADS,
            use_itn=DEFAULT_ITN,
            # SenseVoice detects the language per utterance; sherpa-onnx 2.x has no
            # per-stream language switch, so this is a deployment-wide default.
            language="" if DEFAULT_LANGUAGE in ("auto", "") else DEFAULT_LANGUAGE,
            debug=False,
        )
        self.loads += 1
        log.info("model loaded in %.2fs (rss=%d MB, load #%d)", time.time() - t0, rss_mb(), self.loads)
        return rec

    def preload(self) -> None:
        with self._lock:
            if self._rec is None:
                self._rec = self._build_locked()
            self.last_used = time.time()

    def unload(self, reason: str) -> bool:
        with self._lock:
            if self._rec is None or self.inflight > 0:
                return False
            self._rec = None
            self.unloads += 1
        _trim_heap()
        log.info("model unloaded (%s, rss=%d MB)", reason, rss_mb())
        return True

    def get(self) -> "sherpa_onnx.OfflineRecognizer":
        with self._lock:
            if self._rec is None:
                self._rec = self._build_locked()
            return self._rec

    # -- bounded concurrency ---------------------------------------------- #
    def acquire(self, timeout: float = QUEUE_TIMEOUT) -> bool:
        if not self._slots.acquire(timeout=timeout):
            return False
        with self._lock:
            self.inflight += 1
        return True

    def release(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self.last_used = time.time()
        self._slots.release()

    def state(self) -> dict:
        return {"model_loaded": self._rec is not None, "inflight": self.inflight,
                "max_concurrency": MAX_CONCURRENCY, "loads": self.loads, "unloads": self.unloads,
                "idle_unload_sec": IDLE_UNLOAD_SEC,
                "idle_sec": round(time.time() - self.last_used, 1)}


ENGINE = Engine()
STARTED = time.time()


def _idle_watcher() -> None:
    period = max(5.0, min(60.0, IDLE_UNLOAD_SEC / 2 if IDLE_UNLOAD_SEC else 30.0))
    while True:
        time.sleep(period)
        if IDLE_UNLOAD_SEC and ENGINE._rec is not None and ENGINE.inflight == 0:
            if time.time() - ENGINE.last_used >= IDLE_UNLOAD_SEC:
                ENGINE.unload(f"idle > {IDLE_UNLOAD_SEC:.0f}s")


def build_vad() -> "sherpa_onnx.VoiceActivityDetector":
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = VAD_FILE
    cfg.silero_vad.min_silence_duration = VAD_SILENCE
    cfg.silero_vad.min_speech_duration = VAD_MIN_SPEECH
    cfg.silero_vad.max_speech_duration = VAD_MAX_SPEECH
    cfg.sample_rate = 16000
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


# --------------------------------------------------------------------------- #
# fastapi app + auth
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(ENGINE.preload)
    threading.Thread(target=_idle_watcher, name="idle-watcher", daemon=True).start()
    log.info("ready: threads=%d concurrency=%d idle_unload=%s",
             NUM_THREADS, MAX_CONCURRENCY, int(IDLE_UNLOAD_SEC) or "off")
    yield


app = FastAPI(title="SenseVoice ASR", version="1.1.1", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {
        "error": {"message": str(exc.detail), "type": "invalid_request_error"}
    }
    return JSONResponse(status_code=exc.status_code, content=detail, headers=exc.headers)


def _unauthorized(msg: str):
    raise HTTPException(
        status_code=401,
        detail={"error": {"message": msg, "type": "invalid_request_error", "code": "invalid_api_key"}},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def auth(authorization: Optional[str] = Header(default=None),
               x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if not token or token not in API_KEYS:
        _unauthorized("missing or invalid api key")


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def wav_info(path: str) -> Optional[dict]:
    try:
        with wave.open(path, "rb") as w:
            return {"sr": w.getframerate(), "channels": w.getnchannels(),
                    "width": w.getsampwidth(), "frames": w.getnframes()}
    except Exception:
        return None


def normalize_audio(src: str, dst: str) -> float:
    """Decode (almost) any audio/video to 16 kHz mono s16 wav. Returns duration [s]."""
    info = wav_info(src)
    if info and info["sr"] == 16000 and info["channels"] == 1 and info["width"] == 2:
        shutil.copyfile(src, dst)
    else:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src,
               "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
               "-f", "wav", dst]
        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        if proc.returncode != 0:
            lines = [l for l in proc.stderr.decode("utf-8", "replace").strip().splitlines() if l.strip()]
            msg = re.sub(r"/tmp/asr-[^/\s]+/", "<upload>/", " | ".join(lines[-2:]))
            raise ValueError("ffmpeg could not decode the audio: " + msg)
    info = wav_info(dst)
    if not info:
        raise ValueError("decoded audio is not readable PCM wav")
    return info["frames"] / float(info["sr"])


def iter_chunks(path: str, window: int = 512, block: float = 1.0) -> Iterable[np.ndarray]:
    """Yield 32 ms windows: sherpa-onnx's VAD mis-segments speech when fed big chunks."""
    with wave.open(path, "rb") as w:
        n = int(w.getframerate() * block)
        while True:
            raw = w.readframes(n)
            if not raw:
                return
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            for i in range(0, len(samples), window):
                yield samples[i:i + window]


def read_all(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def decode_samples(rec, samples: np.ndarray) -> dict:
    st = rec.create_stream()
    st.accept_waveform(16000, samples)
    rec.decode_stream(st)
    r = st.result
    return {
        "text": (r.text or "").strip(),
        "tokens": list(getattr(r, "tokens", []) or []),
        "timestamps": [float(x) for x in (getattr(r, "timestamps", []) or [])],
        "lang": (getattr(r, "lang", "") or "").strip("<|>"),
    }


def _emit(rec, pending: dict, segments: List[dict]) -> None:
    samples = np.concatenate(pending["chunks"]) if pending["chunks"] else np.zeros(0, np.float32)
    if len(samples) < 1600:
        return
    r = decode_samples(rec, samples)
    if not r["text"]:
        return
    segments.append({
        "start": round(pending["start"], 3),
        "end": round(pending["start"] + len(samples) / 16000.0, 3),
        "text": r["text"],
        "tokens": r["tokens"],
        "timestamps": [round(t, 3) for t in r["timestamps"]],
        "lang": r["lang"],
    })


def transcribe_whole(rec, path: str, duration: float) -> List[dict]:
    """Short clip: one pass over the full audio (no VAD boundary artefacts)."""
    r = decode_samples(rec, read_all(path))
    if not r["text"]:
        return []
    return [{"start": 0.0, "end": round(duration, 3), "text": r["text"], "tokens": r["tokens"],
             "timestamps": [round(t, 3) for t in r["timestamps"]], "lang": r["lang"]}]


def transcribe_vad(rec, path: str) -> List[dict]:
    """Long audio: Silero VAD segmentation with ±SEG_PAD_SEC context padding."""
    vad = build_vad()
    pad_n = max(0, int(SEG_PAD_SEC * 16000))
    segments: List[dict] = []
    pending: List[dict] = []
    tail = np.zeros(0, dtype=np.float32)

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            vad.pop()
            chunks = []
            if pad_n and seg.start > 0 and tail.size:
                chunks.append(tail[-min(pad_n, tail.size):].copy())
            chunks.append(seg.samples)
            pending.append({"start": max(0.0, seg.start - pad_n) / 16000.0,
                            "chunks": chunks, "need": pad_n})

    def advance(chunk: np.ndarray) -> None:
        keep = []
        for p in pending:
            if p["need"] > 0:
                take = chunk[: p["need"]]
                p["chunks"].append(take)
                p["need"] -= len(take)
            if p["need"] <= 0:
                _emit(rec, p, segments)
            else:
                keep.append(p)
        pending[:] = keep

    for chunk in iter_chunks(path):
        vad.accept_waveform(chunk)
        drain()
        advance(chunk)
        if pad_n:
            tail = np.concatenate([tail, chunk])[-pad_n:]
    vad.flush()
    drain()
    if pending:
        advance(np.zeros(pad_n + 1, dtype=np.float32))
    return segments


# --------------------------------------------------------------------------- #
# cue shaping
# --------------------------------------------------------------------------- #
CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
SENT_SPLIT = re.compile(r"(?<=[。！？!?；;…])")
STRONG_PUNCT = "。！？!?；;…"


def join_texts(parts: List[str]) -> str:
    out = ""
    for p in parts:
        if not p:
            continue
        if out and not CJK.search(out[-1]) and not CJK.search(p[0]) and out[-1] not in "，,、 ":
            out += " "
        out += p
    return out


def cues_from_tokens(seg: dict, duration: float) -> List[dict]:
    """Split one decoded segment into subtitle cues using token timestamps."""
    toks, stamps = seg["tokens"], seg["timestamps"]
    if not toks or not stamps:
        return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}]
    cues, cur, cur_chars, cur_start = [], [], 0, stamps[0]
    for i, tok in enumerate(toks):
        cur.append(tok)
        cur_chars += len(tok)
        nxt = stamps[i + 1] if i + 1 < len(stamps) else max(stamps[i] + 0.08, duration)
        cut = (tok in STRONG_PUNCT and cur_chars >= 6) or cur_chars >= CUE_MAX_CHARS or (nxt - cur_start) >= CUE_MAX_SEC
        if cut:
            text = "".join(cur).strip()
            if text:
                cues.append({"start": round(cur_start, 3), "end": round(nxt, 3), "text": text})
            cur, cur_chars, cur_start = [], 0, nxt
    if cur:
        text = "".join(cur).strip()
        if text:
            cues.append({"start": round(cur_start, 3), "end": round(seg["end"], 3), "text": text})
    return cues


def merge_cues(segments: List[dict]) -> List[dict]:
    cues: List[dict] = []
    for seg in segments:
        if (cues and seg["start"] - cues[-1]["end"] <= CUE_MERGE_GAP
                and len(cues[-1]["text"]) + len(seg["text"]) <= CUE_MAX_CHARS
                and seg["end"] - cues[-1]["start"] <= CUE_MAX_SEC):
            cues[-1]["text"] = join_texts([cues[-1]["text"], seg["text"]])
            cues[-1]["end"] = seg["end"]
        else:
            cues.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
    return [c for c in cues if c["text"].strip()]


def split_long(cues: List[dict]) -> List[dict]:
    out: List[dict] = []
    for cue in cues:
        text = cue["text"]
        if len(text) <= CUE_MAX_CHARS * 1.5:
            out.append(cue)
            continue
        pieces: List[str] = []
        for sent in SENT_SPLIT.split(text):
            if not sent:
                continue
            if len(sent) <= CUE_MAX_CHARS:
                pieces.append(sent)
                continue
            buf = ""
            for ch in sent:
                buf += ch
                if len(buf) >= CUE_MAX_CHARS and ch in "，,、 ":
                    pieces.append(buf)
                    buf = ""
            if buf:
                pieces.append(buf)
        if len(pieces) <= 1:
            out.append(cue)
            continue
        total = sum(len(p) for p in pieces) or 1
        span, t = cue["end"] - cue["start"], cue["start"]
        for p in pieces:
            d = span * len(p) / total
            out.append({"start": round(t, 3), "end": round(t + d, 3), "text": p.strip()})
            t += d
    return [c for c in out if c["text"].strip()]


def srt_ts(sec: float, sep: str = ",") -> str:
    sec = max(0.0, sec)
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def to_srt(cues: List[dict]) -> str:
    return "\n".join(f"{i}\n{srt_ts(c['start'])} --> {srt_ts(c['end'])}\n{c['text']}\n"
                     for i, c in enumerate(cues, 1))


def to_vtt(cues: List[dict]) -> str:
    body = "\n".join(f"{srt_ts(c['start'], '.')} --> {srt_ts(c['end'], '.')}\n{c['text']}\n" for c in cues)
    return "WEBVTT\n\n" + body


def error(message: str, code: str = "invalid_request_error", status: int = 400,
          headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": code, "code": code}},
                        headers=headers)


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    body = {"status": "ok", "service": "sensevoice-asr", "backend": "sherpa-onnx",
            "model": DEFAULT_MODEL_NAME, "model_file": MODEL_FILE, "threads": NUM_THREADS,
            "rss_mb": rss_mb(), "uptime_sec": round(time.time() - STARTED, 1), "auth": "bearer"}
    body.update(ENGINE.state())
    return body


@app.get("/v1/models", dependencies=[Depends(auth)])
async def models():
    return {"object": "list",
            "data": [{"id": DEFAULT_MODEL_NAME, "object": "model", "created": int(STARTED), "owned_by": "local"}]}


@app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
    response_format: str = Form(default="json"),
    prompt: str = Form(default=""),
    timestamp_granularities: str = Form(default=""),
):
    rid = uuid.uuid4().hex[:8]
    fmt = (response_format or "json").lower()
    if fmt not in SUPPORTED_FORMATS:
        return error(f"unsupported response_format '{fmt}', use one of {sorted(SUPPORTED_FORMATS)}")
    if language and language.lower() not in SUPPORTED_LANGS:
        return error(f"unsupported language '{language}', use auto/zh/en/ja/ko/yue")
    lang = "" if (language or DEFAULT_LANGUAGE).lower() in ("auto", "") else (language or DEFAULT_LANGUAGE).lower()

    tmpdir = tempfile.mkdtemp(prefix="asr-")
    suffix = os.path.splitext(file.filename or "")[1][:10] or ".bin"
    raw, wav = os.path.join(tmpdir, "upload" + suffix), os.path.join(tmpdir, "audio.wav")
    try:
        written, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(raw, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    return error(f"file too large (> {MAX_UPLOAD_MB} MB), raise ASR_MAX_UPLOAD_MB",
                                 "file_too_large", 413)
                out.write(chunk)
        if written == 0:
            return error("empty upload")

        try:
            duration = await run_in_threadpool(normalize_audio, raw, wav)
        except subprocess.TimeoutExpired:
            return error("audio decoding timed out", "decode_timeout", 408)
        except ValueError as exc:
            return error(str(exc))
        if duration < 0.1:
            return error("audio is shorter than 0.1 s")
        if duration > MAX_AUDIO_SEC:
            return error(f"audio too long ({duration:.0f}s > {MAX_AUDIO_SEC:.0f}s)", "file_too_long", 413)

        if not await run_in_threadpool(ENGINE.acquire, QUEUE_TIMEOUT):
            return error(f"server busy ({MAX_CONCURRENCY} concurrent jobs), retry later",
                         "server_busy", 503, {"Retry-After": "10"})
        t1 = time.time()
        try:
            rec = ENGINE.get()
            segments = await run_in_threadpool(
                lambda: transcribe_whole(rec, wav, duration) if duration <= WHOLE_MAX_SEC else transcribe_vad(rec, wav)
            )
        except Exception as exc:
            log.exception("inference failed")
            return error(f"inference failed: {exc}", "inference_error", 500)
        finally:
            ENGINE.release()
        took = time.time() - t1
        rtf = took / duration if duration else 0.0

        if segments and duration <= WHOLE_MAX_SEC:
            cues = cues_from_tokens(segments[0], duration)
        else:
            cues = split_long(merge_cues(segments))
        text = (join_texts([s["text"] for s in segments]) if len(segments) > 1
                else (segments[0]["text"] if segments else ""))

        log.info("rid=%s fmt=%s lang=%s audio=%.1fs segs=%d cues=%d took=%.2fs rtf=%.3f inflight=%d",
                 rid, fmt, lang or "auto", duration, len(segments), len(cues), took, rtf, ENGINE.inflight)
        headers = {"X-Request-Id": rid, "X-Audio-Seconds": f"{duration:.3f}",
                   "X-Inference-Seconds": f"{took:.3f}", "X-Rtf": f"{rtf:.3f}"}

        if fmt == "text":
            return PlainTextResponse(text or "", headers=headers)
        if fmt == "srt":
            return PlainTextResponse(to_srt(cues), media_type="text/plain; charset=utf-8", headers=headers)
        if fmt == "vtt":
            return PlainTextResponse(to_vtt(cues), media_type="text/vtt; charset=utf-8", headers=headers)
        if fmt == "verbose_json":
            body = {"task": "transcribe", "model": model or DEFAULT_MODEL_NAME,
                    "language": (segments[0]["lang"] if segments and segments[0]["lang"] else (lang or "auto")),
                    "duration": round(duration, 3), "text": text,
                    "segments": [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"],
                                  "tokens": s["tokens"], "timestamps": s["timestamps"]}
                                 for i, s in enumerate(segments)]}
            if "word" in (timestamp_granularities or ""):
                body["words"] = [{"word": tok, "start": ts,
                                  "end": (s["timestamps"][j + 1] if j + 1 < len(s["timestamps"]) else s["end"])}
                                 for s in segments for j, (tok, ts) in enumerate(zip(s["tokens"], s["timestamps"]))]
            return JSONResponse(body, headers=headers)
        return JSONResponse({"text": text}, headers=headers)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/")
async def index():
    return {"service": "sensevoice-asr", "model": DEFAULT_MODEL_NAME,
            "endpoints": ["/health", "/v1/models", "/v1/audio/transcriptions", "/docs"],
            "auth": "Authorization: Bearer <ASR_API_KEY>"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ASR_PORT", "8000")), log_level=LOG_LEVEL)
