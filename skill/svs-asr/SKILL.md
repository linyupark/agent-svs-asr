---
name: svs-asr
description: 调用自建 SenseVoice ASR 服务做语音转文字 / 生成字幕（HTTP + Bearer token）。当用户要求把音频或视频转成文字、做会议记录/播客转录、生成 SRT/VTT 字幕、或问「这段音频说了什么」时使用。需要服务已部署并可从当前环境访问。
---

# svs-asr skill

把本地音频/视频交给自建的 SenseVoice-Small 服务（OpenAI 兼容接口）转写。
服务地址与 token 在 `config.json` 里，**先读它，不要凭记忆猜地址**。

## 配置

把 `config.example.json` 复制成同目录的 `config.json` 再填自己的值（`config.json` 已被 git 忽略，不会把 token 提交上去）：

```bash
cp config.example.json config.json
```

```json
{ "base_url": "http://192.168.1.58:8001", "api_key": "<ASR_API_KEY>", "model": "sensevoice" }
```

也可以用环境变量：`ASR_BASE_URL` / `ASR_API_KEY` / `ASR_MODEL` / `ASR_TIMEOUT`（默认 900 s），优先级高于 `config.json`。

- 返回 401 → token 错；连接被拒 → 服务没起。直接告诉用户，不要尝试猜别 token 或换地址。
- 首次请求若服务空闲超过 15 分钟，模型已卸载，会多花约 1 s 重载，属正常。

## 用法

`asr.py` 是零依赖 CLI（仅标准库），自动读 `config.json`：

```bash
# 0. 先确认服务活着（顺带看内存/并发占用）
python asr.py health

# 1. 普通转写：stdout 只有文本，stderr 一行耗时
python asr.py transcribe meeting.mp3

# 2. 生成字幕文件（SRT / VTT，按标点+时长自动切行）
python asr.py transcribe video.mp4 --format srt --out video.srt
python asr.py transcribe video.mp4 --format vtt --out video.vtt

# 3. 强制语种（默认 auto，模型自己判）
python asr.py transcribe en.mp3 --language en --format text

# 4. 需要时间戳：段级 / 词级
python asr.py transcribe a.wav --format verbose_json
python asr.py transcribe a.wav --format verbose_json --words

# 5. 机器可读输出（含 request_id、音频秒数、推理秒数、rtf）
python asr.py transcribe a.wav --json
```

参数：`--format json|text|verbose_json|srt|vtt`、`--language auto|zh|en|ja|ko|yue`、
`--out FILE`、`--words`、`--json`、`--quiet`。

## 直接打接口（不想要 CLI 时）

```bash
curl -s -X POST "$BASE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F file=@audio.mp3 -F response_format=text
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://192.168.1.58:8001/v1", api_key="<token>")
print(c.audio.transcriptions.create(model="sensevoice", file=open("a.mp3", "rb")).text)
```

字段：`file`、`model`（仅回显）、`language`、`response_format`、`timestamp_granularities=word`（配
`verbose_json` 才返回 `words[]`）。响应头 `X-Request-Id`、`X-Audio-Seconds`、`X-Inference-Seconds`、`X-Rtf`。

## 使用建议（重要）

- **长音频先转 16 kHz 单声道**，省一半以上上传时间和解码 CPU：
  `ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav`
- **并发上限 3**：同时发 4 个以上请求，排队超过 `ASR_QUEUE_TIMEOUT`（默认 120 s）会收到
  `503 server_busy` + `Retry-After: 10`。批量任务请串行，或最多 3 路并跑，遇 503 等 10 s 重试。
- **字幕行宽**：默认每行 ≤34 字、≤9 s、段间 0.5 s 内合并。要改行宽只能改服务端环境变量
  `ASR_CUE_MAX_CHARS` / `ASR_CUE_MAX_SEC`（改完 `docker compose up -d`）。
- **时长上限**：默认 6 小时、300 MB；超了先切片再分别转写。
- **质量预期**：中文标点、数字/单位归一化由模型自带 ITN 处理；多人重叠、强噪声、方言口音会影响准确率。
  需要「谁说了什么」的说话人分离，本服务不做（只有文本+时间戳）。
- **隐私**：音频会离开当前机器到自建服务（局域网），不要把隐私音频发到公网地址。
- 语音**合成**请用 kokoro-tts skill，本 skill 只做识别。

## 批量示例

```bash
for f in ./audio/*.mp3; do
  echo "== $f"
  python asr.py transcribe "$f" --format text --quiet || echo "failed: $f"
  sleep 1
done
```

## 排错

| 现象 | 处理 |
|---|---|
| `connect refused` / 超时 | `docker compose -f asr/compose.yml ps` 看容器状态，或 `python asr.py health` |
| `HTTP 401` | `config.json` 的 `api_key` 与 `.env` 的 `ASR_API_KEY` 不一致 |
| `HTTP 503 server_busy` | 并发满了，等 `Retry-After` 秒重试或串行 |
| `HTTP 413` | 文件过大/音频过长，先转码或切片 |
| 输出乱码/空 | 确认文件确实有音轨：`ffprobe -show_streams file` |
