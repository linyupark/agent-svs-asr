# 本地 ASR 服务（SenseVoice-Small · CPU · Docker Compose）

给 AMD Ryzen 7 5825U（Zen3，8C/16T，**无 GPU / 无 NPU**）准备的纯 CPU 语音识别服务：
OpenAI 兼容的 `/v1/audio/transcriptions`，带 Bearer 认证、并发上限、空闲自动卸载模型。
实测常驻内存 **≈ 390 MB**（解码 2 分钟音频峰值 ≈ 475 MB），离 2 GB 上限有很大余量。

```
docker compose up -d --build     # 首次会自动下载模型（约 230 MB）
curl -H "Authorization: Bearer $ASR_API_KEY" http://127.0.0.1:8001/health
```

---

## 1. 为什么是这个方案

需求里最关键的两条是「内存 ≤ 2 GB」和「纯 CPU」。常见候选的实测/公开数据：

| 方案 | 模型 | 内存 | 中文效果 | 结论 |
|---|---|---|---|---|
| **本方案** sherpa-onnx + SenseVoice-Small int8 | 229 MB ONNX | **390–475 MB** | 好，自带标点，中英日韩粤 | ✅ 选它 |
| FunASR 官方 `funasr-server`（torch） | SenseVoiceSmall fp32 | ≈ 1.5 GB | 好 | 逼近上限，放弃 |
| `falconia/sense-voice-openai-api` 等第三方镜像 | torch 封装 | 1.5 GB+ | 好 | 无认证、偏重 |
| Speaches + faster-whisper `medium` | CT2 int8 | ≈ 1.2 GB | 中文一般（缺标点、易漏字） | 仅作对照 |
| whisper `large-v3` | — | ≈ 2.5 GB | 好 | 超预算 |

**SenseVoice-Small** 是阿里 FunASR 的多语种非自回归模型：一次前向出整句，30 秒音频百毫秒级，
自带中文标点与 ITN（数字/单位归一化），中英日韩粤自动判语种。它的 ONNX int8 版本只有 229 MB，
用 **sherpa-onnx**（k2-fsa 的 C++ 推理内核，pip 轮子自带原生库，**不依赖 torch / onnxruntime**）加载，
内存只有 torch 路线的 1/3，且长音频靠 Silero VAD 切段，2 小时播客也不会爆内存。

实测（本机 16 核容器、4 线程，仅作量级参考；5825U 上大约慢 1.5–2 倍）：

| 场景 | 音频时长 | 推理耗时 | RTF |
|---|---|---|---|
| 短句（整段单次解码） | 3.6 s | 0.09 s | 0.024 |
| 长音频（VAD 分段） | 145.7 s | 4.8 s | 0.033 |
| 3 路并发（同上长音频） | 145.7 s ×3 | 7.3 s each | 0.050 |

内存曲线：启动加载 385 MB → 长音频峰值 475 MB → 空闲卸载后 **104 MB** → 再次请求 1 s 内重载。

---

## 2. 快速开始

```bash
cd asr
cp .env.example .env
sed -i "s/change-me-please/$(openssl rand -hex 16)/" .env   # 生成随机 token
docker compose up -d --build
docker compose logs -f asr
```

首次启动会调用 `scripts/fetch-models.sh` 把模型下载到 `./models/`（约 230 MB，
默认走 hf-mirror，失败自动回退 huggingface.co）。想先下好再启动：

```bash
bash scripts/fetch-models.sh          # 宿主机需要 curl
# 或者用镜像里的脚本：
docker compose run --rm --entrypoint bash asr /app/scripts/fetch-models.sh
```

### 模型：不入库、不打进镜像、只下一次

- 模型**不在 git 里**：`.gitignore` 有 `models/`、`*.onnx`，仓库只有 21 个文本/脚本文件 + 5 个测试样本。
- 模型**也不在镜像里**：`Dockerfile` 只装 python + ffmpeg + 5 个 pip 包，`docker build` 离线就能过，不碰网络下模型。
- 模型放在宿主机 `./models/`，bind mount 进容器。`docker compose build`、`up -d`、`down && up`、换镜像重建
  都复用同一份文件，**不会重复下载**。
- 唯一会下载的时机：三个文件不齐 且 `ASR_AUTO_DOWNLOAD=1`（默认）时，容器**启动那一次**下 230 MB。
  离线/内网部署就把 `ASR_AUTO_DOWNLOAD=0` 写进 `.env`，再把 `models/` 整个拷过去。

```
models/
├── model.int8.onnx   # 228 MB  SenseVoice-Small int8
├── tokens.txt        # 309 KB
└── silero_vad.onnx   # 629 KB
```

换盘或换目录：改 `compose.yml` 里 `- ./models:/models` 的左边；想用 docker 卷就换成
`- asr-models:/models`，并在顶层声明 `volumes: { asr-models: {} }`（迁机器时不如直接拷目录省事）。

验证：

```bash
source .env
curl -s http://127.0.0.1:8001/health | jq
curl -s -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F file=@../tts-test-params.wav
# {"text":"第一句第二句第三句。"}
```

一键自检（认证、5 种输出格式、5 个语种样本，共 15 项）：

```bash
ASR_API_KEY=xxx bash scripts/smoke-test.sh http://127.0.0.1:8001
```

> 参考实现已在 py3.11 + sherpa-onnx 2.0.1 上跑通全部 15 项检查（2026-11）。

---

## 3. 接口

所有 `/v1/*` 端点都要认证，两种写法等价：

```
Authorization: Bearer <ASR_API_KEY>
X-API-Key: <ASR_API_KEY>
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 免认证。含 `rss_mb`、`model_loaded`、`inflight`、`loads/unloads` |
| GET | `/v1/models` | OpenAI 风格模型列表 |
| POST | `/v1/audio/transcriptions` | multipart 上传，字段见下 |
| GET | `/docs` | Swagger UI |

`/v1/audio/transcriptions` 表单字段（均兼容 OpenAI SDK）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `file` | — | 音频/视频文件，ffmpeg 能解的都行（mp3/m4a/wav/mp4/flac/opus…） |
| `model` | `sensevoice` | 仅回显，不影响行为 |
| `language` | `auto` | `auto/zh/en/ja/ko/yue`；SenseVoice 本身会逐句判语种 |
| `response_format` | `json` | `json` / `text` / `verbose_json` / `srt` / `vtt` |
| `prompt` | 空 | 兼容字段，暂不参与解码 |
| `timestamp_granularities` | 空 | 传 `word` 时 `verbose_json` 额外返回 `words[]` |

响应头带 `X-Request-Id`、`X-Audio-Seconds`、`X-Inference-Seconds`、`X-Rtf`，方便做限流/观测。

### 示例

```bash
# 纯文本
curl -s -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F file=@podcast.mp3 -F response_format=text

# 字幕（SRT，按标点+时长切行，每行 <=34 字 / <=9 s）
curl -s -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F file=@video.mp4 -F response_format=srt -o video.srt

# 带 token 级时间戳
curl -s -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F file=@a.wav -F response_format=verbose_json -F timestamp_granularities=word
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="<ASR_API_KEY>")
with open("podcast.mp3", "rb") as f:
    print(client.audio.transcriptions.create(model="sensevoice", file=f).text)
```

长音频建议先转 16 kHz 单声道再上传（省带宽，解码也快）：

```bash
ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav
```

---

## 4. 并发与内存策略

三道闸门，保证「内存永远碰不到 2 GB」：

1. **并发上限**：`ASR_MAX_CONCURRENCY`（默认 3）——同时最多 3 个转写任务，
   其余请求排队；等待超过 `ASR_QUEUE_TIMEOUT`（默认 120 s）直接回 `503 server_busy`
   + `Retry-After: 10`（不会无限堆积把机器拖死）。
   模型只有一份，并发只增加少量 stream/VAD 内存，所以 3 路并发仍然 ≈ 500 MB。
2. **空闲卸载**：`ASR_IDLE_UNLOAD_SEC`（默认 900 s = 15 分钟）——没有请求且没有在跑的任务时，
   释放 recognizer 并 `malloc_trim`，RSS 从 ~390 MB 降到 ~104 MB；
   下一个请求自动重载（约 1 s，请求本身仍然成功）。设为 `0` 可关闭。
3. **容器硬上限**：`mem_limit: 2g` + `memswap_limit: 2g`（不启用 swap，避免抖动），
   真的失控也是容器被 OOM kill，不会拖垮整台机器（`restart: unless-stopped` 会把它拉起来，
   但更建议看 `docker logs asr` 找原因）。

线程数建议：`ASR_NUM_THREADS × ASR_MAX_CONCURRENCY` 略小于物理线程数。
5825U 是 8C/16T，默认 `4 × 3 = 12` 比较稳；纯离线批量处理可把并发降到 1、线程升到 8。

---

## 5. 目录结构

```
asr/
├── compose.yml               # 服务定义（bridge 网络、2 GB 上限、健康检查）
├── Dockerfile                # python:3.11-slim + ffmpeg + 5 个 pip 包
├── requirements.txt
├── .env.example              # 复制成 .env
├── .gitignore / .gitattributes
├── app/server.py             # FastAPI 服务（认证/并发/空闲卸载/字幕）
├── scripts/
│   ├── fetch-models.sh       # 下载 model.int8.onnx + tokens.txt + silero_vad.onnx
│   ├── docker-entrypoint.sh  # 缺模型则自动下载，然后启动
│   └── smoke-test.sh         # 15 项自检（认证/格式/语种）
├── models/                   # 229 MB 模型 + VAD（挂载进容器，不入库）
├── samples/                  # 5 个语种测试样本（zh/en/ja/ko/yue）
└── skill/asr-transcribe/     # 可给 agent 用的 skill（SKILL.md + asr.py + config.example.json）
```

## 5.1 给 agent 用（skill）

`skill/asr-transcribe/` 是一个标准 skill：`SKILL.md` + 零依赖 CLI `asr.py` + `config.example.json`。
装到 DSH 后 agent 就能直接说「把这段录音转成文字」：

```bash
cp -r skill/asr-transcribe /root/.dsh/skills/        # 或放进你 agent 的 skills 目录
cd /root/.dsh/skills/asr-transcribe
cp config.example.json config.json                   # 真正的 config.json 里放 token，已被 git 忽略
$EDITOR config.json                                  # base_url + api_key 填成自己的
python asr.py health
```

---

## 6. 常见问题

**端口/容器名冲突**：容器名固定为 `asr`，默认宿主端口 8001。如果机器上已有别的 ASR 容器占着这两个位置
（比如之前的 Speaches 方案），`docker compose up` 会报端口占用或容器名冲突。先清掉旧容器：

```bash
docker rm -f asr                                            # 按容器名
docker ps -aq --filter publish=8001 | xargs -r docker rm -f  # 或者按占着 8001 的容器
```

也可以改 `.env` 里的 `ASR_HOST_PORT` 换个端口。

**为什么用自定义 bridge 网络**：`networks: asr-net (driver: bridge)` 是 Docker 推荐的 bridge 用法
（自带 DNS、和别的 compose 项目隔离）。若一定要用默认 bridge，把 compose 里的 `networks` 全删掉、
给服务加 `network_mode: bridge`，端口映射照旧可用。

**只允许本机访问**：`.env` 里 `ASR_BIND_IP=127.0.0.1`。

**要不要加 HTTPS**：这个 token 是明文 Bearer，跨公网请在前面套 Nginx/Caddy（或换 `network_mode: host` + 本地反代）。
局域网内直连没问题。

**模型能换吗**：可以。任何 sherpa-onnx 的 SenseVoice/Paraformer/Whisper ONNX 模型把
`ASR_MODEL_FILE` / `ASR_TOKENS_FILE` 指过去即可（非 SenseVoice 模型需要改 `from_sense_voice` 那行构造）。
内存预算内**不**建议换 whisper large-v3。

**没有 GPU 会不会很慢**：SenseVoice-Small 是非自回归模型，30 s 音频在 4 线程上约 0.3–0.6 s；
1 小时播客约 1.5–3 分钟（5825U 上）。比 whisper large 快一个数量级。

**磁盘占用**：镜像约 600 MB（slim + ffmpeg + wheels），模型 230 MB。

---

## 7. 卸载

```bash
docker compose down            # 保留 ./models
docker image rm sensevoice-asr:1.1.0
```
