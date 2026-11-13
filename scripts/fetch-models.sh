#!/usr/bin/env bash
# Fetch the SenseVoice-Small int8 ONNX model + Silero VAD into $ASR_MODEL_DIR.
#
#   bash scripts/fetch-models.sh            # uses /models (or ASR_MODEL_DIR)
#   ASR_MODEL_DIR=./models bash scripts/fetch-models.sh
#
# Mirrors are tried in order; partial downloads resume (-C -). Re-running is cheap:
# files that already match the expected size are skipped.
set -euo pipefail

MODEL_DIR="${ASR_MODEL_DIR:-/models}"
REPO="${ASR_MODEL_REPO:-csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17}"
MIRROR="${ASR_HF_MIRROR:-https://hf-mirror.com}"
GH="${ASR_GH_MIRROR:-https://github.com}"
VAD_REPO="${ASR_VAD_REPO:-csukuangfj/vad}"

mkdir -p "${MODEL_DIR}"

log() { printf '[fetch-models] %s\n' "$*"; }

# fetch <filename> <min-bytes> <url>...
fetch() {
  local name="$1" min="$2"; shift 2
  local dest="${MODEL_DIR}/${name}"
  if [[ -f "${dest}" ]] && (( $(stat -c%s "${dest}") >= min )); then
    log "ok ${name} ($(du -h "${dest}" | cut -f1)) — already present"
    return 0
  fi
  local tmp="${dest}.part"
  for url in "$@"; do
    log "downloading ${name} <- ${url}"
    if curl -fL --retry 3 --retry-delay 3 --connect-timeout 20 -C - -o "${tmp}" "${url}"; then
      if (( $(stat -c%s "${tmp}" 2>/dev/null || echo 0) >= min )); then
        mv -f "${tmp}" "${dest}"
        log "ok ${name} ($(du -h "${dest}" | cut -f1))"
        return 0
      fi
      log "!! ${name} too small, trying next mirror"
      rm -f "${tmp}"
    else
      log "!! download failed, trying next mirror"
    fi
  done
  log "FAILED: could not download ${name}"
  return 1
}

fetch model.int8.onnx 200000000 \
  "${MIRROR}/${REPO}/resolve/main/model.int8.onnx" \
  "https://huggingface.co/${REPO}/resolve/main/model.int8.onnx"

fetch tokens.txt 100000 \
  "${MIRROR}/${REPO}/resolve/main/tokens.txt" \
  "https://huggingface.co/${REPO}/resolve/main/tokens.txt"

fetch silero_vad.onnx 500000 \
  "${MIRROR}/${VAD_REPO}/resolve/main/silero_vad.onnx" \
  "${GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

log "models ready in ${MODEL_DIR}:"
ls -lh "${MODEL_DIR}"/model.int8.onnx "${MODEL_DIR}"/tokens.txt "${MODEL_DIR}"/silero_vad.onnx
