#!/usr/bin/env bash
# Container entrypoint: make sure the model files exist, then serve.
set -euo pipefail

MODEL_DIR="${ASR_MODEL_DIR:-/models}"
missing=0
for f in "${MODEL_DIR}/${ASR_MODEL_FILE:-model.int8.onnx}" \
         "${MODEL_DIR}/${ASR_TOKENS_FILE:-tokens.txt}" \
         "${ASR_VAD_FILE:-${MODEL_DIR}/silero_vad.onnx}"; do
  [[ -s "$f" ]] || { echo "[entrypoint] missing: $f"; missing=1; }
done

if (( missing )); then
  if [[ "${ASR_AUTO_DOWNLOAD:-1}" == "0" ]]; then
    echo "[entrypoint] models missing and ASR_AUTO_DOWNLOAD=0 -> run scripts/fetch-models.sh on the host" >&2
    exit 2
  fi
  echo "[entrypoint] downloading models (~230 MB, one time) ..."
  bash /app/scripts/fetch-models.sh
fi

exec python /app/app/server.py
