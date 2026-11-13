# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Override for a faster mirror, e.g. https://mirrors.aliyun.com/pypi/simple
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ASR_MODEL_DIR=/models \
    ASR_PORT=8000

# ffmpeg: decode mp3/m4a/mp4/opus/... into 16 kHz mono wav
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY app ./app
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

EXPOSE 8000

# first start: fetch models into /models when missing, then run the API
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
