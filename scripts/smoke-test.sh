#!/usr/bin/env bash
# Smoke-test a running ASR service (auth, formats, multilingual samples).
#
#   ASR_API_KEY=xxx bash scripts/smoke-test.sh [http://127.0.0.1:8001]
set -uo pipefail

BASE="${1:-http://127.0.0.1:8001}"
KEY="${ASR_API_KEY:?export ASR_API_KEY=<your token>}"
SAMPLES="$(cd "$(dirname "$0")/.." && pwd)/samples"
pass=0; fail=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
code() { curl -sS -o /tmp/asr-smoke.body -w '%{http_code}' "$@"; }

echo "== health (no auth) =="
c=$(code "$BASE/health")
if [[ "$c" == 200 ]] && grep -q '"status":"ok"' /tmp/asr-smoke.body; then ok "health 200"; else bad "health -> $c"; fi

echo "== auth =="
c=$(code -X POST "$BASE/v1/audio/transcriptions" -F "file=@$SAMPLES/zh.wav")
[[ "$c" == 401 ]] && ok "no token -> 401" || bad "no token -> $c (expected 401)"
c=$(code -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer wrong" -F "file=@$SAMPLES/zh.wav")
[[ "$c" == 401 ]] && ok "wrong token -> 401" || bad "wrong token -> $c"
c=$(code -X POST "$BASE/v1/audio/transcriptions" -H "X-API-Key: $KEY" -F "file=@$SAMPLES/zh.wav" -F response_format=text)
[[ "$c" == 200 ]] && ok "X-API-Key -> 200: $(head -c 60 /tmp/asr-smoke.body)" || bad "X-API-Key -> $c"

echo "== formats =="
for fmt in json text verbose_json srt vtt; do
  c=$(code -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer $KEY" \
        -F "file=@$SAMPLES/zh.wav" -F "response_format=$fmt")
  [[ "$c" == 200 ]] && ok "response_format=$fmt" || bad "response_format=$fmt -> $c"
done
c=$(code -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer $KEY" \
      -F "file=@$SAMPLES/zh.wav" -F response_format=bogus)
[[ "$c" == 400 ]] && ok "bad format -> 400" || bad "bad format -> $c"

echo "== languages =="
for w in zh en ja ko yue; do
  c=$(code -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer $KEY" \
        -F "file=@$SAMPLES/$w.wav" -F response_format=verbose_json)
  lang=$(python3 -c "import json;print(json.load(open('/tmp/asr-smoke.body'))['language'])" 2>/dev/null || echo "?")
  text=$(python3 -c "import json;print(json.load(open('/tmp/asr-smoke.body'))['text'][:40])" 2>/dev/null || echo "?")
  [[ "$c" == 200 && "$lang" == "$w" ]] && ok "$w -> lang=$lang: $text" || bad "$w -> $c lang=$lang"
done

echo
echo "passed=$pass failed=$fail"
(( fail == 0 ))
