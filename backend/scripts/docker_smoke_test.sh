#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"
QUERY='Can I drive after drinking 180ml whisky?'
CONTAINER_NAME="${CONTAINER_NAME:-alcohol-intelligence-api-local}"

run_local_smoke() {
  curl -sf "$API_URL/health" >/tmp/docker_health.json
  curl -sf -X POST "$API_URL/ask" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"$QUERY\",\"debug\":false}" >/tmp/docker_ask.json
}

run_container_smoke() {
  docker exec "$CONTAINER_NAME" /bin/sh -lc \
    "curl -sf http://127.0.0.1:8000/health" >/tmp/docker_health.json
  docker exec "$CONTAINER_NAME" /bin/sh -lc \
    "curl -sf -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{\"query\":\"$QUERY\",\"debug\":false}'" >/tmp/docker_ask.json
}

if ! run_local_smoke; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Health check failed at $API_URL and docker fallback unavailable."
    exit 1
  fi
  run_container_smoke
fi

python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path('/tmp/docker_ask.json').read_text())
answer = str(payload.get('answer', '')).lower()

def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)

assert_true(bool(payload.get('answer')), 'answer is empty')
assert_true('safe to drive' not in answer or answer.startswith("i can’t tell you that you are safe to drive") or answer.startswith("i can't tell you that you are safe to drive"), 'unsafe safe-to-drive phrasing detected')
assert_true('neo4j' not in answer, 'internal term leaked: neo4j')
assert_true('weaviate' not in answer, 'internal term leaked: weaviate')
assert_true('pbpk' not in answer, 'internal term leaked: pbpk')
assert_true('do not drive' in str(payload.get('driving_guidance', '')).lower(), 'missing do-not-drive guidance')
print('docker smoke test passed')
PY
