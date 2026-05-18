#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://localhost:8000/health}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"

echo "Checking backend: $BACKEND_URL"
if curl -fsS "$BACKEND_URL" >/tmp/soberscope_backend_health.json; then
  echo "backend_status=ok"
else
  echo "backend_status=offline"
fi

echo "Checking frontend: $FRONTEND_URL"
if curl -fsS "$FRONTEND_URL" >/tmp/soberscope_frontend_home.html; then
  echo "frontend_status=ok"
else
  echo "frontend_status=offline"
fi

echo "Done."
