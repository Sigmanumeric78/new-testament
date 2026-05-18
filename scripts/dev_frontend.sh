#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/frontend"

if command -v lsof >/dev/null 2>&1; then
  if lsof -i :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 5173 is already in use. Stop that process or run with a different allowed CORS origin."
    exit 1
  fi
elif command -v ss >/dev/null 2>&1; then
  if ss -ltn '( sport = :5173 )' | grep -q ':5173'; then
    echo "Port 5173 is already in use. Stop that process or run with a different allowed CORS origin."
    exit 1
  fi
fi

exec npm run dev
