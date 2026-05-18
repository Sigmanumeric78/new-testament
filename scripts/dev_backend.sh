#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/backend"

export PYTHONPATH=.
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
