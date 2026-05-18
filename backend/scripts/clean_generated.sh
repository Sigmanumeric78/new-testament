#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$BACKEND_DIR/.." && pwd)"
if [[ ! -d "$REPO_ROOT/backend" ]]; then
  REPO_ROOT="$BACKEND_DIR"
fi

cd "$REPO_ROOT"

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache backend/.pytest_cache

rm -f data/interim/reasoning/app_cli_run_log.jsonl
rm -f backend/data/interim/reasoning/app_cli_run_log.jsonl

echo "clean_generated: completed"
