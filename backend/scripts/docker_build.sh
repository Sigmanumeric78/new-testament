#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$BACKEND_DIR/.." && pwd)"
if [[ ! -d "$REPO_ROOT/backend" ]]; then
  REPO_ROOT="$BACKEND_DIR"
fi

cd "$REPO_ROOT"

docker build -t alcohol-intelligence-api:local -f backend/Dockerfile .
