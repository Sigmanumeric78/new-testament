#!/usr/bin/env bash
set -euo pipefail

# Remove Python caches
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

# Remove transient local run log
rm -f data/interim/reasoning/app_cli_run_log.jsonl

echo "clean_generated: completed"
