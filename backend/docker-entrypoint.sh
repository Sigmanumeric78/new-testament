#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESTORE_ARTIFACTS_ON_STARTUP="${RESTORE_ARTIFACTS_ON_STARTUP:-false}"
ARTIFACT_RESTORE_MODE="${ARTIFACT_RESTORE_MODE:-background}"
ARTIFACT_RELEASE="${ARTIFACT_RELEASE:-v0.6-chemical-explorer}"
ARTIFACT_STORE_BACKEND="${ARTIFACT_STORE_BACKEND:-supabase}"
SUPABASE_ARTIFACT_BUCKET="${SUPABASE_ARTIFACT_BUCKET:-alcohol-intelligence-artifacts}"
RESTORE_WORKSPACE_DIR="${RESTORE_WORKSPACE_DIR:-/tmp/artifact_restore/${ARTIFACT_RELEASE}}"
PYTHONPATH="${PYTHONPATH:-/app/backend}"

export PROJECT_ROOT
export DATA_ROOT
export PYTHONPATH
export RESTORE_WORKSPACE_DIR
export RESTORE_ARTIFACTS_ON_STARTUP
export ARTIFACT_RESTORE_MODE
export ARTIFACT_RELEASE
export ARTIFACT_STORE_BACKEND
export SUPABASE_ARTIFACT_BUCKET

mkdir -p "${DATA_ROOT}"
if [[ -f "/app/data/artifact_manifest.example.json" ]]; then
  cp -f "/app/data/artifact_manifest.example.json" "${DATA_ROOT}/artifact_manifest.example.json"
fi

if [[ "${RESTORE_ARTIFACTS_ON_STARTUP,,}" == "true" ]]; then
  echo "[artifact-restore] startup restore requested; FastAPI will schedule background restore"
fi

exec "$@"
