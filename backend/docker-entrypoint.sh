#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESTORE_ARTIFACTS_ON_STARTUP="${RESTORE_ARTIFACTS_ON_STARTUP:-false}"
ARTIFACT_RELEASE="${ARTIFACT_RELEASE:-v0.6-chemical-explorer}"
ARTIFACT_STORE_BACKEND="${ARTIFACT_STORE_BACKEND:-supabase}"
SUPABASE_ARTIFACT_BUCKET="${SUPABASE_ARTIFACT_BUCKET:-alcohol-intelligence-artifacts}"
RESTORE_WORKSPACE_DIR="${RESTORE_WORKSPACE_DIR:-/tmp/artifact_restore/${ARTIFACT_RELEASE}}"
PYTHONPATH="${PYTHONPATH:-/app/backend}"

export PROJECT_ROOT
export DATA_ROOT
export PYTHONPATH
export RESTORE_WORKSPACE_DIR

mkdir -p "${DATA_ROOT}"
if [[ -f "/app/data/artifact_manifest.example.json" ]]; then
  cp -f "/app/data/artifact_manifest.example.json" "${DATA_ROOT}/artifact_manifest.example.json"
fi

if [[ "${RESTORE_ARTIFACTS_ON_STARTUP,,}" == "true" ]]; then
  echo "[artifact-restore] restore started"
  echo "[artifact-restore] release=${ARTIFACT_RELEASE}"
  echo "[artifact-restore] backend=${ARTIFACT_STORE_BACKEND}"
  echo "[artifact-restore] project_root=${PROJECT_ROOT}"
  echo "[artifact-restore] data_root=${DATA_ROOT}"

  if [[ "${ARTIFACT_STORE_BACKEND,,}" == "mongodb" ]]; then
    mongo_output_root="${MONGODB_RESTORE_OUTPUT_ROOT:-}"
    if [[ -z "${mongo_output_root}" ]]; then
      if [[ "$(basename "${DATA_ROOT}")" == "data" ]]; then
        mongo_output_root="$(dirname "${DATA_ROOT}")"
      else
        mongo_output_root="${PROJECT_ROOT}"
      fi
    fi

    download_output="$(python3 /app/backend/scripts/artifact_download_mongodb.py \
      --release "${ARTIFACT_RELEASE}" \
      --output-root "${mongo_output_root}" \
      --required-only \
      --force)"
    echo "${download_output}"
  elif [[ "${ARTIFACT_STORE_BACKEND,,}" == "supabase" ]]; then
    echo "[artifact-restore] bucket=${SUPABASE_ARTIFACT_BUCKET}"
    download_output="$(python3 /app/backend/scripts/artifact_download_supabase.py \
      --release "${ARTIFACT_RELEASE}" \
      --execute \
      --overwrite \
      --runtime-only \
      --workspace-dir "${RESTORE_WORKSPACE_DIR}")"
    echo "${download_output}"

    verify_output="$(python3 /app/backend/scripts/artifact_verify_release.py \
      --release "${ARTIFACT_RELEASE}" \
      --manifest "${RESTORE_WORKSPACE_DIR}/artifact_manifest.json" \
      --runtime-only \
      --workspace-dir "${RESTORE_WORKSPACE_DIR}")"
    verify_status=$?
    echo "${verify_output}"
    if [[ ${verify_status} -ne 0 ]]; then
      echo "[artifact-restore] verification failed"
      exit ${verify_status}
    fi
  else
    echo "[artifact-restore] unsupported ARTIFACT_STORE_BACKEND=${ARTIFACT_STORE_BACKEND}"
    exit 1
  fi

  direct_count="$(DOWNLOAD_PAYLOAD="${download_output}" python3 - <<'PY'
import json
import os
payload = json.loads(os.environ.get("DOWNLOAD_PAYLOAD", "{}"))
print(int(payload.get("downloaded_count", payload.get("restored_count", 0))))
PY
)"
  chunked_count="$(DOWNLOAD_PAYLOAD="${download_output}" python3 - <<'PY'
import json
import os
payload = json.loads(os.environ.get("DOWNLOAD_PAYLOAD", "{}"))
print(int(payload.get("restored_chunked_count", 0)))
PY
)"
  echo "[artifact-restore] restored artifacts count=${direct_count}"
  echo "[artifact-restore] restored chunked artifacts count=${chunked_count}"
  echo "[artifact-restore] verification passed"
fi

exec "$@"
