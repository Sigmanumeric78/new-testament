#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESTORE_ARTIFACTS_ON_STARTUP="${RESTORE_ARTIFACTS_ON_STARTUP:-false}"
ARTIFACT_RELEASE="${ARTIFACT_RELEASE:-v0.6-chemical-explorer}"
SUPABASE_ARTIFACT_BUCKET="${SUPABASE_ARTIFACT_BUCKET:-alcohol-intelligence-artifacts}"
RESTORE_WORKSPACE_DIR="${RESTORE_WORKSPACE_DIR:-/tmp/artifact_restore/${ARTIFACT_RELEASE}}"
PYTHONPATH="${PYTHONPATH:-/app/backend}"

export PROJECT_ROOT
export DATA_ROOT
export PYTHONPATH
export RESTORE_WORKSPACE_DIR

if [[ "${RESTORE_ARTIFACTS_ON_STARTUP,,}" == "true" ]]; then
  echo "[artifact-restore] restore started"
  echo "[artifact-restore] release=${ARTIFACT_RELEASE}"
  echo "[artifact-restore] bucket=${SUPABASE_ARTIFACT_BUCKET}"
  echo "[artifact-restore] project_root=${PROJECT_ROOT}"
  echo "[artifact-restore] data_root=${DATA_ROOT}"

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

  direct_count="$(DOWNLOAD_PAYLOAD="${download_output}" python3 - <<'PY'
import json
import os
payload = json.loads(os.environ.get("DOWNLOAD_PAYLOAD", "{}"))
print(int(payload.get("downloaded_count", 0)))
PY
)"
  chunked_count="$(DOWNLOAD_PAYLOAD="${download_output}" python3 - <<'PY'
import json
import os
payload = json.loads(os.environ.get("DOWNLOAD_PAYLOAD", "{}"))
print(int(payload.get("restored_chunked_count", 0)))
PY
)"
  echo "[artifact-restore] restored direct artifacts count=${direct_count}"
  echo "[artifact-restore] restored chunked artifacts count=${chunked_count}"
  echo "[artifact-restore] verification passed"
fi

exec "$@"
