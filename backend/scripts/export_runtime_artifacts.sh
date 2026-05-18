#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${BACKEND_ROOT}/.." && pwd)"

RELEASE="${1:-v0.6-chemical-explorer}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${BACKEND_ROOT}"
else
  export PYTHONPATH="${BACKEND_ROOT}:${PYTHONPATH}"
fi

echo "Exporting Neo4j runtime metadata for release: ${RELEASE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/export_neo4j_data.py" --release "${RELEASE}"

echo "Exporting Weaviate runtime metadata for release: ${RELEASE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/export_weaviate_data.py" --release "${RELEASE}"

echo "Verifying runtime exports and generating checksums/manifest for release: ${RELEASE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_runtime_exports.py" --release "${RELEASE}"

echo "Runtime export flow complete."
