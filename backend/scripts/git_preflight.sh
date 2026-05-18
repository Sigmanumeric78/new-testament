#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$BACKEND_DIR/.." && pwd)"
if [[ ! -d "$REPO_ROOT/backend" ]]; then
  REPO_ROOT="$BACKEND_DIR"
fi

cd "$REPO_ROOT"

bash backend/scripts/clean_generated.sh
python3 backend/scripts/check_repo_hygiene.py

PYTHONPATH=backend python3 -m pytest -q \
  backend/tests/test_query_router.py \
  backend/tests/test_hybrid_orchestrator.py \
  backend/tests/test_response_synthesizer.py \
  backend/tests/test_grounding_safety_guard.py \
  backend/tests/test_user_risk_advisor.py \
  backend/tests/test_pipeline_quality_audit.py \
  backend/tests/test_scientific_validity_audit.py \
  backend/tests/test_app_cli.py

python3 - <<'PY'
import json
from pathlib import Path

report = Path("data/interim/reasoning/repo_hygiene_report.json")
if not report.exists():
    raise SystemExit("repo_hygiene_report.json not found")

payload = json.loads(report.read_text(encoding="utf-8"))
if not bool(payload.get("safe_for_git_push")):
    raise SystemExit("Repo hygiene checks failed")
PY

echo "READY_FOR_GIT_PUSH=true"
