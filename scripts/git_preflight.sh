#!/usr/bin/env bash
set -euo pipefail

bash scripts/clean_generated.sh
python3 scripts/check_repo_hygiene.py

pytest -q \
  tests/test_query_router.py \
  tests/test_hybrid_orchestrator.py \
  tests/test_response_synthesizer.py \
  tests/test_grounding_safety_guard.py \
  tests/test_user_risk_advisor.py \
  tests/test_pipeline_quality_audit.py \
  tests/test_scientific_validity_audit.py \
  tests/test_app_cli.py

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
