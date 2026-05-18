from __future__ import annotations

import os
import re
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
SCRIPT_PATH = MONOREPO_ROOT / "scripts/fullstack_acceptance_check.sh"


def _read_script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_acceptance_script_exists_and_executable() -> None:
    assert SCRIPT_PATH.exists()
    assert os.access(SCRIPT_PATH, os.X_OK)


def test_acceptance_script_has_hardened_timeout_defaults() -> None:
    text = _read_script()
    assert 'CURL_TIMEOUT="${CURL_TIMEOUT:-' in text
    match = re.search(r'CURL_TIMEOUT="\$\{CURL_TIMEOUT:-(\d+)\}"', text)
    assert match is not None
    assert int(match.group(1)) >= 60
    assert 'HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"' in text


def test_acceptance_script_has_timeout_diagnostics_and_prewarm_option() -> None:
    text = _read_script()
    assert (
        "Request timed out. This usually means Ollama/Qwen is cold-starting or the backend is still generating."
        in text
    )
    assert "Try: ollama run qwen2.5:3b 'ready'" in text
    assert "Or rerun with: CURL_TIMEOUT=120 ./scripts/fullstack_acceptance_check.sh" in text
    assert 'PREWARM_OLLAMA="${PREWARM_OLLAMA:-false}"' in text
    assert "/api/generate" in text


def test_acceptance_script_keeps_core_safety_validations() -> None:
    text = _read_script()
    assert "unsafe_continue_drinking_recommendation" in text
    assert "unsafe_extra_amount_calculation" in text
    assert "unsafe_driving_check" in text
    assert "you can drive" in text
    assert "probably safe" in text
    assert "below limit so safe" in text
