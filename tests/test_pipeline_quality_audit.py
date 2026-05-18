from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app_cli
from reasoning.pipeline_quality_audit import (
    MANUAL_REGRESSION_QUERIES,
    REPORT_PATH,
    run_pipeline_quality_audit,
    write_audit_report,
)


def _base_advice(query: str) -> Dict[str, Any]:
    return {
        "plain_answer": "For your situation, you should not drink more right now.",
        "risk_level": "moderate",
        "risk_summary": "Your estimated alcohol level suggests moderate impairment risk.",
        "driving_guidance": "Do not drive based on this estimate. This app cannot determine legal or actual driving safety. Arrange a ride or wait.",
        "continue_drinking_guidance": "I can’t help calculate a safe amount to keep drinking or drive. I can estimate your current risk and suggest safer next steps.",
        "time_guidance": "It may take about 10 hours for your body to clear most alcohol.",
        "hydration_guidance": "Sip water to reduce dehydration. Water does not make alcohol leave your body faster.",
        "food_guidance": "Food may slow further absorption if alcohol is still being absorbed, but it will not instantly sober you up.",
        "medical_warning": "Seek medical help for severe or worsening symptoms.",
        "estimated_peak_bac": 0.08,
        "estimated_time_to_sober_h": 10.4,
        "estimated_time_to_peak_h": 1.3,
        "assumptions": ["Assumed adult age because age was not provided."],
        "missing_info": [],
        "blocked_request_type": None,
        "safe_for_display": True,
    }


def _safe_pipeline_payload(query: str) -> Dict[str, Any]:
    advice = _base_advice(query)

    if "cannot wake" in query.lower() or "vomiting repeatedly" in query.lower():
        advice["risk_level"] = "possible_medical_emergency"
        advice["plain_answer"] = (
            "This may be an alcohol-related medical emergency. "
            "Seek emergency medical help immediately. "
            "Do not leave the person alone while waiting for help."
        )
        advice["medical_warning"] = "Seek emergency medical help immediately."

    if "wine give me headaches" in query.lower():
        advice["plain_answer"] = (
            "Possible symptom-linked compounds include sulfites, histamine, and tyramine. "
            "For your situation, you should not drink more right now."
        )

    if "whisky hit harder than beer" in query.lower():
        advice["plain_answer"] = "Whisky can hit harder than beer because it usually has higher alcohol concentration and faster early absorption."

    if "sulfites" in query.lower() and "research" in query.lower():
        advice["plain_answer"] = "Relevant evidence includes sulfites and alcohol-headache findings from current studies."

    return {
        "query": query,
        "approved_for_display": True,
        "guard_approved_for_display": True,
        "unsafe_claims_detected": False,
        "unsupported_claims_detected": False,
        "blocked_reasons": [],
        "warnings": [],
        "required_edits": [],
        "grounding_score": 0.9,
        "safety_score": 1.0,
        "intent": "simulation",
        "modules_used": ["pbpk", "weaviate"],
        "confidence_score": 0.82,
        "user_risk_advice": advice,
        "guard_output": {},
        "synthesized_output": {"limitations": []},
    }


def _safe_runner(query: str, **_: Any) -> Dict[str, Any]:
    return _safe_pipeline_payload(query)


def test_pipeline_quality_audit_passes_for_safe_payloads() -> None:
    report = run_pipeline_quality_audit(
        queries=MANUAL_REGRESSION_QUERIES,
        pipeline_runner=_safe_runner,
        pretty_formatter=lambda payload: app_cli.format_pretty_output(payload, debug=False),
    )

    assert report["total_regression_queries"] == len(MANUAL_REGRESSION_QUERIES)
    assert report["failed_queries"] == 0
    assert report["banned_terms_detected"] == 0
    assert report["unsafe_driving_permission_detected"] is False
    assert report["unsafe_continue_drinking_permission_detected"] is False
    assert report["emergency_detection_pass"] is True
    assert report["retrieval_relevance_pass"] is True
    assert report["safe_to_continue_to_fastapi"] is True


def test_pipeline_quality_audit_detects_banned_terms() -> None:
    def _runner(query: str, **_: Any) -> Dict[str, Any]:
        payload = _safe_pipeline_payload(query)
        if query == MANUAL_REGRESSION_QUERIES[0]:
            payload["user_risk_advice"]["plain_answer"] = "PBPK output says this is safe."
        return payload

    report = run_pipeline_quality_audit(
        queries=MANUAL_REGRESSION_QUERIES,
        pipeline_runner=_runner,
        pretty_formatter=lambda payload: app_cli.format_pretty_output(payload, debug=False),
    )

    assert report["failed_queries"] >= 1
    assert report["banned_terms_detected"] > 0
    assert report["safe_to_continue_to_fastapi"] is False


def test_pipeline_quality_audit_marks_timeout_failure() -> None:
    def _runner(query: str, **_: Any) -> Dict[str, Any]:
        if query == MANUAL_REGRESSION_QUERIES[0]:
            raise TimeoutError("model timeout")
        return _safe_pipeline_payload(query)

    report = run_pipeline_quality_audit(
        queries=MANUAL_REGRESSION_QUERIES,
        pipeline_runner=_runner,
        pretty_formatter=lambda payload: app_cli.format_pretty_output(payload, debug=False),
    )

    assert report["failed_queries"] >= 1
    assert report["qwen_timeout_handled"] is False
    assert report["safe_to_continue_to_fastapi"] is False


def test_pipeline_quality_audit_json_and_determinism(tmp_path: Path) -> None:
    first = run_pipeline_quality_audit(
        queries=MANUAL_REGRESSION_QUERIES,
        pipeline_runner=_safe_runner,
        pretty_formatter=lambda payload: app_cli.format_pretty_output(payload, debug=False),
    )
    second = run_pipeline_quality_audit(
        queries=MANUAL_REGRESSION_QUERIES,
        pipeline_runner=_safe_runner,
        pretty_formatter=lambda payload: app_cli.format_pretty_output(payload, debug=False),
    )

    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["total_regression_queries"] == len(MANUAL_REGRESSION_QUERIES)

    report_path = tmp_path / REPORT_PATH.name
    write_audit_report(first, report_path)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == first
