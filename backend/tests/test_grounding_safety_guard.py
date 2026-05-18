from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning.grounding_safety_guard import (
    MANDATORY_SAFETY_NOTES,
    TOXICITY_SAFETY_NOTE,
    GroundingSafetyGuard,
)


def _base_response_payload(query: str = "Why does whisky hit harder?") -> Dict[str, Any]:
    return {
        "query": query,
        "answer": "Whisky may hit harder because higher ABV and faster absorption can raise blood alcohol sooner.",
        "response_style": "layman",
        "used_facts": [
            "Whisky may have higher ABV than beer.",
            "Faster absorption can increase early intoxication intensity.",
        ],
        "used_causal_paths": [
            "whisky -[CONTAINS]-> ethanol -[METABOLIZED_BY]-> ADH",
        ],
        "used_evidence": [
            {
                "object_id": "ev1",
                "collection": "ScientificEvidence",
                "title": "ABV and absorption evidence",
                "content_excerpt": "Higher ABV and fast absorption increase early blood alcohol rise.",
                "score": 0.91,
                "distance": None,
                "source_dataset": "scientific_evidence",
                "source_file": "abv_absorption.jsonl",
            }
        ],
        "simulation_summary": None,
        "toxicity_summary": None,
        "limitations": ["No personalized physiology inputs provided."],
        "safety_notes": list(MANDATORY_SAFETY_NOTES),
        "confidence_score": 0.82,
        "unsupported_claims_detected": False,
        "safe_for_user_display": True,
    }


def test_safe_layman_response_passes() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()

    result = guard.validate(payload)

    assert result["approved_for_display"] is True
    assert result["unsafe_claims_detected"] is False
    assert result["unsupported_claims_detected"] is False
    assert result["safety_notes_present"] is True
    assert result["medical_disclaimer_present"] is True
    assert result["driving_warning_present"] is True


def test_missing_safety_notes_auto_appended() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()
    payload["safety_notes"] = []

    result = guard.validate(payload)

    assert result["approved_for_display"] is True
    assert result["required_edits"]
    assert "This is an estimate, not medical advice." in result["final_answer"]
    assert "Do not use this to decide whether it is safe to drive." in result["final_answer"]


def test_safe_to_drive_claim_is_blocked() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()
    payload["answer"] = "You are safe to drive after this amount because there is no risk."

    result = guard.validate(payload)

    assert result["approved_for_display"] is False
    assert result["unsafe_claims_detected"] is True
    assert any("safe to drive" in reason.lower() or "no-risk" in reason.lower() or "no risk" in reason.lower() for reason in result["blocked_reasons"])


def test_drink_more_recommendation_is_blocked() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()
    payload["answer"] = "You can drink more now because the effect is low."

    result = guard.validate(payload)

    assert result["approved_for_display"] is False
    assert result["unsafe_claims_detected"] is True
    assert any("drink more" in reason.lower() for reason in result["blocked_reasons"])


def test_unsupported_medical_diagnosis_is_blocked() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()
    payload["answer"] = "This is a medical diagnosis and you have liver disease."

    result = guard.validate(payload)

    assert result["approved_for_display"] is False
    assert result["unsafe_claims_detected"] is True
    assert any("diagnosis" in reason.lower() for reason in result["blocked_reasons"])


def test_toxicity_answer_requires_severe_symptom_warning() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload("Why does wine give me headaches?")
    payload["answer"] = "The evidence links sulfites and histamine with headache risk in sensitive people."
    payload["toxicity_summary"] = {
        "risk_compounds": ["sulfites", "histamine"],
        "risk_types": ["headache_risk"],
        "symptom_modifiers": ["headache", "histamine", "sulfites"],
        "confidence": 0.86,
    }
    payload["used_facts"] = [
        "Toxicity evidence identified sulfites and histamine as risk compounds.",
    ]
    payload["used_causal_paths"] = [
        "wine -[CONTAINS]-> sulfites -[CONTRIBUTES_TO]-> headache_risk",
    ]
    payload["safety_notes"] = list(MANDATORY_SAFETY_NOTES)

    result = guard.validate(payload)

    assert result["approved_for_display"] is True
    assert any(TOXICITY_SAFETY_NOTE in edit for edit in result["required_edits"]) or TOXICITY_SAFETY_NOTE in result["final_answer"]
    assert result["safety_notes_present"] is True


def test_json_serializability() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload()

    result = guard.validate(payload)

    encoded = json.dumps(result, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["query"] == payload["query"]


def test_deterministic_rerun_behavior() -> None:
    guard = GroundingSafetyGuard()
    payload = _base_response_payload("Why does whisky hit harder?")

    first = guard.validate(payload)
    second = guard.validate(payload)

    assert first == second
