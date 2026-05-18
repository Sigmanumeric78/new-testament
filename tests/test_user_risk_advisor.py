from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning.user_risk_advisor import (
    BANNED_TECHNICAL_TERMS,
    build_user_risk_advice,
    extract_query_signals,
)


def _guarded_payload() -> Dict[str, Any]:
    return {
        "query": "",
        "approved_for_display": True,
        "final_answer": "",
        "grounding_score": 0.9,
        "safety_score": 1.0,
        "blocked_reasons": [],
        "warnings": [],
        "required_edits": [],
        "unsupported_claims_detected": False,
        "unsafe_claims_detected": False,
        "safety_notes_present": True,
        "medical_disclaimer_present": True,
        "driving_warning_present": True,
    }


def _synthesized_payload_with_simulation() -> Dict[str, Any]:
    return {
        "simulation_summary": {
            "simulations": [
                {
                    "beverage": "vodka",
                    "volume_ml": 200.0,
                    "abv_percent": 40.0,
                    "peak_bac_percent": 0.091,
                    "time_to_peak_h": 1.7,
                    "time_to_sober_h": 10.4,
                }
            ],
            "defaults_applied": ["age"],
            "personalized_mode": False,
        },
        "toxicity_summary": None,
    }


def test_long_sentence_extraction() -> None:
    query = "I am 75 kg male, fed, I just drank 200 ml vodka in 1 hour, how much more can I drink?"
    signals = extract_query_signals(query)

    assert signals["weight_kg"] == 75.0
    assert signals["sex"] == "male"
    assert signals["fed_state"] == "fed"
    assert signals["drink_type"] == "vodka"
    assert signals["amount_ml"] == 200.0
    assert signals["duration_h"] == 1.0
    assert signals["unsafe_continue_drinking_request"] is True


def test_continue_drinking_request_refusal() -> None:
    query = "How much more can I drink?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    assert advice["blocked_request_type"] == "unsafe_continue_drinking"
    assert "can’t help calculate a safe amount" in advice["continue_drinking_guidance"].lower()
    assert "should not drink more" in advice["plain_answer"].lower()


def test_driving_request_never_grants_safe_to_drive() -> None:
    query = "Can I drive now?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    assert "safe to drive" not in advice["plain_answer"].lower()
    assert "do not drive" in advice["driving_guidance"].lower()


def test_layman_output_removes_banned_technical_terms() -> None:
    query = "How drunk will I get?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    plain = advice["plain_answer"].lower()
    for term in BANNED_TECHNICAL_TERMS:
        assert term.lower() not in plain


def test_simulation_query_includes_estimates_when_available() -> None:
    query = "How drunk will I get after 200 ml vodka?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    assert advice["estimated_peak_bac"] == 0.091
    assert advice["estimated_time_to_sober_h"] == 10.4
    assert "about 10 hours" in advice["time_guidance"].lower()


def test_emergency_symptoms_trigger_emergency_risk() -> None:
    query = "My friend is unconscious with slow breathing and blue lips after alcohol poisoning"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=None,
        orchestrator_payload=None,
    )

    assert advice["risk_level"] == "possible_medical_emergency"
    assert "emergency" in advice["medical_warning"].lower()


def test_intake_mode_query_builder_logic() -> None:
    from app_cli import build_query_from_intake

    intake = {
        "sex": "female",
        "weight": "60",
        "age": "29",
        "fed_state": "fasted",
        "drink_type": "whisky",
        "amount": "180ml",
        "time_period": "1 hour",
        "goal": "drive check",
    }
    query = build_query_from_intake(intake)
    assert "female" in query.lower()
    assert "60 kg" in query.lower()
    assert "180ml whisky" in query.lower()
    assert "can i drive now" in query.lower()


def test_deterministic_rerun_behavior() -> None:
    query = "I am 75 kg male, fed, I just drank 200 ml vodka in 1 hour, how much more can I drink?"
    first = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )
    second = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )
    assert first == second


def test_json_serializable() -> None:
    query = "How drunk will I get after 200 ml vodka?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    encoded = json.dumps(advice, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["risk_level"] == advice["risk_level"]
