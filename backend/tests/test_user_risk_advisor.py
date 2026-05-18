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
            "safe_defaults": {
                "sex": "male",
                "weight": 75.0,
                "fed_state": "fed",
                "age": 30,
                "liver_status": "healthy",
            },
            "personalized_mode": False,
        },
        "limitations": ["PBPK used default ABV for 'vodka': 40.0%."],
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
    assert signals["unsafe_extra_amount_calculation_request"] is False


def test_continue_drinking_request_refusal() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. Should I keep drinking?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload={
            **_synthesized_payload_with_simulation(),
            "simulation_summary": {
                **_synthesized_payload_with_simulation()["simulation_summary"],
                "simulations": [
                    {
                        "beverage": "vodka",
                        "volume_ml": 50.0,
                        "abv_percent": 40.0,
                        "peak_bac_percent": 0.014,
                        "time_to_peak_h": 1.0,
                        "time_to_sober_h": 1.7,
                    }
                ],
            },
        },
        orchestrator_payload=None,
    )

    assert advice["blocked_request_type"] == "unsafe_continue_drinking_recommendation"
    assert "won’t recommend whether you should keep drinking" in advice["plain_answer"].lower()
    assert "can’t calculate a safe extra amount" not in advice["plain_answer"].lower()
    assert advice["estimated_peak_bac"] is not None and advice["estimated_peak_bac"] < 0.08
    assert advice["is_estimated_below_0_08"] is True
    assert advice["estimated_total_volume_for_0_08_ml"] is not None
    assert advice["threshold_explanation"] is not None
    assert "not a recommendation" in advice["threshold_explanation"].lower()
    assert advice["continue_drinking_guidance"].lower() != "you should not drink more right now."


def test_extra_amount_refusal_wording() -> None:
    query = "How much more vodka can I drink before I am too drunk?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    assert advice["blocked_request_type"] == "unsafe_extra_amount_calculation"
    assert advice["plain_answer"].lower().startswith("i can’t calculate a safe extra amount to drink")
    assert "safe extra amount" not in advice["continue_drinking_guidance"].lower()
    assert "not a recommendation" in (advice["threshold_explanation"] or "").lower() or advice[
        "estimated_total_volume_for_0_08_ml"
    ] is None


def test_driving_request_never_grants_safe_to_drive() -> None:
    query = "I am 75 kg male, fed, I drank 30 ml vodka in 1 hour. Can I drive now?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload=_synthesized_payload_with_simulation(),
        orchestrator_payload=None,
    )

    answer = advice["plain_answer"].lower()
    assert answer.startswith("i can’t tell you that you are safe to drive")
    assert "you can drive" not in answer
    assert "probably safe" not in answer
    assert "below limit so safe" not in answer
    assert "cannot determine legal or actual driving safety" in advice["driving_guidance"].lower()


def test_scientific_mode_includes_compounds_and_processes() -> None:
    advice = build_user_risk_advice(
        query="I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. What is happening in my body?",
        guarded_payload=_guarded_payload(),
        synthesized_payload={
            **_synthesized_payload_with_simulation(),
            "response_style": "scientific",
            "simulation_summary": {
                **_synthesized_payload_with_simulation()["simulation_summary"],
                "simulations": [
                    {
                        "beverage": "vodka",
                        "volume_ml": 50.0,
                        "abv_percent": 40.0,
                        "peak_bac_percent": 0.028,
                        "time_to_peak_h": 1.0,
                        "time_to_sober_h": 2.8,
                    }
                ],
            },
        },
        orchestrator_payload=None,
    )

    assert advice["detail_level"] == "scientific"
    assert advice["ethanol_dose_g"] is not None
    assert "ethanol" in [c.lower() for c in advice["likely_compounds"]]
    stages = [item["stage"] for item in advice["body_processes"]]
    assert stages == ["absorption", "distribution", "metabolism", "elimination"]
    assert "structured sections below show dose" in advice["plain_answer"].lower()
    assert advice["estimated_total_volume_for_0_08_ml"] is not None


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
    assert advice["legal_limit_reference_bac"] == 0.08
    assert advice["is_estimated_below_0_08"] is False


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


def test_specific_assumptions_are_reported_without_generic_placeholder() -> None:
    query = "I drank 180 ml whisky in 1 hour. Can I drive?"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=_guarded_payload(),
        synthesized_payload={
            "simulation_summary": {
                "simulations": [
                    {
                        "beverage": "whisky",
                        "volume_ml": 180.0,
                        "abv_percent": 40.0,
                        "peak_bac_percent": 0.091,
                        "time_to_peak_h": 1.7,
                        "time_to_sober_h": 10.4,
                    }
                ],
                "defaults_applied": ["sex", "weight", "age", "fed_state", "liver_status"],
                "safe_defaults": {
                    "sex": "male",
                    "weight": 75.0,
                    "fed_state": "fed",
                    "age": 30,
                    "liver_status": "healthy",
                },
                "personalized_mode": False,
            },
            "limitations": ["PBPK used default ABV for 'whisky': 40.0%."],
            "toxicity_summary": None,
        },
        orchestrator_payload=None,
    )

    assumptions = advice["assumptions"]
    assert "Some personal inputs were assumed because they were not provided." not in assumptions
    assert "Assumed adult age because age was not provided." in assumptions
    assert "Assumed 75 kg body weight because weight was not provided." in assumptions
    assert "Assumed male sex because sex was not provided." in assumptions
    assert "Assumed fed state because meal status was not provided." in assumptions
    assert "Assumed healthy adult metabolism." in assumptions
    assert "Assumed standard whisky ABV of 40%." in assumptions
