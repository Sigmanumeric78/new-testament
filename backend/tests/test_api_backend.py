from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

import pytest

try:
    import fastapi  # noqa: F401
    from pydantic import ValidationError

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi is not installed")

if FASTAPI_AVAILABLE:
    from api.main import ALLOWED_CORS_ORIGINS
    from api.health import health_check
    from api.routes import ask_endpoint, intake_endpoint, route_endpoint
    from api.schemas import AskRequest, IntakeRequest, QueryRequest


EXPECTED_ASK_KEYS: Set[str] = {
    "query",
    "answer",
    "risk_level",
    "risk_summary",
    "estimated_peak_bac",
    "estimated_time_to_sober_h",
    "estimated_time_to_peak_h",
    "ethanol_dose_g",
    "drink_abv_percent",
    "drink_volume_ml",
    "legal_limit_reference_bac",
    "is_estimated_below_0_08",
    "estimated_total_volume_for_0_08_ml",
    "estimated_additional_volume_to_0_08_ml",
    "threshold_explanation",
    "beverage_type",
    "likely_compounds",
    "body_processes",
    "detail_level",
    "driving_guidance",
    "continue_drinking_guidance",
    "hydration_guidance",
    "food_guidance",
    "medical_warning",
    "assumptions",
    "missing_info",
    "safe_for_display",
    "advisor_fallback_used",
    "synthesis_blocked",
    "blocked_synthesis_reasons",
    "blocked_request_type",
}


BANNED_TERMS = (
    "pbpk",
    "neo4j",
    "weaviate",
    "causal path",
    "graph",
    "adh",
    "aldh",
    "cyp2e1",
    "embedding",
    "vector",
    "simulator fallback",
    "confidence score",
)


def _guard_grounding_block_payload(query: str, response_payload: Dict[str, Any]) -> Dict[str, Any]:
    _ = response_payload
    return {
        "query": query,
        "approved_for_display": False,
        "final_answer": "You are safe to drive.",
        "grounding_score": 0.69,
        "safety_score": 1.0,
        "blocked_reasons": ["Grounding score below threshold (0.70)."],
        "warnings": [],
        "required_edits": [],
        "unsupported_claims_detected": True,
        "unsafe_claims_detected": False,
        "safety_notes_present": True,
        "medical_disclaimer_present": True,
        "driving_warning_present": True,
    }


@pytest.fixture(autouse=True)
def _fast_deterministic_pipeline(monkeypatch: Any) -> None:
    from reasoning.hybrid_orchestrator import HybridOrchestrator
    from reasoning.response_synthesizer import ResponseSynthesizer

    def _force_model_fallback(self: ResponseSynthesizer, prompt: str) -> str:
        _ = prompt
        raise RuntimeError("forced fallback for API tests")

    def _neo4j_unavailable(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        _ = (query, route, parsed_inputs)
        return (
            {
                "status": "unavailable",
                "error": "neo4j unavailable in tests",
                "query_templates_used": [],
                "path_count": 0,
                "paths": [],
                "node_names": [],
                "relationship_types": [],
            },
            ["Neo4j module unavailable in tests."],
        )

    def _weaviate_stub(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        _ = (query, route)
        return (
            {
                "status": "success",
                "retrieval_backend": "embedded_fallback",
                "top_k": 8,
                "collections_searched": [],
                "hit_count": 0,
                "hits": [],
            },
            ["Some supporting evidence was unavailable."],
        )

    monkeypatch.setattr(ResponseSynthesizer, "_invoke_ollama", _force_model_fallback)
    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", _neo4j_unavailable)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", _weaviate_stub)


def _assert_no_banned_terms(text: str) -> None:
    lower = text.lower()
    for term in BANNED_TERMS:
        assert term not in lower


def test_health_endpoint_returns_valid_json() -> None:
    payload = health_check()

    assert payload["status"] in {"ok", "degraded", "error"}
    components = payload["components"]
    required = {
        "api",
        "neo4j",
        "weaviate",
        "ollama",
        "artifact_status",
        "pbpk",
        "router",
        "orchestrator",
        "synthesizer",
        "grounding_guard",
        "user_risk_advisor",
    }
    assert required.issubset(set(components.keys()))
    artifact = components["artifact_status"]
    assert "ok" in artifact
    assert "detail" in artifact
    assert "missing_required_count" in artifact
    assert "missing_required" in artifact


def test_cors_origins_include_local_5173_and_no_wildcard() -> None:
    assert "http://localhost:5173" in ALLOWED_CORS_ORIGINS
    assert "http://127.0.0.1:5173" in ALLOWED_CORS_ORIGINS
    assert "*" not in ALLOWED_CORS_ORIGINS


def test_route_endpoint_works() -> None:
    payload = route_endpoint(QueryRequest(query="Why does whisky hit harder than beer?"))
    assert "intent" in payload
    assert "required_modules" in payload


def test_ask_continue_drinking_query_is_safe() -> None:
    query = "I am 75 kg male, fed, I just drank 200 ml vodka in 1 hour, how much more can I drink?"
    payload = ask_endpoint(AskRequest(query=query))

    assert payload["query"] == query
    assert payload["continue_drinking_guidance"]
    assert "safe to drive" not in payload["answer"].lower()
    assert "safe to drink more" not in payload["answer"].lower()
    guidance_lower = payload["continue_drinking_guidance"].lower()
    assert "safe extra amount" not in payload["answer"].lower()
    assert "recommend using this app to decide whether to drink more" in guidance_lower or "increase impairment risk" in guidance_lower
    assert payload["blocked_request_type"] == "unsafe_continue_drinking_recommendation"
    assert payload["advisor_fallback_used"] in {True, False}
    assert payload["synthesis_blocked"] in {True, False}
    assert isinstance(payload["blocked_synthesis_reasons"], list)
    _assert_no_banned_terms(payload["answer"])


def test_ask_driving_query_blocks_safe_to_drive_language(monkeypatch: Any) -> None:
    from reasoning.grounding_safety_guard import GroundingSafetyGuard

    def _blocked_guard(self: GroundingSafetyGuard, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        return _guard_grounding_block_payload(response_payload.get("query", ""), response_payload)

    monkeypatch.setattr(GroundingSafetyGuard, "validate", _blocked_guard)

    query = "Can I drive after drinking 180ml whisky?"
    payload = ask_endpoint(AskRequest(query=query))

    answer = payload["answer"].lower()
    guidance = payload["driving_guidance"].lower()
    assert answer.startswith("i can’t tell you that you are safe to drive")
    assert "you can drive" not in answer
    assert "probably safe" not in answer
    assert "below limit so safe" not in answer
    assert "safe amount to keep drinking" not in answer
    assert not answer.startswith("i can’t calculate a safe amount to keep drinking")
    assert "do not drive" in guidance
    assert "cannot determine legal or actual driving safety" in guidance
    assert payload["blocked_request_type"] == "unsafe_driving_check"
    assert payload["advisor_fallback_used"] is True
    assert payload["synthesis_blocked"] is True
    assert payload["blocked_synthesis_reasons"]


def test_ask_extra_amount_query_uses_extra_amount_refusal() -> None:
    query = "How much more vodka can I drink before I am too drunk?"
    payload = ask_endpoint(AskRequest(query=query))
    assert payload["blocked_request_type"] == "unsafe_extra_amount_calculation"
    assert payload["answer"].lower().startswith("i can’t calculate a safe extra amount to drink")
    assert "not a recommendation" in (payload.get("threshold_explanation") or "").lower() or payload.get(
        "estimated_total_volume_for_0_08_ml"
    ) is None


def test_low_bac_keep_drinking_query_has_threshold_context_without_extra_permission() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. Should I keep drinking?"
    payload = ask_endpoint(AskRequest(query=query, response_style="layman"))

    assert payload["blocked_request_type"] == "unsafe_continue_drinking_recommendation"
    assert payload["estimated_peak_bac"] is not None
    assert float(payload["estimated_peak_bac"]) < 0.08
    assert payload["is_estimated_below_0_08"] is True
    assert payload["estimated_total_volume_for_0_08_ml"] is not None
    assert payload["legal_limit_reference_bac"] == 0.08
    answer = payload["answer"].lower()
    assert not answer.startswith("i can’t calculate")
    assert "you can drink" not in answer
    assert "not a recommendation" in (payload["threshold_explanation"] or "").lower()
    assert payload["continue_drinking_guidance"].lower() != "you should not drink more right now."


def test_ask_scientific_mode_returns_analysis_fields() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. What is happening in my body?"
    payload = ask_endpoint(AskRequest(query=query, response_style="scientific"))
    assert payload["detail_level"] == "scientific"
    assert payload["ethanol_dose_g"] is not None
    assert payload["threshold_explanation"]
    assert payload["estimated_total_volume_for_0_08_ml"] is not None
    assert isinstance(payload["likely_compounds"], list)
    assert isinstance(payload["body_processes"], list)
    assert payload["answer"].lower().count("estimated peak bac") <= 1


def test_technical_mode_is_more_detailed_than_layman() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. What is happening in my body?"
    layman = ask_endpoint(AskRequest(query=query, response_style="layman"))
    technical = ask_endpoint(AskRequest(query=query, response_style="technical"))
    scientific = ask_endpoint(AskRequest(query=query, response_style="scientific"))

    assert layman["detail_level"] == "layman"
    assert technical["detail_level"] == "technical"
    assert scientific["detail_level"] == "scientific"
    assert layman["answer"]
    assert technical["answer"]
    assert scientific["answer"]
    assert "ADH" in technical["answer"] or "ALDH" in technical["answer"] or "CYP2E1" in technical["answer"]
    assert scientific["likely_compounds"] or scientific["body_processes"]


def test_low_bac_driving_guidance_is_strict_without_safe_to_drive_permission() -> None:
    payload = ask_endpoint(
        AskRequest(query="I am 75 kg male, fed, I drank 30 ml vodka in 1 hour. Can I drive?", response_style="layman")
    )
    answer = payload["answer"].lower()
    guidance = payload["driving_guidance"].lower()
    assert payload["blocked_request_type"] == "unsafe_driving_check"
    assert answer.startswith("i can’t tell you that you are safe to drive") or answer.startswith(
        "i can't tell you that you are safe to drive"
    )
    assert "you can drive" not in answer
    assert "probably safe" not in answer
    assert "below limit so safe" not in answer
    assert "cannot determine legal or actual driving safety" in guidance


def test_ask_emergency_query_returns_emergency_guidance() -> None:
    query = "My friend is vomiting repeatedly and cannot wake up after drinking."
    payload = ask_endpoint(AskRequest(query=query))

    assert payload["risk_level"] == "possible_medical_emergency"
    assert "emergency" in payload["medical_warning"].lower() or "emergency" in payload["answer"].lower()


def test_empty_query_rejected() -> None:
    with pytest.raises(ValidationError):
        AskRequest(query="   ")


def test_too_long_query_rejected() -> None:
    with pytest.raises(ValidationError):
        AskRequest(query="a" * 2001)


def test_debug_false_hides_internals() -> None:
    payload = ask_endpoint(AskRequest(query="Why does whisky hit harder?", debug=False))
    assert "debug" not in payload
    combined = " ".join(
        [
            payload["answer"],
            payload["driving_guidance"],
            payload["continue_drinking_guidance"],
            payload["hydration_guidance"],
            payload["food_guidance"],
            payload["medical_warning"],
            " ".join(payload["assumptions"]),
        ]
    )
    _assert_no_banned_terms(combined)


def test_debug_true_includes_internal_payload() -> None:
    payload = ask_endpoint(AskRequest(query="Why does whisky hit harder?", debug=True))

    assert "debug" in payload
    debug = payload["debug"]
    assert "route" in debug
    assert "orchestration" in debug
    assert "synthesis" in debug
    assert "guard" in debug


def test_debug_consistency_when_guard_blocks(monkeypatch: Any) -> None:
    from reasoning.grounding_safety_guard import GroundingSafetyGuard

    def _blocked_guard(self: GroundingSafetyGuard, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        return _guard_grounding_block_payload(response_payload.get("query", ""), response_payload)

    monkeypatch.setattr(GroundingSafetyGuard, "validate", _blocked_guard)

    payload = ask_endpoint(AskRequest(query="Can I drive after drinking 180ml whisky?", debug=True))

    assert payload["debug"]["guard"]["approved_for_display"] is False
    assert payload["synthesis_blocked"] is True
    assert payload["advisor_fallback_used"] is True
    assert payload["blocked_synthesis_reasons"]
    assert payload["answer"] != payload["debug"]["guard"]["final_answer"]
    answer = payload["answer"].lower()
    assert answer.startswith("i can’t tell you that you are safe to drive")
    assert "you can drive" not in answer


def test_intake_endpoint_works() -> None:
    payload = intake_endpoint(
        IntakeRequest(
            sex="male",
            weight_kg=75,
            age=30,
            fed_state="fed",
            drink_type="vodka",
            amount_ml=200,
            duration_h=1,
            goal="should_i_keep_drinking",
        )
    )

    assert payload["query"]
    assert payload["continue_drinking_guidance"]
    assert payload["risk_level"]
    assert "Some personal inputs were assumed because they were not provided." not in payload["assumptions"]


def test_json_schema_stable_default_ask() -> None:
    payload = ask_endpoint(AskRequest(query="How drunk will I get after 180ml whisky?"))

    assert set(payload.keys()) == EXPECTED_ASK_KEYS
    encoded = json.dumps(payload, sort_keys=True)
    decoded: Dict[str, Any] = json.loads(encoded)
    assert decoded["query"] == payload["query"]
