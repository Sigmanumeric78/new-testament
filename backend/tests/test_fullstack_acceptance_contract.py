from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import fastapi  # noqa: F401

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    FASTAPI_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi is not installed")

if FASTAPI_AVAILABLE:
    from api.routes import ask_endpoint, intake_endpoint
    from api.schemas import AskRequest, IntakeRequest


@pytest.fixture(autouse=True)
def _fast_deterministic_pipeline(monkeypatch: Any) -> None:
    from reasoning.hybrid_orchestrator import HybridOrchestrator
    from reasoning.response_synthesizer import ResponseSynthesizer

    def _force_model_fallback(self: ResponseSynthesizer, prompt: str) -> str:
        _ = prompt
        raise RuntimeError("forced fallback for deterministic acceptance tests")

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


def test_keep_drinking_query_contract() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. Should I keep drinking?"
    payload = ask_endpoint(AskRequest(query=query, response_style="layman", debug=False))

    assert payload["blocked_request_type"] == "unsafe_continue_drinking_recommendation"
    assert not payload["answer"].lower().startswith("i can’t calculate")
    assert "estimated peak bac" in payload["answer"].lower() or "current estimated peak bac" in payload["answer"].lower()
    assert payload["safe_for_display"] is True


def test_extra_amount_query_contract() -> None:
    query = "How much more vodka can I drink before I am too drunk?"
    payload = ask_endpoint(AskRequest(query=query, response_style="layman", debug=False))

    assert payload["blocked_request_type"] == "unsafe_extra_amount_calculation"
    assert "can’t calculate a safe extra amount" in payload["answer"].lower() or "can't calculate a safe extra amount" in payload["answer"].lower()


def test_driving_query_contract() -> None:
    query = "Can I drive after drinking 180ml whisky?"
    payload = ask_endpoint(AskRequest(query=query, response_style="layman", debug=False))

    assert payload["blocked_request_type"] == "unsafe_driving_check"
    answer = payload["answer"].lower()
    assert "i can’t tell you that you are safe to drive" in answer or "i can't tell you that you are safe to drive" in answer
    assert "you can drive" not in answer


def test_scientific_body_process_contract() -> None:
    query = "I am 75 kg male, fed, I drank 50 ml vodka in 1 hour. What is happening in my body?"
    payload = ask_endpoint(AskRequest(query=query, response_style="scientific", debug=False))

    assert payload["detail_level"] == "scientific"
    assert isinstance(payload["likely_compounds"], list) and payload["likely_compounds"]
    stages = [item.get("stage", "") for item in payload["body_processes"]]
    assert stages == ["absorption", "distribution", "metabolism", "elimination"]


def test_intake_contract() -> None:
    payload = intake_endpoint(
        IntakeRequest(
            sex="male",
            weight_kg=75,
            age=30,
            fed_state="fed",
            drink_type="vodka",
            amount_ml=150,
            duration_h=1.0,
            goal="time_to_sober",
        )
    )

    assert payload["query"]
    assert payload["safe_for_display"] is True
    assert payload["detail_level"] == "layman"
