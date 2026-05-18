from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning.query_router import (
    INTENT_MODULES,
    LOW_CONFIDENCE_THRESHOLD,
    QueryRouter,
    route_query,
    validate_route_result_schema,
)


REPRESENTATIVE_QUERIES: List[Tuple[str, str]] = [
    # simulation
    ("How drunk will I get from 4 beers?", "simulation"),
    ("How long until sober after vodka shots?", "simulation"),
    ("What will my BAC be at 2 hours?", "simulation"),
    ("Peak BAC after drinking whisky", "simulation"),
    ("Time to sober if I had wine", "simulation"),
    # mechanistic_explanation
    ("Why does whisky hit harder?", "mechanistic_explanation"),
    ("Why does wine make me sleepy?", "mechanistic_explanation"),
    ("Explain ethanol metabolism mechanism", "mechanistic_explanation"),
    ("How does alcohol get metabolized in liver?", "mechanistic_explanation"),
    ("What mechanism causes a quick buzz from champagne?", "mechanistic_explanation"),
    # toxicity_risk
    ("Why do I get headaches from wine?", "toxicity_risk"),
    ("Why does beer upset my stomach?", "toxicity_risk"),
    ("Why do cocktails give me nausea?", "toxicity_risk"),
    ("Is migraine risk higher with red wine?", "toxicity_risk"),
    ("Hangover risk from congeners?", "toxicity_risk"),
    # comparison
    ("Beer vs whisky", "comparison"),
    ("Will vodka hit harder than beer?", "comparison"),
    ("Compare rum and tequila effects", "comparison"),
    ("Is wine stronger than cider for intoxication?", "comparison"),
    ("Which gets me drunk faster, gin or beer?", "comparison"),
    # scientific_evidence
    ("Show studies about sulfites", "scientific_evidence"),
    ("Research on alcohol metabolism", "scientific_evidence"),
    ("Papers on congeners and hangover", "scientific_evidence"),
    ("Evidence for acetaldehyde toxicity", "scientific_evidence"),
    ("Meta-analysis on alcohol absorption rates", "scientific_evidence"),
    # personalized_physiology
    ("I am 60kg female, fasted, drank wine", "personalized_physiology"),
    ("I am 80 kg male, 35 years old, had 2 beers 1 hour ago", "personalized_physiology"),
    ("Female 55kg fed, drank 150 ml vodka", "personalized_physiology"),
    ("I drank 3 shots on empty stomach, 70kg male", "personalized_physiology"),
    ("My profile: 90kg man, 42 years old, with food, drank whisky", "personalized_physiology"),
    # retrieval_only
    ("What is ethanol?", "retrieval_only"),
    ("What are congeners?", "retrieval_only"),
    ("Define acetaldehyde", "retrieval_only"),
    ("Tell me about fusel alcohols", "retrieval_only"),
    ("Meaning of sulfites in wine", "retrieval_only"),
]


def test_representative_query_routing_accuracy_threshold() -> None:
    router = QueryRouter(enable_llm_fallback=False)
    correct = 0
    for query, expected_intent in REPRESENTATIVE_QUERIES:
        result = router.route(query)
        validate_route_result_schema(result)
        if result.intent == expected_intent:
            correct += 1

    accuracy = correct / float(len(REPRESENTATIVE_QUERIES))
    assert len(REPRESENTATIVE_QUERIES) >= 30
    assert accuracy >= 0.90


def test_required_modules_match_intent_mapping() -> None:
    router = QueryRouter(enable_llm_fallback=False)
    for query, expected_intent in REPRESENTATIVE_QUERIES:
        result = router.route(query)
        if result.intent == expected_intent:
            assert result.required_modules == INTENT_MODULES[expected_intent]


def test_router_deterministic_rerun_behavior() -> None:
    query = "Will vodka hit harder than beer?"
    baseline = route_query(query, enable_llm_fallback=False).to_dict()
    for _ in range(10):
        rerun = route_query(query, enable_llm_fallback=False).to_dict()
        assert rerun == baseline


def test_json_safe_output_contract() -> None:
    result = route_query("Why does whisky hit harder?", enable_llm_fallback=False)
    payload = result.to_dict()
    validate_route_result_schema(result)

    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["intent"] == payload["intent"]
    assert isinstance(decoded["sub_intents"], list)
    assert isinstance(decoded["required_modules"], list)
    assert isinstance(decoded["required_inputs"], dict)
    assert isinstance(decoded["missing_required_inputs"], list)
    assert isinstance(decoded["routing_reasoning"], list)
    assert isinstance(decoded["confidence"], float)


def test_response_style_inference_scientific_technical_layman_defaults() -> None:
    scientific = route_query(
        "Explain scientifically why whisky hits harder.",
        enable_llm_fallback=False,
    )
    assert scientific.response_style == "scientific"

    technical = route_query(
        "Give a technical breakdown of ethanol metabolism.",
        enable_llm_fallback=False,
    )
    assert technical.response_style == "technical"

    layman_explicit = route_query(
        "Explain in simple words why wine makes me sleepy.",
        enable_llm_fallback=False,
    )
    assert layman_explicit.response_style == "layman"

    layman_default = route_query(
        "Why does wine make me sleepy?",
        enable_llm_fallback=False,
    )
    assert layman_default.response_style == "layman"


def test_required_input_detection_for_simulation() -> None:
    result = route_query(
        "How long until sober after wine?",
        enable_llm_fallback=False,
    )
    assert result.intent == "simulation"
    assert result.required_inputs["body_weight"] is True
    assert result.required_inputs["sex"] is True
    assert result.required_inputs["age"] is True
    assert result.required_inputs["fed_state"] is True
    assert result.required_inputs["beverages"] is True
    assert result.required_inputs["drink_amount"] is True
    assert result.required_inputs["time_since_drinking"] is True
    assert "body_weight" in result.missing_required_inputs
    assert "sex" in result.missing_required_inputs
    assert "age" in result.missing_required_inputs
    assert "fed_state" in result.missing_required_inputs
    assert "drink_amount" in result.missing_required_inputs
    assert "time_since_drinking" in result.missing_required_inputs


def test_low_confidence_triggers_qwen_fallback() -> None:
    calls: List[Dict[str, Any]] = []

    def fake_disambiguator(query: str, baseline: Dict[str, Any]) -> Dict[str, Any]:
        calls.append({"query": query, "baseline": baseline})
        return {
            "intent": "retrieval_only",
            "sub_intents": ["definition_lookup"],
            "response_style": "layman",
            "confidence": 0.88,
            "routing_reasoning": ["fallback intent selected"],
        }

    router = QueryRouter(
        enable_llm_fallback=True,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        disambiguator=fake_disambiguator,
    )
    result = router.route("hello there")
    assert calls
    assert result.intent == "retrieval_only"
    assert result.confidence >= 0.35
    assert any("Low-confidence fallback engaged" in line for line in result.routing_reasoning)


def test_high_confidence_skips_fallback() -> None:
    called = False

    def fake_disambiguator(_: str, __: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal called
        called = True
        return {"intent": "retrieval_only"}

    router = QueryRouter(
        enable_llm_fallback=True,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        disambiguator=fake_disambiguator,
    )
    result = router.route("How long until sober?")
    assert result.intent == "simulation"
    assert result.confidence >= LOW_CONFIDENCE_THRESHOLD
    assert called is False


def test_fallback_invalid_payload_keeps_deterministic_result() -> None:
    def fake_disambiguator(_: str, __: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "intent": "not_a_real_intent",
            "sub_intents": [],
            "response_style": "layman",
            "confidence": 0.99,
            "routing_reasoning": ["bad payload"],
        }

    router = QueryRouter(
        enable_llm_fallback=True,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        disambiguator=fake_disambiguator,
    )
    result = router.route("hello there")
    assert result.intent == "retrieval_only"
    assert any("invalid intent" in line for line in result.routing_reasoning)


def test_route_query_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        route_query("   ", enable_llm_fallback=False)


def test_cli_compact_output_json() -> None:
    script = REPO_ROOT / "reasoning" / "query_router.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--query",
            "Why does whisky hit harder?",
            "--disable-llm-fallback",
            "--compact",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["intent"] == "mechanistic_explanation"
    assert payload["required_modules"] == ["neo4j", "weaviate"]
