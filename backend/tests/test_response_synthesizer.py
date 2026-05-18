from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import reasoning.response_synthesizer as rs


def _orchestrator_payload_for_query(query: str) -> Dict[str, Any]:
    q = query.strip()

    if q == "Why does whisky hit harder?":
        return {
            "query": q,
            "route": {
                "intent": "mechanistic_explanation",
                "sub_intents": ["causal_why", "relative_potency_mechanism"],
                "required_modules": ["neo4j", "weaviate"],
                "required_inputs": {},
                "missing_required_inputs": [],
                "response_style": "layman",
                "confidence": 0.92,
                "routing_reasoning": ["deterministic route"],
            },
            "module_results": {"pbpk": None, "neo4j": {}, "weaviate": {}, "toxicity": None},
            "evidence_bundle": {
                "key_facts": [
                    "Whisky commonly has higher ABV than beer.",
                    "Faster ethanol absorption can increase early intoxication intensity.",
                ],
                "causal_paths": [
                    "whisky -[CONTAINS]-> ethanol -[METABOLIZED_BY]-> ADH",
                ],
                "retrieved_evidence": [
                    {
                        "object_id": "ev-mech-1",
                        "collection": "ScientificEvidence",
                        "title": "ABV and intoxication intensity",
                        "content_excerpt": "Higher ABV and faster absorption increase early blood alcohol rise.",
                        "score": 0.89,
                        "distance": None,
                        "source_dataset": "scientific_evidence",
                        "source_file": "abv_intensity.jsonl",
                    }
                ],
                "simulation_summary": None,
                "toxicity_summary": None,
                "confidence_score": 0.78,
                "limitations": ["No personalized physiology inputs were provided."],
            },
            "missing_inputs": [],
            "safe_for_response_synthesis": True,
        }

    if q == "How drunk will I get after 180ml whisky?":
        return {
            "query": q,
            "route": {
                "intent": "simulation",
                "sub_intents": ["intoxication_level"],
                "required_modules": ["pbpk"],
                "required_inputs": {},
                "missing_required_inputs": [],
                "response_style": "layman",
                "confidence": 0.94,
                "routing_reasoning": ["deterministic route"],
            },
            "module_results": {"pbpk": {}, "neo4j": None, "weaviate": None, "toxicity": None},
            "evidence_bundle": {
                "key_facts": [
                    "PBPK estimated peak BAC around 0.081.",
                    "PBPK estimated time-to-sober around 7.8 hours.",
                ],
                "causal_paths": [],
                "retrieved_evidence": [
                    {
                        "object_id": "ev-sim-1",
                        "collection": "PBPKKnowledge",
                        "title": "PBPK whisky simulation reference",
                        "content_excerpt": "Simulation shows peak BAC and elimination trajectory.",
                        "score": 0.91,
                        "distance": None,
                        "source_dataset": "pbpk_knowledge",
                        "source_file": "pbpk_reference.jsonl",
                    }
                ],
                "simulation_summary": {
                    "simulations": [
                        {
                            "beverage": "whisky",
                            "volume_ml": 180.0,
                            "abv_percent": 40.0,
                            "peak_bac_percent": 0.081,
                            "time_to_peak_h": 2.1,
                            "time_to_sober_h": 7.8,
                            "ethanol_auc_mg_h_l": 101.0,
                            "acetaldehyde_auc_mg_h_l": 11.0,
                        }
                    ],
                    "defaults_applied": ["sex", "weight", "age", "fed_state", "body_fat_percent", "liver_status"],
                    "personalized_mode": False,
                },
                "toxicity_summary": None,
                "confidence_score": 0.83,
                "limitations": [],
            },
            "missing_inputs": [],
            "safe_for_response_synthesis": True,
        }

    if q == "Why does wine give me headaches?":
        return {
            "query": q,
            "route": {
                "intent": "toxicity_risk",
                "sub_intents": ["headache_risk"],
                "required_modules": ["neo4j", "weaviate", "toxicity"],
                "required_inputs": {},
                "missing_required_inputs": [],
                "response_style": "layman",
                "confidence": 0.9,
                "routing_reasoning": ["deterministic route"],
            },
            "module_results": {"pbpk": None, "neo4j": {}, "weaviate": {}, "toxicity": {}},
            "evidence_bundle": {
                "key_facts": [
                    "Toxicity evidence identified risk compounds and risk types.",
                ],
                "causal_paths": [
                    "wine -[CONTAINS]-> sulfites -[CONTRIBUTES_TO]-> headache_risk",
                ],
                "retrieved_evidence": [
                    {
                        "object_id": "ev-tox-1",
                        "collection": "ToxicityKnowledge",
                        "title": "Wine headache risk modifiers",
                        "content_excerpt": "Sulfites and histamine are associated with headache symptoms in sensitive individuals.",
                        "score": 0.88,
                        "distance": None,
                        "source_dataset": "toxicity_knowledge",
                        "source_file": "wine_headache.jsonl",
                    }
                ],
                "simulation_summary": None,
                "toxicity_summary": {
                    "risk_compounds": ["sulfites", "histamine"],
                    "risk_types": ["headache_risk"],
                    "symptom_modifiers": ["headache", "histamine", "sulfites"],
                    "confidence": 0.86,
                },
                "confidence_score": 0.8,
                "limitations": [],
            },
            "missing_inputs": [],
            "safe_for_response_synthesis": True,
        }

    if q == "Show research on sulfites":
        return {
            "query": q,
            "route": {
                "intent": "scientific_evidence",
                "sub_intents": ["literature_retrieval"],
                "required_modules": ["weaviate"],
                "required_inputs": {},
                "missing_required_inputs": [],
                "response_style": "layman",
                "confidence": 0.95,
                "routing_reasoning": ["deterministic route"],
            },
            "module_results": {"pbpk": None, "neo4j": None, "weaviate": {}, "toxicity": None},
            "evidence_bundle": {
                "key_facts": [
                    "Retrieved scientific evidence includes sulfite-related studies.",
                ],
                "causal_paths": [],
                "retrieved_evidence": [
                    {
                        "object_id": "ev-sci-1",
                        "collection": "ScientificEvidence",
                        "title": "Sulfites and intolerance evidence",
                        "content_excerpt": "Study evidence describing sulfite exposure and symptom reports.",
                        "score": 0.93,
                        "distance": None,
                        "source_dataset": "scientific_evidence",
                        "source_file": "sulfites_studies.jsonl",
                    },
                    {
                        "object_id": "ev-sci-2",
                        "collection": "ScientificEvidence",
                        "title": "Wine additives and adverse symptom prevalence",
                        "content_excerpt": "Observational evidence on sulfites and symptom associations.",
                        "score": 0.9,
                        "distance": None,
                        "source_dataset": "scientific_evidence",
                        "source_file": "wine_additives.jsonl",
                    },
                ],
                "simulation_summary": None,
                "toxicity_summary": None,
                "confidence_score": 0.85,
                "limitations": [],
            },
            "missing_inputs": [],
            "safe_for_response_synthesis": True,
        }

    raise ValueError(f"Unhandled mock query: {q}")


def _mock_orchestrate_query(query: str, enable_llm_fallback: bool = False) -> Dict[str, Any]:
    _ = enable_llm_fallback
    return _orchestrator_payload_for_query(query)


def _mock_ollama_response_for_prompt(prompt: str) -> str:
    if "Why does whisky hit harder?" in prompt:
        payload = {
            "answer": "Whisky can feel stronger because higher ABV and faster absorption can raise alcohol levels sooner.",
            "used_facts": [
                "Whisky commonly has higher ABV than beer.",
                "Faster ethanol absorption can increase early intoxication intensity.",
            ],
            "used_causal_paths": [
                "whisky -[CONTAINS]-> ethanol -[METABOLIZED_BY]-> ADH",
            ],
            "used_evidence_ids": ["ev-mech-1"],
            "limitations": ["No personalized physiology inputs were provided."],
        }
        return json.dumps(payload)

    if "How drunk will I get after 180ml whisky?" in prompt:
        payload = {
            "answer": "The PBPK estimate suggests peak BAC around 0.081 and time-to-sober around 7.8 hours.",
            "used_facts": [
                "PBPK estimated peak BAC around 0.081.",
                "PBPK estimated time-to-sober around 7.8 hours.",
            ],
            "used_causal_paths": [],
            "used_evidence_ids": ["ev-sim-1"],
            "limitations": [],
        }
        return json.dumps(payload)

    if "Why does wine give me headaches?" in prompt:
        payload = {
            "answer": "Based on the provided evidence, wine headache risk is linked to sulfites and histamine in susceptible people.",
            "used_facts": [
                "Toxicity evidence identified risk compounds and risk types.",
            ],
            "used_causal_paths": [
                "wine -[CONTAINS]-> sulfites -[CONTRIBUTES_TO]-> headache_risk",
            ],
            "used_evidence_ids": ["ev-tox-1"],
            "limitations": [],
        }
        return json.dumps(payload)

    if "Show research on sulfites" in prompt:
        payload = {
            "answer": "Evidence-oriented summary: retrieved studies include sulfite-focused scientific evidence with symptom association findings.",
            "used_facts": [
                "Retrieved scientific evidence includes sulfite-related studies.",
            ],
            "used_causal_paths": [],
            "used_evidence_ids": ["ev-sci-1", "ev-sci-2"],
            "limitations": [],
        }
        return json.dumps(payload)

    return json.dumps(
        {
            "answer": "Evidence summary generated from provided bundle.",
            "used_facts": [],
            "used_causal_paths": [],
            "used_evidence_ids": [],
            "limitations": [],
        }
    )


def test_layman_mechanistic_explanation(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Why does whisky hit harder?")

    assert result["answer"].strip()
    assert result["response_style"] == "layman"
    assert result["unsupported_claims_detected"] is False
    assert "This is an estimate, not medical advice." in result["safety_notes"]
    assert "Do not use this to decide whether it is safe to drive." in result["safety_notes"]


def test_simulation_answer_includes_pbpk_summary(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("How drunk will I get after 180ml whisky?")

    assert result["simulation_summary"] is not None
    sim = result["simulation_summary"]["simulations"][0]
    assert sim["peak_bac_percent"] is not None
    assert sim["time_to_sober_h"] is not None
    assert ("peak bac" in result["answer"].lower()) or ("time-to-sober" in result["answer"].lower())
    assert "Do not use this to decide whether it is safe to drive." in result["safety_notes"]


def test_toxicity_answer_includes_toxicity_summary(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Why does wine give me headaches?")

    assert result["toxicity_summary"] is not None
    assert "sulfites" in result["toxicity_summary"]["risk_compounds"]
    assert "Seek medical help for severe symptoms." in result["safety_notes"]


def test_scientific_evidence_answer_uses_scientific_style(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Show research on sulfites")

    assert result["response_style"] == "scientific"
    assert result["used_evidence"]
    assert any(item["source_file"] for item in result["used_evidence"])


def test_fallback_behavior_when_ollama_fails(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)

    def _raise_ollama(self: rs.ResponseSynthesizer, prompt: str) -> str:
        _ = prompt
        raise RuntimeError("simulated ollama failure")

    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", _raise_ollama)

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Why does whisky hit harder?")

    assert result["answer"].strip()
    assert result["unsupported_claims_detected"] is False
    assert result["safe_for_user_display"] is True
    assert any("fallback" in item.lower() for item in result["limitations"])


def test_timeout_fallback_behavior(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)

    def _timeout(self: rs.ResponseSynthesizer, prompt: str) -> str:
        _ = prompt
        raise subprocess.TimeoutExpired(cmd=["ollama"], timeout=30)

    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", _timeout)

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Why does whisky hit harder?")

    assert result["answer"].strip()
    assert result["safe_for_user_display"] is True
    assert any("fallback" in item.lower() for item in result["limitations"])


def test_json_serializability(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    result = synthesizer.synthesize_from_query("Why does whisky hit harder?")

    encoded = json.dumps(result, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["query"] == "Why does whisky hit harder?"


def test_deterministic_rerun_behavior(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "orchestrate_query", _mock_orchestrate_query)
    monkeypatch.setattr(rs.ResponseSynthesizer, "_invoke_ollama", lambda self, prompt: _mock_ollama_response_for_prompt(prompt))

    synthesizer = rs.ResponseSynthesizer()
    first = synthesizer.synthesize_from_query("Show research on sulfites")
    second = synthesizer.synthesize_from_query("Show research on sulfites")

    assert first == second


def test_default_timeout_seconds_is_30() -> None:
    synthesizer = rs.ResponseSynthesizer()
    assert synthesizer.timeout_seconds == 30
