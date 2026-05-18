from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning.hybrid_orchestrator import HybridOrchestrator


def _fake_pbpk_success() -> Dict[str, Any]:
    return {
        "status": "success",
        "personalized_mode": False,
        "defaults_applied": ["sex", "weight", "age", "fed_state", "body_fat_percent", "liver_status"],
        "safe_defaults": {
            "sex": "male",
            "weight": 75.0,
            "fed_state": "fed",
            "age": 30,
            "body_fat_percent": None,
            "liver_status": "healthy",
        },
        "simulations": [
            {
                "beverage": "whisky",
                "volume_ml": 180.0,
                "abv_percent": 40.0,
                "peak_bac_percent": 0.08,
                "time_to_peak_h": 2.0,
                "time_to_sober_h": 8.0,
                "ethanol_auc_mg_h_l": 100.0,
                "acetaldehyde_auc_mg_h_l": 10.0,
            }
        ],
    }


def _fake_neo4j_success() -> Dict[str, Any]:
    return {
        "status": "success",
        "query_templates_used": ["A", "B", "C", "D"],
        "path_count": 2,
        "node_names": ["whisky", "acetaldehyde", "headache_risk"],
        "relationship_types": ["CONTAINS", "CONTRIBUTES_TO"],
        "paths": [
            {
                "template": "A",
                "path": "whisky -[CONTAINS]-> acetaldehyde -[METABOLIZED_BY]-> ALDH",
                "nodes": ["whisky", "acetaldehyde", "ALDH"],
                "relationship_types": ["CONTAINS", "METABOLIZED_BY"],
                "confidence": 0.9,
                "beverage": "whisky",
                "compound": "acetaldehyde",
                "enzyme": "ALDH",
            },
            {
                "template": "B",
                "path": "wine -[CONTAINS]-> sulfites -[CONTRIBUTES_TO]-> headache_risk",
                "nodes": ["wine", "sulfites", "headache_risk"],
                "relationship_types": ["CONTAINS", "CONTRIBUTES_TO"],
                "confidence": 0.86,
                "beverage": "wine",
                "compound": "sulfites",
                "risk_type": "headache_risk",
            },
        ],
    }


def _fake_weaviate_success() -> Dict[str, Any]:
    return {
        "status": "success",
        "retrieval_backend": "embedded_fallback",
        "top_k": 8,
        "collections_searched": ["ScientificEvidence"],
        "hit_count": 2,
        "hits": [
            {
                "object_id": "obj-1",
                "collection": "ScientificEvidence",
                "title": "Evidence on sulfites and headaches",
                "content_excerpt": "Sulfites and histamine are linked to headache symptoms in sensitive people.",
                "score": 0.84,
                "distance": None,
                "source_dataset": "scientific_evidence",
                "source_file": "sample1.jsonl",
            },
            {
                "object_id": "obj-2",
                "collection": "ToxicityKnowledge",
                "title": "Wine toxicity modifier",
                "content_excerpt": "Acetaldehyde and congeners can amplify hangover and headache risk.",
                "score": 0.77,
                "distance": None,
                "source_dataset": "toxicity_knowledge",
                "source_file": "sample2.jsonl",
            },
        ],
    }


def test_mechanistic_explanation_calls_neo4j_and_weaviate(monkeypatch: Any) -> None:
    calls: List[str] = []

    def fake_neo4j(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        calls.append("neo4j")
        return _fake_neo4j_success(), []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        calls.append("weaviate")
        return _fake_weaviate_success(), []

    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", fake_neo4j)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("Why does whisky hit harder?")

    assert calls == ["neo4j", "weaviate"]
    assert result["module_results"]["neo4j"]["status"] == "success"
    assert result["module_results"]["weaviate"]["status"] == "success"


def test_simulation_query_runs_pbpk_with_defaults_when_allowed() -> None:
    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("How drunk will I get after 180ml whisky?")

    pbpk = result["module_results"]["pbpk"]
    assert pbpk is not None
    assert pbpk["status"] == "success"
    assert pbpk["simulations"]
    assert pbpk["simulations"][0]["beverage"] == "whisky"
    assert pbpk["simulations"][0]["volume_ml"] == 180.0


def test_toxicity_query_calls_neo4j_weaviate_toxicity(monkeypatch: Any) -> None:
    calls: List[str] = []
    original_toxicity = HybridOrchestrator._execute_toxicity

    def fake_neo4j(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        calls.append("neo4j")
        return _fake_neo4j_success(), []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        calls.append("weaviate")
        return _fake_weaviate_success(), []

    def spy_toxicity(self: HybridOrchestrator, query: str, neo4j_result: Any, weaviate_result: Any) -> Any:
        calls.append("toxicity")
        return original_toxicity(self, query, neo4j_result, weaviate_result)

    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", fake_neo4j)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)
    monkeypatch.setattr(HybridOrchestrator, "_execute_toxicity", spy_toxicity)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("Why does wine give me headaches?")

    assert calls == ["neo4j", "weaviate", "toxicity"]
    toxicity = result["module_results"]["toxicity"]
    assert toxicity is not None
    assert toxicity["status"] == "success"
    assert "sulfites" in toxicity["risk_compounds"]


def test_comparison_query_calls_pbpk_neo4j_weaviate(monkeypatch: Any) -> None:
    calls: List[str] = []

    def fake_pbpk(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        calls.append("pbpk")
        return _fake_pbpk_success(), [], []

    def fake_neo4j(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        calls.append("neo4j")
        return _fake_neo4j_success(), []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        calls.append("weaviate")
        return _fake_weaviate_success(), []

    monkeypatch.setattr(HybridOrchestrator, "_execute_pbpk", fake_pbpk)
    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", fake_neo4j)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("Beer vs whisky, which hits harder?")

    assert calls == ["pbpk", "neo4j", "weaviate"]
    assert result["module_results"]["pbpk"]["status"] == "success"
    assert result["module_results"]["neo4j"]["status"] == "success"
    assert result["module_results"]["weaviate"]["status"] == "success"


def test_scientific_evidence_query_calls_weaviate_only(monkeypatch: Any) -> None:
    calls: List[str] = []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        calls.append("weaviate")
        return _fake_weaviate_success(), []

    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("Show research on sulfites")

    assert calls == ["weaviate"]
    assert result["module_results"]["pbpk"] is None
    assert result["module_results"]["neo4j"] is None
    assert result["module_results"]["toxicity"] is None
    assert result["module_results"]["weaviate"]["status"] == "success"


def test_personalized_missing_input_avoids_unsafe_pbpk_simulation() -> None:
    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("For my body, how drunk will I get?")

    pbpk = result["module_results"]["pbpk"]
    assert pbpk is not None
    assert pbpk["status"] == "skipped_missing_inputs"
    assert "body_weight" in result["missing_inputs"]
    assert result["safe_for_response_synthesis"] is False


def test_hybrid_orchestrator_output_is_json_serializable(monkeypatch: Any) -> None:
    def fake_neo4j(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        return _fake_neo4j_success(), []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        return _fake_weaviate_success(), []

    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", fake_neo4j)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    result = orchestrator.orchestrate("Why does whisky hit harder?")

    encoded = json.dumps(result, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["query"] == "Why does whisky hit harder?"
    assert isinstance(decoded["module_results"], dict)
    assert isinstance(decoded["evidence_bundle"], dict)


def test_hybrid_orchestrator_deterministic_rerun_behavior(monkeypatch: Any) -> None:
    def fake_neo4j(self: HybridOrchestrator, query: str, route: Dict[str, Any], parsed_inputs: Any) -> Any:
        return _fake_neo4j_success(), []

    def fake_weaviate(self: HybridOrchestrator, query: str, route: Dict[str, Any]) -> Any:
        return _fake_weaviate_success(), []

    monkeypatch.setattr(HybridOrchestrator, "_execute_neo4j", fake_neo4j)
    monkeypatch.setattr(HybridOrchestrator, "_execute_weaviate", fake_weaviate)

    orchestrator = HybridOrchestrator(enable_llm_fallback=False)
    query = "Why does whisky hit harder?"
    first = orchestrator.orchestrate(query)
    second = orchestrator.orchestrate(query)

    assert first == second
