from __future__ import annotations

from typing import Any, Dict, List

import pytest

from reasoning.hybrid_orchestrator import HybridOrchestrator
from utils.config import get_pinecone_config
from vectorstores.pinecone_store import PineconeVectorStore


def test_pinecone_config_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.config as config_module

    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX", "healthlens-knowledge")
    monkeypatch.setenv("PINECONE_NAMESPACE", "production")
    monkeypatch.setenv("PINECONE_DIMENSION", "768")
    monkeypatch.setenv("PINECONE_METRIC", "cosine")

    config = get_pinecone_config(require=True)

    assert config["api_key"] == "test-key"
    assert config["index"] == "healthlens-knowledge"
    assert config["namespace"] == "production"
    assert config["dimension"] == 768
    assert config["metric"] == "cosine"


def test_pinecone_dimension_validation() -> None:
    store = PineconeVectorStore(
        config={
            "api_key": "test-key",
            "index": "idx",
            "namespace": "ns",
            "dimension": 3,
            "metric": "cosine",
        },
        index=object(),
    )

    assert store.validate_vector_dimension([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.validate_vector_dimension([0.1, 0.2])


def test_pinecone_metadata_normalization() -> None:
    class _Index:
        def query(self, **kwargs: Any) -> Dict[str, Any]:
            assert kwargs["namespace"] == "production"
            return {
                "matches": [
                    {
                        "id": "chunk-1",
                        "score": 0.92,
                        "metadata": {
                            "object_id": "obj-1",
                            "chunk_id": "chunk-1",
                            "collection": "ScientificEvidence",
                            "source_collection": "ScientificEvidence",
                            "title": "Sulfites and headache evidence",
                            "content": "Sulfites and headache evidence content.",
                            "content_excerpt": "Sulfites and headache evidence content.",
                            "metadata_json": '{"compound":"sulfites"}',
                            "provenance_json": '{"source":"unit"}',
                            "source_file": "scientific_evidence_embeddings.parquet",
                        },
                    }
                ]
            }

    store = PineconeVectorStore(
        config={
            "api_key": "test-key",
            "index": "idx",
            "namespace": "production",
            "dimension": 3,
            "metric": "cosine",
        },
        index=_Index(),
    )

    hits = store.query_by_vector([0.1, 0.2, 0.3], top_k=5, collections=["ScientificEvidence"])

    assert hits == [
        {
            "id": "chunk-1",
            "object_id": "obj-1",
            "chunk_id": "chunk-1",
            "collection": "ScientificEvidence",
            "title": "Sulfites and headache evidence",
            "content_excerpt": "Sulfites and headache evidence content.",
            "content": "Sulfites and headache evidence content.",
            "score": 0.92,
            "distance": 0.08,
            "metadata": {"compound": "sulfites"},
            "provenance": {"source": "unit"},
            "source_file": "scientific_evidence_embeddings.parquet",
            "source_dataset": "",
            "retrieval_backend": "pinecone",
        }
    ]


def test_pinecone_collection_filter_construction() -> None:
    query_filter = PineconeVectorStore.build_collection_filter(["ScientificEvidence", "PBPKKnowledge"])

    assert query_filter == {
        "$or": [
            {"collection": {"$in": ["PBPKKnowledge", "ScientificEvidence"]}},
            {"source_collection": {"$in": ["PBPKKnowledge", "ScientificEvidence"]}},
        ]
    }


def test_orchestrator_uses_pinecone_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import reasoning.hybrid_orchestrator as orchestrator_module

    calls: List[Dict[str, Any]] = []

    class _Store:
        def query_by_vector(self, vector: List[float], top_k: int, collections: List[str]) -> List[Dict[str, Any]]:
            calls.append({"vector_len": len(vector), "top_k": top_k, "collections": collections})
            return [
                {
                    "object_id": "obj-1",
                    "chunk_id": "chunk-1",
                    "collection": "ScientificEvidence",
                    "title": "Sulfites headache evidence",
                    "content_excerpt": "Sulfites and alcohol headache evidence.",
                    "content": "Sulfites and alcohol headache evidence.",
                    "score": 0.95,
                    "distance": 0.05,
                    "metadata": {},
                    "provenance": {},
                    "source_file": "scientific_evidence_embeddings.parquet",
                    "retrieval_backend": "pinecone",
                }
            ]

        def close(self) -> None:
            pass

    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    monkeypatch.setattr(orchestrator_module, "PineconeVectorStore", _Store)
    monkeypatch.setattr(
        HybridOrchestrator,
        "_build_query_vector",
        lambda self, query, collections: (
            [0.001] * 768,
            {
                "query_vector_source": "test_vector",
                "query_vector_dimension": 768,
                "query_vector_seed_rows": 1,
                "embedding_model_reference": "nomic-ai/nomic-embed-text-v1",
            },
        ),
    )

    result, limitations = HybridOrchestrator()._execute_semantic_retrieval(
        "Show research on sulfites and alcohol headaches",
        {"intent": "scientific_evidence"},
    )

    assert result["retrieval_backend"] == "pinecone"
    assert result["query_vector_source"] == "test_vector"
    assert result["query_vector_dimension"] == 768
    assert result["embedding_model_reference"] == "nomic-ai/nomic-embed-text-v1"
    assert calls == [{"vector_len": 768, "top_k": 8, "collections": ["ScientificEvidence"]}]
    assert limitations == []


def test_orchestrator_falls_back_when_pinecone_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import reasoning.hybrid_orchestrator as orchestrator_module

    class _FailingStore:
        def query_by_vector(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
            raise RuntimeError("pinecone unavailable")

        def close(self) -> None:
            pass

    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    monkeypatch.setattr(orchestrator_module, "PineconeVectorStore", _FailingStore)
    monkeypatch.setattr(
        HybridOrchestrator,
        "_build_query_vector",
        lambda self, query, collections: (
            [0.001] * 768,
            {"query_vector_source": "test_vector", "query_vector_dimension": 768},
        ),
    )
    monkeypatch.setattr(
        HybridOrchestrator,
        "_embedded_fallback_search",
        lambda self, query, collections, top_k: [
            {
                "object_id": "fallback-1",
                "collection": "ScientificEvidence",
                "title": "Sulfites headache evidence",
                "content_excerpt": "Sulfites and headache evidence.",
                "score": 1.0,
                "distance": None,
                "source_dataset": "",
                "source_file": "",
            }
        ],
    )

    result, limitations = HybridOrchestrator()._execute_semantic_retrieval(
        "Show research on sulfites and alcohol headaches",
        {"intent": "scientific_evidence"},
    )

    assert result["retrieval_backend"] == "embedded_fallback"
    assert result["hit_count"] == 1
    assert any("Pinecone query failed" in item for item in limitations)


def test_health_adds_pinecone_component_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.health as health_module

    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    monkeypatch.setattr(health_module, "_neo4j_probe", lambda: {"ok": True, "detail": "ok"})
    monkeypatch.setattr(health_module, "_mongodb_probe", lambda: {"ok": True, "detail": "ok"})
    monkeypatch.setattr(health_module, "_artifact_probe", lambda: {"ok": True, "detail": "ok", "missing_required_count": 0, "missing_required": []})
    monkeypatch.setattr(health_module, "_ollama_probe", lambda: {"ok": True, "detail": "standby (LLM_PROVIDER=disabled)"})
    monkeypatch.setattr(health_module, "_pinecone_probe", lambda: {"ok": True, "detail": "ok"})

    payload = health_module.build_health_payload()

    assert payload["status"] == "healthy"
    assert payload["components"]["pinecone"]["ok"] is True
    assert "mongodb" in payload["components"]
    assert "weaviate" not in payload["components"]
