from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import (
    check_all_artifacts,
    filter_runtime_specs,
    get_missing_required,
    load_manifest,
    summarize_artifacts,
)
from artifacts.local_store import sha256
from utils.config import get_project_root, resolve_project_path
from reasoning.hybrid_orchestrator import HybridOrchestrator


def _write_manifest(path: Path, artifacts: List[Dict[str, Any]]) -> Path:
    payload = {"manifest_version": "test", "artifacts": artifacts}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_manifest_loads_from_example() -> None:
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())

    assert specs
    assert len(specs) >= 30
    ids = {spec.artifact_id for spec in specs}
    assert "core_master_beverage_reference_repaired" in ids
    assert "weaviate_schema_design" in ids


def test_project_root_respects_project_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", "/app")
    assert get_project_root().as_posix() == "/app"


def test_resolve_project_path_for_backend_and_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", "/app")
    backend_path = resolve_project_path("backend/rag/neo4j/neo4j_graph_schema_design.md")
    data_path = resolve_project_path("data/artifact_manifest.example.json")
    assert backend_path.as_posix() == "/app/backend/rag/neo4j/neo4j_graph_schema_design.md"
    assert data_path.as_posix() == "/app/data/artifact_manifest.example.json"


def test_manifest_schema_design_paths_point_to_backend_monorepo() -> None:
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())
    by_id = {spec.artifact_id: spec for spec in specs}

    assert by_id["neo4j_graph_schema_design"].local_path == "backend/rag/neo4j/neo4j_graph_schema_design.md"
    assert by_id["weaviate_schema_design"].local_path == "backend/rag/weaviate/weaviate_schema_design.md"


def test_manifest_schema_design_files_validate_when_present() -> None:
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())
    filtered = [
        spec
        for spec in specs
        if spec.artifact_id in {"neo4j_graph_schema_design", "weaviate_schema_design"}
    ]
    statuses = check_all_artifacts(filtered)
    by_id = {status.artifact_id: status for status in statuses}

    assert by_id["neo4j_graph_schema_design"].validation_status == "ok"
    assert by_id["weaviate_schema_design"].validation_status == "ok"


def test_runtime_filter_includes_schema_design_inputs() -> None:
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())
    runtime_specs = filter_runtime_specs(specs)
    runtime_ids = {spec.artifact_id for spec in runtime_specs}
    assert "neo4j_graph_schema_design" in runtime_ids
    assert "weaviate_schema_design" in runtime_ids


def test_runtime_filter_includes_scientific_embedding_for_lambda_fallback() -> None:
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())
    runtime_specs = filter_runtime_specs(specs)
    runtime_ids = {spec.artifact_id for spec in runtime_specs}
    assert "weaviate_emb_scientific_evidence" in runtime_ids


def test_scientific_embedding_path_uses_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", "/app")
    monkeypatch.setenv("DATA_ROOT", "/tmp/data")
    orchestrator = HybridOrchestrator()
    path = orchestrator._embedded_corpus_path("ScientificEvidence")
    assert path.as_posix() == "/tmp/data/processed/weaviate/embedded/scientific_evidence_embeddings.parquet"


def test_query_vector_builder_uses_lambda_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    embedded_dir = data_root / "processed/weaviate/embedded"
    embedded_dir.mkdir(parents=True)
    parquet_path = embedded_dir / "scientific_evidence_embeddings.parquet"
    pd.DataFrame(
        [
            {
                "object_id": "evidence-1",
                "chunk_id": "chunk-1",
                "collection": "ScientificEvidence",
                "title": "Sulfites and headache",
                "content": "Sulfites and alcohol headaches in sensitive people.",
                "metadata": "{}",
                "provenance": "{}",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "object_id": "evidence-2",
                "chunk_id": "chunk-2",
                "collection": "ScientificEvidence",
                "title": "Unrelated",
                "content": "Different evidence topic.",
                "metadata": "{}",
                "provenance": "{}",
                "embedding": [0.0, 1.0, 0.0],
            },
        ]
    ).to_parquet(parquet_path)

    monkeypatch.setenv("PROJECT_ROOT", "/app")
    monkeypatch.setenv("DATA_ROOT", data_root.as_posix())
    orchestrator = HybridOrchestrator()

    vector, details = orchestrator._build_query_vector("sulfites alcohol headaches", ["ScientificEvidence"])

    assert orchestrator._embedded_corpus_path("ScientificEvidence") == parquet_path
    assert len(vector) == 3
    assert details["query_vector_dimension"] == 3
    assert details["query_vector_source"] == "embedded_corpus_weighted_average"
    assert details["query_vector_seed_rows"] >= 1


def test_weaviate_retrieval_uses_near_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    import reasoning.hybrid_orchestrator as orchestrator_module

    class FakeMetadata:
        score = None
        distance = 0.05
        certainty = None

    class FakeObject:
        properties = {
            "object_id": "evidence-1",
            "chunk_id": "chunk-1",
            "collection": "ScientificEvidence",
            "title": "Sulfites and alcohol headaches",
            "content": "Sulfites may be relevant to headache reports in sensitive people.",
            "metadata": "{}",
            "provenance": "{}",
        }
        metadata = FakeMetadata()

    class FakeResponse:
        objects = [FakeObject()]

    class FakeQuery:
        def __init__(self) -> None:
            self.near_vector_calls: List[Dict[str, Any]] = []

        def near_vector(self, **kwargs: Any) -> FakeResponse:
            self.near_vector_calls.append(kwargs)
            return FakeResponse()

        def hybrid(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("near_vector must be used instead of hybrid/near_text")

        def near_text(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("near_text must not be used for vectorizer-none collections")

    class FakeCollection:
        def __init__(self) -> None:
            self.query = FakeQuery()

    fake_collection = FakeCollection()

    class FakeCollections:
        def exists(self, _name: str) -> bool:
            return True

        def get(self, _name: str) -> FakeCollection:
            return fake_collection

    class FakeClient:
        collections = FakeCollections()

        def is_ready(self) -> bool:
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        orchestrator_module,
        "get_weaviate_config",
        lambda: {
            "url": "https://example.weaviate.cloud",
            "grpc_host": "grpc-example.weaviate.cloud",
            "grpc_port": "443",
            "api_key": "loaded",
        },
    )
    monkeypatch.setattr(orchestrator_module, "weaviate", object())
    monkeypatch.setattr(HybridOrchestrator, "_connect_weaviate", lambda self, config: FakeClient())
    monkeypatch.setattr(
        HybridOrchestrator,
        "_build_query_vector",
        lambda self, query, collections: (
            [0.001] * 768,
            {
                "query_vector_source": "test_vector",
                "query_vector_dimension": 768,
                "query_vector_seed_rows": 1,
                "query_vector_seed_overlap_max": 1,
            },
        ),
    )

    result, limitations = HybridOrchestrator()._execute_weaviate(
        "Show research on sulfites and alcohol headaches",
        {"intent": "scientific_evidence"},
    )

    assert result["status"] == "success"
    assert result["retrieval_backend"] == "weaviate_near_vector"
    assert result["hit_count"] == 1
    assert result["query_vector_dimension"] == 768
    assert fake_collection.query.near_vector_calls
    assert "query" not in fake_collection.query.near_vector_calls[0]
    assert not any("VectorFromInput" in item for item in limitations)


def test_weaviate_near_vector_failure_uses_embedded_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import reasoning.hybrid_orchestrator as orchestrator_module

    class FakeQuery:
        def near_vector(self, **_kwargs: Any) -> None:
            raise RuntimeError("cloud query unavailable")

    class FakeCollection:
        query = FakeQuery()

    class FakeCollections:
        def exists(self, _name: str) -> bool:
            return True

        def get(self, _name: str) -> FakeCollection:
            return FakeCollection()

    class FakeClient:
        collections = FakeCollections()

        def is_ready(self) -> bool:
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        orchestrator_module,
        "get_weaviate_config",
        lambda: {"url": "https://example.weaviate.cloud", "grpc_host": "grpc", "grpc_port": "443", "api_key": "loaded"},
    )
    monkeypatch.setattr(orchestrator_module, "weaviate", object())
    monkeypatch.setattr(HybridOrchestrator, "_connect_weaviate", lambda self, config: FakeClient())
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

    result, limitations = HybridOrchestrator()._execute_weaviate(
        "Show research on sulfites and alcohol headaches",
        {"intent": "scientific_evidence"},
    )

    assert result["retrieval_backend"] == "embedded_fallback"
    assert result["hit_count"] == 1
    assert any("Weaviate query failed" in item for item in limitations)


def test_weaviate_execution_source_uses_near_vector_not_near_text() -> None:
    source = inspect.getsource(HybridOrchestrator._execute_weaviate)
    assert ".near_vector(" in source
    assert ".near_text(" not in source
    assert ".hybrid(" not in source


def test_manifest_schema_design_files_validate_with_explicit_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", REPO_ROOT.as_posix())
    manifest_path = REPO_ROOT / "data/artifact_manifest.example.json"
    specs = load_manifest(manifest_path.as_posix())
    filtered = [spec for spec in specs if spec.artifact_id in {"neo4j_graph_schema_design", "weaviate_schema_design"}]
    statuses = check_all_artifacts(filtered)
    assert all(item.validation_status == "ok" for item in statuses)


def test_missing_artifacts_detected_with_temp_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifact_id": "missing_required_csv",
                "category": "test",
                "local_path": (tmp_path / "does_not_exist.csv").as_posix(),
                "required_for": ["unit"],
                "required": True,
                "expected_type": "csv",
                "min_size_bytes": 1,
                "description": "missing",
            },
            {
                "artifact_id": "missing_optional_md",
                "category": "test",
                "local_path": (tmp_path / "missing.md").as_posix(),
                "required_for": ["unit"],
                "required": False,
                "expected_type": "md",
                "min_size_bytes": 1,
                "description": "missing optional",
            },
        ],
    )

    specs = load_manifest(manifest_path.as_posix())
    statuses = check_all_artifacts(specs)
    summary = summarize_artifacts(statuses)

    assert summary["all_required_available"] is False
    assert summary["missing_required_count"] == 1
    assert "missing_required_csv" in summary["missing_required"]
    assert get_missing_required(statuses) == ["missing_required_csv"]


def test_present_artifacts_validate_for_csv_json_jsonl_md(tmp_path: Path) -> None:
    csv_file = tmp_path / "table.csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")

    json_file = tmp_path / "doc.json"
    json_file.write_text('{"ok": true}\n', encoding="utf-8")

    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")

    md_file = tmp_path / "design.md"
    md_file.write_text("# Design\nArtifact notes\n", encoding="utf-8")

    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifact_id": "csv_ok",
                "category": "test",
                "local_path": csv_file.as_posix(),
                "required_for": ["unit"],
                "required": True,
                "expected_type": "csv",
                "min_size_bytes": 2,
                "description": "csv",
            },
            {
                "artifact_id": "json_ok",
                "category": "test",
                "local_path": json_file.as_posix(),
                "required_for": ["unit"],
                "required": True,
                "expected_type": "json",
                "min_size_bytes": 2,
                "description": "json",
            },
            {
                "artifact_id": "jsonl_ok",
                "category": "test",
                "local_path": jsonl_file.as_posix(),
                "required_for": ["unit"],
                "required": True,
                "expected_type": "jsonl",
                "min_size_bytes": 2,
                "description": "jsonl",
            },
            {
                "artifact_id": "md_ok",
                "category": "test",
                "local_path": md_file.as_posix(),
                "required_for": ["unit"],
                "required": True,
                "expected_type": "md",
                "min_size_bytes": 2,
                "description": "md",
            },
        ],
    )

    specs = load_manifest(manifest_path.as_posix())
    statuses = check_all_artifacts(specs)

    assert all(status.validation_status == "ok" for status in statuses)


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    file_path = tmp_path / "value.txt"
    file_path.write_text("deterministic-hash\n", encoding="utf-8")

    first = sha256(file_path.as_posix())
    second = sha256(file_path.as_posix())

    assert first == second
    assert len(first) == 64


def test_artifact_status_cli_returns_valid_json() -> None:
    cmd = [sys.executable, "backend/scripts/artifact_status.py"]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)

    assert completed.returncode in {0, 2}
    payload = json.loads(completed.stdout.strip())
    assert "all_required_available" in payload
    assert "missing_required" in payload
    assert "available_count" in payload
    assert "missing_count" in payload
    assert "categories" in payload


def test_health_includes_artifact_status() -> None:
    pytest.importorskip("fastapi")
    from api.health import health_check

    payload = health_check()
    components = payload["components"]

    assert "artifact_status" in components
    artifact = components["artifact_status"]
    assert "ok" in artifact
    assert "detail" in artifact
    assert "missing_required_count" in artifact
    assert "missing_required" in artifact


def test_missing_artifacts_degrade_health_without_crash(monkeypatch: Any, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import api.health as health_module

    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
                {
                    "artifact_id": "required_missing",
                    "category": "core_processed_tables",
                    "local_path": (tmp_path / "not_here.csv").as_posix(),
                    "required_for": ["unit"],
                    "required": True,
                "expected_type": "csv",
                "min_size_bytes": 1,
                "description": "missing",
            }
        ],
    )

    monkeypatch.setattr(health_module, "ARTIFACT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(health_module, "_neo4j_probe", lambda: health_module._component(True, "ok"))
    monkeypatch.setattr(health_module, "_weaviate_probe", lambda: health_module._component(True, "ok"))
    monkeypatch.setattr(health_module, "_ollama_probe", lambda: health_module._component(True, "ok"))

    payload = health_module.build_health_payload()

    assert payload["status"] == "degraded"
    artifact = payload["components"]["artifact_status"]
    assert artifact["ok"] is False
    assert artifact["missing_required_count"] == 1
    assert "required_missing" in artifact["missing_required"]
