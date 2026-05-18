from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import check_all_artifacts, get_missing_required, load_manifest, summarize_artifacts
from artifacts.local_store import sha256


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
                "category": "test",
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
