from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.export_neo4j_data import (  # noqa: E402
    DEFAULT_RELEASE,
    _to_json_safe,
    expected_neo4j_files,
    runtime_export_root as neo_runtime_export_root,
)
from scripts.export_weaviate_data import (  # noqa: E402
    expected_weaviate_files,
    runtime_export_root as weav_runtime_export_root,
)
from scripts.verify_runtime_exports import expected_runtime_files, verify_runtime_exports  # noqa: E402


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runtime_export_path_generation_is_deterministic(tmp_path: Path) -> None:
    release = "v0.6-chemical-explorer"
    neo_root = neo_runtime_export_root(release, output_root=tmp_path.as_posix())
    weav_root = weav_runtime_export_root(release, output_root=tmp_path.as_posix())

    assert neo_root == weav_root
    assert neo_root.as_posix().endswith(f"/{release}")


def test_expected_file_lists_are_complete() -> None:
    neo = expected_neo4j_files()
    weav = expected_weaviate_files()
    combined = expected_runtime_files()

    assert neo == [
        "cypher_counts.json",
        "node_counts.json",
        "relationship_counts.json",
        "schema_snapshot.json",
        "graph_export_manifest.json",
        "neo4j_dump_instructions.md",
    ]
    assert weav == [
        "collection_counts.json",
        "schema_snapshot.json",
        "backup_manifest.json",
        "weaviate_backup_instructions.md",
    ]
    assert len(combined["neo4j"]) == len(neo)
    assert len(combined["weaviate"]) == len(weav)


def test_verify_generates_manifest_and_checksums_when_files_present(tmp_path: Path) -> None:
    release = "test-release"
    runtime_root = tmp_path / release
    (runtime_root / "neo4j").mkdir(parents=True, exist_ok=True)
    (runtime_root / "weaviate").mkdir(parents=True, exist_ok=True)

    for rel in expected_runtime_files()["neo4j"] + expected_runtime_files()["weaviate"]:
        file_path = runtime_root / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == ".md":
            file_path.write_text("# snapshot\n", encoding="utf-8")
        elif rel == "neo4j/graph_export_manifest.json":
            file_path.write_text(json.dumps({"status": "ok", "read_only": True}) + "\n", encoding="utf-8")
        elif rel == "weaviate/backup_manifest.json":
            file_path.write_text(json.dumps({"status": "ok", "read_only_by_default": True}) + "\n", encoding="utf-8")
        else:
            file_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    result = verify_runtime_exports(release=release, output_root=tmp_path.as_posix())
    manifest_path = runtime_root / "artifact_manifest.runtime.json"
    checksums_path = runtime_root / "checksums.sha256"

    assert result["neo4j_export_ok"] is True
    assert result["weaviate_export_ok"] is True
    assert result["checksums_written"] is True
    assert result["safe_for_supabase_upload"] is True
    assert manifest_path.exists()
    assert checksums_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["release"] == release
    assert manifest_payload["missing_files"] == []

    checksum_text = checksums_path.read_text(encoding="utf-8")
    assert "artifact_manifest.runtime.json" in checksum_text


def test_verify_detects_missing_files(tmp_path: Path) -> None:
    release = "missing-release"
    result = verify_runtime_exports(release=release, output_root=tmp_path.as_posix())

    assert result["neo4j_export_ok"] is False
    assert result["weaviate_export_ok"] is False
    assert result["checksums_written"] is True
    assert result["safe_for_supabase_upload"] is False

    manifest_path = tmp_path / release / "artifact_manifest.runtime.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["missing_files"]


def test_sanitizer_handles_datetime_like_object() -> None:
    class FakeDateTime:
        def iso_format(self) -> str:
            return "2026-05-19T12:00:00+00:00"

    safe = _to_json_safe({"dt": FakeDateTime()})
    assert safe["dt"] == "2026-05-19T12:00:00+00:00"


def test_sanitizer_handles_nested_structures() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird-object"

    payload = {
        "a": [1, {"b": (2, 3)}, {4, 5}],
        "c": Weird(),
    }
    safe = _to_json_safe(payload)
    assert safe["a"][0] == 1
    assert safe["a"][1]["b"] == [2, 3]
    assert sorted(safe["a"][2]) == [4, 5]
    assert safe["c"] == "weird-object"


def test_verify_uses_manifest_status_for_neo4j_and_weaviate(tmp_path: Path) -> None:
    release = "status-check"
    runtime_root = tmp_path / release
    (runtime_root / "neo4j").mkdir(parents=True, exist_ok=True)
    (runtime_root / "weaviate").mkdir(parents=True, exist_ok=True)

    for rel in expected_runtime_files()["neo4j"] + expected_runtime_files()["weaviate"]:
        file_path = runtime_root / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == ".md":
            file_path.write_text("# snapshot\n", encoding="utf-8")
        elif rel == "neo4j/graph_export_manifest.json":
            file_path.write_text(json.dumps({"status": "unavailable"}) + "\n", encoding="utf-8")
        elif rel == "weaviate/backup_manifest.json":
            file_path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
        else:
            file_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    result = verify_runtime_exports(release=release, output_root=tmp_path.as_posix())
    assert result["neo4j_export_ok"] is False
    assert result["weaviate_export_ok"] is True
    assert result["safe_for_supabase_upload"] is False
    assert result["neo4j_manifest_status"] == "unavailable"


def test_verify_safe_for_supabase_upload_true_only_when_both_ok(tmp_path: Path) -> None:
    release = "status-ok"
    runtime_root = tmp_path / release
    (runtime_root / "neo4j").mkdir(parents=True, exist_ok=True)
    (runtime_root / "weaviate").mkdir(parents=True, exist_ok=True)

    for rel in expected_runtime_files()["neo4j"] + expected_runtime_files()["weaviate"]:
        file_path = runtime_root / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == ".md":
            file_path.write_text("# snapshot\n", encoding="utf-8")
        elif rel == "neo4j/graph_export_manifest.json":
            file_path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
        elif rel == "weaviate/backup_manifest.json":
            file_path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
        else:
            file_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    result = verify_runtime_exports(release=release, output_root=tmp_path.as_posix())
    assert result["neo4j_export_ok"] is True
    assert result["weaviate_export_ok"] is True
    assert result["safe_for_supabase_upload"] is True


def test_scripts_contain_no_hardcoded_passwords() -> None:
    files = [
        BACKEND_ROOT / "scripts" / "export_neo4j_data.py",
        BACKEND_ROOT / "scripts" / "export_weaviate_data.py",
        BACKEND_ROOT / "scripts" / "export_runtime_artifacts.sh",
        BACKEND_ROOT / "scripts" / "verify_runtime_exports.py",
    ]
    forbidden_markers = [
        "change_me",
        "password=\"",
        "password='",
        "neo4j_password=",
        "supabase_service_role_key=",
    ]

    for path in files:
        text = _read(path).lower()
        for marker in forbidden_markers:
            assert marker not in text, f"forbidden marker found in {path}: {marker}"


def test_scripts_default_to_safe_read_only_modes() -> None:
    neo_text = _read(BACKEND_ROOT / "scripts" / "export_neo4j_data.py").lower()
    weav_text = _read(BACKEND_ROOT / "scripts" / "export_weaviate_data.py").lower()

    for keyword in (" merge ", " delete ", " detach ", " set "):
        assert keyword not in neo_text
    for keyword in (" merge ", " delete ", " detach ", " set "):
        assert keyword not in weav_text

    assert '"read_only": true' in neo_text
    assert "--execute-backup" in weav_text
    assert "store_true" in weav_text
    assert "metadata-only" in weav_text
    assert "execute_backup: bool = false" in weav_text


def test_docs_exist() -> None:
    path = REPO_ROOT / "docs" / "runtime_data_export_plan.md"
    assert path.exists()
    text = _read(path).lower()
    assert "docker images code-only" in text or "code-only" in text
    assert "mongodb atlas is" in text and "not replacing neo4j or weaviate" in text
