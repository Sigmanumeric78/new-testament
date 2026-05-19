from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from artifacts.release_manifest import build_release_bundle, build_remote_path
from artifacts.supabase_store import SupabaseArtifactStore
from scripts import artifact_download_supabase, artifact_upload_supabase
from scripts.artifact_verify_release import verify_release_manifest
from utils.config import get_supabase_config
import utils.config as config_module


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_release_bundle_files(release_dir: Path, payload: Dict[str, Any]) -> None:
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (release_dir / "release_metadata.json").write_text(
        json.dumps({"release_name": payload.get("release_name", "test")}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (release_dir / "checksums.sha256").write_text("dummy  dummy\n", encoding="utf-8")


def _write_chunk_manifest(
    *,
    original_path: str,
    original_bytes: bytes,
    chunk_size: int,
    chunk_dir: Path,
    release_name: str,
) -> Dict[str, Any]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for idx in range(0, len(original_bytes), chunk_size):
        payload = original_bytes[idx : idx + chunk_size]
        part_name = f"part_{idx // chunk_size:05d}"
        part_path = chunk_dir / part_name
        part_path.write_bytes(payload)
        chunks.append(
            {
                "part_name": part_name,
                "size_bytes": len(payload),
                "sha256": _sha(part_path),
                "local_path": part_path.as_posix(),
                "remote_path": f"releases/{release_name}/chunks/{original_path}/{part_name}",
            }
        )

    manifest = {
        "original_path": original_path,
        "original_size_bytes": len(original_bytes),
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "chunk_size_bytes": chunk_size,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "restore_command_hint": "PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py --execute",
    }
    (chunk_dir / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def test_supabase_config_fails_clearly_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ARTIFACT_BUCKET", raising=False)

    optional = get_supabase_config(require=False)
    assert optional["artifact_bucket"] == "alcohol-intelligence-artifacts"

    with pytest.raises(ValueError) as exc:
        _ = get_supabase_config(require=True)
    assert "SUPABASE_URL" in str(exc.value)
    assert "SUPABASE_SERVICE_ROLE_KEY" in str(exc.value)


def test_supabase_sdk_lazy_import_does_not_break_normal_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_ARTIFACT_BUCKET", "alcohol-intelligence-artifacts")
    store = SupabaseArtifactStore(require_credentials=False)
    assert store.bucket == "alcohol-intelligence-artifacts"


def test_release_manifest_generation_is_deterministic_from_fake_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    local_manifest = tmp_path / "artifact_manifest.local.json"
    local_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "pbpk_library",
                        "category": "core_processed_tables",
                        "local_path": artifact.as_posix(),
                        "required_for": ["pbpk"],
                        "required": True,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha(artifact),
                        "exists": True,
                        "validation_status": "ok",
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out1 = tmp_path / "release1"
    out2 = tmp_path / "release2"
    payload1 = build_release_bundle(
        release_name="v0.1-local-intelligence",
        manifest_path=local_manifest,
        output_dir=out1,
        allow_missing=False,
        generated_at_utc="2026-05-18T00:00:00+00:00",
        max_upload_mb=45,
    )
    payload2 = build_release_bundle(
        release_name="v0.1-local-intelligence",
        manifest_path=local_manifest,
        output_dir=out2,
        allow_missing=False,
        generated_at_utc="2026-05-18T00:00:00+00:00",
        max_upload_mb=45,
    )

    assert payload1 == payload2


def test_remote_path_mapping_is_deterministic() -> None:
    local = "data/processed/pbpk/pbpk_parameter_library.csv"
    expected = "releases/v0.1-local-intelligence/data/processed/pbpk/pbpk_parameter_library.csv"
    assert build_remote_path("v0.1-local-intelligence", local) == expected


def test_oversized_artifact_detection_and_chunk_manifest(tmp_path: Path) -> None:
    release = "v0.6-chemical-explorer"
    payload_path = tmp_path / "data" / "processed" / "weaviate" / "embedded" / "big.parquet"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"A" * (2 * 1024 * 1024 + 64))

    local_manifest = tmp_path / "artifact_manifest.local.json"
    local_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "scientific_embeddings",
                        "category": "weaviate_embeddings",
                        "local_path": payload_path.as_posix(),
                        "required_for": ["weaviate"],
                        "required": True,
                        "size_bytes": payload_path.stat().st_size,
                        "sha256": _sha(payload_path),
                        "exists": True,
                        "validation_status": "ok",
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    bundle = build_release_bundle(
        release_name=release,
        manifest_path=local_manifest,
        output_dir=tmp_path / "release",
        allow_missing=False,
        generated_at_utc="2026-05-19T00:00:00+00:00",
        max_upload_mb=1,
    )
    artifact = bundle["artifacts"][0]
    assert artifact["upload_strategy"] == "chunked"
    assert artifact["direct_upload"] is False
    assert artifact["chunk_count"] >= 3
    assert artifact["original_sha256"] == _sha(payload_path)
    manifest_path = Path(artifact["chunk_manifest_path"])
    assert manifest_path.exists()
    chunk_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_names = [entry["part_name"] for entry in chunk_payload["chunks"]]
    assert part_names == sorted(part_names)
    assert part_names[0] == "part_00000"


def test_oversized_detection_uses_env_max_upload_mb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPABASE_MAX_UPLOAD_MB", "1")
    release = "v0.6-env"
    payload_path = tmp_path / "payload.parquet"
    payload_path.write_bytes(b"C" * (2 * 1024 * 1024 + 32))

    local_manifest = tmp_path / "artifact_manifest.local.json"
    local_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "artifact_env_limit",
                        "category": "test",
                        "local_path": payload_path.as_posix(),
                        "required_for": ["weaviate"],
                        "required": True,
                        "size_bytes": payload_path.stat().st_size,
                        "sha256": _sha(payload_path),
                        "exists": True,
                        "validation_status": "ok",
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    bundle = build_release_bundle(
        release_name=release,
        manifest_path=local_manifest,
        output_dir=tmp_path / "release",
        allow_missing=False,
        generated_at_utc="2026-05-19T00:00:00+00:00",
    )
    assert bundle["artifacts"][0]["upload_strategy"] == "chunked"


def test_upload_candidates_replace_oversized_direct_file(tmp_path: Path) -> None:
    release = "v0.6-chemical-explorer"
    payload_path = tmp_path / "data" / "processed" / "weaviate" / "embedded" / "big.parquet"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"B" * (2 * 1024 * 1024 + 32))

    local_manifest = tmp_path / "artifact_manifest.local.json"
    local_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "scientific_embeddings",
                        "category": "weaviate_embeddings",
                        "local_path": payload_path.as_posix(),
                        "required_for": ["weaviate"],
                        "required": True,
                        "size_bytes": payload_path.stat().st_size,
                        "sha256": _sha(payload_path),
                        "exists": True,
                        "validation_status": "ok",
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    release_dir = tmp_path / "release"
    bundle = build_release_bundle(
        release_name=release,
        manifest_path=local_manifest,
        output_dir=release_dir,
        allow_missing=False,
        generated_at_utc="2026-05-19T00:00:00+00:00",
        max_upload_mb=1,
    )
    plan = artifact_upload_supabase.collect_upload_candidates(bundle, release_dir)
    upload_locals = [entry["local_path"] for entry in plan["upload_items"]]
    assert payload_path.as_posix() not in upload_locals
    assert any(path.endswith("chunk_manifest.json") for path in upload_locals)
    assert any(Path(path).name.startswith("part_") for path in upload_locals)
    assert "scientific_embeddings" in plan["skipped_oversized_direct_upload"]
    assert plan["chunked_upload_count"] > 0


def test_dry_run_upload_does_not_call_supabase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "artifact_1",
                "category": "test",
                "local_path": artifact.as_posix(),
                "remote_path": f"releases/{release}/{artifact.as_posix()}",
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "direct",
                "direct_upload": True,
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    class _ForbiddenStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Supabase store should not be instantiated during dry-run")

    monkeypatch.setattr(artifact_upload_supabase, "SupabaseArtifactStore", _ForbiddenStore)
    monkeypatch.setattr(artifact_upload_supabase, "default_release_dir", lambda _release: release_dir)
    monkeypatch.setattr(
        artifact_upload_supabase,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {"release": release, "dry_run": True, "execute": False, "overwrite": False},
        )(),
    )

    rc = artifact_upload_supabase.main()
    assert rc == 0


def test_dry_run_upload_supports_overwrite_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "artifact_1",
                "category": "test",
                "local_path": artifact.as_posix(),
                "remote_path": f"releases/{release}/{artifact.as_posix()}",
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "direct",
                "direct_upload": True,
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    monkeypatch.setattr(artifact_upload_supabase, "default_release_dir", lambda _release: release_dir)
    monkeypatch.setattr(
        artifact_upload_supabase,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {"release": release, "dry_run": True, "execute": False, "overwrite": True},
        )(),
    )

    rc = artifact_upload_supabase.main()
    assert rc == 0


def test_dry_run_download_does_not_call_supabase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "artifact_1",
                "category": "test",
                "local_path": artifact.as_posix(),
                "remote_path": f"releases/{release}/{artifact.as_posix()}",
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "direct",
                "direct_upload": True,
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    class _ForbiddenStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Supabase store should not be instantiated during dry-run")

    monkeypatch.setattr(artifact_download_supabase, "SupabaseArtifactStore", _ForbiddenStore)
    monkeypatch.setattr(artifact_download_supabase, "default_release_dir", lambda _release: release_dir)
    monkeypatch.setattr(
        artifact_download_supabase,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "release": release,
                "dry_run": True,
                "execute": False,
                "overwrite": False,
                "workspace_dir": "",
                "all_artifacts": True,
                "runtime_only": False,
            },
        )(),
    )

    rc = artifact_download_supabase.main()
    assert rc == 0


def test_download_plan_runtime_only_filters_non_runtime(tmp_path: Path) -> None:
    runtime_file = tmp_path / "backend" / "rag" / "weaviate" / "weaviate_schema_design.md"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("schema", encoding="utf-8")
    non_runtime_file = tmp_path / "data" / "interim" / "ignored.json"
    non_runtime_file.parent.mkdir(parents=True, exist_ok=True)
    non_runtime_file.write_text('{"ok":true}', encoding="utf-8")

    manifest = {
        "release_name": "v0.6",
        "artifacts": [
            {
                "artifact_id": "weaviate_schema_design",
                "category": "weaviate_schema_design_inputs",
                "local_path": runtime_file.as_posix(),
                "remote_path": "releases/v0.6/backend/rag/weaviate/weaviate_schema_design.md",
                "required": True,
                "available": True,
                "sha256": _sha(runtime_file),
                "upload_strategy": "direct",
            },
            {
                "artifact_id": "interim_noise",
                "category": "validation_reports",
                "local_path": non_runtime_file.as_posix(),
                "remote_path": "releases/v0.6/data/interim/ignored.json",
                "required": False,
                "available": True,
                "sha256": _sha(non_runtime_file),
                "upload_strategy": "direct",
            },
        ],
    }
    plan = artifact_download_supabase.compute_download_plan(
        manifest,
        runtime_only=True,
        workspace_dir=tmp_path / "workspace",
    )
    assert plan["selected_artifact_count"] == 1
    assert plan["skipped_non_runtime_count"] == 1


def test_execute_download_supports_overwrite_with_mock_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.6"
    workspace = tmp_path / "workspace"
    local_target = tmp_path / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv"

    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "pbpk_library",
                "category": "core_processed_tables",
                "local_path": local_target.as_posix(),
                "remote_path": f"releases/{release}/data/processed/pbpk/pbpk_parameter_library.csv",
                "required": True,
                "available": True,
                "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
                "upload_strategy": "direct",
            }
        ],
    }

    class _Store:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: List[tuple[str, str, bool]] = []

        def download_file(self, remote_path: str, local_path: str, overwrite: bool = False) -> Path:
            self.calls.append((remote_path, local_path, overwrite))
            target = Path(local_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if remote_path.endswith("artifact_manifest.json"):
                target.write_text(json.dumps(manifest_payload), encoding="utf-8")
            elif remote_path.endswith(".csv"):
                target.write_bytes(b"a,b\n1,2\n")
            else:
                target.write_text("{}", encoding="utf-8")
            return target

    monkeypatch.setattr(artifact_download_supabase, "SupabaseArtifactStore", _Store)
    monkeypatch.setattr(
        artifact_download_supabase,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "release": release,
                "dry_run": False,
                "execute": True,
                "overwrite": True,
                "workspace_dir": workspace.as_posix(),
                "all_artifacts": True,
                "runtime_only": False,
            },
        )(),
    )

    rc = artifact_download_supabase.main()
    assert rc == 0
    assert local_target.exists()


def test_download_chunk_reassembly_validates_sha(tmp_path: Path) -> None:
    original_path = (tmp_path / "data" / "processed" / "weaviate" / "embedded" / "big.parquet").as_posix()
    original_bytes = b"0123456789" * 1000
    chunk_dir = tmp_path / "data" / "chunks" / "v0.6" / "data/processed/weaviate/embedded/big.parquet"
    chunk_manifest = _write_chunk_manifest(
        original_path=original_path,
        original_bytes=original_bytes,
        chunk_size=1024,
        chunk_dir=chunk_dir,
        release_name="v0.6",
    )

    restored_path = artifact_download_supabase.reassemble_chunked_artifact(chunk_manifest, overwrite=True)
    assert restored_path.read_bytes() == original_bytes
    assert _sha(restored_path) == chunk_manifest["original_sha256"]


def test_verify_detects_missing_chunk(tmp_path: Path) -> None:
    release = "v0.6"
    original_path = (tmp_path / "target.parquet").as_posix()
    chunk_dir = tmp_path / "chunk_dir"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_manifest = {
        "original_path": original_path,
        "original_size_bytes": 10,
        "original_sha256": "abc",
        "chunk_size_bytes": 10,
        "chunk_count": 1,
        "chunks": [
            {
                "part_name": "part_00000",
                "sha256": "abc",
                "size_bytes": 10,
                "local_path": (chunk_dir / "part_00000").as_posix(),
                "remote_path": "releases/v0.6/chunks/target.parquet/part_00000",
            }
        ],
    }
    chunk_manifest_path = chunk_dir / "chunk_manifest.json"
    chunk_manifest_path.write_text(json.dumps(chunk_manifest, indent=2, sort_keys=True), encoding="utf-8")

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "big_one",
                "category": "test",
                "local_path": original_path,
                "remote_path": "releases/v0.6/target.parquet",
                "size_bytes": 10,
                "sha256": "abc",
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "chunked",
                "direct_upload": False,
                "chunk_manifest_path": chunk_manifest_path.as_posix(),
                "chunk_manifest_remote_path": "releases/v0.6/chunks/target.parquet/chunk_manifest.json",
                "chunk_count": 1,
                "original_sha256": "abc",
                "original_size_bytes": 10,
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    payload = verify_release_manifest(release, release_dir / "artifact_manifest.json", runtime_only=False)
    assert payload["all_required_valid"] is False
    assert any("part_00000" in item for item in payload["missing"])


def test_verify_detects_chunk_checksum_mismatch(tmp_path: Path) -> None:
    release = "v0.6"
    original_path = (tmp_path / "target.parquet").as_posix()
    chunk_dir = tmp_path / "chunk_dir"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    part_path = chunk_dir / "part_00000"
    part_path.write_bytes(b"not-matching")
    chunk_manifest = {
        "original_path": original_path,
        "original_size_bytes": 12,
        "original_sha256": "abc",
        "chunk_size_bytes": 12,
        "chunk_count": 1,
        "chunks": [
            {
                "part_name": "part_00000",
                "sha256": "deadbeef",
                "size_bytes": part_path.stat().st_size,
                "local_path": part_path.as_posix(),
                "remote_path": "releases/v0.6/chunks/target.parquet/part_00000",
            }
        ],
    }
    chunk_manifest_path = chunk_dir / "chunk_manifest.json"
    chunk_manifest_path.write_text(json.dumps(chunk_manifest, indent=2, sort_keys=True), encoding="utf-8")

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "big_one",
                "category": "test",
                "local_path": original_path,
                "remote_path": "releases/v0.6/target.parquet",
                "size_bytes": 12,
                "sha256": "abc",
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "chunked",
                "direct_upload": False,
                "chunk_manifest_path": chunk_manifest_path.as_posix(),
                "chunk_manifest_remote_path": "releases/v0.6/chunks/target.parquet/chunk_manifest.json",
                "chunk_count": 1,
                "original_sha256": "abc",
                "original_size_bytes": 12,
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    payload = verify_release_manifest(release, release_dir / "artifact_manifest.json", runtime_only=False)
    assert payload["all_required_valid"] is False
    assert payload["checksum_mismatches"]


def test_verify_chunked_restorable_when_chunks_are_present(tmp_path: Path) -> None:
    release = "v0.6"
    original_path = (tmp_path / "target.parquet").as_posix()
    original_bytes = b"ABCDEFGHIJ"
    chunk_dir = tmp_path / "chunk_dir"
    chunk_manifest = _write_chunk_manifest(
        original_path=original_path,
        original_bytes=original_bytes,
        chunk_size=4,
        chunk_dir=chunk_dir,
        release_name=release,
    )
    chunk_manifest_path = chunk_dir / "chunk_manifest.json"

    release_dir = tmp_path / "release"
    manifest_payload = {
        "release_name": release,
        "artifacts": [
            {
                "artifact_id": "big_one",
                "category": "test",
                "local_path": original_path,
                "remote_path": "releases/v0.6/target.parquet",
                "size_bytes": len(original_bytes),
                "sha256": hashlib.sha256(original_bytes).hexdigest(),
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
                "upload_strategy": "chunked",
                "direct_upload": False,
                "chunk_manifest_path": chunk_manifest_path.as_posix(),
                "chunk_manifest_remote_path": "releases/v0.6/chunks/target.parquet/chunk_manifest.json",
                "chunk_count": chunk_manifest["chunk_count"],
                "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "original_size_bytes": len(original_bytes),
            }
        ],
    }
    _write_release_bundle_files(release_dir, manifest_payload)

    payload = verify_release_manifest(release, release_dir / "artifact_manifest.json", runtime_only=False)
    assert payload["all_required_valid"] is True
    assert "big_one" in payload["restorable_chunked"]


def test_env_and_generated_paths_are_never_upload_candidates() -> None:
    assert artifact_upload_supabase._is_forbidden_upload_path(".env") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("config/.env") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("frontend/node_modules/a") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("frontend/dist/index.js") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("data/interim/reasoning/app_cli_run_log.jsonl") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("backend/__pycache__/a.pyc") is True
