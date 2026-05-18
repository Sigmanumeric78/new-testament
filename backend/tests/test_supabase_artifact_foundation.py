from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from artifacts.release_manifest import (
    build_release_bundle,
    build_remote_path,
)
from artifacts.supabase_store import SupabaseArtifactStore
from scripts import artifact_download_supabase, artifact_upload_supabase
from scripts.artifact_verify_release import verify_release_manifest
from utils.config import get_supabase_config


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_release_manifest(release_dir: Path, release_name: str, artifact_local_path: str, artifact_sha: str) -> None:
    payload = {
        "release_name": release_name,
        "generated_at_utc": "2026-05-18T00:00:00+00:00",
        "artifact_count": 1,
        "required_artifact_count": 1,
        "available_artifact_count": 1,
        "missing_artifact_count": 0,
        "artifacts": [
            {
                "artifact_id": "artifact_1",
                "category": "test",
                "local_path": artifact_local_path,
                "remote_path": f"releases/{release_name}/{artifact_local_path}",
                "size_bytes": 5,
                "sha256": artifact_sha,
                "required": True,
                "required_for": ["unit"],
                "available": True,
                "validation_status": "ok",
            }
        ],
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (release_dir / "release_metadata.json").write_text(
        json.dumps({"release_name": release_name}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (release_dir / "checksums.sha256").write_text("dummy  dummy\n", encoding="utf-8")


def test_supabase_config_fails_clearly_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
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
    )
    payload2 = build_release_bundle(
        release_name="v0.1-local-intelligence",
        manifest_path=local_manifest,
        output_dir=out2,
        allow_missing=False,
        generated_at_utc="2026-05-18T00:00:00+00:00",
    )

    assert payload1 == payload2


def test_remote_path_mapping_is_deterministic() -> None:
    local = "data/processed/pbpk/pbpk_parameter_library.csv"
    expected = "releases/v0.1-local-intelligence/data/processed/pbpk/pbpk_parameter_library.csv"
    assert build_remote_path("v0.1-local-intelligence", local) == expected


def test_dry_run_upload_does_not_call_supabase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    _write_release_manifest(release_dir, release, artifact.as_posix(), _sha(artifact))

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


def test_dry_run_download_does_not_call_supabase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    _write_release_manifest(release_dir, release, artifact.as_posix(), _sha(artifact))

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
            {"release": release, "dry_run": True, "execute": False, "overwrite": False},
        )(),
    )

    rc = artifact_download_supabase.main()
    assert rc == 0


def test_verify_detects_missing_artifact(tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    missing_local_path = (tmp_path / "missing.csv").as_posix()

    release_dir = tmp_path / "release"
    _write_release_manifest(release_dir, release, missing_local_path, "deadbeef")

    payload = verify_release_manifest(release, release_dir / "artifact_manifest.json")
    assert payload["all_required_valid"] is False
    assert payload["missing"]


def test_verify_detects_checksum_mismatch(tmp_path: Path) -> None:
    release = "v0.1-local-intelligence"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    _write_release_manifest(release_dir, release, artifact.as_posix(), "badchecksum")

    payload = verify_release_manifest(release, release_dir / "artifact_manifest.json")
    assert payload["all_required_valid"] is False
    assert payload["checksum_mismatches"]


def test_env_is_never_upload_candidate() -> None:
    assert artifact_upload_supabase._is_forbidden_upload_path(".env") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("config/.env") is True


def test_frontend_node_modules_and_dist_never_upload_candidates() -> None:
    assert artifact_upload_supabase._is_forbidden_upload_path("frontend/node_modules/a") is True
    assert artifact_upload_supabase._is_forbidden_upload_path("frontend/dist/index.js") is True
