from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from artifacts.mongodb_store import ARTIFACT_SOURCE, MongoArtifactStore
from scripts import artifact_download_mongodb, artifact_upload_mongodb


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeCollection:
    def __init__(self, find_one_result: Dict[str, Any] | None = None, find_rows: List[Dict[str, Any]] | None = None) -> None:
        self.find_one_result = find_one_result
        self.find_rows = find_rows or []
        self.find_one_queries: List[Dict[str, Any]] = []
        self.find_queries: List[Dict[str, Any]] = []
        self.updates: List[Dict[str, Any]] = []
        self.inserts: List[Dict[str, Any]] = []

    def find_one(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        self.find_one_queries.append(query)
        return self.find_one_result

    def find(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.find_queries.append(query)
        return list(self.find_rows)

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> None:
        self.updates.append({"query": query, "update": update, "upsert": upsert})

    def insert_one(self, record: Dict[str, Any]) -> None:
        self.inserts.append(record)


class _FakeDb:
    def __init__(self) -> None:
        self.collections: Dict[str, _FakeCollection] = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeBucket:
    def __init__(self) -> None:
        self.uploads: List[Dict[str, Any]] = []

    def upload_from_stream(self, filename: str, source: Any, metadata: Dict[str, Any]) -> str:
        payload = source.read()
        self.uploads.append({"filename": filename, "payload": payload, "metadata": metadata})
        return "gridfs-file-id"


def test_sha256_mismatch_raises_error(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        artifact_upload_mongodb.validate_artifact_source(
            {
                "artifact_id": "artifact_1",
                "local_path": artifact.as_posix(),
                "sha256": "not-the-real-sha",
            }
        )


def test_dry_run_upload_does_not_connect_to_mongodb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = "v0.6-test"
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_name": release,
                "artifacts": [
                    {
                        "artifact_id": "artifact_1",
                        "category": "test",
                        "local_path": artifact.as_posix(),
                        "remote_path": f"releases/{release}/payload.csv",
                        "sha256": _sha(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "required": True,
                        "required_for": ["unit"],
                        "upload_strategy": "direct",
                        "validation_status": "ok",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    class _ForbiddenStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("MongoArtifactStore should not be instantiated during dry-run")

    monkeypatch.setattr(artifact_upload_mongodb, "MongoArtifactStore", _ForbiddenStore)
    monkeypatch.setattr(
        artifact_upload_mongodb,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "manifest_path": manifest.as_posix(),
                "release": release,
                "dry_run": True,
                "force": False,
            },
        )(),
    )

    assert artifact_upload_mongodb.main() == 0


def test_metadata_shape_is_written_to_manifest_entries(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    release = "v0.6-test"
    metadata = artifact_upload_mongodb.build_artifact_metadata(
        release=release,
        artifact={
            "artifact_id": "artifact_1",
            "category": "test",
            "local_path": artifact.as_posix(),
            "remote_path": f"releases/{release}/payload.csv",
            "required": True,
            "required_for": ["unit"],
            "upload_strategy": "direct",
            "validation_status": "ok",
        },
        observed_sha256=_sha(artifact),
        observed_size_bytes=artifact.stat().st_size,
    )
    fake_db = _FakeDb()
    fake_bucket = _FakeBucket()
    store = MongoArtifactStore(db=fake_db, gridfs_bucket=fake_bucket)

    record = store.upload_file(artifact.as_posix(), metadata)
    stored = fake_db["artifact_manifest_entries"].updates[0]["update"]["$set"]

    for key in (
        "release",
        "artifact_id",
        "category",
        "local_path",
        "remote_path",
        "sha256",
        "size_bytes",
        "required",
        "required_for",
        "upload_strategy",
        "validation_status",
        "gridfs_file_id",
        "uploaded_at_utc",
        "content_type",
        "source",
    ):
        assert key in stored
    assert stored["release"] == release
    assert stored["gridfs_file_id"] == "gridfs-file-id"
    assert stored["source"] == ARTIFACT_SOURCE
    assert record["status"] == "uploaded"
    assert fake_bucket.uploads[0]["metadata"]["source"] == ARTIFACT_SOURCE


def test_restore_path_is_constructed_from_local_path(tmp_path: Path) -> None:
    output_root = tmp_path / "restore"
    target = artifact_download_mongodb.restore_target_path(
        output_root,
        "data/processed/pbpk/pbpk_parameter_library.csv",
    )
    assert target == output_root / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv"

    with pytest.raises(ValueError, match="unsafe restore path"):
        artifact_download_mongodb.restore_target_path(output_root, "../outside.csv")


def test_restore_downloads_to_output_root_local_path_and_verifies_sha(tmp_path: Path) -> None:
    payload = b"a,b\n1,2\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    output_root = tmp_path / "restore"

    class _Store:
        def __init__(self) -> None:
            self.logged: List[Dict[str, Any]] = []

        def list_release_artifacts(self, release: str, required_only: bool = False) -> List[Dict[str, Any]]:
            return [
                {
                    "release": release,
                    "artifact_id": "artifact_1",
                    "category": "core_processed_tables",
                    "local_path": "data/processed/payload.csv",
                    "sha256": expected_sha,
                    "required": True,
                    "gridfs_file_id": "gridfs-file-id",
                    "source": ARTIFACT_SOURCE,
                }
            ]

        def download_file(self, gridfs_file_id: str, local_path: str, overwrite: bool = False) -> Path:
            target = Path(local_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return target

        def log_restore(self, release: str, report: Dict[str, Any]) -> None:
            self.logged.append({"release": release, "report": report})

    report = artifact_download_mongodb.restore_release(
        _Store(),  # type: ignore[arg-type]
        release="v0.6-test",
        output_root=output_root,
        required_only=True,
        force=False,
    )

    restored = output_root / "data" / "processed" / "payload.csv"
    assert restored.exists()
    assert _sha(restored) == expected_sha
    assert report["restored_count"] == 1
    assert report["all_required_verified"] is True


def test_same_artifact_is_skipped_if_sha256_already_exists(tmp_path: Path) -> None:
    release = "v0.6-test"
    manifest = {"release_name": release, "artifacts": []}
    artifact = tmp_path / "payload.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    sha = _sha(artifact)

    class _Store:
        def __init__(self) -> None:
            self.release_updates: List[Dict[str, Any]] = []

        def find_existing_artifact(self, release: str, artifact_id: str, sha256: str) -> Dict[str, Any]:
            return {"release": release, "artifact_id": artifact_id, "sha256": sha256}

        def upload_file(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise AssertionError("upload_file should not be called for existing sha")

        def create_or_update_release(self, release: str, **kwargs: Any) -> Dict[str, Any]:
            self.release_updates.append({"release": release, **kwargs})
            return {"release": release}

    summary = artifact_upload_mongodb.upload_artifacts(
        _Store(),  # type: ignore[arg-type]
        release=release,
        manifest=manifest,
        manifest_path=tmp_path / "artifact_manifest.json",
        upload_items=[
            {
                "artifact_id": "artifact_1",
                "source_path": artifact.as_posix(),
                "metadata": {
                    "release": release,
                    "artifact_id": "artifact_1",
                    "sha256": sha,
                    "local_path": artifact.as_posix(),
                },
            }
        ],
        force=False,
    )

    assert summary["uploaded_count"] == 0
    assert summary["skipped_existing_count"] == 1
    assert summary["skipped_existing"] == ["artifact_1"]
