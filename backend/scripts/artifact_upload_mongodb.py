#!/usr/bin/env python3
"""Upload release artifacts to MongoDB Atlas GridFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.local_store import resolve_path  # noqa: E402
from artifacts.mongodb_store import (  # noqa: E402
    ARTIFACT_SOURCE,
    MongoArtifactStore,
    clean_text,
    infer_content_type,
    is_forbidden_artifact_path,
)


DEFAULT_RELEASE = "v0.6-chemical-explorer"
DEFAULT_MANIFEST_PATH = "data/releases/v0.6-chemical-explorer/artifact_manifest.json"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    path = resolve_path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact manifest not found: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Artifact manifest must be a JSON object.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Artifact manifest must contain an 'artifacts' list.")
    return payload


def validate_release_name(value: str) -> str:
    release = clean_text(value)
    if not release:
        raise ValueError("release name is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in release):
        raise ValueError("release name contains unsupported characters")
    return release


def validate_artifact_source(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    artifact_id = clean_text(artifact.get("artifact_id")) or clean_text(artifact.get("local_path"))
    local_path = clean_text(artifact.get("local_path"))
    if not artifact_id:
        raise ValueError("manifest artifact is missing artifact_id")
    if not local_path:
        raise ValueError(f"Artifact {artifact_id} is missing local_path")
    if is_forbidden_artifact_path(local_path):
        raise PermissionError(f"Refusing to upload forbidden artifact path: {local_path}")

    source_path = resolve_path(local_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {local_path}")

    observed_sha = sha256_file(source_path)
    expected_sha = clean_text(artifact.get("sha256"))
    if expected_sha and observed_sha != expected_sha:
        raise ValueError(
            f"sha256 mismatch for {artifact_id}: expected {expected_sha}, observed {observed_sha}"
        )
    return {
        "artifact_id": artifact_id,
        "local_path": local_path,
        "source_path": source_path,
        "sha256": observed_sha,
        "size_bytes": int(source_path.stat().st_size),
    }


def build_artifact_metadata(
    *,
    release: str,
    artifact: Mapping[str, Any],
    observed_sha256: str,
    observed_size_bytes: int,
    gridfs_file_id: Any = None,
) -> Dict[str, Any]:
    local_path = clean_text(artifact.get("local_path"))
    metadata: Dict[str, Any] = {
        "release": release,
        "artifact_id": clean_text(artifact.get("artifact_id")) or local_path,
        "category": clean_text(artifact.get("category")),
        "local_path": local_path,
        "remote_path": clean_text(artifact.get("remote_path")),
        "sha256": observed_sha256,
        "size_bytes": int(observed_size_bytes),
        "required": bool_value(artifact.get("required"), default=True),
        "required_for": list(artifact.get("required_for", []) or []),
        "upload_strategy": clean_text(artifact.get("upload_strategy")) or "direct",
        "validation_status": clean_text(artifact.get("validation_status")),
        "gridfs_file_id": gridfs_file_id,
        "uploaded_at_utc": "",
        "content_type": infer_content_type(local_path),
        "source": ARTIFACT_SOURCE,
    }
    return metadata


def collect_upload_candidates(manifest: Mapping[str, Any], *, release: str) -> Dict[str, Any]:
    artifacts = list(manifest.get("artifacts", []) or [])
    upload_items: List[Dict[str, Any]] = []
    required_missing: List[str] = []
    skipped_optional_missing: List[str] = []
    skipped_forbidden: List[str] = []

    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact = dict(raw_artifact)
        artifact_id = clean_text(artifact.get("artifact_id")) or clean_text(artifact.get("local_path"))
        local_path = clean_text(artifact.get("local_path"))
        required = bool_value(artifact.get("required"), default=True)

        if local_path and is_forbidden_artifact_path(local_path):
            skipped_forbidden.append(artifact_id or local_path)
            continue

        try:
            validated = validate_artifact_source(artifact)
        except FileNotFoundError:
            if required:
                required_missing.append(artifact_id or local_path)
            else:
                skipped_optional_missing.append(artifact_id or local_path)
            continue
        except PermissionError:
            skipped_forbidden.append(artifact_id or local_path)
            continue

        metadata = build_artifact_metadata(
            release=release,
            artifact=artifact,
            observed_sha256=validated["sha256"],
            observed_size_bytes=validated["size_bytes"],
        )
        upload_items.append(
            {
                "artifact_id": validated["artifact_id"],
                "local_path": validated["local_path"],
                "source_path": validated["source_path"].as_posix(),
                "sha256": validated["sha256"],
                "size_bytes": validated["size_bytes"],
                "metadata": metadata,
            }
        )

    return {
        "upload_items": upload_items,
        "required_missing": sorted(set(required_missing)),
        "skipped_optional_missing": sorted(set(skipped_optional_missing)),
        "skipped_forbidden": sorted(set(skipped_forbidden)),
        "manifest_artifact_count": len([item for item in artifacts if isinstance(item, Mapping)]),
    }


def upload_artifacts(
    store: MongoArtifactStore,
    *,
    release: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    upload_items: List[Mapping[str, Any]],
    force: bool,
) -> Dict[str, Any]:
    uploaded: List[str] = []
    skipped_existing: List[str] = []

    for item in upload_items:
        metadata = dict(item["metadata"])
        artifact_id = clean_text(metadata.get("artifact_id"))
        sha256 = clean_text(metadata.get("sha256"))
        existing = store.find_existing_artifact(release, artifact_id, sha256)
        if existing and not force:
            skipped_existing.append(artifact_id)
            continue

        record = store.upload_file(
            clean_text(item.get("source_path")),
            metadata,
            force=force,
        )
        if clean_text(record.get("status")) == "skipped_existing":
            skipped_existing.append(artifact_id)
        else:
            uploaded.append(artifact_id)

    summary = {
        "candidate_count": len(upload_items),
        "uploaded_count": len(uploaded),
        "uploaded": uploaded,
        "skipped_existing_count": len(skipped_existing),
        "skipped_existing": skipped_existing,
        "force": bool(force),
    }
    store.create_or_update_release(
        release,
        manifest=manifest,
        manifest_path=manifest_path.as_posix(),
        upload_summary=summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload release artifacts to MongoDB Atlas GridFS")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH, help="Release artifact manifest path")
    parser.add_argument("--release", default="", help="Release name; defaults to manifest release_name")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without connecting to MongoDB")
    parser.add_argument("--force", action="store_true", help="Upload even when same release/artifact/sha already exists")
    return parser.parse_args()


def _json_default(value: Any) -> str:
    return str(value)


def main() -> int:
    args = parse_args()
    try:
        manifest_path = resolve_path(args.manifest_path)
        manifest = load_manifest(manifest_path)
        release = validate_release_name(args.release or clean_text(manifest.get("release_name")) or DEFAULT_RELEASE)
        plan = collect_upload_candidates(manifest, release=release)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "backend": "mongodb",
                    "dry_run": bool(getattr(args, "dry_run", False)),
                    "error": True,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    if plan["required_missing"]:
        print(
            json.dumps(
                {
                    "backend": "mongodb",
                    "release": release,
                    "manifest_path": manifest_path.as_posix(),
                    "dry_run": bool(args.dry_run),
                    "error": True,
                    "message": "Missing required artifacts; upload aborted.",
                    "required_missing": plan["required_missing"],
                    "skipped_optional_missing": plan["skipped_optional_missing"],
                    "skipped_forbidden": plan["skipped_forbidden"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    payload: Dict[str, Any] = {
        "backend": "mongodb",
        "release": release,
        "manifest_path": manifest_path.as_posix(),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "manifest_artifact_count": plan["manifest_artifact_count"],
        "candidate_count": len(plan["upload_items"]),
        "uploaded_count": 0,
        "uploaded": [],
        "skipped_existing_count": 0,
        "skipped_existing": [],
        "skipped_optional_missing": plan["skipped_optional_missing"],
        "skipped_forbidden": plan["skipped_forbidden"],
        "source": ARTIFACT_SOURCE,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return 0

    store: MongoArtifactStore | None = None
    try:
        store = MongoArtifactStore()
        store.ping()
        summary = upload_artifacts(
            store,
            release=release,
            manifest=manifest,
            manifest_path=manifest_path,
            upload_items=plan["upload_items"],
            force=bool(args.force),
        )
        payload.update(summary)
    except Exception as exc:
        payload.update({"error": True, "message": str(exc)})
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return 1
    finally:
        if store is not None:
            store.close()

    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
