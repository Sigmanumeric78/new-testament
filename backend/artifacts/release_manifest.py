"""Release manifest generation for artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from artifacts.artifact_manager import check_all_artifacts, load_manifest, write_local_manifest
from artifacts.local_store import get_project_root, resolve_path, sha256 as compute_sha256, size_bytes as compute_size_bytes

DEFAULT_SUPABASE_MAX_UPLOAD_MB = 45


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return ""
    return text


def build_remote_path(release_name: str, local_path: str) -> str:
    normalized_release = _clean_text(release_name).strip("/")
    normalized_local = local_path.replace("\\", "/").lstrip("/")
    return f"releases/{normalized_release}/{normalized_local}"


def get_max_upload_bytes(max_upload_mb: float | None = None) -> int:
    if max_upload_mb is None:
        raw = _clean_text(os.getenv("SUPABASE_MAX_UPLOAD_MB", ""))
        if raw:
            try:
                parsed = float(raw)
            except ValueError:
                parsed = float(DEFAULT_SUPABASE_MAX_UPLOAD_MB)
        else:
            parsed = float(DEFAULT_SUPABASE_MAX_UPLOAD_MB)
    else:
        parsed = float(max_upload_mb)

    if parsed <= 0:
        parsed = float(DEFAULT_SUPABASE_MAX_UPLOAD_MB)
    return int(parsed * 1024 * 1024)


def _to_repo_relative(path: Path) -> str:
    root = get_project_root()
    normalized = path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return normalized.lstrip("/")


def chunk_artifact_for_release(
    *,
    release_name: str,
    local_path: str,
    original_size_bytes: int,
    original_sha256: str,
    chunk_size_bytes: int,
) -> Dict[str, Any]:
    source_path = resolve_path(local_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Cannot chunk missing artifact: {local_path}")

    relative_original = _to_repo_relative(source_path)
    chunk_dir = get_project_root() / "data" / "chunks" / release_name / relative_original
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Ensure deterministic chunk directory contents across reruns.
    for stale in chunk_dir.glob("part_*"):
        if stale.is_file():
            stale.unlink()
    stale_manifest = chunk_dir / "chunk_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()

    chunks: List[Dict[str, Any]] = []
    with source_path.open("rb") as handle:
        part_index = 0
        while True:
            payload = handle.read(chunk_size_bytes)
            if not payload:
                break
            part_name = f"part_{part_index:05d}"
            part_path = chunk_dir / part_name
            part_path.write_bytes(payload)
            part_sha = _file_sha256(part_path)
            remote_path = build_remote_path(
                release_name,
                f"chunks/{relative_original}/{part_name}",
            )
            chunks.append(
                {
                    "part_name": part_name,
                    "size_bytes": int(len(payload)),
                    "sha256": part_sha,
                    "remote_path": remote_path,
                    "local_path": _to_repo_relative(part_path),
                }
            )
            part_index += 1

    chunk_manifest = {
        "original_path": local_path,
        "original_size_bytes": int(original_size_bytes),
        "original_sha256": original_sha256,
        "chunk_size_bytes": int(chunk_size_bytes),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "restore_command_hint": (
            f"PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py "
            f"--release {release_name} --execute"
        ),
    }

    chunk_manifest_path = chunk_dir / "chunk_manifest.json"
    chunk_manifest_path.write_text(json.dumps(chunk_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chunk_manifest_remote_path = build_remote_path(
        release_name,
        f"chunks/{relative_original}/chunk_manifest.json",
    )

    return {
        "chunk_manifest_path": _to_repo_relative(chunk_manifest_path),
        "chunk_manifest_remote_path": chunk_manifest_remote_path,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _load_manifest_items(manifest_path: Path) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        items = payload.get("artifacts", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Manifest payload must be a list or object with artifacts list")

    if not isinstance(items, list):
        raise ValueError("Manifest artifacts entry must be a list")

    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def ensure_local_manifest(local_manifest_path: Path, *, source_manifest_path: Path) -> Path:
    if local_manifest_path.exists():
        return local_manifest_path

    specs = load_manifest(source_manifest_path.as_posix())
    statuses = check_all_artifacts(specs)
    write_local_manifest(statuses, local_manifest_path.as_posix())
    return local_manifest_path


def build_release_bundle(
    *,
    release_name: str,
    manifest_path: Path,
    output_dir: Path,
    allow_missing: bool = False,
    max_upload_mb: float | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    items = _load_manifest_items(manifest_path)
    artifacts: List[Dict[str, Any]] = []
    max_upload_size_bytes = get_max_upload_bytes(max_upload_mb)

    required_count = 0
    available_required_count = 0
    chunked_artifact_count = 0

    for raw in items:
        artifact_id = _clean_text(raw.get("artifact_id"))
        category = _clean_text(raw.get("category"))
        local_path = _clean_text(raw.get("local_path"))
        required = bool(raw.get("required", True))
        required_for_raw = raw.get("required_for", [])
        required_for = [
            _clean_text(item)
            for item in (required_for_raw if isinstance(required_for_raw, list) else [])
            if _clean_text(item)
        ]

        exists = bool(raw.get("exists")) if "exists" in raw else resolve_path(local_path).exists()
        validation_status = _clean_text(raw.get("validation_status"))
        if not validation_status:
            validation_status = "ok" if exists else "missing"

        artifact_size = int(raw.get("size_bytes", 0) or 0)
        if artifact_size <= 0 and exists:
            artifact_size = compute_size_bytes(local_path)

        artifact_sha = _clean_text(raw.get("sha256"))
        if exists and not artifact_sha:
            artifact_sha = compute_sha256(local_path)

        available = bool(exists and artifact_sha and artifact_size > 0 and validation_status == "ok")

        if required:
            required_count += 1
            if available:
                available_required_count += 1

        artifact_record: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "category": category,
            "local_path": local_path,
            "remote_path": build_remote_path(release_name, local_path),
            "size_bytes": artifact_size,
            "sha256": artifact_sha,
            "required": required,
            "required_for": required_for,
            "available": available,
            "validation_status": validation_status,
            "upload_strategy": "direct",
            "direct_upload": True,
        }

        if available and artifact_size > max_upload_size_bytes:
            chunk_details = chunk_artifact_for_release(
                release_name=release_name,
                local_path=local_path,
                original_size_bytes=artifact_size,
                original_sha256=artifact_sha,
                chunk_size_bytes=max_upload_size_bytes,
            )
            artifact_record.update(
                {
                    "upload_strategy": "chunked",
                    "direct_upload": False,
                    "chunk_manifest_path": chunk_details["chunk_manifest_path"],
                    "chunk_manifest_remote_path": chunk_details["chunk_manifest_remote_path"],
                    "chunk_count": int(chunk_details["chunk_count"]),
                    "original_sha256": artifact_sha,
                    "original_size_bytes": int(artifact_size),
                }
            )
            chunked_artifact_count += 1

        artifacts.append(artifact_record)

    missing_required = [item for item in artifacts if item["required"] and not item["available"]]
    if missing_required and not allow_missing:
        ids = ", ".join(sorted(item["artifact_id"] for item in missing_required))
        raise RuntimeError(f"Missing required artifacts for release bundle: {ids}")

    artifact_count = len(artifacts)
    available_count = sum(1 for item in artifacts if item["available"])
    missing_count = artifact_count - available_count

    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()

    release_payload = {
        "release_name": release_name,
        "generated_at_utc": generated,
        "max_upload_size_bytes": int(max_upload_size_bytes),
        "artifact_count": artifact_count,
        "required_artifact_count": required_count,
        "available_artifact_count": available_count,
        "missing_artifact_count": missing_count,
        "chunked_artifact_count": int(chunked_artifact_count),
        "artifacts": artifacts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "release_metadata.json"
    manifest_out_path = output_dir / "artifact_manifest.json"
    checksums_path = output_dir / "checksums.sha256"

    metadata_path.write_text(
        json.dumps(
            {
                "release_name": release_name,
                "generated_at_utc": generated,
                "artifact_count": artifact_count,
                "required_artifact_count": required_count,
                "available_artifact_count": available_count,
                "missing_artifact_count": missing_count,
                "chunked_artifact_count": int(chunked_artifact_count),
                "max_upload_size_bytes": int(max_upload_size_bytes),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_out_path.write_text(json.dumps(release_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_lines: List[str] = []
    for item in sorted(artifacts, key=lambda x: str(x.get("local_path", ""))):
        if item["available"]:
            checksum_lines.append(f"{item['sha256']}  {item['local_path']}")

    checksum_lines.append(f"{_file_sha256(metadata_path)}  {metadata_path.as_posix()}")
    checksum_lines.append(f"{_file_sha256(manifest_out_path)}  {manifest_out_path.as_posix()}")

    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return release_payload


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def default_release_dir(release_name: str) -> Path:
    return get_project_root() / "data" / "releases" / release_name


def default_local_manifest_path() -> Path:
    return get_project_root() / "data" / "artifact_manifest.local.json"


def default_example_manifest_path() -> Path:
    return get_project_root() / "data" / "artifact_manifest.example.json"
