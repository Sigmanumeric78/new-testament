"""Artifact manifest loading, checking, and summarization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

from artifacts.local_store import (
    exists,
    modified_time,
    resolve_path,
    sha256,
    size_bytes,
    validate_min_size,
    validate_type,
)
from artifacts.manifest import ArtifactSpec, ArtifactStatus


def load_manifest(path: str) -> List[ArtifactSpec]:
    manifest_path = resolve_path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, Mapping):
        items = raw.get("artifacts", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Manifest must be a list or an object with an 'artifacts' list.")

    if not isinstance(items, list):
        raise ValueError("Manifest 'artifacts' must be a list.")

    specs: List[ArtifactSpec] = []
    seen_ids = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Each manifest artifact must be an object.")
        spec = ArtifactSpec.from_dict(item)
        if spec.artifact_id in seen_ids:
            raise ValueError(f"Duplicate artifact_id in manifest: {spec.artifact_id}")
        seen_ids.add(spec.artifact_id)
        specs.append(spec)
    return specs


def check_artifact(spec: ArtifactSpec) -> ArtifactStatus:
    found = exists(spec.local_path)
    file_size = size_bytes(spec.local_path) if found else 0
    file_sha = sha256(spec.local_path) if found else ""
    file_mtime = modified_time(spec.local_path) if found else None

    if not found:
        return ArtifactStatus(
            artifact_id=spec.artifact_id,
            exists=False,
            size_bytes=0,
            sha256="",
            modified_time=None,
            validation_status="missing",
            failure_reason="artifact path does not exist",
            category=spec.category,
            required=spec.required,
            local_path=spec.local_path,
        )

    type_ok, type_reason = validate_type(spec.local_path, spec.expected_type)
    if not type_ok:
        return ArtifactStatus(
            artifact_id=spec.artifact_id,
            exists=True,
            size_bytes=file_size,
            sha256=file_sha,
            modified_time=file_mtime,
            validation_status="invalid_type",
            failure_reason=type_reason,
            category=spec.category,
            required=spec.required,
            local_path=spec.local_path,
        )

    size_ok, size_reason = validate_min_size(spec.local_path, spec.min_size_bytes)
    if not size_ok:
        return ArtifactStatus(
            artifact_id=spec.artifact_id,
            exists=True,
            size_bytes=file_size,
            sha256=file_sha,
            modified_time=file_mtime,
            validation_status="too_small",
            failure_reason=size_reason,
            category=spec.category,
            required=spec.required,
            local_path=spec.local_path,
        )

    return ArtifactStatus(
        artifact_id=spec.artifact_id,
        exists=True,
        size_bytes=file_size,
        sha256=file_sha,
        modified_time=file_mtime,
        validation_status="ok",
        failure_reason="",
        category=spec.category,
        required=spec.required,
        local_path=spec.local_path,
    )


def check_all_artifacts(manifest: Sequence[ArtifactSpec]) -> List[ArtifactStatus]:
    return [check_artifact(spec) for spec in manifest]


def get_missing_required(statuses: Sequence[ArtifactStatus]) -> List[str]:
    missing = [s.artifact_id for s in statuses if s.required and s.validation_status != "ok"]
    return sorted(set(missing))


def summarize_artifacts(statuses: Sequence[ArtifactStatus]) -> Dict[str, Any]:
    required_statuses = [s for s in statuses if s.required]
    missing_required = get_missing_required(statuses)

    categories: Dict[str, Dict[str, int]] = {}
    for status in statuses:
        category = status.category or "uncategorized"
        if category not in categories:
            categories[category] = {"total": 0, "available": 0, "missing": 0}
        categories[category]["total"] += 1
        if status.validation_status == "ok":
            categories[category]["available"] += 1
        else:
            categories[category]["missing"] += 1

    required_count = len(required_statuses)
    available_required_count = sum(1 for s in required_statuses if s.validation_status == "ok")

    return {
        "all_required_available": len(missing_required) == 0,
        "required_count": required_count,
        "available_required_count": available_required_count,
        "missing_required_count": len(missing_required),
        "missing_required": missing_required,
        "categories": categories,
    }


def write_local_manifest(statuses: Sequence[ArtifactStatus], output_path: str) -> Dict[str, Any]:
    summary = summarize_artifacts(statuses)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "artifacts": [status.to_dict() for status in statuses],
    }

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
