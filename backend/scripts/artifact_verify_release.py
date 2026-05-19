#!/usr/bin/env python3
"""Verify local artifacts against a release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import is_runtime_artifact_record  # noqa: E402
from artifacts.local_store import resolve_path  # noqa: E402
from artifacts.release_manifest import default_release_dir  # noqa: E402


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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_chunk_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_release_manifest(
    release: str,
    manifest_path: Path,
    *,
    runtime_only: bool = False,
    workspace_dir: Path | None = None,
) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts_all = [item for item in list(payload.get("artifacts", []) or []) if isinstance(item, dict)]
    artifacts = [item for item in artifacts_all if (not runtime_only or is_runtime_artifact_record(item))]
    skipped_non_runtime = len(artifacts_all) - len(artifacts)

    missing: List[str] = []
    checksum_mismatches: List[str] = []
    valid_count = 0
    invalid_count = 0
    restorable_chunked: List[str] = []

    required_invalid = False

    for artifact in artifacts:
        artifact_id = _clean_text(artifact.get("artifact_id")) or _clean_text(artifact.get("local_path"))
        local_path_raw = _clean_text(artifact.get("local_path"))
        expected_sha = _clean_text(artifact.get("sha256"))
        required = bool(artifact.get("required", True))
        available = bool(artifact.get("available", False))
        upload_strategy = _clean_text(artifact.get("upload_strategy")) or "direct"

        if not available:
            if required:
                missing.append(artifact_id)
                required_invalid = True
                invalid_count += 1
            continue

        if upload_strategy == "chunked":
            expected_original_sha = _clean_text(artifact.get("original_sha256")) or expected_sha
            original_local_path = resolve_path(local_path_raw)
            if original_local_path.exists():
                observed_original_sha = _sha256_file(original_local_path)
                if expected_original_sha and observed_original_sha != expected_original_sha:
                    checksum_mismatches.append(artifact_id)
                    invalid_count += 1
                    if required:
                        required_invalid = True
                else:
                    valid_count += 1
                continue

            chunk_manifest_path = None
            if workspace_dir is not None:
                candidate = workspace_dir / "chunks" / artifact_id / "chunk_manifest.json"
                if candidate.exists():
                    chunk_manifest_path = candidate

            if chunk_manifest_path is None:
                chunk_manifest_path_raw = _clean_text(artifact.get("chunk_manifest_path"))
                if chunk_manifest_path_raw:
                    chunk_manifest_path = resolve_path(chunk_manifest_path_raw)

            if chunk_manifest_path is None:
                missing.append(f"{artifact_id}:chunk_manifest")
                invalid_count += 1
                if required:
                    required_invalid = True
                continue
            if not chunk_manifest_path.exists():
                missing.append(f"{artifact_id}:chunk_manifest")
                invalid_count += 1
                if required:
                    required_invalid = True
                continue

            try:
                chunk_manifest = _load_chunk_manifest(chunk_manifest_path)
                chunk_entries = list(chunk_manifest.get("chunks", []) or [])
                if not chunk_entries:
                    raise ValueError("chunk manifest has no chunks")
                chunk_invalid = False
                for idx, chunk in enumerate(chunk_entries):
                    if not isinstance(chunk, dict):
                        missing.append(f"{artifact_id}:chunk_{idx}:invalid")
                        chunk_invalid = True
                        continue
                    part_name = _clean_text(chunk.get("part_name")) or f"part_{idx:05d}"
                    part_local_path_raw = _clean_text(chunk.get("local_path"))
                    if not part_local_path_raw:
                        part_local_path_raw = (chunk_manifest_path.parent / part_name).as_posix()
                    part_local_path = resolve_path(part_local_path_raw)
                    if not part_local_path.exists():
                        missing.append(f"{artifact_id}:{part_name}")
                        chunk_invalid = True
                        continue
                    expected_part_sha = _clean_text(chunk.get("sha256"))
                    observed_part_sha = _sha256_file(part_local_path)
                    if expected_part_sha and observed_part_sha != expected_part_sha:
                        checksum_mismatches.append(f"{artifact_id}:{part_name}")
                        chunk_invalid = True

                if chunk_invalid:
                    invalid_count += 1
                    if required:
                        required_invalid = True
                else:
                    valid_count += 1
                    restorable_chunked.append(artifact_id)
            except Exception:
                missing.append(f"{artifact_id}:chunk_manifest_unreadable")
                invalid_count += 1
                if required:
                    required_invalid = True
            continue

        local_path = resolve_path(local_path_raw)
        if not local_path.exists():
            missing.append(artifact_id)
            invalid_count += 1
            if required:
                required_invalid = True
            continue

        observed = _sha256_file(local_path)
        if expected_sha and observed != expected_sha:
            checksum_mismatches.append(artifact_id)
            invalid_count += 1
            if required:
                required_invalid = True
        else:
            valid_count += 1

    return {
        "release": release,
        "all_required_valid": (not required_invalid),
        "runtime_only": bool(runtime_only),
        "selected_artifact_count": len(artifacts),
        "skipped_non_runtime_count": int(max(skipped_non_runtime, 0)),
        "missing": sorted(set(missing)),
        "checksum_mismatches": sorted(set(checksum_mismatches)),
        "valid_count": int(valid_count),
        "invalid_count": int(invalid_count),
        "restorable_chunked": sorted(set(restorable_chunked)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local release artifacts")
    parser.add_argument("--release", required=True, help="Release name")
    parser.add_argument("--manifest", default="", help="Optional explicit release manifest path")
    parser.add_argument(
        "--workspace-dir",
        default="",
        help="Optional workspace directory containing downloaded manifest/chunk files.",
    )
    parser.add_argument("--runtime-only", action="store_true", help="Verify runtime-only artifact subset.")
    parser.add_argument("--all-artifacts", action="store_true", help="Verify full artifact set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release.strip()
    runtime_only = bool(args.runtime_only)
    if bool(args.all_artifacts):
        runtime_only = False
    workspace_dir: Path | None = None
    if _clean_text(args.workspace_dir):
        candidate = Path(args.workspace_dir)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        workspace_dir = candidate

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
    elif workspace_dir is not None:
        manifest_path = workspace_dir / "artifact_manifest.json"
    else:
        manifest_path = default_release_dir(release) / "artifact_manifest.json"

    if not manifest_path.exists():
        payload = {
            "release": release,
            "all_required_valid": False,
            "runtime_only": runtime_only,
            "missing": [f"manifest:{manifest_path.as_posix()}"],
            "checksum_mismatches": [],
            "valid_count": 0,
            "invalid_count": 1,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    payload = verify_release_manifest(
        release,
        manifest_path,
        runtime_only=runtime_only,
        workspace_dir=workspace_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("all_required_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
