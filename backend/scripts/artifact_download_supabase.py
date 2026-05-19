#!/usr/bin/env python3
"""Download release artifacts from Supabase Storage.

Dry-run is default. Use --execute to perform downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import is_runtime_artifact_record  # noqa: E402
from artifacts.local_store import resolve_path  # noqa: E402
from artifacts.release_manifest import default_release_dir  # noqa: E402
from artifacts.supabase_store import SupabaseArtifactStore  # noqa: E402


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


def _validate_release_name(value: str) -> str:
    release = _clean_text(value)
    if not release:
        raise ValueError("release name is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in release):
        raise ValueError("release name contains unsupported characters")
    return release


def _workspace_dir_from_args(release: str, workspace_dir_raw: str) -> Path:
    if _clean_text(workspace_dir_raw):
        base = Path(workspace_dir_raw)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return Path("/tmp") / "artifact_restore" / release


def _artifact_workspace_manifest_path(workspace_dir: Path) -> Path:
    return workspace_dir / "artifact_manifest.json"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_chunk_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_release_manifest_local(release: str, release_dir: Path | None = None) -> Tuple[Dict[str, Any], Path]:
    base = release_dir or default_release_dir(release)
    manifest_path = base / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Release manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def compute_download_plan(
    release_manifest: Dict[str, Any],
    *,
    runtime_only: bool,
    workspace_dir: Path,
) -> Dict[str, Any]:
    artifacts_all = list(release_manifest.get("artifacts", []) or [])
    artifacts = [item for item in artifacts_all if isinstance(item, dict)]
    if runtime_only:
        artifacts = [item for item in artifacts if is_runtime_artifact_record(item)]
    candidates: List[Dict[str, str]] = []
    checksum_mismatches: List[str] = []
    missing_local: List[str] = []
    unavailable_required: List[str] = []
    chunked_restore: List[Dict[str, Any]] = []
    chunked_total_chunks = 0
    chunked_total_bytes = 0

    for artifact in artifacts:
        local_path_raw = _clean_text(artifact.get("local_path"))
        expected_sha = _clean_text(artifact.get("sha256"))
        available = bool(artifact.get("available", False))
        required = bool(artifact.get("required", True))
        remote_path = _clean_text(artifact.get("remote_path"))
        artifact_id = _clean_text(artifact.get("artifact_id")) or local_path_raw
        upload_strategy = _clean_text(artifact.get("upload_strategy")) or "direct"

        if not available:
            if required:
                unavailable_required.append(artifact_id)
            continue

        if upload_strategy == "chunked":
            original_local_path = resolve_path(local_path_raw)
            expected_original_sha = _clean_text(artifact.get("original_sha256")) or expected_sha
            if original_local_path.exists():
                observed = _sha256_file(original_local_path)
                if expected_original_sha and observed == expected_original_sha:
                    continue
                checksum_mismatches.append(artifact_id)
            else:
                missing_local.append(artifact_id)

            chunk_manifest_remote_path = _clean_text(artifact.get("chunk_manifest_remote_path"))
            chunk_count = _to_int(artifact.get("chunk_count"), 0)
            chunked_total_chunks += max(chunk_count, 0)
            chunked_total_bytes += _to_int(artifact.get("original_size_bytes"), 0)
            chunk_manifest_path = (workspace_dir / "chunks" / artifact_id / "chunk_manifest.json").as_posix()
            chunked_restore.append(
                {
                    "artifact_id": artifact_id,
                    "required": bool(required),
                    "original_local_path": local_path_raw,
                    "original_sha256": expected_original_sha,
                    "chunk_manifest_local_path": chunk_manifest_path,
                    "chunk_manifest_remote_path": chunk_manifest_remote_path,
                    "chunk_count": chunk_count,
                    "remote_path": remote_path,
                }
            )
            continue

        local_path = resolve_path(local_path_raw)
        if not local_path.exists():
            missing_local.append(artifact_id)
            candidates.append(
                {
                    "artifact_id": artifact_id,
                    "remote_path": remote_path,
                    "local_path": local_path_raw,
                    "required": str(required).lower(),
                }
            )
            continue

        observed_sha = _sha256_file(local_path)
        if expected_sha and observed_sha != expected_sha:
            checksum_mismatches.append(artifact_id)
            candidates.append(
                {
                    "artifact_id": artifact_id,
                    "remote_path": remote_path,
                    "local_path": local_path_raw,
                    "required": str(required).lower(),
                }
            )

    skipped_non_runtime_ids = []
    if runtime_only:
        selected_ids = {_clean_text(item.get("artifact_id")) for item in artifacts}
        skipped_non_runtime_ids = sorted(
            {
                _clean_text(item.get("artifact_id")) or _clean_text(item.get("local_path"))
                for item in artifacts_all
                if isinstance(item, dict)
                and (_clean_text(item.get("artifact_id")) or _clean_text(item.get("local_path"))) not in selected_ids
            }
        )

    return {
        "candidates": candidates,
        "missing_local": sorted(set(missing_local)),
        "checksum_mismatches": sorted(set(checksum_mismatches)),
        "chunked_restore": chunked_restore,
        "chunked_artifact_count": len(chunked_restore),
        "chunked_total_chunks": int(chunked_total_chunks),
        "chunked_total_bytes": int(chunked_total_bytes),
        "unavailable_required": sorted(set(unavailable_required)),
        "runtime_only": bool(runtime_only),
        "selected_artifact_count": len(artifacts),
        "skipped_non_runtime_count": len(skipped_non_runtime_ids),
        "skipped_non_runtime": skipped_non_runtime_ids,
    }


def reassemble_chunked_artifact(chunk_manifest: Dict[str, Any], *, overwrite: bool) -> Path:
    original_path_raw = _clean_text(chunk_manifest.get("original_path"))
    if not original_path_raw:
        raise ValueError("chunk_manifest missing original_path")
    original_path = resolve_path(original_path_raw)
    if original_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact without --overwrite: {original_path.as_posix()}")

    chunks = list(chunk_manifest.get("chunks", []) or [])
    if not chunks:
        raise ValueError("chunk_manifest contains no chunks")

    original_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = original_path.with_suffix(original_path.suffix + ".reassemble_tmp")
    with tmp_path.open("wb") as out_handle:
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Invalid chunk entry at index {idx}")
            part_name = _clean_text(chunk.get("part_name")) or f"part_{idx:05d}"
            local_part_path_raw = _clean_text(chunk.get("local_path"))
            if not local_part_path_raw:
                raise ValueError(f"chunk entry missing local_path for {part_name}")
            local_part_path = resolve_path(local_part_path_raw)
            if not local_part_path.exists():
                raise FileNotFoundError(f"Missing chunk part: {local_part_path.as_posix()}")
            observed_part_sha = _sha256_file(local_part_path)
            expected_part_sha = _clean_text(chunk.get("sha256"))
            if expected_part_sha and observed_part_sha != expected_part_sha:
                raise ValueError(
                    f"Chunk checksum mismatch for {part_name}: expected {expected_part_sha}, observed {observed_part_sha}"
                )
            out_handle.write(local_part_path.read_bytes())

    observed_original_sha = _sha256_file(tmp_path)
    expected_original_sha = _clean_text(chunk_manifest.get("original_sha256"))
    if expected_original_sha and observed_original_sha != expected_original_sha:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Reassembled file checksum mismatch: expected {expected_original_sha}, observed {observed_original_sha}"
        )

    if original_path.exists():
        original_path.unlink()
    tmp_path.rename(original_path)
    return original_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download release artifacts from Supabase")
    parser.add_argument("--release", required=True, help="Release name")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode (default)")
    parser.add_argument("--execute", action="store_true", help="Perform downloads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files")
    parser.add_argument(
        "--workspace-dir",
        default="",
        help="Workspace directory for remote manifest/chunk metadata (default: /tmp/artifact_restore/<release>)",
    )
    parser.add_argument(
        "--all-artifacts",
        action="store_true",
        help="Download all artifacts from release manifest (default is runtime-only subset).",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Force runtime-only subset restore (default behavior).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    dry_run = bool(args.dry_run or not execute)
    runtime_only = bool(args.runtime_only)
    if bool(args.all_artifacts):
        runtime_only = False

    release = _validate_release_name(args.release)
    workspace_dir = _workspace_dir_from_args(release, args.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_manifest = _artifact_workspace_manifest_path(workspace_dir)
    release_dir = default_release_dir(release)

    if execute:
        store = SupabaseArtifactStore(require_credentials=True)
        remote_manifest = f"releases/{release}/artifact_manifest.json"
        store.download_file(
            remote_path=remote_manifest,
            local_path=workspace_manifest.as_posix(),
            overwrite=True,
        )

        for filename in ("release_metadata.json", "checksums.sha256"):
            remote = f"releases/{release}/{filename}"
            local = workspace_dir / filename
            try:
                store.download_file(remote_path=remote, local_path=local.as_posix(), overwrite=True)
            except Exception:
                # Optional for restore; keep going if not present.
                pass

        release_manifest = json.loads(workspace_manifest.read_text(encoding="utf-8"))
        manifest_source = workspace_manifest.as_posix()
    else:
        if workspace_manifest.exists():
            release_manifest = json.loads(workspace_manifest.read_text(encoding="utf-8"))
            manifest_source = workspace_manifest.as_posix()
        else:
            release_manifest, local_manifest_path = load_release_manifest_local(release, release_dir)
            manifest_source = local_manifest_path.as_posix()

    plan = compute_download_plan(release_manifest, runtime_only=runtime_only, workspace_dir=workspace_dir)
    if execute and plan["unavailable_required"]:
        payload = {
            "release": release,
            "dry_run": dry_run,
            "execute": execute,
            "error": True,
            "message": "Required artifacts are unavailable in release manifest.",
            "manifest_source": manifest_source,
            "unavailable_required": plan["unavailable_required"],
            "runtime_only": runtime_only,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    downloaded: List[str] = []
    restored_chunked: List[str] = []
    skipped_chunked_existing: List[str] = []
    if execute:
        store = SupabaseArtifactStore(require_credentials=True)
        for item in plan["candidates"]:
            store.download_file(
                remote_path=item["remote_path"],
                local_path=resolve_path(item["local_path"]).as_posix(),
                overwrite=bool(args.overwrite),
            )
            downloaded.append(item["artifact_id"])

        for item in plan["chunked_restore"]:
            manifest_local_path_raw = _clean_text(item.get("chunk_manifest_local_path"))
            manifest_remote_path = _clean_text(item.get("chunk_manifest_remote_path"))
            artifact_id = _clean_text(item.get("artifact_id")) or _clean_text(item.get("original_local_path"))
            if not manifest_local_path_raw:
                raise ValueError(f"Chunked artifact missing chunk_manifest_local_path: {artifact_id}")
            if not manifest_remote_path:
                raise ValueError(f"Chunked artifact missing chunk_manifest_remote_path: {artifact_id}")

            manifest_local_path = resolve_path(manifest_local_path_raw)
            store.download_file(
                remote_path=manifest_remote_path,
                local_path=manifest_local_path.as_posix(),
                overwrite=True,
            )
            downloaded.append(f"{artifact_id}:chunk_manifest")

            chunk_manifest = _load_chunk_manifest(manifest_local_path)
            chunk_entries = list(chunk_manifest.get("chunks", []) or [])
            for idx, chunk in enumerate(chunk_entries):
                if not isinstance(chunk, dict):
                    raise ValueError(f"Invalid chunk entry at index {idx} for {artifact_id}")
                part_remote_path = _clean_text(chunk.get("remote_path"))
                part_name = _clean_text(chunk.get("part_name")) or f"part_{idx:05d}"
                part_local_path = manifest_local_path.parent / part_name
                if not part_remote_path:
                    remote_base = _clean_text(item.get("chunk_manifest_remote_path")) or _clean_text(item.get("remote_path"))
                    if remote_base.endswith("/chunk_manifest.json"):
                        part_remote_path = remote_base.rsplit("/", 1)[0] + f"/{part_name}"
                if not part_remote_path:
                    raise ValueError(f"Missing chunk remote_path for {artifact_id}:{part_name}")
                chunk["local_path"] = part_local_path.as_posix()

                store.download_file(
                    remote_path=part_remote_path,
                    local_path=part_local_path.as_posix(),
                    overwrite=bool(args.overwrite),
                )
                downloaded.append(f"{artifact_id}:{part_name}")

            manifest_local_path.write_text(json.dumps(chunk_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                reassemble_chunked_artifact(chunk_manifest, overwrite=bool(args.overwrite))
                restored_chunked.append(artifact_id)
            except FileExistsError:
                skipped_chunked_existing.append(artifact_id)

    payload = {
        "release": release,
        "manifest_source": manifest_source,
        "workspace_dir": workspace_dir.as_posix(),
        "runtime_only": runtime_only,
        "selected_artifact_count": plan["selected_artifact_count"],
        "skipped_non_runtime_count": plan["skipped_non_runtime_count"],
        "skipped_non_runtime": plan["skipped_non_runtime"],
        "dry_run": dry_run,
        "execute": execute,
        "candidate_count": len(plan["candidates"]),
        "chunked_artifact_count": plan["chunked_artifact_count"],
        "chunked_total_chunks": plan["chunked_total_chunks"],
        "chunked_total_bytes": plan["chunked_total_bytes"],
        "unavailable_required": plan["unavailable_required"],
        "downloaded_count": len(downloaded),
        "downloaded": downloaded,
        "restored_chunked_count": len(restored_chunked),
        "restored_chunked": restored_chunked,
        "skipped_chunked_existing": skipped_chunked_existing,
        "missing_local": plan["missing_local"],
        "checksum_mismatches": plan["checksum_mismatches"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
