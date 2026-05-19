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
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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


def load_release_manifest_local(release: str, release_dir: Path | None = None) -> Dict[str, Any]:
    base = release_dir or default_release_dir(release)
    manifest_path = base / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Release manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def compute_download_plan(release_manifest: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = list(release_manifest.get("artifacts", []) or [])
    candidates: List[Dict[str, str]] = []
    checksum_mismatches: List[str] = []
    missing_local: List[str] = []
    chunked_restore: List[Dict[str, Any]] = []
    chunked_total_chunks = 0
    chunked_total_bytes = 0

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        local_path_raw = _clean_text(artifact.get("local_path"))
        expected_sha = _clean_text(artifact.get("sha256"))
        available = bool(artifact.get("available", False))
        required = bool(artifact.get("required", True))
        remote_path = _clean_text(artifact.get("remote_path"))
        artifact_id = _clean_text(artifact.get("artifact_id")) or local_path_raw
        upload_strategy = _clean_text(artifact.get("upload_strategy")) or "direct"

        if not available:
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

            chunk_manifest_path = _clean_text(artifact.get("chunk_manifest_path"))
            chunk_manifest_remote_path = _clean_text(artifact.get("chunk_manifest_remote_path"))
            chunk_count = _to_int(artifact.get("chunk_count"), 0)
            chunked_total_chunks += max(chunk_count, 0)
            chunked_total_bytes += _to_int(artifact.get("original_size_bytes"), 0)
            chunked_restore.append(
                {
                    "artifact_id": artifact_id,
                    "required": bool(required),
                    "original_local_path": local_path_raw,
                    "original_sha256": expected_original_sha,
                    "chunk_manifest_local_path": chunk_manifest_path,
                    "chunk_manifest_remote_path": chunk_manifest_remote_path,
                    "chunk_count": chunk_count,
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

    return {
        "candidates": candidates,
        "missing_local": sorted(set(missing_local)),
        "checksum_mismatches": sorted(set(checksum_mismatches)),
        "chunked_restore": chunked_restore,
        "chunked_artifact_count": len(chunked_restore),
        "chunked_total_chunks": int(chunked_total_chunks),
        "chunked_total_bytes": int(chunked_total_bytes),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    dry_run = bool(args.dry_run or not execute)

    release = args.release.strip()
    release_dir = default_release_dir(release)
    release_dir.mkdir(parents=True, exist_ok=True)

    if execute:
        store = SupabaseArtifactStore(require_credentials=True)
        remote_manifest = f"releases/{release}/artifact_manifest.json"
        store.download_json(
            remote_path=remote_manifest,
            local_path=(release_dir / "artifact_manifest.json").as_posix(),
            overwrite=True,
        )

    release_manifest = load_release_manifest_local(release, release_dir)
    plan = compute_download_plan(release_manifest)

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
                part_local_path_raw = _clean_text(chunk.get("local_path"))
                if not part_remote_path:
                    raise ValueError(f"Missing chunk remote_path for {artifact_id}:{part_name}")
                if not part_local_path_raw:
                    part_local_path_raw = (manifest_local_path.parent / part_name).as_posix()
                    chunk["local_path"] = part_local_path_raw

                store.download_file(
                    remote_path=part_remote_path,
                    local_path=part_local_path_raw,
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
        "dry_run": dry_run,
        "execute": execute,
        "candidate_count": len(plan["candidates"]),
        "chunked_artifact_count": plan["chunked_artifact_count"],
        "chunked_total_chunks": plan["chunked_total_chunks"],
        "chunked_total_bytes": plan["chunked_total_bytes"],
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
