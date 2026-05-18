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

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        local_path_raw = _clean_text(artifact.get("local_path"))
        expected_sha = _clean_text(artifact.get("sha256"))
        available = bool(artifact.get("available", False))
        required = bool(artifact.get("required", True))
        remote_path = _clean_text(artifact.get("remote_path"))
        artifact_id = _clean_text(artifact.get("artifact_id")) or local_path_raw

        if not available:
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
    }


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
    if execute:
        store = SupabaseArtifactStore(require_credentials=True)
        for item in plan["candidates"]:
            store.download_file(
                remote_path=item["remote_path"],
                local_path=resolve_path(item["local_path"]).as_posix(),
                overwrite=bool(args.overwrite),
            )
            downloaded.append(item["artifact_id"])

    payload = {
        "release": release,
        "dry_run": dry_run,
        "execute": execute,
        "candidate_count": len(plan["candidates"]),
        "downloaded_count": len(downloaded),
        "downloaded": downloaded,
        "missing_local": plan["missing_local"],
        "checksum_mismatches": plan["checksum_mismatches"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
