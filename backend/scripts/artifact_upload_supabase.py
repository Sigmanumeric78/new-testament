#!/usr/bin/env python3
"""Upload release artifacts to Supabase Storage.

Dry-run is default. Use --execute to perform uploads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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


def _is_forbidden_upload_path(local_path: str) -> bool:
    normalized = local_path.replace("\\", "/")
    forbidden_prefixes = (
        ".env",
        "frontend/node_modules/",
        "frontend/dist/",
        "docs/research_papers/",
    )
    if normalized == ".env" or normalized.endswith("/.env") or normalized.endswith("/.env.local"):
        return True
    return any(normalized.startswith(prefix) for prefix in forbidden_prefixes)


def load_release_manifest(release: str, release_dir: Path | None = None) -> Dict[str, Any]:
    base = release_dir or default_release_dir(release)
    manifest_path = base / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Release manifest not found at {manifest_path}. Run create_release_bundle.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_upload_candidates(release_manifest: Dict[str, Any], release_dir: Path) -> Dict[str, Any]:
    artifacts = list(release_manifest.get("artifacts", []) or [])

    required_missing: List[str] = []
    upload_items: List[Dict[str, str]] = []
    skipped_optional_missing: List[str] = []
    skipped_forbidden: List[str] = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        local_path = _clean_text(artifact.get("local_path"))
        remote_path = _clean_text(artifact.get("remote_path"))
        artifact_id = _clean_text(artifact.get("artifact_id"))
        required = bool(artifact.get("required", True))
        available = bool(artifact.get("available", False))

        if _is_forbidden_upload_path(local_path):
            skipped_forbidden.append(artifact_id or local_path)
            continue

        if not available:
            if required:
                required_missing.append(artifact_id or local_path)
            else:
                skipped_optional_missing.append(artifact_id or local_path)
            continue

        upload_items.append({"local_path": local_path, "remote_path": remote_path, "artifact_id": artifact_id})

    manifest_files = [
        {
            "local_path": (release_dir / "release_metadata.json").as_posix(),
            "remote_path": f"releases/{release_manifest.get('release_name', '')}/release_metadata.json",
            "artifact_id": "release_metadata",
        },
        {
            "local_path": (release_dir / "artifact_manifest.json").as_posix(),
            "remote_path": f"releases/{release_manifest.get('release_name', '')}/artifact_manifest.json",
            "artifact_id": "release_artifact_manifest",
        },
        {
            "local_path": (release_dir / "checksums.sha256").as_posix(),
            "remote_path": f"releases/{release_manifest.get('release_name', '')}/checksums.sha256",
            "artifact_id": "release_checksums",
        },
    ]

    for entry in manifest_files:
        if not Path(entry["local_path"]).exists():
            required_missing.append(entry["artifact_id"])
        else:
            upload_items.append(entry)

    return {
        "required_missing": sorted(set(required_missing)),
        "upload_items": upload_items,
        "skipped_optional_missing": sorted(set(skipped_optional_missing)),
        "skipped_forbidden": sorted(set(skipped_forbidden)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload release artifacts to Supabase")
    parser.add_argument("--release", required=True, help="Release name")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode (default)")
    parser.add_argument("--execute", action="store_true", help="Perform actual uploads")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite in bucket")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    dry_run = bool(args.dry_run or not execute)

    release = args.release.strip()
    release_dir = default_release_dir(release)
    release_manifest = load_release_manifest(release, release_dir)
    plan = collect_upload_candidates(release_manifest, release_dir)

    required_missing = list(plan["required_missing"])
    if required_missing:
        payload = {
            "release": release,
            "dry_run": dry_run,
            "execute": execute,
            "error": True,
            "message": "Missing required artifacts; upload aborted.",
            "required_missing": required_missing,
            "skipped_optional_missing": plan["skipped_optional_missing"],
            "skipped_forbidden": plan["skipped_forbidden"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    uploaded: List[str] = []
    if execute:
        store = SupabaseArtifactStore(require_credentials=True)
        for entry in plan["upload_items"]:
            store.upload_file(entry["local_path"], entry["remote_path"], overwrite=bool(args.overwrite))
            uploaded.append(entry["artifact_id"])

    payload = {
        "release": release,
        "dry_run": dry_run,
        "execute": execute,
        "candidate_count": len(plan["upload_items"]),
        "uploaded_count": len(uploaded),
        "uploaded": uploaded,
        "skipped_optional_missing": plan["skipped_optional_missing"],
        "skipped_forbidden": plan["skipped_forbidden"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
