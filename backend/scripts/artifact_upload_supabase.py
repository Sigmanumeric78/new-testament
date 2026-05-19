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
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.local_store import resolve_path  # noqa: E402
from artifacts.release_manifest import (  # noqa: E402
    build_remote_path,
    chunk_artifact_for_release,
    default_release_dir,
    get_max_upload_bytes,
)
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
        "data/interim/",
        "__pycache__/",
    )
    forbidden_suffixes = (".pyc",)
    if (
        normalized == ".env"
        or normalized.endswith("/.env")
        or normalized.endswith("/.env.local")
        or normalized.endswith(forbidden_suffixes)
    ):
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
    release_name = _clean_text(release_manifest.get("release_name"))
    max_upload_size_bytes = get_max_upload_bytes()

    required_missing: List[str] = []
    upload_items: List[Dict[str, str]] = []
    skipped_optional_missing: List[str] = []
    skipped_forbidden: List[str] = []
    skipped_oversized_direct_upload: List[str] = []
    chunked_originals: List[str] = []
    chunked_upload_count = 0

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        local_path = _clean_text(artifact.get("local_path"))
        remote_path = _clean_text(artifact.get("remote_path"))
        artifact_id = _clean_text(artifact.get("artifact_id"))
        required = bool(artifact.get("required", True))
        available = bool(artifact.get("available", False))
        upload_strategy = _clean_text(artifact.get("upload_strategy")) or "direct"
        artifact_size = int(artifact.get("size_bytes", 0) or 0)

        if _is_forbidden_upload_path(local_path):
            skipped_forbidden.append(artifact_id or local_path)
            continue

        if not available:
            if required:
                required_missing.append(artifact_id or local_path)
            else:
                skipped_optional_missing.append(artifact_id or local_path)
            continue

        if upload_strategy != "chunked":
            local_file = resolve_path(local_path)
            if local_file.exists():
                if artifact_size <= 0:
                    artifact_size = int(local_file.stat().st_size)
                if artifact_size > max_upload_size_bytes:
                    chunk_details = chunk_artifact_for_release(
                        release_name=release_name,
                        local_path=local_path,
                        original_size_bytes=artifact_size,
                        original_sha256=_clean_text(artifact.get("sha256")),
                        chunk_size_bytes=max_upload_size_bytes,
                    )
                    artifact["upload_strategy"] = "chunked"
                    artifact["direct_upload"] = False
                    artifact["chunk_manifest_path"] = chunk_details["chunk_manifest_path"]
                    artifact["chunk_manifest_remote_path"] = chunk_details["chunk_manifest_remote_path"]
                    artifact["chunk_count"] = int(chunk_details["chunk_count"])
                    artifact["original_sha256"] = _clean_text(artifact.get("sha256"))
                    artifact["original_size_bytes"] = int(artifact_size)
                    upload_strategy = "chunked"

        if upload_strategy == "chunked":
            skipped_oversized_direct_upload.append(artifact_id or local_path)
            chunked_originals.append(artifact_id or local_path)

            chunk_manifest_path = _clean_text(artifact.get("chunk_manifest_path"))
            chunk_manifest_remote = _clean_text(artifact.get("chunk_manifest_remote_path"))
            if not chunk_manifest_remote:
                chunk_manifest_remote = build_remote_path(release_name, f"chunks/{local_path}/chunk_manifest.json")

            if not chunk_manifest_path or not Path(chunk_manifest_path).exists():
                required_missing.append(f"{artifact_id or local_path}:chunk_manifest_missing")
                continue

            upload_items.append(
                {
                    "local_path": chunk_manifest_path,
                    "remote_path": chunk_manifest_remote,
                    "artifact_id": f"{artifact_id or local_path}:chunk_manifest",
                }
            )
            chunked_upload_count += 1

            chunk_payload = json.loads(Path(chunk_manifest_path).read_text(encoding="utf-8"))
            chunk_entries = list(chunk_payload.get("chunks", []) or [])
            if not chunk_entries:
                required_missing.append(f"{artifact_id or local_path}:chunk_parts_missing")
                continue
            for idx, chunk in enumerate(chunk_entries):
                if not isinstance(chunk, dict):
                    required_missing.append(f"{artifact_id or local_path}:invalid_chunk_entry_{idx}")
                    continue
                part_name = _clean_text(chunk.get("part_name")) or f"part_{idx:05d}"
                part_local_path = _clean_text(chunk.get("local_path"))
                if not part_local_path:
                    part_local_path = (Path(chunk_manifest_path).parent / part_name).as_posix()
                part_remote_path = _clean_text(chunk.get("remote_path"))
                if not part_remote_path:
                    part_remote_path = build_remote_path(
                        release_name,
                        f"chunks/{local_path}/{part_name}",
                    )
                if not Path(part_local_path).exists():
                    required_missing.append(f"{artifact_id or local_path}:{part_name}:missing")
                    continue
                upload_items.append(
                    {
                        "local_path": part_local_path,
                        "remote_path": part_remote_path,
                        "artifact_id": f"{artifact_id or local_path}:{part_name}",
                    }
                )
                chunked_upload_count += 1
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
        "skipped_oversized_direct_upload": sorted(set(skipped_oversized_direct_upload)),
        "chunked_upload_count": int(chunked_upload_count),
        "chunked_originals": sorted(set(chunked_originals)),
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
            "skipped_oversized_direct_upload": plan["skipped_oversized_direct_upload"],
            "chunked_upload_count": plan["chunked_upload_count"],
            "chunked_originals": plan["chunked_originals"],
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
        "skipped_oversized_direct_upload": plan["skipped_oversized_direct_upload"],
        "chunked_upload_count": plan["chunked_upload_count"],
        "chunked_originals": plan["chunked_originals"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
