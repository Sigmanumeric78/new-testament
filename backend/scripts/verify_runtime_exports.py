#!/usr/bin/env python3
"""Verify runtime export file layout and generate checksums/manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.export_neo4j_data import DEFAULT_RELEASE, expected_neo4j_files, runtime_export_root
from scripts.export_weaviate_data import expected_weaviate_files


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def expected_runtime_files() -> Dict[str, List[str]]:
    return {
        "neo4j": [f"neo4j/{name}" for name in expected_neo4j_files()],
        "weaviate": [f"weaviate/{name}" for name in expected_weaviate_files()],
    }


def _collect_file_entries(root: Path, relative_paths: Sequence[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for rel in sorted(set(relative_paths)):
        path = root / rel
        exists = path.exists() and path.is_file()
        size = int(path.stat().st_size) if exists else 0
        checksum = _sha256(path) if exists else ""
        entries.append(
            {
                "path": rel,
                "exists": bool(exists),
                "size_bytes": size,
                "sha256": checksum,
            }
        )
    return entries


def _read_manifest_status(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_json"
    if isinstance(payload, Mapping):
        status = payload.get("status")
        if isinstance(status, str):
            text = status.strip().lower()
            if text:
                return text
    return "missing_status"


def verify_runtime_exports(release: str = DEFAULT_RELEASE, output_root: Optional[str] = None) -> Dict[str, Any]:
    export_root = runtime_export_root(release, output_root)
    export_root.mkdir(parents=True, exist_ok=True)

    expected = expected_runtime_files()
    neo_paths = expected["neo4j"]
    weav_paths = expected["weaviate"]
    all_expected_paths = neo_paths + weav_paths

    generated_at = datetime.now(timezone.utc).isoformat()
    file_entries = _collect_file_entries(export_root, all_expected_paths)
    missing = [entry["path"] for entry in file_entries if not entry["exists"]]

    neo4j_files_exist = all((export_root / rel).exists() for rel in neo_paths)
    weaviate_files_exist = all((export_root / rel).exists() for rel in weav_paths)

    neo_manifest_status = _read_manifest_status(export_root / "neo4j" / "graph_export_manifest.json")
    weav_manifest_status = _read_manifest_status(export_root / "weaviate" / "backup_manifest.json")

    neo4j_export_ok = bool(neo4j_files_exist and neo_manifest_status == "ok")
    weaviate_export_ok = bool(weaviate_files_exist and weav_manifest_status == "ok")

    runtime_manifest = {
        "release": release,
        "generated_at_utc": generated_at,
        "export_root": export_root.as_posix(),
        "expected_files": {"neo4j": neo_paths, "weaviate": weav_paths},
        "files": file_entries,
        "missing_files": missing,
        "neo4j_manifest_status": neo_manifest_status,
        "weaviate_manifest_status": weav_manifest_status,
    }
    manifest_path = export_root / "artifact_manifest.runtime.json"
    manifest_path.write_text(json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = sorted([entry["path"] for entry in file_entries if entry["exists"]] + ["artifact_manifest.runtime.json"])
    checksum_lines: List[str] = []
    for rel in checksum_paths:
        checksum_lines.append(f"{_sha256(export_root / rel)}  {rel}")
    checksums_path = export_root / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    checksums_written = checksums_path.exists()
    result = {
        "release": release,
        "neo4j_export_ok": bool(neo4j_export_ok),
        "weaviate_export_ok": bool(weaviate_export_ok),
        "checksums_written": bool(checksums_written),
        "neo4j_manifest_status": neo_manifest_status,
        "weaviate_manifest_status": weav_manifest_status,
        "safe_for_supabase_upload": bool(neo4j_export_ok and weaviate_export_ok and checksums_written),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify runtime exports and generate checksums/manifest")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Release name")
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional root directory for runtime exports. Defaults to data/exports/runtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = verify_runtime_exports(release=args.release, output_root=args.output_root or None)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
