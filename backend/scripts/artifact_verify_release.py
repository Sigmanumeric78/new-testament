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


def verify_release_manifest(release: str, manifest_path: Path) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = list(payload.get("artifacts", []) or [])

    missing: List[str] = []
    checksum_mismatches: List[str] = []
    valid_count = 0
    invalid_count = 0

    required_invalid = False

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue

        artifact_id = _clean_text(artifact.get("artifact_id")) or _clean_text(artifact.get("local_path"))
        local_path_raw = _clean_text(artifact.get("local_path"))
        expected_sha = _clean_text(artifact.get("sha256"))
        required = bool(artifact.get("required", True))
        available = bool(artifact.get("available", False))

        if not available:
            if required:
                missing.append(artifact_id)
                required_invalid = True
                invalid_count += 1
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
        "missing": sorted(set(missing)),
        "checksum_mismatches": sorted(set(checksum_mismatches)),
        "valid_count": int(valid_count),
        "invalid_count": int(invalid_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local release artifacts")
    parser.add_argument("--release", required=True, help="Release name")
    parser.add_argument("--manifest", default="", help="Optional explicit release manifest path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release.strip()

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
    else:
        manifest_path = default_release_dir(release) / "artifact_manifest.json"

    if not manifest_path.exists():
        payload = {
            "release": release,
            "all_required_valid": False,
            "missing": [f"manifest:{manifest_path.as_posix()}"],
            "checksum_mismatches": [],
            "valid_count": 0,
            "invalid_count": 1,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    payload = verify_release_manifest(release, manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("all_required_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
