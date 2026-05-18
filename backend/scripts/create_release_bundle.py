#!/usr/bin/env python3
"""Create deterministic release manifest bundle for artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.release_manifest import (  # noqa: E402
    build_release_bundle,
    default_example_manifest_path,
    default_local_manifest_path,
    default_release_dir,
    ensure_local_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create release artifact bundle")
    parser.add_argument("--release", required=True, help="Release name, e.g. v0.1-local-intelligence")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing required artifacts")
    parser.add_argument("--manifest", default="", help="Optional explicit artifact manifest path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release.strip()
    if not release:
        raise SystemExit("--release is required")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
    else:
        local_manifest = default_local_manifest_path()
        source_manifest = default_example_manifest_path()
        manifest_path = ensure_local_manifest(local_manifest, source_manifest_path=source_manifest)

    output_dir = default_release_dir(release)
    bundle = build_release_bundle(
        release_name=release,
        manifest_path=manifest_path,
        output_dir=output_dir,
        allow_missing=bool(args.allow_missing),
    )

    summary = {
        "release_name": bundle.get("release_name"),
        "artifact_count": bundle.get("artifact_count"),
        "required_artifact_count": bundle.get("required_artifact_count"),
        "available_artifact_count": bundle.get("available_artifact_count"),
        "missing_artifact_count": bundle.get("missing_artifact_count"),
        "output_dir": output_dir.as_posix(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
