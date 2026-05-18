#!/usr/bin/env python3
"""Build local artifact status manifest from the committed example manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import check_all_artifacts, load_manifest, summarize_artifacts, write_local_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local artifact status manifest")
    parser.add_argument(
        "--manifest",
        default=(REPO_ROOT / "data" / "artifact_manifest.example.json").as_posix(),
        help="Input artifact manifest JSON",
    )
    parser.add_argument(
        "--output",
        default=(REPO_ROOT / "data" / "artifact_manifest.local.json").as_posix(),
        help="Output local artifact status JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    specs = load_manifest(manifest_path.as_posix())
    statuses = check_all_artifacts(specs)
    payload = write_local_manifest(statuses, output_path.as_posix())
    print(json.dumps(payload.get("summary", summarize_artifacts(statuses)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
