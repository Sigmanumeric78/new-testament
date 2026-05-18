#!/usr/bin/env python3
"""Report local artifact availability status from the example manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import check_all_artifacts, load_manifest, summarize_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Artifact availability status")
    parser.add_argument(
        "--manifest",
        default=(REPO_ROOT / "data" / "artifact_manifest.example.json").as_posix(),
        help="Artifact manifest JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest = load_manifest(manifest_path.as_posix())
        statuses = check_all_artifacts(manifest)
        summary = summarize_artifacts(statuses)

        total_count = len(statuses)
        available_count = sum(1 for item in statuses if item.validation_status == "ok")
        missing_count = total_count - available_count

        payload = {
            "all_required_available": bool(summary["all_required_available"]),
            "missing_required": list(summary["missing_required"]),
            "available_count": int(available_count),
            "missing_count": int(missing_count),
            "categories": summary["categories"],
        }

        print(json.dumps(payload, sort_keys=True))
        if payload["all_required_available"]:
            return 0
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": True,
                    "message": str(exc),
                    "stage": "artifact_status",
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
