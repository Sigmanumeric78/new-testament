#!/usr/bin/env python3
"""Download release artifacts from MongoDB Atlas GridFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from artifacts.artifact_manager import is_runtime_artifact_record  # noqa: E402
from artifacts.mongodb_store import ARTIFACT_SOURCE, MongoArtifactStore, clean_text  # noqa: E402


DEFAULT_REPORT_NAME = "artifact_restore_report.json"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def validate_release_name(value: str) -> str:
    release = clean_text(value)
    if not release:
        raise ValueError("release name is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in release):
        raise ValueError("release name contains unsupported characters")
    return release


def restore_target_path(output_root: str | Path, local_path: str) -> Path:
    root = Path(output_root).expanduser()
    normalized = clean_text(local_path).replace("\\", "/")
    if not normalized:
        raise ValueError("artifact entry missing local_path")
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Refusing unsafe restore path: {local_path}")
    return root.joinpath(*parts)


def _failure(entry: Mapping[str, Any], *, target_path: Path, reason: str, observed_sha256: str = "") -> Dict[str, Any]:
    return {
        "artifact_id": clean_text(entry.get("artifact_id")) or clean_text(entry.get("local_path")),
        "local_path": clean_text(entry.get("local_path")),
        "target_path": target_path.as_posix(),
        "required": bool_value(entry.get("required"), default=True),
        "reason": reason,
        "expected_sha256": clean_text(entry.get("sha256")),
        "observed_sha256": observed_sha256,
    }


def restore_release(
    store: MongoArtifactStore,
    *,
    release: str,
    output_root: str | Path,
    required_only: bool,
    force: bool,
    report_name: str = DEFAULT_REPORT_NAME,
) -> Dict[str, Any]:
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    entries = store.list_release_artifacts(release, required_only=required_only)
    if required_only:
        entries = [
            entry
            for entry in entries
            if bool_value(entry.get("required"), default=True) and is_runtime_artifact_record(entry)
        ]

    restored: List[str] = []
    skipped_existing: List[str] = []
    verified: List[str] = []
    failures: List[Dict[str, Any]] = []

    for entry in entries:
        artifact_id = clean_text(entry.get("artifact_id")) or clean_text(entry.get("local_path"))
        expected_sha = clean_text(entry.get("sha256"))
        target_path = restore_target_path(root, clean_text(entry.get("local_path")))
        gridfs_file_id = entry.get("gridfs_file_id")
        if not gridfs_file_id:
            failures.append(_failure(entry, target_path=target_path, reason="missing_gridfs_file_id"))
            continue

        if target_path.exists():
            observed_existing = sha256_file(target_path)
            if expected_sha and observed_existing == expected_sha:
                skipped_existing.append(artifact_id)
                verified.append(artifact_id)
                continue
            if not force:
                failures.append(
                    _failure(
                        entry,
                        target_path=target_path,
                        reason="existing_file_checksum_mismatch_requires_force",
                        observed_sha256=observed_existing,
                    )
                )
                continue

        try:
            store.download_file(gridfs_file_id, target_path.as_posix(), overwrite=True)
            observed_sha = sha256_file(target_path)
        except Exception as exc:
            failures.append(_failure(entry, target_path=target_path, reason=str(exc)))
            continue

        if expected_sha and observed_sha != expected_sha:
            failures.append(
                _failure(
                    entry,
                    target_path=target_path,
                    reason="downloaded_checksum_mismatch",
                    observed_sha256=observed_sha,
                )
            )
            continue

        restored.append(artifact_id)
        verified.append(artifact_id)

    required_failures = [item for item in failures if bool_value(item.get("required"), default=True)]
    report_path = root / report_name
    report: Dict[str, Any] = {
        "backend": "mongodb",
        "source": ARTIFACT_SOURCE,
        "release": release,
        "output_root": root.as_posix(),
        "required_only": bool(required_only),
        "force": bool(force),
        "selected_artifact_count": len(entries),
        "restored_count": len(restored),
        "restored": restored,
        "skipped_existing_count": len(skipped_existing),
        "skipped_existing": skipped_existing,
        "checksum_verified_count": len(verified),
        "checksum_verified": verified,
        "failure_count": len(failures),
        "required_failure_count": len(required_failures),
        "all_required_verified": len(required_failures) == 0,
        "failures": failures,
        "report_path": report_path.as_posix(),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.log_restore(release, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download release artifacts from MongoDB Atlas GridFS")
    parser.add_argument("--release", required=True, help="Release name")
    parser.add_argument("--output-root", required=True, help="Restore root; artifacts are written to output_root/local_path")
    parser.add_argument("--required-only", action="store_true", help="Restore only entries marked required")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files when checksums differ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store: MongoArtifactStore | None = None
    try:
        release = validate_release_name(args.release)
        store = MongoArtifactStore()
        store.ping()
        report = restore_release(
            store,
            release=release,
            output_root=args.output_root,
            required_only=bool(args.required_only),
            force=bool(args.force),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["required_failure_count"] == 0 else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "backend": "mongodb",
                    "error": True,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
