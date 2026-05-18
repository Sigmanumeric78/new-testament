"""Local filesystem helpers for artifact resolution and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def _is_repo_root(path: Path) -> bool:
    return (path / "backend").is_dir() and (path / "README.md").exists()


def _is_backend_root(path: Path) -> bool:
    return (path / "api").is_dir() and (path / "reasoning").is_dir() and (path / "simulation").is_dir()


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if _is_repo_root(parent):
            return parent
    for parent in current.parents:
        if _is_backend_root(parent):
            return parent
    return Path(__file__).resolve().parents[1]


def resolve_path(local_path: str) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute():
        return candidate
    return get_project_root() / candidate


def exists(local_path: str) -> bool:
    return resolve_path(local_path).exists()


def size_bytes(local_path: str) -> int:
    path = resolve_path(local_path)
    if not path.exists():
        return 0
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def sha256(local_path: str) -> str:
    path = resolve_path(local_path)
    if not path.exists() or not path.is_file():
        return ""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def modified_time(local_path: str) -> Optional[str]:
    path = resolve_path(local_path)
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return mtime.isoformat()
    except Exception:
        return None


def validate_type(local_path: str, expected_type: str) -> Tuple[bool, str]:
    path = resolve_path(local_path)
    if not path.exists() or not path.is_file():
        return False, "artifact path does not exist"

    expected = expected_type.strip().lower()
    try:
        if expected == "csv":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                first_line = handle.readline()
            if not first_line.strip():
                return False, "csv first line is empty"
            return True, ""

        if expected == "json":
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            return True, ""

        if expected == "jsonl":
            valid_lines = 0
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        json.loads(text)
                    except Exception:
                        return False, f"invalid jsonl line at {idx}"
                    valid_lines += 1
            if valid_lines < 1:
                return False, "jsonl contains no valid JSON lines"
            return True, ""

        if expected == "parquet":
            if path.suffix.lower() != ".parquet":
                return False, "parquet artifact must have .parquet extension"
            return True, ""

        if expected == "md":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.read(256)
            return True, ""

        return False, f"unsupported expected type: {expected_type}"
    except Exception as exc:
        return False, str(exc)


def validate_min_size(local_path: str, min_size_bytes: int) -> Tuple[bool, str]:
    size = size_bytes(local_path)
    if size < int(min_size_bytes):
        return False, f"artifact size {size} < minimum {int(min_size_bytes)}"
    return True, ""
