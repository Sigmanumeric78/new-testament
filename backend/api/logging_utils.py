"""Lightweight JSONL logging for API requests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.config import project_root

LOG_PATH = project_root() / "data" / "interim" / "api" / "api_request_log.jsonl"


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


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def log_request(
    *,
    endpoint: str,
    query: str,
    response_style: Optional[str],
    risk_level: Optional[str],
    safe_for_display: Optional[bool],
    latency_ms: float,
    error: Optional[str],
    stage: str,
) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": _clean_text(endpoint),
            "query_hash": query_hash(query),
            "query_length": len(query),
            "response_style": _clean_text(response_style) or None,
            "risk_level": _clean_text(risk_level) or None,
            "safe_for_display": safe_for_display,
            "latency_ms": round(float(latency_ms), 3),
            "error": _clean_text(error) or None,
            "stage": _clean_text(stage),
        }
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:
        # Logging must never break request handling (e.g., read-only mounts in Docker).
        return


def structured_error(message: str, stage: str) -> Mapping[str, Any]:
    return {
        "error": True,
        "message": _clean_text(message) or "unknown error",
        "stage": _clean_text(stage) or "unknown_stage",
    }
