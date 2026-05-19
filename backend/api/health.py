"""Health helpers and endpoint router."""

from __future__ import annotations

import socket
import json
from typing import Any, Dict, List
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from fastapi import APIRouter

from artifacts.artifact_manager import check_all_artifacts, load_manifest, summarize_artifacts
from reasoning.grounding_safety_guard import GroundingSafetyGuard
from reasoning.hybrid_orchestrator import orchestrate_query
from reasoning.query_router import route_query
from reasoning.response_synthesizer import ResponseSynthesizer
from reasoning.user_risk_advisor import build_user_risk_advice
from simulation.pbpk import pbpk_master_simulator
from utils.config import get_neo4j_config, get_ollama_config, get_weaviate_config, resolve_project_path

router = APIRouter()
ARTIFACT_MANIFEST_PATH = resolve_project_path("data/artifact_manifest.example.json")


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


def _component(ok: bool, detail: str) -> Dict[str, Any]:
    return {"ok": bool(ok), "detail": _clean_text(detail) or ("ok" if ok else "unavailable")}


def _artifact_component(ok: bool, detail: str, *, missing_required: List[str]) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "detail": _clean_text(detail) or ("ok" if ok else "unavailable"),
        "missing_required_count": int(len(missing_required)),
        "missing_required": sorted(set([_clean_text(item) for item in missing_required if _clean_text(item)])),
    }


def _socket_probe(host: str, port: int, timeout_seconds: float = 1.5) -> Dict[str, Any]:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds):
            return _component(True, "ok")
    except Exception as exc:
        return _component(False, str(exc))


def _neo4j_probe() -> Dict[str, Any]:
    try:
        config = get_neo4j_config()
        parsed = urlparse(_clean_text(config.get("uri")))
        host = parsed.hostname or "localhost"
        port = int(parsed.port or 7687)
        return _socket_probe(host, port)
    except Exception as exc:
        return _component(False, str(exc))


def _weaviate_probe() -> Dict[str, Any]:
    try:
        config = get_weaviate_config()
        parsed = urlparse(_clean_text(config.get("url")))
        host = parsed.hostname or "localhost"
        port = int(parsed.port or 8080)
        return _socket_probe(host, port)
    except Exception as exc:
        return _component(False, str(exc))


def _ollama_probe() -> Dict[str, Any]:
    try:
        config = get_ollama_config()
        host = _clean_text(config.get("host")) or "http://localhost:11434"
        model = _clean_text(config.get("model"))
        url = urljoin(host.rstrip("/") + "/", "api/tags")
        req = Request(url=url, method="GET")
        with urlopen(req, timeout=4) as response:  # noqa: S310 - local host URL from config
            status = int(getattr(response, "status", 200))
            body = response.read().decode("utf-8", errors="replace")
        if status != 200:
            return _component(False, f"ollama http {status}")
        payload = json.loads(body or "{}")
        models = payload.get("models", []) if isinstance(payload, dict) else []
        detail = "ok"
        if model and isinstance(models, list):
            model_names = {
                _clean_text(item.get("name"))
                for item in models
                if isinstance(item, dict) and _clean_text(item.get("name"))
            }
            if model not in model_names:
                detail = f"ok (model {model} not listed)"
        return _component(True, detail)
    except Exception as exc:
        return _component(False, str(exc))


def _artifact_probe() -> Dict[str, Any]:
    if not ARTIFACT_MANIFEST_PATH.exists():
        return _artifact_component(
            False,
            f"artifact manifest not found: {ARTIFACT_MANIFEST_PATH.as_posix()}",
            missing_required=[],
        )

    try:
        manifest = load_manifest(ARTIFACT_MANIFEST_PATH.as_posix())
        statuses = check_all_artifacts(manifest)
        summary = summarize_artifacts(statuses)
    except Exception as exc:
        return _artifact_component(False, f"artifact status check failed: {exc}", missing_required=[])

    missing = list(summary.get("missing_required", []) or [])
    if missing:
        return _artifact_component(
            False,
            f"{len(missing)} required artifacts missing.",
            missing_required=missing,
        )
    return _artifact_component(True, "ok", missing_required=[])


def build_health_payload() -> Dict[str, Any]:
    components = {
        "api": _component(True, "ok"),
        "neo4j": _neo4j_probe(),
        "weaviate": _weaviate_probe(),
        "ollama": _ollama_probe(),
        "artifact_status": _artifact_probe(),
    }

    try:
        _ = pbpk_master_simulator.run_simulation
        components["pbpk"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["pbpk"] = _component(False, str(exc))

    try:
        _ = route_query
        components["router"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["router"] = _component(False, str(exc))

    try:
        _ = orchestrate_query
        components["orchestrator"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["orchestrator"] = _component(False, str(exc))

    try:
        _ = ResponseSynthesizer
        components["synthesizer"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["synthesizer"] = _component(False, str(exc))

    try:
        _ = GroundingSafetyGuard
        components["grounding_guard"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["grounding_guard"] = _component(False, str(exc))

    try:
        _ = build_user_risk_advice
        components["user_risk_advisor"] = _component(True, "ok")
    except Exception as exc:  # pragma: no cover
        components["user_risk_advisor"] = _component(False, str(exc))

    core_keys = ("api", "pbpk", "router", "orchestrator", "synthesizer", "grounding_guard", "user_risk_advisor")
    external_keys = ("neo4j", "weaviate", "ollama", "artifact_status")

    if any(not bool(components[key]["ok"]) for key in core_keys):
        status = "error"
    elif any(not bool(components[key]["ok"]) for key in external_keys):
        status = "degraded"
    else:
        status = "ok"

    return {"status": status, "components": components}


@router.get("/health")
def health_check() -> Dict[str, Any]:
    return build_health_payload()
