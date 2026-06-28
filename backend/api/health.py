"""Health helpers and endpoint router."""

from __future__ import annotations

import socket
import json
from typing import Any, Dict, List, Mapping
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from fastapi import APIRouter

from artifacts.artifact_restore_manager import get_restore_status
from artifacts.artifact_manager import check_all_artifacts, filter_runtime_specs, load_manifest, summarize_artifacts
from reasoning.grounding_safety_guard import GroundingSafetyGuard
from reasoning.hybrid_orchestrator import orchestrate_query
from reasoning.query_router import route_query
from reasoning.response_synthesizer import ResponseSynthesizer
from reasoning.user_risk_advisor import build_user_risk_advice
from simulation.pbpk import pbpk_master_simulator
from utils.config import get_mongodb_config, get_neo4j_config, get_ollama_config, get_vector_backend, resolve_project_path
from vectorstores.pinecone_store import PineconeVectorStore

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


def _bool_from_text(value: Any, default: bool = True) -> bool:
    text = _clean_text(value).lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _component(ok: bool, detail: str) -> Dict[str, Any]:
    return {"ok": bool(ok), "detail": _clean_text(detail) or ("ok" if ok else "unavailable")}


def _artifact_component(
    ok: bool,
    detail: str,
    *,
    missing_required: List[str],
    missing_required_count: int | None = None,
    artifact_state: str = "",
    restore_status: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    missing_items = sorted(set([_clean_text(item) for item in missing_required if _clean_text(item)]))
    payload: Dict[str, Any] = {
        "ok": bool(ok),
        "detail": _clean_text(detail) or ("ok" if ok else "unavailable"),
        "missing_required_count": int(len(missing_items) if missing_required_count is None else missing_required_count),
        "missing_required": missing_items,
    }
    if _clean_text(artifact_state):
        payload["status"] = _clean_text(artifact_state)
    if restore_status is not None:
        payload["backend"] = _clean_text(restore_status.get("backend")) or "mongodb"
        payload["release"] = _clean_text(restore_status.get("release"))
        payload["started_at"] = _clean_text(restore_status.get("started_at"))
        payload["completed_at"] = _clean_text(restore_status.get("completed_at"))
        payload["restored_count"] = int(restore_status.get("restored_count", 0) or 0)
        payload["restore_status"] = {
            "status": _clean_text(restore_status.get("status")) or "idle",
            "started_at": _clean_text(restore_status.get("started_at")),
            "completed_at": _clean_text(restore_status.get("completed_at")),
            "error_summary": _clean_text(restore_status.get("error_summary")),
            "restored_count": int(restore_status.get("restored_count", 0) or 0),
            "missing_required_count": int(restore_status.get("missing_required_count", 0) or 0),
        }
    return payload


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


def _mongodb_probe() -> Dict[str, Any]:
    try:
        config = get_mongodb_config(require=True)
        try:
            import certifi  # type: ignore
            from pymongo import MongoClient  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency branch
            return _component(False, f"mongodb dependency unavailable: {exc}")

        kwargs: Dict[str, Any] = {"serverSelectionTimeoutMS": 2500}
        try:
            kwargs["tlsCAFile"] = certifi.where()
        except Exception:
            pass
        client = MongoClient(_clean_text(config.get("uri")), **kwargs)
        try:
            client.admin.command("ping")
        finally:
            client.close()
        database = _clean_text(config.get("database"))
        bucket = _clean_text(config.get("gridfs_bucket"))
        return _component(True, f"ok database={database} bucket={bucket}")
    except Exception as exc:
        return _component(False, str(exc))


def _pinecone_probe() -> Dict[str, Any]:
    store: PineconeVectorStore | None = None
    try:
        store = PineconeVectorStore()
        payload = store.ping()
        total = int(payload.get("total_vector_count", 0) or 0)
        namespace = _clean_text(payload.get("namespace"))
        index = _clean_text(payload.get("index"))
        dimension = int(payload.get("dimension", 0) or 0)
        detail = f"ok index={index} namespace={namespace} dimension={dimension} total_vector_count={total}"
        return _component(True, detail)
    except Exception as exc:
        return _component(False, str(exc))
    finally:
        if store is not None:
            store.close()


def _ollama_is_disabled(config: Mapping[str, Any]) -> bool:
    provider = _clean_text(config.get("provider")).lower() or "ollama"
    enabled = _bool_from_text(config.get("enabled"), default=True)
    host = _clean_text(config.get("host"))
    normalized = host.lower().rstrip("/")
    if provider == "disabled":
        return True
    if not enabled:
        return True
    return normalized in {"", "disabled", "http://disabled", "https://disabled", "off", "none"}


def _build_ollama_url(host: str, endpoint: str) -> str:
    base = _clean_text(host).rstrip("/")
    if not base:
        return ""
    normalized_endpoint = _clean_text(endpoint).strip().lstrip("/")
    if not normalized_endpoint:
        return base
    base_norm = _clean_text(base).lower().rstrip("/")
    if base_norm.endswith("/api"):
        tail = normalized_endpoint.split("/", 1)[-1]
        return f"{base}/{tail}"
    return urljoin(base + "/", normalized_endpoint)


def _extract_ollama_model_names(payload: Any) -> List[str]:
    if not isinstance(payload, Mapping):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        models = payload.get("data")
    if not isinstance(models, list):
        return []
    names: List[str] = []
    for item in models:
        if not isinstance(item, Mapping):
            continue
        name = _clean_text(item.get("name")) or _clean_text(item.get("model")) or _clean_text(item.get("id"))
        if name:
            names.append(name)
    deduped = sorted(set(names))
    return deduped


def _ollama_probe() -> Dict[str, Any]:
    try:
        config = get_ollama_config()
        if _ollama_is_disabled(config):
            provider = _clean_text(config.get("provider")).lower() or "ollama"
            if provider == "disabled":
                return _component(True, "standby (LLM_PROVIDER=disabled)")
            if not _bool_from_text(config.get("enabled"), default=True):
                return _component(True, "standby (OLLAMA_ENABLED=false)")
            return _component(True, "standby (OLLAMA_HOST=disabled)")
        host = _clean_text(config.get("host")) or "http://localhost:11434"
        model = _clean_text(config.get("model"))
        allow_unlisted_model = _bool_from_text(config.get("allow_unlisted_model"), default=False)
        auto_select_model = _bool_from_text(config.get("auto_select_model"), default=False)
        api_key = _clean_text(config.get("api_key"))
        urls = [_build_ollama_url(host, "api/tags"), _build_ollama_url(host, "v1/models")]
        urls = [item for item in urls if item]
        if not urls:
            return _component(False, "OLLAMA_HOST is required when LLM_PROVIDER=ollama")
        payload: Any = {}
        last_error = ""
        for url in urls:
            req = Request(url=url, method="GET")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            try:
                with urlopen(req, timeout=4) as response:  # noqa: S310 - URL comes from runtime config
                    status = int(getattr(response, "status", 200))
                    body = response.read().decode("utf-8", errors="replace")
                if status != 200:
                    last_error = f"ollama http {status}"
                    continue
                payload = json.loads(body or "{}")
                break
            except Exception as exc:
                last_error = str(exc)
        if not payload:
            return _component(False, last_error or "ollama model list unavailable")
        model_names = _extract_ollama_model_names(payload)
        if model:
            if model in model_names:
                return _component(True, f"ok model={model}")
            if allow_unlisted_model:
                return _component(True, f"ok model={model}")
            return _component(False, f"configured model not available: {model}")

        if auto_select_model and model_names:
            return _component(True, f"ok model={model_names[0]}")

        return _component(False, "configured model not available: <empty>")
    except Exception as exc:
        return _component(False, str(exc))


def _artifact_probe() -> Dict[str, Any]:
    restore_status = get_restore_status()
    restore_state = _clean_text(restore_status.get("status")).lower()
    if restore_state == "restoring":
        missing_required = list(restore_status.get("missing_required", []) or [])
        return _artifact_component(
            False,
            "restoring",
            missing_required=missing_required,
            missing_required_count=int(restore_status.get("missing_required_count", len(missing_required)) or 0),
            artifact_state="restoring",
            restore_status=restore_status,
        )
    if restore_state == "failed":
        missing_required = list(restore_status.get("missing_required", []) or [])
        detail = _clean_text(restore_status.get("error_summary")) or "artifact restore failed"
        return _artifact_component(
            False,
            f"restore failed: {detail}",
            missing_required=missing_required,
            missing_required_count=int(restore_status.get("missing_required_count", len(missing_required)) or 0),
            artifact_state="failed",
            restore_status=restore_status,
        )

    if not ARTIFACT_MANIFEST_PATH.exists():
        return _artifact_component(
            False,
            f"artifact manifest not found: {ARTIFACT_MANIFEST_PATH.as_posix()}",
            missing_required=[],
            artifact_state="failed",
            restore_status=restore_status,
        )

    try:
        manifest = load_manifest(ARTIFACT_MANIFEST_PATH.as_posix())
        runtime_specs = filter_runtime_specs(manifest)
        if not runtime_specs:
            return _artifact_component(
                False,
                "no runtime artifacts selected from manifest",
                missing_required=[],
                artifact_state="failed",
                restore_status=restore_status,
            )
        statuses = check_all_artifacts(runtime_specs)
        summary = summarize_artifacts(statuses)
    except Exception as exc:
        return _artifact_component(
            False,
            f"artifact status check failed: {exc}",
            missing_required=[],
            artifact_state="failed",
            restore_status=restore_status,
        )

    missing = list(summary.get("missing_required", []) or [])
    if missing:
        return _artifact_component(
            False,
            f"{len(missing)} required artifacts missing.",
            missing_required=missing,
            artifact_state="degraded",
            restore_status=restore_status,
        )
    return _artifact_component(True, "ok", missing_required=[], artifact_state="ok", restore_status=restore_status)


def build_health_payload() -> Dict[str, Any]:
    _ = get_vector_backend()
    components = {
        "api": _component(True, "ok"),
        "neo4j": _neo4j_probe(),
        "mongodb": _mongodb_probe(),
        "ollama": _ollama_probe(),
        "artifact_status": _artifact_probe(),
        "pinecone": _pinecone_probe(),
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
    external_keys = ["neo4j", "mongodb", "artifact_status", "pinecone"]
    if not _clean_text(components.get("ollama", {}).get("detail", "")).lower().startswith("standby"):
        external_keys.append("ollama")

    if any(not bool(components[key]["ok"]) for key in core_keys):
        status = "error"
    elif any(not bool(components[key]["ok"]) for key in external_keys):
        status = "degraded"
    else:
        status = "healthy"

    return {"status": status, "components": components}


@router.get("/health")
def health_check() -> Dict[str, Any]:
    return build_health_payload()
