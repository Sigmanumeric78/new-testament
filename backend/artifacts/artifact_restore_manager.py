"""Background artifact restore scheduling and status tracking."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from artifacts.artifact_manager import check_all_artifacts, filter_runtime_specs, load_manifest, summarize_artifacts
from utils.config import get_project_root, resolve_project_path

DEFAULT_ARTIFACT_RELEASE = "v0.6-chemical-explorer"
DEFAULT_ARTIFACT_BACKEND = "supabase"
DEFAULT_MANIFEST_PATH = "data/artifact_manifest.example.json"
RESTORE_LOG_PREFIX = "[artifact-restore]"
SENSITIVE_ENV_MARKERS = ("URI", "KEY", "TOKEN", "PASSWORD", "SECRET")


@dataclass
class ArtifactRestoreStatus:
    status: str = "idle"
    started_at: str = ""
    completed_at: str = ""
    error_summary: str = ""
    restored_count: int = 0
    missing_required_count: int = 0
    missing_required: List[str] = field(default_factory=list)
    backend: str = ""
    release: str = ""


_restore_lock = threading.Lock()
_restore_status = ArtifactRestoreStatus()
_restore_thread: threading.Thread | None = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _truthy(value: Any) -> bool:
    return _clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_dict_locked() -> Dict[str, Any]:
    payload = asdict(_restore_status)
    payload["running"] = _restore_status.status == "restoring"
    return payload


def _set_status(**updates: Any) -> Dict[str, Any]:
    with _restore_lock:
        for key, value in updates.items():
            if hasattr(_restore_status, key):
                setattr(_restore_status, key, value)
        return _status_dict_locked()


def get_restore_status() -> Dict[str, Any]:
    with _restore_lock:
        return _status_dict_locked()


def artifact_restore_enabled() -> bool:
    if not _truthy(os.getenv("RESTORE_ARTIFACTS_ON_STARTUP", "false")):
        return False
    mode = _clean_text(os.getenv("ARTIFACT_RESTORE_MODE", "background")).lower()
    return mode not in {"0", "false", "no", "off", "none", "disabled"}


def schedule_background_restore_if_enabled() -> Dict[str, Any]:
    if not artifact_restore_enabled():
        return get_restore_status()
    return schedule_background_restore()


def schedule_background_restore() -> Dict[str, Any]:
    global _restore_thread
    with _restore_lock:
        if _restore_thread is not None and _restore_thread.is_alive():
            return _status_dict_locked()

        backend = _artifact_backend()
        release = _artifact_release()
        _restore_status.status = "restoring"
        _restore_status.started_at = _now_iso()
        _restore_status.completed_at = ""
        _restore_status.error_summary = ""
        _restore_status.restored_count = 0
        _restore_status.missing_required_count = 0
        _restore_status.missing_required = []
        _restore_status.backend = backend
        _restore_status.release = release
        _restore_thread = threading.Thread(target=_run_restore, name="artifact-restore", daemon=True)
        payload = _status_dict_locked()

    _log("background restore scheduled")
    _restore_thread.start()
    return payload


def _artifact_backend() -> str:
    return _clean_text(os.getenv("ARTIFACT_STORE_BACKEND", DEFAULT_ARTIFACT_BACKEND)).lower() or DEFAULT_ARTIFACT_BACKEND


def _artifact_release() -> str:
    return _clean_text(os.getenv("ARTIFACT_RELEASE", DEFAULT_ARTIFACT_RELEASE)) or DEFAULT_ARTIFACT_RELEASE


def _restore_workspace_dir(release: str) -> Path:
    raw = _clean_text(os.getenv("RESTORE_WORKSPACE_DIR"))
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp") / "artifact_restore" / release


def _mongodb_output_root() -> Path:
    raw = _clean_text(os.getenv("MONGODB_RESTORE_OUTPUT_ROOT"))
    if raw:
        return Path(raw).expanduser()

    data_root_raw = _clean_text(os.getenv("DATA_ROOT"))
    data_root = Path(data_root_raw).expanduser() if data_root_raw else resolve_project_path("data")
    if data_root.name == "data":
        return data_root.parent
    return get_project_root()


def _artifact_availability_summary(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    manifest = resolve_project_path(manifest_path)
    if not manifest.exists():
        return {
            "missing_required_count": 0,
            "missing_required": [],
            "error": f"artifact manifest not found: {manifest.as_posix()}",
        }

    specs = filter_runtime_specs(load_manifest(manifest.as_posix()))
    statuses = check_all_artifacts(specs)
    summary = summarize_artifacts(statuses)
    return {
        "missing_required_count": int(summary.get("missing_required_count", 0) or 0),
        "missing_required": list(summary.get("missing_required", []) or []),
        "error": "",
    }


def _run_restore() -> None:
    backend = _artifact_backend()
    release = _artifact_release()
    _set_status(status="restoring", started_at=_now_iso(), completed_at="", backend=backend, release=release)
    _log("restore started")
    try:
        initial = _artifact_availability_summary()
        _set_status(
            missing_required_count=int(initial.get("missing_required_count", 0) or 0),
            missing_required=list(initial.get("missing_required", []) or []),
        )

        report = _restore_artifacts(backend=backend, release=release)
        restored_count = _restored_count(report)
        availability = _artifact_availability_summary()
        missing_count = int(availability.get("missing_required_count", 0) or 0)
        missing_required = list(availability.get("missing_required", []) or [])
        availability_error = _clean_text(availability.get("error"))
        _set_status(
            restored_count=restored_count,
            missing_required_count=missing_count,
            missing_required=missing_required,
        )

        if availability_error:
            raise RuntimeError(availability_error)
        if missing_count:
            raise RuntimeError(f"{missing_count} required artifacts missing after restore")

        _set_status(
            status="ok",
            completed_at=_now_iso(),
            error_summary="",
            restored_count=restored_count,
            missing_required_count=0,
            missing_required=[],
        )
        _log("restore completed")
    except Exception as exc:
        summary = _sanitize_error(str(exc))
        current = get_restore_status()
        _set_status(
            status="failed",
            completed_at=_now_iso(),
            error_summary=summary,
            restored_count=int(current.get("restored_count", 0) or 0),
        )
        _log(f"restore failed: {summary}")


def _restore_artifacts(*, backend: str, release: str) -> Dict[str, Any]:
    if backend == "mongodb":
        return _restore_mongodb(release)
    if backend == "supabase":
        return _restore_supabase(release)
    raise ValueError(f"unsupported ARTIFACT_STORE_BACKEND={backend}")


def _restore_mongodb(release: str) -> Dict[str, Any]:
    from artifacts.mongodb_store import MongoArtifactStore
    from scripts import artifact_download_mongodb

    store: MongoArtifactStore | None = None
    try:
        validated_release = artifact_download_mongodb.validate_release_name(release)
        store = MongoArtifactStore()
        store.ping()
        report = artifact_download_mongodb.restore_release(
            store,
            release=validated_release,
            output_root=_mongodb_output_root(),
            required_only=True,
            force=True,
        )
        if int(report.get("required_failure_count", 0) or 0):
            raise RuntimeError(f"{int(report.get('required_failure_count', 0) or 0)} required MongoDB artifacts failed")
        return report
    finally:
        if store is not None:
            store.close()


def _restore_supabase(release: str) -> Dict[str, Any]:
    from scripts import artifact_download_supabase
    from scripts.artifact_verify_release import verify_release_manifest

    workspace_dir = _restore_workspace_dir(release)
    report = artifact_download_supabase.restore_release(
        release,
        execute=True,
        overwrite=True,
        runtime_only=True,
        workspace_dir=workspace_dir,
    )
    if bool(report.get("error")):
        message = _clean_text(report.get("message")) or "Supabase artifact restore failed"
        raise RuntimeError(message)

    manifest_path = workspace_dir / "artifact_manifest.json"
    verification = verify_release_manifest(
        release,
        manifest_path,
        runtime_only=True,
        workspace_dir=workspace_dir,
    )
    if not bool(verification.get("all_required_valid")):
        invalid_count = int(verification.get("invalid_count", 0) or 0)
        raise RuntimeError(f"{invalid_count} required Supabase artifacts failed verification")

    payload = dict(report)
    payload["verification"] = verification
    return payload


def _restored_count(report: Mapping[str, Any]) -> int:
    direct_count = int(report.get("restored_count", report.get("downloaded_count", 0)) or 0)
    chunked_count = int(report.get("restored_chunked_count", 0) or 0)
    return direct_count + chunked_count


def _sanitize_error(message: str) -> str:
    sanitized = _clean_text(message) or "artifact restore failed"
    for key, value in os.environ.items():
        if not any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS):
            continue
        secret = _clean_text(value)
        if len(secret) >= 6:
            sanitized = sanitized.replace(secret, "<redacted>")

    sanitized = re.sub(r"mongodb(?:\+srv)?://[^\s]+", "mongodb://<redacted>", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"Bearer\s+[^\s]+", "Bearer <redacted>", sanitized, flags=re.IGNORECASE)
    return sanitized[:500]


def _log(message: str) -> None:
    print(f"{RESTORE_LOG_PREFIX} {message}", flush=True)


def _reset_restore_state_for_tests() -> None:
    global _restore_thread, _restore_status
    with _restore_lock:
        _restore_thread = None
        _restore_status = ArtifactRestoreStatus()
