#!/usr/bin/env python3
"""Read-only Weaviate runtime export foundation.

Exports vector DB metadata snapshots to:
data/exports/runtime/<release>/weaviate/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    import weaviate  # type: ignore
except Exception:  # pragma: no cover - environment without weaviate client
    weaviate = None

from utils.config import get_weaviate_config, project_root

DEFAULT_RELEASE = "v0.6-chemical-explorer"


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


def runtime_export_root(release: str, output_root: Optional[str] = None) -> Path:
    if output_root:
        base = Path(output_root)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / release
    return project_root() / "data" / "exports" / "runtime" / release


def weaviate_export_dir(release: str, output_root: Optional[str] = None) -> Path:
    return runtime_export_root(release, output_root) / "weaviate"


def expected_weaviate_files() -> List[str]:
    return [
        "collection_counts.json",
        "schema_snapshot.json",
        "backup_manifest.json",
        "weaviate_backup_instructions.md",
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_primitive(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _as_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_primitive(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _as_primitive(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _as_primitive(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _as_primitive(vars(value))
        except Exception:
            pass
    return _clean_text(value)


def _parse_weaviate_url(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid WEAVIATE_URL: '{url}'. Expected http(s)://host[:port]")
    secure = parsed.scheme.lower() == "https"
    return {
        "host": parsed.hostname,
        "port": int(parsed.port or (443 if secure else 80)),
        "secure": secure,
    }


def _connect_weaviate(config: Mapping[str, str]) -> Any:
    if weaviate is None:
        raise RuntimeError("weaviate-client is not installed")

    parsed = _parse_weaviate_url(config["url"])
    grpc_host = _clean_text(config.get("grpc_host", "")) or "localhost"
    grpc_port = int(_clean_text(config.get("grpc_port", "")) or "50051")
    api_key = _clean_text(config.get("api_key", ""))

    auth_credentials = None
    if api_key:
        try:
            from weaviate.classes.init import Auth  # type: ignore

            auth_credentials = Auth.api_key(api_key)
        except Exception:
            from weaviate.auth import AuthApiKey  # type: ignore

            auth_credentials = AuthApiKey(api_key)

    try:
        return weaviate.connect_to_custom(
            http_host=parsed["host"],
            http_port=parsed["port"],
            http_secure=parsed["secure"],
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=parsed["secure"],
            auth_credentials=auth_credentials,
        )
    except Exception:
        return weaviate.connect_to_local(
            host=parsed["host"],
            port=parsed["port"],
            grpc_port=grpc_port,
            auth_credentials=auth_credentials,
        )


def _list_collection_names(client: Any) -> List[str]:
    listing = _as_primitive(client.collections.list_all())
    if isinstance(listing, dict):
        return sorted([_clean_text(name) for name in listing.keys() if _clean_text(name)])
    names: List[str] = []
    if isinstance(listing, list):
        for item in listing:
            if isinstance(item, dict):
                name = _clean_text(item.get("name") or item.get("class"))
            else:
                name = _clean_text(item)
            if name:
                names.append(name)
    return sorted(set(names))


def _collection_total_count(client: Any, collection_name: str) -> int:
    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(total_count=True)
    count = getattr(result, "total_count", None)
    if count is None and isinstance(result, dict):
        count = result.get("total_count")
    try:
        return int(count or 0)
    except Exception:
        return 0


def _weaviate_backup_instructions() -> str:
    return (
        "# Weaviate Backup Instructions\n\n"
        "Default export mode is metadata-only and read-only.\n\n"
        "Optional backup execution can be requested with:\n\n"
        "```bash\n"
        "PYTHONPATH=backend python3 backend/scripts/export_weaviate_data.py \\\n"
        "  --release v0.6-chemical-explorer \\\n"
        "  --backup-id runtime-backup-001 \\\n"
        "  --backend filesystem \\\n"
        "  --execute-backup\n"
        "```\n\n"
        "Rebuild alternative:\n"
        "- re-ingest from JSONL/parquet embedding artifacts using existing ETL scripts.\n"
    )


def _export_unavailable(
    target_dir: Path,
    release: str,
    reason: str,
    *,
    backup_id: str,
    backend: str,
    execute_backup: bool,
) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        target_dir / "collection_counts.json",
        {
            "status": "unavailable",
            "reason": reason,
            "generated_at_utc": generated_at,
            "collections": [],
        },
    )
    _write_json(
        target_dir / "schema_snapshot.json",
        {
            "status": "unavailable",
            "reason": reason,
            "generated_at_utc": generated_at,
            "collections": {},
        },
    )
    backup_manifest = {
        "status": "unavailable",
        "reason": reason,
        "release": release,
        "generated_at_utc": generated_at,
        "backup_id": backup_id,
        "backend": backend,
        "execute_backup": execute_backup,
        "backup_execution": {
            "requested": bool(execute_backup),
            "attempted": False,
            "executed": False,
            "detail": "metadata-only export with unavailable runtime",
        },
    }
    _write_json(target_dir / "backup_manifest.json", backup_manifest)
    _write_text(target_dir / "weaviate_backup_instructions.md", _weaviate_backup_instructions())
    return backup_manifest


def export_weaviate_data(
    release: str = DEFAULT_RELEASE,
    output_root: Optional[str] = None,
    *,
    backup_id: str = "",
    backend: str = "filesystem",
    execute_backup: bool = False,
) -> Dict[str, Any]:
    target_dir = weaviate_export_dir(release, output_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    backup_id_clean = _clean_text(backup_id) or f"{release}-metadata-snapshot"
    backend_clean = _clean_text(backend) or "filesystem"

    if weaviate is None:
        return _export_unavailable(
            target_dir,
            release,
            "weaviate-client is not installed.",
            backup_id=backup_id_clean,
            backend=backend_clean,
            execute_backup=execute_backup,
        )

    try:
        config = get_weaviate_config()
    except Exception as exc:
        return _export_unavailable(
            target_dir,
            release,
            str(exc),
            backup_id=backup_id_clean,
            backend=backend_clean,
            execute_backup=execute_backup,
        )

    client = None
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        client = _connect_weaviate(config)
        if not bool(client.is_ready()):
            return _export_unavailable(
                target_dir,
                release,
                "weaviate client connected but is_ready() returned False.",
                backup_id=backup_id_clean,
                backend=backend_clean,
                execute_backup=execute_backup,
            )

        collection_names = _list_collection_names(client)
        collection_counts: List[Dict[str, Any]] = []
        schema_details: Dict[str, Any] = {}
        for name in collection_names:
            count = _collection_total_count(client, name)
            collection_counts.append({"collection": name, "object_count": count})
            try:
                cfg_obj = client.collections.get(name).config.get()
                schema_details[name] = _as_primitive(cfg_obj)
            except Exception as exc:
                schema_details[name] = {"error": _clean_text(exc)}

        _write_json(
            target_dir / "collection_counts.json",
            {
                "status": "ok",
                "generated_at_utc": generated_at,
                "collections": collection_counts,
                "total_collections": len(collection_counts),
                "total_objects": sum(int(item["object_count"]) for item in collection_counts),
                "read_only": True,
            },
        )
        _write_json(
            target_dir / "schema_snapshot.json",
            {
                "status": "ok",
                "generated_at_utc": generated_at,
                "collections": schema_details,
                "collection_names": collection_names,
                "read_only": True,
            },
        )

        backup_execution: Dict[str, Any] = {
            "requested": bool(execute_backup),
            "attempted": False,
            "executed": False,
            "detail": "metadata-only mode",
        }
        if execute_backup:
            backup_execution["attempted"] = True
            try:
                result_obj: Any = None
                if hasattr(client, "backup") and hasattr(client.backup, "create"):
                    try:
                        result_obj = client.backup.create(
                            backup_id=backup_id_clean,
                            backend=backend_clean,
                        )
                    except TypeError:
                        result_obj = client.backup.create(backup_id_clean, backend_clean)
                backup_execution["executed"] = result_obj is not None
                backup_execution["result"] = _as_primitive(result_obj)
                if result_obj is None:
                    backup_execution["detail"] = "backup API unavailable in current client; no backup created"
                else:
                    backup_execution["detail"] = "backup create invoked"
            except Exception as exc:
                backup_execution["executed"] = False
                backup_execution["detail"] = f"backup invocation failed: {_clean_text(exc)}"

        backup_manifest = {
            "status": "ok",
            "release": release,
            "generated_at_utc": generated_at,
            "backup_id": backup_id_clean,
            "backend": backend_clean,
            "execute_backup": bool(execute_backup),
            "backup_execution": backup_execution,
            "read_only_by_default": True,
        }
        _write_json(target_dir / "backup_manifest.json", backup_manifest)
    except Exception as exc:
        return _export_unavailable(
            target_dir,
            release,
            f"weaviate export error: {exc}",
            backup_id=backup_id_clean,
            backend=backend_clean,
            execute_backup=execute_backup,
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    _write_text(target_dir / "weaviate_backup_instructions.md", _weaviate_backup_instructions())
    return {
        "status": "ok",
        "release": release,
        "generated_at_utc": generated_at,
        "files": expected_weaviate_files(),
        "read_only_by_default": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Weaviate runtime metadata (read-only by default)")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Release name")
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional root directory for runtime exports. Defaults to data/exports/runtime.",
    )
    parser.add_argument("--backup-id", default="", help="Optional backup id")
    parser.add_argument(
        "--backend",
        default="filesystem",
        choices=["filesystem", "s3", "gcs", "azure"],
        help="Optional backup backend selector",
    )
    parser.add_argument(
        "--execute-backup",
        action="store_true",
        help="Attempt backup invocation. Default is metadata-only safe mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = export_weaviate_data(
        release=_clean_text(args.release) or DEFAULT_RELEASE,
        output_root=args.output_root or None,
        backup_id=args.backup_id,
        backend=args.backend,
        execute_backup=bool(args.execute_backup),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
