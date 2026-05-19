"""Supabase Storage abstraction for artifact scripts.

This module is intentionally lazy-loaded so normal backend runtime does not
require supabase dependencies or credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import get_supabase_config


class SupabaseArtifactStore:
    def __init__(self, *, require_credentials: bool = True) -> None:
        self._config = get_supabase_config(require=require_credentials)
        self._client: Any = None

    @property
    def bucket(self) -> str:
        return self._config["artifact_bucket"]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from supabase import create_client  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency branch
            raise RuntimeError(
                "Supabase SDK is not installed. Install `supabase` to use artifact upload/download scripts."
            ) from exc

        url = self._config.get("url", "")
        service_role_key = self._config.get("service_role_key", "")
        if not url or not service_role_key:
            raise RuntimeError(
                "Supabase URL/service role key is missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )

        self._client = create_client(url, service_role_key)
        return self._client

    @staticmethod
    def _is_forbidden_local_path(local_path: str) -> bool:
        normalized = local_path.replace("\\", "/")
        forbidden_prefixes = (
            ".env",
            "frontend/node_modules/",
            "frontend/dist/",
            "docs/research_papers/",
            "data/interim/",
            "__pycache__/",
        )
        forbidden_suffixes = (".pyc",)
        if (
            normalized == ".env"
            or normalized.endswith("/.env")
            or normalized.endswith("/.env.local")
            or normalized.endswith(forbidden_suffixes)
        ):
            return True
        return any(normalized.startswith(prefix) for prefix in forbidden_prefixes)

    def list_artifacts(self, prefix: str) -> List[Dict[str, Any]]:
        client = self._get_client()
        storage = client.storage.from_(self.bucket)

        normalized = prefix.strip("/")
        folder = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
        search = normalized.rsplit("/", 1)[-1] if normalized else ""

        try:
            rows = storage.list(path=folder or None, options={"search": search} if search else None)
        except TypeError:
            rows = storage.list(folder)

        if not isinstance(rows, list):
            return []
        return [item for item in rows if isinstance(item, dict)]

    def exists(self, remote_path: str) -> bool:
        normalized = remote_path.strip("/")
        rows = self.list_artifacts(normalized)
        target_name = normalized.rsplit("/", 1)[-1]
        for row in rows:
            name = str(row.get("name", "")).strip()
            if name == target_name:
                return True
        return False

    def upload_file(self, local_path: str, remote_path: str, overwrite: bool = False) -> Dict[str, Any]:
        if self._is_forbidden_local_path(local_path):
            raise ValueError(f"Refusing to upload forbidden local path: {local_path}")

        source = Path(local_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        client = self._get_client()
        storage = client.storage.from_(self.bucket)

        payload = source.read_bytes()
        options = {"upsert": "true" if overwrite else "false"}

        try:
            result = storage.upload(remote_path.strip("/"), payload, file_options=options)
        except TypeError:
            result = storage.upload(remote_path.strip("/"), payload, options)
        return result if isinstance(result, dict) else {"result": result}

    def download_file(self, remote_path: str, local_path: str, overwrite: bool = False) -> Path:
        target = Path(local_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Local target already exists: {local_path}")

        client = self._get_client()
        storage = client.storage.from_(self.bucket)

        data = storage.download(remote_path.strip("/"))
        if isinstance(data, (bytes, bytearray)):
            content = bytes(data)
        elif hasattr(data, "read"):
            content = data.read()
        else:
            content = bytes(data)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def upload_json(self, local_path: str, remote_path: str, overwrite: bool = False) -> Dict[str, Any]:
        source = Path(local_path)
        with source.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        serialized = json.dumps(parsed, indent=2, sort_keys=True).encode("utf-8")

        if self._is_forbidden_local_path(local_path):
            raise ValueError(f"Refusing to upload forbidden local path: {local_path}")

        client = self._get_client()
        storage = client.storage.from_(self.bucket)
        options = {"upsert": "true" if overwrite else "false", "content-type": "application/json"}
        try:
            result = storage.upload(remote_path.strip("/"), serialized, file_options=options)
        except TypeError:
            result = storage.upload(remote_path.strip("/"), serialized, options)
        return result if isinstance(result, dict) else {"result": result}

    def download_json(self, remote_path: str, local_path: str, overwrite: bool = False) -> Dict[str, Any]:
        target = self.download_file(remote_path=remote_path, local_path=local_path, overwrite=overwrite)
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
