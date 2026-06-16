"""MongoDB Atlas/GridFS artifact storage abstraction.

The MongoDB client and GridFS dependencies are imported lazily so normal
backend imports do not require MongoDB credentials or optional runtime packages.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_DATABASE = "healthlens_artifacts"
DEFAULT_GRIDFS_BUCKET = "artifact_files"
ARTIFACT_SOURCE = "mongodb_gridfs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def infer_content_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix == ".parquet":
        return "application/octet-stream"
    if suffix == ".md":
        return "text/markdown"
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def is_forbidden_artifact_path(local_path: str) -> bool:
    normalized = local_path.replace("\\", "/").strip().lstrip("./")
    lowered = normalized.lower()
    forbidden_prefixes = (
        "frontend/node_modules/",
        "frontend/dist/",
        "docs/research_papers/",
        "__pycache__/",
    )
    if (
        lowered == ".env"
        or lowered.startswith(".env.")
        or lowered.endswith("/.env")
        or lowered.endswith("/.env.local")
        or lowered.endswith(".pyc")
    ):
        return True
    if lowered == "data/raw/food_health" or lowered.startswith("data/raw/food_health/"):
        return True
    if lowered.startswith("data/chunks/pytest") or ("/pytest-" in lowered and "/chunks/" in lowered):
        return True
    return any(lowered.startswith(prefix) for prefix in forbidden_prefixes)


class MongoArtifactStore:
    """Small MongoDB/GridFS adapter for artifact release upload and restore."""

    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        bucket: Optional[str] = None,
        client: Any = None,
        db: Any = None,
        gridfs_bucket: Any = None,
    ) -> None:
        self.uri = clean_text(uri if uri is not None else os.getenv("MONGODB_URI"))
        self.database_name = clean_text(database if database is not None else os.getenv("MONGODB_DATABASE")) or DEFAULT_DATABASE
        self.bucket_name = (
            clean_text(bucket if bucket is not None else os.getenv("MONGODB_GRIDFS_BUCKET")) or DEFAULT_GRIDFS_BUCKET
        )
        self._client = client
        self._db = db
        self._bucket = gridfs_bucket
        if self._client is None and self._db is None and not self.uri:
            raise ValueError(
                "Missing MongoDB configuration value: MONGODB_URI. "
                "Set it in the environment before using MongoDB artifact storage."
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from pymongo import MongoClient  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError(
                "pymongo is not installed. Install `pymongo` to use MongoDB artifact storage."
            ) from exc

        self._client = MongoClient(
            self.uri,
            serverSelectionTimeoutMS=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "10000")),
        )
        return self._client

    def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        self._db = self._get_client()[self.database_name]
        return self._db

    def _get_bucket(self) -> Any:
        if self._bucket is not None:
            return self._bucket
        try:
            from gridfs import GridFSBucket  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError(
                "pymongo GridFS support is not installed. Install `pymongo` to use MongoDB artifact storage."
            ) from exc
        self._bucket = GridFSBucket(self._get_db(), bucket_name=self.bucket_name)
        return self._bucket

    def _collection(self, name: str) -> Any:
        return self._get_db()[name]

    def ping(self) -> bool:
        client = self._get_client()
        if hasattr(client, "admin"):
            client.admin.command("ping")
        else:
            self._get_db().command("ping")
        return True

    def find_existing_artifact(self, release: str, artifact_id: str, sha256: str) -> Optional[Dict[str, Any]]:
        query = {
            "release": clean_text(release),
            "artifact_id": clean_text(artifact_id),
            "sha256": clean_text(sha256),
            "source": ARTIFACT_SOURCE,
        }
        existing = self._collection("artifact_manifest_entries").find_one(query)
        return dict(existing) if isinstance(existing, Mapping) else existing

    def upload_file(self, local_path: str, metadata: Mapping[str, Any], *, force: bool = False) -> Dict[str, Any]:
        metadata_dict = dict(metadata)
        manifest_local_path = clean_text(metadata_dict.get("local_path")) or local_path
        if is_forbidden_artifact_path(manifest_local_path):
            raise ValueError(f"Refusing to upload forbidden artifact path: {manifest_local_path}")

        source = Path(local_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Local artifact file does not exist: {source.as_posix()}")

        release = clean_text(metadata_dict.get("release"))
        artifact_id = clean_text(metadata_dict.get("artifact_id"))
        sha256 = clean_text(metadata_dict.get("sha256"))
        if not release or not artifact_id or not sha256:
            raise ValueError("Artifact metadata must include release, artifact_id, and sha256.")

        existing = self.find_existing_artifact(release, artifact_id, sha256)
        if existing and not force:
            existing["status"] = "skipped_existing"
            return existing

        uploaded_at = utc_now_iso()
        record = dict(metadata_dict)
        record["uploaded_at_utc"] = uploaded_at
        record["content_type"] = clean_text(record.get("content_type")) or infer_content_type(manifest_local_path)
        record["source"] = ARTIFACT_SOURCE

        gridfs_metadata = {
            key: value
            for key, value in record.items()
            if key not in {"gridfs_file_id", "_id"}
        }
        filename = clean_text(record.get("remote_path")) or manifest_local_path
        with source.open("rb") as handle:
            gridfs_file_id = self._get_bucket().upload_from_stream(
                filename,
                handle,
                metadata=gridfs_metadata,
            )

        record["gridfs_file_id"] = gridfs_file_id
        self._collection("artifact_manifest_entries").update_one(
            {"release": release, "artifact_id": artifact_id},
            {
                "$set": record,
                "$setOnInsert": {"created_at_utc": uploaded_at},
            },
            upsert=True,
        )
        record["status"] = "uploaded"
        return record

    @staticmethod
    def _coerce_gridfs_file_id(file_id: Any) -> Any:
        if not isinstance(file_id, str):
            return file_id
        try:
            from bson import ObjectId  # type: ignore

            if ObjectId.is_valid(file_id):
                return ObjectId(file_id)
        except Exception:
            return file_id
        return file_id

    def download_file(self, gridfs_file_id: Any, local_path: str, *, overwrite: bool = False) -> Path:
        target = Path(local_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Local target already exists: {target.as_posix()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            self._get_bucket().download_to_stream(
                self._coerce_gridfs_file_id(gridfs_file_id),
                handle,
            )
        return target

    def list_release_artifacts(self, release: str, *, required_only: bool = False) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"release": clean_text(release), "source": ARTIFACT_SOURCE}
        if required_only:
            query["required"] = True
        cursor = self._collection("artifact_manifest_entries").find(query)
        try:
            cursor = cursor.sort([("local_path", 1), ("artifact_id", 1)])
            rows = list(cursor)
        except Exception:
            rows = list(cursor)
            rows.sort(key=lambda item: (clean_text(item.get("local_path")), clean_text(item.get("artifact_id"))))
        return [dict(row) if isinstance(row, Mapping) else row for row in rows]

    def create_or_update_release(
        self,
        release: str,
        *,
        manifest: Optional[Mapping[str, Any]] = None,
        manifest_path: str = "",
        upload_summary: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        manifest_metadata = {
            key: value
            for key, value in dict(manifest or {}).items()
            if key != "artifacts"
        }
        record: Dict[str, Any] = {
            "release": clean_text(release),
            "release_name": clean_text((manifest or {}).get("release_name")) or clean_text(release),
            "source": ARTIFACT_SOURCE,
            "database": self.database_name,
            "gridfs_bucket": self.bucket_name,
            "manifest_path": clean_text(manifest_path),
            "manifest_metadata": manifest_metadata,
            "updated_at_utc": now,
        }
        if manifest is not None:
            artifacts = list(manifest.get("artifacts", []) or [])
            record["artifact_count"] = int(manifest.get("artifact_count", len(artifacts)) or len(artifacts))
            record["required_artifact_count"] = int(manifest.get("required_artifact_count", 0) or 0)
            record["missing_artifact_count"] = int(manifest.get("missing_artifact_count", 0) or 0)
            record["chunked_artifact_count"] = int(manifest.get("chunked_artifact_count", 0) or 0)
        if upload_summary is not None:
            record["upload_summary"] = dict(upload_summary)

        self._collection("artifact_releases").update_one(
            {"release": clean_text(release)},
            {"$set": record, "$setOnInsert": {"created_at_utc": now}},
            upsert=True,
        )
        return record

    def log_restore(self, release: str, report: Mapping[str, Any]) -> Dict[str, Any]:
        record = {
            "release": clean_text(release),
            "source": ARTIFACT_SOURCE,
            "logged_at_utc": utc_now_iso(),
            "report": dict(report),
        }
        self._collection("artifact_restore_logs").insert_one(record)
        return record

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
