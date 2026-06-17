"""Pinecone vector store adapter for runtime retrieval."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from utils.config import get_pinecone_config


PINECONE_RETRIEVAL_BACKEND = "pinecone"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict()
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            converted = value.model_dump()
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass
    return {}


def _coerce_matches(response: Any) -> List[Any]:
    if isinstance(response, Mapping):
        matches = response.get("matches", [])
    else:
        matches = getattr(response, "matches", [])
    if not isinstance(matches, list):
        return list(matches or [])
    return matches


def _parse_json_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = _clean_text(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _truncate(text: str, limit: int = 320) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit]


class PineconeUnavailableError(RuntimeError):
    """Raised when Pinecone cannot be reached or queried."""


class PineconeVectorStore:
    """Runtime Pinecone retrieval adapter.

    The Pinecone SDK is imported lazily so tests and non-Pinecone deployments do
    not need credentials or a network connection at import time.
    """

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        client: Any = None,
        index: Any = None,
    ) -> None:
        self.config = dict(config or get_pinecone_config(require=True))
        self.api_key = _clean_text(self.config.get("api_key"))
        self.index_name = _clean_text(self.config.get("index")) or "healthlens-knowledge"
        self.namespace = _clean_text(self.config.get("namespace")) or "production"
        self.dimension = int(self.config.get("dimension", 768) or 768)
        self.metric = _clean_text(self.config.get("metric")) or "cosine"
        self._client = client
        self._index = index
        if self.dimension <= 0:
            raise ValueError("PINECONE_DIMENSION must be a positive integer.")

    def _sanitize_error(self, exc: Exception) -> str:
        message = str(exc)
        if self.api_key:
            message = message.replace(self.api_key, "<redacted>")
        return message.replace("PINECONE_API_KEY", "PINECONE_API_KEY")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise PineconeUnavailableError("PINECONE_API_KEY is required for Pinecone vector retrieval.")
        try:
            from pinecone import Pinecone  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise PineconeUnavailableError("pinecone SDK is not installed.") from exc
        try:
            self._client = Pinecone(api_key=self.api_key)
        except Exception as exc:
            raise PineconeUnavailableError(f"Pinecone client initialization failed: {self._sanitize_error(exc)}") from exc
        return self._client

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index
        try:
            self._index = self._get_client().Index(self.index_name)
        except Exception as exc:
            raise PineconeUnavailableError(f"Pinecone index connection failed: {self._sanitize_error(exc)}") from exc
        return self._index

    def ping(self) -> Dict[str, Any]:
        try:
            stats = self._get_index().describe_index_stats()
            stats_dict = _coerce_mapping(stats)
            observed_dimension = stats_dict.get("dimension")
            if observed_dimension is not None and int(observed_dimension) != self.dimension:
                raise PineconeUnavailableError(
                    f"Pinecone dimension mismatch: expected {self.dimension}, observed {observed_dimension}."
                )
            return {
                "ok": True,
                "index": self.index_name,
                "namespace": self.namespace,
                "dimension": self.dimension,
                "metric": self.metric,
                "total_vector_count": int(stats_dict.get("total_vector_count", 0) or 0),
            }
        except PineconeUnavailableError:
            raise
        except Exception as exc:
            raise PineconeUnavailableError(f"Pinecone ping failed: {self._sanitize_error(exc)}") from exc

    def validate_vector_dimension(self, vector: Sequence[float]) -> List[float]:
        values = [float(item) for item in vector]
        if len(values) != self.dimension:
            raise ValueError(f"Pinecone query vector dimension mismatch: expected {self.dimension}, got {len(values)}.")
        return values

    @staticmethod
    def build_collection_filter(collections: Optional[Iterable[str]]) -> Optional[Dict[str, Any]]:
        selected = sorted({_clean_text(item) for item in list(collections or []) if _clean_text(item)})
        if not selected:
            return None
        return {
            "$or": [
                {"collection": {"$in": selected}},
                {"source_collection": {"$in": selected}},
            ]
        }

    def _normalize_match(self, match: Any) -> Dict[str, Any]:
        match_dict = _coerce_mapping(match)
        match_id = _clean_text(match_dict.get("id") or getattr(match, "id", ""))
        score = _to_float(match_dict.get("score") if "score" in match_dict else getattr(match, "score", None))
        metadata_raw = match_dict.get("metadata") if "metadata" in match_dict else getattr(match, "metadata", {})
        metadata = _coerce_mapping(metadata_raw)

        metadata_payload = _parse_json_dict(metadata.get("metadata_json") or metadata.get("metadata"))
        provenance_payload = _parse_json_dict(metadata.get("provenance_json") or metadata.get("provenance"))
        content = _clean_text(metadata.get("content"))
        content_excerpt = _clean_text(metadata.get("content_excerpt")) or _truncate(content)
        collection = _clean_text(metadata.get("collection")) or _clean_text(metadata.get("source_collection"))

        if self.metric == "cosine" and score is not None:
            distance = max(0.0, 1.0 - float(score))
        else:
            distance = None

        return {
            "id": match_id,
            "object_id": _clean_text(metadata.get("object_id")) or match_id,
            "chunk_id": _clean_text(metadata.get("chunk_id")) or match_id,
            "collection": collection,
            "title": _clean_text(metadata.get("title")),
            "content_excerpt": content_excerpt,
            "content": content,
            "score": _safe_round(score),
            "distance": _safe_round(distance),
            "metadata": metadata_payload,
            "provenance": provenance_payload,
            "source_file": _clean_text(metadata.get("source_file")),
            "source_dataset": _clean_text(metadata.get("source_dataset")),
            "retrieval_backend": PINECONE_RETRIEVAL_BACKEND,
        }

    def query_by_vector(
        self,
        vector: Sequence[float],
        top_k: int,
        collections: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        values = self.validate_vector_dimension(vector)
        query_filter = self.build_collection_filter(collections)
        try:
            response = self._get_index().query(
                vector=values,
                top_k=int(top_k),
                namespace=self.namespace,
                include_metadata=True,
                filter=query_filter,
            )
            matches = [_match for _match in _coerce_matches(response)]
            return [self._normalize_match(match) for match in matches]
        except ValueError:
            raise
        except Exception as exc:
            raise PineconeUnavailableError(f"Pinecone query failed: {self._sanitize_error(exc)}") from exc

    def close(self) -> None:
        if self._index is not None and hasattr(self._index, "close"):
            self._index.close()
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
