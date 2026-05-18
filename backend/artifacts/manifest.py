"""Data models for artifact manifest specification and status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional

ExpectedType = Literal["csv", "json", "jsonl", "parquet", "md"]


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


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    category: str
    local_path: str
    required_for: List[str]
    required: bool
    expected_type: ExpectedType
    min_size_bytes: int
    description: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactSpec":
        expected_type = _clean_text(raw.get("expected_type"))
        if expected_type not in {"csv", "json", "jsonl", "parquet", "md"}:
            raise ValueError(f"Unsupported expected_type: {expected_type}")

        required_for_raw = raw.get("required_for")
        required_for: List[str] = []
        if isinstance(required_for_raw, list):
            for item in required_for_raw:
                text = _clean_text(item)
                if text:
                    required_for.append(text)

        artifact_id = _clean_text(raw.get("artifact_id"))
        if not artifact_id:
            raise ValueError("artifact_id is required")

        category = _clean_text(raw.get("category"))
        if not category:
            raise ValueError(f"category is required for {artifact_id}")

        local_path = _clean_text(raw.get("local_path"))
        if not local_path:
            raise ValueError(f"local_path is required for {artifact_id}")

        min_size = raw.get("min_size_bytes", 0)
        try:
            min_size_int = int(min_size)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"min_size_bytes must be an integer for {artifact_id}") from exc
        if min_size_int < 0:
            raise ValueError(f"min_size_bytes must be >= 0 for {artifact_id}")

        return cls(
            artifact_id=artifact_id,
            category=category,
            local_path=local_path,
            required_for=required_for,
            required=bool(raw.get("required", True)),
            expected_type=expected_type,  # type: ignore[arg-type]
            min_size_bytes=min_size_int,
            description=_clean_text(raw.get("description")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "category": self.category,
            "local_path": self.local_path,
            "required_for": list(self.required_for),
            "required": bool(self.required),
            "expected_type": self.expected_type,
            "min_size_bytes": int(self.min_size_bytes),
            "description": self.description,
        }


@dataclass(frozen=True)
class ArtifactStatus:
    artifact_id: str
    exists: bool
    size_bytes: int
    sha256: str
    modified_time: Optional[str]
    validation_status: str
    failure_reason: str
    category: str = ""
    required: bool = True
    local_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "exists": bool(self.exists),
            "size_bytes": int(self.size_bytes),
            "sha256": self.sha256,
            "modified_time": self.modified_time,
            "validation_status": self.validation_status,
            "failure_reason": self.failure_reason,
            "category": self.category,
            "required": bool(self.required),
            "local_path": self.local_path,
        }
