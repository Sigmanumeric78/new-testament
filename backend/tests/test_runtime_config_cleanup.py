from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import get_artifact_backend, get_vector_backend


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_default_artifact_backend_is_mongodb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTIFACT_STORE_BACKEND", raising=False)
    assert get_artifact_backend() == "mongodb"


def test_unsupported_artifact_backend_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORE_BACKEND", "local")
    with pytest.raises(ValueError, match="Unsupported artifact backend: local. Supported backend: mongodb"):
        get_artifact_backend()


def test_default_vector_backend_is_pinecone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    assert get_vector_backend() == "pinecone"


def test_unsupported_vector_backend_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_BACKEND", "legacy")
    with pytest.raises(ValueError, match="Unsupported vector backend: legacy. Supported backend: pinecone"):
        get_vector_backend()


def test_no_supabase_or_weaviate_imports_in_active_backend_source() -> None:
    offenders: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.parts[-2:-1] == ("tests",):
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        supabase_import = "supabase"
        weaviate_import = "weaviate"
        if f"import {supabase_import}" in lowered or f"from {supabase_import}" in lowered:
            offenders.append(path.relative_to(BACKEND_ROOT).as_posix())
        if f"import {weaviate_import}" in lowered or f"from {weaviate_import}" in lowered:
            offenders.append(path.relative_to(BACKEND_ROOT).as_posix())
    assert offenders == []


def test_env_example_contains_only_current_runtime_selectors() -> None:
    env_example = BACKEND_ROOT.parent / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    assert "ARTIFACT_STORE_BACKEND=mongodb" in text
    assert "VECTOR_BACKEND=pinecone" in text
    assert ("SUPABASE" + "_") not in text
    assert ("WEAVIATE" + "_") not in text
