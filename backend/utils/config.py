"""Shared deterministic project configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

REQUIRED_NEO4J_KEYS = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)

SUPPORTED_ARTIFACT_BACKEND = "mongodb"
SUPPORTED_VECTOR_BACKEND = "pinecone"


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def backend_root() -> Path:
    return get_backend_root()


def _resolve_env_root(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _is_project_root_candidate(path: Path) -> bool:
    return (path / "backend").is_dir()


def project_root() -> Path:
    env_root = _clean_text(os.getenv("PROJECT_ROOT"))
    if env_root:
        return _resolve_env_root(env_root)

    current = Path(__file__).resolve()
    best_with_data: Path | None = None
    best_without_data: Path | None = None

    for parent in current.parents:
        if not _is_project_root_candidate(parent):
            continue
        if (parent / "data").exists():
            best_with_data = parent
            break
        if best_without_data is None:
            best_without_data = parent

    if best_with_data is not None:
        return best_with_data
    if best_without_data is not None:
        return best_without_data
    return get_backend_root().parent if (get_backend_root().parent / "backend").is_dir() else get_backend_root()


def get_project_root() -> Path:
    return project_root()

def resolve_project_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    text = candidate.as_posix().lstrip("./")
    if text == "data":
        return get_data_root()
    if text.startswith("data/"):
        return get_data_root() / text[len("data/") :]
    return get_project_root() / candidate


def resolve_data_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    text = candidate.as_posix().lstrip("./")
    if text == "data":
        return get_data_root()
    if text.startswith("data/"):
        return get_data_root() / text[len("data/") :]
    return get_data_root() / candidate


def get_data_root() -> Path:
    env_data = _clean_text(os.getenv("DATA_ROOT"))
    if env_data:
        candidate = _resolve_env_root(env_data)
        return candidate
    return get_project_root() / "data"


def env_file_path() -> Path:
    repo_env = project_root() / ".env"
    if repo_env.exists():
        return repo_env
    backend_env = get_backend_root() / ".env"
    if backend_env.exists():
        return backend_env
    return repo_env


def _load_dotenv() -> None:
    # override=False preserves runtime environment precedence over .env values.
    load_dotenv(dotenv_path=env_file_path(), override=False)


def get_neo4j_config() -> Dict[str, str]:
    _load_dotenv()
    username = os.getenv("NEO4J_USERNAME", "").strip() or os.getenv("NEO4J_USER", "").strip()
    config = {
        "uri": os.getenv("NEO4J_URI", "").strip(),
        "user": username,
        "password": os.getenv("NEO4J_PASSWORD", "").strip(),
        "database": os.getenv("NEO4J_DATABASE", "").strip() or "neo4j",
    }
    missing: List[str] = []
    if not config["uri"]:
        missing.append("NEO4J_URI")
    if not config["user"]:
        missing.append("NEO4J_USERNAME")
    if not config["password"]:
        missing.append("NEO4J_PASSWORD")
    if not config["database"]:
        missing.append("NEO4J_DATABASE")
    if missing:
        raise ValueError(
            "Missing Neo4j configuration values: "
            + ", ".join(missing)
            + ". Provide them via environment variables or project .env."
        )
    return config


def get_artifact_backend() -> str:
    _load_dotenv()
    backend = os.getenv("ARTIFACT_STORE_BACKEND", "").strip().lower() or SUPPORTED_ARTIFACT_BACKEND
    if backend != SUPPORTED_ARTIFACT_BACKEND:
        raise ValueError(f"Unsupported artifact backend: {backend}. Supported backend: mongodb")
    return backend


def get_vector_backend() -> str:
    _load_dotenv()
    backend = os.getenv("VECTOR_BACKEND", "").strip().lower() or SUPPORTED_VECTOR_BACKEND
    if backend != SUPPORTED_VECTOR_BACKEND:
        raise ValueError(f"Unsupported vector backend: {backend}. Supported backend: pinecone")
    return backend


def get_mongodb_config(*, require: bool = True) -> Dict[str, str]:
    _load_dotenv()
    config = {
        "uri": os.getenv("MONGODB_URI", "").strip(),
        "database": os.getenv("MONGODB_DATABASE", "").strip() or "healthlens_artifacts",
        "gridfs_bucket": os.getenv("MONGODB_GRIDFS_BUCKET", "").strip() or "artifact_files",
    }
    if require and not config["uri"]:
        raise ValueError(
            "Missing MongoDB configuration values: MONGODB_URI. "
            "This is required for MongoDB Atlas/GridFS artifact storage."
        )
    return config


def get_pinecone_config(*, require: bool = True) -> Dict[str, str | int]:
    _load_dotenv()
    raw_dimension = os.getenv("PINECONE_DIMENSION", "").strip() or "768"
    try:
        dimension = int(raw_dimension)
    except ValueError as exc:
        raise ValueError("PINECONE_DIMENSION must be an integer.") from exc
    if dimension <= 0:
        raise ValueError("PINECONE_DIMENSION must be a positive integer.")

    config: Dict[str, str | int] = {
        "api_key": os.getenv("PINECONE_API_KEY", "").strip(),
        "index": os.getenv("PINECONE_INDEX", "").strip() or "healthlens-knowledge",
        "namespace": os.getenv("PINECONE_NAMESPACE", "").strip() or "production",
        "dimension": dimension,
        "metric": os.getenv("PINECONE_METRIC", "").strip() or "cosine",
    }
    if require and not config["api_key"]:
        raise ValueError(
            "Missing Pinecone configuration values: PINECONE_API_KEY. "
            "This is only required when VECTOR_BACKEND=pinecone or when running Pinecone scripts."
        )
    return config


def get_ollama_config() -> Dict[str, str]:
    _load_dotenv()
    raw_host = os.getenv("OLLAMA_HOST")
    host = raw_host.strip() if raw_host is not None else "http://localhost:11434"
    raw_model = os.getenv("OLLAMA_MODEL")
    model = raw_model.strip() if raw_model is not None else ""
    config = {
        "host": host,
        "model": model,
        "provider": os.getenv("LLM_PROVIDER", "").strip() or "ollama",
        "enabled": os.getenv("OLLAMA_ENABLED", "").strip() or "true",
        "api_key": os.getenv("OLLAMA_API_KEY", "").strip(),
        "allow_unlisted_model": os.getenv("OLLAMA_ALLOW_UNLISTED_MODEL", "").strip() or "false",
        "auto_select_model": os.getenv("OLLAMA_AUTO_SELECT_MODEL", "").strip() or "false",
    }
    return config
