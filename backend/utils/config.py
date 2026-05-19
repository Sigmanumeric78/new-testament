"""Shared deterministic project configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

REQUIRED_NEO4J_KEYS = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)


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


def get_data_root() -> Path:
    return get_project_root() / "data"


def resolve_project_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return get_project_root() / candidate


def resolve_data_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return get_data_root() / candidate


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
    config = {
        "uri": os.getenv("NEO4J_URI", "").strip(),
        "user": os.getenv("NEO4J_USER", "").strip(),
        "password": os.getenv("NEO4J_PASSWORD", "").strip(),
        "database": os.getenv("NEO4J_DATABASE", "").strip() or "neo4j",
    }
    missing: List[str] = []
    if not config["uri"]:
        missing.append("NEO4J_URI")
    if not config["user"]:
        missing.append("NEO4J_USER")
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


def get_weaviate_config() -> Dict[str, str]:
    _load_dotenv()
    default_grpc_host = "localhost"
    default_grpc_port = "50051"
    config = {
        "url": os.getenv("WEAVIATE_URL", "").strip(),
        "grpc_host": os.getenv("WEAVIATE_GRPC_HOST", "").strip() or default_grpc_host,
        "grpc_port": os.getenv("WEAVIATE_GRPC_PORT", "").strip() or default_grpc_port,
        "api_key": os.getenv("WEAVIATE_API_KEY", "").strip(),
    }
    missing: List[str] = []
    if not config["url"]:
        missing.append("WEAVIATE_URL")
    if missing:
        raise ValueError(
            "Missing Weaviate configuration values: "
            + ", ".join(missing)
            + ". Provide them via environment variables or project .env."
        )
    try:
        int(config["grpc_port"])
    except ValueError as exc:
        raise ValueError(
            "WEAVIATE_GRPC_PORT must be an integer."
        ) from exc
    return config


def get_ollama_config() -> Dict[str, str]:
    _load_dotenv()
    config = {
        "host": os.getenv("OLLAMA_HOST", "").strip() or "http://localhost:11434",
        "model": os.getenv("OLLAMA_MODEL", "").strip() or "qwen2.5:3b",
    }
    return config


def get_supabase_config(*, require: bool = False) -> Dict[str, str]:
    _load_dotenv()
    config = {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        "anon_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "artifact_bucket": os.getenv("SUPABASE_ARTIFACT_BUCKET", "").strip() or "alcohol-intelligence-artifacts",
    }
    if require:
        missing: List[str] = []
        if not config["url"]:
            missing.append("SUPABASE_URL")
        if not config["service_role_key"]:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise ValueError(
                "Missing Supabase configuration values: "
                + ", ".join(missing)
                + ". These are only required when running Supabase artifact scripts."
            )
    return config
