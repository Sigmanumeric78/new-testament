from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dockerfile_exists() -> None:
    path = BACKEND_ROOT / "Dockerfile"
    assert path.exists()


def test_dockerignore_exists() -> None:
    path = MONOREPO_ROOT / ".dockerignore"
    assert path.exists()


def test_dockerfile_does_not_copy_env() -> None:
    text = _read(BACKEND_ROOT / "Dockerfile").lower()
    assert "copy .env" not in text


def test_dockerignore_excludes_data_raw_and_embeddings() -> None:
    text = _read(MONOREPO_ROOT / ".dockerignore")
    assert "data/raw" in text
    assert "data/processed/weaviate/embedded" in text


def test_docker_compose_exists() -> None:
    path = BACKEND_ROOT / "docker-compose.local.yml"
    assert path.exists()


def test_compose_has_port_and_host_mapping() -> None:
    text = _read(BACKEND_ROOT / "docker-compose.local.yml")
    assert '"8000:8000"' in text
    assert "host.docker.internal:host-gateway" in text


def test_requirements_exists() -> None:
    path = BACKEND_ROOT / "requirements.txt"
    assert path.exists()


def test_required_scripts_exist_and_executable() -> None:
    scripts = [
        BACKEND_ROOT / "scripts/docker_build.sh",
        BACKEND_ROOT / "scripts/docker_run_api.sh",
        BACKEND_ROOT / "scripts/docker_smoke_test.sh",
    ]
    for script in scripts:
        assert script.exists(), f"missing script: {script}"
        assert os.access(script, os.X_OK), f"script is not executable: {script}"
