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


def test_dockerfile_copies_only_manifest_from_data() -> None:
    text = _read(BACKEND_ROOT / "Dockerfile")
    assert "COPY data/artifact_manifest.example.json /app/data/artifact_manifest.example.json" in text
    assert "COPY data/ /app/data" not in text
    assert "COPY data /app/data" not in text


def test_dockerignore_excludes_data_raw_and_embeddings() -> None:
    text = _read(MONOREPO_ROOT / ".dockerignore")
    assert "data/raw" in text
    assert "data/processed" in text
    assert "data/interim" in text
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


def test_docker_run_script_sets_project_root() -> None:
    text = _read(BACKEND_ROOT / "scripts/docker_run_api.sh")
    assert 'PROJECT_ROOT=/app' in text


def test_docker_publish_workflow_exists() -> None:
    path = MONOREPO_ROOT / ".github/workflows/docker-publish.yml"
    assert path.exists()


def test_docker_publish_workflow_references_backend_dockerfile_and_ghcr() -> None:
    text = _read(MONOREPO_ROOT / ".github/workflows/docker-publish.yml")
    assert "file: backend/Dockerfile" in text
    assert "ghcr.io/sigmanumeric78/new-testament-api" in text
    assert "docker/login-action" in text
    assert "docker/setup-buildx-action" in text
    assert "docker/build-push-action" in text


def test_docker_publish_workflow_does_not_echo_secrets() -> None:
    text = _read(MONOREPO_ROOT / ".github/workflows/docker-publish.yml").lower()
    assert "echo ${{ secrets." not in text
    assert "supabase_service_role_key" not in text
    assert "neo4j_password" not in text
