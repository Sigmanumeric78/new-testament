# Monorepo Structure

## Top-level Layout
- `backend/`: all backend runtime and tests
- `frontend/`: future frontend app
- `infra/`: deployment/ops placeholders
- `docs/`: architecture and planning docs
- `data/artifact_manifest.example.json`: committed artifact manifest only

## Backend Contents
- `api/`, `artifacts/`, `etl/`, `rag/`, `reasoning/`, `simulation/`, `scripts/`, `tests/`, `utils/`
- `app_cli.py`, `Dockerfile`, `docker-compose.local.yml`, `requirements.txt`

## Data Policy
Large local data is excluded from Git:
- raw datasets
- processed heavy artifacts
- embeddings
- runtime DB files
- logs and caches

Only lightweight manifest metadata is versioned under `data/`.
