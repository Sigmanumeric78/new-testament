# Docker Deployment Plan (Phase 09C)

## Local Full-Stack Development

Terminal 1:

```bash
cd backend
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## Local Build

```bash
bash backend/scripts/docker_build.sh
```

## Local Run (single container)

```bash
bash backend/scripts/docker_run_api.sh
```

Manual run should include:
- `PROJECT_ROOT=/app`
- `PYTHONPATH=/app/backend`

## Local Run (compose)

```bash
docker compose -f backend/docker-compose.local.yml up --build
```

## Local Smoke Test

```bash
bash backend/scripts/docker_smoke_test.sh
```

## Required Host Services (for full functionality)

- Neo4j on host (`bolt://localhost:7687`)
- Weaviate on host (`http://localhost:8080`, gRPC `50051`)
- Ollama on host (`http://localhost:11434`)

The Docker container connects to these through `host.docker.internal`.

## Environment Variables

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `WEAVIATE_URL`
- `WEAVIATE_GRPC_HOST`
- `WEAVIATE_GRPC_PORT`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`

## Artifact Strategy

- GitHub stores code and lightweight committed manifests/reports.
- Supabase Storage (future phase) will store heavy artifacts and runtime data artifacts.
- Docker image remains code-first; no raw datasets or embedding parquet files are baked in.

## Future Plan

1. GitHub Actions builds Docker image from main branch.
2. Push image to GHCR.
3. Azure Container Apps pulls and runs the image.
4. Supabase provides artifact storage/config for runtime artifact resolution.
