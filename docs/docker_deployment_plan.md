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

Frontend build:

```bash
docker build \
  -f frontend/Dockerfile \
  -t alcohol-intelligence-frontend:local \
  --build-arg VITE_API_BASE_URL=http://localhost:8000 \
  frontend
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

Frontend smoke test:

```bash
docker run -d --name alcohol-intelligence-frontend-test -p 5173:80 alcohol-intelligence-frontend:local
curl -I http://localhost:5173
curl http://localhost:5173 | head
docker stop alcohol-intelligence-frontend-test
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

1. GitHub Actions builds Docker images from main branch.
2. Push backend and frontend images to GHCR.
3. Azure Container Apps pulls and runs backend image.
4. Frontend static image is served separately with production API URL baked via `VITE_API_BASE_URL`.
5. Supabase provides artifact storage/config for runtime artifact resolution.
