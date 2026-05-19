# Alcohol Intelligence Monorepo

Local-first alcohol risk estimation and safety guidance system with deterministic guards.

## Safety Disclaimer
- Estimates only.
- Not medical advice.
- Not legal/driving advice.
- Never use this system to decide whether it is safe to drive.

## Monorepo Layout
- `backend/`: FastAPI backend, reasoning pipeline, simulation, RAG integration, scripts, tests, Docker config.
- `frontend/`: React + TypeScript + Vite user interface (Zer0 G0nd0g0l).
- `infra/`: Placeholder docs for Docker/CI/CD/Azure/DNS setup.
- `docs/`: Architecture, memory, deployment planning.
- `data/artifact_manifest.example.json`: committed lightweight artifact manifest only.

## Backend Architecture
- PBPK simulator: `backend/simulation/pbpk/pbpk_master_simulator.py`
- Neo4j causal graph integration
- Weaviate semantic retrieval
- Qwen2.5 3B via Ollama (grounded synthesis)
- Grounding/safety guard before user display
- User risk advisor for plain-language conservative guidance
- Chemical Explorer API for compound search/detail/conformer retrieval

## Local Services
- Neo4j
- Weaviate
- Ollama with `qwen2.5:3b`

## Setup
1. Copy env file:
   - `cp .env.example .env`
2. Install backend dependencies:
   - `pip install -r backend/requirements.txt`
3. Start Neo4j and Weaviate locally.
4. Ensure Ollama is running:
   - `ollama pull qwen2.5:3b`

## Run Backend API
```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Chemical Explorer Endpoints
- `GET /chemicals`
- `GET /chemicals/{compound_id}`
- `GET /chemicals/{compound_id}/conformer`

These endpoints are read-only and built from local processed beverage compound data and local PubChem JSON/SDF structure files.

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

Terminal 3:
```bash
./scripts/fullstack_acceptance_check.sh
```

Recommended when Ollama may be cold:
```bash
PREWARM_OLLAMA=true CURL_TIMEOUT=120 ./scripts/fullstack_acceptance_check.sh
```

Open `http://localhost:5173`.

Routes:
- `/` Ask and intake workflow
- `/explorer` Chemical Explorer (compound catalog + conformer viewer)

If port 5173 is in use:
```bash
lsof -i :5173
kill <PID>
```
Then restart the frontend dev server.

## Run Backend Tests
```bash
PYTHONPATH=backend python3 -m pytest -q backend/tests
```

## Run Frontend
```bash
cd frontend
npm install
npm run dev
```

## Docker (Local)
```bash
docker compose -f backend/docker-compose.local.yml up --build
```

Manual `docker run` should set:
- `PROJECT_ROOT=/app`
- `DATA_ROOT=/app/data`
- `PYTHONPATH=/app/backend`
- `RESTORE_ARTIFACTS_ON_STARTUP=false` (default)
- `ARTIFACT_RELEASE=v0.6-chemical-explorer` (default)

To test Supabase restore-on-startup (no local `data/` mount):

```bash
docker run --rm \
  --name alcohol-intelligence-api-test \
  --network host \
  --env-file .env \
  -e PROJECT_ROOT=/app \
  -e DATA_ROOT=/app/data \
  -e RESTORE_ARTIFACTS_ON_STARTUP=true \
  -e ARTIFACT_RELEASE=v0.6-chemical-explorer \
  -e PYTHONPATH=/app/backend \
  alcohol-intelligence-api:local
```

Startup behavior when restore mode is enabled:
- Downloads release manifest from Supabase.
- Restores runtime artifacts (runtime-only subset by default).
- Reassembles chunked artifacts if needed.
- Verifies checksums before API starts.
- Fails fast if required runtime artifacts cannot be restored.

Frontend image (static Nginx):

```bash
docker build \
  -f frontend/Dockerfile \
  -t alcohol-intelligence-frontend:local \
  --build-arg VITE_API_BASE_URL=http://localhost:8000 \
  frontend

docker run --rm \
  --name alcohol-intelligence-frontend-test \
  -p 5173:80 \
  alcohol-intelligence-frontend:local
```

Open `http://localhost:5173`.

## Useful Commands
```bash
PYTHONPATH=backend python3 backend/app_cli.py --health
PYTHONPATH=backend python3 backend/app_cli.py --demo
PYTHONPATH=backend python3 backend/app_cli.py --query "I am 75 kg male, fed, I drank 200 ml vodka in 1 hour, should I keep drinking?" --pretty
PYTHONPATH=backend python3 backend/app_cli.py --intake
```

## Artifact Policy
- GitHub stores code, tests, docs, and lightweight reproducible metadata.
- Large/raw/generated artifacts stay outside Git (planned Supabase Storage workflow).
- See `ARTIFACTS.md` and `docs/supabase_artifact_plan.md`.

## Supabase Artifact Workflow (Phase 09E)
```bash
PYTHONPATH=backend python3 backend/scripts/create_release_bundle.py --release v0.1-local-intelligence
PYTHONPATH=backend python3 backend/scripts/artifact_upload_supabase.py --release v0.1-local-intelligence --dry-run
PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py --release v0.1-local-intelligence --dry-run
PYTHONPATH=backend python3 backend/scripts/artifact_verify_release.py --release v0.1-local-intelligence
```

Use `--execute` only when you intentionally want live Supabase uploads/downloads.
Oversized artifacts are auto-chunked using `SUPABASE_MAX_UPLOAD_MB` (default `45`), and reassembled on download with SHA256 verification.

Runtime restore commands:

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py \
  --release v0.6-chemical-explorer \
  --execute \
  --overwrite \
  --runtime-only

PYTHONPATH=backend python3 backend/scripts/artifact_verify_release.py \
  --release v0.6-chemical-explorer \
  --runtime-only
```
