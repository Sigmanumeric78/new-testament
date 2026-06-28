# HealthLens Alcohol Intelligence

Production FastAPI/Lambda backend and React frontend for alcohol risk estimation, chemical exploration, grounded scientific retrieval, and conservative safety guidance.

## Safety
- Estimates only.
- Not medical advice.
- Not legal or driving advice.
- Never use this system to decide whether it is safe to drive.

## Architecture
- Frontend: AWS Amplify, React, TypeScript, Vite.
- Backend: FastAPI in an AWS Lambda container image.
- Artifact storage: MongoDB Atlas GridFS.
- Vector retrieval: Pinecone.
- Graph database: Neo4j Aura.
- Generation: verified remote Ollama model when enabled; deterministic grounded synthesis when Ollama is disabled or unavailable.

## Backend Components
- PBPK simulator: `backend/simulation/pbpk/pbpk_master_simulator.py`
- Semantic retrieval: Pinecone index `healthlens-knowledge`, namespace `production`, dimension `768`.
- Neo4j causal graph integration.
- Grounding and safety guard before user display.
- User risk advisor for plain-language conservative guidance.
- Chemical Explorer API for compound search, detail, and conformer retrieval.

## Environment
Copy `.env.example` to a local ignored env file and fill secrets outside Git.

Current runtime selectors:

```bash
ARTIFACT_STORE_BACKEND=mongodb
VECTOR_BACKEND=pinecone
```

Unsupported selector values fail at startup/configuration with a clear error.

## Run Backend API
```bash
cd backend
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

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

Open `http://localhost:5173`.

Routes:
- `/` Ask and intake workflow.
- `/explorer` Chemical Explorer.

## Docker
```bash
docker compose -f backend/docker-compose.local.yml up --build
```

Manual `docker run` should set:
- `PROJECT_ROOT=/app`
- `DATA_ROOT=/app/data` for local Docker, `/tmp/data` for Lambda.
- `PYTHONPATH=/app/backend`
- `ARTIFACT_STORE_BACKEND=mongodb`
- `VECTOR_BACKEND=pinecone`
- `RESTORE_ARTIFACTS_ON_STARTUP=true` when testing artifact restore.
- `ARTIFACT_RESTORE_MODE=background`
- `ARTIFACT_RELEASE=v0.6-chemical-explorer`

Startup restore uses MongoDB Atlas GridFS and restores only the runtime-required artifact subset.

## Health
`GET /health` reports these production components:
- `api`
- `neo4j`
- `mongodb`
- `artifact_status`
- `pinecone`
- `ollama`
- `pbpk`
- `router`
- `orchestrator`
- `synthesizer`
- `grounding_guard`
- `user_risk_advisor`

The health response must not include legacy storage or vector components. A deliberately disabled LLM reports as Ollama standby and does not degrade platform health.

## API Debug Shape
Semantic retrieval appears under `semantic_retrieval`:

```json
{
  "debug": {
    "route": {
      "required_modules": ["semantic_retrieval"]
    },
    "orchestration": {
      "module_results": {
        "semantic_retrieval": {
          "retrieval_backend": "pinecone",
          "query_vector_dimension": 768
        }
      }
    }
  }
}
```

## Useful Commands
```bash
PYTHONPATH=backend python3 backend/app_cli.py --health
PYTHONPATH=backend python3 backend/app_cli.py --demo
PYTHONPATH=backend python3 backend/app_cli.py --query "Show research on sulfites and alcohol headaches" --pretty
PYTHONPATH=backend python3 backend/app_cli.py --intake
```

## Artifact Policy
- GitHub stores code, tests, docs, and lightweight reproducible metadata.
- Large, raw, generated, and restored artifacts stay outside Git.
- Runtime artifact restore comes from MongoDB Atlas GridFS.
- Old artifact IDs or paths may contain historical backend names; do not rename or delete them unless runtime consumption and checksums are fully validated.

## Legacy Migrations
Completed migrations:
- Supabase Storage to MongoDB Atlas/GridFS.
- Weaviate Cloud to Pinecone.

Retained legacy migration utilities live under `tools/legacy_migrations/` and are not used in production runtime or copied into the Lambda image.
