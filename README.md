# Alcohol Intelligence Monorepo

Local-first alcohol risk estimation and safety guidance system with deterministic guards.

## Safety Disclaimer
- Estimates only.
- Not medical advice.
- Not legal/driving advice.
- Never use this system to decide whether it is safe to drive.

## Monorepo Layout
- `backend/`: FastAPI backend, reasoning pipeline, simulation, RAG integration, scripts, tests, Docker config.
- `frontend/`: Placeholder for future React app.
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

## Run Backend Tests
```bash
PYTHONPATH=backend python3 -m pytest -q backend/tests
```

## Docker (Local)
```bash
docker compose -f backend/docker-compose.local.yml up --build
```

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
