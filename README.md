# Alcohol Intelligence Local V1

A local alcohol-risk reasoning system that combines simulation, causal knowledge, semantic retrieval, grounded synthesis, and safety guarding to provide conservative end-user guidance.

## Safety Disclaimer
- Estimates only; outputs are uncertainty-aware approximations.
- Not medical advice.
- Not legal or driving advice.
- Never use this system to decide whether it is safe to drive.

## What The System Does
- Parses natural-language alcohol questions.
- Routes queries to the required reasoning modules.
- Runs deterministic PBPK-style alcohol simulations when appropriate.
- Uses graph/retrieval evidence to support mechanistic and toxicity context.
- Produces plain-language risk guidance with conservative safety constraints.
- Blocks unsafe or unsupported responses before display.

## Architecture Summary
- PBPK simulator: `simulation/pbpk/pbpk_master_simulator.py`
- Neo4j causal graph reasoning
- Weaviate semantic retrieval
- Qwen2.5 3B via Ollama (grounded synthesis layer)
- User risk advisor: plain-language conservative guidance
- Grounding/safety guard: deterministic validation before display

## Local Services Required
- Neo4j
- Weaviate (Docker/local)
- Ollama with `qwen2.5:3b`

## Local Setup
1. Create environment file:
   - `cp .env.example .env`
2. Install Python dependencies in your environment (example):
   - `pip install -U pandas scipy neo4j weaviate-client pytest`
3. Start Weaviate (local Docker or equivalent service).
4. Start Neo4j and confirm DB credentials match `.env`.
5. Verify Ollama model:
   - `ollama pull qwen2.5:3b`
   - `ollama list`

## CLI Commands
- Health check:
  - `python3 app_cli.py --health`
- Demo run:
  - `python3 app_cli.py --demo`
- Single query pretty output:
  - `python3 app_cli.py --query "I am 75 kg male, fed, I drank 200 ml vodka in 1 hour, should I keep drinking?" --pretty`
- Guided intake mode:
  - `python3 app_cli.py --intake`

## Testing and Audits
- Full tests:
  - `pytest -q`
- Pipeline quality audit:
  - `python3 reasoning/pipeline_quality_audit.py --compact`
- Scientific validity/truthfulness audit:
  - `python3 reasoning/scientific_validity_audit.py`

## Artifact Policy
- GitHub stores code, tests, docs, and lightweight reproducible artifacts.
- Azure Blob stores raw/large/generated artifacts (datasets, PDFs, embeddings, runtime DB artifacts, model artifacts).
- See `ARTIFACTS.md` for the Azure container layout and regeneration commands.
