# Artifact Storage Policy

## In Git (Versioned)
- Source code under `backend/`
- Tests under `backend/tests/`
- Documentation under `docs/`
- Lightweight manifest metadata (`data/artifact_manifest.example.json`)
- Final lightweight validation/audit summaries required for reproducibility

## Outside Git (Artifact Storage)
- Raw datasets
- PDFs/research dumps
- PubChem JSON/SDF bulk artifacts
- Embedding parquet/vector outputs
- Weaviate runtime data
- Neo4j runtime data
- Model fine-tuning artifacts
- Large logs/intermediate ETL outputs

## Planned External Storage (Supabase)
Bucket suggestion:
- `alcohol-intelligence-artifacts`

Suggested layout:
```text
releases/v0.1-local-intelligence/
processed/
embeddings/
neo4j/
weaviate/
reports/
```

## Regeneration Commands
Run from repo root with `PYTHONPATH=backend`:

```bash
PYTHONPATH=backend python3 backend/etl/etl_07b_weaviate_materialization.py
PYTHONPATH=backend python3 backend/etl/etl_07c_embedding_generation.py
PYTHONPATH=backend python3 backend/etl/etl_07d_weaviate_schema_init.py
PYTHONPATH=backend python3 backend/etl/etl_07e_weaviate_ingestion.py
PYTHONPATH=backend python3 backend/reasoning/pipeline_quality_audit.py --compact
PYTHONPATH=backend python3 backend/reasoning/scientific_validity_audit.py
```

Supabase upload/download integration is planned, not implemented in this phase.
