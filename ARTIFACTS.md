# Artifact Storage Policy

## In Git (Versioned)
- Source code (`app_cli.py`, `reasoning/`, `simulation/`, `etl/`, `rag/`, `utils/`)
- Tests (`tests/`)
- Documentation and architecture notes (`docs/`, schema/design markdown/cypher files)
- Small processed tables and lightweight JSONL assets used for reproducible local runs
- Final validation and audit reports required for reproducibility and release gates

## Outside Git (Artifact Storage)
- Raw datasets (food composition, pathway corpora, PubChem source files)
- Large scientific PDFs and paper dumps
- PubChem JSON/SDF bulk artifacts
- Embedding parquet files and generated vector payloads
- Weaviate vector database runtime contents
- Neo4j runtime databases and transaction files
- Model training/fine-tuning artifacts (e.g., LoRA checkpoints)

## Azure Artifact Layout
Use one Blob container for non-Git artifacts:

`alcohol-intelligence-artifacts/`

```text
raw/
  food_composition/
  pubchem/
  human_metabolism_pdfs/
  jecfa/

processed/
  weaviate_jsonl/
  embeddings_parquet/
  pbpk_tables/
  beverage_tables/

model/
  qwen_lora/
  prompt_templates/

reports/
  validation/
  audits/

releases/
  v0.1-local-intelligence/
```

## Regeneration Commands
Run from repo root:

```bash
python3 etl/etl_07b_weaviate_materialization.py
python3 etl/etl_07c_embedding_generation.py
python3 etl/etl_07d_weaviate_schema_init.py
python3 etl/etl_07e_weaviate_ingestion.py
python3 reasoning/pipeline_quality_audit.py --compact
python3 reasoning/scientific_validity_audit.py
```
