# Runtime Data Export Plan (Neo4j + Weaviate)

## Purpose
- Keep Docker images code-only.
- Store runtime graph/vector state as separate artifacts.
- Preserve repeatable restore paths without baking databases into containers.

## Export Target
- `data/exports/runtime/<release>/`
- Default release: `v0.6-chemical-explorer`

Expected layout:

```text
data/exports/runtime/<release>/
  neo4j/
    cypher_counts.json
    node_counts.json
    relationship_counts.json
    schema_snapshot.json
    graph_export_manifest.json
    neo4j_dump_instructions.md
  weaviate/
    collection_counts.json
    schema_snapshot.json
    backup_manifest.json
    weaviate_backup_instructions.md
  artifact_manifest.runtime.json
  checksums.sha256
```

## Scripts
- `backend/scripts/export_neo4j_data.py`
  - Read-only Cypher snapshots.
  - Node counts by label.
  - Relationship counts by type.
  - Schema snapshot (constraints/indexes when accessible).
- `backend/scripts/export_weaviate_data.py`
  - Read-only metadata export by default.
  - Collection names/counts.
  - Schema/config snapshot.
  - Optional backup invocation only when `--execute-backup` is passed.
- `backend/scripts/verify_runtime_exports.py`
  - Verifies expected files.
  - Writes `artifact_manifest.runtime.json` and `checksums.sha256`.
- `backend/scripts/export_runtime_artifacts.sh`
  - Runs Neo4j export -> Weaviate export -> verify.

## Safety Defaults
- Read-only by default.
- No graph mutation.
- No vector/object mutation.
- No hardcoded credentials.
- Uses env/config through `backend/utils/config.py`.

## Supabase Storage Role
- Supabase Storage stores release artifacts/backups outside Git.
- Runtime export outputs are intended upload candidates for release packaging.

## Restore Strategies
### Neo4j
- Rebuild from processed CSV artifacts using existing ETL ingestion scripts.
- Or restore a native dump manually (see `neo4j_dump_instructions.md`).

### Weaviate
- Rebuild from JSONL/parquet embeddings using existing ingestion scripts.
- Or restore from backup where backup backend is configured.

## Architecture Clarification
- MongoDB Atlas is **not** replacing Neo4j or Weaviate.
- MongoDB Atlas is not replacing Neo4j or Weaviate.
- Neo4j remains graph reasoning storage.
- Weaviate remains vector retrieval storage.
- MongoDB may be added later for sessions/history/analytics.
