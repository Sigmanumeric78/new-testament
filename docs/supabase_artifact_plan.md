# Supabase Artifact Plan

## Purpose
Store non-Git artifact payloads in Supabase Storage while keeping the repository code-first.

## Bucket
- `alcohol-intelligence-artifacts`
- Large files are chunked automatically using `SUPABASE_MAX_UPLOAD_MB` (default `45`).

## Proposed Layout
```text
releases/
  v0.1-local-intelligence/
    release_metadata.json
    artifact_manifest.json
    checksums.sha256
    data/...
processed/
embeddings/
neo4j/
weaviate/
reports/
```

## Commands
Create release bundle metadata:

```bash
PYTHONPATH=backend python3 backend/scripts/create_release_bundle.py --release v0.1-local-intelligence
```

Upload dry-run (default safe mode):

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_upload_supabase.py --release v0.1-local-intelligence --dry-run
```

Upload execute:

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_upload_supabase.py --release v0.1-local-intelligence --execute
```

Upload execute with overwrite (safe re-run after partial upload):

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_upload_supabase.py --release v0.1-local-intelligence --execute --overwrite
```

Download dry-run:

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py --release v0.1-local-intelligence --dry-run
```

Download execute:

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_download_supabase.py --release v0.1-local-intelligence --execute
```

Chunked artifact restoration behavior:
- Oversized artifacts are uploaded as `releases/<release>/chunks/.../part_XXXXX` plus `chunk_manifest.json`.
- Download restores originals by downloading chunk manifest + parts and reassembling with SHA256 validation.

Verify local release integrity:

```bash
PYTHONPATH=backend python3 backend/scripts/artifact_verify_release.py --release v0.1-local-intelligence
```

## Security
- `SUPABASE_SERVICE_ROLE_KEY` is sensitive and must never be logged, committed, or exposed in frontend code.
- Scripts are dry-run by default and only perform remote operations with `--execute`.
- Forbidden upload candidates include `.env`, frontend build/dependency folders, and research-paper paths.
- `data/chunks/`, `data/releases/`, and `data/exports/` are generated local runtime artifacts and are git-ignored.

## Relation to Future Azure Container Apps
- Container startup will later validate required artifacts via release manifests/checksums.
- Artifact download/verify can be invoked in deployment automation before app startup.
- This phase does not configure Azure; it only establishes release manifest and storage contract foundations.
