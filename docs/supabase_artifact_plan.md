# Supabase Artifact Plan

## Objective
Store large/generated artifacts outside Git in Supabase Storage.

## Proposed Bucket
- `alcohol-intelligence-artifacts`

## Proposed Layout
- `releases/v0.1-local-intelligence/`
- `processed/`
- `embeddings/`
- `neo4j/`
- `weaviate/`
- `reports/`

## Planned Future Scripts
- `artifact_upload_supabase.py`
- `artifact_download_supabase.py`
- `artifact_verify_release.py`

## Validation Rules
- Use checksums and manifest validation before/after transfer.
- Keep secrets out of storage buckets and payload files.
