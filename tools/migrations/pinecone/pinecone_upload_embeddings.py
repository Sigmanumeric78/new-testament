#!/usr/bin/env python3
"""Upload existing parquet embeddings to Pinecone.

This script migrates precomputed vectors only. It does not generate embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.config import get_pinecone_config, resolve_project_path  # noqa: E402
from vectorstores.pinecone_store import PineconeVectorStore  # noqa: E402


DEFAULT_ROOT = "data/processed/weaviate/embedded"
UPLOAD_SOURCE = "weaviate_parquet_migration"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        text = _clean_text(value)
        if not text:
            return "{}"
        try:
            parsed = json.loads(text)
        except Exception:
            return text
        return json.dumps(parsed, sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    return "{}"


def _parse_vector(value: Any) -> List[float]:
    data = value
    if isinstance(data, str):
        data = json.loads(data)
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, (list, tuple)):
        raise ValueError("embedding value must be a list-like vector")
    return [float(item) for item in data]


def _chunks(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def build_vector_record(row: Mapping[str, Any], *, source_file: str, dimension: int) -> Dict[str, Any]:
    chunk_id = _clean_text(row.get("chunk_id"))
    if not chunk_id:
        raise ValueError("Embedding row is missing chunk_id; cannot build stable Pinecone id.")
    vector = _parse_vector(row.get("embedding"))
    if len(vector) != int(dimension):
        raise ValueError(f"Embedding dimension mismatch for {chunk_id}: expected {dimension}, got {len(vector)}.")

    content = _clean_text(row.get("content"))
    collection = _clean_text(row.get("collection"))
    metadata = {
        "object_id": _clean_text(row.get("object_id")),
        "chunk_id": chunk_id,
        "collection": collection,
        "source_collection": collection,
        "title": _clean_text(row.get("title")),
        "content": content,
        "content_excerpt": content[:320],
        "metadata_json": _json_string(row.get("metadata")),
        "provenance_json": _json_string(row.get("provenance")),
        "source_file": source_file,
        "source": UPLOAD_SOURCE,
    }
    return {"id": chunk_id, "values": vector, "metadata": metadata}


def load_vectors(root: Path, *, dimension: int) -> List[Dict[str, Any]]:
    parquet_files = sorted(root.glob("*_embeddings.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No *_embeddings.parquet files found under {root.as_posix()}")

    vectors: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        required_columns = {"object_id", "chunk_id", "collection", "title", "content", "embedding", "metadata", "provenance"}
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{parquet_path.as_posix()} missing required columns: {', '.join(missing)}")
        for _, row in df.iterrows():
            record = build_vector_record(row, source_file=parquet_path.as_posix(), dimension=dimension)
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate chunk_id across embedding parquet files: {record['id']}")
            seen_ids.add(record["id"])
            vectors.append(record)
    return vectors


def existing_ids(index: Any, *, namespace: str, ids: Sequence[str]) -> set[str]:
    if not ids:
        return set()
    response = index.fetch(ids=list(ids), namespace=namespace)
    if isinstance(response, Mapping):
        vectors = response.get("vectors", {})
    else:
        vectors = getattr(response, "vectors", {})
    if isinstance(vectors, Mapping):
        return set(str(item) for item in vectors.keys())
    return set()


def upload_vectors(
    store: PineconeVectorStore,
    vectors: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    force: bool,
) -> Dict[str, Any]:
    index = store._get_index()
    uploaded = 0
    skipped = 0
    for batch in _chunks(list(vectors), batch_size):
        upload_batch = batch
        if not force:
            found = existing_ids(index, namespace=store.namespace, ids=[item["id"] for item in batch])
            upload_batch = [item for item in batch if item["id"] not in found]
            skipped += len(batch) - len(upload_batch)
        if upload_batch:
            index.upsert(vectors=upload_batch, namespace=store.namespace)
            uploaded += len(upload_batch)
    return {
        "uploaded_count": int(uploaded),
        "skipped_existing_count": int(skipped),
        "force": bool(force),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload existing parquet embeddings to Pinecone")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root containing *_embeddings.parquet files")
    parser.add_argument("--namespace", default="", help="Pinecone namespace")
    parser.add_argument("--batch-size", type=int, default=100, help="Pinecone upsert batch size")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and report without Pinecone writes")
    parser.add_argument("--force", action="store_true", help="Upsert vectors even when ids already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = get_pinecone_config(require=not bool(args.dry_run))
        if _clean_text(args.namespace):
            config["namespace"] = _clean_text(args.namespace)
        root = resolve_project_path(args.root)
        batch_size = max(1, int(args.batch_size))
        vectors = load_vectors(root, dimension=int(config["dimension"]))
    except Exception as exc:
        print(json.dumps({"backend": "pinecone", "error": True, "message": str(exc)}, indent=2, sort_keys=True))
        return 1

    payload: Dict[str, Any] = {
        "backend": "pinecone",
        "index": _clean_text(config.get("index")),
        "namespace": _clean_text(config.get("namespace")),
        "root": root.as_posix(),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "candidate_count": len(vectors),
        "uploaded_count": 0,
        "skipped_existing_count": 0,
        "source": UPLOAD_SOURCE,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    store = PineconeVectorStore(config=config)
    try:
        summary = upload_vectors(store, vectors, batch_size=batch_size, force=bool(args.force))
        payload.update(summary)
    except Exception as exc:
        payload.update({"error": True, "message": str(exc)})
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    finally:
        store.close()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
