#!/usr/bin/env python3
"""Smoke test Pinecone retrieval against the first scientific evidence vector."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from reasoning.hybrid_orchestrator import _parse_vector_value  # noqa: E402
from utils.config import get_data_root  # noqa: E402
from vectorstores.pinecone_store import PineconeVectorStore  # noqa: E402


def main() -> int:
    parquet_path = get_data_root() / "processed" / "weaviate" / "embedded" / "scientific_evidence_embeddings.parquet"
    if not parquet_path.exists():
        print(json.dumps({"error": True, "message": f"missing parquet: {parquet_path.as_posix()}"}, indent=2, sort_keys=True))
        return 1

    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(json.dumps({"error": True, "message": "scientific evidence parquet is empty"}, indent=2, sort_keys=True))
        return 1

    first = df.iloc[0]
    expected_chunk_id = str(first["chunk_id"])
    vector = _parse_vector_value(first["embedding"])
    store = PineconeVectorStore()
    try:
        matches = store.query_by_vector(vector, top_k=5, collections=["ScientificEvidence"])
    finally:
        store.close()

    top_matches: List[Dict[str, Any]] = [
        {
            "rank": idx + 1,
            "id": item.get("id"),
            "chunk_id": item.get("chunk_id"),
            "object_id": item.get("object_id"),
            "collection": item.get("collection"),
            "title": item.get("title"),
            "score": item.get("score"),
        }
        for idx, item in enumerate(matches[:5])
    ]
    observed = str(top_matches[0]["chunk_id"]) if top_matches else ""
    payload = {
        "backend": "pinecone",
        "expected_chunk_id": expected_chunk_id,
        "observed_top_chunk_id": observed,
        "match_count": len(matches),
        "top_matches": top_matches,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if observed != expected_chunk_id:
        raise AssertionError(f"Expected top match {expected_chunk_id}, observed {observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
