"""Unified pipeline test for Ethanol across storage layers.

Checks:
1) Pandas math engine (standardized_toxicity.csv)
2) Weaviate vector DB (ScientificMonograph)
3) Neo4j knowledge graph (XmlNode)
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import weaviate
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOX_PATH = os.path.join(BASE_DIR, "data", "processed", "standardized_toxicity.csv")

WEAVIATE_PORT = 8080
WEAVIATE_GRPC = 50051
WEAVIATE_COLLECTION = "ScientificMonograph"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ihatepassword"


def find_ethanol_row(df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str]]:
    cas_col = "cas_number"
    pubchem_col = "pubchem_cid"

    if cas_col in df.columns:
        mask = df[cas_col].astype(str).str.fullmatch("64-17-5", case=False, na=False)
        if mask.any():
            return df[mask].iloc[0], f"{cas_col}=64-17-5"

    if pubchem_col in df.columns:
        mask = df[pubchem_col].astype(str).str.fullmatch("702", case=False, na=False)
        if mask.any():
            return df[mask].iloc[0], f"{pubchem_col}=702"

    # look for canonical ethanol identifiers if present
    for col, value in [
        ("inchi", "InChI=1S/C2H6O"),
        ("inchikey", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        ("rdkit_smiles", "CCO"),
        ("canonical_smiles", "CCO"),
        ("smiles", "CCO"),
        ("molecular_formula", "C2H6O"),
    ]:
        if col in df.columns:
            mask = df[col].astype(str).str.fullmatch(value, case=False, na=False)
            if mask.any():
                return df[mask].iloc[0], f"{col}={value}"

    # fallback: search any object column for whole-word "ethanol"
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        mask = pd.Series([False] * len(df))
        for c in text_cols:
            mask = mask | df[c].astype(str).str.contains(
                r"\bethanol\b", case=False, na=False
            )
        if mask.any():
            return df[mask].iloc[0], "text_match=ethanol"

    return None, None


def test_math_engine() -> dict:
    df = pd.read_csv(TOX_PATH)
    row, match_source = find_ethanol_row(df)
    if row is None:
        return {
            "status": "NOT_FOUND",
            "message": "Ethanol not found in standardized_toxicity.csv",
        }

    exact_mw = row.get("exact_molecular_weight")
    ld50_raw = row.get("ld50_value")
    norm_tox = row.get("Normalized_Toxicity")

    return {
        "status": "OK",
        "match_source": match_source,
        "exact_molecular_weight": exact_mw,
        "ld50_value": ld50_raw,
        "normalized_toxicity": norm_tox,
    }


def test_weaviate() -> dict:
    try:
        client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC)
    except Exception as exc:
        return {"status": "ERROR", "message": f"Weaviate connection failed: {exc}"}

    try:
        if not client.collections.exists(WEAVIATE_COLLECTION):
            return {
                "status": "NOT_FOUND",
                "message": f"Collection {WEAVIATE_COLLECTION} not found",
            }

        collection = client.collections.get(WEAVIATE_COLLECTION)
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        query_vector = model.encode(
            ["Ethanol physical properties and toxicity"], normalize_embeddings=True
        )[0].tolist()

        try:
            result = collection.query.hybrid(
                query="Ethanol physical properties and toxicity",
                vector=query_vector,
                limit=2,
            )
        except Exception as exc:
            return {"status": "ERROR", "message": f"Weaviate query failed: {exc}"}

        chunks: List[str] = []
        for obj in result.objects:
            content = obj.properties.get("content") if obj.properties else None
            if content:
                chunks.append(content)

        if not chunks:
            return {"status": "NOT_FOUND", "message": "No chunks returned"}

        return {"status": "OK", "chunks": chunks}
    finally:
        client.close()


def test_neo4j() -> dict:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as exc:
        return {"status": "ERROR", "message": f"Neo4j connection failed: {exc}"}

    query = (
        "MATCH (n:XmlNode) "
        "WHERE (n.text IS NOT NULL AND toLower(n.text) CONTAINS 'ethanol') "
        "   OR (n.attrs IS NOT NULL AND toLower(n.attrs) CONTAINS 'ethanol') "
        "RETURN n.tag AS tag, n.text AS text, n.attrs AS attrs "
        "LIMIT 3"
    )
    with driver.session() as session:
        rows = list(session.run(query))
    driver.close()

    if not rows:
        return {"status": "NOT_FOUND", "message": "No ethanol nodes found"}

    results = []
    for r in rows:
        results.append({"tag": r["tag"], "text": r["text"], "attrs": r["attrs"]})

    return {"status": "OK", "results": results}


def main() -> None:
    print("=== Ethanol Pipeline Integration Test ===")

    math_result = test_math_engine()
    print("\n[1] Math Engine (Pandas)")
    if math_result["status"] == "OK":
        print(f"Match source: {math_result.get('match_source')}")
        print(f"Exact MW: {math_result['exact_molecular_weight']}")
        print(f"LD50 (raw): {math_result['ld50_value']}")
        print(f"Normalized_Toxicity: {math_result['normalized_toxicity']}")
    else:
        print(math_result["message"])

    weaviate_result = test_weaviate()
    print("\n[2] Vector DB (Weaviate)")
    if weaviate_result["status"] == "OK":
        for idx, chunk in enumerate(weaviate_result["chunks"], start=1):
            print(f"--- Chunk {idx} ---")
            print(chunk)
    else:
        print(weaviate_result["message"])

    neo4j_result = test_neo4j()
    print("\n[3] Knowledge Graph (Neo4j)")
    if neo4j_result["status"] == "OK":
        for idx, row in enumerate(neo4j_result["results"], start=1):
            print(f"--- Node {idx} ---")
            print(f"tag: {row['tag']}")
            print(f"text: {row['text']}")
            print(f"attrs: {row['attrs']}")
    else:
        print(neo4j_result["message"])

    all_ok = (
        math_result["status"] == "OK"
        and weaviate_result["status"] == "OK"
        and neo4j_result["status"] == "OK"
    )

    print("\n=== Summary ===")
    print(f"Math Engine: {math_result['status']}")
    print(f"Weaviate: {weaviate_result['status']}")
    print(f"Neo4j: {neo4j_result['status']}")
    print(f"Pipeline Verified: {'YES' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()
