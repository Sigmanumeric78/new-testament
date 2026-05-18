"""Full pipeline integration test across Pandas, Weaviate, and Neo4j."""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import weaviate
from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOX_PATH = os.path.join(BASE_DIR, "data", "processed", "standardized_toxicity.csv")

WEAVIATE_PORT = 8080
WEAVIATE_GRPC = 50051

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ihatepassword"


def pandas_summary() -> Dict:
    if not os.path.exists(TOX_PATH):
        return {"status": "MISSING", "message": f"Missing {TOX_PATH}"}

    df = pd.read_csv(TOX_PATH)
    total = len(df)
    non_null_norm = (
        df["Normalized_Toxicity"].notna().sum()
        if "Normalized_Toxicity" in df.columns
        else 0
    )
    non_null_ld50 = df["ld50_value"].notna().sum() if "ld50_value" in df.columns else 0
    non_null_mw = (
        df["exact_molecular_weight"].notna().sum()
        if "exact_molecular_weight" in df.columns
        else 0
    )

    unique_cas = (
        df["cas_number"].dropna().astype(str).nunique()
        if "cas_number" in df.columns
        else 0
    )
    unique_pubchem = (
        df["pubchem_cid"].dropna().astype(str).nunique()
        if "pubchem_cid" in df.columns
        else 0
    )

    smiles_col = (
        "canonical_smiles" if "canonical_smiles" in df.columns else "rdkit_smiles"
    )
    unique_smiles = (
        df[smiles_col].dropna().astype(str).nunique() if smiles_col in df.columns else 0
    )

    return {
        "status": "OK",
        "total_rows": total,
        "non_null_normalized_toxicity": non_null_norm,
        "non_null_ld50": non_null_ld50,
        "non_null_exact_mw": non_null_mw,
        "unique_cas": unique_cas,
        "unique_pubchem_cid": unique_pubchem,
        "unique_smiles": unique_smiles,
        "smiles_column": smiles_col,
    }


def weaviate_summary() -> Dict:
    try:
        client = weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC)
    except Exception as exc:
        return {"status": "ERROR", "message": f"Weaviate connection failed: {exc}"}

    try:
        collections = client.collections.list_all()
        names = list(collections.keys()) if isinstance(collections, dict) else []
        summary = {"status": "OK", "collections": names}
        return summary
    finally:
        client.close()


def neo4j_summary() -> Dict:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as exc:
        return {"status": "ERROR", "message": f"Neo4j connection failed: {exc}"}

    with driver.session() as session:
        total_res = session.run(
            "MATCH (n:XmlNode) RETURN count(n) AS total, count(distinct n.tag) AS tag_types"
        ).single()
        total = total_res["total"] if total_res else 0
        tag_types = total_res["tag_types"] if total_res else 0

        top_tags = session.run(
            "MATCH (n:XmlNode) RETURN n.tag AS tag, count(*) AS count "
            "ORDER BY count DESC LIMIT 10"
        )
        top = [{"tag": r["tag"], "count": r["count"]} for r in top_tags]

    driver.close()

    return {
        "status": "OK",
        "total_nodes": total,
        "tag_types": tag_types,
        "top_tags": top,
    }


def main() -> None:
    print("=== Full Pipeline Integration Test ===")

    p = pandas_summary()
    print("\n[1] Pandas (standardized_toxicity.csv)")
    if p["status"] == "OK":
        print(f"Rows: {p['total_rows']}")
        print(f"Non-null Normalized_Toxicity: {p['non_null_normalized_toxicity']}")
        print(f"Non-null LD50: {p['non_null_ld50']}")
        print(f"Non-null Exact MW: {p['non_null_exact_mw']}")
        print(f"Unique CAS: {p['unique_cas']}")
        print(f"Unique PubChem CID: {p['unique_pubchem_cid']}")
        print(f"Unique SMILES ({p['smiles_column']}): {p['unique_smiles']}")
    else:
        print(p.get("message", p))

    w = weaviate_summary()
    print("\n[2] Weaviate")
    if w["status"] == "OK":
        print(f"Collections: {w['collections']}")
    else:
        print(w.get("message", w))

    n = neo4j_summary()
    print("\n[3] Neo4j")
    if n["status"] == "OK":
        print(f"Total XmlNode nodes: {n['total_nodes']}")
        print(f"Distinct tag types: {n['tag_types']}")
        print("Top tags:")
        for t in n["top_tags"]:
            print(f"  {t['tag']}: {t['count']}")
    else:
        print(n.get("message", n))

    all_ok = p["status"] == "OK" and w["status"] == "OK" and n["status"] == "OK"
    print("\n=== Summary ===")
    print(f"Pandas: {p['status']}")
    print(f"Weaviate: {w['status']}")
    print(f"Neo4j: {n['status']}")
    print(f"Pipeline Healthy: {'YES' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()
