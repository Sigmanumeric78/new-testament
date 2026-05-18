"""ETL step 6 (streaming): ingest full AOP-Wiki XML into Neo4j.

- Streams XML with iterparse to avoid large memory usage.
- Batches node + relationship writes to Neo4j.
"""

from __future__ import annotations

import gzip
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "04_biological_pathways")
XML_NAME = "aop-wiki-xml-2026-04-01.xml"
GZ_NAME = "aop-wiki-xml-2026-04-01.gz"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ihatepassword"

LABEL = "XmlNode"
REL_TYPE = "PARENT_OF"
BATCH_SIZE = 1000
CLEAR_EXISTING = True


def ensure_xml_exists() -> str:
    xml_path = os.path.join(DATA_DIR, XML_NAME)
    gz_path = os.path.join(DATA_DIR, GZ_NAME)

    if os.path.exists(xml_path):
        return xml_path

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing XML and GZ: {xml_path} / {gz_path}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with gzip.open(gz_path, "rb") as gz_file, open(xml_path, "wb") as out_file:
        out_file.write(gz_file.read())

    return xml_path


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def flush_batch(session, nodes: List[Dict], rels: List[Dict]) -> None:
    if nodes:
        session.run(
            f"UNWIND $rows AS row CREATE (n:{LABEL} "
            "{xml_id: row.id, tag: row.tag, attrs: row.attrs, text: row.text})",
            rows=nodes,
        )
    if rels:
        session.run(
            f"UNWIND $rows AS row MATCH (p:{LABEL} {{xml_id: row.parent_id}}), "
            f"(c:{LABEL} {{xml_id: row.child_id}}) CREATE (p)-[:{REL_TYPE}]->(c)",
            rows=rels,
        )


def main() -> None:
    xml_path = ensure_xml_exists()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        if CLEAR_EXISTING:
            session.run(f"MATCH (n:{LABEL}) DETACH DELETE n")

        session.run(
            f"CREATE CONSTRAINT xml_id_unique IF NOT EXISTS "
            f"FOR (n:{LABEL}) REQUIRE n.xml_id IS UNIQUE"
        )

        node_batch: List[Dict] = []
        rel_batch: List[Dict] = []
        stack: List[int] = []
        next_id = 1
        total_nodes = 0

        for event, elem in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start":
                node_id = next_id
                next_id += 1

                parent_id: Optional[int] = stack[-1] if stack else None

                tag = strip_ns(elem.tag)
                attrs = json.dumps(elem.attrib) if elem.attrib else None
                text = elem.text.strip() if elem.text and elem.text.strip() else None

                node_batch.append(
                    {"id": node_id, "tag": tag, "attrs": attrs, "text": text}
                )
                if parent_id is not None:
                    rel_batch.append({"parent_id": parent_id, "child_id": node_id})

                stack.append(node_id)

                if len(node_batch) >= BATCH_SIZE:
                    flush_batch(session, node_batch, rel_batch)
                    total_nodes += len(node_batch)
                    node_batch.clear()
                    rel_batch.clear()
                    if total_nodes % 10000 == 0:
                        print(f"Inserted {total_nodes} nodes...")

            elif event == "end":
                if stack:
                    stack.pop()
                elem.clear()

        if node_batch or rel_batch:
            flush_batch(session, node_batch, rel_batch)
            total_nodes += len(node_batch)

        print(f"Ingested nodes: {total_nodes}")

    driver.close()


if __name__ == "__main__":
    main()
