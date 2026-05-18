"""ETL step 6: ingest AOP-Wiki XML into Neo4j via APOC (streaming).

- Ensures the XML file exists (decompresses .gz if needed).
- Uses apoc.load.xml + apoc.periodic.iterate to batch ingest and prints node count.
"""

from __future__ import annotations

import gzip
import os
import shutil

from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "04_biological_pathways")
XML_NAME = "aop-wiki-xml-2026-04-01.xml"
GZ_NAME = "aop-wiki-xml-2026-04-01.gz"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ihatepassword"


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


def to_file_uri(abs_path: str) -> str:
    # Neo4j file URI format: file:///absolute/path
    return f"file:///{abs_path.lstrip('/')}"


def ensure_neo4j_import_mirror(abs_path: str) -> str:
    """Mirror the XML into Neo4j import directory if possible.

    Neo4j APOC import is restricted to the import directory, which means
    file:///absolute/path maps to <import_dir>/absolute/path.
    """
    import_root = os.environ.get("NEO4J_IMPORT_DIR", "/var/lib/neo4j/import")
    mirror_path = os.path.join(import_root, abs_path.lstrip("/"))
    mirror_dir = os.path.dirname(mirror_path)

    if os.path.exists(mirror_path):
        return mirror_path

    os.makedirs(mirror_dir, exist_ok=True)
    shutil.copy2(abs_path, mirror_path)
    return mirror_path


def main() -> None:
    xml_path = ensure_xml_exists()
    abs_path = os.path.abspath(xml_path)
    try:
        ensure_neo4j_import_mirror(abs_path)
    except Exception as exc:
        print(
            "Warning: could not mirror XML into Neo4j import directory. "
            f"Proceeding with direct file URI. Error: {exc}"
        )

    file_uri = to_file_uri(abs_path)

    xpath = "/*/*"
    cypher = (
        "CALL apoc.periodic.iterate("
        f'\'CALL apoc.load.xml("{file_uri}", "{xpath}") YIELD value RETURN value\', '
        "'CREATE (n:XmlWord) SET n.payload = apoc.convert.toJson(value), n.tag = value._type', "
        "{batchSize:1000, parallel:false}) "
        "YIELD total RETURN total AS node_count"
    )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(cypher)
        record = result.single()
        count = record["node_count"] if record else 0
        print(f"Ingested nodes: {count}")

    driver.close()


if __name__ == "__main__":
    main()
