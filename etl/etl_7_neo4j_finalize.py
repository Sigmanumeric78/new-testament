"""ETL step 7: finalize Neo4j XML ingestion with indexes and counts."""

from __future__ import annotations

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ihatepassword"
LABEL = "XmlNode"


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Create indexes for tag and commonly used attributes
        session.run(
            f"CREATE INDEX xmlnode_tag_index IF NOT EXISTS FOR (n:{LABEL}) ON (n.tag)"
        )
        session.run(
            f"CREATE INDEX xmlnode_attrs_index IF NOT EXISTS FOR (n:{LABEL}) ON (n.attrs)"
        )
        session.run(
            f"CREATE INDEX xmlnode_text_index IF NOT EXISTS FOR (n:{LABEL}) ON (n.text)"
        )

        # Node type counts by tag
        result = session.run(
            f"MATCH (n:{LABEL}) RETURN n.tag AS tag, count(*) AS count ORDER BY count DESC"
        )
        counts = list(result)

    driver.close()

    print("Node type counts (by tag):")
    for record in counts:
        print(f"{record['tag']}: {record['count']}")


if __name__ == "__main__":
    main()
