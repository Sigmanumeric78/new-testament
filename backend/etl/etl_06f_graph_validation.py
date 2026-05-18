"""ETL step 06f: Neo4j graph validation and explainability readiness audit."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency availability branch
    GraphDatabase = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import get_neo4j_config

LOGGER = logging.getLogger("etl_06f_graph_validation")

ENCODING = "utf-8"

REQUIRED_NODE_LABELS: Tuple[str, ...] = (
    "Beverage",
    "Compound",
    "ChemicalClass",
    "Enzyme",
    "PopulationGroup",
    "PBPKParameter",
    "BodyCompartment",
    "ToxicityRisk",
    "PhysiologyCondition",
)

REQUIRED_RELATIONSHIPS: Tuple[str, ...] = (
    "CONTAINS",
    "BELONGS_TO",
    "METABOLIZED_BY",
    "MODIFIES",
    "AFFECTS",
    "CONTRIBUTES_TO",
    "INCREASES",
    "DECREASES",
)

REL_DIRECTION_SPEC: Mapping[str, Tuple[str, str]] = {
    "CONTAINS": ("Beverage", "Compound"),
    "BELONGS_TO": ("Compound", "ChemicalClass"),
    "METABOLIZED_BY": ("Compound", "Enzyme"),
    "MODIFIES": ("PopulationGroup", "PBPKParameter"),
    "AFFECTS": ("PBPKParameter", "BodyCompartment"),
    "CONTRIBUTES_TO": ("Compound", "ToxicityRisk"),
    "INCREASES": ("PhysiologyCondition", "PBPKParameter"),
    "DECREASES": ("PhysiologyCondition", "PBPKParameter"),
}

EXPLAINABILITY_TESTS: Mapping[str, str] = {
    "test_1_beverage_compound_enzyme": (
        "MATCH p=(b:Beverage)-[:CONTAINS]->(:Compound)-[:METABOLIZED_BY]->(:Enzyme) "
        "RETURN count(p) AS path_count"
    ),
    "test_2_population_parameter_compartment": (
        "MATCH p=(g:PopulationGroup)-[:MODIFIES]->(:PBPKParameter)-[:AFFECTS]->(:BodyCompartment) "
        "RETURN count(p) AS path_count"
    ),
    "test_3_beverage_compound_toxicity": (
        "MATCH p=(b:Beverage)-[:CONTAINS]->(:Compound)-[:CONTRIBUTES_TO]->(:ToxicityRisk) "
        "RETURN count(p) AS path_count"
    ),
    "test_4_physiology_to_parameter": (
        "MATCH p=(pc:PhysiologyCondition)-[:INCREASES|DECREASES]->(:PBPKParameter) "
        "RETURN count(p) AS path_count"
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def report_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "neo4j" / "neo4j_graph_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def queries_output_path(root: Path) -> Path:
    path = root / "rag" / "neo4j" / "example_explainability_queries.cypher"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def redact_env(config: Mapping[str, str]) -> Dict[str, str]:
    return {
        "uri": config.get("uri", ""),
        "user": config.get("user", ""),
        "database": config.get("database", ""),
        "password_provided": "yes" if bool(config.get("password")) else "no",
    }


def write_example_queries(path: Path) -> None:
    content = """// Neo4j Explainability Queries (deterministic, read-only)

// 1) Why would whisky hit faster?
// Compares fasted vs fed modifiers and affected absorption/emptying parameters.
MATCH (g:PopulationGroup)-[m:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(bc:BodyCompartment)
WHERE g.group_name IN ['fasted', 'fed']
  AND p.parameter_name IN ['gastric_emptying_rate', 'intestinal_absorption_rate']
RETURN g.group_name AS group_name,
       p.parameter_name AS parameter_name,
       bc.name AS compartment,
       m.modifier AS modifier,
       m.source_dataset AS source_dataset,
       m.confidence_score AS confidence_score
ORDER BY p.parameter_name, group_name;

// 2) Why would fed state reduce BAC?
MATCH (pc:PhysiologyCondition)-[r:DECREASES]->(p:PBPKParameter)-[:AFFECTS]->(bc:BodyCompartment)
WHERE toLower(pc.condition) CONTAINS 'food' OR toLower(pc.condition) CONTAINS 'fed'
RETURN pc.condition AS condition,
       p.parameter_name AS parameter_name,
       bc.name AS compartment,
       r.source_file AS source_file,
       r.confidence_score AS confidence_score
ORDER BY pc.condition, p.parameter_name;

// 3) Which compounds contribute to hangover risk?
MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:CONTRIBUTES_TO]->(t:ToxicityRisk)
WHERE t.risk_type = 'hangover_amplification_modifier'
RETURN b.name AS beverage,
       c.name AS compound,
       t.risk_type AS risk_type,
       t.modifier AS modifier,
       t.source_compound_class AS source_compound_class,
       t.confidence_score AS confidence_score
ORDER BY beverage, compound;

// 4) Which enzymes metabolize beverage compounds?
MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:METABOLIZED_BY]->(e:Enzyme)
RETURN b.name AS beverage,
       c.name AS compound,
       e.name AS enzyme,
       e.family AS enzyme_family
ORDER BY beverage, compound, enzyme;
"""
    path.write_text(content, encoding=ENCODING)


def node_counts(session) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in REQUIRED_NODE_LABELS:
        record = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
        counts[label] = int(record["c"]) if record else 0
    return counts


def relationship_counts(session) -> Dict[str, int]:
    query = "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY t"
    rows = session.run(query)
    counts = {str(row["t"]): int(row["c"]) for row in rows}
    return counts


def orphan_counts(session) -> Dict[str, int]:
    queries = {
        "Beverage": "MATCH (n:Beverage) WHERE NOT (n)-[:CONTAINS]->(:Compound) RETURN count(n) AS c",
        "Compound": (
            "MATCH (n:Compound) "
            "WHERE NOT (:Beverage)-[:CONTAINS]->(n) "
            "   OR NOT (n)-[:BELONGS_TO]->(:ChemicalClass) "
            "RETURN count(n) AS c"
        ),
        "ChemicalClass": "MATCH (n:ChemicalClass) WHERE NOT (:Compound)-[:BELONGS_TO]->(n) RETURN count(n) AS c",
        "Enzyme": "MATCH (n:Enzyme) WHERE NOT (:Compound)-[:METABOLIZED_BY]->(n) RETURN count(n) AS c",
        "PopulationGroup": "MATCH (n:PopulationGroup) WHERE NOT (n)-[:MODIFIES]->(:PBPKParameter) RETURN count(n) AS c",
        "PBPKParameter": (
            "MATCH (n:PBPKParameter) "
            "WHERE NOT (:PopulationGroup)-[:MODIFIES]->(n) "
            "   OR NOT (n)-[:AFFECTS]->(:BodyCompartment) "
            "RETURN count(n) AS c"
        ),
        "BodyCompartment": "MATCH (n:BodyCompartment) WHERE NOT (:PBPKParameter)-[:AFFECTS]->(n) RETURN count(n) AS c",
        "ToxicityRisk": "MATCH (n:ToxicityRisk) WHERE NOT (:Compound)-[:CONTRIBUTES_TO]->(n) RETURN count(n) AS c",
        "PhysiologyCondition": (
            "MATCH (n:PhysiologyCondition) "
            "WHERE NOT (n)-[:INCREASES|DECREASES]->(:PBPKParameter) "
            "RETURN count(n) AS c"
        ),
    }
    counts: Dict[str, int] = {}
    for label, query in queries.items():
        record = session.run(query).single()
        counts[label] = int(record["c"]) if record else 0
    return counts


def invalid_directionality(session) -> Dict[str, int]:
    invalid: Dict[str, int] = {}
    for rel_type, (from_label, to_label) in REL_DIRECTION_SPEC.items():
        query = (
            f"MATCH (a)-[r:{rel_type}]->(b) "
            f"WHERE NOT (a:{from_label} AND b:{to_label}) "
            "RETURN count(r) AS c"
        )
        record = session.run(query).single()
        invalid[rel_type] = int(record["c"]) if record else 0
    return invalid


def duplicate_semantic_paths(session) -> Dict[str, Dict[str, int]]:
    duplicates: Dict[str, Dict[str, int]] = {}
    for rel_type in REQUIRED_RELATIONSHIPS:
        query = (
            f"MATCH (a)-[r:{rel_type}]->(b) "
            "WITH elementId(a) AS a_id, elementId(b) AS b_id, count(r) AS rel_count "
            "WHERE rel_count > 1 "
            "RETURN count(*) AS duplicate_pairs, coalesce(sum(rel_count - 1), 0) AS duplicate_relationships"
        )
        record = session.run(query).single()
        duplicates[rel_type] = {
            "duplicate_pairs": int(record["duplicate_pairs"]) if record else 0,
            "duplicate_relationships": int(record["duplicate_relationships"]) if record else 0,
        }
    return duplicates


def fetch_subgraph_for_components(session) -> Tuple[Set[str], List[Tuple[str, str]]]:
    node_query = (
        "MATCH (n) "
        "WHERE any(l IN labels(n) WHERE l IN $labels) "
        "RETURN elementId(n) AS id"
    )
    rel_query = (
        "MATCH (a)-[r]->(b) "
        "WHERE any(l IN labels(a) WHERE l IN $labels) "
        "  AND any(l IN labels(b) WHERE l IN $labels) "
        "  AND type(r) IN $types "
        "RETURN elementId(a) AS source, elementId(b) AS target"
    )
    nodes = {str(row["id"]) for row in session.run(node_query, labels=list(REQUIRED_NODE_LABELS))}
    edges = [
        (str(row["source"]), str(row["target"]))
        for row in session.run(
            rel_query,
            labels=list(REQUIRED_NODE_LABELS),
            types=list(REQUIRED_RELATIONSHIPS),
        )
    ]
    return nodes, edges


def connected_components(nodes: Set[str], edges: Sequence[Tuple[str, str]]) -> Tuple[int, int]:
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    visited: Set[str] = set()
    component_count = 0
    orphan_count = 0
    for node in sorted(nodes):
        if node in visited:
            continue
        component_count += 1
        stack = [node]
        comp_nodes: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            comp_nodes.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(comp_nodes) == 1 and len(adjacency[comp_nodes[0]]) == 0:
            orphan_count += 1
    return component_count, orphan_count


def explainability_tests(session) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, query in EXPLAINABILITY_TESTS.items():
        record = session.run(query).single()
        path_count = int(record["path_count"]) if record else 0
        results[name] = {
            "path_count": path_count,
            "passes": path_count > 0,
            "query": query,
        }
    return results


def build_failed_report(config: Mapping[str, str], missing_env: Sequence[str], error: str) -> Dict[str, Any]:
    return {
        "status": "failed",
        "neo4j_env": redact_env(config),
        "missing_environment_variables": list(missing_env),
        "error": error,
        "graph_integrity": {},
        "explainability_path_tests": {},
        "graph_quality_metrics": {
            "path_coverage": 0.0,
            "reasoning_readiness_score": 0.0,
            "orphan_rate": 1.0,
            "connected_component_count": 0,
        },
        "safe_for_weaviate_ingestion": False,
        "reasoning": [
            "Graph validation did not run successfully.",
            "Resolve Neo4j driver/environment prerequisites and rerun.",
        ],
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = report_output_path(root)
    query_path = queries_output_path(root)
    write_example_queries(query_path)

    config: Dict[str, str] = {"uri": "", "user": "", "password": "", "database": "neo4j"}
    missing_env: List[str] = []
    try:
        config = get_neo4j_config()
    except ValueError as exc:
        report = build_failed_report(config, [], str(exc))
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j graph validation report -> %s", report_path)
        LOGGER.info("safe_for_weaviate_ingestion=%s", report["safe_for_weaviate_ingestion"])
        return

    if GraphDatabase is None:
        report = build_failed_report(config, [], "neo4j Python driver is not installed in this environment.")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j graph validation report -> %s", report_path)
        LOGGER.info("safe_for_weaviate_ingestion=%s", report["safe_for_weaviate_ingestion"])
        return

    try:
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            nodes_by_label = node_counts(session)
            rel_counts = relationship_counts(session)
            orphan_by_label = orphan_counts(session)
            invalid_dir = invalid_directionality(session)
            duplicate_paths = duplicate_semantic_paths(session)
            nodes, edges = fetch_subgraph_for_components(session)
            component_count, zero_degree_components = connected_components(nodes, edges)
            path_tests = explainability_tests(session)
        driver.close()

        total_nodes = sum(nodes_by_label.values())
        orphan_total = sum(orphan_by_label.values())
        orphan_rate = float(orphan_total / total_nodes) if total_nodes > 0 else 1.0
        missing_relationship_types = sorted(
            [rel for rel in REQUIRED_RELATIONSHIPS if rel_counts.get(rel, 0) == 0]
        )
        invalid_directionality_total = int(sum(invalid_dir.values()))
        duplicate_relationship_total = int(
            sum(item["duplicate_relationships"] for item in duplicate_paths.values())
        )
        passed_path_tests = sum(1 for result in path_tests.values() if result["passes"])
        path_coverage = passed_path_tests / float(len(EXPLAINABILITY_TESTS)) if EXPLAINABILITY_TESTS else 0.0
        directionality_score = 1.0 if invalid_directionality_total == 0 else 0.0
        reasoning_readiness_score = round(
            (0.6 * path_coverage) + (0.2 * max(0.0, 1.0 - orphan_rate)) + (0.2 * directionality_score),
            4,
        )

        safe_for_weaviate_ingestion = (
            len(missing_relationship_types) == 0
            and invalid_directionality_total == 0
            and duplicate_relationship_total == 0
            and path_coverage == 1.0
            and orphan_rate <= 0.25
            and component_count >= 1
        )

        report: Dict[str, Any] = {
            "status": "success",
            "neo4j_env": redact_env(config),
            "missing_environment_variables": [],
            "graph_integrity": {
                "node_counts": nodes_by_label,
                "relationship_counts": rel_counts,
                "missing_required_relationship_types": missing_relationship_types,
                "orphan_nodes_by_label": orphan_by_label,
                "orphan_nodes_total": orphan_total,
                "disconnected_subgraphs_detected": component_count > 1,
                "duplicate_semantic_paths": duplicate_paths,
                "invalid_relationship_directionality": invalid_dir,
                "invalid_directionality_total": invalid_directionality_total,
            },
            "explainability_path_tests": path_tests,
            "graph_quality_metrics": {
                "path_coverage": round(path_coverage, 4),
                "reasoning_readiness_score": reasoning_readiness_score,
                "orphan_rate": round(orphan_rate, 6),
                "connected_component_count": int(component_count),
                "zero_degree_component_count": int(zero_degree_components),
            },
            "safe_for_weaviate_ingestion": safe_for_weaviate_ingestion,
            "reasoning": [
                f"Required relationship types missing: {len(missing_relationship_types)}.",
                f"Invalid directionality relationships: {invalid_directionality_total}.",
                f"Duplicate semantic relationships: {duplicate_relationship_total}.",
                f"Explainability path coverage: {round(path_coverage, 4)}.",
                f"Orphan rate: {round(orphan_rate, 6)}.",
            ],
            "artifacts": {
                "example_explainability_queries": str(query_path.relative_to(root)),
            },
        }
    except Exception as exc:  # pragma: no cover - runtime branch
        report = build_failed_report(config, [], str(exc))

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Neo4j graph validation report -> %s", report_path)
    LOGGER.info("safe_for_weaviate_ingestion=%s", report["safe_for_weaviate_ingestion"])


if __name__ == "__main__":
    main()
