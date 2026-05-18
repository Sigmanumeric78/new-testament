"""ETL step 06c: initialize Neo4j schema for the frozen ontology."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency availability branch
    GraphDatabase = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import get_neo4j_config

LOGGER = logging.getLogger("etl_06c_graph_schema")

ENCODING = "utf-8"


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    label: str
    property_key: str


@dataclass(frozen=True)
class IndexSpec:
    name: str
    label: str
    property_key: str


REQUIRED_LABELS: Tuple[str, ...] = (
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

# Deterministic unique keys aligned to frozen ontology design.
CONSTRAINT_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec("uq_beverage_beverage_id", "Beverage", "beverage_id"),
    ConstraintSpec("uq_compound_compound_id", "Compound", "compound_id"),
    ConstraintSpec("uq_chemicalclass_class_key", "ChemicalClass", "class_key"),
    ConstraintSpec("uq_enzyme_enzyme_id", "Enzyme", "enzyme_id"),
    ConstraintSpec("uq_populationgroup_group_name", "PopulationGroup", "group_name"),
    ConstraintSpec("uq_pbpkparameter_parameter_id", "PBPKParameter", "parameter_id"),
    ConstraintSpec("uq_bodycompartment_compartment_key", "BodyCompartment", "compartment_key"),
    ConstraintSpec("uq_toxicityrisk_risk_id", "ToxicityRisk", "risk_id"),
    ConstraintSpec("uq_physiologycondition_condition_key", "PhysiologyCondition", "condition_key"),
)

INDEX_SPECS: Tuple[IndexSpec, ...] = (
    IndexSpec("idx_beverage_name", "Beverage", "name"),
    IndexSpec("idx_beverage_normalized_name", "Beverage", "normalized_name"),
    IndexSpec("idx_beverage_category", "Beverage", "category"),
    IndexSpec("idx_compound_name", "Compound", "name"),
    IndexSpec("idx_compound_normalized_name", "Compound", "normalized_name"),
    IndexSpec("idx_compound_chemical_category", "Compound", "chemical_category"),
    IndexSpec("idx_compound_pubchem_cid", "Compound", "pubchem_cid"),
    IndexSpec("idx_chemicalclass_name", "ChemicalClass", "class_name"),
    IndexSpec("idx_populationgroup_name", "PopulationGroup", "name"),
    IndexSpec("idx_pbpkparameter_name", "PBPKParameter", "parameter_name"),
    IndexSpec("idx_physiologycondition_domain", "PhysiologyCondition", "domain"),
    IndexSpec("idx_toxicityrisk_type", "ToxicityRisk", "risk_type"),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "neo4j" / "neo4j_schema_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def redact_config(config: Mapping[str, str]) -> Dict[str, str]:
    return {
        "uri": config.get("uri", ""),
        "user": config.get("user", ""),
        "database": config.get("database", ""),
        "password_provided": "yes" if bool(config.get("password")) else "no",
    }


def run_query(session, query: str) -> None:
    session.run(query).consume()


def show_constraints(session) -> List[Dict[str, Any]]:
    query = (
        "SHOW CONSTRAINTS "
        "YIELD name, type, entityType, labelsOrTypes, properties "
        "RETURN name, type, entityType, labelsOrTypes, properties "
        "ORDER BY name"
    )
    records = session.run(query).data()
    return records


def show_indexes(session) -> List[Dict[str, Any]]:
    query = (
        "SHOW INDEXES "
        "YIELD name, type, entityType, labelsOrTypes, properties, state "
        "RETURN name, type, entityType, labelsOrTypes, properties, state "
        "ORDER BY name"
    )
    records = session.run(query).data()
    return records


def create_constraint_query(spec: ConstraintSpec) -> str:
    return (
        f"CREATE CONSTRAINT {spec.name} IF NOT EXISTS "
        f"FOR (n:{spec.label}) REQUIRE n.{spec.property_key} IS UNIQUE"
    )


def create_index_query(spec: IndexSpec) -> str:
    return f"CREATE INDEX {spec.name} IF NOT EXISTS FOR (n:{spec.label}) ON (n.{spec.property_key})"


def constraint_signature(record: Mapping[str, Any]) -> str:
    return str(record.get("name", ""))


def index_signature(record: Mapping[str, Any]) -> str:
    return str(record.get("name", ""))


def build_failed_report(
    root: Path,
    config: Mapping[str, str],
    error: str,
    missing_env: Sequence[str],
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error": error,
        "neo4j_env": redact_config(config),
        "missing_environment_variables": list(missing_env),
        "required_labels": list(REQUIRED_LABELS),
        "constraint_specs": [spec.__dict__ for spec in CONSTRAINT_SPECS],
        "index_specs": [spec.__dict__ for spec in INDEX_SPECS],
        "existing_constraints": [],
        "existing_indexes": [],
        "created_constraints": [],
        "created_indexes": [],
        "safe_for_node_ingestion": False,
        "reasoning": [
            "Neo4j schema initialization did not complete.",
            "Connection or configuration prerequisites were not satisfied.",
        ],
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = output_report_path(root)
    try:
        config = get_neo4j_config()
    except ValueError as exc:
        report = build_failed_report(
            root=root,
            config={"uri": "", "user": "", "password": "", "database": "neo4j"},
            error=str(exc),
            missing_env=[],
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j schema report -> %s", report_path)
        LOGGER.info("safe_for_node_ingestion=%s", report["safe_for_node_ingestion"])
        return

    if GraphDatabase is None:
        report = build_failed_report(
            root=root,
            config=config,
            error="neo4j Python driver is not installed in this environment.",
            missing_env=[],
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j schema report -> %s", report_path)
        LOGGER.info("safe_for_node_ingestion=%s", report["safe_for_node_ingestion"])
        return

    try:
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            existing_constraints_before = show_constraints(session)
            existing_indexes_before = show_indexes(session)

            for spec in CONSTRAINT_SPECS:
                run_query(session, create_constraint_query(spec))
            for spec in INDEX_SPECS:
                run_query(session, create_index_query(spec))

            existing_constraints_after = show_constraints(session)
            existing_indexes_after = show_indexes(session)
        driver.close()

        before_constraint_names = {constraint_signature(item) for item in existing_constraints_before}
        after_constraint_names = {constraint_signature(item) for item in existing_constraints_after}
        created_constraint_names = sorted(after_constraint_names - before_constraint_names)

        before_index_names = {index_signature(item) for item in existing_indexes_before}
        after_index_names = {index_signature(item) for item in existing_indexes_after}
        created_index_names = sorted(after_index_names - before_index_names)

        required_constraint_names = {spec.name for spec in CONSTRAINT_SPECS}
        required_index_names = {spec.name for spec in INDEX_SPECS}
        constraints_ok = required_constraint_names.issubset(after_constraint_names)
        indexes_ok = required_index_names.issubset(after_index_names)
        safe_for_node_ingestion = constraints_ok and indexes_ok

        report: Dict[str, Any] = {
            "status": "success",
            "neo4j_env": redact_config(config),
            "missing_environment_variables": [],
            "required_labels": list(REQUIRED_LABELS),
            "constraint_specs": [spec.__dict__ for spec in CONSTRAINT_SPECS],
            "index_specs": [spec.__dict__ for spec in INDEX_SPECS],
            "existing_constraints_before": existing_constraints_before,
            "existing_indexes_before": existing_indexes_before,
            "existing_constraints": existing_constraints_after,
            "existing_indexes": existing_indexes_after,
            "created_constraints": created_constraint_names,
            "created_indexes": created_index_names,
            "schema_validation": {
                "required_constraint_names": sorted(required_constraint_names),
                "required_index_names": sorted(required_index_names),
                "constraints_present": constraints_ok,
                "indexes_present": indexes_ok,
            },
            "safe_for_node_ingestion": safe_for_node_ingestion,
            "reasoning": [
                f"Constraints present after initialization: {constraints_ok}.",
                f"Indexes present after initialization: {indexes_ok}.",
                "Schema creation is idempotent via IF NOT EXISTS.",
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive runtime branch
        report = build_failed_report(
            root=root,
            config=config,
            error=str(exc),
            missing_env=[],
        )

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Neo4j schema report -> %s", report_path)
    LOGGER.info("safe_for_node_ingestion=%s", report["safe_for_node_ingestion"])


if __name__ == "__main__":
    main()
