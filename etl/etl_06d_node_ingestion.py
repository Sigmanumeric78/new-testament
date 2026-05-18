"""ETL step 06d: ingest Neo4j ontology nodes only (no relationships)."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency availability branch
    GraphDatabase = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import get_neo4j_config

LOGGER = logging.getLogger("etl_06d_node_ingestion")

ENCODING = "utf-8"
UNKNOWN = "unknown"

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

BODY_COMPARTMENTS: Tuple[str, ...] = (
    "stomach",
    "gut",
    "blood",
    "liver",
    "brain",
    "muscle",
    "fat",
    "elimination",
)

ENZYME_MAP: Mapping[str, Tuple[str, str]] = {
    "adh_activity": ("ENZ_ADH", "ADH"),
    "stomach_adh_activity": ("ENZ_ADH", "ADH"),
    "aldh_activity": ("ENZ_ALDH", "ALDH"),
    "cyp2e1_activity": ("ENZ_CYP2E1", "CYP2E1"),
    "catalase_activity": ("ENZ_CATALASE", "Catalase"),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "neo4j" / "neo4j_node_ingestion_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def input_paths(root: Path) -> Dict[str, Path]:
    return {
        "master_beverage_reference_repaired": root
        / "data"
        / "processed"
        / "beverage"
        / "reference_tables"
        / "master_beverage_reference_repaired.csv",
        "beverage_compound_matrix_expanded": root
        / "data"
        / "processed"
        / "beverage"
        / "compound_profiles"
        / "beverage_compound_matrix_expanded.csv",
        "human_metabolism_parameters": root / "data" / "processed" / "human" / "human_metabolism_parameters.csv",
        "pbpk_parameter_library": root / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv",
        "population_modifiers": root / "data" / "processed" / "pbpk" / "population_modifiers.csv",
        "beverage_effect_modifiers": root / "data" / "processed" / "pbpk" / "beverage_effect_modifiers.csv",
    }


def redact_config(config: Mapping[str, str]) -> Dict[str, str]:
    return {
        "uri": config.get("uri", ""),
        "user": config.get("user", ""),
        "database": config.get("database", ""),
        "password_provided": "yes" if bool(config.get("password")) else "no",
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return ""
    return text


def normalize_token(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=ENCODING)
    for column in df.columns:
        df[column] = df[column].map(clean_text)
    return df


def merge_props(base: MutableMapping[str, str], new_props: Mapping[str, str]) -> None:
    for key, value in new_props.items():
        if key not in base or not clean_text(base.get(key, "")):
            base[key] = clean_text(value)


def merge_record(
    records: MutableMapping[str, Dict[str, Any]],
    key: str,
    props: Mapping[str, Any],
) -> None:
    if not key:
        return
    normalized_props = {field: clean_text(value) for field, value in props.items()}
    if key not in records:
        records[key] = dict(normalized_props)
        return
    merge_props(records[key], normalized_props)


def build_beverage_nodes(master_df: pd.DataFrame, matrix_df: pd.DataFrame, effects_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for _, row in master_df.iterrows():
        beverage_id = clean_text(row.get("beverage_id"))
        props = {
            "beverage_id": beverage_id,
            "name": clean_text(row.get("beverage_name")),
            "normalized_name": clean_text(row.get("normalized_name")),
            "category": clean_text(row.get("category")),
            "subcategory": clean_text(row.get("subcategory")),
            "baseline_abv": clean_text(row.get("baseline_abv")),
            "country_origin": clean_text(row.get("country_origin")),
            "carbonation": clean_text(row.get("carbonation")),
            "sugar_g_per_100ml": clean_text(row.get("sugar_g_per_100ml")),
            "source_dataset": clean_text(row.get("source_dataset")),
            "source_file": clean_text(row.get("source_file")),
            "source_row": clean_text(row.get("source_row")),
            "confidence_score": clean_text(row.get("confidence_score")),
            "provenance": "master_beverage_reference_repaired",
        }
        merge_record(records, beverage_id, props)

    for _, row in matrix_df.iterrows():
        beverage_id = clean_text(row.get("beverage_id"))
        props = {
            "beverage_id": beverage_id,
            "name": clean_text(row.get("beverage_name")),
            "category": clean_text(row.get("category")),
            "provenance": "beverage_compound_matrix_expanded",
        }
        merge_record(records, beverage_id, props)

    for _, row in effects_df.iterrows():
        beverage_id = clean_text(row.get("beverage_id"))
        props = {
            "beverage_id": beverage_id,
            "name": clean_text(row.get("beverage_name")),
            "category": clean_text(row.get("category")),
            "provenance": "beverage_effect_modifiers",
        }
        merge_record(records, beverage_id, props)

    return [records[key] for key in sorted(records)]


def build_compound_nodes(matrix_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for _, row in matrix_df.iterrows():
        normalized_name = normalize_token(row.get("normalized_compound_name"))
        cid = clean_text(row.get("pubchem_cid"))
        if cid:
            compound_id = f"CID_{cid}"
        else:
            compound_id = f"CMP_{normalized_name}" if normalized_name else ""
        props = {
            "compound_id": compound_id,
            "name": clean_text(row.get("compound_name")),
            "normalized_name": clean_text(row.get("normalized_compound_name")),
            "pubchem_cid": cid or UNKNOWN,
            "chemical_category": clean_text(row.get("chemical_category")),
            "compound_role": clean_text(row.get("compound_role")),
            "estimated_concentration": clean_text(row.get("estimated_concentration")),
            "concentration_unit": clean_text(row.get("concentration_unit")),
            "digestion_effect": clean_text(row.get("digestion_effect")),
            "metabolic_burden": clean_text(row.get("metabolic_burden")),
            "confidence_score": clean_text(row.get("confidence_score")),
            "source_dataset": clean_text(row.get("source_dataset")),
            "source_file": clean_text(row.get("source_file")),
            "source_row": clean_text(row.get("source_row")),
            "provenance": "beverage_compound_matrix_expanded",
        }
        merge_record(records, compound_id, props)
    return [records[key] for key in sorted(records)]


def build_chemical_class_nodes(matrix_df: pd.DataFrame, effects_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for _, row in matrix_df.iterrows():
        class_name = clean_text(row.get("source_compound_class"))
        class_key = normalize_token(class_name)
        props = {
            "class_key": class_key,
            "class_name": class_name or UNKNOWN,
            "expansion_type": clean_text(row.get("expansion_type")),
            "provenance": "beverage_compound_matrix_expanded",
        }
        merge_record(records, class_key, props)
    for _, row in effects_df.iterrows():
        class_name = clean_text(row.get("source_compound_class"))
        class_key = normalize_token(class_name)
        props = {
            "class_key": class_key,
            "class_name": class_name or UNKNOWN,
            "provenance": "beverage_effect_modifiers",
        }
        merge_record(records, class_key, props)
    return [records[key] for key in sorted(records)]


def build_enzyme_nodes(human_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for _, row in human_df.iterrows():
        parameter_name = clean_text(row.get("parameter_name"))
        if parameter_name not in ENZYME_MAP:
            continue
        enzyme_id, enzyme_name = ENZYME_MAP[parameter_name]
        props = {
            "enzyme_id": enzyme_id,
            "name": enzyme_name,
            "family": "oxidative_metabolism",
            "source_parameter_names": parameter_name,
            "source_parameter_ids": clean_text(row.get("parameter_id")),
            "source_document": clean_text(row.get("source_document")),
            "provenance": "human_metabolism_parameters",
        }
        merge_record(records, enzyme_id, props)
    return [records[key] for key in sorted(records)]


def build_population_group_nodes(pop_df: pd.DataFrame, human_df: pd.DataFrame) -> List[Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for group_name in sorted(set(pop_df["population_group"].tolist()) | set(human_df["population_group"].tolist())):
        group_name = clean_text(group_name)
        if not group_name:
            continue
        props = {
            "group_name": group_name,
            "name": group_name,
            "provenance": "population_modifiers|human_metabolism_parameters",
        }
        groups[group_name] = props
    return [groups[key] for key in sorted(groups)]


def build_pbpk_parameter_nodes(pbpk_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for _, row in pbpk_df.iterrows():
        records.append(
            {
                "parameter_id": clean_text(row.get("parameter_id")),
                "parameter_name": clean_text(row.get("parameter_name")),
                "compartment": clean_text(row.get("compartment")),
                "base_value": clean_text(row.get("base_value")),
                "unit": clean_text(row.get("unit")),
                "population_group": clean_text(row.get("population_group")),
                "modifier": clean_text(row.get("modifier")),
                "modifier_reason": clean_text(row.get("modifier_reason")),
                "confidence_score": clean_text(row.get("confidence_score")),
                "source_document": clean_text(row.get("source_document")),
                "source_parameter_id": clean_text(row.get("source_parameter_id")),
                "provenance": "pbpk_parameter_library",
            }
        )
    records = sorted(records, key=lambda item: item.get("parameter_id", ""))
    return records


def build_body_compartment_nodes(pbpk_df: pd.DataFrame) -> List[Dict[str, str]]:
    discovered = {normalize_token(value) for value in pbpk_df["compartment"].tolist() if clean_text(value)}
    merged = sorted(set(BODY_COMPARTMENTS) | discovered)
    return [
        {
            "compartment_key": key,
            "name": key,
            "provenance": "pbpk_parameter_library|frozen_schema",
        }
        for key in merged
    ]


def build_toxicity_risk_nodes(effects_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for _, row in effects_df.iterrows():
        records.append(
            {
                "risk_id": clean_text(row.get("modifier_id")),
                "risk_type": clean_text(row.get("parameter_name")),
                "modifier": clean_text(row.get("modifier")),
                "modifier_reason": clean_text(row.get("modifier_reason")),
                "trigger_compounds": clean_text(row.get("trigger_compounds")),
                "source_compound_class": clean_text(row.get("source_compound_class")),
                "confidence_score": clean_text(row.get("confidence_score")),
                "beverage_id": clean_text(row.get("beverage_id")),
                "beverage_name": clean_text(row.get("beverage_name")),
                "category": clean_text(row.get("category")),
                "provenance": "beverage_effect_modifiers",
            }
        )
    records = sorted(records, key=lambda item: item.get("risk_id", ""))
    return records


def build_physiology_condition_nodes(human_df: pd.DataFrame) -> List[Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for _, row in human_df.iterrows():
        domain = clean_text(row.get("domain"))
        condition = clean_text(row.get("condition"))
        modifier_type = clean_text(row.get("modifier_type"))
        condition_key = normalize_token(f"{domain}|{condition}|{modifier_type}")
        props = {
            "condition_key": condition_key,
            "condition": condition,
            "domain": domain,
            "modifier_type": modifier_type,
            "effect_direction": clean_text(row.get("effect_direction")),
            "evidence_text": clean_text(row.get("evidence_text")),
            "source_document": clean_text(row.get("source_document")),
            "source_page": clean_text(row.get("source_page")),
            "extract_method": clean_text(row.get("extract_method")),
            "confidence_score": clean_text(row.get("confidence_score")),
            "population_group": clean_text(row.get("population_group")),
            "source_parameter_id": clean_text(row.get("parameter_id")),
            "source_parameter_name": clean_text(row.get("parameter_name")),
            "provenance": "human_metabolism_parameters",
        }
        merge_record(records, condition_key, props)
    return [records[key] for key in sorted(records)]


def merge_query(label: str, key_field: str) -> str:
    return (
        f"MERGE (n:{label} {{{key_field}: $key_value}}) "
        "SET n += $props"
    )


def ingest_label_records(session, label: str, key_field: str, records: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    created_count = 0
    failed_rows = 0
    failures: List[Dict[str, str]] = []
    query = merge_query(label, key_field)
    for row in records:
        key_value = clean_text(row.get(key_field))
        if not key_value:
            failed_rows += 1
            failures.append({"key": "", "error": f"missing key_field '{key_field}'"})
            continue
        props = {key: clean_text(value) for key, value in row.items()}
        try:
            summary = session.run(query, key_value=key_value, props=props).consume()
            created_count += int(summary.counters.nodes_created)
        except Exception as exc:  # pragma: no cover - runtime branch
            failed_rows += 1
            failures.append({"key": key_value, "error": str(exc)})
    total_rows = len(records)
    matched_existing_count = total_rows - created_count - failed_rows
    return {
        "rows_attempted": total_rows,
        "created_count": created_count,
        "matched_existing_count": matched_existing_count,
        "failed_rows": failed_rows,
        "failures": failures[:100],
    }


def query_node_count(session, label: str) -> int:
    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS node_count")
    record = result.single()
    return int(record["node_count"]) if record is not None else 0


def build_failure_report(
    config: Mapping[str, str],
    missing_env: Sequence[str],
    error: str,
    inputs: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "neo4j_env": redact_config(config),
        "missing_environment_variables": list(missing_env),
        "error": error,
        "inputs": dict(inputs),
        "required_labels": list(REQUIRED_LABELS),
        "node_ingestion_metrics": {},
        "graph_validation": {},
        "safe_for_relationship_ingestion": False,
        "reasoning": [
            "Node ingestion did not complete successfully.",
            "Resolve environment/driver/connectivity prerequisites before retry.",
        ],
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = output_report_path(root)
    missing_env: List[str] = []
    paths = input_paths(root)
    input_refs = {name: str(path.relative_to(root)) for name, path in paths.items()}

    try:
        config = get_neo4j_config()
    except ValueError as exc:
        report = build_failure_report(
            config={"uri": "", "user": "", "password": "", "database": "neo4j"},
            missing_env=[],
            error=str(exc),
            inputs=input_refs,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j node ingestion report -> %s", report_path)
        LOGGER.info("safe_for_relationship_ingestion=%s", report["safe_for_relationship_ingestion"])
        return

    if GraphDatabase is None:
        report = build_failure_report(
            config=config,
            missing_env=[],
            error="neo4j Python driver is not installed in this environment.",
            inputs=input_refs,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j node ingestion report -> %s", report_path)
        LOGGER.info("safe_for_relationship_ingestion=%s", report["safe_for_relationship_ingestion"])
        return

    try:
        master_df = read_csv(paths["master_beverage_reference_repaired"])
        matrix_df = read_csv(paths["beverage_compound_matrix_expanded"])
        human_df = read_csv(paths["human_metabolism_parameters"])
        pbpk_df = read_csv(paths["pbpk_parameter_library"])
        pop_df = read_csv(paths["population_modifiers"])
        effects_df = read_csv(paths["beverage_effect_modifiers"])

        node_payloads: List[Tuple[str, str, List[Dict[str, str]]]] = [
            ("Beverage", "beverage_id", build_beverage_nodes(master_df, matrix_df, effects_df)),
            ("Compound", "compound_id", build_compound_nodes(matrix_df)),
            ("ChemicalClass", "class_key", build_chemical_class_nodes(matrix_df, effects_df)),
            ("Enzyme", "enzyme_id", build_enzyme_nodes(human_df)),
            ("PopulationGroup", "group_name", build_population_group_nodes(pop_df, human_df)),
            ("PBPKParameter", "parameter_id", build_pbpk_parameter_nodes(pbpk_df)),
            ("BodyCompartment", "compartment_key", build_body_compartment_nodes(pbpk_df)),
            ("ToxicityRisk", "risk_id", build_toxicity_risk_nodes(effects_df)),
            ("PhysiologyCondition", "condition_key", build_physiology_condition_nodes(human_df)),
        ]

        node_metrics: Dict[str, Any] = {}
        node_counts: Dict[str, int] = {}

        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            for label, key_field, records in node_payloads:
                metrics = ingest_label_records(session, label, key_field, records)
                node_metrics[label] = metrics
            for label in REQUIRED_LABELS:
                node_counts[label] = query_node_count(session, label)
        driver.close()

        failed_total = sum(int(metrics["failed_rows"]) for metrics in node_metrics.values())
        created_total = sum(int(metrics["created_count"]) for metrics in node_metrics.values())
        safe_for_relationship_ingestion = (failed_total == 0) and all(node_counts[label] > 0 for label in REQUIRED_LABELS)

        report: Dict[str, Any] = {
            "status": "success",
            "neo4j_env": redact_config(config),
            "missing_environment_variables": [],
            "inputs": input_refs,
            "required_labels": list(REQUIRED_LABELS),
            "node_ingestion_metrics": node_metrics,
            "graph_validation": {
                "node_counts": node_counts,
                "total_nodes_created_this_run": created_total,
                "total_failed_rows": failed_total,
            },
            "safe_for_relationship_ingestion": safe_for_relationship_ingestion,
            "reasoning": [
                f"Node ingestion failed rows: {failed_total}.",
                "All required labels have at least one node."
                if all(node_counts[label] > 0 for label in REQUIRED_LABELS)
                else "One or more required labels have zero nodes.",
                "MERGE-based node ingestion is idempotent and rerunnable.",
            ],
        }
    except Exception as exc:  # pragma: no cover - runtime branch
        report = build_failure_report(
            config=config,
            missing_env=[],
            error=str(exc),
            inputs=input_refs,
        )

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Neo4j node ingestion report -> %s", report_path)
    LOGGER.info("safe_for_relationship_ingestion=%s", report["safe_for_relationship_ingestion"])


if __name__ == "__main__":
    main()
