"""ETL step 06e: ingest frozen-ontology Neo4j relationships (no APOC)."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

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

LOGGER = logging.getLogger("etl_06e_relationship_ingestion")

ENCODING = "utf-8"
UNKNOWN = "unknown"

RELATIONSHIP_TYPES: Tuple[str, ...] = (
    "CONTAINS",
    "BELONGS_TO",
    "METABOLIZED_BY",
    "MODIFIES",
    "AFFECTS",
    "CONTRIBUTES_TO",
    "INCREASES",
    "DECREASES",
)

SOURCE_TO_PBPK_PARAMETER: Mapping[str, Tuple[str, ...]] = {
    "gastric_emptying_rate_constant": ("gastric_emptying_rate",),
    "food_effect_on_gastric_emptying": ("gastric_emptying_rate",),
    "meal_composition_effect_on_gastric_emptying": ("gastric_emptying_rate",),
    "alcohol_absorption_rate": ("intestinal_absorption_rate",),
    "food_effect_on_alcohol_absorption": ("intestinal_absorption_rate",),
    "gastric_emptying_absorption_link": ("intestinal_absorption_rate",),
    "ethanol_elimination_rate": ("ethanol_elimination_rate",),
    "food_effect_on_ethanol_elimination_rate": ("ethanol_elimination_rate",),
    "ethanol_elimination_variability": ("ethanol_elimination_rate",),
    "volume_of_distribution": ("ethanol_distribution_volume", "fat_partition_coefficient"),
    "peak_bac_same_dose": ("ethanol_distribution_volume", "blood_brain_partition"),
    "body_water_normalized_bac_difference": ("body_water_fraction",),
    "total_body_water_volume": ("body_water_fraction",),
    "adh_activity": ("adh_metabolism_rate",),
    "stomach_adh_activity": ("adh_metabolism_rate",),
    "aldh_activity": ("aldh_metabolism_rate",),
    "cyp2e1_activity": ("cyp2e1_modifier",),
    "first_pass_metabolism": ("first_pass_metabolism",),
    "drug_effect_on_first_pass_metabolism": ("first_pass_metabolism", "intestinal_absorption_rate"),
    "gastric_emptying_first_pass_link": ("first_pass_metabolism",),
    "liver_blood_flow": ("liver_blood_flow",),
    "liver_function_role": ("liver_blood_flow",),
    "liver_function_modifier": ("liver_blood_flow",),
}

COMPOUND_ENZYME_MAP: Mapping[str, Tuple[str, ...]] = {
    "ethanol": ("ENZ_ADH", "ENZ_CYP2E1", "ENZ_CATALASE"),
    "acetaldehyde": ("ENZ_ALDH",),
}


@dataclass(frozen=True)
class RelationshipSpec:
    rel_type: str
    from_label: str
    from_key: str
    to_label: str
    to_key: str


REL_SPECS: Mapping[str, RelationshipSpec] = {
    "CONTAINS": RelationshipSpec("CONTAINS", "Beverage", "beverage_id", "Compound", "compound_id"),
    "BELONGS_TO": RelationshipSpec("BELONGS_TO", "Compound", "compound_id", "ChemicalClass", "class_key"),
    "METABOLIZED_BY": RelationshipSpec("METABOLIZED_BY", "Compound", "compound_id", "Enzyme", "enzyme_id"),
    "MODIFIES": RelationshipSpec("MODIFIES", "PopulationGroup", "group_name", "PBPKParameter", "parameter_id"),
    "AFFECTS": RelationshipSpec("AFFECTS", "PBPKParameter", "parameter_id", "BodyCompartment", "compartment_key"),
    "CONTRIBUTES_TO": RelationshipSpec("CONTRIBUTES_TO", "Compound", "compound_id", "ToxicityRisk", "risk_id"),
    "INCREASES": RelationshipSpec("INCREASES", "PhysiologyCondition", "condition_key", "PBPKParameter", "parameter_id"),
    "DECREASES": RelationshipSpec("DECREASES", "PhysiologyCondition", "condition_key", "PBPKParameter", "parameter_id"),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "neo4j" / "neo4j_relationship_ingestion_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def input_paths(root: Path) -> Dict[str, Path]:
    return {
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


def compound_id_for_row(row: Mapping[str, Any]) -> str:
    cid = clean_text(row.get("pubchem_cid"))
    normalized = normalize_token(row.get("normalized_compound_name"))
    if cid:
        return f"CID_{cid}"
    if normalized:
        return f"CMP_{normalized}"
    return ""


def aggregate_record(
    container: MutableMapping[Tuple[str, str], Dict[str, Any]],
    from_key: str,
    to_key: str,
    source_dataset: str,
    source_file: str,
    confidence_score: str,
) -> None:
    if not from_key or not to_key:
        return
    key = (from_key, to_key)
    if key not in container:
        container[key] = {
            "from_key": from_key,
            "to_key": to_key,
            "source_dataset_set": set(),
            "source_file_set": set(),
            "confidence_score_set": set(),
        }
    if source_dataset:
        container[key]["source_dataset_set"].add(source_dataset)
    if source_file:
        container[key]["source_file_set"].add(source_file)
    if confidence_score:
        container[key]["confidence_score_set"].add(confidence_score)


def finalize_aggregated(container: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, value in sorted(container.items(), key=lambda item: item[0]):
        rows.append(
            {
                "from_key": value["from_key"],
                "to_key": value["to_key"],
                "props": {
                    "source_dataset": "|".join(sorted(value["source_dataset_set"])) or UNKNOWN,
                    "source_file": "|".join(sorted(value["source_file_set"])) or UNKNOWN,
                    "confidence_score": "|".join(sorted(value["confidence_score_set"])) or UNKNOWN,
                },
            }
        )
    return rows


def build_relationship_payloads(
    matrix_df: pd.DataFrame,
    human_df: pd.DataFrame,
    pbpk_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    effects_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    payloads: Dict[str, List[Dict[str, Any]]] = {}

    parameter_id_by_name: Dict[str, str] = {}
    for _, row in pbpk_df.iterrows():
        name = clean_text(row.get("parameter_name"))
        parameter_id = clean_text(row.get("parameter_id"))
        if name and parameter_id:
            parameter_id_by_name[name] = parameter_id

    compound_id_by_normalized: Dict[str, str] = {}
    for _, row in matrix_df.iterrows():
        normalized = normalize_token(row.get("normalized_compound_name"))
        compound_id = compound_id_for_row(row)
        if normalized and compound_id and normalized not in compound_id_by_normalized:
            compound_id_by_normalized[normalized] = compound_id

    # Beverage CONTAINS Compound
    contains: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Compound BELONGS_TO ChemicalClass
    belongs_to: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Compound METABOLIZED_BY Enzyme
    metabolized_by: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # PopulationGroup MODIFIES PBPKParameter
    modifies: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # PBPKParameter AFFECTS BodyCompartment
    affects: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Compound CONTRIBUTES_TO ToxicityRisk
    contributes_to: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # PhysiologyCondition INCREASES/DECREASES PBPKParameter
    increases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    decreases: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in matrix_df.iterrows():
        beverage_id = clean_text(row.get("beverage_id"))
        compound_id = compound_id_for_row(row)
        class_key = normalize_token(row.get("source_compound_class"))
        source_dataset = clean_text(row.get("source_dataset"))
        source_file = clean_text(row.get("source_file"))
        confidence = clean_text(row.get("confidence_score"))

        aggregate_record(contains, beverage_id, compound_id, source_dataset, source_file, confidence)
        aggregate_record(belongs_to, compound_id, class_key, source_dataset, source_file, confidence)

        normalized_compound = normalize_token(row.get("normalized_compound_name"))
        for enzyme_id in COMPOUND_ENZYME_MAP.get(normalized_compound, ()):
            aggregate_record(
                metabolized_by,
                compound_id,
                enzyme_id,
                "deterministic_enzyme_map",
                "etl/etl_06e_relationship_ingestion.py",
                "1.0",
            )

    for _, row in pop_df.iterrows():
        group_name = clean_text(row.get("population_group"))
        parameter_name = clean_text(row.get("parameter_name"))
        parameter_id = parameter_id_by_name.get(parameter_name, "")
        aggregate_record(
            modifies,
            group_name,
            parameter_id,
            "population_modifiers",
            clean_text(row.get("source_document")),
            clean_text(row.get("confidence_score")),
        )

    for _, row in pbpk_df.iterrows():
        parameter_id = clean_text(row.get("parameter_id"))
        compartment = normalize_token(row.get("compartment"))
        aggregate_record(
            affects,
            parameter_id,
            compartment,
            "pbpk_parameter_library",
            clean_text(row.get("source_document")),
            clean_text(row.get("confidence_score")),
        )

    for _, row in effects_df.iterrows():
        risk_id = clean_text(row.get("modifier_id"))
        source_dataset = "beverage_effect_modifiers"
        source_file = clean_text(row.get("source_compound_class"))
        confidence = clean_text(row.get("confidence_score"))

        trigger_compounds = clean_text(row.get("trigger_compounds"))
        triggers = [normalize_token(token) for token in trigger_compounds.split("|") if clean_text(token)]
        for token in triggers:
            compound_id = compound_id_by_normalized.get(token, "")
            aggregate_record(contributes_to, compound_id, risk_id, source_dataset, source_file, confidence)

    for _, row in human_df.iterrows():
        source_parameter_name = clean_text(row.get("parameter_name"))
        targets = SOURCE_TO_PBPK_PARAMETER.get(source_parameter_name, ())
        if not targets:
            continue
        domain = clean_text(row.get("domain"))
        condition = clean_text(row.get("condition"))
        modifier_type = clean_text(row.get("modifier_type"))
        condition_key = normalize_token(f"{domain}|{condition}|{modifier_type}")
        direction = clean_text(row.get("effect_direction")).lower()
        rel_bucket: Dict[Tuple[str, str], Dict[str, Any]]
        if direction.startswith("increase"):
            rel_bucket = increases
        elif direction.startswith("decrease"):
            rel_bucket = decreases
        else:
            continue
        for parameter_name in targets:
            parameter_id = parameter_id_by_name.get(parameter_name, "")
            aggregate_record(
                rel_bucket,
                condition_key,
                parameter_id,
                "human_metabolism_parameters",
                clean_text(row.get("source_document")),
                clean_text(row.get("confidence_score")),
            )

    payloads["CONTAINS"] = finalize_aggregated(contains)
    payloads["BELONGS_TO"] = finalize_aggregated(belongs_to)
    payloads["METABOLIZED_BY"] = finalize_aggregated(metabolized_by)
    payloads["MODIFIES"] = finalize_aggregated(modifies)
    payloads["AFFECTS"] = finalize_aggregated(affects)
    payloads["CONTRIBUTES_TO"] = finalize_aggregated(contributes_to)
    payloads["INCREASES"] = finalize_aggregated(increases)
    payloads["DECREASES"] = finalize_aggregated(decreases)
    return payloads


def key_set_query(label: str, key_field: str) -> str:
    return (
        f"MATCH (n:{label}) "
        f"WHERE n.{key_field} IS NOT NULL "
        f"RETURN n.{key_field} AS key_value"
    )


def relationship_merge_query(spec: RelationshipSpec) -> str:
    return (
        f"MATCH (a:{spec.from_label} {{{spec.from_key}: $from_key}}) "
        f"MATCH (b:{spec.to_label} {{{spec.to_key}: $to_key}}) "
        f"MERGE (a)-[r:{spec.rel_type}]->(b) "
        "ON CREATE SET r += $props "
        "ON MATCH SET r += $props "
        "RETURN 1 AS ok"
    )


def load_existing_keys(session, specs: Mapping[str, RelationshipSpec]) -> Dict[str, Set[str]]:
    label_key_pairs = sorted({(spec.from_label, spec.from_key) for spec in specs.values()} | {(spec.to_label, spec.to_key) for spec in specs.values()})
    key_sets: Dict[str, Set[str]] = {}
    for label, key_field in label_key_pairs:
        query = key_set_query(label, key_field)
        records = session.run(query).data()
        key_sets[f"{label}.{key_field}"] = {clean_text(record.get("key_value")) for record in records if clean_text(record.get("key_value"))}
    return key_sets


def ingest_relationship_type(
    session,
    spec: RelationshipSpec,
    records: Sequence[Mapping[str, Any]],
    key_sets: Mapping[str, Set[str]],
) -> Dict[str, Any]:
    query = relationship_merge_query(spec)
    created_relationships = 0
    failed_relationships = 0
    missing_from_node = 0
    missing_to_node = 0
    failures: List[Dict[str, str]] = []

    from_key_set = key_sets.get(f"{spec.from_label}.{spec.from_key}", set())
    to_key_set = key_sets.get(f"{spec.to_label}.{spec.to_key}", set())

    for row in records:
        from_key = clean_text(row.get("from_key"))
        to_key = clean_text(row.get("to_key"))
        props = row.get("props", {})

        if not from_key or not to_key:
            failed_relationships += 1
            failures.append({"from_key": from_key, "to_key": to_key, "error": "missing_endpoint_key"})
            continue
        has_from = from_key in from_key_set
        has_to = to_key in to_key_set
        if not has_from or not has_to:
            failed_relationships += 1
            if not has_from:
                missing_from_node += 1
            if not has_to:
                missing_to_node += 1
            failures.append(
                {
                    "from_key": from_key,
                    "to_key": to_key,
                    "error": "endpoint_node_missing",
                }
            )
            continue

        try:
            summary = session.run(query, from_key=from_key, to_key=to_key, props=props).consume()
            created_relationships += int(summary.counters.relationships_created)
        except Exception as exc:  # pragma: no cover - runtime branch
            failed_relationships += 1
            failures.append({"from_key": from_key, "to_key": to_key, "error": str(exc)})

    attempted = len(records)
    matched_existing_relationships = attempted - created_relationships - failed_relationships
    return {
        "attempted_relationships": attempted,
        "created_relationships": created_relationships,
        "matched_existing_relationships": matched_existing_relationships,
        "failed_relationships": failed_relationships,
        "missing_from_node": missing_from_node,
        "missing_to_node": missing_to_node,
        "failures": failures[:100],
    }


def relationship_counts(session) -> Dict[str, int]:
    result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS rel_count ORDER BY rel_type")
    counts = {clean_text(record["rel_type"]): int(record["rel_count"]) for record in result}
    return counts


def orphan_node_detection(session) -> Dict[str, int]:
    queries = {
        "Beverage_without_CONTAINS": "MATCH (n:Beverage) WHERE NOT (n)-[:CONTAINS]->(:Compound) RETURN count(n) AS count",
        "Compound_without_BELONGS_TO": "MATCH (n:Compound) WHERE NOT (n)-[:BELONGS_TO]->(:ChemicalClass) RETURN count(n) AS count",
        "Compound_without_CONTRIBUTES_TO": "MATCH (n:Compound) WHERE NOT (n)-[:CONTRIBUTES_TO]->(:ToxicityRisk) RETURN count(n) AS count",
        "Compound_without_METABOLIZED_BY": "MATCH (n:Compound) WHERE NOT (n)-[:METABOLIZED_BY]->(:Enzyme) RETURN count(n) AS count",
        "PopulationGroup_without_MODIFIES": "MATCH (n:PopulationGroup) WHERE NOT (n)-[:MODIFIES]->(:PBPKParameter) RETURN count(n) AS count",
        "PBPKParameter_without_AFFECTS": "MATCH (n:PBPKParameter) WHERE NOT (n)-[:AFFECTS]->(:BodyCompartment) RETURN count(n) AS count",
        "BodyCompartment_without_incoming_AFFECTS": "MATCH (n:BodyCompartment) WHERE NOT (:PBPKParameter)-[:AFFECTS]->(n) RETURN count(n) AS count",
        "ToxicityRisk_without_incoming_CONTRIBUTES_TO": "MATCH (n:ToxicityRisk) WHERE NOT (:Compound)-[:CONTRIBUTES_TO]->(n) RETURN count(n) AS count",
        "PhysiologyCondition_without_signed_effect": "MATCH (n:PhysiologyCondition) WHERE NOT (n)-[:INCREASES|DECREASES]->(:PBPKParameter) RETURN count(n) AS count",
        "Enzyme_without_incoming_METABOLIZED_BY": "MATCH (n:Enzyme) WHERE NOT (:Compound)-[:METABOLIZED_BY]->(n) RETURN count(n) AS count",
        "ChemicalClass_without_incoming_BELONGS_TO": "MATCH (n:ChemicalClass) WHERE NOT (:Compound)-[:BELONGS_TO]->(n) RETURN count(n) AS count",
    }
    results: Dict[str, int] = {}
    for key, query in queries.items():
        record = session.run(query).single()
        results[key] = int(record["count"]) if record else 0
    return results


def build_failed_report(config: Mapping[str, str], missing_env: Sequence[str], error: str, input_refs: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "status": "failed",
        "neo4j_env": {
            "uri": config.get("uri", ""),
            "user": config.get("user", ""),
            "database": config.get("database", ""),
            "password_provided": "yes" if bool(config.get("password")) else "no",
        },
        "missing_environment_variables": list(missing_env),
        "error": error,
        "inputs": dict(input_refs),
        "relationship_metrics": {},
        "graph_validation": {},
        "safe_for_graph_validation": False,
        "reasoning": [
            "Relationship ingestion did not complete successfully.",
            "Resolve environment/driver/connectivity prerequisites before retry.",
        ],
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = output_report_path(root)
    missing_env: List[str] = []
    config: Dict[str, str] = {"uri": "", "user": "", "password": "", "database": "neo4j"}
    paths = input_paths(root)
    input_refs = {key: str(path.relative_to(root)) for key, path in paths.items()}

    try:
        config = get_neo4j_config()
    except ValueError as exc:
        report = build_failed_report(config, missing_env, str(exc), input_refs)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j relationship ingestion report -> %s", report_path)
        LOGGER.info("safe_for_graph_validation=%s", report["safe_for_graph_validation"])
        return

    if GraphDatabase is None:
        report = build_failed_report(config, [], "neo4j Python driver is not installed in this environment.", input_refs)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Neo4j relationship ingestion report -> %s", report_path)
        LOGGER.info("safe_for_graph_validation=%s", report["safe_for_graph_validation"])
        return

    try:
        matrix_df = read_csv(paths["beverage_compound_matrix_expanded"])
        human_df = read_csv(paths["human_metabolism_parameters"])
        pbpk_df = read_csv(paths["pbpk_parameter_library"])
        pop_df = read_csv(paths["population_modifiers"])
        effects_df = read_csv(paths["beverage_effect_modifiers"])

        payloads = build_relationship_payloads(
            matrix_df=matrix_df,
            human_df=human_df,
            pbpk_df=pbpk_df,
            pop_df=pop_df,
            effects_df=effects_df,
        )

        relationship_metrics: Dict[str, Any] = {}
        graph_validation: Dict[str, Any] = {}

        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            key_sets = load_existing_keys(session, REL_SPECS)
            for rel_type in RELATIONSHIP_TYPES:
                spec = REL_SPECS[rel_type]
                records = payloads.get(rel_type, [])
                relationship_metrics[rel_type] = ingest_relationship_type(
                    session=session,
                    spec=spec,
                    records=records,
                    key_sets=key_sets,
                )

            graph_validation["relationship_counts"] = relationship_counts(session)
            graph_validation["orphan_node_detection"] = orphan_node_detection(session)
        driver.close()

        total_failed = sum(int(metrics["failed_relationships"]) for metrics in relationship_metrics.values())
        missing_linkage_stats = {
            rel_type: {
                "missing_from_node": int(metrics["missing_from_node"]),
                "missing_to_node": int(metrics["missing_to_node"]),
            }
            for rel_type, metrics in relationship_metrics.items()
        }
        graph_validation["missing_linkage_stats"] = missing_linkage_stats

        required_rel_present = all(rel_type in graph_validation["relationship_counts"] for rel_type in RELATIONSHIP_TYPES)
        safe_for_graph_validation = (total_failed == 0) and required_rel_present

        report: Dict[str, Any] = {
            "status": "success",
            "neo4j_env": redact_config(config),
            "missing_environment_variables": [],
            "inputs": input_refs,
            "relationship_metrics": relationship_metrics,
            "graph_validation": graph_validation,
            "safe_for_graph_validation": safe_for_graph_validation,
            "reasoning": [
                f"Failed relationships total: {total_failed}.",
                f"All required relationship types present in graph counts: {required_rel_present}.",
                "Relationship ingestion uses deterministic MATCH + MERGE and is idempotent.",
            ],
        }
    except Exception as exc:  # pragma: no cover - runtime branch
        report = build_failed_report(config, [], str(exc), input_refs)

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Neo4j relationship ingestion report -> %s", report_path)
    LOGGER.info("safe_for_graph_validation=%s", report["safe_for_graph_validation"])


if __name__ == "__main__":
    main()
