"""ETL step 06b: audit Neo4j ingestion scripts against frozen ontology schema."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

LOGGER = logging.getLogger("etl_06b_neo4j_ingestion_audit")

ENCODING = "utf-8"

SCRIPT_PATHS: Tuple[str, ...] = (
    "etl/etl_6_neo4j.py",
    "etl/etl_6_neo4j_stream.py",
    "etl/etl_7_neo4j_finalize.py",
)

FROZEN_SCHEMA_PATH = "rag/neo4j/neo4j_graph_schema_design.md"

REQUIRED_NODES: Tuple[str, ...] = (
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

REQUIRED_INPUT_REFERENCES: Tuple[str, ...] = (
    "data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv",
    "data/processed/human/human_metabolism_parameters.csv",
    "data/processed/pbpk/pbpk_parameter_library.csv",
    "data/processed/pbpk/population_modifiers.csv",
    "data/processed/pbpk/beverage_effect_modifiers.csv",
)

OUTDATED_TOKENS: Tuple[str, ...] = (
    "aop-wiki-xml",
    "data/raw/04_biological_pathways",
    "XmlNode",
    "XmlWord",
    "PARENT_OF",
)

LABEL_PATTERN = re.compile(r"\(([A-Za-z_][A-Za-z0-9_]*)?:([A-Za-z_][A-Za-z0-9_]*)")
REL_PATTERN = re.compile(r"\[:([A-Z_]+)\]")
MERGE_PATTERN = re.compile(r"\b(MERGE|CREATE)\b")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "neo4j" / "neo4j_ingestion_audit_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_label(label: str) -> str:
    return label.strip()


def extract_script_tokens(text: str) -> Dict[str, Any]:
    labels = sorted({normalize_label(match[1]) for match in LABEL_PATTERN.findall(text) if match[1]})
    rels = sorted(set(REL_PATTERN.findall(text)))
    merge_statements = len(MERGE_PATTERN.findall(text))
    return {
        "labels": labels,
        "relationships": rels,
        "merge_or_create_statements": merge_statements,
    }


def evaluate_script(path: Path, frozen_schema_text: str) -> Dict[str, Any]:
    text = path.read_text(encoding=ENCODING)
    tokens = extract_script_tokens(text)

    found_required_nodes = sorted(node for node in REQUIRED_NODES if node in text or node in tokens["labels"])
    missing_required_nodes = sorted(node for node in REQUIRED_NODES if node not in found_required_nodes)

    found_required_relationships = sorted(rel for rel in REQUIRED_RELATIONSHIPS if rel in text or rel in tokens["relationships"])
    missing_required_relationships = sorted(rel for rel in REQUIRED_RELATIONSHIPS if rel not in found_required_relationships)

    referenced_inputs = sorted(reference for reference in REQUIRED_INPUT_REFERENCES if reference in text)
    missing_inputs = sorted(reference for reference in REQUIRED_INPUT_REFERENCES if reference not in referenced_inputs)

    outdated_markers = sorted(token for token in OUTDATED_TOKENS if token in text)
    uses_frozen_schema = FROZEN_SCHEMA_PATH in text or "PBPKParameter" in text

    hardcoded_credentials = ("NEO4J_PASSWORD" in text) and ("Ihatepassword" in text)
    graph_write_operations = any(keyword in text for keyword in ("CREATE (", "MERGE (", "DETACH DELETE"))

    incompatibilities: List[str] = []
    if missing_required_nodes:
        incompatibilities.append("Missing required ontology node labels in script logic.")
    if missing_required_relationships:
        incompatibilities.append("Missing required ontology relationship types in script logic.")
    if missing_inputs:
        incompatibilities.append("Does not reference current processed ETL CSV inputs required by frozen ontology.")
    if outdated_markers:
        incompatibilities.append("Uses outdated XML-centric ingestion assumptions and labels.")
    if not uses_frozen_schema:
        incompatibilities.append("No linkage to frozen ontology schema entities.")
    if hardcoded_credentials:
        incompatibilities.append("Contains hardcoded Neo4j credentials.")

    compatibility_score = 0.0
    if REQUIRED_NODES:
        compatibility_score += 0.4 * (len(found_required_nodes) / float(len(REQUIRED_NODES)))
    if REQUIRED_RELATIONSHIPS:
        compatibility_score += 0.4 * (len(found_required_relationships) / float(len(REQUIRED_RELATIONSHIPS)))
    if REQUIRED_INPUT_REFERENCES:
        compatibility_score += 0.2 * (len(referenced_inputs) / float(len(REQUIRED_INPUT_REFERENCES)))
    compatibility_score = round(compatibility_score, 4)

    script_safe = (
        not missing_required_nodes
        and not missing_required_relationships
        and not missing_inputs
        and not outdated_markers
        and uses_frozen_schema
    )

    return {
        "script": str(path),
        "assumptions": {
            "detected_labels": tokens["labels"],
            "detected_relationship_types": tokens["relationships"],
            "merge_or_create_statements": tokens["merge_or_create_statements"],
            "graph_write_operations_present": graph_write_operations,
        },
        "ontology_comparison": {
            "required_nodes_found": found_required_nodes,
            "required_nodes_missing": missing_required_nodes,
            "required_relationships_found": found_required_relationships,
            "required_relationships_missing": missing_required_relationships,
            "required_inputs_referenced": referenced_inputs,
            "required_inputs_missing": missing_inputs,
        },
        "detected_issues": {
            "outdated_markers": outdated_markers,
            "hardcoded_credentials": hardcoded_credentials,
            "uses_frozen_schema_entities": uses_frozen_schema,
            "incompatibilities": incompatibilities,
        },
        "compatibility_score": compatibility_score,
        "safe_for_ontology_ingestion": script_safe,
    }


def aggregate_decision(per_script: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = [float(item["compatibility_score"]) for item in per_script]
    avg_score = round(sum(scores) / float(len(scores)), 4) if scores else 0.0
    any_safe = any(bool(item["safe_for_ontology_ingestion"]) for item in per_script)
    all_safe = all(bool(item["safe_for_ontology_ingestion"]) for item in per_script) if per_script else False

    if all_safe:
        recommendation = "reuse_existing"
    elif any_safe:
        recommendation = "patch_existing"
    else:
        recommendation = "rewrite_required"

    safe_for_build = all_safe
    reasoning: List[str] = [
        f"Scripts audited: {len(per_script)}.",
        f"Average compatibility score: {avg_score}.",
        f"Scripts individually safe for frozen ontology ingestion: {sum(1 for item in per_script if item['safe_for_ontology_ingestion'])}.",
    ]
    if recommendation == "rewrite_required":
        reasoning.append(
            "Current scripts are XML-pathway ingesters and do not construct the required PBPK/beverage ontology nodes or relationships."
        )
    elif recommendation == "patch_existing":
        reasoning.append("Partial compatibility detected; significant patching is required.")
    else:
        reasoning.append("Existing scripts are compatible with frozen ontology.")

    return {
        "reuse_recommendation": recommendation,
        "safe_for_neo4j_build": safe_for_build,
        "average_compatibility_score": avg_score,
        "reasoning": reasoning,
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    schema_path = root / FROZEN_SCHEMA_PATH
    frozen_schema_text = schema_path.read_text(encoding=ENCODING)

    per_script: List[Dict[str, Any]] = []
    for relative_path in SCRIPT_PATHS:
        script_path = root / relative_path
        if not script_path.exists():
            per_script.append(
                {
                    "script": str(script_path),
                    "missing_script": True,
                    "compatibility_score": 0.0,
                    "safe_for_ontology_ingestion": False,
                    "detected_issues": {"incompatibilities": ["Script file missing."]},
                }
            )
            continue
        per_script.append(evaluate_script(script_path, frozen_schema_text))

    decision = aggregate_decision(per_script)
    report: Dict[str, Any] = {
        "frozen_schema": FROZEN_SCHEMA_PATH,
        "required_nodes": list(REQUIRED_NODES),
        "required_relationships": list(REQUIRED_RELATIONSHIPS),
        "required_input_files": list(REQUIRED_INPUT_REFERENCES),
        "scripts_audited": list(SCRIPT_PATHS),
        "script_findings": per_script,
        "decision": decision,
    }

    out_path = output_path(root)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Neo4j ingestion audit report -> %s", out_path)
    LOGGER.info("safe_for_neo4j_build=%s", decision["safe_for_neo4j_build"])
    LOGGER.info("reuse_recommendation=%s", decision["reuse_recommendation"])


if __name__ == "__main__":
    main()
