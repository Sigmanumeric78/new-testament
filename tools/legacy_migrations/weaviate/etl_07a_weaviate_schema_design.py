"""ETL step 07a: deterministic Weaviate retrieval ontology design (no ingestion)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_07a_weaviate_schema_design")

ENCODING = "utf-8"

REQUIRED_NEO4J_NODES: Tuple[str, ...] = (
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

REQUIRED_NEO4J_RELATIONSHIPS: Tuple[str, ...] = (
    "CONTAINS",
    "BELONGS_TO",
    "METABOLIZED_BY",
    "MODIFIES",
    "AFFECTS",
    "CONTRIBUTES_TO",
    "INCREASES",
    "DECREASES",
)

INPUT_FILES: Mapping[str, Dict[str, Any]] = {
    "neo4j_ontology_design": {
        "type": "markdown",
        "path": "rag/neo4j/neo4j_graph_schema_design.md",
        "required_terms": list(REQUIRED_NEO4J_NODES) + list(REQUIRED_NEO4J_RELATIONSHIPS),
    },
    "beverage_compound_matrix": {
        "type": "csv",
        "path": "data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv",
        "required_columns": [
            "beverage_id",
            "beverage_name",
            "category",
            "compound_name",
            "normalized_compound_name",
            "pubchem_cid",
            "chemical_category",
            "source_compound_class",
            "expansion_type",
            "source_dataset",
            "source_file",
            "source_row",
            "confidence_score",
        ],
    },
    "human_metabolism_parameters": {
        "type": "csv",
        "path": "data/processed/human/human_metabolism_parameters.csv",
        "required_columns": [
            "parameter_id",
            "parameter_name",
            "domain",
            "population_group",
            "condition",
            "value",
            "unit",
            "modifier_type",
            "effect_direction",
            "confidence_score",
            "evidence_text",
            "source_document",
            "source_page",
            "extract_method",
        ],
    },
    "pbpk_parameter_library": {
        "type": "csv",
        "path": "data/processed/pbpk/pbpk_parameter_library.csv",
        "required_columns": [
            "parameter_id",
            "parameter_name",
            "compartment",
            "base_value",
            "unit",
            "population_group",
            "modifier",
            "modifier_reason",
            "confidence_score",
            "source_document",
            "source_parameter_id",
        ],
    },
    "beverage_effect_modifiers": {
        "type": "csv",
        "path": "data/processed/pbpk/beverage_effect_modifiers.csv",
        "required_columns": [
            "modifier_id",
            "beverage_id",
            "beverage_name",
            "category",
            "parameter_name",
            "compartment",
            "modifier",
            "modifier_reason",
            "trigger_compounds",
            "source_compound_class",
            "confidence_score",
        ],
    },
}


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    source_keys: Tuple[str, ...]
    primary_key: str
    searchable_content_fields: Tuple[str, ...]
    metadata_fields: Tuple[str, ...]
    provenance_fields: Tuple[str, ...]
    retrieval_intent: str
    expected_query_examples: Tuple[str, ...]


COLLECTION_SPECS: Tuple[CollectionSpec, ...] = (
    CollectionSpec(
        name="BeverageKnowledge",
        source_keys=("beverage_compound_matrix", "beverage_effect_modifiers"),
        primary_key="beverage_id",
        searchable_content_fields=(
            "beverage_name",
            "category",
            "compound_name",
            "source_compound_class",
            "modifier_reason",
            "trigger_compounds",
        ),
        metadata_fields=(
            "beverage_id",
            "category",
            "expansion_type",
            "chemical_category",
            "compound_role",
            "estimated_concentration",
            "concentration_unit",
        ),
        provenance_fields=("source_dataset", "source_file", "source_row", "confidence_score"),
        retrieval_intent="Resolve beverage composition, class expansions, and likely downstream exposure drivers.",
        expected_query_examples=(
            "What compounds are in whisky and which are family-expanded?",
            "Which beverage categories carry sulfur-related compounds?",
            "Which compounds in bourbon are linked to modifier signals?",
        ),
    ),
    CollectionSpec(
        name="CompoundKnowledge",
        source_keys=("beverage_compound_matrix",),
        primary_key="normalized_compound_name",
        searchable_content_fields=(
            "compound_name",
            "normalized_compound_name",
            "pubchem_cid",
            "chemical_category",
            "compound_role",
            "source_compound_class",
        ),
        metadata_fields=(
            "chemical_category",
            "compound_role",
            "estimated_concentration",
            "concentration_unit",
            "digestion_effect",
            "metabolic_burden",
            "expansion_type",
        ),
        provenance_fields=("source_dataset", "source_file", "source_row", "confidence_score"),
        retrieval_intent="Answer molecule-level lookup and category membership questions with provenance.",
        expected_query_examples=(
            "What is the PubChem CID for isoamyl alcohol?",
            "Which compounds belong to organic acid class?",
            "Which compounds are marked digestion-relevant?",
        ),
    ),
    CollectionSpec(
        name="MetabolismKnowledge",
        source_keys=("human_metabolism_parameters",),
        primary_key="parameter_id",
        searchable_content_fields=(
            "parameter_name",
            "domain",
            "population_group",
            "condition",
            "value",
            "unit",
            "evidence_text",
        ),
        metadata_fields=("modifier_type", "effect_direction", "extract_method", "source_page"),
        provenance_fields=("source_document", "source_page", "confidence_score", "extract_method"),
        retrieval_intent="Retrieve physiology and metabolism evidence by domain, population, and condition.",
        expected_query_examples=(
            "What evidence supports slower gastric emptying in fed state?",
            "How does sex affect body water distribution parameters?",
            "Which records quantify ethanol elimination rate?",
        ),
    ),
    CollectionSpec(
        name="PBPKKnowledge",
        source_keys=("pbpk_parameter_library",),
        primary_key="parameter_id",
        searchable_content_fields=(
            "parameter_name",
            "compartment",
            "base_value",
            "unit",
            "modifier_reason",
            "source_parameter_id",
        ),
        metadata_fields=("population_group", "modifier", "confidence_score"),
        provenance_fields=("source_document", "source_parameter_id", "confidence_score"),
        retrieval_intent="Ground simulator parameter selection and compartment-specific interpretation.",
        expected_query_examples=(
            "What is the base ethanol elimination rate and evidence source?",
            "Which parameters affect the liver compartment?",
            "How is first-pass metabolism parameterized?",
        ),
    ),
    CollectionSpec(
        name="ToxicityKnowledge",
        source_keys=("beverage_effect_modifiers",),
        primary_key="modifier_id",
        searchable_content_fields=(
            "beverage_name",
            "category",
            "parameter_name",
            "modifier_reason",
            "trigger_compounds",
            "source_compound_class",
        ),
        metadata_fields=("beverage_id", "compartment", "modifier", "confidence_score"),
        provenance_fields=("source_compound_class", "confidence_score"),
        retrieval_intent="Explain toxicity and hangover-linked modifier signals from beverage chemistry.",
        expected_query_examples=(
            "Which compounds contribute to hangover amplification risk?",
            "What modifier signals are associated with histamine in beer?",
            "Which beverages have sulfite-linked sensitivity modifiers?",
        ),
    ),
    CollectionSpec(
        name="PopulationKnowledge",
        source_keys=("human_metabolism_parameters", "pbpk_parameter_library"),
        primary_key="population_group",
        searchable_content_fields=(
            "population_group",
            "domain",
            "condition",
            "effect_direction",
            "parameter_name",
            "modifier_reason",
        ),
        metadata_fields=("value", "unit", "modifier_type", "extract_method", "compartment"),
        provenance_fields=("source_document", "source_page", "confidence_score", "source_parameter_id"),
        retrieval_intent="Resolve group-specific physiology effects and PBPK implications.",
        expected_query_examples=(
            "How does fasted state affect absorption-relevant physiology?",
            "Which parameters differ for female vs male groups?",
            "What evidence exists for liver impairment modifiers?",
        ),
    ),
    CollectionSpec(
        name="ScientificEvidence",
        source_keys=("human_metabolism_parameters", "pbpk_parameter_library", "beverage_compound_matrix"),
        primary_key="derived:evidence_id=sha1(source_document|source_page|source_parameter_id|source_row|parameter_name)",
        searchable_content_fields=(
            "evidence_text",
            "source_document",
            "parameter_name",
            "domain",
            "condition",
            "modifier_reason",
            "compound_name",
        ),
        metadata_fields=("source_page", "unit", "value", "extract_method", "chemical_category", "compartment"),
        provenance_fields=("source_document", "source_page", "source_file", "source_row", "confidence_score"),
        retrieval_intent="Provide auditable evidence retrieval and citation surfaces for downstream QA/RAG.",
        expected_query_examples=(
            "Show evidence text for ethanol elimination rate assumptions.",
            "Which source documents support body water distribution differences?",
            "What are the source rows behind whisky-related chemistry claims?",
        ),
    ),
)

NODE_PATTERN = re.compile(r"`([A-Za-z][A-Za-z0-9_]*)`")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def markdown_output_path(root: Path) -> Path:
    path = root / "rag" / "weaviate" / "weaviate_schema_design.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def report_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "weaviate_schema_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalized_set(values: Sequence[str]) -> List[str]:
    return sorted({str(value) for value in values if str(value)})


def analyze_markdown(path: Path, required_terms: Sequence[str]) -> Dict[str, Any]:
    exists = path.exists()
    if not exists:
        return {"exists": False, "missing_required_terms": list(required_terms), "detected_tokens": []}
    text = path.read_text(encoding=ENCODING)
    detected_tokens = normalized_set(NODE_PATTERN.findall(text))
    missing_terms = sorted(term for term in required_terms if term not in text)
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "missing_required_terms": missing_terms,
        "detected_tokens": detected_tokens,
    }


def analyze_csv(path: Path, required_columns: Sequence[str]) -> Dict[str, Any]:
    exists = path.exists()
    if not exists:
        return {
            "exists": False,
            "missing_required_columns": list(required_columns),
            "row_count": 0,
            "column_count": 0,
            "columns": [],
        }
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=ENCODING)
    columns = list(df.columns)
    missing_required = sorted(col for col in required_columns if col not in df.columns)
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "missing_required_columns": missing_required,
    }


def collect_source_columns(input_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, List[str]]:
    source_columns: Dict[str, List[str]] = {}
    for source_key, result in input_results.items():
        columns = result.get("columns", [])
        if isinstance(columns, list):
            source_columns[source_key] = [str(col) for col in columns]
        else:
            source_columns[source_key] = []
    return source_columns


def field_is_derived(field_name: str) -> bool:
    return field_name.startswith("derived:")


def evaluate_field_presence(field_name: str, source_keys: Sequence[str], source_columns: Mapping[str, Sequence[str]]) -> bool:
    if field_is_derived(field_name):
        return True
    for source_key in source_keys:
        if field_name in set(source_columns.get(source_key, [])):
            return True
    return False


def evaluate_collections(source_columns: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    collection_rows: List[Dict[str, Any]] = []
    for spec in COLLECTION_SPECS:
        searchable_presence = {
            field: evaluate_field_presence(field, spec.source_keys, source_columns)
            for field in spec.searchable_content_fields
        }
        metadata_presence = {
            field: evaluate_field_presence(field, spec.source_keys, source_columns) for field in spec.metadata_fields
        }
        provenance_presence = {
            field: evaluate_field_presence(field, spec.source_keys, source_columns)
            for field in spec.provenance_fields
        }
        primary_key_present = evaluate_field_presence(spec.primary_key, spec.source_keys, source_columns)

        missing_searchable = sorted([field for field, ok in searchable_presence.items() if not ok])
        missing_metadata = sorted([field for field, ok in metadata_presence.items() if not ok])
        missing_provenance = sorted([field for field, ok in provenance_presence.items() if not ok])

        design_ready = primary_key_present and not missing_searchable and not missing_provenance
        collection_rows.append(
            {
                "name": spec.name,
                "source_keys": list(spec.source_keys),
                "primary_key": spec.primary_key,
                "searchable_content_fields": list(spec.searchable_content_fields),
                "metadata_fields": list(spec.metadata_fields),
                "provenance_fields": list(spec.provenance_fields),
                "retrieval_intent": spec.retrieval_intent,
                "expected_query_examples": list(spec.expected_query_examples),
                "validation": {
                    "primary_key_present": primary_key_present,
                    "missing_searchable_fields": missing_searchable,
                    "missing_metadata_fields": missing_metadata,
                    "missing_provenance_fields": missing_provenance,
                    "design_ready": design_ready,
                },
            }
        )
    return collection_rows


def chunking_strategy() -> Dict[str, Any]:
    return {
        "deterministic_rules": [
            "No random splitting.",
            "No token-length randomization or stochastic overlap.",
            "Chunk boundaries follow semantic entity boundaries only.",
            "Chunk IDs are deterministic from collection + primary key.",
            "Provenance fields are copied to every chunk.",
        ],
        "chunk_id_contract": {
            "format": "WVC::<CollectionName>::<normalized_primary_key_value>",
            "normalization": [
                "trim whitespace",
                "lowercase",
                "replace non-alphanumeric runs with underscore",
                "strip leading/trailing underscores",
            ],
        },
        "collection_boundaries": {
            "BeverageKnowledge": "one chunk per beverage_id",
            "CompoundKnowledge": "one chunk per normalized_compound_name",
            "MetabolismKnowledge": "one chunk per parameter_id row",
            "PBPKKnowledge": "one chunk per parameter_id row",
            "ToxicityKnowledge": "one chunk per modifier_id row",
            "PopulationKnowledge": "one chunk per population_group, grouped deterministically by parameter_name then condition",
            "ScientificEvidence": "one chunk per deterministic evidence_id from source provenance tuple",
        },
        "ordering_rules": [
            "Within each chunk, fields are serialized in fixed schema order.",
            "Rows are sorted by primary key ascending before chunk materialization.",
            "Multi-value lists are sorted lexicographically before serialization.",
        ],
    }


def embedding_contract() -> Dict[str, Any]:
    return {
        "model": "nomic-ai/nomic-embed-text-v1",
        "vector_dimensions": 768,
        "deterministic_preprocessing": {
            "text_normalization_rules": [
                "Unicode normalization: NFC.",
                "Normalize line endings to LF.",
                "Trim leading/trailing whitespace per field.",
                "Collapse internal whitespace runs to a single space.",
                "Lowercase field keys; preserve original field values except whitespace normalization.",
                "Serialize fields in fixed per-collection order using '\\n' separators.",
            ],
            "missing_field_handling": [
                "Missing/empty values become literal token 'unknown'.",
                "Do not drop missing fields; preserve placeholders for positional consistency.",
                "If all searchable fields are missing, skip vectorization and flag chunk as 'insufficient_text'.",
            ],
        },
        "reproducibility_guards": [
            "No random seed usage required; transformations are pure string functions.",
            "Identical input rows must produce byte-identical embedding payload text.",
            "Schema changes require version bump before new vector generation.",
        ],
    }


def build_markdown(
    input_analysis: Mapping[str, Mapping[str, Any]],
    collections: Sequence[Mapping[str, Any]],
    chunking: Mapping[str, Any],
    embedding: Mapping[str, Any],
    safe_for_materialization: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Weaviate Schema Design (Deterministic Freeze)")
    lines.append("")
    lines.append("## Scope")
    lines.append("This document freezes retrieval ontology design only. No ingestion, no live Weaviate calls, and no vector generation are performed in ETL_07A.")
    lines.append("")
    lines.append("Input artifacts analyzed:")
    for key, cfg in INPUT_FILES.items():
        path = cfg["path"]
        status = "present" if input_analysis[key]["exists"] else "missing"
        lines.append(f"- `{path}` ({status})")
    lines.append("")
    lines.append("## Retrieval Collections")
    for item in collections:
        lines.append(f"### {item['name']}")
        lines.append(f"- Primary key: `{item['primary_key']}`")
        lines.append(f"- Source artifacts: `{', '.join(item['source_keys'])}`")
        lines.append(f"- Retrieval intent: {item['retrieval_intent']}")
        lines.append("- Searchable content fields: " + ", ".join(f"`{f}`" for f in item["searchable_content_fields"]))
        lines.append("- Metadata fields: " + ", ".join(f"`{f}`" for f in item["metadata_fields"]))
        lines.append("- Provenance fields: " + ", ".join(f"`{f}`" for f in item["provenance_fields"]))
        lines.append("- Expected query examples:")
        for query in item["expected_query_examples"]:
            lines.append(f"  - {query}")
        lines.append(
            f"- Design readiness: `{item['validation']['design_ready']}` "
            f"(missing searchable={len(item['validation']['missing_searchable_fields'])}, "
            f"missing provenance={len(item['validation']['missing_provenance_fields'])})"
        )
        lines.append("")
    lines.append("## Deterministic Chunking Strategy")
    for rule in chunking["deterministic_rules"]:
        lines.append(f"- {rule}")
    lines.append(f"- Chunk ID format: `{chunking['chunk_id_contract']['format']}`")
    lines.append("- Chunk ID normalization rules:")
    for rule in chunking["chunk_id_contract"]["normalization"]:
        lines.append(f"  - {rule}")
    lines.append("- Per-collection semantic boundaries:")
    for name, boundary in chunking["collection_boundaries"].items():
        lines.append(f"  - `{name}`: {boundary}")
    lines.append("- Ordering rules:")
    for rule in chunking["ordering_rules"]:
        lines.append(f"  - {rule}")
    lines.append("")
    lines.append("## Embedding Contract")
    lines.append(f"- Model: `{embedding['model']}`")
    lines.append(f"- Vector dimensions: `{embedding['vector_dimensions']}`")
    lines.append("- Deterministic preprocessing:")
    for rule in embedding["deterministic_preprocessing"]["text_normalization_rules"]:
        lines.append(f"  - {rule}")
    lines.append("- Missing field handling:")
    for rule in embedding["deterministic_preprocessing"]["missing_field_handling"]:
        lines.append(f"  - {rule}")
    lines.append("- Reproducibility guards:")
    for guard in embedding["reproducibility_guards"]:
        lines.append(f"  - {guard}")
    lines.append("")
    lines.append("## Final Gate")
    lines.append(f"- `safe_for_weaviate_materialization`: `{str(safe_for_materialization).lower()}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = report_output_path(root)
    md_path = markdown_output_path(root)

    input_analysis: Dict[str, Dict[str, Any]] = {}
    for key, cfg in INPUT_FILES.items():
        file_path = root / cfg["path"]
        if cfg["type"] == "markdown":
            input_analysis[key] = analyze_markdown(file_path, cfg.get("required_terms", []))
        else:
            input_analysis[key] = analyze_csv(file_path, cfg.get("required_columns", []))
        input_analysis[key]["path"] = cfg["path"]

    source_columns = collect_source_columns(input_analysis)
    collection_rows = evaluate_collections(source_columns)
    chunking = chunking_strategy()
    embedding = embedding_contract()

    missing_inputs = sorted([key for key, value in input_analysis.items() if not bool(value.get("exists"))])
    schema_term_issues = input_analysis["neo4j_ontology_design"].get("missing_required_terms", [])
    csv_column_issues = {
        key: value.get("missing_required_columns", [])
        for key, value in input_analysis.items()
        if key != "neo4j_ontology_design"
    }
    missing_required_columns_total = sum(len(v) for v in csv_column_issues.values())

    collection_ready_count = sum(1 for row in collection_rows if row["validation"]["design_ready"])
    collection_total = len(collection_rows)

    safe_for_weaviate_materialization = (
        len(missing_inputs) == 0
        and len(schema_term_issues) == 0
        and missing_required_columns_total == 0
        and collection_ready_count == collection_total
    )

    report: Dict[str, Any] = {
        "status": "success",
        "inputs": input_analysis,
        "collections": collection_rows,
        "chunking_strategy": chunking,
        "embedding_contract": embedding,
        "quality_metrics": {
            "collections_total": collection_total,
            "collections_design_ready": collection_ready_count,
            "missing_input_count": len(missing_inputs),
            "missing_required_columns_total": missing_required_columns_total,
            "neo4j_required_term_gaps": len(schema_term_issues),
        },
        "safe_for_weaviate_materialization": safe_for_weaviate_materialization,
        "reasoning": [
            f"Input files missing: {len(missing_inputs)}.",
            f"Neo4j ontology required-term gaps: {len(schema_term_issues)}.",
            f"Required CSV column gaps: {missing_required_columns_total}.",
            f"Collections design-ready: {collection_ready_count}/{collection_total}.",
        ],
    }

    markdown = build_markdown(
        input_analysis=input_analysis,
        collections=collection_rows,
        chunking=chunking,
        embedding=embedding,
        safe_for_materialization=safe_for_weaviate_materialization,
    )

    md_path.write_text(markdown, encoding=ENCODING)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)

    LOGGER.info("Wrote Weaviate schema design -> %s", md_path)
    LOGGER.info("Wrote Weaviate schema report -> %s", report_path)
    LOGGER.info("safe_for_weaviate_materialization=%s", safe_for_weaviate_materialization)


if __name__ == "__main__":
    main()
