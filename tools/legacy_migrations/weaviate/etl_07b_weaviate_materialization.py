"""ETL step 07b: deterministic semantic object materialization for Weaviate ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_07b_weaviate_materialization")

ENCODING = "utf-8"
UNKNOWN = "unknown"

REQUIRED_COLLECTIONS: Tuple[str, ...] = (
    "BeverageKnowledge",
    "CompoundKnowledge",
    "MetabolismKnowledge",
    "PBPKKnowledge",
    "ToxicityKnowledge",
    "PopulationKnowledge",
    "ScientificEvidence",
)

INPUT_PATHS: Mapping[str, str] = {
    "beverage_compound_matrix": "data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv",
    "beverage_reference_table": "data/processed/beverage/reference_tables/master_beverage_reference_repaired.csv",
    "human_metabolism_parameters": "data/processed/human/human_metabolism_parameters.csv",
    "pbpk_parameter_library": "data/processed/pbpk/pbpk_parameter_library.csv",
    "population_modifiers": "data/processed/pbpk/population_modifiers.csv",
    "beverage_effect_modifiers": "data/processed/pbpk/beverage_effect_modifiers.csv",
    "weaviate_schema_design": "rag/weaviate/weaviate_schema_design.md",
}

OUTPUT_FILES: Mapping[str, str] = {
    "BeverageKnowledge": "data/processed/weaviate/beverage_knowledge.jsonl",
    "CompoundKnowledge": "data/processed/weaviate/compound_knowledge.jsonl",
    "MetabolismKnowledge": "data/processed/weaviate/metabolism_knowledge.jsonl",
    "PBPKKnowledge": "data/processed/weaviate/pbpk_knowledge.jsonl",
    "ToxicityKnowledge": "data/processed/weaviate/toxicity_knowledge.jsonl",
    "PopulationKnowledge": "data/processed/weaviate/population_knowledge.jsonl",
    "ScientificEvidence": "data/processed/weaviate/scientific_evidence.jsonl",
}

SCHEMA_REQUIRED_PHRASES: Tuple[str, ...] = REQUIRED_COLLECTIONS

RISK_PARAMETER_MAP: Mapping[str, str] = {
    "hangover_amplification": "hangover_amplification",
    "sensitivity": "sensitivity",
    "toxicity": "toxicity_response",
    "histamine": "toxicity_response",
    "sulfite": "sensitivity",
    "congener": "hangover_amplification",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def report_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "weaviate_materialization_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def normalize_key(value: Any) -> str:
    token = clean_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def list_unique(values: Iterable[Any]) -> List[str]:
    items = sorted({clean_text(v) for v in values if clean_text(v)})
    return items


def clip_join(values: Sequence[str], limit: int) -> str:
    if not values:
        return UNKNOWN
    selected = list(values[:limit])
    return ", ".join(selected)


def parse_confidence(value: Any) -> float:
    text = clean_text(value)
    if not text:
        return 0.0
    lowered = text.lower()
    if lowered in {"high", "strong"}:
        return 0.9
    if lowered in {"medium", "adequate"}:
        return 0.6
    if lowered in {"low", "weak"}:
        return 0.3
    try:
        score = float(text)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, score))


def mean_confidence(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    scores = [parse_confidence(v) for v in values]
    if not scores:
        return 0.0
    return round(sum(scores) / float(len(scores)), 4)


def build_chunk_id(collection: str, primary_key: str) -> str:
    return f"WVC::{collection}::{normalize_key(primary_key)}"


def build_object_id(collection: str, primary_key: str) -> str:
    return f"WVO::{collection}::{normalize_key(primary_key)}"


def sha1_token(text: str) -> str:
    return hashlib.sha1(text.encode(ENCODING)).hexdigest()


def ensure_schema_markdown(path: Path) -> Tuple[bool, List[str]]:
    if not path.exists():
        return False, list(SCHEMA_REQUIRED_PHRASES)
    text = path.read_text(encoding=ENCODING)
    missing = [phrase for phrase in SCHEMA_REQUIRED_PHRASES if phrase not in text]
    return True, missing


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=ENCODING)


def build_base_object(
    collection: str,
    primary_key: str,
    title: str,
    content: str,
    metadata: Mapping[str, Any],
    source_dataset: str,
    source_file: str,
    confidence_score: float,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    chunk_id = build_chunk_id(collection, primary_key)
    object_id = build_object_id(collection, primary_key)
    return {
        "object_id": object_id,
        "collection": collection,
        "title": clean_text(title) or f"{collection} object",
        "content": clean_text(content),
        "metadata": dict(metadata),
        "source_dataset": clean_text(source_dataset) or UNKNOWN,
        "source_file": clean_text(source_file) or UNKNOWN,
        "confidence_score": round(float(confidence_score), 4),
        "chunk_id": chunk_id,
        "provenance": dict(provenance),
    }


def build_beverage_knowledge(
    beverage_ref_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    bev_mod_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0

    ref_map = {clean_text(row["beverage_id"]): row for _, row in beverage_ref_df.iterrows()}
    grouped = matrix_df.groupby("beverage_id", sort=True, dropna=False)

    for beverage_id, group in grouped:
        b_id = clean_text(beverage_id)
        if not b_id:
            continue
        ref_row = ref_map.get(b_id)
        name = clean_text(group["beverage_name"].iloc[0]) or (clean_text(ref_row["beverage_name"]) if ref_row is not None else "")
        category = clean_text(group["category"].iloc[0]) or (clean_text(ref_row["category"]) if ref_row is not None else UNKNOWN)

        compounds = list_unique(group["compound_name"].tolist())
        classes = list_unique(group["source_compound_class"].tolist())
        expansions = list_unique(group["expansion_type"].tolist())
        chem_categories = list_unique(group["chemical_category"].tolist())
        roles = list_unique(group["compound_role"].tolist())
        conc_levels = list_unique(group["estimated_concentration"].tolist())

        mod_rows = bev_mod_df[bev_mod_df["beverage_id"] == b_id]
        modifier_reasons = list_unique(mod_rows["modifier_reason"].tolist())
        trigger_compounds = list_unique(mod_rows["trigger_compounds"].tolist())
        modifier_parameters = list_unique(mod_rows["parameter_name"].tolist())

        abv = clean_text(ref_row["baseline_abv"]) if ref_row is not None else UNKNOWN
        carbonation = clean_text(ref_row["carbonation"]) if ref_row is not None else UNKNOWN
        sugar = clean_text(ref_row["sugar_g_per_100ml"]) if ref_row is not None else UNKNOWN

        title = f"{name or b_id} beverage chemistry profile"
        content = (
            f"{name or b_id} is categorized as {category or UNKNOWN} with baseline ABV {abv or UNKNOWN}. "
            f"Detected compound classes include {clip_join(classes, 6)}. "
            f"Representative compounds include {clip_join(compounds, 8)}. "
            f"Observed chemical categories include {clip_join(chem_categories, 6)} and roles {clip_join(roles, 6)}. "
            f"Concentration signals are {clip_join(conc_levels, 4)}. "
            f"Carbonation is {carbonation or UNKNOWN} and sugar is {sugar or UNKNOWN} g/100ml. "
            f"Modifier pathways reference parameters {clip_join(modifier_parameters, 6)} with triggers {clip_join(trigger_compounds, 6)}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        source_dataset = clean_text(group["source_dataset"].iloc[0]) or (clean_text(ref_row["source_dataset"]) if ref_row is not None else UNKNOWN)
        source_file = clean_text(group["source_file"].iloc[0]) or (clean_text(ref_row["source_file"]) if ref_row is not None else UNKNOWN)
        confidence = mean_confidence(group["confidence_score"].tolist() + mod_rows["confidence_score"].tolist())

        src_rows = list_unique(group["source_row"].tolist())
        provenance = {
            "source_csvs": [
                "data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv",
                "data/processed/beverage/reference_tables/master_beverage_reference_repaired.csv",
                "data/processed/pbpk/beverage_effect_modifiers.csv",
            ],
            "source_rows": src_rows,
            "modifier_ids": list_unique(mod_rows["modifier_id"].tolist()),
            "source_dataset_values": list_unique(group["source_dataset"].tolist()),
            "confidence_values": list_unique(group["confidence_score"].tolist()),
        }
        metadata = {
            "beverage_id": b_id,
            "beverage_name": name or UNKNOWN,
            "category": category or UNKNOWN,
            "compound_count": int(group["normalized_compound_name"].nunique()),
            "compound_classes": classes,
            "expansion_types": expansions,
            "chemical_categories": chem_categories,
            "modifier_count": int(len(mod_rows)),
            "modifier_reasons": modifier_reasons[:8],
        }
        objects.append(
            build_base_object(
                collection="BeverageKnowledge",
                primary_key=b_id,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset=source_dataset,
                source_file=source_file,
                confidence_score=confidence,
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def build_compound_knowledge(matrix_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0
    grouped = matrix_df.groupby("normalized_compound_name", sort=True, dropna=False)
    for normalized_name, group in grouped:
        n_name = clean_text(normalized_name)
        if not n_name:
            continue
        compound_name = clean_text(group["compound_name"].iloc[0]) or n_name
        pubchem_cids = list_unique(group["pubchem_cid"].tolist())
        chem_categories = list_unique(group["chemical_category"].tolist())
        roles = list_unique(group["compound_role"].tolist())
        class_names = list_unique(group["source_compound_class"].tolist())
        digestion_effects = list_unique(group["digestion_effect"].tolist())
        metabolic_burden = list_unique(group["metabolic_burden"].tolist())
        expansion_types = list_unique(group["expansion_type"].tolist())
        beverages = list_unique(group["beverage_id"].tolist())

        title = f"{compound_name} compound profile"
        content = (
            f"{compound_name} (normalized as {n_name}) appears across {len(beverages)} beverages. "
            f"PubChem identifiers include {clip_join(pubchem_cids, 4)}. "
            f"It is classified as {clip_join(chem_categories, 4)} with roles {clip_join(roles, 4)}. "
            f"Source chemical families include {clip_join(class_names, 4)} and expansion types {clip_join(expansion_types, 4)}. "
            f"Digestion effect tags are {clip_join(digestion_effects, 3)} and metabolic burden tags are {clip_join(metabolic_burden, 3)}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        metadata = {
            "normalized_compound_name": n_name,
            "compound_name": compound_name,
            "pubchem_cids": pubchem_cids,
            "chemical_categories": chem_categories,
            "compound_roles": roles,
            "source_compound_classes": class_names,
            "expansion_types": expansion_types,
            "beverage_count": len(beverages),
        }
        provenance = {
            "source_csvs": ["data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv"],
            "source_rows": list_unique(group["source_row"].tolist()),
            "source_dataset_values": list_unique(group["source_dataset"].tolist()),
            "confidence_values": list_unique(group["confidence_score"].tolist()),
        }
        objects.append(
            build_base_object(
                collection="CompoundKnowledge",
                primary_key=n_name,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset=clean_text(group["source_dataset"].iloc[0]) or UNKNOWN,
                source_file=clean_text(group["source_file"].iloc[0]) or UNKNOWN,
                confidence_score=mean_confidence(group["confidence_score"].tolist()),
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def build_metabolism_knowledge(human_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0
    ordered = human_df.sort_values(by=["parameter_id", "population_group", "condition"], kind="mergesort")
    for row_idx, row in ordered.iterrows():
        parameter_id = clean_text(row["parameter_id"])
        if not parameter_id:
            continue
        parameter_name = clean_text(row["parameter_name"]) or UNKNOWN
        domain = clean_text(row["domain"]) or UNKNOWN
        population_group = clean_text(row["population_group"]) or UNKNOWN
        condition = clean_text(row["condition"]) or UNKNOWN
        value = clean_text(row["value"]) or UNKNOWN
        unit = clean_text(row["unit"]) or UNKNOWN
        modifier_type = clean_text(row["modifier_type"]) or UNKNOWN
        direction = clean_text(row["effect_direction"]) or UNKNOWN
        evidence_text = clean_text(row["evidence_text"]) or UNKNOWN
        source_document = clean_text(row["source_document"]) or UNKNOWN
        source_page = clean_text(row["source_page"]) or UNKNOWN
        extract_method = clean_text(row["extract_method"]) or UNKNOWN

        title = f"{parameter_name} metabolism evidence for {population_group}"
        content = (
            f"In the {domain} domain, parameter {parameter_name} for population {population_group} under "
            f"condition {condition} has value {value} {unit}. "
            f"Effect direction is {direction} with modifier type {modifier_type}. "
            f"Evidence states: {evidence_text}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        metadata = {
            "parameter_id": parameter_id,
            "parameter_name": parameter_name,
            "domain": domain,
            "population_group": population_group,
            "condition": condition,
            "value": value,
            "unit": unit,
            "modifier_type": modifier_type,
            "effect_direction": direction,
            "extract_method": extract_method,
            "source_page": source_page,
        }
        provenance = {
            "source_csvs": ["data/processed/human/human_metabolism_parameters.csv"],
            "source_rows": [str(row_idx)],
            "source_document": source_document,
            "source_page": source_page,
            "extract_method": extract_method,
        }
        objects.append(
            build_base_object(
                collection="MetabolismKnowledge",
                primary_key=parameter_id,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="human_metabolism_parameters",
                source_file=source_document,
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def build_pbpk_knowledge(pbpk_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0
    ordered = pbpk_df.sort_values(by=["parameter_id"], kind="mergesort")
    for row_idx, row in ordered.iterrows():
        parameter_id = clean_text(row["parameter_id"])
        if not parameter_id:
            continue
        parameter_name = clean_text(row["parameter_name"]) or UNKNOWN
        compartment = clean_text(row["compartment"]) or UNKNOWN
        base_value = clean_text(row["base_value"]) or UNKNOWN
        unit = clean_text(row["unit"]) or UNKNOWN
        population_group = clean_text(row["population_group"]) or UNKNOWN
        modifier = clean_text(row["modifier"]) or UNKNOWN
        modifier_reason = clean_text(row["modifier_reason"]) or UNKNOWN
        source_document = clean_text(row["source_document"]) or UNKNOWN
        source_parameter_id = clean_text(row["source_parameter_id"]) or UNKNOWN

        title = f"{parameter_name} PBPK parameter in {compartment}"
        content = (
            f"PBPK parameter {parameter_name} affects the {compartment} compartment with base value "
            f"{base_value} {unit}. The baseline population context is {population_group}. "
            f"Modifier mode is {modifier} because {modifier_reason}. Source parameter linkage is {source_parameter_id}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        metadata = {
            "parameter_id": parameter_id,
            "parameter_name": parameter_name,
            "compartment": compartment,
            "base_value": base_value,
            "unit": unit,
            "population_group": population_group,
            "modifier": modifier,
            "modifier_reason": modifier_reason,
            "source_parameter_id": source_parameter_id,
        }
        provenance = {
            "source_csvs": ["data/processed/pbpk/pbpk_parameter_library.csv"],
            "source_rows": [str(row_idx)],
            "source_document": source_document,
            "source_parameter_id": source_parameter_id,
        }
        objects.append(
            build_base_object(
                collection="PBPKKnowledge",
                primary_key=parameter_id,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="pbpk_parameter_library",
                source_file=source_document,
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def infer_risk_type(row: Mapping[str, Any]) -> str:
    parameter_name = clean_text(row.get("parameter_name", "")).lower()
    modifier_reason = clean_text(row.get("modifier_reason", "")).lower()
    trigger = clean_text(row.get("trigger_compounds", "")).lower()
    combined = "|".join([parameter_name, modifier_reason, trigger])
    for token, risk_type in RISK_PARAMETER_MAP.items():
        if token in combined:
            return risk_type
    return clean_text(row.get("parameter_name")) or "general_modifier"


def build_toxicity_knowledge(bev_mod_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0
    ordered = bev_mod_df.sort_values(by=["modifier_id"], kind="mergesort")
    for row_idx, row in ordered.iterrows():
        modifier_id = clean_text(row["modifier_id"])
        if not modifier_id:
            continue
        beverage_name = clean_text(row["beverage_name"]) or UNKNOWN
        beverage_id = clean_text(row["beverage_id"]) or UNKNOWN
        category = clean_text(row["category"]) or UNKNOWN
        parameter_name = clean_text(row["parameter_name"]) or UNKNOWN
        compartment = clean_text(row["compartment"]) or UNKNOWN
        modifier = clean_text(row["modifier"]) or UNKNOWN
        reason = clean_text(row["modifier_reason"]) or UNKNOWN
        triggers = clean_text(row["trigger_compounds"]) or UNKNOWN
        source_compound_class = clean_text(row["source_compound_class"]) or UNKNOWN
        risk_type = infer_risk_type(row)

        title = f"{beverage_name} toxicity modifier {modifier_id}"
        content = (
            f"For beverage {beverage_name} ({category}), modifier {modifier_id} contributes to {risk_type} risk. "
            f"It targets parameter {parameter_name} in compartment {compartment} with factor {modifier}. "
            f"Trigger compounds are {triggers} from class {source_compound_class}. "
            f"Causal rationale: {reason}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        metadata = {
            "modifier_id": modifier_id,
            "beverage_id": beverage_id,
            "beverage_name": beverage_name,
            "category": category,
            "parameter_name": parameter_name,
            "compartment": compartment,
            "modifier": modifier,
            "risk_type": risk_type,
            "trigger_compounds": triggers,
            "source_compound_class": source_compound_class,
        }
        provenance = {
            "source_csvs": ["data/processed/pbpk/beverage_effect_modifiers.csv"],
            "source_rows": [str(row_idx)],
            "source_dataset": "beverage_effect_modifiers",
            "source_compound_class": source_compound_class,
        }
        objects.append(
            build_base_object(
                collection="ToxicityKnowledge",
                primary_key=modifier_id,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="beverage_effect_modifiers",
                source_file="data/processed/pbpk/beverage_effect_modifiers.csv",
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def build_population_knowledge(
    human_df: pd.DataFrame,
    pop_mod_df: pd.DataFrame,
    pbpk_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0
    groups = sorted(
        {
            clean_text(v)
            for v in list(human_df["population_group"].tolist()) + list(pop_mod_df["population_group"].tolist()) + list(pbpk_df["population_group"].tolist())
            if clean_text(v)
        }
    )
    for group_name in groups:
        human_rows = human_df[human_df["population_group"] == group_name]
        mod_rows = pop_mod_df[pop_mod_df["population_group"] == group_name]
        pbpk_rows = pbpk_df[pbpk_df["population_group"] == group_name]

        domains = list_unique(human_rows["domain"].tolist())
        effects = list_unique(human_rows["effect_direction"].tolist())
        conditions = list_unique(human_rows["condition"].tolist())
        mod_pairs = sorted(
            {
                f"{clean_text(r['parameter_name'])}:{clean_text(r['modifier'])}"
                for _, r in mod_rows.iterrows()
                if clean_text(r["parameter_name"])
            }
        )
        pbpk_params = list_unique(pbpk_rows["parameter_name"].tolist())

        title = f"{group_name} population physiology and PBPK profile"
        content = (
            f"Population group {group_name} is represented across domains {clip_join(domains, 6)} "
            f"with conditions {clip_join(conditions, 6)} and effect directions {clip_join(effects, 4)}. "
            f"Population modifiers include {clip_join(mod_pairs, 8)}. "
            f"Direct PBPK library parameters for this group include {clip_join(pbpk_params, 6)}."
        )
        if not clean_text(content):
            missing_content_rows += 1

        source_docs = list_unique(human_rows["source_document"].tolist()) + list_unique(mod_rows["source_document"].tolist())
        confidence_values = (
            list(human_rows["confidence_score"].tolist())
            + list(mod_rows["confidence_score"].tolist())
            + list(pbpk_rows["confidence_score"].tolist())
        )
        metadata = {
            "population_group": group_name,
            "domain_count": int(len(domains)),
            "domains": domains,
            "conditions": conditions[:12],
            "effect_directions": effects,
            "population_modifier_count": int(len(mod_rows)),
            "pbpk_parameter_count": int(len(pbpk_rows)),
        }
        provenance = {
            "source_csvs": [
                "data/processed/human/human_metabolism_parameters.csv",
                "data/processed/pbpk/population_modifiers.csv",
                "data/processed/pbpk/pbpk_parameter_library.csv",
            ],
            "human_source_rows": [str(i) for i in human_rows.index.tolist()],
            "modifier_ids": list_unique(mod_rows["modifier_id"].tolist()),
            "source_documents": source_docs,
            "source_parameter_ids": list_unique(mod_rows["source_parameter_id"].tolist() + pbpk_rows["source_parameter_id"].tolist()),
        }
        objects.append(
            build_base_object(
                collection="PopulationKnowledge",
                primary_key=group_name,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="population_merged_knowledge",
                source_file=clip_join(source_docs, 3),
                confidence_score=mean_confidence(confidence_values),
                provenance=provenance,
            )
        )
    return objects, missing_content_rows


def build_scientific_evidence(
    human_df: pd.DataFrame,
    pbpk_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    objects: List[Dict[str, Any]] = []
    missing_content_rows = 0

    human_rows = human_df.sort_values(by=["parameter_id", "source_document", "source_page"], kind="mergesort")
    for idx, row in human_rows.iterrows():
        doc = clean_text(row["source_document"]) or UNKNOWN
        page = clean_text(row["source_page"]) or UNKNOWN
        param_id = clean_text(row["parameter_id"]) or UNKNOWN
        param = clean_text(row["parameter_name"]) or UNKNOWN
        domain = clean_text(row["domain"]) or UNKNOWN
        condition = clean_text(row["condition"]) or UNKNOWN
        unit = clean_text(row["unit"]) or UNKNOWN
        value = clean_text(row["value"]) or UNKNOWN
        evidence_text = clean_text(row["evidence_text"]) or UNKNOWN

        primary = sha1_token(f"human|{doc}|{page}|{param_id}|{idx}")
        title = f"Evidence for {param} from {Path(doc).name if doc != UNKNOWN else 'unknown_source'}"
        content = (
            f"Source {doc} page {page} reports {param} ({domain}) under condition {condition} "
            f"with value {value} {unit}. Evidence excerpt: {evidence_text}."
        )
        if not clean_text(content):
            missing_content_rows += 1
        metadata = {
            "evidence_type": "human_metabolism",
            "parameter_id": param_id,
            "parameter_name": param,
            "domain": domain,
            "condition": condition,
            "value": value,
            "unit": unit,
            "source_page": page,
        }
        provenance = {
            "source_csvs": ["data/processed/human/human_metabolism_parameters.csv"],
            "source_rows": [str(idx)],
            "source_document": doc,
            "source_page": page,
        }
        objects.append(
            build_base_object(
                collection="ScientificEvidence",
                primary_key=primary,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="human_metabolism_parameters",
                source_file=doc,
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )

    pbpk_rows = pbpk_df.sort_values(by=["parameter_id", "source_document"], kind="mergesort")
    for idx, row in pbpk_rows.iterrows():
        doc = clean_text(row["source_document"]) or UNKNOWN
        param_id = clean_text(row["parameter_id"]) or UNKNOWN
        param = clean_text(row["parameter_name"]) or UNKNOWN
        comp = clean_text(row["compartment"]) or UNKNOWN
        reason = clean_text(row["modifier_reason"]) or UNKNOWN
        unit = clean_text(row["unit"]) or UNKNOWN
        value = clean_text(row["base_value"]) or UNKNOWN
        src_param_id = clean_text(row["source_parameter_id"]) or UNKNOWN

        primary = sha1_token(f"pbpk|{doc}|{src_param_id}|{param_id}|{idx}")
        title = f"PBPK evidence for {param}"
        content = (
            f"PBPK source {doc} links parameter {param} (id {param_id}) to compartment {comp} "
            f"with base value {value} {unit}. Parameter rationale: {reason}. "
            f"Source parameter identifier is {src_param_id}."
        )
        if not clean_text(content):
            missing_content_rows += 1
        metadata = {
            "evidence_type": "pbpk_parameterization",
            "parameter_id": param_id,
            "parameter_name": param,
            "compartment": comp,
            "value": value,
            "unit": unit,
            "source_parameter_id": src_param_id,
        }
        provenance = {
            "source_csvs": ["data/processed/pbpk/pbpk_parameter_library.csv"],
            "source_rows": [str(idx)],
            "source_document": doc,
            "source_parameter_id": src_param_id,
        }
        objects.append(
            build_base_object(
                collection="ScientificEvidence",
                primary_key=primary,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset="pbpk_parameter_library",
                source_file=doc,
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )

    matrix_rows = matrix_df.sort_values(by=["beverage_id", "normalized_compound_name", "source_row"], kind="mergesort")
    for idx, row in matrix_rows.iterrows():
        src_file = clean_text(row["source_file"]) or UNKNOWN
        src_row = clean_text(row["source_row"]) or str(idx)
        beverage_id = clean_text(row["beverage_id"]) or UNKNOWN
        beverage_name = clean_text(row["beverage_name"]) or UNKNOWN
        compound = clean_text(row["compound_name"]) or UNKNOWN
        class_name = clean_text(row["source_compound_class"]) or UNKNOWN
        category = clean_text(row["chemical_category"]) or UNKNOWN
        expansion = clean_text(row["expansion_type"]) or UNKNOWN

        primary = sha1_token(f"bevchem|{src_file}|{src_row}|{beverage_id}|{compound}")
        title = f"Chemistry evidence {beverage_name} -> {compound}"
        content = (
            f"Source row {src_row} in {src_file} records beverage {beverage_name} ({beverage_id}) "
            f"containing compound {compound}. Chemical category is {category}, source class is {class_name}, "
            f"and expansion type is {expansion}."
        )
        if not clean_text(content):
            missing_content_rows += 1
        metadata = {
            "evidence_type": "beverage_compound_observation",
            "beverage_id": beverage_id,
            "beverage_name": beverage_name,
            "compound_name": compound,
            "chemical_category": category,
            "source_compound_class": class_name,
            "expansion_type": expansion,
            "source_row": src_row,
        }
        provenance = {
            "source_csvs": ["data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv"],
            "source_rows": [src_row],
            "source_document": src_file,
        }
        objects.append(
            build_base_object(
                collection="ScientificEvidence",
                primary_key=primary,
                title=title,
                content=content,
                metadata=metadata,
                source_dataset=clean_text(row["source_dataset"]) or "beverage_compound_matrix_expanded",
                source_file=src_file,
                confidence_score=parse_confidence(row["confidence_score"]),
                provenance=provenance,
            )
        )

    return objects, missing_content_rows


def semantic_signature(obj: Mapping[str, Any]) -> str:
    collection = clean_text(obj.get("collection"))
    title = re.sub(r"\s+", " ", clean_text(obj.get("title"))).strip().lower()
    content = re.sub(r"\s+", " ", clean_text(obj.get("content"))).strip().lower()
    return sha1_token(f"{collection}|{title}|{content}")


def deduplicate_objects(objects: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    removed = 0
    ordered = sorted(objects, key=lambda x: (clean_text(x.get("collection")), clean_text(x.get("chunk_id"))))
    for obj in ordered:
        sig = semantic_signature(obj)
        chunk_id = clean_text(obj.get("chunk_id"))
        if sig in seen:
            removed += 1
            continue
        seen[sig] = chunk_id
        kept.append(dict(obj))
    return kept, removed


def has_provenance(obj: Mapping[str, Any]) -> bool:
    dataset = clean_text(obj.get("source_dataset"))
    source_file = clean_text(obj.get("source_file"))
    conf = obj.get("confidence_score", None)
    provenance = obj.get("provenance", {})
    has_conf = False
    try:
        has_conf = float(conf) >= 0.0
    except Exception:
        has_conf = False
    return bool(dataset and dataset != UNKNOWN and source_file and has_conf and isinstance(provenance, dict) and len(provenance) > 0)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda x: clean_text(x.get("chunk_id")))
    with path.open("w", encoding=ENCODING) as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def main() -> None:
    configure_logging()
    root = repo_root()
    report_path = report_output_path(root)

    schema_path = root / INPUT_PATHS["weaviate_schema_design"]
    schema_exists, missing_schema_terms = ensure_schema_markdown(schema_path)

    try:
        matrix_df = load_csv(root / INPUT_PATHS["beverage_compound_matrix"])
        beverage_ref_df = load_csv(root / INPUT_PATHS["beverage_reference_table"])
        human_df = load_csv(root / INPUT_PATHS["human_metabolism_parameters"])
        pbpk_df = load_csv(root / INPUT_PATHS["pbpk_parameter_library"])
        pop_mod_df = load_csv(root / INPUT_PATHS["population_modifiers"])
        bev_mod_df = load_csv(root / INPUT_PATHS["beverage_effect_modifiers"])
    except Exception as exc:
        report = {
            "status": "failed",
            "error": str(exc),
            "safe_for_embedding_generation": False,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate materialization report -> %s", report_path)
        return

    objects_by_collection: Dict[str, List[Dict[str, Any]]] = {}
    missing_content_rows = 0

    beverage_objects, missing = build_beverage_knowledge(beverage_ref_df, matrix_df, bev_mod_df)
    objects_by_collection["BeverageKnowledge"] = beverage_objects
    missing_content_rows += missing

    compound_objects, missing = build_compound_knowledge(matrix_df)
    objects_by_collection["CompoundKnowledge"] = compound_objects
    missing_content_rows += missing

    metabolism_objects, missing = build_metabolism_knowledge(human_df)
    objects_by_collection["MetabolismKnowledge"] = metabolism_objects
    missing_content_rows += missing

    pbpk_objects, missing = build_pbpk_knowledge(pbpk_df)
    objects_by_collection["PBPKKnowledge"] = pbpk_objects
    missing_content_rows += missing

    toxicity_objects, missing = build_toxicity_knowledge(bev_mod_df)
    objects_by_collection["ToxicityKnowledge"] = toxicity_objects
    missing_content_rows += missing

    population_objects, missing = build_population_knowledge(human_df, pop_mod_df, pbpk_df)
    objects_by_collection["PopulationKnowledge"] = population_objects
    missing_content_rows += missing

    evidence_objects, missing = build_scientific_evidence(human_df, pbpk_df, matrix_df)
    objects_by_collection["ScientificEvidence"] = evidence_objects
    missing_content_rows += missing

    pre_dedup_count = sum(len(rows) for rows in objects_by_collection.values())
    dedup_removed_total = 0
    deduped_by_collection: Dict[str, List[Dict[str, Any]]] = {}
    for collection in REQUIRED_COLLECTIONS:
        rows = objects_by_collection.get(collection, [])
        deduped_rows, removed = deduplicate_objects(rows)
        deduped_by_collection[collection] = deduped_rows
        dedup_removed_total += removed

    for collection, output_rel in OUTPUT_FILES.items():
        output_path = root / output_rel
        write_jsonl(output_path, deduped_by_collection.get(collection, []))

    objects_per_collection = {
        collection: int(len(deduped_by_collection.get(collection, []))) for collection in REQUIRED_COLLECTIONS
    }
    total_objects_created = int(sum(objects_per_collection.values()))
    all_objects: List[Dict[str, Any]] = []
    for collection in REQUIRED_COLLECTIONS:
        all_objects.extend(deduped_by_collection.get(collection, []))
    provenance_count = sum(1 for obj in all_objects if has_provenance(obj))
    provenance_coverage = round(provenance_count / float(total_objects_created), 6) if total_objects_created > 0 else 0.0

    input_rows_total = len(matrix_df) + len(beverage_ref_df) + len(human_df) + len(pbpk_df) + len(pop_mod_df) + len(bev_mod_df)
    content_coverage = 1.0 if input_rows_total == 0 else max(0.0, 1.0 - (missing_content_rows / float(input_rows_total)))
    dedup_ratio = 0.0 if pre_dedup_count == 0 else (dedup_removed_total / float(pre_dedup_count))
    semantic_materialization_score = round(max(0.0, (0.45 * content_coverage) + (0.45 * provenance_coverage) + (0.10 * (1.0 - dedup_ratio))), 4)

    safe_for_embedding_generation = (
        schema_exists
        and len(missing_schema_terms) == 0
        and all(objects_per_collection.get(collection, 0) > 0 for collection in REQUIRED_COLLECTIONS)
        and missing_content_rows == 0
        and provenance_coverage >= 0.99
        and semantic_materialization_score >= 0.95
    )

    report: Dict[str, Any] = {
        "status": "success",
        "inputs": {key: str(path) for key, path in INPUT_PATHS.items()},
        "schema_validation": {
            "schema_file_exists": schema_exists,
            "missing_required_collection_terms": missing_schema_terms,
        },
        "metrics": {
            "total_objects_created": total_objects_created,
            "objects_per_collection": objects_per_collection,
            "duplicate_objects_removed": int(dedup_removed_total),
            "missing_content_rows": int(missing_content_rows),
            "provenance_coverage": provenance_coverage,
            "semantic_materialization_score": semantic_materialization_score,
        },
        "safe_for_embedding_generation": safe_for_embedding_generation,
        "artifacts": OUTPUT_FILES,
        "reasoning": [
            f"Schema term gaps: {len(missing_schema_terms)}.",
            f"Collections materialized: {len([c for c in REQUIRED_COLLECTIONS if objects_per_collection.get(c, 0) > 0])}/{len(REQUIRED_COLLECTIONS)}.",
            f"Missing semantic content rows: {missing_content_rows}.",
            f"Provenance coverage: {provenance_coverage}.",
            f"Semantic materialization score: {semantic_materialization_score}.",
        ],
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote Weaviate materialization report -> %s", report_path)
    LOGGER.info("safe_for_embedding_generation=%s", safe_for_embedding_generation)


if __name__ == "__main__":
    main()
