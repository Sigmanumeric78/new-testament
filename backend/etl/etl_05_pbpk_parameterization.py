"""ETL step 05: build PBPK simulation-ready parameterization from human and beverage ETL outputs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_05_pbpk_parameterization")

ENCODING = "utf-8"
UNKNOWN = "unknown"

CANONICAL_COMPARTMENTS: Tuple[str, ...] = (
    "stomach",
    "gut",
    "blood",
    "liver",
    "brain",
    "muscle",
    "fat",
    "kidney",
    "elimination",
)

POPULATION_GROUPS: Tuple[str, ...] = (
    "male",
    "female",
    "elderly",
    "young_adult",
    "high_bmi",
    "low_bmi",
    "fasted",
    "fed",
    "liver_impairment",
    "general_population",
)

PARAMETER_LIBRARY_COLUMNS: Tuple[str, ...] = (
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
)

POPULATION_MODIFIER_COLUMNS: Tuple[str, ...] = (
    "modifier_id",
    "population_group",
    "parameter_name",
    "compartment",
    "modifier",
    "modifier_reason",
    "confidence_score",
    "source_document",
    "source_parameter_id",
)

BEVERAGE_MODIFIER_COLUMNS: Tuple[str, ...] = (
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
)

EFFECT_DIRECTION_TO_FACTOR: Mapping[str, float] = {
    "increase": 1.1,
    "decrease": 0.9,
    "neutral": 1.0,
    "modifier": 1.0,
    "variable": 1.0,
    "low": 0.85,
    "primary": 1.0,
    "increase_impairment": 1.15,
    "decrease_bioavailability": 0.9,
}

CONFIDENCE_TEXT_TO_SCORE: Mapping[str, float] = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
    "unknown": 0.6,
}

NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class NumericEvidence:
    value: float
    unit: str
    confidence: float
    source_document: str
    source_parameter_id: str


@dataclass(frozen=True)
class ParameterDefinition:
    parameter_name: str
    compartment: str
    unit: str


PARAMETER_DEFINITIONS: Tuple[ParameterDefinition, ...] = (
    ParameterDefinition("gastric_emptying_rate", "stomach", "1/min"),
    ParameterDefinition("intestinal_absorption_rate", "gut", "1/min"),
    ParameterDefinition("ethanol_distribution_volume", "blood", "L/kg"),
    ParameterDefinition("body_water_fraction", "blood", "fraction"),
    ParameterDefinition("adh_metabolism_rate", "liver", "g/hr"),
    ParameterDefinition("aldh_metabolism_rate", "liver", "g/hr"),
    ParameterDefinition("cyp2e1_modifier", "liver", "ratio"),
    ParameterDefinition("ethanol_elimination_rate", "elimination", "g/hr"),
    ParameterDefinition("acetaldehyde_clearance_rate", "liver", "g/hr"),
    ParameterDefinition("blood_brain_partition", "brain", "ratio"),
    ParameterDefinition("fat_partition_coefficient", "fat", "ratio"),
    ParameterDefinition("first_pass_metabolism", "gut", "ratio"),
    ParameterDefinition("liver_blood_flow", "liver", "ml/min"),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_human_parameters_path(root: Path) -> Path:
    return root / "data" / "processed" / "human" / "human_metabolism_parameters.csv"


def input_beverage_matrix_path(root: Path) -> Path:
    return root / "data" / "processed" / "beverage" / "compound_profiles" / "beverage_compound_matrix_expanded.csv"


def output_parameter_library_path(root: Path) -> Path:
    path = root / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_population_modifiers_path(root: Path) -> Path:
    path = root / "data" / "processed" / "pbpk" / "population_modifiers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_beverage_modifiers_path(root: Path) -> Path:
    path = root / "data" / "processed" / "pbpk" / "beverage_effect_modifiers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "pbpk" / "pbpk_parameterization_report.json"
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
    return clean_text(value).lower()


def parse_confidence(value: Any) -> float:
    text = normalize_key(value)
    if text in CONFIDENCE_TEXT_TO_SCORE:
        return CONFIDENCE_TEXT_TO_SCORE[text]
    try:
        numeric = float(clean_text(value))
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, numeric))


def parse_numeric_value(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    matches = list(NUMERIC_PATTERN.finditer(text))
    if not matches:
        return None
    numeric_values: List[float] = []
    for match in matches:
        token = match.group(0)
        try:
            number = float(token)
        except ValueError:
            continue
        if token.startswith("-") and match.start() > 0 and text[match.start() - 1].isdigit():
            number = abs(number)
        numeric_values.append(number)
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return numeric_values[0]
    return float(mean(numeric_values))


def collect_numeric_evidence(
    df: pd.DataFrame,
    parameter_name: str,
    unit: str = "",
    population_group: str = "",
) -> List[NumericEvidence]:
    subset = df[df["parameter_name"] == parameter_name]
    if unit:
        subset = subset[subset["unit"] == unit]
    if population_group:
        subset = subset[subset["population_group"] == population_group]
    evidence: List[NumericEvidence] = []
    for _, row in subset.iterrows():
        numeric_value = parse_numeric_value(row["value"])
        if numeric_value is None:
            continue
        evidence.append(
            NumericEvidence(
                value=float(numeric_value),
                unit=clean_text(row["unit"]) or UNKNOWN,
                confidence=parse_confidence(row["confidence_score"]),
                source_document=clean_text(row["source_document"]) or UNKNOWN,
                source_parameter_id=clean_text(row["parameter_id"]) or UNKNOWN,
            )
        )
    return evidence


def summarize_evidence(evidence: Sequence[NumericEvidence]) -> Tuple[Optional[float], float, str, str]:
    if not evidence:
        return None, 0.6, UNKNOWN, UNKNOWN
    value = float(mean(item.value for item in evidence))
    confidence = float(mean(item.confidence for item in evidence))
    documents = sorted({item.source_document for item in evidence if item.source_document})
    parameter_ids = sorted({item.source_parameter_id for item in evidence if item.source_parameter_id})
    return value, confidence, "|".join(documents) if documents else UNKNOWN, "|".join(parameter_ids) if parameter_ids else UNKNOWN


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_parameter_library(human_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    rows: List[Dict[str, Any]] = []
    numeric_parameters_used = 0
    qualitative_parameters_derived = 0
    counter = 1

    gastric_general = summarize_evidence(
        collect_numeric_evidence(human_df, "gastric_emptying_rate_constant", population_group="general_population")
    )
    ethanol_elimination = summarize_evidence(
        collect_numeric_evidence(human_df, "ethanol_elimination_rate", unit="g/hr", population_group="general_population")
    )
    liver_flow = summarize_evidence(collect_numeric_evidence(human_df, "liver_blood_flow", unit="ml/min"))
    tbw = summarize_evidence(collect_numeric_evidence(human_df, "total_body_water_volume", unit="L"))
    body_weight = summarize_evidence(collect_numeric_evidence(human_df, "body_weight_reference", unit="kg"))

    tbw_fraction: Optional[float] = None
    tbw_docs = UNKNOWN
    tbw_ids = UNKNOWN
    tbw_confidence = 0.6
    if tbw[0] is not None and body_weight[0] is not None and body_weight[0] > 0:
        tbw_fraction = float(tbw[0] / body_weight[0])
        tbw_confidence = min(tbw[1], body_weight[1])
        tbw_doc_parts = sorted({token for token in (tbw[2] + "|" + body_weight[2]).split("|") if token and token != UNKNOWN})
        tbw_id_parts = sorted({token for token in (tbw[3] + "|" + body_weight[3]).split("|") if token and token != UNKNOWN})
        tbw_docs = "|".join(tbw_doc_parts) if tbw_doc_parts else UNKNOWN
        tbw_ids = "|".join(tbw_id_parts) if tbw_id_parts else UNKNOWN

    base_map: Dict[str, Dict[str, Any]] = {}
    base_map["gastric_emptying_rate"] = {
        "value": gastric_general[0],
        "confidence": gastric_general[1],
        "source_document": gastric_general[2],
        "source_parameter_id": gastric_general[3],
        "modifier": "direct_numeric",
        "reason": "mean_general_population_gastric_emptying_rate_constant",
    }
    base_map["intestinal_absorption_rate"] = {
        "value": gastric_general[0],
        "confidence": max(0.7, gastric_general[1] * 0.9),
        "source_document": gastric_general[2],
        "source_parameter_id": gastric_general[3],
        "modifier": "derived_from_gastric_emptying",
        "reason": "gastric_emptying_is_primary_determinant_of_ethanol_absorption",
    }
    base_map["ethanol_distribution_volume"] = {
        "value": tbw_fraction,
        "confidence": tbw_confidence,
        "source_document": tbw_docs,
        "source_parameter_id": tbw_ids,
        "modifier": "derived_from_total_body_water_over_body_weight",
        "reason": "volume_distribution_proxy_from_reference_total_body_water",
    }
    base_map["body_water_fraction"] = {
        "value": tbw_fraction,
        "confidence": tbw_confidence,
        "source_document": tbw_docs,
        "source_parameter_id": tbw_ids,
        "modifier": "derived_from_total_body_water_over_body_weight",
        "reason": "body_water_fraction_from_reference_total_body_water",
    }
    base_map["adh_metabolism_rate"] = {
        "value": ethanol_elimination[0],
        "confidence": ethanol_elimination[1],
        "source_document": ethanol_elimination[2],
        "source_parameter_id": ethanol_elimination[3],
        "modifier": "proxy_from_ethanol_elimination_rate",
        "reason": "adh_dominant_pathway_proxy_under_reference_elimination_rate",
    }
    if ethanol_elimination[0] is not None:
        aldh_proxy = float(ethanol_elimination[0] * 0.85)
        acaldehyde_proxy = float(ethanol_elimination[0] * 0.8)
    else:
        aldh_proxy = None
        acaldehyde_proxy = None
    base_map["aldh_metabolism_rate"] = {
        "value": aldh_proxy,
        "confidence": 0.7,
        "source_document": ethanol_elimination[2],
        "source_parameter_id": ethanol_elimination[3],
        "modifier": "derived_proxy",
        "reason": "scaled_from_reference_ethanol_elimination_rate_due_to_missing_direct_aldh_numeric",
    }
    base_map["cyp2e1_modifier"] = {
        "value": 1.0,
        "confidence": 0.75,
        "source_document": "data/raw/08_human_metabolism/reviews/nihms445232.pdf",
        "source_parameter_id": "HMP000018",
        "modifier": "baseline_ratio",
        "reason": "baseline_cyp2e1_modifier_before_high_exposure_induction",
    }
    base_map["ethanol_elimination_rate"] = {
        "value": ethanol_elimination[0],
        "confidence": ethanol_elimination[1],
        "source_document": ethanol_elimination[2],
        "source_parameter_id": ethanol_elimination[3],
        "modifier": "direct_numeric",
        "reason": "reference_average_metabolic_rate",
    }
    base_map["acetaldehyde_clearance_rate"] = {
        "value": acaldehyde_proxy,
        "confidence": 0.68,
        "source_document": ethanol_elimination[2],
        "source_parameter_id": ethanol_elimination[3],
        "modifier": "derived_proxy",
        "reason": "scaled_from_ethanol_elimination_rate_due_to_missing_direct_acetaldehyde_clearance_numeric",
    }
    base_map["blood_brain_partition"] = {
        "value": 1.0,
        "confidence": 0.74,
        "source_document": "data/raw/08_human_metabolism/reviews/nihms-402840.pdf",
        "source_parameter_id": "HMP000010",
        "modifier": "qualitative_baseline_ratio",
        "reason": "ethanol_distribution_in_body_water_supports_unit_ratio_baseline",
    }
    base_map["fat_partition_coefficient"] = {
        "value": 1.0,
        "confidence": 0.7,
        "source_document": "data/raw/08_human_metabolism/reviews/nihms445232.pdf",
        "source_parameter_id": "HMP000013",
        "modifier": "qualitative_baseline_ratio",
        "reason": "sex_specific_shift_applied_in_population_modifiers_from_volume_of_distribution_evidence",
    }
    base_map["first_pass_metabolism"] = {
        "value": 1.0,
        "confidence": 0.8,
        "source_document": "data/raw/08_human_metabolism/reviews/nihms-402840.pdf",
        "source_parameter_id": "HMP000052",
        "modifier": "baseline_ratio",
        "reason": "first_pass_present_at_baseline_with_population_and_feeding_modifiers",
    }
    base_map["liver_blood_flow"] = {
        "value": liver_flow[0],
        "confidence": liver_flow[1],
        "source_document": liver_flow[2],
        "source_parameter_id": liver_flow[3],
        "modifier": "direct_numeric",
        "reason": "reference_table_liver_blood_flow",
    }

    for definition in PARAMETER_DEFINITIONS:
        meta = base_map[definition.parameter_name]
        value = meta["value"]
        if isinstance(value, (float, int)):
            numeric_parameters_used += 1
            base_value = format_value(float(value))
        else:
            qualitative_parameters_derived += 1
            base_value = UNKNOWN
        rows.append(
            {
                "parameter_id": f"PBPKP{counter:05d}",
                "parameter_name": definition.parameter_name,
                "compartment": definition.compartment,
                "base_value": base_value,
                "unit": definition.unit,
                "population_group": "general_population",
                "modifier": meta["modifier"],
                "modifier_reason": meta["reason"],
                "confidence_score": f"{float(meta['confidence']):.4f}",
                "source_document": meta["source_document"] or UNKNOWN,
                "source_parameter_id": meta["source_parameter_id"] or UNKNOWN,
            }
        )
        counter += 1

    return pd.DataFrame(rows, columns=list(PARAMETER_LIBRARY_COLUMNS)), numeric_parameters_used, qualitative_parameters_derived


def effect_factor(direction: str) -> float:
    key = normalize_key(direction)
    return EFFECT_DIRECTION_TO_FACTOR.get(key, 1.0)


def source_to_target_parameter(source_name: str) -> List[str]:
    mapping: Mapping[str, Tuple[str, ...]] = {
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
        "aldh_activity": ("aldh_metabolism_rate",),
        "cyp2e1_activity": ("cyp2e1_modifier",),
        "first_pass_metabolism": ("first_pass_metabolism",),
        "stomach_adh_activity": ("first_pass_metabolism",),
        "drug_effect_on_first_pass_metabolism": ("first_pass_metabolism", "intestinal_absorption_rate"),
        "gastric_emptying_first_pass_link": ("first_pass_metabolism",),
        "liver_blood_flow": ("liver_blood_flow",),
        "liver_function_role": ("liver_blood_flow",),
        "liver_function_modifier": ("liver_blood_flow",),
    }
    if source_name in mapping:
        return list(mapping[source_name])
    return []


def parameter_to_compartment(parameter_name: str) -> str:
    for definition in PARAMETER_DEFINITIONS:
        if definition.parameter_name == parameter_name:
            return definition.compartment
    return UNKNOWN


def build_population_modifiers(human_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    counter = 1

    # Start with neutral defaults for full population coverage.
    for group in POPULATION_GROUPS:
        for definition in PARAMETER_DEFINITIONS:
            rows.append(
                {
                    "modifier_id": f"PBPKM{counter:06d}",
                    "population_group": group,
                    "parameter_name": definition.parameter_name,
                    "compartment": definition.compartment,
                    "modifier": "1.0000",
                    "modifier_reason": "default_neutral_modifier_due_to_no_specific_extracted_signal",
                    "confidence_score": "0.5000",
                    "source_document": UNKNOWN,
                    "source_parameter_id": UNKNOWN,
                }
            )
            counter += 1

    row_lookup: Dict[Tuple[str, str], int] = {}
    for idx, row in enumerate(rows):
        row_lookup[(row["population_group"], row["parameter_name"])] = idx

    scoped = human_df[human_df["population_group"].isin(POPULATION_GROUPS)].copy()
    scoped = scoped[scoped["population_group"] != "general_population"]
    for _, source in scoped.iterrows():
        group = clean_text(source["population_group"])
        source_parameter_name = clean_text(source["parameter_name"])
        targets = source_to_target_parameter(source_parameter_name)
        if not targets:
            continue
        factor = effect_factor(clean_text(source["effect_direction"]))
        confidence = parse_confidence(source["confidence_score"])
        for target_parameter in targets:
            key = (group, target_parameter)
            if key not in row_lookup:
                continue
            idx = row_lookup[key]
            existing = rows[idx]
            current_factor = float(existing["modifier"])
            updated_factor = current_factor * factor
            source_ids = [token for token in str(existing["source_parameter_id"]).split("|") if token and token != UNKNOWN]
            source_docs = [token for token in str(existing["source_document"]).split("|") if token and token != UNKNOWN]
            if clean_text(source["parameter_id"]):
                source_ids.append(clean_text(source["parameter_id"]))
            if clean_text(source["source_document"]):
                source_docs.append(clean_text(source["source_document"]))
            reason = (
                f"combined_effect_from_{source_parameter_name}_condition_{clean_text(source['condition']).replace(' ', '_')}"
            )
            rows[idx] = {
                "modifier_id": existing["modifier_id"],
                "population_group": group,
                "parameter_name": target_parameter,
                "compartment": parameter_to_compartment(target_parameter),
                "modifier": f"{updated_factor:.4f}",
                "modifier_reason": reason,
                "confidence_score": f"{max(float(existing['confidence_score']), confidence):.4f}",
                "source_document": "|".join(sorted(set(source_docs))) if source_docs else UNKNOWN,
                "source_parameter_id": "|".join(sorted(set(source_ids))) if source_ids else UNKNOWN,
            }

    return pd.DataFrame(rows, columns=list(POPULATION_MODIFIER_COLUMNS))


def map_beverage_trigger_rows(beverage_group: pd.DataFrame) -> List[Dict[str, Any]]:
    normalized_names = {normalize_key(value) for value in beverage_group["normalized_compound_name"].tolist() if clean_text(value)}
    class_names = {normalize_key(value) for value in beverage_group["source_compound_class"].tolist() if clean_text(value)}
    confidence_values = [parse_confidence(value) for value in beverage_group["confidence_score"].tolist()]
    confidence_score = float(max(confidence_values)) if confidence_values else 0.6

    triggers: List[Dict[str, Any]] = []

    sugar_names = {"glucose", "fructose", "sucrose", "residual sugars"}
    sugar_trigger_names = sorted(name for name in normalized_names if name in sugar_names)
    if "residual sugars" in class_names:
        sugar_trigger_names.append("residual sugars")
    if sugar_trigger_names:
        triggers.append(
            {
                "parameter_name": "intestinal_absorption_rate",
                "compartment": "gut",
                "modifier": "0.9000",
                "modifier_reason": "high_sugar_signal_associated_with_slower_absorption",
                "trigger_compounds": "|".join(sorted(set(sugar_trigger_names))),
                "source_compound_class": "residual sugars",
                "confidence_score": confidence_score,
            }
        )

    carb_trigger_names = sorted(name for name in normalized_names if name == "carbonation")
    if carb_trigger_names:
        triggers.append(
            {
                "parameter_name": "gastric_emptying_rate",
                "compartment": "stomach",
                "modifier": "1.1000",
                "modifier_reason": "carbonation_signal_associated_with_faster_gastric_transition",
                "trigger_compounds": "|".join(carb_trigger_names),
                "source_compound_class": "unknown",
                "confidence_score": confidence_score,
            }
        )

    histamine_trigger_names = sorted(name for name in normalized_names if name == "histamine")
    if histamine_trigger_names:
        triggers.append(
            {
                "parameter_name": "toxicity_response_modifier",
                "compartment": "elimination",
                "modifier": "1.1500",
                "modifier_reason": "histamine_presence_increases_toxicity_sensitivity_signal",
                "trigger_compounds": "|".join(histamine_trigger_names),
                "source_compound_class": "nitrogen compounds",
                "confidence_score": confidence_score,
            }
        )

    sulfite_name_set = {"sulfites", "sulfur dioxide", "potassium metabisulfite"}
    sulfite_trigger_names = sorted(name for name in normalized_names if name in sulfite_name_set)
    if sulfite_trigger_names:
        triggers.append(
            {
                "parameter_name": "sensitivity_modifier",
                "compartment": "elimination",
                "modifier": "1.1000",
                "modifier_reason": "sulfite_signal_associated_with_sensitivity_amplification",
                "trigger_compounds": "|".join(sulfite_trigger_names),
                "source_compound_class": "unknown",
                "confidence_score": confidence_score,
            }
        )

    congener_trigger = "congeners" in class_names or "congeners" in normalized_names
    if congener_trigger:
        triggers.append(
            {
                "parameter_name": "hangover_amplification_modifier",
                "compartment": "elimination",
                "modifier": "1.2000",
                "modifier_reason": "congener_signal_associated_with_hangover_amplification",
                "trigger_compounds": "congeners",
                "source_compound_class": "congeners",
                "confidence_score": confidence_score,
            }
        )

    return triggers


def build_beverage_modifiers(beverage_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    counter = 1
    grouped = beverage_df.groupby(["beverage_id", "beverage_name", "category"], dropna=False, sort=True)
    for (beverage_id, beverage_name, category), group in grouped:
        triggers = map_beverage_trigger_rows(group)
        for trigger in triggers:
            rows.append(
                {
                    "modifier_id": f"PBPKB{counter:07d}",
                    "beverage_id": clean_text(beverage_id),
                    "beverage_name": clean_text(beverage_name),
                    "category": clean_text(category),
                    "parameter_name": trigger["parameter_name"],
                    "compartment": trigger["compartment"],
                    "modifier": trigger["modifier"],
                    "modifier_reason": trigger["modifier_reason"],
                    "trigger_compounds": trigger["trigger_compounds"],
                    "source_compound_class": trigger["source_compound_class"],
                    "confidence_score": f"{float(trigger['confidence_score']):.4f}",
                }
            )
            counter += 1
    return pd.DataFrame(rows, columns=list(BEVERAGE_MODIFIER_COLUMNS))


def build_final_decision(
    parameter_library: pd.DataFrame,
    population_modifiers: pd.DataFrame,
    beverage_modifiers: pd.DataFrame,
) -> Dict[str, Any]:
    required_parameters = {
        "gastric_emptying_rate",
        "intestinal_absorption_rate",
        "ethanol_distribution_volume",
        "body_water_fraction",
        "adh_metabolism_rate",
        "aldh_metabolism_rate",
        "cyp2e1_modifier",
        "ethanol_elimination_rate",
        "acetaldehyde_clearance_rate",
        "blood_brain_partition",
        "fat_partition_coefficient",
    }
    observed_parameters = set(parameter_library["parameter_name"].tolist())
    missing_required = sorted(required_parameters - observed_parameters)

    numeric_base_count = int((parameter_library["base_value"] != UNKNOWN).sum())
    has_population_coverage = set(population_modifiers["population_group"].tolist()) == set(POPULATION_GROUPS)
    safe = (
        len(missing_required) == 0
        and numeric_base_count >= 8
        and has_population_coverage
        and len(beverage_modifiers) > 0
    )

    reasoning: List[str] = [
        f"Canonical PBPK compartments defined: {', '.join(CANONICAL_COMPARTMENTS)}.",
        f"Parameter library rows: {len(parameter_library)}.",
        f"Parameters with numeric base values: {numeric_base_count}.",
        f"Population modifier rows: {len(population_modifiers)} across {len(set(population_modifiers['population_group'].tolist()))} groups.",
        f"Beverage effect modifier rows: {len(beverage_modifiers)}.",
    ]
    if missing_required:
        reasoning.append(f"Missing required PBPK parameters: {', '.join(missing_required)}.")
    if not has_population_coverage:
        reasoning.append("Population modifier coverage is incomplete.")
    reasoning.append(
        "ETL_06 simulation is allowed."
        if safe
        else "ETL_06 simulation is blocked because one or more deterministic parameterization readiness gates failed."
    )

    return {
        "safe_for_etl_06_simulation": safe,
        "missing_required_parameters": missing_required,
        "population_groups_covered": sorted(set(population_modifiers["population_group"].tolist())),
        "reasoning": reasoning,
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    human_path = input_human_parameters_path(root)
    beverage_path = input_beverage_matrix_path(root)
    library_path = output_parameter_library_path(root)
    population_path = output_population_modifiers_path(root)
    beverage_mod_path = output_beverage_modifiers_path(root)
    report_path = output_report_path(root)

    LOGGER.info("Loading human parameter source: %s", human_path)
    human_df = pd.read_csv(human_path, dtype=str, keep_default_na=False, encoding=ENCODING)
    LOGGER.info("Loading beverage compound matrix source: %s", beverage_path)
    beverage_df = pd.read_csv(beverage_path, dtype=str, keep_default_na=False, encoding=ENCODING)
    for column in human_df.columns:
        human_df[column] = human_df[column].map(clean_text)
    for column in beverage_df.columns:
        beverage_df[column] = beverage_df[column].map(clean_text)

    parameter_library_df, numeric_parameters_used, qualitative_parameters_derived = build_parameter_library(human_df)
    population_modifiers_df = build_population_modifiers(human_df)
    beverage_modifiers_df = build_beverage_modifiers(beverage_df)

    parameter_library_df.to_csv(library_path, index=False, encoding=ENCODING)
    population_modifiers_df.to_csv(population_path, index=False, encoding=ENCODING)
    beverage_modifiers_df.to_csv(beverage_mod_path, index=False, encoding=ENCODING)

    decision = build_final_decision(parameter_library_df, population_modifiers_df, beverage_modifiers_df)
    report: Dict[str, Any] = {
        "input_files": {
            "human_parameters": str(human_path.relative_to(root)),
            "beverage_compound_matrix_expanded": str(beverage_path.relative_to(root)),
        },
        "canonical_compartments": list(CANONICAL_COMPARTMENTS),
        "outputs": {
            "pbpk_parameter_library": str(library_path.relative_to(root)),
            "population_modifiers": str(population_path.relative_to(root)),
            "beverage_effect_modifiers": str(beverage_mod_path.relative_to(root)),
        },
        "metrics": {
            "parameters_created": int(len(parameter_library_df)),
            "numeric_parameters_used": int(numeric_parameters_used),
            "qualitative_parameters_derived": int(qualitative_parameters_derived),
            "population_modifier_count": int(len(population_modifiers_df)),
            "beverage_modifier_count": int(len(beverage_modifiers_df)),
        },
        "final_decision": decision,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)

    LOGGER.info("Wrote PBPK parameter library -> %s (rows=%d)", library_path, len(parameter_library_df))
    LOGGER.info("Wrote population modifiers -> %s (rows=%d)", population_path, len(population_modifiers_df))
    LOGGER.info("Wrote beverage effect modifiers -> %s (rows=%d)", beverage_mod_path, len(beverage_modifiers_df))
    LOGGER.info(
        "Wrote PBPK parameterization report -> %s | safe_for_etl_06_simulation=%s",
        report_path,
        decision["safe_for_etl_06_simulation"],
    )


if __name__ == "__main__":
    main()
