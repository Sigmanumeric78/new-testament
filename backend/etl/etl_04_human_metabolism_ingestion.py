"""ETL step 04: ingest structured human metabolism parameters from review PDFs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PDF_BACKEND = "unavailable"
try:
    from pypdf import PdfReader  # type: ignore

    PDF_BACKEND = "pypdf"
except Exception:
    from PyPDF2 import PdfReader  # type: ignore

    PDF_BACKEND = "PyPDF2_fallback"

ENCODING = "utf-8"
UNKNOWN = "unknown"

DOMAIN_ORDER: Tuple[str, ...] = (
    "gastric_emptying",
    "alcohol_absorption",
    "food_effects",
    "body_water_distribution",
    "sex_differences",
    "age_effects",
    "body_mass_effects",
    "lean_body_mass",
    "enzyme_variation",
    "liver_function",
    "ethanol_elimination_rate",
    "bac_kinetics",
    "distribution_volume",
    "metabolic_modifiers",
)

POPULATION_GROUP_ORDER: Tuple[str, ...] = (
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

PBPK_CRITICAL_DOMAINS: Tuple[str, ...] = (
    "gastric_emptying",
    "alcohol_absorption",
    "body_water_distribution",
    "enzyme_variation",
    "ethanol_elimination_rate",
    "bac_kinetics",
    "body_mass_effects",
)

OUTPUT_COLUMNS: Tuple[str, ...] = (
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
)

LOGGER = logging.getLogger("etl_04_human_metabolism_ingestion")


@dataclass(frozen=True)
class PageRecord:
    source_document: str
    source_page: int
    text: str
    flat_text: str


@dataclass(frozen=True)
class QualitativeRule:
    parameter_name: str
    domain: str
    pattern: str
    value: str
    condition: str
    modifier_type: str
    effect_direction: str
    confidence_score: float
    population_group: str = ""
    extract_method: str = "regex_qualitative_sentence"


QUALITATIVE_RULES: Tuple[QualitativeRule, ...] = (
    QualitativeRule(
        parameter_name="peak_bac_same_dose",
        domain="bac_kinetics",
        pattern=r"Women will have higher peak blood alcohol levels than men",
        value="higher_peak_bac",
        condition="same_dose_g_per_kg_body_weight",
        modifier_type="sex_difference",
        effect_direction="increase",
        confidence_score=0.84,
        population_group="female",
    ),
    QualitativeRule(
        parameter_name="body_water_normalized_bac_difference",
        domain="body_water_distribution",
        pattern=r"no differences occur when given the same dose per liter of body water",
        value="no_difference",
        condition="same_dose_per_liter_body_water",
        modifier_type="sex_difference",
        effect_direction="neutral",
        confidence_score=0.83,
    ),
    QualitativeRule(
        parameter_name="first_pass_metabolism",
        domain="sex_differences",
        pattern=r"First pass metabolism of alcohol by the stomach, which may be greater in males",
        value="greater_in_males",
        condition="stomach_first_pass_metabolism",
        modifier_type="sex_difference",
        effect_direction="increase",
        confidence_score=0.82,
        population_group="male",
    ),
    QualitativeRule(
        parameter_name="gastric_emptying_absorption_link",
        domain="alcohol_absorption",
        pattern=r"rate of gastric emptying is an important determinant of the rate of absorption of orally administered alcohol",
        value="important_determinant",
        condition="oral_ethanol",
        modifier_type="physiology_link",
        effect_direction="modifier",
        confidence_score=0.81,
    ),
    QualitativeRule(
        parameter_name="food_effect_on_gastric_emptying",
        domain="food_effects",
        pattern=r"presence of food in the stomach retards gastric emptying",
        value="retards_gastric_emptying",
        condition="food_present_in_stomach",
        modifier_type="food_effect",
        effect_direction="decrease",
        confidence_score=0.9,
        population_group="fed",
    ),
    QualitativeRule(
        parameter_name="food_effect_on_alcohol_absorption",
        domain="alcohol_absorption",
        pattern=r"presence of food in the stomach retards gastric emptying and thus will reduce the absorption of alcohol",
        value="reduces_absorption",
        condition="food_present_in_stomach",
        modifier_type="food_effect",
        effect_direction="decrease",
        confidence_score=0.92,
        population_group="fed",
    ),
    QualitativeRule(
        parameter_name="meal_composition_effect_on_gastric_emptying",
        domain="food_effects",
        pattern=r"Meals high in either fat, or carbohydrate or protein are equally effective in retarding gastric emptying",
        value="meal_types_equally_effective",
        condition="fat_or_carbohydrate_or_protein_meals",
        modifier_type="food_effect",
        effect_direction="decrease",
        confidence_score=0.88,
        population_group="fed",
    ),
    QualitativeRule(
        parameter_name="alcohol_absorption_rate",
        domain="alcohol_absorption",
        pattern=r"The rate of alcohol absorption depends on the rate of gastric emptying.*?more rapid in the fasted state",
        value="more_rapid",
        condition="fasted_state",
        modifier_type="feeding_state",
        effect_direction="increase",
        confidence_score=0.89,
        population_group="fasted",
    ),
    QualitativeRule(
        parameter_name="first_pass_metabolism",
        domain="metabolic_modifiers",
        pattern=r"Some of the alcohol which is ingested orally does not enter the systemic circulation but may be oxidized in the stomach by ADH isoforms",
        value="stomach_oxidation_present",
        condition="oral_ethanol",
        modifier_type="first_pass_metabolism",
        effect_direction="decrease_bioavailability",
        confidence_score=0.86,
    ),
    QualitativeRule(
        parameter_name="first_pass_metabolism",
        domain="metabolic_modifiers",
        pattern=r"This first pass metabolism could modulate alcohol toxicity since its efficiency determines the bioavailability of alcohol",
        value="modulates_bioavailability",
        condition="oral_ethanol",
        modifier_type="first_pass_metabolism",
        effect_direction="modifier",
        confidence_score=0.85,
    ),
    QualitativeRule(
        parameter_name="first_pass_metabolism",
        domain="food_effects",
        pattern=r"This will minimize first pass metabolism and thereby play a role in the higher blood alcohol concentrations observed in the fasted versus the fed state",
        value="minimized_in_fasted_state",
        condition="fasted_vs_fed",
        modifier_type="feeding_state",
        effect_direction="decrease",
        confidence_score=0.88,
        population_group="fasted",
    ),
    QualitativeRule(
        parameter_name="adh_activity",
        domain="enzyme_variation",
        pattern=r"First pass metabolism has been reported to be low in alcoholics, especially in alcoholic women because of decreased ADH activity",
        value="decreased",
        condition="alcoholic_women",
        modifier_type="enzyme_activity",
        effect_direction="decrease",
        confidence_score=0.84,
        population_group="female",
    ),
    QualitativeRule(
        parameter_name="stomach_adh_activity",
        domain="metabolic_modifiers",
        pattern=r"Several drugs, including H2 receptor blockers such as cimetidine or ranitidine, or aspirin inhibit stomach ADH activity",
        value="inhibited",
        condition="h2_blockers_or_aspirin",
        modifier_type="drug_interaction",
        effect_direction="decrease",
        confidence_score=0.87,
    ),
    QualitativeRule(
        parameter_name="drug_effect_on_first_pass_metabolism",
        domain="metabolic_modifiers",
        pattern=r"This will decrease first pass metabolism by the stomach, and hence, increase blood alcohol concentrations",
        value="decreased_first_pass_increases_bac",
        condition="stomach_adh_inhibition",
        modifier_type="drug_interaction",
        effect_direction="increase",
        confidence_score=0.86,
    ),
    QualitativeRule(
        parameter_name="gastric_emptying_first_pass_link",
        domain="metabolic_modifiers",
        pattern=r"The speed of gastric emptying modulates gastric and hepatic first pass metabolism of alcohol",
        value="modulates_first_pass_metabolism",
        condition="oral_ethanol",
        modifier_type="physiology_link",
        effect_direction="modifier",
        confidence_score=0.84,
    ),
    QualitativeRule(
        parameter_name="ethanol_elimination_rate",
        domain="lean_body_mass",
        pattern=r"There is a faster rate of alcohol elimination by women when rates are corrected for lean body mass",
        value="higher_when_corrected_for_lean_body_mass",
        condition="lean_body_mass_normalized",
        modifier_type="sex_difference",
        effect_direction="increase",
        confidence_score=0.89,
        population_group="female",
    ),
    QualitativeRule(
        parameter_name="ethanol_elimination_rate",
        domain="sex_differences",
        pattern=r"Men and women generally have similar alcohol elimination rates when results are expressed as g per hr or g per liter liver volume",
        value="similar_after_normalization",
        condition="g_per_hr_or_liter_liver_volume",
        modifier_type="sex_difference",
        effect_direction="neutral",
        confidence_score=0.87,
    ),
    QualitativeRule(
        parameter_name="ethanol_elimination_rate",
        domain="age_effects",
        pattern=r"There may be a small decline in alcohol elimination with aging",
        value="small_decline",
        condition="aging",
        modifier_type="age_effect",
        effect_direction="decrease",
        confidence_score=0.82,
        population_group="elderly",
    ),
    QualitativeRule(
        parameter_name="liver_function_modifier",
        domain="liver_function",
        pattern=r"Liver mass may explain ethnic and gender differences in alcohol elimination rates",
        value="liver_mass_contributes_to_variation",
        condition="population_variability",
        modifier_type="liver_function",
        effect_direction="modifier",
        confidence_score=0.8,
    ),
    QualitativeRule(
        parameter_name="food_effect_on_ethanol_elimination_rate",
        domain="food_effects",
        pattern=r"The increase in the alcohol elimination rate by food was similar for meals of different compositions",
        value="food_increases_elimination_rate",
        condition="meal_composition_comparison",
        modifier_type="food_effect",
        effect_direction="increase",
        confidence_score=0.86,
        population_group="fed",
    ),
    QualitativeRule(
        parameter_name="adh_activity",
        domain="metabolic_modifiers",
        pattern=r"Agents which inhibit ADH .*? will decrease the alcohol elimination rate",
        value="adh_inhibition_decreases_elimination",
        condition="adh_inhibition_or_competition",
        modifier_type="drug_interaction",
        effect_direction="decrease",
        confidence_score=0.84,
    ),
    QualitativeRule(
        parameter_name="ethanol_distribution_compartment",
        domain="body_water_distribution",
        pattern=r"alcohol is not stored and remains in body water until eliminated",
        value="remains_in_body_water",
        condition="systemic_distribution",
        modifier_type="distribution",
        effect_direction="neutral",
        confidence_score=0.83,
    ),
    QualitativeRule(
        parameter_name="liver_function_role",
        domain="liver_function",
        pattern=r"it seems likely that liver plays the major role in alcohol metabolism",
        value="liver_primary_metabolic_site",
        condition="ethanol_metabolism",
        modifier_type="organ_role",
        effect_direction="primary",
        confidence_score=0.84,
    ),
    QualitativeRule(
        parameter_name="adh_activity",
        domain="enzyme_variation",
        pattern=r"particularly active ADH enzymes, resulting in more rapid conversion of alcohol .*? to acetaldehyde",
        value="particularly_active",
        condition="ADH1B_or_ADH1C_variant",
        modifier_type="genetic_variation",
        effect_direction="increase",
        confidence_score=0.89,
    ),
    QualitativeRule(
        parameter_name="aldh_activity",
        domain="enzyme_variation",
        pattern=r"essentially inactive ALDH enzyme, resulting in acetaldehyde accumulation",
        value="essentially_inactive",
        condition="ALDH2_variant",
        modifier_type="genetic_variation",
        effect_direction="decrease",
        confidence_score=0.9,
    ),
    QualitativeRule(
        parameter_name="cyp2e1_activity",
        domain="enzyme_variation",
        pattern=r"CYP2E1 only is active after a person has consumed large amounts of alcohol",
        value="activated_after_large_amounts",
        condition="high_alcohol_exposure",
        modifier_type="enzyme_induction",
        effect_direction="increase",
        confidence_score=0.88,
    ),
    QualitativeRule(
        parameter_name="catalase_activity",
        domain="enzyme_variation",
        pattern=r"catalase metabolizes only a small fraction of alcohol in the body",
        value="small_fraction",
        condition="baseline_ethanol_metabolism",
        modifier_type="enzyme_contribution",
        effect_direction="low",
        confidence_score=0.87,
    ),
    QualitativeRule(
        parameter_name="volume_of_distribution",
        domain="distribution_volume",
        pattern=r"Women generally have a smaller volume of distribution for alcohol than men",
        value="smaller_in_women",
        condition="sex_comparison",
        modifier_type="sex_difference",
        effect_direction="decrease",
        confidence_score=0.89,
        population_group="female",
    ),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def corpus_root(root: Path) -> Path:
    return root / "data" / "raw" / "08_human_metabolism"


def output_parameters_path(root: Path) -> Path:
    path = root / "data" / "processed" / "human" / "human_metabolism_parameters.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_candidates_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "human_parameter_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "human_metabolism_ingestion_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def collapse_whitespace(value: Any) -> str:
    text = clean_text(value)
    return re.sub(r"\s+", " ", text).strip()


def normalize_numeric(value: str) -> str:
    return collapse_whitespace(value).replace(",", "")


def normalize_range(low: str, high: str) -> str:
    return f"{normalize_numeric(low)}-{normalize_numeric(high)}"


def iter_pdf_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.pdf")):
        if path.is_file():
            yield path


def load_pages(root: Path) -> List[PageRecord]:
    pages: List[PageRecord] = []
    for path in iter_pdf_paths(corpus_root(root)):
        relative_path = str(path.relative_to(root))
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            flat = collapse_whitespace(text)
            if not flat:
                continue
            pages.append(
                PageRecord(
                    source_document=relative_path,
                    source_page=page_number,
                    text=text,
                    flat_text=flat,
                )
            )
    return pages


def evidence_snippet(flat_text: str, start: int, end: int, radius: int = 220) -> str:
    left = max(0, start - radius)
    right = min(len(flat_text), end + radius)
    return collapse_whitespace(flat_text[left:right])


def infer_population_group(evidence_text: str, explicit_group: str = "") -> str:
    if explicit_group:
        return explicit_group
    lowered = evidence_text.lower()
    if "elderly" in lowered or "aging" in lowered or "older adult" in lowered:
        return "elderly"
    if "young" in lowered:
        return "young_adult"
    if "high bmi" in lowered or "obesity" in lowered or "obese" in lowered or "overweight" in lowered:
        return "high_bmi"
    if "low bmi" in lowered or "underweight" in lowered:
        return "low_bmi"
    if "fasted" in lowered or "fasting" in lowered or "empty stomach" in lowered:
        return "fasted"
    if "fed" in lowered or "meal" in lowered or "food" in lowered:
        return "fed"
    if "cirrhosis" in lowered or "liver disease" in lowered or "hepatic" in lowered:
        return "liver_impairment"
    if ("women" in lowered or "female" in lowered) and not ("men" in lowered or "male" in lowered):
        return "female"
    if ("men" in lowered or "male" in lowered) and not ("women" in lowered or "female" in lowered):
        return "male"
    return "general_population"


def build_row(
    parameter_name: str,
    domain: str,
    population_group: str,
    condition: str,
    value: str,
    unit: str,
    modifier_type: str,
    effect_direction: str,
    confidence_score: float,
    evidence_text: str,
    source_document: str,
    source_page: int,
    extract_method: str,
) -> Dict[str, Any]:
    return {
        "parameter_id": "",
        "parameter_name": parameter_name,
        "domain": domain,
        "population_group": population_group or "general_population",
        "condition": condition or UNKNOWN,
        "value": value or UNKNOWN,
        "unit": unit or UNKNOWN,
        "modifier_type": modifier_type or UNKNOWN,
        "effect_direction": effect_direction or UNKNOWN,
        "confidence_score": round(float(confidence_score), 4),
        "evidence_text": evidence_text or UNKNOWN,
        "source_document": source_document,
        "source_page": str(source_page),
        "extract_method": extract_method,
    }


def add_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    rows.append(row)


def extract_total_body_water(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    pattern = re.compile(
        r"(?P<weight>\d+(?:\.\d+)?)\s*[-]?\s*kg man whose total body water content is (?P<tbw>\d+(?:\.\d+)?)\s*l",
        re.IGNORECASE,
    )
    for match in pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        add_row(
            rows,
            build_row(
                parameter_name="body_weight_reference",
                domain="body_mass_effects",
                population_group="male",
                condition="standard_man_reference",
                value=normalize_numeric(match.group("weight")),
                unit="kg",
                modifier_type="reference_anthropometry",
                effect_direction="neutral",
                confidence_score=0.95,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )
        add_row(
            rows,
            build_row(
                parameter_name="total_body_water_volume",
                domain="body_water_distribution",
                population_group="male",
                condition=f"standard_man_{normalize_numeric(match.group('weight'))}_kg",
                value=normalize_numeric(match.group("tbw")),
                unit="L",
                modifier_type="reference_anthropometry",
                effect_direction="neutral",
                confidence_score=0.95,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )


def extract_liver_table_metrics(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    pattern = re.compile(
        r"Liver\s+Liver\s+(?P<water>\d+(?:\.\d+)?)\s+(?P<flow>[\d,]+)\s+(?P<perfusion>\d+(?:\.\d+)?)\s+(?P<residence>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end(), radius=140)
        add_row(
            rows,
            build_row(
                parameter_name="liver_water_volume",
                domain="liver_function",
                population_group="male",
                condition="standard_man_table_reference",
                value=normalize_numeric(match.group("water")),
                unit="L",
                modifier_type="reference_anthropometry",
                effect_direction="neutral",
                confidence_score=0.93,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_table",
            ),
        )
        add_row(
            rows,
            build_row(
                parameter_name="liver_blood_flow",
                domain="liver_function",
                population_group="male",
                condition="standard_man_table_reference",
                value=normalize_numeric(match.group("flow")),
                unit="ml/min",
                modifier_type="reference_physiology",
                effect_direction="neutral",
                confidence_score=0.93,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_table",
            ),
        )


def extract_stomach_emptying_table(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    pattern = re.compile(
        r"Stomach-emptying rate constants.*?Ethanol \(g/kg\).*?0\.15\s+0\.3\s+0\.45\s+0\.6\s+Current work,\s*k\s*S\s*(?P<c1>\d+\.\d+)\s+(?P<c2>\d+\.\d+)\s+(?P<c3>\d+\.\d+)\s+(?P<c4>\d+\.\d+)\s+Wilkinson et al\.[a-z,]*kS\s*(?P<w1>\d+\.\d+)\s+(?P<w2>\d+\.\d+)\s+(?P<w3>\d+\.\d+)\s+(?P<w4>\d+\.\d+)",
        re.IGNORECASE,
    )
    dose_keys = (("0.15", "c1"), ("0.3", "c2"), ("0.45", "c3"), ("0.6", "c4"))
    wilkinson_keys = (("0.15", "w1"), ("0.3", "w2"), ("0.45", "w3"), ("0.6", "w4"))
    for match in pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end(), radius=80)
        for dose, group_key in dose_keys:
            add_row(
                rows,
                build_row(
                    parameter_name="gastric_emptying_rate_constant",
                    domain="gastric_emptying",
                    population_group="general_population",
                    condition=f"ethanol_dose_g_per_kg={dose};source=current_work",
                    value=normalize_numeric(match.group(group_key)),
                    unit=UNKNOWN,
                    modifier_type="dose_dependent_rate_constant",
                    effect_direction="neutral",
                    confidence_score=0.9,
                    evidence_text=evidence,
                    source_document=page.source_document,
                    source_page=page.source_page,
                    extract_method="regex_numeric_table",
                ),
            )
        for dose, group_key in wilkinson_keys:
            add_row(
                rows,
                build_row(
                    parameter_name="gastric_emptying_rate_constant",
                    domain="gastric_emptying",
                    population_group="fasted",
                    condition=f"ethanol_dose_g_per_kg={dose};source=wilkinson_et_al",
                    value=normalize_numeric(match.group(group_key)),
                    unit=UNKNOWN,
                    modifier_type="dose_dependent_rate_constant",
                    effect_direction="neutral",
                    confidence_score=0.9,
                    evidence_text=evidence,
                    source_document=page.source_document,
                    source_page=page.source_page,
                    extract_method="regex_numeric_table",
                ),
            )


def extract_elimination_numeric(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    range_pattern = re.compile(
        r"about (?P<low>\d+)\s+to\s+(?P<high>\d+)\s+g per day for a person with a body weight of (?P<weight>\d+)\s*kg",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        add_row(
            rows,
            build_row(
                parameter_name="ethanol_elimination_capacity",
                domain="ethanol_elimination_rate",
                population_group="general_population",
                condition=f"body_weight={normalize_numeric(match.group('weight'))}_kg_reference",
                value=normalize_range(match.group("low"), match.group("high")),
                unit="g/day",
                modifier_type="reference_capacity",
                effect_direction="neutral",
                confidence_score=0.94,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )

    rate_pattern = re.compile(r"about (?P<value>\d+(?:\.\d+)?)\s*g/hr", re.IGNORECASE)
    for match in rate_pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        if "drink per hr" not in evidence.lower() and "metabolic rate" not in evidence.lower():
            continue
        add_row(
            rows,
            build_row(
                parameter_name="ethanol_elimination_rate",
                domain="ethanol_elimination_rate",
                population_group="general_population",
                condition="average_metabolic_rate_reference",
                value=normalize_numeric(match.group("value")),
                unit="g/hr",
                modifier_type="reference_rate",
                effect_direction="neutral",
                confidence_score=0.94,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )

    variability_pattern = re.compile(r"(?P<low>\d+)\s*[–-]\s*(?P<high>\d+)\s*fold variability in the rate of alcohol elimination", re.IGNORECASE)
    for match in variability_pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        add_row(
            rows,
            build_row(
                parameter_name="ethanol_elimination_variability",
                domain="ethanol_elimination_rate",
                population_group="general_population",
                condition="population_variability",
                value=normalize_range(match.group("low"), match.group("high")),
                unit="fold",
                modifier_type="interindividual_variation",
                effect_direction="variable",
                confidence_score=0.9,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )

    four_fold_pattern = re.compile(r"actual ethanol elimination rates have been shown to differ about (?P<value>\d+)-fold", re.IGNORECASE)
    for match in four_fold_pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        add_row(
            rows,
            build_row(
                parameter_name="ethanol_elimination_variability",
                domain="ethanol_elimination_rate",
                population_group="young_adult",
                condition="healthy_young_european_americans",
                value=normalize_numeric(match.group("value")),
                unit="fold",
                modifier_type="interindividual_variation",
                effect_direction="variable",
                confidence_score=0.88,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )


def extract_bac_numeric(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    pattern = re.compile(r"BAC as low as (?P<value>\d+(?:\.\d+)?)%", re.IGNORECASE)
    for match in pattern.finditer(page.flat_text):
        evidence = evidence_snippet(page.flat_text, match.start(), match.end())
        add_row(
            rows,
            build_row(
                parameter_name="residual_bac_impairment_threshold",
                domain="bac_kinetics",
                population_group="general_population",
                condition="complex_task_impairment",
                value=normalize_numeric(match.group("value")),
                unit="%",
                modifier_type="impairment_threshold",
                effect_direction="increase_impairment",
                confidence_score=0.93,
                evidence_text=evidence,
                source_document=page.source_document,
                source_page=page.source_page,
                extract_method="regex_numeric_sentence",
            ),
        )


def apply_qualitative_rules(page: PageRecord, rows: List[Dict[str, Any]]) -> None:
    for rule in QUALITATIVE_RULES:
        pattern = re.compile(rule.pattern, re.IGNORECASE)
        for match in pattern.finditer(page.flat_text):
            evidence = evidence_snippet(page.flat_text, match.start(), match.end())
            add_row(
                rows,
                build_row(
                    parameter_name=rule.parameter_name,
                    domain=rule.domain,
                    population_group=infer_population_group(evidence, rule.population_group),
                    condition=rule.condition,
                    value=rule.value,
                    unit=UNKNOWN,
                    modifier_type=rule.modifier_type,
                    effect_direction=rule.effect_direction,
                    confidence_score=rule.confidence_score,
                    evidence_text=evidence,
                    source_document=page.source_document,
                    source_page=page.source_page,
                    extract_method=rule.extract_method,
                ),
            )


def extract_parameters(pages: Sequence[PageRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in pages:
        extract_total_body_water(page, rows)
        extract_liver_table_metrics(page, rows)
        extract_stomach_emptying_table(page, rows)
        extract_elimination_numeric(page, rows)
        extract_bac_numeric(page, rows)
        apply_qualitative_rules(page, rows)
    return rows


def row_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        clean_text(row.get("domain", "")),
        clean_text(row.get("parameter_name", "")),
        clean_text(row.get("population_group", "")),
        clean_text(row.get("condition", "")),
        clean_text(row.get("value", "")),
        clean_text(row.get("source_document", "")),
        int(clean_text(row.get("source_page", "0")) or "0"),
        clean_text(row.get("evidence_text", "")),
    )


def deduplicate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[Any, ...]] = set()
    output: List[Dict[str, Any]] = []
    for row in sorted((dict(item) for item in rows), key=row_sort_key):
        key = tuple(
            clean_text(row.get(column, "")) for column in OUTPUT_COLUMNS if column != "parameter_id"
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def assign_parameter_ids(rows: Sequence[Mapping[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for index, row in enumerate(sorted((dict(item) for item in rows), key=row_sort_key), start=1):
        row["parameter_id"] = f"{prefix}{index:06d}"
        output.append(row)
    return output


def is_numeric_value(value: str, unit: str, extract_method: str) -> bool:
    if "numeric" not in extract_method:
        return False
    return bool(re.search(r"\d", value))


def compute_domain_coverage(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        subset = [row for row in rows if clean_text(row.get("domain", "")) == domain]
        coverage[domain] = {
            "count": len(subset),
            "numeric_count": sum(1 for row in subset if is_numeric_value(clean_text(row.get("value", "")), clean_text(row.get("unit", "")), clean_text(row.get("extract_method", "")))),
            "qualitative_count": sum(1 for row in subset if not is_numeric_value(clean_text(row.get("value", "")), clean_text(row.get("unit", "")), clean_text(row.get("extract_method", "")))),
            "covered": len(subset) > 0,
        }
    return coverage


def compute_population_coverage(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {group: 0 for group in POPULATION_GROUP_ORDER}
    for row in rows:
        group = clean_text(row.get("population_group", "")) or "general_population"
        counts[group] = counts.get(group, 0) + 1
    return counts


def final_decision(
    documents_processed: int,
    parameters: Sequence[Mapping[str, Any]],
    domain_coverage: Mapping[str, Mapping[str, Any]],
    numeric_parameters: int,
) -> Dict[str, Any]:
    missing_domains = sorted(domain for domain, meta in domain_coverage.items() if not meta["covered"])
    pbpk_missing = sorted(domain for domain in PBPK_CRITICAL_DOMAINS if not domain_coverage[domain]["covered"])
    safe = (
        documents_processed >= 4
        and len(parameters) >= 15
        and numeric_parameters >= 8
        and len(missing_domains) == 0
        and len(pbpk_missing) == 0
    )
    reasoning: List[str] = [
        f"Documents processed: {documents_processed}.",
        f"Structured parameters extracted: {len(parameters)}.",
        f"Numeric parameters extracted: {numeric_parameters}.",
        f"Covered domains: {len([domain for domain, meta in domain_coverage.items() if meta['covered']])} of {len(DOMAIN_ORDER)}.",
    ]
    if missing_domains:
        reasoning.append("Missing extracted domains: " + ", ".join(missing_domains) + ".")
    if pbpk_missing:
        reasoning.append("PBPK-critical extracted domains missing: " + ", ".join(pbpk_missing) + ".")
    if safe:
        reasoning.append("ETL_05 is allowed because document count, parameter count, numeric evidence count, and domain coverage thresholds are satisfied.")
    else:
        reasoning.append("ETL_05 is blocked because one or more deterministic ingestion thresholds were not satisfied.")
    return {
        "safe_for_etl_05": safe,
        "missing_domains": missing_domains,
        "pbpk_critical_missing_domains": pbpk_missing,
        "reasoning": reasoning,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(list(rows), columns=list(OUTPUT_COLUMNS))
    df.to_csv(path, index=False, encoding=ENCODING)


def main() -> None:
    configure_logging()
    root = repo_root()
    corpus = corpus_root(root)
    parameters_path = output_parameters_path(root)
    candidates_path = output_candidates_path(root)
    report_path = output_report_path(root)

    if not corpus.exists():
        raise FileNotFoundError(f"Missing human metabolism corpus directory: {corpus}")

    LOGGER.info("Loading human metabolism PDFs from %s", corpus)
    pages = load_pages(root)
    documents_processed = len({page.source_document for page in pages})
    if documents_processed == 0:
        raise RuntimeError("No PDF documents were processed from the human metabolism corpus.")

    candidate_rows = extract_parameters(pages)
    candidate_rows = assign_parameter_ids(candidate_rows, prefix="HMPCAND")
    parameter_rows = deduplicate_rows(candidate_rows)
    parameter_rows = assign_parameter_ids(parameter_rows, prefix="HMP")

    write_csv(candidates_path, candidate_rows)
    write_csv(parameters_path, parameter_rows)

    numeric_parameters = sum(
        1
        for row in parameter_rows
        if is_numeric_value(clean_text(row.get("value", "")), clean_text(row.get("unit", "")), clean_text(row.get("extract_method", "")))
    )
    qualitative_parameters = len(parameter_rows) - numeric_parameters
    domain_coverage = compute_domain_coverage(parameter_rows)
    population_group_coverage = compute_population_coverage(parameter_rows)
    decision = final_decision(
        documents_processed=documents_processed,
        parameters=parameter_rows,
        domain_coverage=domain_coverage,
        numeric_parameters=numeric_parameters,
    )

    report = {
        "metadata": {
            "script": "etl/etl_04_human_metabolism_ingestion.py",
            "corpus_root": str(corpus.relative_to(root)),
            "pdf_backend": PDF_BACKEND,
            "documents_processed": documents_processed,
        },
        "metrics": {
            "documents_processed": documents_processed,
            "parameters_extracted": len(parameter_rows),
            "numeric_parameters": numeric_parameters,
            "qualitative_parameters": qualitative_parameters,
            "domain_coverage": domain_coverage,
            "population_group_coverage": population_group_coverage,
        },
        "artifacts": {
            "human_metabolism_parameters_csv": str(parameters_path.relative_to(root)),
            "human_parameter_candidates_csv": str(candidates_path.relative_to(root)),
            "human_metabolism_ingestion_report_json": str(report_path.relative_to(root)),
        },
        "final_decision": decision,
    }

    with report_path.open("w", encoding=ENCODING) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info("Wrote candidate parameter rows=%d -> %s", len(candidate_rows), candidates_path)
    LOGGER.info("Wrote canonical parameter rows=%d -> %s", len(parameter_rows), parameters_path)
    LOGGER.info("Wrote ingestion report -> %s | safe_for_etl_05=%s", report_path, decision["safe_for_etl_05"])


if __name__ == "__main__":
    main()
