"""PBPK V1 simulator using ETL_05 parameterization artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

LOGGER = logging.getLogger("pbpk_master_simulator")

ENCODING = "utf-8"
UNKNOWN = "unknown"

REQUIRED_LIBRARY_PARAMETERS: Tuple[str, ...] = (
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
    "first_pass_metabolism",
    "liver_blood_flow",
)

SIMULATION_CONSTANTS: Mapping[str, float] = {
    "ethanol_density_g_per_ml": 0.789,
    "bac_sober_threshold_percent": 0.02,
    "time_start_h": 0.0,
    "time_end_h": 24.0,
    "time_step_h": 0.05,
    "solve_rtol": 1e-5,
    "solve_atol": 1e-8,
    "denominator_guard": 1e-9,
    "blood_volume_fraction_of_vd": 0.12,
    "liver_volume_l_per_kg": 0.026,
    "brain_volume_l_per_kg": 0.020,
    "muscle_volume_l_per_kg": 0.280,
    "fat_volume_l_per_kg": 0.190,
    "min_blood_volume_l": 3.0,
    "min_liver_volume_l": 1.0,
    "min_brain_volume_l": 1.0,
    "min_muscle_volume_l": 10.0,
    "min_fat_volume_l": 5.0,
    "distribution_scale_from_absorption": 0.6,
    "hepatic_exchange_fraction": 0.06,
    "muscle_distribution_weight": 0.45,
    "acetaldehyde_blood_to_liver_fraction": 0.35,
    "acetaldehyde_liver_to_blood_fraction": 0.55,
    "cyp2e1_relative_capacity": 0.25,
    "ethanol_to_acetaldehyde_mass_ratio": 44.053 / 46.068,
    "first_pass_rate_scale": 0.03,
    "first_pass_to_acetaldehyde_fraction": 0.7,
    "widmark_reference_male": 0.68,
    "widmark_reference_female": 0.55,
    "widmark_reference_body_fat_percent": 20.0,
    "widmark_body_fat_slope_per_percent": 0.003,
    "widmark_min_fraction": 0.40,
    "widmark_max_fraction": 0.80,
}

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

CORE_PARAMETER_NAMES: Tuple[str, ...] = (
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
    "first_pass_metabolism",
    "liver_blood_flow",
)


@dataclass(frozen=True)
class UserProfile:
    sex: str
    weight: float
    height: float
    age: int
    body_fat_percent: float
    fed_or_fasted: str
    liver_status: str


@dataclass(frozen=True)
class DrinkProfile:
    beverage: str
    volume_ml: float
    abv: float
    serving_time: float


@dataclass(frozen=True)
class NormalizedInput:
    user: UserProfile
    drink: DrinkProfile


@dataclass(frozen=True)
class BaseParameter:
    name: str
    value: float
    unit: str
    source_document: str
    source_parameter_id: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "backend").is_dir() and (parent / "data").exists():
            return parent
    for parent in current.parents:
        if (parent / "reasoning").is_dir() and (parent / "simulation").is_dir():
            return parent
    return current.parents[2]


def parameter_library_path(root: Path) -> Path:
    return root / "data" / "processed" / "pbpk" / "pbpk_parameter_library.csv"


def population_modifiers_path(root: Path) -> Path:
    return root / "data" / "processed" / "pbpk" / "population_modifiers.csv"


def beverage_modifiers_path(root: Path) -> Path:
    return root / "data" / "processed" / "pbpk" / "beverage_effect_modifiers.csv"


def default_validation_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "pbpk" / "pbpk_simulation_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def default_calibration_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "pbpk" / "pbpk_calibration_report.json"
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


def parse_float(value: Any, field_name: str) -> float:
    try:
        return float(clean_text(value))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for '{field_name}': {value}") from exc


def parse_int(value: Any, field_name: str) -> int:
    try:
        return int(float(clean_text(value)))
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for '{field_name}': {value}") from exc


def normalize_sex(value: str) -> str:
    lowered = clean_text(value).lower()
    if lowered in {"male", "m"}:
        return "male"
    if lowered in {"female", "f"}:
        return "female"
    raise ValueError("sex must be one of: male, female")


def normalize_fed_state(value: str) -> str:
    lowered = clean_text(value).lower()
    if lowered in {"fed", "with_food"}:
        return "fed"
    if lowered in {"fasted", "empty_stomach"}:
        return "fasted"
    raise ValueError("fed_or_fasted must be one of: fed, fasted")


def normalize_liver_status(value: str) -> str:
    lowered = clean_text(value).lower()
    if lowered in {"healthy", "normal"}:
        return "healthy"
    if lowered in {"liver_impairment", "impaired", "hepatic_severe", "cirrhosis"}:
        return "liver_impairment"
    raise ValueError("liver_status must be one of: healthy, liver_impairment")


def normalize_user_profile(payload: Mapping[str, Any]) -> UserProfile:
    user = UserProfile(
        sex=normalize_sex(clean_text(payload.get("sex"))),
        weight=parse_float(payload.get("weight"), "weight"),
        height=parse_float(payload.get("height"), "height"),
        age=parse_int(payload.get("age"), "age"),
        body_fat_percent=parse_float(payload.get("body_fat_percent"), "body_fat_percent"),
        fed_or_fasted=normalize_fed_state(clean_text(payload.get("fed_or_fasted"))),
        liver_status=normalize_liver_status(clean_text(payload.get("liver_status"))),
    )
    if user.weight <= 0:
        raise ValueError("weight must be > 0")
    if user.height <= 0:
        raise ValueError("height must be > 0")
    if user.age <= 0:
        raise ValueError("age must be > 0")
    if user.body_fat_percent < 0 or user.body_fat_percent > 80:
        raise ValueError("body_fat_percent must be within [0, 80]")
    return user


def normalize_drink_profile(payload: Mapping[str, Any]) -> DrinkProfile:
    drink = DrinkProfile(
        beverage=clean_text(payload.get("beverage")),
        volume_ml=parse_float(payload.get("volume_ml"), "volume_ml"),
        abv=parse_float(payload.get("abv"), "abv"),
        serving_time=parse_float(payload.get("serving_time", 0.0), "serving_time"),
    )
    if not drink.beverage:
        raise ValueError("beverage is required")
    if drink.volume_ml <= 0:
        raise ValueError("volume_ml must be > 0")
    if drink.abv <= 0 or drink.abv > 100:
        raise ValueError("abv must be within (0, 100]")
    if drink.serving_time < 0:
        raise ValueError("serving_time must be >= 0")
    return drink


def normalize_payload(user_payload: Mapping[str, Any], drink_payload: Mapping[str, Any]) -> NormalizedInput:
    return NormalizedInput(
        user=normalize_user_profile(user_payload),
        drink=normalize_drink_profile(drink_payload),
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=ENCODING)


def convert_base_unit(parameter_name: str, value: float, unit: str) -> float:
    normalized_unit = clean_text(unit)
    if parameter_name in {"gastric_emptying_rate", "intestinal_absorption_rate"} and normalized_unit == "1/min":
        return value * 60.0
    if parameter_name == "liver_blood_flow" and normalized_unit == "ml/min":
        return value * 0.06
    return value


def load_base_parameters(library_df: pd.DataFrame) -> Dict[str, BaseParameter]:
    params: Dict[str, BaseParameter] = {}
    for _, row in library_df.iterrows():
        name = clean_text(row.get("parameter_name"))
        value_text = clean_text(row.get("base_value"))
        if not name or value_text in {"", UNKNOWN}:
            continue
        value = parse_float(value_text, f"base_value[{name}]")
        unit = clean_text(row.get("unit"))
        converted_value = convert_base_unit(name, value, unit)
        params[name] = BaseParameter(
            name=name,
            value=converted_value,
            unit=unit,
            source_document=clean_text(row.get("source_document")) or UNKNOWN,
            source_parameter_id=clean_text(row.get("source_parameter_id")) or UNKNOWN,
        )
    missing = [name for name in REQUIRED_LIBRARY_PARAMETERS if name not in params]
    if missing:
        raise ValueError(f"Missing required PBPK base parameters: {', '.join(missing)}")
    return params


def derive_population_groups(user: UserProfile) -> List[str]:
    groups: List[str] = ["general_population", user.sex, user.fed_or_fasted]
    bmi = user.weight / ((user.height / 100.0) ** 2)
    if user.age >= 65:
        groups.append("elderly")
    else:
        groups.append("young_adult")
    if bmi >= 30.0 or user.body_fat_percent >= 30.0:
        groups.append("high_bmi")
    if bmi <= 18.5 or user.body_fat_percent <= 10.0:
        groups.append("low_bmi")
    if user.liver_status == "liver_impairment":
        groups.append("liver_impairment")
    unique_groups = sorted(set(group for group in groups if group in POPULATION_GROUPS))
    return unique_groups


def load_population_modifiers(pop_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for _, row in pop_df.iterrows():
        group = clean_text(row.get("population_group"))
        parameter = clean_text(row.get("parameter_name"))
        if not group or not parameter:
            continue
        factor = parse_float(row.get("modifier"), f"population_modifier[{group}:{parameter}]")
        table.setdefault(group, {})[parameter] = factor
    return table


def select_beverage_modifier_rows(beverage_df: pd.DataFrame, beverage: str) -> pd.DataFrame:
    if beverage_df.empty:
        return beverage_df.copy()
    normalized = clean_text(beverage).lower()
    candidates: List[pd.DataFrame] = []
    by_id = beverage_df[beverage_df["beverage_id"].str.lower() == normalized]
    if not by_id.empty:
        candidates.append(by_id)
    by_name = beverage_df[beverage_df["beverage_name"].str.lower() == normalized]
    if not by_name.empty:
        candidates.append(by_name)
    by_category = beverage_df[beverage_df["category"].str.lower() == normalized]
    if not by_category.empty:
        candidates.append(by_category)

    if not candidates:
        return beverage_df.iloc[0:0].copy()
    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.drop_duplicates(subset=["modifier_id"])
    return combined


def load_beverage_modifiers(beverage_df: pd.DataFrame, beverage: str) -> Dict[str, List[Dict[str, Any]]]:
    selected = select_beverage_modifier_rows(beverage_df, beverage)
    modifier_map: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in selected.iterrows():
        parameter = clean_text(row.get("parameter_name"))
        if not parameter:
            continue
        payload = {
            "modifier_id": clean_text(row.get("modifier_id")) or UNKNOWN,
            "modifier": parse_float(row.get("modifier"), f"beverage_modifier[{parameter}]"),
            "modifier_reason": clean_text(row.get("modifier_reason")) or UNKNOWN,
            "trigger_compounds": clean_text(row.get("trigger_compounds")) or UNKNOWN,
            "source_compound_class": clean_text(row.get("source_compound_class")) or UNKNOWN,
            "confidence_score": parse_float(row.get("confidence_score"), "confidence_score"),
        }
        modifier_map.setdefault(parameter, []).append(payload)
    return modifier_map


def apply_modifiers(
    base_parameters: Mapping[str, BaseParameter],
    population_groups: Sequence[str],
    population_modifier_table: Mapping[str, Mapping[str, float]],
    beverage_modifier_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    effective: Dict[str, float] = {}
    provenance: Dict[str, Any] = {}

    for parameter_name in CORE_PARAMETER_NAMES:
        base = base_parameters[parameter_name]
        value = float(base.value)
        pop_factors: List[Dict[str, Any]] = []
        for group in population_groups:
            factor = float(population_modifier_table.get(group, {}).get(parameter_name, 1.0))
            value *= factor
            pop_factors.append({"group": group, "factor": factor})
        bev_factors: List[Dict[str, Any]] = []
        for item in beverage_modifier_table.get(parameter_name, []):
            factor = float(item["modifier"])
            value *= factor
            bev_factors.append(
                {
                    "modifier_id": item["modifier_id"],
                    "factor": factor,
                    "modifier_reason": item["modifier_reason"],
                    "trigger_compounds": item["trigger_compounds"],
                }
            )
        effective[parameter_name] = value
        provenance[parameter_name] = {
            "base_value": base.value,
            "base_unit": base.unit,
            "source_document": base.source_document,
            "source_parameter_id": base.source_parameter_id,
            "population_factors": pop_factors,
            "beverage_factors": bev_factors,
            "effective_value": value,
        }
    return effective, provenance


def clamp_positive(value: float, minimum: float) -> float:
    return max(minimum, value)


def build_model_parameters(
    user: UserProfile,
    drink: DrinkProfile,
    effective_parameters: Mapping[str, float],
) -> Dict[str, float]:
    c = SIMULATION_CONSTANTS
    weight = float(user.weight)
    body_fat_fraction = float(user.body_fat_percent / 100.0)

    base_body_water_fraction = float(np.clip(effective_parameters["body_water_fraction"], 0.35, 0.80))
    reference_body_water_fraction = 0.587896
    if user.sex == "male":
        sex_ratio = 1.0
        widmark_reference = c["widmark_reference_male"]
    else:
        sex_ratio = c["widmark_reference_female"] / reference_body_water_fraction
        widmark_reference = c["widmark_reference_female"]
    body_water_ratio = base_body_water_fraction / reference_body_water_fraction
    body_fat_delta = user.body_fat_percent - c["widmark_reference_body_fat_percent"]
    body_fat_factor = 1.0 - (body_fat_delta * c["widmark_body_fat_slope_per_percent"])
    widmark_fraction = base_body_water_fraction * sex_ratio * body_water_ratio * body_fat_factor
    widmark_fraction = float(np.clip(widmark_fraction, c["widmark_min_fraction"], c["widmark_max_fraction"]))
    vd_l_per_kg = widmark_fraction
    vd_l = vd_l_per_kg * weight
    body_water_fraction = base_body_water_fraction

    blood_volume_l = clamp_positive(vd_l * c["blood_volume_fraction_of_vd"], c["min_blood_volume_l"])
    liver_volume_l = clamp_positive(weight * c["liver_volume_l_per_kg"], c["min_liver_volume_l"])
    brain_volume_l = clamp_positive(weight * c["brain_volume_l_per_kg"], c["min_brain_volume_l"])
    muscle_volume_l = clamp_positive(
        weight * c["muscle_volume_l_per_kg"] * (1.0 - body_fat_fraction * 0.5),
        c["min_muscle_volume_l"],
    )
    fat_volume_l = clamp_positive(weight * c["fat_volume_l_per_kg"] * (0.7 + body_fat_fraction), c["min_fat_volume_l"])

    k_ge_h = clamp_positive(effective_parameters["gastric_emptying_rate"], 1e-6)
    k_abs_h = clamp_positive(effective_parameters["intestinal_absorption_rate"], 1e-6)
    q_liver_l_h = clamp_positive(effective_parameters["liver_blood_flow"], 1e-6)

    distribution_scale = clamp_positive(k_abs_h * c["distribution_scale_from_absorption"], 1e-6)
    brain_partition = float(np.clip(effective_parameters["blood_brain_partition"], 0.4, 2.5))
    fat_partition = float(np.clip(effective_parameters["fat_partition_coefficient"], 0.4, 3.0))
    muscle_partition = float(np.clip(1.0 + (1.0 - body_fat_fraction) * 0.2, 0.8, 1.3))

    k_blood_to_brain = distribution_scale * brain_partition
    k_brain_to_blood = distribution_scale / brain_partition
    k_blood_to_muscle = distribution_scale * c["muscle_distribution_weight"] * muscle_partition
    k_muscle_to_blood = distribution_scale * c["muscle_distribution_weight"] / muscle_partition
    k_blood_to_fat = distribution_scale * fat_partition
    k_fat_to_blood = distribution_scale / fat_partition

    k_blood_to_liver = (q_liver_l_h / blood_volume_l) * c["hepatic_exchange_fraction"]
    k_liver_to_blood = (q_liver_l_h / liver_volume_l) * c["hepatic_exchange_fraction"]

    first_pass_ratio = float(np.clip(effective_parameters["first_pass_metabolism"], 0.3, 1.7))
    k_first_pass = k_abs_h * c["first_pass_rate_scale"] * first_pass_ratio

    dose_g = drink.volume_ml * (drink.abv / 100.0) * c["ethanol_density_g_per_ml"]

    return {
        "dose_g": dose_g,
        "serving_time_h": drink.serving_time,
        "blood_volume_l": blood_volume_l,
        "liver_volume_l": liver_volume_l,
        "brain_volume_l": brain_volume_l,
        "muscle_volume_l": muscle_volume_l,
        "fat_volume_l": fat_volume_l,
        "vd_l": vd_l,
        "vd_l_per_kg": vd_l_per_kg,
        "widmark_fraction": widmark_fraction,
        "body_water_fraction": body_water_fraction,
        "k_ge_h": k_ge_h,
        "k_abs_h": k_abs_h,
        "k_first_pass_h": k_first_pass,
        "k_blood_to_brain_h": k_blood_to_brain,
        "k_brain_to_blood_h": k_brain_to_blood,
        "k_blood_to_muscle_h": k_blood_to_muscle,
        "k_muscle_to_blood_h": k_muscle_to_blood,
        "k_blood_to_fat_h": k_blood_to_fat,
        "k_fat_to_blood_h": k_fat_to_blood,
        "k_blood_to_liver_h": k_blood_to_liver,
        "k_liver_to_blood_h": k_liver_to_blood,
        "k_adh_g_h": clamp_positive(effective_parameters["adh_metabolism_rate"], 1e-6),
        "k_aldh_g_h": clamp_positive(effective_parameters["aldh_metabolism_rate"], 1e-6),
        "k_cyp2e1_modifier": clamp_positive(effective_parameters["cyp2e1_modifier"], 1e-6),
        "k_elim0_g_h": clamp_positive(effective_parameters["ethanol_elimination_rate"], 1e-6),
        "k_acetaldehyde_clear_g_h": clamp_positive(effective_parameters["acetaldehyde_clearance_rate"], 1e-6),
    }


def state_index() -> Dict[str, int]:
    return {
        "stomach_ethanol": 0,
        "gut_ethanol": 1,
        "blood_ethanol": 2,
        "liver_ethanol": 3,
        "brain_ethanol": 4,
        "muscle_ethanol": 5,
        "fat_ethanol": 6,
        "eliminated_ethanol": 7,
        "liver_acetaldehyde": 8,
        "blood_acetaldehyde": 9,
        "eliminated_acetaldehyde": 10,
    }


def compute_metabolic_rates(y: np.ndarray, p: Mapping[str, float]) -> Dict[str, float]:
    idx = state_index()
    c = SIMULATION_CONSTANTS
    liver_ethanol = max(float(y[idx["liver_ethanol"]]), 0.0)
    liver_acet = max(float(y[idx["liver_acetaldehyde"]]), 0.0)
    blood_acet = max(float(y[idx["blood_acetaldehyde"]]), 0.0)
    gut_ethanol = max(float(y[idx["gut_ethanol"]]), 0.0)

    adh_raw = min(p["k_adh_g_h"], liver_ethanol / c["denominator_guard"])
    cyp_raw = min(
        p["k_adh_g_h"] * p["k_cyp2e1_modifier"] * c["cyp2e1_relative_capacity"],
        liver_ethanol / c["denominator_guard"],
    )
    raw_total = adh_raw + cyp_raw
    cap = min(p["k_elim0_g_h"], liver_ethanol / c["denominator_guard"])
    if raw_total <= 0:
        scale = 0.0
    else:
        scale = cap / raw_total
    r_adh = adh_raw * scale
    r_cyp = cyp_raw * scale
    r_total_ethanol = r_adh + r_cyp

    r_first_pass = p["k_first_pass_h"] * gut_ethanol
    r_acet_prod = (r_total_ethanol + r_first_pass * c["first_pass_to_acetaldehyde_fraction"]) * c[
        "ethanol_to_acetaldehyde_mass_ratio"
    ]

    r_aldh = min(p["k_aldh_g_h"], liver_acet / c["denominator_guard"])
    r_acet_clear = min(p["k_acetaldehyde_clear_g_h"], blood_acet / c["denominator_guard"])
    return {
        "r_adh_g_h": r_adh,
        "r_cyp2e1_g_h": r_cyp,
        "r_ethanol_total_g_h": r_total_ethanol,
        "r_first_pass_g_h": r_first_pass,
        "r_acetaldehyde_production_g_h": r_acet_prod,
        "r_aldh_clearance_g_h": r_aldh,
        "r_acetaldehyde_clearance_g_h": r_acet_clear,
    }


def pbpk_ode_system(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    idx = state_index()
    c = SIMULATION_CONSTANTS
    dy = np.zeros_like(y)

    s = max(float(y[idx["stomach_ethanol"]]), 0.0)
    g = max(float(y[idx["gut_ethanol"]]), 0.0)
    b = max(float(y[idx["blood_ethanol"]]), 0.0)
    l = max(float(y[idx["liver_ethanol"]]), 0.0)
    br = max(float(y[idx["brain_ethanol"]]), 0.0)
    m = max(float(y[idx["muscle_ethanol"]]), 0.0)
    f = max(float(y[idx["fat_ethanol"]]), 0.0)
    la = max(float(y[idx["liver_acetaldehyde"]]), 0.0)
    ba = max(float(y[idx["blood_acetaldehyde"]]), 0.0)

    rates = compute_metabolic_rates(y, p)
    r_ethanol_met = rates["r_ethanol_total_g_h"]
    r_first_pass = rates["r_first_pass_g_h"]
    r_acet_prod = rates["r_acetaldehyde_production_g_h"]
    r_aldh = rates["r_aldh_clearance_g_h"]
    r_acet_clear = rates["r_acetaldehyde_clearance_g_h"]

    flow_stomach_to_gut = p["k_ge_h"] * s
    flow_gut_to_blood = p["k_abs_h"] * g
    flow_blood_to_brain = p["k_blood_to_brain_h"] * b
    flow_brain_to_blood = p["k_brain_to_blood_h"] * br
    flow_blood_to_muscle = p["k_blood_to_muscle_h"] * b
    flow_muscle_to_blood = p["k_muscle_to_blood_h"] * m
    flow_blood_to_fat = p["k_blood_to_fat_h"] * b
    flow_fat_to_blood = p["k_fat_to_blood_h"] * f
    flow_blood_to_liver = p["k_blood_to_liver_h"] * b
    flow_liver_to_blood = p["k_liver_to_blood_h"] * l

    flow_acet_liver_to_blood = p["k_liver_to_blood_h"] * c["acetaldehyde_liver_to_blood_fraction"] * la
    flow_acet_blood_to_liver = p["k_blood_to_liver_h"] * c["acetaldehyde_blood_to_liver_fraction"] * ba

    dy[idx["stomach_ethanol"]] = -flow_stomach_to_gut
    dy[idx["gut_ethanol"]] = flow_stomach_to_gut - flow_gut_to_blood - r_first_pass
    dy[idx["blood_ethanol"]] = (
        flow_gut_to_blood
        + flow_brain_to_blood
        + flow_muscle_to_blood
        + flow_fat_to_blood
        + flow_liver_to_blood
        - flow_blood_to_brain
        - flow_blood_to_muscle
        - flow_blood_to_fat
        - flow_blood_to_liver
    )
    dy[idx["liver_ethanol"]] = flow_blood_to_liver - flow_liver_to_blood - r_ethanol_met
    dy[idx["brain_ethanol"]] = flow_blood_to_brain - flow_brain_to_blood
    dy[idx["muscle_ethanol"]] = flow_blood_to_muscle - flow_muscle_to_blood
    dy[idx["fat_ethanol"]] = flow_blood_to_fat - flow_fat_to_blood
    dy[idx["eliminated_ethanol"]] = r_ethanol_met + r_first_pass

    dy[idx["liver_acetaldehyde"]] = r_acet_prod + flow_acet_blood_to_liver - flow_acet_liver_to_blood - r_aldh
    dy[idx["blood_acetaldehyde"]] = flow_acet_liver_to_blood - flow_acet_blood_to_liver - r_acet_clear
    dy[idx["eliminated_acetaldehyde"]] = r_aldh + r_acet_clear

    return dy


def initial_state(model_parameters: Mapping[str, float]) -> np.ndarray:
    y0 = np.zeros(11, dtype=float)
    y0[state_index()["stomach_ethanol"]] = model_parameters["dose_g"]
    return y0


def build_time_grid() -> np.ndarray:
    c = SIMULATION_CONSTANTS
    return np.arange(c["time_start_h"], c["time_end_h"] + c["time_step_h"], c["time_step_h"])


def simulate(model_parameters: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = build_time_grid()
    c = SIMULATION_CONSTANTS
    solution = solve_ivp(
        pbpk_ode_system,
        (c["time_start_h"], c["time_end_h"]),
        initial_state(model_parameters),
        method="BDF",
        t_eval=t_eval,
        args=(model_parameters,),
        rtol=c["solve_rtol"],
        atol=c["solve_atol"],
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(f"PBPK solve_ivp failed: {solution.message}")
    return solution.t, solution.y


def first_index_at_or_below(values: np.ndarray, threshold: float, start_index: int) -> Optional[int]:
    for index in range(start_index, len(values)):
        if values[index] <= threshold:
            return index
    return None


def summarize_results(
    time_h: np.ndarray,
    y: np.ndarray,
    model_parameters: Mapping[str, float],
    beverage_modifier_table: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_population_groups: Sequence[str],
    effective_parameter_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    idx = state_index()
    blood_ethanol = np.maximum(y[idx["blood_ethanol"]], 0.0)
    liver_ethanol = np.maximum(y[idx["liver_ethanol"]], 0.0)
    brain_ethanol = np.maximum(y[idx["brain_ethanol"]], 0.0)
    muscle_ethanol = np.maximum(y[idx["muscle_ethanol"]], 0.0)
    fat_ethanol = np.maximum(y[idx["fat_ethanol"]], 0.0)
    blood_acet = np.maximum(y[idx["blood_acetaldehyde"]], 0.0)
    blood_volume_l = model_parameters["blood_volume_l"]
    vd_l = model_parameters["vd_l"]
    systemic_ethanol_mass = blood_ethanol + liver_ethanol + brain_ethanol + muscle_ethanol + fat_ethanol
    apparent_ethanol_mg_l = 1000.0 * systemic_ethanol_mass / vd_l
    blood_ethanol_mg_l = 1000.0 * blood_ethanol / blood_volume_l
    bac_percent = apparent_ethanol_mg_l / 10000.0
    blood_acet_mg_l = 1000.0 * blood_acet / blood_volume_l

    peak_index = int(np.argmax(bac_percent))
    time_to_peak_h = float(time_h[peak_index])
    peak_bac_percent = float(bac_percent[peak_index])
    threshold = SIMULATION_CONSTANTS["bac_sober_threshold_percent"]
    sober_index = first_index_at_or_below(bac_percent, threshold, peak_index)
    time_to_sober_h = None if sober_index is None else float(time_h[sober_index])

    rates = [compute_metabolic_rates(y[:, i], model_parameters) for i in range(y.shape[1])]
    metabolism_rate_series = [float(item["r_ethanol_total_g_h"]) for item in rates]
    acetaldehyde_rate_series = [float(item["r_acetaldehyde_production_g_h"]) for item in rates]

    ethanol_burden_auc = float(np.trapezoid(apparent_ethanol_mg_l, time_h))
    acetaldehyde_burden_auc = float(np.trapezoid(blood_acet_mg_l, time_h))

    toxicity_risk_inputs: Dict[str, Any] = {
        "peak_bac_percent": peak_bac_percent,
        "peak_acetaldehyde_mg_l": float(np.max(blood_acet_mg_l)),
        "applied_beverage_modifiers": [],
    }
    for parameter_name, modifiers in beverage_modifier_table.items():
        if "modifier" not in parameter_name:
            continue
        for modifier in modifiers:
            toxicity_risk_inputs["applied_beverage_modifiers"].append(
                {
                    "parameter_name": parameter_name,
                    "modifier_id": modifier["modifier_id"],
                    "factor": modifier["modifier"],
                    "reason": modifier["modifier_reason"],
                    "trigger_compounds": modifier["trigger_compounds"],
                    "source_compound_class": modifier["source_compound_class"],
                    "confidence_score": modifier["confidence_score"],
                }
            )

    return {
        "selected_population_groups": list(selected_population_groups),
        "effective_parameter_provenance": effective_parameter_provenance,
        "model_parameters": {key: float(value) for key, value in model_parameters.items()},
        "bac_curve": [
            {"time_h": float(time_h[i]), "bac_percent": float(bac_percent[i])}
            for i in range(len(time_h))
        ],
        "blood_ethanol_concentration_curve": [
            {"time_h": float(time_h[i]), "blood_ethanol_mg_l": float(blood_ethanol_mg_l[i])}
            for i in range(len(time_h))
        ],
        "peak_bac_percent": peak_bac_percent,
        "time_to_peak_h": time_to_peak_h,
        "time_to_sober_h": time_to_sober_h,
        "acetaldehyde_curve": [
            {"time_h": float(time_h[i]), "acetaldehyde_mg_l": float(blood_acet_mg_l[i])}
            for i in range(len(time_h))
        ],
        "metabolism_rate": [
            {
                "time_h": float(time_h[i]),
                "ethanol_metabolism_rate_g_h": float(metabolism_rate_series[i]),
                "acetaldehyde_production_rate_g_h": float(acetaldehyde_rate_series[i]),
            }
            for i in range(len(time_h))
        ],
        "compound_burden": {
            "ethanol_auc_mg_h_l": ethanol_burden_auc,
            "acetaldehyde_auc_mg_h_l": acetaldehyde_burden_auc,
        },
        "toxicity_risk_inputs": toxicity_risk_inputs,
    }


def run_simulation(
    user_payload: Mapping[str, Any],
    drink_payload: Mapping[str, Any],
    library_df: pd.DataFrame,
    population_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
) -> Dict[str, Any]:
    normalized = normalize_payload(user_payload, drink_payload)
    base_parameters = load_base_parameters(library_df)
    population_groups = derive_population_groups(normalized.user)
    population_modifier_table = load_population_modifiers(population_df)
    beverage_modifier_table = load_beverage_modifiers(beverage_df, normalized.drink.beverage)
    effective_parameters, provenance = apply_modifiers(
        base_parameters=base_parameters,
        population_groups=population_groups,
        population_modifier_table=population_modifier_table,
        beverage_modifier_table=beverage_modifier_table,
    )
    model_parameters = build_model_parameters(normalized.user, normalized.drink, effective_parameters)
    time_h, states = simulate(model_parameters)
    summary = summarize_results(
        time_h=time_h,
        y=states,
        model_parameters=model_parameters,
        beverage_modifier_table=beverage_modifier_table,
        selected_population_groups=population_groups,
        effective_parameter_provenance=provenance,
    )
    return {
        "input_user_profile": normalized.user.__dict__,
        "input_drink_profile": normalized.drink.__dict__,
        "summary": summary,
    }


def estimate_post_peak_bac_elimination_rate(sim_result: Mapping[str, Any], window_h: float = 2.0) -> Optional[float]:
    curve = sim_result["summary"]["bac_curve"]
    if not curve:
        return None
    times = np.array([float(point["time_h"]) for point in curve], dtype=float)
    bac = np.array([float(point["bac_percent"]) for point in curve], dtype=float)
    peak_idx = int(np.argmax(bac))
    if bac[peak_idx] < 0.04:
        return None
    start_t = min(times[-1], times[peak_idx] + 1.0)
    end_t = min(times[-1], start_t + window_h)
    start_idx = int(np.searchsorted(times, start_t, side="left"))
    end_idx = int(np.searchsorted(times, end_t, side="left"))
    if end_idx <= start_idx:
        return None
    dt = float(times[end_idx] - times[start_idx])
    if dt <= 0:
        return None
    rate = float((bac[start_idx] - bac[end_idx]) / dt)
    return max(0.0, rate)


def audit_simulation_chain(sim_result: Mapping[str, Any]) -> Dict[str, Any]:
    user = sim_result["input_user_profile"]
    drink = sim_result["input_drink_profile"]
    summary = sim_result["summary"]
    params = summary["model_parameters"]
    constants = SIMULATION_CONSTANTS
    dose_g = float(params["dose_g"])
    ethanol_density = constants["ethanol_density_g_per_ml"]
    expected_dose_g = float(drink["volume_ml"]) * (float(drink["abv"]) / 100.0) * ethanol_density
    dose_delta = abs(dose_g - expected_dose_g)

    elimination_rate = estimate_post_peak_bac_elimination_rate(sim_result)
    return {
        "ethanol_mass_audit": {
            "formula": "grams_ethanol = volume_ml * (abv/100) * ethanol_density_g_per_ml",
            "volume_ml": float(drink["volume_ml"]),
            "abv_percent": float(drink["abv"]),
            "ethanol_density_g_per_ml": ethanol_density,
            "expected_grams_ethanol": expected_dose_g,
            "model_grams_ethanol": dose_g,
            "absolute_difference_g": dose_delta,
            "passes": dose_delta < 1e-9,
        },
        "distribution_audit": {
            "sex": user["sex"],
            "body_water_fraction": float(params["body_water_fraction"]),
            "widmark_fraction_effective": float(params["widmark_fraction"]),
            "expected_widmark_reference": 0.68 if user["sex"] == "male" else 0.55,
            "ethanol_distribution_volume_l_per_kg": float(params["vd_l_per_kg"]),
            "ethanol_distribution_volume_l": float(params["vd_l"]),
        },
        "bac_conversion_audit": {
            "formula": "BAC_percent = (systemic_ethanol_mass_g / Vd_L) / 10",
            "peak_bac_percent": float(summary["peak_bac_percent"]),
            "time_to_peak_h": float(summary["time_to_peak_h"]),
        },
        "absorption_audit": {
            "gastric_emptying_rate_h": float(params["k_ge_h"]),
            "intestinal_absorption_rate_h": float(params["k_abs_h"]),
            "first_pass_rate_h": float(params["k_first_pass_h"]),
            "fed_or_fasted": user["fed_or_fasted"],
        },
        "elimination_audit": {
            "ethanol_elimination_rate_parameter_g_h": float(params["k_elim0_g_h"]),
            "observed_post_peak_bac_decline_per_h": elimination_rate,
            "realistic_range_bac_per_h": {"min": 0.010, "max": 0.020},
            "within_realistic_range": elimination_rate is not None and 0.010 <= elimination_rate <= 0.020,
        },
    }


def validation_user_profile() -> Dict[str, Any]:
    return {
        "sex": "male",
        "weight": 75.0,
        "height": 178.0,
        "age": 35,
        "body_fat_percent": 20.0,
        "fed_or_fasted": "fed",
        "liver_status": "healthy",
    }


def validation_drink_profile() -> Dict[str, Any]:
    return {
        "beverage": "whisky",
        "volume_ml": 180.0,
        "abv": 40.0,
        "serving_time": 0.0,
    }


def calibration_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "scenario_id": "scenario_1_male_fed_180ml_40abv",
            "description": "75kg male, fed, 180ml whisky, 40% ABV",
            "user": {
                "sex": "male",
                "weight": 75.0,
                "height": 178.0,
                "age": 35,
                "body_fat_percent": 20.0,
                "fed_or_fasted": "fed",
                "liver_status": "healthy",
            },
            "drink": {"beverage": "whisky", "volume_ml": 180.0, "abv": 40.0, "serving_time": 0.0},
            "expected": {"peak_bac_min": 0.08, "peak_bac_max": 0.12},
        },
        {
            "scenario_id": "scenario_2_male_fasted_180ml_40abv",
            "description": "75kg male, fasted, 180ml whisky, 40% ABV",
            "user": {
                "sex": "male",
                "weight": 75.0,
                "height": 178.0,
                "age": 35,
                "body_fat_percent": 20.0,
                "fed_or_fasted": "fasted",
                "liver_status": "healthy",
            },
            "drink": {"beverage": "whisky", "volume_ml": 180.0, "abv": 40.0, "serving_time": 0.0},
            "expected": {"relative_to": "scenario_1_male_fed_180ml_40abv", "higher_and_faster_peak": True},
        },
        {
            "scenario_id": "scenario_3_female_fasted_180ml_40abv",
            "description": "60kg female, fasted, 180ml whisky, 40% ABV",
            "user": {
                "sex": "female",
                "weight": 60.0,
                "height": 165.0,
                "age": 35,
                "body_fat_percent": 28.0,
                "fed_or_fasted": "fasted",
                "liver_status": "healthy",
            },
            "drink": {"beverage": "whisky", "volume_ml": 180.0, "abv": 40.0, "serving_time": 0.0},
            "expected": {"relative_to": "scenario_2_male_fasted_180ml_40abv", "higher_peak_than_reference": True},
        },
        {
            "scenario_id": "scenario_4_male_fed_30ml_40abv",
            "description": "75kg male, fed, 30ml whisky, 40% ABV",
            "user": {
                "sex": "male",
                "weight": 75.0,
                "height": 178.0,
                "age": 35,
                "body_fat_percent": 20.0,
                "fed_or_fasted": "fed",
                "liver_status": "healthy",
            },
            "drink": {"beverage": "whisky", "volume_ml": 30.0, "abv": 40.0, "serving_time": 0.0},
            "expected": {"peak_bac_less_than": 0.04},
        },
    ]


def run_calibration(
    library_df: pd.DataFrame,
    population_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
) -> Dict[str, Any]:
    scenario_results: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    all_pass = True

    for scenario in calibration_scenarios():
        sim_result = run_simulation(
            user_payload=scenario["user"],
            drink_payload=scenario["drink"],
            library_df=library_df,
            population_df=population_df,
            beverage_df=beverage_df,
        )
        audit = audit_simulation_chain(sim_result)
        summary = sim_result["summary"]
        peak = float(summary["peak_bac_percent"])
        t_peak = float(summary["time_to_peak_h"])
        expected = scenario["expected"]
        checks: List[Dict[str, Any]] = []

        if "peak_bac_min" in expected and "peak_bac_max" in expected:
            passes = float(expected["peak_bac_min"]) <= peak <= float(expected["peak_bac_max"])
            checks.append(
                {
                    "check": "peak_bac_within_expected_range",
                    "expected": {"min": expected["peak_bac_min"], "max": expected["peak_bac_max"]},
                    "actual": peak,
                    "passes": passes,
                }
            )

        if "peak_bac_less_than" in expected:
            threshold = float(expected["peak_bac_less_than"])
            passes = peak < threshold
            checks.append(
                {
                    "check": "peak_bac_below_threshold",
                    "expected": {"less_than": threshold},
                    "actual": peak,
                    "passes": passes,
                }
            )

        if expected.get("higher_and_faster_peak"):
            ref = by_id.get(str(expected["relative_to"]))
            if ref is None:
                passes = False
                checks.append(
                    {
                        "check": "higher_and_faster_than_reference",
                        "expected": {"reference": expected["relative_to"]},
                        "actual": "reference_missing",
                        "passes": False,
                    }
                )
            else:
                passes = peak > ref["peak_bac_percent"] and t_peak < ref["time_to_peak_h"]
                checks.append(
                    {
                        "check": "higher_and_faster_than_reference",
                        "expected": {
                            "peak_bac_greater_than": ref["peak_bac_percent"],
                            "time_to_peak_less_than": ref["time_to_peak_h"],
                        },
                        "actual": {"peak_bac_percent": peak, "time_to_peak_h": t_peak},
                        "passes": passes,
                    }
                )

        if expected.get("higher_peak_than_reference"):
            ref = by_id.get(str(expected["relative_to"]))
            if ref is None:
                passes = False
                checks.append(
                    {
                        "check": "higher_peak_than_reference",
                        "expected": {"reference": expected["relative_to"]},
                        "actual": "reference_missing",
                        "passes": False,
                    }
                )
            else:
                passes = peak > ref["peak_bac_percent"]
                checks.append(
                    {
                        "check": "higher_peak_than_reference",
                        "expected": {"peak_bac_greater_than": ref["peak_bac_percent"]},
                        "actual": {"peak_bac_percent": peak},
                        "passes": passes,
                    }
                )

        elimination_observed = audit["elimination_audit"]["observed_post_peak_bac_decline_per_h"]
        elimination_check = audit["elimination_audit"]["within_realistic_range"]
        elimination_applicable = elimination_observed is not None
        checks.append(
            {
                "check": "elimination_rate_within_realistic_bac_decline_range",
                "expected": {"min": 0.010, "max": 0.020},
                "actual": elimination_observed,
                "passes": bool(elimination_check) if elimination_applicable else True,
                "status": "applicable" if elimination_applicable else "not_applicable_low_peak_bac",
            }
        )

        scenario_pass = all(bool(item["passes"]) for item in checks)
        all_pass = all_pass and scenario_pass

        result_row = {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "expected_vs_actual": checks,
            "passes": scenario_pass,
            "actual_summary": {
                "peak_bac_percent": peak,
                "time_to_peak_h": t_peak,
                "time_to_sober_h": summary["time_to_sober_h"],
            },
            "audits": audit,
        }
        scenario_results.append(result_row)
        by_id[scenario["scenario_id"]] = {
            "peak_bac_percent": peak,
            "time_to_peak_h": t_peak,
        }

    return {
        "pbpk_calibrated": all_pass,
        "scenarios": scenario_results,
    }


def validate_result(sim_result: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    summary = sim_result["summary"]
    bac_curve = summary["bac_curve"]
    acet_curve = summary["acetaldehyde_curve"]
    if not bac_curve:
        reasons.append("BAC curve is empty.")
    if not acet_curve:
        reasons.append("Acetaldehyde curve is empty.")
    peak_bac = float(summary["peak_bac_percent"])
    if peak_bac <= 0:
        reasons.append("Peak BAC is non-positive.")
    time_to_peak = float(summary["time_to_peak_h"])
    if time_to_peak < 0:
        reasons.append("time_to_peak_h is negative.")
    compound_burden = summary["compound_burden"]
    if float(compound_burden["ethanol_auc_mg_h_l"]) <= 0:
        reasons.append("Ethanol burden AUC is non-positive.")
    if float(compound_burden["acetaldehyde_auc_mg_h_l"]) < 0:
        reasons.append("Acetaldehyde burden AUC is negative.")
    return len(reasons) == 0, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PBPK V1 master simulator")
    parser.add_argument("--user-profile-json", type=str, default="", help="Path to user profile JSON payload.")
    parser.add_argument("--drink-profile-json", type=str, default="", help="Path to drink profile JSON payload.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path to simulation output JSON. Defaults to data/interim/pbpk/pbpk_simulation_validation.json",
    )
    parser.add_argument(
        "--run-validation-scenario",
        action="store_true",
        help="Run deterministic validation scenario (75kg male, fed, 180ml whisky, 40%% ABV).",
    )
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run deterministic BAC realism calibration scenarios and write pbpk_calibration_report.json.",
    )
    parser.add_argument(
        "--calibration-output-json",
        type=str,
        default="",
        help="Path to calibration output JSON. Defaults to data/interim/pbpk/pbpk_calibration_report.json",
    )
    return parser.parse_args()


def read_json_payload(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"JSON input not found: {path}")
    data = json.loads(path.read_text(encoding=ENCODING))
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return data


def main() -> None:
    configure_logging()
    args = parse_args()
    root = repo_root()

    library_df = load_dataframe(parameter_library_path(root))
    population_df = load_dataframe(population_modifiers_path(root))
    beverage_df = load_dataframe(beverage_modifiers_path(root))
    for frame in (library_df, population_df, beverage_df):
        for column in frame.columns:
            frame[column] = frame[column].map(clean_text)

    if args.run_calibration:
        calibration_report = run_calibration(
            library_df=library_df,
            population_df=population_df,
            beverage_df=beverage_df,
        )
        calibration_output = (
            Path(args.calibration_output_json) if args.calibration_output_json else default_calibration_output_path(root)
        )
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_output.write_text(json.dumps(calibration_report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote PBPK calibration report -> %s", calibration_output)
        LOGGER.info("pbpk_calibrated=%s", calibration_report["pbpk_calibrated"])
        return

    if args.run_validation_scenario:
        user_payload = validation_user_profile()
        drink_payload = validation_drink_profile()
    else:
        if not args.user_profile_json or not args.drink_profile_json:
            raise ValueError(
                "Provide --user-profile-json and --drink-profile-json, or use --run-validation-scenario."
            )
        user_payload = read_json_payload(args.user_profile_json)
        drink_payload = read_json_payload(args.drink_profile_json)

    sim_result = run_simulation(
        user_payload=user_payload,
        drink_payload=drink_payload,
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    validation_pass, reasons = validate_result(sim_result)
    audit = audit_simulation_chain(sim_result)

    output_path = Path(args.output_json) if args.output_json else default_validation_output_path(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "scenario": "validation" if args.run_validation_scenario else "custom",
        "safe_for_phase_6b": validation_pass,
        "validation_fail_reasons": reasons,
        "audit": audit,
        "simulation_result": sim_result,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)

    LOGGER.info("Wrote PBPK simulation validation report -> %s", output_path)
    LOGGER.info("safe_for_phase_6b=%s", validation_pass)


if __name__ == "__main__":
    main()
