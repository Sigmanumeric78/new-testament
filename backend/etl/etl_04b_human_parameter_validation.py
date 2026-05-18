"""ETL step 04b: validate extracted human metabolism parameters for PBPK readiness."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_04b_human_parameter_validation")

ENCODING = "utf-8"
UNKNOWN = "unknown"

REQUIRED_COLUMNS: Tuple[str, ...] = (
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

ALLOWED_POPULATION_GROUPS: Tuple[str, ...] = (
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
ALLOWED_POPULATION_SET: Set[str] = set(ALLOWED_POPULATION_GROUPS)

PARAMETER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
POPULATION_GROUP_PATTERN = re.compile(r"^[a-z]+(?:_[a-z]+)*$")
NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

ALLOWED_UNITS: Set[str] = {
    UNKNOWN,
    "%",
    "L",
    "kg",
    "g/hr",
    "g/day",
    "fold",
    "ml/min",
    "BAC/hour",
    "%/hour",
    "hr",
    "min",
    "L/hr",
    "L/kg",
    "dimensionless",
    "ratio",
}

CANONICAL_PARAMETER_EXAMPLES: Tuple[str, ...] = (
    "gastric_emptying_rate",
    "widmark_factor",
    "total_body_water_percent",
    "adh_activity",
    "aldh_activity",
    "cyp2e1_activity",
    "ethanol_elimination_rate",
    "bac_peak_delay",
    "first_pass_metabolism",
    "liver_blood_flow",
    "volume_of_distribution",
)

PBPK_CORE_PARAMETERS: Tuple[str, ...] = (
    "gastric_emptying_rate_constant",
    "ethanol_elimination_rate",
    "liver_blood_flow",
    "volume_of_distribution",
    "first_pass_metabolism",
)


@dataclass(frozen=True)
class Issue:
    issue_code: str
    severity: str
    detail: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_parameters_path(root: Path) -> Path:
    return root / "data" / "processed" / "human" / "human_metabolism_parameters.csv"


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "human_parameter_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_suspicious_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "suspicious_human_parameters.csv"
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


def normalize_param_name(name: str) -> str:
    text = clean_text(name).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def extract_numeric_values(value: str) -> List[float]:
    numbers: List[float] = []
    text = clean_text(value)
    for match in NUMERIC_PATTERN.finditer(text):
        token = match.group(0)
        try:
            numeric_value = float(token)
        except ValueError:
            continue
        # Interpret "x-y" as a range delimiter, not a negative value.
        if token.startswith("-") and match.start() > 0 and text[match.start() - 1].isdigit():
            numeric_value = abs(numeric_value)
        numbers.append(numeric_value)
    return numbers


def add_row_issue(
    row_issues: MutableMapping[int, List[Issue]],
    row_index: int,
    issue_code: str,
    severity: str,
    detail: str,
) -> None:
    row_issues.setdefault(row_index, []).append(Issue(issue_code=issue_code, severity=severity, detail=detail))


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def validate_parameter_names(df: pd.DataFrame, row_issues: MutableMapping[int, List[Issue]]) -> Dict[str, Any]:
    malformed: Set[str] = set()
    for name in sorted(set(df["parameter_name"].astype(str).tolist())):
        if not PARAMETER_NAME_PATTERN.match(name):
            malformed.add(name)

    normalized_groups: Dict[str, Set[str]] = {}
    for name in sorted(set(df["parameter_name"].astype(str).tolist())):
        normalized_groups.setdefault(normalize_param_name(name), set()).add(name)
    inconsistent_sets = [sorted(list(names)) for names in normalized_groups.values() if len(names) > 1]

    duplicate_id_count = int(df["parameter_id"].duplicated(keep=False).sum())
    duplicate_record_count = int(
        df.duplicated(
            subset=["parameter_name", "domain", "population_group", "condition", "value", "unit", "source_document", "source_page"],
            keep=False,
        ).sum()
    )

    for idx, row in df.iterrows():
        name = clean_text(row["parameter_name"])
        if name in malformed:
            add_row_issue(
                row_issues,
                idx,
                "malformed_parameter_name",
                "warning",
                f"Parameter name '{name}' is not lowercase snake_case.",
            )

    for idx in df[df["parameter_id"].duplicated(keep=False)].index.tolist():
        add_row_issue(
            row_issues,
            int(idx),
            "duplicate_parameter_id",
            "critical",
            "Duplicate parameter_id detected.",
        )

    for idx in df[
        df.duplicated(
            subset=["parameter_name", "domain", "population_group", "condition", "value", "unit", "source_document", "source_page"],
            keep=False,
        )
    ].index.tolist():
        add_row_issue(
            row_issues,
            int(idx),
            "duplicate_parameter_record",
            "warning",
            "Duplicate parameter record detected for same source and context.",
        )

    observed_names: Set[str] = set(df["parameter_name"].astype(str).tolist())
    canonical_present = [name for name in CANONICAL_PARAMETER_EXAMPLES if name in observed_names]
    canonical_missing = [name for name in CANONICAL_PARAMETER_EXAMPLES if name not in observed_names]

    return {
        "malformed_parameter_names": sorted(malformed),
        "inconsistent_naming_groups": inconsistent_sets,
        "duplicate_parameter_id_count": duplicate_id_count,
        "duplicate_parameter_record_count": duplicate_record_count,
        "canonical_parameter_examples_present": canonical_present,
        "canonical_parameter_examples_missing": canonical_missing,
    }


def validate_population_groups(df: pd.DataFrame, row_issues: MutableMapping[int, List[Issue]]) -> Dict[str, Any]:
    unknown_groups: Set[str] = set()
    malformed_groups: Set[str] = set()
    for idx, row in df.iterrows():
        group = clean_text(row["population_group"])
        if not group:
            add_row_issue(row_issues, int(idx), "empty_population_group", "warning", "Population group is empty.")
            continue
        if not POPULATION_GROUP_PATTERN.match(group):
            malformed_groups.add(group)
            add_row_issue(
                row_issues,
                int(idx),
                "malformed_population_group",
                "critical",
                f"Malformed population group '{group}'.",
            )
        if group not in ALLOWED_POPULATION_SET:
            unknown_groups.add(group)
            add_row_issue(
                row_issues,
                int(idx),
                "unknown_population_group",
                "critical",
                f"Unknown population group '{group}'.",
            )

    return {
        "unknown_population_groups": sorted(unknown_groups),
        "malformed_population_groups": sorted(malformed_groups),
    }


def validate_confidence_scores(df: pd.DataFrame, row_issues: MutableMapping[int, List[Issue]]) -> Dict[str, Any]:
    non_numeric_count = 0
    out_of_range_count = 0
    for idx, row in df.iterrows():
        value = clean_text(row["confidence_score"])
        try:
            score = float(value)
        except ValueError:
            non_numeric_count += 1
            add_row_issue(
                row_issues,
                int(idx),
                "confidence_non_numeric",
                "critical",
                f"Confidence score '{value}' is not numeric.",
            )
            continue
        if score < 0.0 or score > 1.0:
            out_of_range_count += 1
            add_row_issue(
                row_issues,
                int(idx),
                "confidence_out_of_range",
                "critical",
                f"Confidence score {score} is outside [0,1].",
            )

    return {
        "confidence_non_numeric_count": non_numeric_count,
        "confidence_out_of_range_count": out_of_range_count,
    }


def validate_numeric_values(df: pd.DataFrame, row_issues: MutableMapping[int, List[Issue]]) -> Dict[str, Any]:
    numeric_rows = 0
    numeric_issue_rows = 0
    warning_issue_rows = 0
    seen_critical_issue_row: Set[int] = set()
    seen_warning_issue_row: Set[int] = set()

    for idx, row in df.iterrows():
        value = clean_text(row["value"])
        parameter_name = clean_text(row["parameter_name"])
        unit = clean_text(row["unit"]) or UNKNOWN
        numeric_values = extract_numeric_values(value)
        if not numeric_values:
            continue
        numeric_rows += 1

        def mark_issue(issue_code: str, severity: str, detail: str) -> None:
            add_row_issue(row_issues, int(idx), issue_code, severity, detail)
            nonlocal numeric_issue_rows, warning_issue_rows
            if severity == "critical":
                if int(idx) not in seen_critical_issue_row:
                    seen_critical_issue_row.add(int(idx))
                    numeric_issue_rows += 1
            elif severity == "warning":
                if int(idx) not in seen_warning_issue_row:
                    seen_warning_issue_row.add(int(idx))
                    warning_issue_rows += 1

        if any(number < 0 for number in numeric_values):
            mark_issue("negative_numeric_value", "critical", f"Negative numeric value in '{value}'.")

        if unit not in ALLOWED_UNITS:
            mark_issue("invalid_unit_vocabulary", "critical", f"Unit '{unit}' is not in allowed unit vocabulary.")

        if unit == UNKNOWN:
            mark_issue("numeric_value_with_unknown_unit", "warning", "Numeric value has unit set to 'unknown'.")

        if parameter_name == "widmark_factor":
            if min(numeric_values) < 0.4 or max(numeric_values) > 1.2:
                mark_issue(
                    "widmark_factor_out_of_range",
                    "critical",
                    f"Widmark factor {value} is outside plausible range [0.4, 1.2].",
                )

        if parameter_name == "ethanol_elimination_rate":
            if unit == "g/hr" and (min(numeric_values) < 1.0 or max(numeric_values) > 25.0):
                mark_issue(
                    "ethanol_elimination_rate_unrealistic",
                    "critical",
                    f"Ethanol elimination rate {value} g/hr outside plausible range [1, 25].",
                )
            if unit in {"BAC/hour", "%/hour"} and (min(numeric_values) < 0.005 or max(numeric_values) > 0.06):
                mark_issue(
                    "bac_elimination_rate_unrealistic",
                    "critical",
                    f"BAC elimination rate {value} {unit} outside plausible range [0.005, 0.06].",
                )

        if parameter_name == "total_body_water_percent" and max(numeric_values) > 100.0:
            mark_issue(
                "body_water_percent_over_100",
                "critical",
                f"Total body water percent {value} exceeds 100%.",
            )

        if "body_water" in parameter_name and unit == "%" and max(numeric_values) > 100.0:
            mark_issue(
                "body_water_percent_over_100",
                "critical",
                f"Body-water-linked percent {value} exceeds 100%.",
            )

    numeric_validity_score = 1.0 if numeric_rows == 0 else max(0.0, 1.0 - (numeric_issue_rows / float(numeric_rows)))
    return {
        "numeric_rows": numeric_rows,
        "critical_numeric_issue_rows": numeric_issue_rows,
        "warning_numeric_issue_rows": warning_issue_rows,
        "numeric_validity_score": round(numeric_validity_score, 4),
    }


def schema_integrity_score(df: pd.DataFrame, missing_columns: Sequence[str]) -> float:
    required_present = (len(REQUIRED_COLUMNS) - len(missing_columns)) / float(len(REQUIRED_COLUMNS))
    if df.empty:
        nonempty_score = 0.0
    else:
        required_present_columns = [column for column in REQUIRED_COLUMNS if column in df.columns]
        if not required_present_columns:
            nonempty_score = 0.0
        else:
            required_slice = df[required_present_columns].apply(lambda column: column.map(clean_text))
            nonempty_cells = int((required_slice != "").sum().sum())
            total_cells = int(required_slice.shape[0] * required_slice.shape[1])
            nonempty_score = nonempty_cells / float(total_cells) if total_cells else 0.0
    score = (0.7 * required_present) + (0.3 * nonempty_score)
    return round(score, 4)


def create_suspicious_dataframe(df: pd.DataFrame, row_issues: Mapping[int, Sequence[Issue]]) -> pd.DataFrame:
    if not row_issues:
        columns = list(df.columns) + ["issue_count", "issue_codes", "issue_severities", "issue_details", "row_index"]
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for idx in sorted(row_issues.keys()):
        base = dict(df.loc[idx])
        issues = row_issues[idx]
        base["row_index"] = int(idx)
        base["issue_count"] = len(issues)
        base["issue_codes"] = "|".join(issue.issue_code for issue in issues)
        base["issue_severities"] = "|".join(issue.severity for issue in issues)
        base["issue_details"] = " || ".join(issue.detail for issue in issues)
        rows.append(base)
    return pd.DataFrame(rows)


def compute_issue_summary(row_issues: Mapping[int, Sequence[Issue]]) -> Dict[str, Any]:
    severity_counts: Dict[str, int] = {"critical": 0, "warning": 0}
    issue_code_counts: Dict[str, int] = {}
    for issues in row_issues.values():
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            issue_code_counts[issue.issue_code] = issue_code_counts.get(issue.issue_code, 0) + 1
    return {
        "rows_with_issues": len(row_issues),
        "severity_counts": severity_counts,
        "issue_code_counts": dict(sorted(issue_code_counts.items())),
    }


def build_final_decision(
    missing_columns: Sequence[str],
    schema_score: float,
    numeric_score: float,
    issue_summary: Mapping[str, Any],
    population_validation: Mapping[str, Any],
    confidence_validation: Mapping[str, Any],
    parameter_name_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    critical_issue_count = int(issue_summary["severity_counts"].get("critical", 0))
    unknown_population_count = len(population_validation["unknown_population_groups"])
    malformed_population_count = len(population_validation["malformed_population_groups"])
    confidence_critical_count = int(confidence_validation["confidence_non_numeric_count"]) + int(
        confidence_validation["confidence_out_of_range_count"]
    )
    pbpk_missing = [name for name in PBPK_CORE_PARAMETERS if name not in set(parameter_name_validation["canonical_parameter_examples_present"]) and name not in set(parameter_name_validation.get("observed_parameter_names", []))]

    # Use observed names directly for PBPK checks.
    observed_names = set(parameter_name_validation.get("observed_parameter_names", []))
    pbpk_missing = [name for name in PBPK_CORE_PARAMETERS if name not in observed_names]

    ready = (
        len(missing_columns) == 0
        and schema_score >= 0.95
        and numeric_score >= 0.9
        and critical_issue_count == 0
        and unknown_population_count == 0
        and malformed_population_count == 0
        and confidence_critical_count == 0
        and len(pbpk_missing) == 0
    )

    reasoning: List[str] = [
        f"Schema integrity score: {schema_score:.4f}.",
        f"Numeric validity score: {numeric_score:.4f}.",
        f"Critical issues detected: {critical_issue_count}.",
        f"Rows with issues: {issue_summary['rows_with_issues']}.",
    ]
    if missing_columns:
        reasoning.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if pbpk_missing:
        reasoning.append(f"Missing PBPK-core parameter names: {', '.join(pbpk_missing)}.")
    if unknown_population_count or malformed_population_count:
        reasoning.append(
            f"Population-group issues -> unknown: {unknown_population_count}, malformed: {malformed_population_count}."
        )
    if confidence_critical_count:
        reasoning.append(f"Confidence-score issues detected: {confidence_critical_count}.")
    reasoning.append(
        "PBPK parameterization is allowed."
        if ready
        else "PBPK parameterization is blocked because one or more deterministic readiness gates failed."
    )

    return {
        "pbpk_parameter_readiness": ready,
        "safe_for_etl_05_parameterization": ready,
        "missing_required_columns": list(missing_columns),
        "missing_pbpk_core_parameters": pbpk_missing,
        "critical_issue_count": critical_issue_count,
        "reasoning": reasoning,
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    source_path = input_parameters_path(root)
    report_path = output_report_path(root)
    suspicious_path = output_suspicious_path(root)

    LOGGER.info("Loading human parameter table from %s", source_path)
    df_raw = pd.read_csv(source_path, dtype=str, keep_default_na=False, encoding=ENCODING)
    df = df_raw.copy()
    for column in df.columns:
        df[column] = df[column].map(clean_text)

    missing_columns = validate_required_columns(df)
    row_issues: Dict[int, List[Issue]] = {}

    if missing_columns:
        LOGGER.warning("Missing required columns detected: %s", ", ".join(missing_columns))
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    name_validation = validate_parameter_names(df, row_issues)
    name_validation["observed_parameter_names"] = sorted(set(df["parameter_name"].astype(str).tolist()))
    population_validation = validate_population_groups(df, row_issues)
    confidence_validation = validate_confidence_scores(df, row_issues)
    numeric_validation = validate_numeric_values(df, row_issues)

    schema_score = schema_integrity_score(df, missing_columns)
    issue_summary = compute_issue_summary(row_issues)
    decision = build_final_decision(
        missing_columns=missing_columns,
        schema_score=schema_score,
        numeric_score=float(numeric_validation["numeric_validity_score"]),
        issue_summary=issue_summary,
        population_validation=population_validation,
        confidence_validation=confidence_validation,
        parameter_name_validation=name_validation,
    )

    suspicious_df = create_suspicious_dataframe(df, row_issues)
    suspicious_df.to_csv(suspicious_path, index=False, encoding=ENCODING)

    report: Dict[str, Any] = {
        "source_file": str(source_path.relative_to(root)),
        "rows_evaluated": int(len(df)),
        "required_columns": list(REQUIRED_COLUMNS),
        "allowed_population_groups": list(ALLOWED_POPULATION_GROUPS),
        "schema_validation": {
            "missing_columns": list(missing_columns),
            "schema_integrity_score": schema_score,
        },
        "parameter_name_validation": name_validation,
        "population_group_validation": population_validation,
        "confidence_validation": confidence_validation,
        "numeric_validation": numeric_validation,
        "issue_summary": issue_summary,
        "schema_integrity_score": schema_score,
        "numeric_validity_score": float(numeric_validation["numeric_validity_score"]),
        "pbpk_parameter_readiness": decision["pbpk_parameter_readiness"],
        "final_decision": decision,
        "suspicious_output_file": str(suspicious_path.relative_to(root)),
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote suspicious parameters -> %s (rows=%d)", suspicious_path, len(suspicious_df))
    LOGGER.info(
        "Wrote parameter validation report -> %s | safe_for_etl_05_parameterization=%s",
        report_path,
        decision["safe_for_etl_05_parameterization"],
    )


if __name__ == "__main__":
    main()
