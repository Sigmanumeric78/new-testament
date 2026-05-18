"""ETL step 01b: beverage CSV schema repair and integrity reporting.

- Inspects all CSV files under data/raw/07_beverage_knowledge.
- Detects malformed CSV structures and row-level anomalies.
- Deterministically repairs alcohol_compounds_digestion.csv into interim/repaired.
- Writes integrity and version overlap reports.

This script never modifies raw files.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_01b_schema_repair")

CSV_SUFFIX = ".csv"
DEFAULT_DELIMITER = ","
DELIMITER_CANDIDATES: Tuple[str, ...] = (",", ";", "\t", "|")
ENCODING = "utf-8"
SNIFFER_SAMPLE_BYTES = 8192
YES_NO_VALUES = {"yes", "no", "true", "false"}

REPAIR_TARGET = "alcohol_compounds_digestion.csv"
V2_FILE = "beverage_compound_profile_v2.csv"
V3_FILE = "beverage_compound_profile_v3.csv"

ACTION_KEEP = "keep"
ACTION_MERGE = "merge"
ACTION_ARCHIVE = "archive"
ACTION_REPAIR = "repair"
ACTION_REMOVE = "remove"


@dataclass
class MalformedRow:
    row_number: int
    expected_columns: int
    detected_columns: Optional[int]
    issue_types: List[str]
    parse_error: Optional[str]
    raw_row: str


@dataclass
class FileInspection:
    file_path: str
    delimiter: str
    expected_columns: int
    header_columns: List[str]
    header_issues: List[str]
    total_data_rows: int
    malformed_rows: List[MalformedRow] = field(default_factory=list)
    variable_column_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class RepairEvent:
    row_number: int
    action: str
    confidence_score: float
    reason: str
    raw_row: str
    repaired_row: Dict[str, str]


@dataclass
class RepairOutcome:
    file_path: str
    repaired_copy_path: str
    repaired: bool
    repaired_rows: int
    copied_rows: int
    dropped_rows: int
    confidence_score: float
    repair_events: List[RepairEvent]
    unresolved_issues: List[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_beverage_dir(root: Path) -> Path:
    return root / "data" / "raw" / "07_beverage_knowledge"


def interim_beverage_dir(root: Path) -> Path:
    return root / "data" / "interim" / "beverage"


def repaired_dir(root: Path) -> Path:
    out = interim_beverage_dir(root) / "repaired"
    out.mkdir(parents=True, exist_ok=True)
    return out


def integrity_report_path(root: Path) -> Path:
    return interim_beverage_dir(root) / "data_integrity_report.json"


def version_report_path(root: Path) -> Path:
    return interim_beverage_dir(root) / "version_comparison_report.json"


def list_csv_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() == CSV_SUFFIX)


def detect_delimiter(path: Path) -> str:
    try:
        sample = path.read_text(encoding=ENCODING, errors="replace")[:SNIFFER_SAMPLE_BYTES]
        if not sample.strip():
            return DEFAULT_DELIMITER
        dialect = csv.Sniffer().sniff(sample, delimiters=DELIMITER_CANDIDATES)
        return dialect.delimiter
    except Exception:
        return DEFAULT_DELIMITER


def parse_line(
    line: str,
    delimiter: str,
    strict: bool,
) -> Tuple[Optional[List[str]], Optional[str]]:
    try:
        reader = csv.reader(
            [line],
            delimiter=delimiter,
            quotechar='"',
            escapechar="\\",
            strict=strict,
        )
        values = next(reader)
        return values, None
    except Exception as exc:
        return None, str(exc)


def normalize_header(columns: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for col in columns:
        cleaned = re.sub(r"\s+", "_", str(col).strip().lower())
        cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
        normalized.append(cleaned)
    return normalized


def detect_header_issues(columns: Sequence[str]) -> List[str]:
    issues: List[str] = []
    if not columns:
        issues.append("empty_header")
        return issues

    if any(not str(col).strip() for col in columns):
        issues.append("blank_header_column_name")

    normalized = normalize_header(columns)
    duplicate_headers = [name for name in set(normalized) if normalized.count(name) > 1 and name]
    if duplicate_headers:
        issues.append(f"duplicate_header_names:{sorted(duplicate_headers)}")

    return issues


def detect_inconsistent_delimiter(raw_line: str, delimiter: str) -> bool:
    expected_count = raw_line.count(delimiter)
    if expected_count == 0:
        return False

    for candidate in DELIMITER_CANDIDATES:
        if candidate == delimiter:
            continue
        if raw_line.count(candidate) > expected_count:
            return True
    return False


def detect_missing_escapes(raw_line: str) -> bool:
    quote_count = raw_line.count('"')
    return quote_count % 2 == 1


def inspect_csv(path: Path, root: Path) -> FileInspection:
    rel = path.relative_to(root).as_posix()
    delimiter = detect_delimiter(path)

    lines = path.read_text(encoding=ENCODING, errors="replace").splitlines()
    if not lines:
        return FileInspection(
            file_path=rel,
            delimiter=delimiter,
            expected_columns=0,
            header_columns=[],
            header_issues=["empty_file"],
            total_data_rows=0,
        )

    header_values, header_err = parse_line(lines[0], delimiter=delimiter, strict=False)
    if header_values is None:
        return FileInspection(
            file_path=rel,
            delimiter=delimiter,
            expected_columns=0,
            header_columns=[],
            header_issues=[f"malformed_header:{header_err}"],
            total_data_rows=max(0, len(lines) - 1),
        )

    header_issues = detect_header_issues(header_values)
    expected_columns = len(header_values)

    malformed_rows: List[MalformedRow] = []
    column_count_histogram: Dict[str, int] = {}

    for idx, raw_line in enumerate(lines[1:], start=2):
        parsed_strict, strict_err = parse_line(raw_line, delimiter=delimiter, strict=True)
        parsed_loose, _ = parse_line(raw_line, delimiter=delimiter, strict=False)

        parsed = parsed_loose if parsed_loose is not None else parsed_strict
        detected_columns = len(parsed) if parsed is not None else None

        if detected_columns is not None:
            key = str(detected_columns)
            column_count_histogram[key] = column_count_histogram.get(key, 0) + 1

        issue_types: List[str] = []
        if strict_err is not None:
            issue_types.append("broken_quoting")

        if detect_inconsistent_delimiter(raw_line, delimiter=delimiter):
            issue_types.append("inconsistent_delimiters")

        if detect_missing_escapes(raw_line):
            issue_types.append("missing_escapes_or_unbalanced_quotes")

        if detected_columns is not None and detected_columns != expected_columns:
            issue_types.append("variable_column_count")
            if detected_columns > expected_columns and delimiter == ",":
                issue_types.append("embedded_commas")

        if issue_types:
            malformed_rows.append(
                MalformedRow(
                    row_number=idx,
                    expected_columns=expected_columns,
                    detected_columns=detected_columns,
                    issue_types=sorted(set(issue_types)),
                    parse_error=strict_err,
                    raw_row=raw_line,
                )
            )

    return FileInspection(
        file_path=rel,
        delimiter=delimiter,
        expected_columns=expected_columns,
        header_columns=[str(c) for c in header_values],
        header_issues=header_issues,
        total_data_rows=max(0, len(lines) - 1),
        malformed_rows=malformed_rows,
        variable_column_counts=column_count_histogram,
    )


def repair_digestion_row(fields: List[str], expected_columns: int) -> Tuple[List[str], str, float, str]:
    if len(fields) == expected_columns:
        return fields, "as_is", 1.0, "row already matches expected column count"

    if len(fields) < expected_columns:
        padded = fields + [""] * (expected_columns - len(fields))
        return padded, "pad_missing_columns", 0.55, "row had fewer columns than expected"

    # len(fields) > expected_columns
    yes_idx: Optional[int] = None
    for i in range(2, len(fields)):
        if fields[i].strip().lower() in YES_NO_VALUES:
            yes_idx = i
            break

    if yes_idx is not None and yes_idx >= 3:
        pre = fields[:yes_idx]
        compound = ",".join(pre[:-2]).strip() if len(pre) > 2 else pre[0].strip()
        chemical_class = pre[-2].strip() if len(pre) > 1 else ""
        description = pre[-1].strip() if len(pre) > 0 else ""

        affects_digestion = fields[yes_idx].strip()
        tail = fields[yes_idx + 1 :]

        if len(tail) >= 4:
            mechanism = ",".join(tail[:-3]).strip()
            beverages_found_in = tail[-3].strip()
            typical_concentration = tail[-2].strip()
            source = tail[-1].strip()
        else:
            padded_tail = tail + [""] * (4 - len(tail))
            mechanism, beverages_found_in, typical_concentration, source = (
                padded_tail[0].strip(),
                padded_tail[1].strip(),
                padded_tail[2].strip(),
                padded_tail[3].strip(),
            )

        repaired = [
            compound,
            chemical_class,
            description,
            affects_digestion,
            mechanism,
            beverages_found_in,
            typical_concentration,
            source,
        ]

        return (
            repaired,
            "semantic_rebuild_with_yes_no_anchor",
            0.9,
            "rebuilt row by anchoring affects_digestion token and right-aligning trailing fields",
        )

    overflow_merged = fields[: expected_columns - 1] + [",".join(fields[expected_columns - 1 :]).strip()]
    return (
        overflow_merged,
        "merge_overflow_into_last_column",
        0.65,
        "fallback overflow merge due to missing yes/no anchor",
    )


def repair_or_copy_file(
    path: Path,
    inspection: FileInspection,
    root: Path,
    repaired_root: Path,
) -> RepairOutcome:
    rel_path = path.relative_to(root)
    out_path = repaired_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if path.name != REPAIR_TARGET:
        shutil.copy2(path, out_path)
        return RepairOutcome(
            file_path=rel_path.as_posix(),
            repaired_copy_path=out_path.relative_to(root).as_posix(),
            repaired=False,
            repaired_rows=0,
            copied_rows=inspection.total_data_rows,
            dropped_rows=0,
            confidence_score=1.0,
            repair_events=[],
            unresolved_issues=[],
        )

    lines = path.read_text(encoding=ENCODING, errors="replace").splitlines()
    header_values, _ = parse_line(lines[0], delimiter=inspection.delimiter, strict=False)
    if header_values is None:
        shutil.copy2(path, out_path)
        return RepairOutcome(
            file_path=rel_path.as_posix(),
            repaired_copy_path=out_path.relative_to(root).as_posix(),
            repaired=False,
            repaired_rows=0,
            copied_rows=0,
            dropped_rows=0,
            confidence_score=0.0,
            repair_events=[],
            unresolved_issues=["header_not_parseable_no_repair_performed"],
        )

    expected_columns = len(header_values)
    repaired_rows: List[List[str]] = []
    repair_events: List[RepairEvent] = []

    for line_no, raw_line in enumerate(lines[1:], start=2):
        loose_fields, parse_err = parse_line(raw_line, delimiter=inspection.delimiter, strict=False)
        if loose_fields is None:
            # Preserve row text by placing it in last column if parsing fully fails.
            filler = [""] * max(0, expected_columns - 1)
            fallback = filler + [raw_line]
            repaired_rows.append(fallback)
            repair_events.append(
                RepairEvent(
                    row_number=line_no,
                    action="unparsed_row_fallback",
                    confidence_score=0.4,
                    reason=f"csv parse failed: {parse_err}",
                    raw_row=raw_line,
                    repaired_row={
                        str(col): fallback[idx] if idx < len(fallback) else ""
                        for idx, col in enumerate(header_values)
                    },
                )
            )
            continue

        repaired, action, confidence, reason = repair_digestion_row(
            fields=loose_fields,
            expected_columns=expected_columns,
        )
        repaired_rows.append(repaired)
        if action != "as_is":
            repair_events.append(
                RepairEvent(
                    row_number=line_no,
                    action=action,
                    confidence_score=confidence,
                    reason=reason,
                    raw_row=raw_line,
                    repaired_row={
                        str(col): repaired[idx] if idx < len(repaired) else ""
                        for idx, col in enumerate(header_values)
                    },
                )
            )

    repaired_df = pd.DataFrame(repaired_rows, columns=[str(c) for c in header_values])
    repaired_df.to_csv(out_path, index=False)

    avg_conf = (
        round(sum(evt.confidence_score for evt in repair_events) / len(repair_events), 4)
        if repair_events
        else 1.0
    )

    return RepairOutcome(
        file_path=rel_path.as_posix(),
        repaired_copy_path=out_path.relative_to(root).as_posix(),
        repaired=True,
        repaired_rows=len(repair_events),
        copied_rows=len(repaired_rows),
        dropped_rows=0,
        confidence_score=avg_conf,
        repair_events=repair_events,
        unresolved_issues=[],
    )


def dataset_recommendation(
    inspection: FileInspection,
    repaired: RepairOutcome,
) -> str:
    if inspection.file_path.endswith(V2_FILE):
        return ACTION_ARCHIVE

    if inspection.file_path.endswith(REPAIR_TARGET):
        if repaired.repaired_rows > 0:
            return ACTION_REPAIR
        return ACTION_KEEP

    if inspection.file_path.endswith(V3_FILE):
        return ACTION_KEEP

    if inspection.malformed_rows:
        return ACTION_REPAIR

    return ACTION_KEEP


def normalized_row_hashes(df: pd.DataFrame) -> pd.Series:
    normalized = df.fillna("").astype(str)
    normalized = normalized.apply(lambda col: col.str.strip().str.lower())
    return normalized.agg("|".join, axis=1)


def compute_version_overlap(
    root: Path,
    raw_dir: Path,
    delimiter_cache: Dict[str, str],
) -> Dict[str, Any]:
    v2_path = raw_dir / V2_FILE
    v3_path = raw_dir / V3_FILE

    unresolved: List[str] = []
    if not v2_path.exists():
        unresolved.append(f"missing_file:{v2_path.relative_to(root).as_posix()}")
    if not v3_path.exists():
        unresolved.append(f"missing_file:{v3_path.relative_to(root).as_posix()}")

    if not v2_path.exists() or not v3_path.exists():
        return {
            "v2_file": v2_path.relative_to(root).as_posix(),
            "v3_file": v3_path.relative_to(root).as_posix(),
            "row_overlap_percent": None,
            "schema_overlap_percent": None,
            "conflicting_rows": [],
            "new_fields_in_v3": [],
            "v2_can_be_deprecated": True,
            "deprecation_rationale": "User instruction: use v3 only for beverage compound profile.",
            "unresolved_issues": unresolved,
        }

    v2_delim = delimiter_cache.get(v2_path.relative_to(root).as_posix(), DEFAULT_DELIMITER)
    v3_delim = delimiter_cache.get(v3_path.relative_to(root).as_posix(), DEFAULT_DELIMITER)

    v2_df = pd.read_csv(v2_path, sep=v2_delim, engine="python", on_bad_lines="error")
    v3_df = pd.read_csv(v3_path, sep=v3_delim, engine="python", on_bad_lines="error")

    v2_cols = list(v2_df.columns)
    v3_cols = list(v3_df.columns)
    common_cols = [col for col in v2_cols if col in v3_cols]
    union_cols = sorted(set(v2_cols) | set(v3_cols))

    schema_overlap = (len(common_cols) / len(union_cols) * 100.0) if union_cols else 100.0

    v2_hash = set(normalized_row_hashes(v2_df[common_cols]).tolist()) if common_cols else set()
    v3_hash = set(normalized_row_hashes(v3_df[common_cols]).tolist()) if common_cols else set()

    row_overlap = (len(v2_hash & v3_hash) / len(v2_hash) * 100.0) if v2_hash else 0.0

    key_candidates = [c for c in ["main_category", "subcategory", "compound_id", "compound"] if c in common_cols]

    conflicting_rows: List[Dict[str, Any]] = []
    if key_candidates:
        v2_idx = v2_df.set_index(key_candidates, drop=False)
        v3_idx = v3_df.set_index(key_candidates, drop=False)
        common_keys = sorted(set(v2_idx.index) & set(v3_idx.index))

        compare_cols = [c for c in common_cols if c not in key_candidates]
        for key in common_keys:
            left = v2_idx.loc[key]
            right = v3_idx.loc[key]

            if isinstance(left, pd.DataFrame):
                left = left.iloc[0]
            if isinstance(right, pd.DataFrame):
                right = right.iloc[0]

            diffs: Dict[str, Dict[str, Any]] = {}
            for col in compare_cols:
                lv = None if pd.isna(left[col]) else str(left[col])
                rv = None if pd.isna(right[col]) else str(right[col])
                if lv != rv:
                    diffs[col] = {"v2": lv, "v3": rv}

            if diffs:
                conflicting_rows.append(
                    {
                        "key": {k: str(left[k]) for k in key_candidates},
                        "differences": diffs,
                    }
                )

    new_fields_in_v3 = [c for c in v3_cols if c not in v2_cols]

    return {
        "v2_file": v2_path.relative_to(root).as_posix(),
        "v3_file": v3_path.relative_to(root).as_posix(),
        "row_overlap_percent": round(row_overlap, 4),
        "schema_overlap_percent": round(schema_overlap, 4),
        "conflicting_rows": conflicting_rows[:250],
        "new_fields_in_v3": new_fields_in_v3,
        "v2_can_be_deprecated": True,
        "deprecation_rationale": "User instruction: use v3 only for beverage compound profile.",
        "unresolved_issues": unresolved,
    }


def build_recommendations(
    inspections: Sequence[FileInspection],
    repair_outcomes: Dict[str, RepairOutcome],
    root: Path,
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    for inspection in inspections:
        outcome = repair_outcomes[inspection.file_path]
        action = dataset_recommendation(inspection=inspection, repaired=outcome)
        recommendations.append(
            {
                "dataset": inspection.file_path,
                "recommended_action": action,
                "notes": {
                    "malformed_row_count": len(inspection.malformed_rows),
                    "header_issues": inspection.header_issues,
                    "repair_applied": outcome.repaired,
                    "repair_events": outcome.repaired_rows,
                },
            }
        )

    # explicit archival recommendation for v2 (even when missing), per user instruction
    v2_rel = (raw_beverage_dir(root) / V2_FILE).relative_to(root).as_posix()
    if not any(item["dataset"] == v2_rel for item in recommendations):
        recommendations.append(
            {
                "dataset": v2_rel,
                "recommended_action": ACTION_ARCHIVE,
                "notes": {
                    "malformed_row_count": None,
                    "header_issues": ["file_not_found"],
                    "repair_applied": False,
                    "repair_events": 0,
                },
            }
        )

    return recommendations


def main() -> None:
    configure_logging()
    root = repo_root()
    raw_dir = raw_beverage_dir(root)
    repaired_root = repaired_dir(root)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw beverage directory: {raw_dir}")

    csv_files = list_csv_files(raw_dir)
    LOGGER.info("Inspecting %d beverage CSV files", len(csv_files))

    inspections: List[FileInspection] = []
    repair_outcomes: Dict[str, RepairOutcome] = {}
    delimiter_cache: Dict[str, str] = {}

    for path in csv_files:
        inspection = inspect_csv(path=path, root=root)
        inspections.append(inspection)
        delimiter_cache[inspection.file_path] = inspection.delimiter

        outcome = repair_or_copy_file(
            path=path,
            inspection=inspection,
            root=root,
            repaired_root=repaired_root,
        )
        repair_outcomes[inspection.file_path] = outcome

        LOGGER.info(
            "Processed %s | malformed_rows=%d | repaired_events=%d",
            inspection.file_path,
            len(inspection.malformed_rows),
            outcome.repaired_rows,
        )

    malformed_files = [ins for ins in inspections if ins.header_issues or ins.malformed_rows]

    unresolved_issues: List[str] = []
    for inspection in inspections:
        if inspection.header_issues:
            unresolved_issues.append(
                f"{inspection.file_path}:header_issues:{inspection.header_issues}"
            )

    recommendations = build_recommendations(
        inspections=inspections,
        repair_outcomes=repair_outcomes,
        root=root,
    )

    integrity_report: Dict[str, Any] = {
        "metadata": {
            "raw_directory": raw_dir.relative_to(root).as_posix(),
            "repaired_directory": repaired_root.relative_to(root).as_posix(),
            "inspected_files": len(inspections),
            "malformed_files": len(malformed_files),
        },
        "files": [
            {
                **asdict(inspection),
                "malformed_rows": [asdict(mr) for mr in inspection.malformed_rows],
                "repair_outcome": {
                    **asdict(repair_outcomes[inspection.file_path]),
                    "repair_events": [
                        asdict(event)
                        for event in repair_outcomes[inspection.file_path].repair_events
                    ],
                },
            }
            for inspection in inspections
        ],
        "malformed_files": [inspection.file_path for inspection in malformed_files],
        "malformed_row_numbers": {
            inspection.file_path: [row.row_number for row in inspection.malformed_rows]
            for inspection in inspections
            if inspection.malformed_rows
        },
        "repair_actions_taken": {
            path: [event.action for event in outcome.repair_events]
            for path, outcome in repair_outcomes.items()
            if outcome.repair_events
        },
        "confidence_scores": {
            path: outcome.confidence_score for path, outcome in repair_outcomes.items()
        },
        "unresolved_issues": unresolved_issues,
        "recommendations": recommendations,
    }

    integrity_path = integrity_report_path(root)
    integrity_path.write_text(json.dumps(integrity_report, indent=2), encoding=ENCODING)

    version_report = compute_version_overlap(
        root=root,
        raw_dir=raw_dir,
        delimiter_cache=delimiter_cache,
    )
    version_report["recommendations"] = {
        "beverage_compound_profile": {
            "preferred_dataset": str((raw_dir / V3_FILE).relative_to(root).as_posix()),
            "deprecated_dataset": str((raw_dir / V2_FILE).relative_to(root).as_posix()),
            "action": ACTION_ARCHIVE,
            "rationale": "Use v3 only; older versions are not needed.",
        }
    }

    version_path = version_report_path(root)
    version_path.write_text(json.dumps(version_report, indent=2), encoding=ENCODING)

    LOGGER.info("Wrote integrity report: %s", integrity_path.relative_to(root).as_posix())
    LOGGER.info("Wrote version report: %s", version_path.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
