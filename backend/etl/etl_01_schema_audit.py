"""ETL step 01: schema audit for beverage ingestion CSVs.

Scans all CSV files under data/raw/07_beverage_knowledge, profiles each dataset,
detects cross-file schema overlaps/conflicts/redundancies, and writes audit outputs to:
- data/interim/beverage/schema_audit_report.json
- data/interim/beverage/schema_summary.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


LOGGER = logging.getLogger("etl_01_schema_audit")

DEFAULT_DELIMITER = ","
DELIMITER_CANDIDATES: Tuple[str, ...] = (",", ";", "\t", "|")
CSV_SUFFIX = ".csv"
ENCODING = "utf-8"
SNIFFER_SAMPLE_BYTES = 8192
N_PREVIEW_ROWS = 3
NULL_PERCENT_DECIMALS = 2
ROW_FINGERPRINT_SAMPLE_SIZE = 200
MERGE_OVERLAP_MIN = 2
MERGE_JACCARD_MIN = 0.30
REDUNDANT_JACCARD_MIN = 0.80

BEVERAGE_CATEGORY_KEYWORDS: Tuple[str, ...] = (
    "beverage",
    "drink",
    "category",
    "subcategory",
    "style",
    "type",
)

IDENTIFIER_KEYWORDS: Tuple[str, ...] = (
    "beverage_id",
    "drink_id",
    "product_id",
    "compound_id",
    "id",
    "uuid",
    "code",
    "sku",
    "upc",
    "ean",
    "name",
    "product",
)


@dataclass
class FileAudit:
    """Holds per-file audit results and optional load artifacts."""

    file_path: str
    delimiter: str
    status: str
    error: Optional[str]
    row_count: Optional[int]
    column_names: List[str]
    inferred_dtypes: Dict[str, str]
    null_percentage: Dict[str, float]
    duplicate_row_count: Optional[int]
    first_3_rows: List[Dict[str, Any]]
    unique_beverage_category_columns: Dict[str, List[str]]
    content_sha256: str
    normalized_columns: List[str]
    identifier_duplicates: Dict[str, int]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_input_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "07_beverage_knowledge"


def get_output_paths(repo_root: Path) -> Tuple[Path, Path]:
    out_dir = repo_root / "data" / "interim" / "beverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "schema_audit_report.json", out_dir / "schema_summary.csv"


def iter_csv_files(input_dir: Path) -> List[Path]:
    files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() == CSV_SUFFIX)
    return files


def detect_delimiter(path: Path) -> str:
    try:
        sample = path.read_text(encoding=ENCODING, errors="replace")[:SNIFFER_SAMPLE_BYTES]
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=DELIMITER_CANDIDATES)
        delimiter = dialect.delimiter
        LOGGER.debug("Detected delimiter '%s' for %s", delimiter, path)
        return delimiter
    except Exception as exc:
        LOGGER.warning(
            "Could not detect delimiter for %s (%s). Falling back to '%s'.",
            path,
            exc,
            DEFAULT_DELIMITER,
        )
        return DEFAULT_DELIMITER


def compute_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_column_name(column_name: str) -> str:
    return re.sub(r"\s+", "_", column_name.strip().lower())


def sanitize_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for row in records:
        sanitized.append({key: sanitize_value(val) for key, val in row.items()})
    return sanitized


def infer_beverage_category_columns(columns: Sequence[str]) -> List[str]:
    matched: List[str] = []
    for column in columns:
        normalized = normalize_column_name(column)
        if any(keyword in normalized for keyword in BEVERAGE_CATEGORY_KEYWORDS):
            matched.append(column)
    return matched


def infer_identifier_columns(columns: Sequence[str]) -> List[str]:
    matched: List[str] = []
    for column in columns:
        normalized = normalize_column_name(column)
        if normalized in IDENTIFIER_KEYWORDS:
            matched.append(column)
            continue
        if normalized.endswith("_id"):
            matched.append(column)
            continue
        if any(token in normalized for token in ("beverage", "drink", "product")) and any(
            token in normalized for token in ("id", "code", "name")
        ):
            matched.append(column)
    return matched


def load_csv_strict(path: Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        on_bad_lines="error",
    )


def build_file_audit(path: Path, repo_root: Path) -> FileAudit:
    rel_path = path.relative_to(repo_root).as_posix()
    delimiter = detect_delimiter(path)
    content_sha256 = compute_content_sha256(path)

    try:
        df = load_csv_strict(path=path, delimiter=delimiter)
    except Exception as exc:
        LOGGER.exception("Failed to parse %s", rel_path)
        return FileAudit(
            file_path=rel_path,
            delimiter=delimiter,
            status="error",
            error=str(exc),
            row_count=None,
            column_names=[],
            inferred_dtypes={},
            null_percentage={},
            duplicate_row_count=None,
            first_3_rows=[],
            unique_beverage_category_columns={},
            content_sha256=content_sha256,
            normalized_columns=[],
            identifier_duplicates={},
        )

    column_names = [str(col) for col in df.columns]
    inferred_dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    null_percentage = {
        str(col): round(float(df[col].isna().mean() * 100.0), NULL_PERCENT_DECIMALS)
        for col in df.columns
    }
    duplicate_row_count = int(df.duplicated(keep="first").sum())
    first_3_rows = sanitize_records(df.head(N_PREVIEW_ROWS).to_dict(orient="records"))

    beverage_category_columns = infer_beverage_category_columns(column_names)
    unique_beverage_category_columns: Dict[str, List[str]] = {}
    for column in beverage_category_columns:
        unique_values = sorted(
            str(value)
            for value in df[column].dropna().astype(str).str.strip().unique().tolist()
            if value
        )
        unique_beverage_category_columns[column] = unique_values

    identifier_columns = infer_identifier_columns(column_names)
    identifier_duplicates: Dict[str, int] = {}
    for column in identifier_columns:
        series = df[column].dropna().astype(str).str.strip()
        if series.empty:
            identifier_duplicates[column] = 0
            continue
        duplicate_count = int(series.duplicated(keep="first").sum())
        identifier_duplicates[column] = duplicate_count

    return FileAudit(
        file_path=rel_path,
        delimiter=delimiter,
        status="ok",
        error=None,
        row_count=int(len(df)),
        column_names=column_names,
        inferred_dtypes=inferred_dtypes,
        null_percentage=null_percentage,
        duplicate_row_count=duplicate_row_count,
        first_3_rows=first_3_rows,
        unique_beverage_category_columns=unique_beverage_category_columns,
        content_sha256=content_sha256,
        normalized_columns=[normalize_column_name(col) for col in column_names],
        identifier_duplicates=identifier_duplicates,
    )


def build_overlap_report(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    overlap: List[Dict[str, Any]] = []
    ok_audits = [audit for audit in file_audits if audit.status == "ok"]

    for left, right in combinations(ok_audits, 2):
        left_cols = set(left.normalized_columns)
        right_cols = set(right.normalized_columns)
        shared = sorted(left_cols & right_cols)
        if not shared:
            continue

        union = left_cols | right_cols
        jaccard = round(len(shared) / len(union), 4) if union else 0.0
        overlap.append(
            {
                "file_a": left.file_path,
                "file_b": right.file_path,
                "shared_columns": shared,
                "shared_column_count": len(shared),
                "jaccard_similarity": jaccard,
            }
        )

    overlap.sort(key=lambda item: (item["shared_column_count"], item["jaccard_similarity"]), reverse=True)
    return overlap


def build_conflicting_schema_report(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    column_dtype_map: Dict[str, Dict[str, Set[str]]] = {}

    for audit in file_audits:
        if audit.status != "ok":
            continue
        for column, dtype in audit.inferred_dtypes.items():
            normalized = normalize_column_name(column)
            if normalized not in column_dtype_map:
                column_dtype_map[normalized] = {}
            column_dtype_map[normalized].setdefault(dtype, set()).add(audit.file_path)

    conflicts: List[Dict[str, Any]] = []
    for normalized_col, dtype_sources in column_dtype_map.items():
        if len(dtype_sources) <= 1:
            continue
        conflicts.append(
            {
                "column": normalized_col,
                "dtype_variants": {
                    dtype: sorted(paths) for dtype, paths in sorted(dtype_sources.items())
                },
            }
        )

    conflicts.sort(key=lambda item: item["column"])
    return conflicts


def version_of_stem(path_stem: str) -> Optional[int]:
    match = re.search(r"_v(\d+)$", path_stem.lower())
    if not match:
        return None
    return int(match.group(1))


def stem_without_version(path_stem: str) -> str:
    return re.sub(r"_v\d+$", "", path_stem.lower())


def build_redundancy_report(file_audits: Sequence[FileAudit]) -> Dict[str, Any]:
    ok_audits = [audit for audit in file_audits if audit.status == "ok"]

    identical_content_groups: List[List[str]] = []
    by_hash: Dict[str, List[str]] = {}
    for audit in ok_audits:
        by_hash.setdefault(audit.content_sha256, []).append(audit.file_path)

    for file_paths in by_hash.values():
        if len(file_paths) > 1:
            identical_content_groups.append(sorted(file_paths))

    high_schema_similarity_pairs: List[Dict[str, Any]] = []
    for left, right in combinations(ok_audits, 2):
        left_cols = set(left.normalized_columns)
        right_cols = set(right.normalized_columns)
        union = left_cols | right_cols
        if not union:
            continue
        jaccard = len(left_cols & right_cols) / len(union)
        if jaccard >= REDUNDANT_JACCARD_MIN:
            high_schema_similarity_pairs.append(
                {
                    "file_a": left.file_path,
                    "file_b": right.file_path,
                    "jaccard_similarity": round(jaccard, 4),
                    "row_count_a": left.row_count,
                    "row_count_b": right.row_count,
                }
            )

    version_groups: Dict[str, List[Dict[str, Any]]] = {}
    for audit in ok_audits:
        stem = Path(audit.file_path).stem
        version = version_of_stem(stem)
        if version is None:
            continue
        key = stem_without_version(stem)
        version_groups.setdefault(key, []).append(
            {"file_path": audit.file_path, "version": version, "row_count": audit.row_count}
        )

    versioned_datasets: Dict[str, List[Dict[str, Any]]] = {}
    for key, group in sorted(version_groups.items()):
        if len(group) <= 1:
            continue
        versioned_datasets[key] = sorted(group, key=lambda item: item["version"])

    return {
        "identical_content_groups": identical_content_groups,
        "high_schema_similarity_pairs": high_schema_similarity_pairs,
        "versioned_datasets": versioned_datasets,
    }


def collect_identifier_values(
    file_audits: Sequence[FileAudit], repo_root: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    within_file_duplicates: List[Dict[str, Any]] = []
    cross_file_overlap: List[Dict[str, Any]] = []

    cross_file_bucket: Dict[str, Dict[str, Set[str]]] = {}

    for audit in file_audits:
        if audit.status != "ok":
            continue

        abs_path = repo_root / audit.file_path
        try:
            df = load_csv_strict(path=abs_path, delimiter=audit.delimiter)
        except Exception as exc:
            LOGGER.warning("Skipping identifier value collection for %s: %s", audit.file_path, exc)
            continue

        identifier_columns = infer_identifier_columns(df.columns.tolist())

        for column in identifier_columns:
            series = df[column].dropna().astype(str).str.strip()
            series = series[series != ""]
            duplicate_count = int(series.duplicated(keep="first").sum())

            if duplicate_count > 0:
                within_file_duplicates.append(
                    {
                        "file_path": audit.file_path,
                        "column": column,
                        "duplicate_count": duplicate_count,
                    }
                )

            normalized_col = normalize_column_name(column)
            values = set(series.tolist())
            cross_file_bucket.setdefault(normalized_col, {})[audit.file_path] = values

    for column, file_values in sorted(cross_file_bucket.items()):
        files = sorted(file_values.keys())
        for file_a, file_b in combinations(files, 2):
            overlap_values = file_values[file_a] & file_values[file_b]
            if not overlap_values:
                continue
            cross_file_overlap.append(
                {
                    "column": column,
                    "file_a": file_a,
                    "file_b": file_b,
                    "overlap_count": len(overlap_values),
                    "sample_values": sorted(list(overlap_values))[:10],
                }
            )

    cross_file_overlap.sort(key=lambda item: item["overlap_count"], reverse=True)
    within_file_duplicates.sort(key=lambda item: item["duplicate_count"], reverse=True)
    return within_file_duplicates, cross_file_overlap


def build_merge_candidates(overlap_report: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in overlap_report:
        if (
            item["shared_column_count"] >= MERGE_OVERLAP_MIN
            and item["jaccard_similarity"] >= MERGE_JACCARD_MIN
        ):
            candidates.append(
                {
                    "file_a": item["file_a"],
                    "file_b": item["file_b"],
                    "shared_column_count": item["shared_column_count"],
                    "jaccard_similarity": item["jaccard_similarity"],
                    "shared_columns": item["shared_columns"],
                }
            )
    return candidates


def build_source_of_truth_candidates(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    ok_audits = [audit for audit in file_audits if audit.status == "ok"]

    grouped: Dict[str, List[FileAudit]] = {}
    for audit in ok_audits:
        stem = Path(audit.file_path).stem
        group_key = stem_without_version(stem)
        grouped.setdefault(group_key, []).append(audit)

    for group_key, group in sorted(grouped.items()):
        if len(group) == 1:
            audit = group[0]
            candidates.append(
                {
                    "dataset_group": group_key,
                    "source_of_truth": audit.file_path,
                    "reason": "only dataset in group",
                }
            )
            continue

        ranked: List[Tuple[int, int, FileAudit]] = []
        for audit in group:
            version = version_of_stem(Path(audit.file_path).stem) or 0
            row_count = audit.row_count or 0
            ranked.append((version, row_count, audit))

        ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
        winner = ranked[0][2]
        candidates.append(
            {
                "dataset_group": group_key,
                "source_of_truth": winner.file_path,
                "reason": "highest detected version and row count in version group",
            }
        )

    return candidates


def build_recommendations(
    source_of_truth_candidates: Sequence[Dict[str, Any]],
    merge_candidates: Sequence[Dict[str, Any]],
    redundancy_report: Dict[str, Any],
) -> Dict[str, Any]:
    redundant_files: List[Dict[str, Any]] = []

    for group in redundancy_report.get("identical_content_groups", []):
        redundant_files.append(
            {
                "files": group,
                "reason": "identical file content hash",
            }
        )

    for pair in redundancy_report.get("high_schema_similarity_pairs", []):
        redundant_files.append(
            {
                "files": [pair["file_a"], pair["file_b"]],
                "reason": f"high schema similarity (jaccard={pair['jaccard_similarity']})",
            }
        )

    for key, group in redundancy_report.get("versioned_datasets", {}).items():
        if len(group) <= 1:
            continue
        latest = max(group, key=lambda item: item["version"])
        older = [item["file_path"] for item in group if item["file_path"] != latest["file_path"]]
        if older:
            redundant_files.append(
                {
                    "files": older,
                    "reason": f"older version(s) in versioned dataset group '{key}'",
                    "preferred": latest["file_path"],
                }
            )

    return {
        "candidate_source_of_truth_datasets": list(source_of_truth_candidates),
        "merge_candidates": list(merge_candidates),
        "redundant_files": redundant_files,
    }


def print_file_audit_to_stdout(audit: FileAudit) -> None:
    print("=" * 80)
    print(f"File: {audit.file_path}")
    print(f"Status: {audit.status}")
    print(f"Delimiter: {audit.delimiter}")
    if audit.error:
        print(f"Error: {audit.error}")
        return

    print(f"Row count: {audit.row_count}")
    print(f"Column names: {audit.column_names}")
    print(f"Inferred dtypes: {audit.inferred_dtypes}")
    print(f"Null percentage: {audit.null_percentage}")
    print(f"Duplicate row count: {audit.duplicate_row_count}")
    print("First 3 rows:")
    for row in audit.first_3_rows:
        print(row)

    if audit.unique_beverage_category_columns:
        print("Unique beverage/category columns:")
        for column, values in audit.unique_beverage_category_columns.items():
            print(f"  - {column}: {values}")
    else:
        print("Unique beverage/category columns: none detected")


def build_schema_summary_rows(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for audit in file_audits:
        if audit.status != "ok":
            rows.append(
                {
                    "file_path": audit.file_path,
                    "status": audit.status,
                    "error": audit.error,
                    "delimiter": audit.delimiter,
                    "row_count": audit.row_count,
                    "duplicate_row_count": audit.duplicate_row_count,
                    "column_name": None,
                    "normalized_column": None,
                    "dtype": None,
                    "null_percentage": None,
                }
            )
            continue

        for column in audit.column_names:
            rows.append(
                {
                    "file_path": audit.file_path,
                    "status": audit.status,
                    "error": audit.error,
                    "delimiter": audit.delimiter,
                    "row_count": audit.row_count,
                    "duplicate_row_count": audit.duplicate_row_count,
                    "column_name": column,
                    "normalized_column": normalize_column_name(column),
                    "dtype": audit.inferred_dtypes.get(column),
                    "null_percentage": audit.null_percentage.get(column),
                }
            )
    return rows


def main() -> None:
    configure_logging()

    repo_root = get_repo_root()
    input_dir = get_input_dir(repo_root)
    report_path, summary_path = get_output_paths(repo_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    csv_files = iter_csv_files(input_dir)
    LOGGER.info("Found %d CSV file(s) in %s", len(csv_files), input_dir)

    file_audits: List[FileAudit] = []
    for csv_file in csv_files:
        LOGGER.info("Auditing %s", csv_file.relative_to(repo_root).as_posix())
        audit = build_file_audit(path=csv_file, repo_root=repo_root)
        file_audits.append(audit)
        print_file_audit_to_stdout(audit)

    overlap_report = build_overlap_report(file_audits)
    conflicting_schema_report = build_conflicting_schema_report(file_audits)
    redundancy_report = build_redundancy_report(file_audits)
    within_file_dup_ids, cross_file_identifier_overlap = collect_identifier_values(
        file_audits=file_audits,
        repo_root=repo_root,
    )

    merge_candidates = build_merge_candidates(overlap_report)
    source_of_truth_candidates = build_source_of_truth_candidates(file_audits)
    recommendations = build_recommendations(
        source_of_truth_candidates=source_of_truth_candidates,
        merge_candidates=merge_candidates,
        redundancy_report=redundancy_report,
    )

    report: Dict[str, Any] = {
        "metadata": {
            "input_directory": input_dir.relative_to(repo_root).as_posix(),
            "total_csv_files": len(csv_files),
            "successful_files": sum(1 for audit in file_audits if audit.status == "ok"),
            "failed_files": sum(1 for audit in file_audits if audit.status != "ok"),
        },
        "files": [audit.__dict__ for audit in file_audits],
        "overlapping_columns_between_files": overlap_report,
        "conflicting_schemas": conflicting_schema_report,
        "probable_redundant_datasets": redundancy_report,
        "duplicate_beverage_identifiers": {
            "within_file_duplicates": within_file_dup_ids,
            "cross_file_value_overlap": cross_file_identifier_overlap,
        },
        "recommendations": recommendations,
    }

    summary_rows = build_schema_summary_rows(file_audits)
    summary_df = pd.DataFrame(summary_rows)

    report_path.write_text(json.dumps(report, indent=2), encoding=ENCODING)
    summary_df.to_csv(summary_path, index=False)

    LOGGER.info("Wrote JSON report: %s", report_path.relative_to(repo_root).as_posix())
    LOGGER.info("Wrote summary CSV: %s", summary_path.relative_to(repo_root).as_posix())


if __name__ == "__main__":
    main()
