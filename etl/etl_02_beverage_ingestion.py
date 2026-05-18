"""ETL step 02: canonical beverage ontology ingestion.

Builds a canonical beverage reference table from curated beverage datasets,
with deterministic IDs, normalization, deduplication, provenance tracking,
and ingestion reporting.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_02_beverage_ingestion")

ENCODING = "utf-8"
UNKNOWN = "unknown"

CATEGORY_ORDER: Tuple[str, ...] = (
    "beer",
    "wine",
    "whisky",
    "vodka",
    "rum",
    "gin",
    "tequila",
    "brandy",
    "liqueur",
    "cider",
    "cocktail",
    "sake",
    "mead",
    "hard_seltzer",
    "fortified_wine",
    "unknown",
)
VALID_CATEGORIES = set(CATEGORY_ORDER)

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "beverage_id",
    "beverage_name",
    "normalized_name",
    "category",
    "subcategory",
    "baseline_abv",
    "min_abv",
    "max_abv",
    "serving_size_ml",
    "carbonation",
    "sugar_g_per_100ml",
    "caffeine",
    "country_origin",
    "aliases",
    "source_dataset",
    "source_file",
    "source_row",
    "confidence_score",
)

NUMERIC_FIELDS: Tuple[str, ...] = (
    "baseline_abv",
    "min_abv",
    "max_abv",
    "serving_size_ml",
    "sugar_g_per_100ml",
)

TEXT_FIELDS: Tuple[str, ...] = (
    "beverage_name",
    "normalized_name",
    "category",
    "subcategory",
    "carbonation",
    "caffeine",
    "country_origin",
    "aliases",
)

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
SOURCE_PRIORITY = {
    "alcohol_abv_reference": 1,
    "alcoholic_drinks_open_units": 2,
    "alcohol_compounds_dataset": 3,
    "beverage_compound_profile_v3": 4,
    "alcohol_compounds_digestion_repaired": 5,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def processed_output_path(root: Path) -> Path:
    out = root / "data" / "processed" / "beverage" / "reference_tables"
    out.mkdir(parents=True, exist_ok=True)
    return out / "master_beverage_reference.csv"


def report_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "beverage_ingestion_report.json"


def path_candidates(root: Path) -> Dict[str, List[Path]]:
    return {
        "alcohol_abv_reference": [
            root / "data" / "raw" / "07_beverage_knowledge" / "alcohol_abv_reference.csv",
        ],
        "alcohol_compounds_dataset": [
            root / "data" / "raw" / "07_beverage_knowledge" / "alcohol_compounds_dataset.csv",
            root
            / "data"
            / "interim"
            / "beverage"
            / "repaired"
            / "data"
            / "raw"
            / "07_beverage_knowledge"
            / "alcohol_compounds_dataset.csv",
        ],
        "beverage_compound_profile_v3": [
            root / "data" / "raw" / "07_beverage_knowledge" / "beverage_compound_profile_v3.csv",
        ],
        "alcoholic_drinks_open_units": [
            root
            / "data"
            / "raw"
            / "07_beverage_knowledge"
            / "alcoholic_drinks"
            / "open_units.csv",
        ],
        "alcohol_compounds_digestion_repaired": [
            root
            / "data"
            / "interim"
            / "beverage"
            / "repaired"
            / "alcohol_compounds_digestion.csv",
            root
            / "data"
            / "interim"
            / "beverage"
            / "repaired"
            / "data"
            / "raw"
            / "07_beverage_knowledge"
            / "alcohol_compounds_digestion.csv",
        ],
    }


def resolve_inputs(root: Path) -> Tuple[Dict[str, Path], Dict[str, List[str]]]:
    resolved: Dict[str, Path] = {}
    missing: Dict[str, List[str]] = {}

    for dataset, candidates in path_candidates(root).items():
        existing = [candidate for candidate in candidates if candidate.exists()]
        if existing:
            resolved[dataset] = existing[0]
        else:
            missing[dataset] = [candidate.relative_to(root).as_posix() for candidate in candidates]

    return resolved, missing


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="error")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)

    text = clean_text(value)
    if not text:
        return None

    text = text.replace("%", "")
    text = text.replace(",", ".") if re.match(r"^\d+,\d+$", text) else text
    try:
        return float(text)
    except ValueError:
        return None


def normalize_beverage_name(name: str) -> str:
    text = clean_text(name).lower()
    if not text:
        return UNKNOWN

    text = text.replace("whiskey", "whisky")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok]
    if not tokens:
        return UNKNOWN

    # deduplicate tokens and sort for deterministic canonical naming
    unique_tokens = sorted(set(tokens))
    normalized = " ".join(unique_tokens)
    return normalized if normalized else UNKNOWN


def canonicalize_category(
    raw_category: str,
    raw_subcategory: str,
    normalized_name: str,
) -> str:
    haystack = " ".join(
        part.lower()
        for part in [clean_text(raw_category), clean_text(raw_subcategory), clean_text(normalized_name)]
        if clean_text(part)
    )

    if not haystack:
        return UNKNOWN

    if any(k in haystack for k in ["hard seltzer", "hard_seltzer", "seltzer"]):
        return "hard_seltzer"
    if any(k in haystack for k in ["fortified", "port", "sherry", "madeira", "vermouth"]):
        return "fortified_wine"
    if any(k in haystack for k in ["whisky", "whiskey", "bourbon", "scotch", "rye"]):
        return "whisky"
    if "vodka" in haystack:
        return "vodka"
    if "rum" in haystack:
        return "rum"
    if "gin" in haystack:
        return "gin"
    if any(k in haystack for k in ["tequila", "mezcal"]):
        return "tequila"
    if any(k in haystack for k in ["brandy", "cognac", "pisco", "grappa"]):
        return "brandy"
    if any(k in haystack for k in ["liqueur", "schnapps", "amaretto", "sambuca", "triple sec", "bitters"]):
        return "liqueur"
    if "cider" in haystack or "perry" in haystack:
        return "cider"
    if any(k in haystack for k in ["cocktail", "rtd", "alcopop", "cooler", "premix"]):
        return "cocktail"
    if "sake" in haystack:
        return "sake"
    if "mead" in haystack:
        return "mead"
    if "beer" in haystack or "ale" in haystack or "lager" in haystack or "stout" in haystack:
        return "beer"
    if "wine" in haystack or "champagne" in haystack or "prosecco" in haystack:
        return "wine"

    return UNKNOWN


def normalize_aliases(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return UNKNOWN
    return text


def default_record() -> Dict[str, Any]:
    return {
        "beverage_id": "",
        "beverage_name": UNKNOWN,
        "normalized_name": UNKNOWN,
        "category": UNKNOWN,
        "subcategory": UNKNOWN,
        "baseline_abv": None,
        "min_abv": None,
        "max_abv": None,
        "serving_size_ml": None,
        "carbonation": UNKNOWN,
        "sugar_g_per_100ml": None,
        "caffeine": UNKNOWN,
        "country_origin": UNKNOWN,
        "aliases": UNKNOWN,
        "source_dataset": "",
        "source_file": "",
        "source_row": "",
        "confidence_score": "low",
    }


def determine_confidence(record: Dict[str, Any]) -> str:
    has_abv = record.get("baseline_abv") is not None
    has_name = clean_text(record.get("normalized_name")) not in {"", UNKNOWN}
    has_category = clean_text(record.get("category")) not in {"", UNKNOWN}

    if has_abv and has_name and has_category:
        return "high"
    if has_name and has_category:
        return "medium"
    return "low"


def set_common_fields(
    record: Dict[str, Any],
    beverage_name: str,
    raw_category: str,
    raw_subcategory: str,
    source_dataset: str,
    source_file: str,
    source_row: int,
    unresolved_categories: Dict[str, int],
) -> None:
    name_clean = clean_text(beverage_name)
    record["beverage_name"] = name_clean if name_clean else UNKNOWN
    record["normalized_name"] = normalize_beverage_name(record["beverage_name"])
    record["subcategory"] = clean_text(raw_subcategory) or UNKNOWN

    category = canonicalize_category(
        raw_category=raw_category,
        raw_subcategory=raw_subcategory,
        normalized_name=record["normalized_name"],
    )
    record["category"] = category
    if category == UNKNOWN and (clean_text(raw_category) or clean_text(raw_subcategory)):
        unresolved_key = f"{clean_text(raw_category)}|{clean_text(raw_subcategory)}"
        unresolved_categories[unresolved_key] = unresolved_categories.get(unresolved_key, 0) + 1

    record["source_dataset"] = source_dataset
    record["source_file"] = source_file
    record["source_row"] = str(source_row)
    record["confidence_score"] = determine_confidence(record)


def ingest_alcohol_abv_reference(
    df: pd.DataFrame,
    source_file: str,
    unresolved_categories: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        rec = default_record()
        raw_cat = clean_text(row.get("main_category"))
        raw_sub = clean_text(row.get("subcategory"))

        beverage_name = raw_sub or raw_cat or UNKNOWN
        set_common_fields(
            record=rec,
            beverage_name=beverage_name,
            raw_category=raw_cat,
            raw_subcategory=raw_sub,
            source_dataset="alcohol_abv_reference",
            source_file=source_file,
            source_row=int(idx) + 2,
            unresolved_categories=unresolved_categories,
        )

        rec["baseline_abv"] = parse_float(row.get("baseline_abv"))
        rec["min_abv"] = parse_float(row.get("min_abv"))
        rec["max_abv"] = parse_float(row.get("max_abv"))
        rec["aliases"] = normalize_aliases(row.get("aliases"))

        confidence_text = clean_text(row.get("confidence")).lower()
        if confidence_text in CONFIDENCE_RANK:
            rec["confidence_score"] = confidence_text
        else:
            rec["confidence_score"] = determine_confidence(rec)

        records.append(rec)

    return records


def infer_carbonation(category: str) -> str:
    if category in {"beer", "cider", "hard_seltzer"}:
        return "yes"
    return UNKNOWN


def ingest_open_units(
    df: pd.DataFrame,
    source_file: str,
    unresolved_categories: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        rec = default_record()

        raw_cat = clean_text(row.get("Category"))
        raw_sub = clean_text(row.get("Style"))
        beverage_name = clean_text(row.get("Product")) or raw_sub or raw_cat or UNKNOWN

        set_common_fields(
            record=rec,
            beverage_name=beverage_name,
            raw_category=raw_cat,
            raw_subcategory=raw_sub,
            source_dataset="alcoholic_drinks_open_units",
            source_file=source_file,
            source_row=int(idx) + 2,
            unresolved_categories=unresolved_categories,
        )

        abv = parse_float(row.get("ABV"))
        rec["baseline_abv"] = abv
        rec["min_abv"] = abv
        rec["max_abv"] = abv
        rec["serving_size_ml"] = parse_float(row.get("Volume"))
        rec["aliases"] = clean_text(row.get("Brand")) or UNKNOWN
        rec["carbonation"] = infer_carbonation(rec["category"])
        rec["confidence_score"] = determine_confidence(rec)

        records.append(rec)

    return records


def ingest_compound_profile_like(
    df: pd.DataFrame,
    source_file: str,
    source_dataset: str,
    unresolved_categories: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        rec = default_record()

        raw_cat = clean_text(row.get("main_category"))
        raw_sub = clean_text(row.get("subcategory"))
        beverage_name = raw_sub or raw_cat or UNKNOWN

        set_common_fields(
            record=rec,
            beverage_name=beverage_name,
            raw_category=raw_cat,
            raw_subcategory=raw_sub,
            source_dataset=source_dataset,
            source_file=source_file,
            source_row=int(idx) + 2,
            unresolved_categories=unresolved_categories,
        )

        rec["aliases"] = UNKNOWN
        rec["confidence_score"] = determine_confidence(rec)
        records.append(rec)

    return records


def parse_beverage_mentions(text: str) -> List[str]:
    raw = clean_text(text)
    if not raw:
        return []

    cleaned = re.sub(r"\([^)]*\)", "", raw)
    parts: List[str] = []
    for chunk in cleaned.split(";"):
        for sub in chunk.split(","):
            token = clean_text(sub)
            if token:
                parts.append(token)

    # deterministic unique order
    unique = sorted(set(parts), key=lambda x: x.lower())
    return unique


def ingest_digestion_repaired(
    df: pd.DataFrame,
    source_file: str,
    unresolved_categories: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        mentions = parse_beverage_mentions(clean_text(row.get("beverages_found_in")))
        if not mentions:
            mentions = [UNKNOWN]

        for mention in mentions:
            rec = default_record()
            set_common_fields(
                record=rec,
                beverage_name=mention,
                raw_category=mention,
                raw_subcategory=clean_text(row.get("chemical_class")),
                source_dataset="alcohol_compounds_digestion_repaired",
                source_file=source_file,
                source_row=int(idx) + 2,
                unresolved_categories=unresolved_categories,
            )
            rec["aliases"] = clean_text(row.get("compound")) or UNKNOWN
            rec["confidence_score"] = "low"
            records.append(rec)

    return records


def baseline_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN
    try:
        return f"{float(value):.6f}"
    except Exception:
        return clean_text(value) or UNKNOWN


def non_unknown(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = clean_text(value).lower()
    return text not in {"", UNKNOWN}


def source_rank(value: str) -> int:
    return SOURCE_PRIORITY.get(value, 999)


def merge_group(group: pd.DataFrame) -> Dict[str, Any]:
    group_sorted = group.sort_values(
        by=["source_priority", "source_file", "source_row_num"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    out = default_record()

    # Pick first non-unknown for most fields, with deterministic source priority.
    for field in [
        "beverage_name",
        "normalized_name",
        "category",
        "subcategory",
        "baseline_abv",
        "min_abv",
        "max_abv",
        "serving_size_ml",
        "carbonation",
        "sugar_g_per_100ml",
        "caffeine",
        "country_origin",
    ]:
        chosen: Any = None
        for _, row in group_sorted.iterrows():
            value = row[field]
            if field in NUMERIC_FIELDS:
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    chosen = value
                    break
            else:
                if non_unknown(value):
                    chosen = value
                    break
        out[field] = chosen

    aliases_values = [
        clean_text(value)
        for value in group_sorted["aliases"].tolist()
        if non_unknown(value)
    ]
    out["aliases"] = ";".join(sorted(set(aliases_values))) if aliases_values else UNKNOWN

    datasets = [clean_text(v) for v in group_sorted["source_dataset"].tolist() if clean_text(v)]
    files = [clean_text(v) for v in group_sorted["source_file"].tolist() if clean_text(v)]
    rows = [str(int(v)) for v in group_sorted["source_row_num"].tolist()]

    out["source_dataset"] = ";".join(dict.fromkeys(datasets))
    out["source_file"] = ";".join(dict.fromkeys(files))
    out["source_row"] = ";".join(dict.fromkeys(rows))

    best_conf = "low"
    best_rank = -1
    for value in group_sorted["confidence_score"].tolist():
        rank = CONFIDENCE_RANK.get(clean_text(value).lower(), 0)
        if rank > best_rank:
            best_rank = rank
            best_conf = clean_text(value).lower() if clean_text(value) else "low"
    out["confidence_score"] = best_conf if best_conf in CONFIDENCE_RANK else "low"

    return out


def merge_records(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, int, int, int]:
    df = pd.DataFrame(records)
    if df.empty:
        return df, 0, 0, 0

    df["source_priority"] = df["source_dataset"].map(source_rank)
    df["source_row_num"] = pd.to_numeric(df["source_row"], errors="coerce").fillna(0).astype(int)
    df["baseline_abv_key"] = df["baseline_abv"].apply(baseline_key)

    keys = ["normalized_name", "category", "baseline_abv_key"]
    df_sorted = df.sort_values(
        by=keys + ["source_priority", "source_file", "source_row_num"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    )

    merged_rows: List[Dict[str, Any]] = []
    for _, group in df_sorted.groupby(keys, dropna=False, sort=True):
        merged_rows.append(merge_group(group))

    merged_df = pd.DataFrame(merged_rows)

    rows_before = len(df)
    rows_after = len(merged_df)
    duplicates_removed = rows_before - rows_after

    return merged_df, rows_before, rows_after, duplicates_removed


def assign_beverage_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = ["category", "normalized_name", "baseline_abv", "subcategory", "beverage_name"]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = UNKNOWN

    df = df.sort_values(by=sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
    df["beverage_id"] = [f"BVG{i:06d}" for i in range(1, len(df) + 1)]
    return df


def finalize_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in TEXT_FIELDS + ("source_dataset", "source_file", "source_row", "confidence_score"):
        df[col] = df[col].apply(lambda x: clean_text(x) if clean_text(x) else UNKNOWN)

    for col in NUMERIC_FIELDS:
        df[col] = df[col].apply(lambda x: x if x is not None and not (isinstance(x, float) and pd.isna(x)) else UNKNOWN)

    df["category"] = df["category"].apply(lambda x: x if x in VALID_CATEGORIES else UNKNOWN)
    df["confidence_score"] = df["confidence_score"].apply(
        lambda x: x if x in CONFIDENCE_RANK else "low"
    )

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = UNKNOWN

    return df.loc[:, list(CANONICAL_COLUMNS)]


def missing_values_summary(df: pd.DataFrame) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            summary[col] = len(df)
            continue

        count = 0
        for value in df[col].tolist():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                count += 1
                continue
            if clean_text(value).lower() in {"", UNKNOWN}:
                count += 1
        summary[col] = count

    return summary


def category_distribution(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}
    vc = df["category"].value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def beverages_missing_abv(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    missing = df[df["baseline_abv"].astype(str).str.lower() == UNKNOWN]
    names = sorted(set(missing["normalized_name"].astype(str).tolist()))
    return names


def build_records(resolved_inputs: Dict[str, Path], root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    records: List[Dict[str, Any]] = []
    unresolved_categories: Dict[str, int] = {}
    ingest_errors: List[str] = []

    for dataset in [
        "alcohol_abv_reference",
        "alcohol_compounds_dataset",
        "beverage_compound_profile_v3",
        "alcoholic_drinks_open_units",
        "alcohol_compounds_digestion_repaired",
    ]:
        path = resolved_inputs.get(dataset)
        if path is None:
            continue

        rel = path.relative_to(root).as_posix()
        try:
            df = read_csv(path)
        except Exception as exc:
            error = f"failed_read:{dataset}:{rel}:{exc}"
            LOGGER.error(error)
            ingest_errors.append(error)
            continue

        LOGGER.info("Ingesting %s (%s rows) from %s", dataset, len(df), rel)

        if dataset == "alcohol_abv_reference":
            records.extend(ingest_alcohol_abv_reference(df, rel, unresolved_categories))
        elif dataset == "alcoholic_drinks_open_units":
            records.extend(ingest_open_units(df, rel, unresolved_categories))
        elif dataset == "beverage_compound_profile_v3":
            records.extend(
                ingest_compound_profile_like(
                    df,
                    rel,
                    source_dataset="beverage_compound_profile_v3",
                    unresolved_categories=unresolved_categories,
                )
            )
        elif dataset == "alcohol_compounds_dataset":
            records.extend(
                ingest_compound_profile_like(
                    df,
                    rel,
                    source_dataset="alcohol_compounds_dataset",
                    unresolved_categories=unresolved_categories,
                )
            )
        elif dataset == "alcohol_compounds_digestion_repaired":
            records.extend(ingest_digestion_repaired(df, rel, unresolved_categories))

    return records, unresolved_categories, ingest_errors


def main() -> None:
    configure_logging()
    root = repo_root()

    resolved_inputs, missing_inputs = resolve_inputs(root)
    LOGGER.info("Resolved %d canonical dataset(s), missing %d", len(resolved_inputs), len(missing_inputs))

    records, unresolved_category_map, ingest_errors = build_records(resolved_inputs, root)
    LOGGER.info("Constructed %d raw ontology records", len(records))

    merged_df, rows_before, rows_after, duplicates_removed = merge_records(records)
    merged_df = assign_beverage_ids(merged_df)
    merged_df = finalize_unknowns(merged_df)

    processed_path = processed_output_path(root)
    merged_df.to_csv(processed_path, index=False)

    unresolved_categories = [
        {"raw_category_subcategory": key, "count": count}
        for key, count in sorted(unresolved_category_map.items(), key=lambda item: (-item[1], item[0]))
    ]

    report: Dict[str, Any] = {
        "metadata": {
            "script": "etl/etl_02_beverage_ingestion.py",
            "rows_before_merge": rows_before,
            "rows_after_merge": rows_after,
            "duplicates_removed": duplicates_removed,
            "resolved_input_datasets": {
                name: path.relative_to(root).as_posix() for name, path in resolved_inputs.items()
            },
            "missing_input_datasets": missing_inputs,
            "ingest_errors": ingest_errors,
        },
        "metrics": {
            "rows_before_merge": rows_before,
            "rows_after_merge": rows_after,
            "duplicates_removed": duplicates_removed,
            "missing_values": missing_values_summary(merged_df),
            "category_distribution": category_distribution(merged_df),
            "unresolved_categories": unresolved_categories,
            "beverages_missing_abv": {
                "count": len(beverages_missing_abv(merged_df)),
                "items": beverages_missing_abv(merged_df)[:250],
            },
        },
        "recommendations": {
            "beverage_compound_profile_v2": {
                "action": "deprecate",
                "reason": "Use v3 only as canonical compound profile source.",
            }
        },
    }

    report_path = report_output_path(root)
    report_path.write_text(json.dumps(report, indent=2), encoding=ENCODING)

    LOGGER.info("Wrote master beverage reference: %s", processed_path.relative_to(root).as_posix())
    LOGGER.info("Wrote ingestion report: %s", report_path.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
