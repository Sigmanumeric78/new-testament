"""ETL step 02b: strict canonical beverage ontology validation.

Validates the canonical beverage reference table prior to downstream ETL_03.
Produces deterministic QA artifacts:
- beverage_validation_report.json
- suspicious_beverages.csv
- category_distribution.csv
"""

from __future__ import annotations

import json
import logging
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_02b_beverage_validation")

ENCODING = "utf-8"
UNKNOWN = "unknown"
ID_PATTERN = re.compile(r"^BVG(\d{6})$")

ALLOWED_CATEGORIES: Tuple[str, ...] = (
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
ALLOWED_CATEGORY_SET: Set[str] = set(ALLOWED_CATEGORIES)

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "beverage_id",
    "beverage_name",
    "normalized_name",
    "category",
    "subcategory",
    "baseline_abv",
    "country_origin",
)

CATEGORY_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "hard_seltzer": ("hard seltzer", "hard_seltzer", "seltzer"),
    "fortified_wine": ("fortified wine", "fortified", "port", "sherry", "madeira", "vermouth"),
    "whisky": ("whisky", "whiskey", "bourbon", "scotch", "rye"),
    "vodka": ("vodka",),
    "rum": ("rum",),
    "gin": ("gin",),
    "tequila": ("tequila", "mezcal"),
    "brandy": ("brandy", "cognac", "pisco", "grappa"),
    "liqueur": ("liqueur", "schnapps", "amaretto", "sambuca", "triple sec", "chartreuse", "cointreau"),
    "cider": ("cider", "perry"),
    "cocktail": ("cocktail", "rtd", "alcopop", "premix", "cooler", "highball"),
    "sake": ("sake",),
    "mead": ("mead",),
    "beer": ("beer", "ale", "lager", "stout", "ipa", "pilsner", "porter", "bock"),
    "wine": ("wine", "champagne", "prosecco"),
}

CATEGORY_ABV_RANGES: Mapping[str, Tuple[float, float]] = {
    "beer": (0.0, 20.0),
    "wine": (5.0, 25.0),
    "whisky": (30.0, 70.0),
    "vodka": (30.0, 70.0),
    "rum": (20.0, 80.0),
    "gin": (30.0, 65.0),
    "tequila": (30.0, 60.0),
    "brandy": (20.0, 70.0),
    "liqueur": (10.0, 70.0),
    "cider": (0.0, 15.0),
    "cocktail": (0.0, 60.0),
    "sake": (5.0, 22.0),
    "mead": (3.0, 20.0),
    "hard_seltzer": (0.0, 12.0),
    "fortified_wine": (12.0, 25.0),
}

UNKNOWN_THRESHOLDS_PCT: Mapping[str, float] = {
    "category": 2.0,
    "abv": 5.0,
    "subcategory": 25.0,
    "country_origin": 25.0,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_master_path(root: Path) -> Path:
    return root / "data" / "processed" / "beverage" / "reference_tables" / "master_beverage_reference.csv"


def report_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "beverage_validation_report.json"


def suspicious_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "suspicious_beverages.csv"


def category_distribution_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "category_distribution.csv"


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


def is_unknown_text(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"", UNKNOWN}


def normalize_category(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return UNKNOWN
    text = re.sub(r"[\s\-]+", "_", text)
    return text


def parse_abv(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("%", "")
    text = text.replace(",", ".") if re.match(r"^\d+,\d+$", text) else text
    try:
        return float(text)
    except ValueError:
        return None


def normalize_tokenized_text(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: Any) -> List[str]:
    normalized = normalize_tokenized_text(value)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def has_repeated_adjacent_token(value: Any) -> bool:
    tokens = tokenize(value)
    if len(tokens) < 2:
        return False
    return any(tokens[idx] == tokens[idx - 1] for idx in range(1, len(tokens)))


def abv_key(value: Optional[float]) -> str:
    if value is None:
        return UNKNOWN
    return f"{value:.6f}"


def infer_category_from_text(beverage_name: str, normalized_name: str, subcategory: str) -> str:
    haystack = " ".join(
        [
            normalize_tokenized_text(subcategory),
            normalize_tokenized_text(beverage_name),
            normalize_tokenized_text(normalized_name),
        ]
    )
    if not haystack:
        return UNKNOWN

    for category in (
        "hard_seltzer",
        "fortified_wine",
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
        "beer",
        "wine",
    ):
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in haystack:
                return category
    return UNKNOWN


def category_keyword_hits(beverage_name: str, normalized_name: str, subcategory: str) -> Set[str]:
    haystack = " ".join(
        [
            normalize_tokenized_text(beverage_name),
            normalize_tokenized_text(normalized_name),
            normalize_tokenized_text(subcategory),
        ]
    )
    hits: Set[str] = set()
    if not haystack:
        return hits

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            hits.add(category)
    return hits


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    return missing


def add_flag(
    flagged: MutableMapping[int, Set[str]],
    indices: Iterable[int],
    reason: str,
) -> None:
    for idx in indices:
        flagged.setdefault(int(idx), set()).add(reason)


def validate_ids(df: pd.DataFrame) -> Dict[str, Any]:
    ids = df["beverage_id"].fillna("").astype(str).str.strip()
    missing_mask = ids.eq("") | ids.str.lower().eq(UNKNOWN)
    malformed_mask = ~missing_mask & ~ids.str.match(ID_PATTERN)
    duplicate_mask = ids.duplicated(keep=False)

    valid_numeric_ids = ids[~malformed_mask & ~missing_mask].str.extract(ID_PATTERN)[0].astype(int)
    unique_numeric_ids = sorted(valid_numeric_ids.unique().tolist())

    gap_values: List[int] = []
    if unique_numeric_ids:
        low = unique_numeric_ids[0]
        high = unique_numeric_ids[-1]
        observed = set(unique_numeric_ids)
        gap_values = [value for value in range(low, high + 1) if value not in observed]

    summary: Dict[str, Any] = {
        "pattern": ID_PATTERN.pattern,
        "total_rows": int(len(df)),
        "missing_id_count": int(missing_mask.sum()),
        "malformed_id_count": int(malformed_mask.sum()),
        "duplicate_id_count": int(duplicate_mask.sum()),
        "duplicate_ids": sorted(ids[duplicate_mask].unique().tolist()),
        "gap_count": int(len(gap_values)),
        "gap_examples": [f"BVG{value:06d}" for value in gap_values[:50]],
        "id_min_numeric": int(unique_numeric_ids[0]) if unique_numeric_ids else None,
        "id_max_numeric": int(unique_numeric_ids[-1]) if unique_numeric_ids else None,
    }
    return summary


def validate_categories(df: pd.DataFrame) -> Dict[str, Any]:
    raw = df["category"].fillna("").astype(str)
    normalized = raw.map(normalize_category)

    invalid_mask = ~normalized.isin(ALLOWED_CATEGORY_SET)
    invalid_values = sorted(normalized[invalid_mask].unique().tolist())

    misspelling_samples: List[Dict[str, str]] = []
    for invalid in invalid_values:
        suggested = get_close_matches(invalid, list(ALLOWED_CATEGORIES), n=1, cutoff=0.6)
        if suggested:
            misspelling_samples.append({"invalid": invalid, "suggested": suggested[0]})

    expected = df.apply(
        lambda row: infer_category_from_text(
            beverage_name=clean_text(row.get("beverage_name", "")),
            normalized_name=clean_text(row.get("normalized_name", "")),
            subcategory=clean_text(row.get("subcategory", "")),
        ),
        axis=1,
    )
    drift_mask = (expected != UNKNOWN) & (normalized != expected)

    drift_examples = (
        df.loc[drift_mask, ["beverage_id", "beverage_name", "category", "subcategory", "normalized_name"]]
        .head(50)
        .to_dict(orient="records")
    )

    return {
        "allowed_categories": list(ALLOWED_CATEGORIES),
        "invalid_category_count": int(invalid_mask.sum()),
        "invalid_categories": invalid_values,
        "misspelling_suggestions": misspelling_samples,
        "category_drift_count": int(drift_mask.sum()),
        "category_drift_examples": drift_examples,
        "normalized_category_series": normalized,
        "invalid_mask": invalid_mask,
        "drift_mask": drift_mask,
    }


def detect_normalization_failures(df: pd.DataFrame) -> Dict[str, Any]:
    repeated_in_normalized = df["normalized_name"].map(has_repeated_adjacent_token)
    repeated_in_beverage_name = df["beverage_name"].map(has_repeated_adjacent_token)

    slash_hybrid_mask = df["beverage_name"].fillna("").astype(str).str.contains("/", regex=False)

    inferred_hits = df.apply(
        lambda row: category_keyword_hits(
            beverage_name=clean_text(row.get("beverage_name", "")),
            normalized_name=clean_text(row.get("normalized_name", "")),
            subcategory=clean_text(row.get("subcategory", "")),
        ),
        axis=1,
    )
    cross_category_hybrid_mask = inferred_hits.map(lambda hits: len(hits) > 1)

    spirit_word_conflict_mask = df.apply(
        lambda row: "whiskey whiskey" in clean_text(row.get("beverage_name", "")).lower()
        or "vodka vodka" in clean_text(row.get("beverage_name", "")).lower()
        or "beer beer" in clean_text(row.get("beverage_name", "")).lower()
        or "bourbon bourbon whisky" in clean_text(row.get("beverage_name", "")).lower()
        or "vodka vodka" in clean_text(row.get("normalized_name", "")).lower()
        or "beer beer" in clean_text(row.get("normalized_name", "")).lower(),
        axis=1,
    )

    suspicious_mask = (
        repeated_in_normalized
        | repeated_in_beverage_name
        | slash_hybrid_mask
        | cross_category_hybrid_mask
        | spirit_word_conflict_mask
    )

    return {
        "suspicious_normalization_count": int(suspicious_mask.sum()),
        "repeated_token_in_normalized_count": int(repeated_in_normalized.sum()),
        "repeated_token_in_beverage_name_count": int(repeated_in_beverage_name.sum()),
        "slash_or_hybrid_name_count": int(slash_hybrid_mask.sum()),
        "cross_category_keyword_hybrid_count": int(cross_category_hybrid_mask.sum()),
        "explicit_pattern_hits_count": int(spirit_word_conflict_mask.sum()),
        "suspicious_examples": (
            df.loc[suspicious_mask, ["beverage_id", "beverage_name", "normalized_name", "category", "subcategory"]]
            .head(100)
            .to_dict(orient="records")
        ),
        "suspicious_mask": suspicious_mask,
    }


def detect_duplicate_leakage(df: pd.DataFrame, category_normalized: pd.Series, abv_numeric: pd.Series) -> Dict[str, Any]:
    key_df = pd.DataFrame(
        {
            "normalized_name_key": df["normalized_name"].fillna("").astype(str).str.strip().str.lower(),
            "category_key": category_normalized,
            "abv_key": abv_numeric.map(abv_key),
        }
    )
    dup_mask = key_df.duplicated(subset=["normalized_name_key", "category_key", "abv_key"], keep=False)

    duplicates = (
        df.loc[dup_mask, ["beverage_id", "beverage_name", "normalized_name", "category", "baseline_abv"]]
        .copy()
        .sort_values(by=["normalized_name", "category", "baseline_abv", "beverage_id"], kind="mergesort")
    )
    grouped = (
        duplicates.assign(
            normalized_name_key=duplicates["normalized_name"].fillna("").astype(str).str.strip().str.lower(),
            category_key=duplicates["category"].map(normalize_category),
            abv_key=duplicates["baseline_abv"].map(lambda x: abv_key(parse_abv(x))),
        )
        .groupby(["normalized_name_key", "category_key", "abv_key"], sort=True)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values(by=["duplicate_count", "normalized_name_key"], ascending=[False, True], kind="mergesort")
    )

    return {
        "duplicate_rows_count": int(dup_mask.sum()),
        "duplicate_groups_count": int(len(grouped)),
        "duplicate_group_examples": grouped.head(100).to_dict(orient="records"),
        "duplicate_mask": dup_mask,
    }


def validate_abv(df: pd.DataFrame, category_normalized: pd.Series, abv_numeric: pd.Series) -> Dict[str, Any]:
    missing_abv_mask = abv_numeric.isna()
    negative_abv_mask = abv_numeric < 0
    over_100_abv_mask = abv_numeric > 100
    impossible_abv_mask = negative_abv_mask | over_100_abv_mask

    mismatch_mask = pd.Series(False, index=df.index)
    for category, (lower, upper) in CATEGORY_ABV_RANGES.items():
        out_of_range = (category_normalized == category) & abv_numeric.notna() & (
            (abv_numeric < lower) | (abv_numeric > upper)
        )
        mismatch_mask = mismatch_mask | out_of_range

    mismatch_breakdown: List[Dict[str, Any]] = []
    for category in ALLOWED_CATEGORIES:
        if category not in CATEGORY_ABV_RANGES:
            continue
        category_mask = (category_normalized == category) & mismatch_mask
        if int(category_mask.sum()) > 0:
            lower, upper = CATEGORY_ABV_RANGES[category]
            mismatch_breakdown.append(
                {
                    "category": category,
                    "expected_abv_range": [lower, upper],
                    "count": int(category_mask.sum()),
                }
            )

    return {
        "missing_abv_count": int(missing_abv_mask.sum()),
        "negative_abv_count": int(negative_abv_mask.sum()),
        "over_100_abv_count": int(over_100_abv_mask.sum()),
        "impossible_abv_count": int(impossible_abv_mask.sum()),
        "category_mismatch_count": int(mismatch_mask.sum()),
        "category_mismatch_breakdown": mismatch_breakdown,
        "impossible_abv_examples": (
            df.loc[impossible_abv_mask, ["beverage_id", "beverage_name", "category", "baseline_abv"]]
            .head(100)
            .to_dict(orient="records")
        ),
        "category_mismatch_examples": (
            df.loc[mismatch_mask, ["beverage_id", "beverage_name", "category", "baseline_abv", "subcategory"]]
            .head(100)
            .to_dict(orient="records")
        ),
        "missing_abv_mask": missing_abv_mask,
        "impossible_abv_mask": impossible_abv_mask,
        "category_mismatch_mask": mismatch_mask,
    }


def compute_unknown_explosion(
    df: pd.DataFrame,
    category_normalized: pd.Series,
    abv_numeric: pd.Series,
) -> Dict[str, Any]:
    total = int(len(df))
    if total == 0:
        return {
            "total_rows": 0,
            "metrics": {},
            "thresholds_pct": dict(UNKNOWN_THRESHOLDS_PCT),
            "explosion_fields": [],
        }

    unknown_category = ((category_normalized == UNKNOWN) | (category_normalized == "")).sum()
    unknown_abv = abv_numeric.isna().sum()
    unknown_subcategory = df["subcategory"].map(is_unknown_text).sum()
    unknown_country = df["country_origin"].map(is_unknown_text).sum()

    metrics = {
        "category": {
            "unknown_count": int(unknown_category),
            "unknown_pct": round((float(unknown_category) / total) * 100.0, 4),
        },
        "abv": {
            "unknown_count": int(unknown_abv),
            "unknown_pct": round((float(unknown_abv) / total) * 100.0, 4),
        },
        "subcategory": {
            "unknown_count": int(unknown_subcategory),
            "unknown_pct": round((float(unknown_subcategory) / total) * 100.0, 4),
        },
        "country_origin": {
            "unknown_count": int(unknown_country),
            "unknown_pct": round((float(unknown_country) / total) * 100.0, 4),
        },
    }

    explosion_fields = [
        field
        for field, detail in metrics.items()
        if detail["unknown_pct"] > UNKNOWN_THRESHOLDS_PCT[field]
    ]

    return {
        "total_rows": total,
        "metrics": metrics,
        "thresholds_pct": dict(UNKNOWN_THRESHOLDS_PCT),
        "explosion_fields": sorted(explosion_fields),
    }


def build_category_distribution(df: pd.DataFrame, category_normalized: pd.Series) -> pd.DataFrame:
    total = max(int(len(df)), 1)
    counts = category_normalized.value_counts(dropna=False, sort=False).to_dict()

    rows: List[Dict[str, Any]] = []
    for category in ALLOWED_CATEGORIES:
        count = int(counts.get(category, 0))
        rows.append(
            {
                "category": category,
                "count": count,
                "percentage": round((count / total) * 100.0, 4),
                "is_allowed": True,
            }
        )

    extra_categories = sorted(category for category in counts if category not in ALLOWED_CATEGORY_SET)
    for category in extra_categories:
        count = int(counts.get(category, 0))
        rows.append(
            {
                "category": category,
                "count": count,
                "percentage": round((count / total) * 100.0, 4),
                "is_allowed": False,
            }
        )

    return pd.DataFrame(rows, columns=["category", "count", "percentage", "is_allowed"])


def serialize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): serialize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [serialize_for_json(item) for item in payload]
    if isinstance(payload, tuple):
        return [serialize_for_json(item) for item in payload]
    if isinstance(payload, pd.Series):
        return serialize_for_json(payload.tolist())
    if isinstance(payload, (pd.Timestamp,)):
        return payload.isoformat()
    return payload


def main() -> None:
    configure_logging()
    root = repo_root()
    input_path = input_master_path(root)
    report_path = report_output_path(root)
    suspicious_path = suspicious_output_path(root)
    category_distribution_path = category_distribution_output_path(root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input master file not found: {input_path}")

    LOGGER.info("Loading master beverage reference: %s", input_path)
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    LOGGER.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    missing_columns = validate_required_columns(df)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    category_normalized = df["category"].map(normalize_category)
    abv_numeric = df["baseline_abv"].map(parse_abv)

    id_validation = validate_ids(df)
    category_validation = validate_categories(df)
    normalization_validation = detect_normalization_failures(df)
    duplicate_validation = detect_duplicate_leakage(df, category_normalized, abv_numeric)
    abv_validation = validate_abv(df, category_normalized, abv_numeric)
    unknown_validation = compute_unknown_explosion(df, category_normalized, abv_numeric)

    flagged: Dict[int, Set[str]] = {}

    ids = df["beverage_id"].fillna("").astype(str).str.strip()
    add_flag(flagged, ids[ids.eq("") | ids.str.lower().eq(UNKNOWN)].index.tolist(), "missing_id")
    add_flag(flagged, ids[~ids.eq("") & ~ids.str.match(ID_PATTERN)].index.tolist(), "malformed_id")
    add_flag(flagged, ids[ids.duplicated(keep=False)].index.tolist(), "duplicate_id")
    add_flag(
        flagged,
        category_validation["invalid_mask"][category_validation["invalid_mask"]].index.tolist(),
        "invalid_category",
    )
    add_flag(
        flagged,
        category_validation["drift_mask"][category_validation["drift_mask"]].index.tolist(),
        "category_drift",
    )
    add_flag(
        flagged,
        normalization_validation["suspicious_mask"][normalization_validation["suspicious_mask"]].index.tolist(),
        "normalization_suspicious",
    )
    add_flag(
        flagged,
        duplicate_validation["duplicate_mask"][duplicate_validation["duplicate_mask"]].index.tolist(),
        "duplicate_leakage",
    )
    add_flag(
        flagged,
        abv_validation["missing_abv_mask"][abv_validation["missing_abv_mask"]].index.tolist(),
        "missing_abv",
    )
    add_flag(
        flagged,
        abv_validation["impossible_abv_mask"][abv_validation["impossible_abv_mask"]].index.tolist(),
        "impossible_abv",
    )
    add_flag(
        flagged,
        abv_validation["category_mismatch_mask"][abv_validation["category_mismatch_mask"]].index.tolist(),
        "abv_category_mismatch",
    )

    suspicious_indices = sorted(flagged.keys())
    suspicious_df = df.loc[suspicious_indices].copy() if suspicious_indices else df.iloc[0:0].copy()
    suspicious_df["qa_flags"] = suspicious_df.index.map(lambda idx: ";".join(sorted(flagged.get(int(idx), set()))))
    suspicious_df = suspicious_df.sort_values(by=["beverage_id", "beverage_name"], kind="mergesort")
    suspicious_df.to_csv(suspicious_path, index=False, encoding=ENCODING)
    LOGGER.info("Wrote suspicious rows: %s (rows=%d)", suspicious_path, len(suspicious_df))

    category_distribution_df = build_category_distribution(df, category_normalized)
    category_distribution_df.to_csv(category_distribution_path, index=False, encoding=ENCODING)
    LOGGER.info("Wrote category distribution: %s", category_distribution_path)

    blocking_issues: List[str] = []
    if id_validation["missing_id_count"] > 0:
        blocking_issues.append("missing_deterministic_ids")
    if id_validation["malformed_id_count"] > 0:
        blocking_issues.append("malformed_deterministic_ids")
    if id_validation["duplicate_id_count"] > 0:
        blocking_issues.append("duplicate_deterministic_ids")
    if id_validation["gap_count"] > 0:
        blocking_issues.append("deterministic_id_gaps")
    if category_validation["invalid_category_count"] > 0:
        blocking_issues.append("invalid_categories")
    if category_validation["category_drift_count"] > 0:
        blocking_issues.append("category_drift")
    if duplicate_validation["duplicate_rows_count"] > 0:
        blocking_issues.append("duplicate_leakage")
    if abv_validation["impossible_abv_count"] > 0:
        blocking_issues.append("impossible_abv_values")
    if abv_validation["category_mismatch_count"] > 0:
        blocking_issues.append("abv_category_mismatch")
    if unknown_validation["explosion_fields"]:
        blocking_issues.append("unknown_explosion")

    safe_for_etl_03 = len(blocking_issues) == 0

    report: Dict[str, Any] = {
        "metadata": {
            "script": "etl/etl_02b_beverage_validation.py",
            "input_file": str(input_path.relative_to(root)),
            "total_rows": int(len(df)),
            "required_columns": list(REQUIRED_COLUMNS),
        },
        "validation": {
            "deterministic_ids": id_validation,
            "categories": {
                key: value
                for key, value in category_validation.items()
                if key not in {"normalized_category_series", "invalid_mask", "drift_mask"}
            },
            "normalization_failures": {
                key: value
                for key, value in normalization_validation.items()
                if key not in {"suspicious_mask"}
            },
            "duplicate_leakage": {
                key: value
                for key, value in duplicate_validation.items()
                if key not in {"duplicate_mask"}
            },
            "abv": {
                key: value
                for key, value in abv_validation.items()
                if key
                not in {"missing_abv_mask", "impossible_abv_mask", "category_mismatch_mask"}
            },
            "unknown_explosion": unknown_validation,
        },
        "artifacts": {
            "validation_report_json": str(report_path.relative_to(root)),
            "suspicious_beverages_csv": str(suspicious_path.relative_to(root)),
            "category_distribution_csv": str(category_distribution_path.relative_to(root)),
        },
        "recommendation": {
            "safe_for_etl_03": safe_for_etl_03,
            "blocking_issues": blocking_issues,
        },
    }

    report_clean = serialize_for_json(report)
    with report_path.open("w", encoding=ENCODING) as fh:
        json.dump(report_clean, fh, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("Wrote validation report: %s", report_path)
    LOGGER.info("safe_for_etl_03=%s | blocking_issues=%s", safe_for_etl_03, blocking_issues)


if __name__ == "__main__":
    main()
