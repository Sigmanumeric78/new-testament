"""ETL step 03c: deterministic chemical class ontology expansion.

Expands unresolved family/class placeholder rows from ETL_03 into
representative molecules backed by the local PubChem library.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableSequence, Optional, Sequence, Tuple

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from etl.etl_03_beverage_compounds import (
    ENCODING,
    UNKNOWN,
    build_compound_resolver_index,
    clean_text,
    load_pubchem_index,
    load_rdkit,
    normalize_compound_name,
    pubchem_library_root,
    repo_root,
    resolve_pubchem_cid,
)

LOGGER = logging.getLogger("etl_03c_chemical_class_expansion")

INPUT_MATRIX_RELATIVE = Path("data/processed/beverage/compound_profiles/beverage_compound_matrix.csv")
OUTPUT_MATRIX_RELATIVE = Path("data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv")
INPUT_REPORT_RELATIVE = Path("data/interim/beverage/compound_ingestion_report.json")
OUTPUT_REPORT_RELATIVE = Path("data/interim/beverage/chemical_class_expansion_report.json")

CLASS_ALIAS_TO_KEY: Mapping[str, str] = {
    "esters": "esters",
    "congeners": "congeners",
    "fusel alcohols": "fusel_alcohols",
    "nitrogen compounds": "nitrogen_compounds",
    "fatty acids": "fatty_acids",
    "organic acids": "organic_acids",
    "polyphenols": "polyphenols",
    "tannins": "tannins",
    "terpenoids": "terpenoids",
    "terpenes": "terpenoids",
    "hop terpenes": "hop_terpenes",
    "hop acids": "hop_acids",
    "lactones": "lactones",
    "phenols": "phenols",
    "residual sugars": "residual_sugars",
    "smoke compounds": "smoke_compounds_equiv",
    "smoke compounds equiv": "smoke_compounds_equiv",
}

CLASS_TO_CATEGORY: Mapping[str, str] = {
    "esters": "ester",
    "congeners": "congener",
    "fusel_alcohols": "fusel_alcohol",
    "nitrogen_compounds": "metabolite",
    "fatty_acids": "organic_acid",
    "organic_acids": "organic_acid",
    "polyphenols": "polyphenol",
    "tannins": "polyphenol",
    "terpenoids": "flavor_compound",
    "hop_terpenes": "flavor_compound",
    "hop_acids": "flavor_compound",
    "lactones": "flavor_compound",
    "phenols": "flavor_compound",
    "residual_sugars": "sugar",
    "smoke_compounds_equiv": "flavor_compound",
}

# These manual assertions are only used when the local PubChem file exists and RDKit
# successfully parses the structure. Each CID was chosen from the on-disk local
# library for class expansion targets that ETL_03 did not yet expose via aliases.
MANUAL_LOCAL_CID_MAP: Mapping[str, str] = {
    "anethole": "637563",
    "cinnamaldehyde": "637511",
    "ellagic acid": "5281855",
    "eucalyptol": "2758",
    "fenchone": "14525",
    "fructose": "5984",
    "glucose": "5793",
    "guaiacol": "460",
    "histamine": "774",
    "isoamyl acetate": "31276",
    "limonene": "22311",
    "syringaldehyde": "8655",
    "tyramine": "5610",
    "4 vinylphenol": "62453",
}

CLASS_ONTOLOGY: Mapping[str, Tuple[str, ...]] = {
    "esters": (
        "ethyl acetate",
        "isoamyl acetate",
        "ethyl hexanoate",
        "ethyl octanoate",
    ),
    "congeners": (
        "methanol",
        "acetaldehyde",
        "acetone",
        "formaldehyde",
        "furfural",
    ),
    "fusel_alcohols": (
        "isoamyl alcohol",
        "isobutanol",
        "1-propanol",
        "2,3-butanediol",
    ),
    "nitrogen_compounds": (
        "histamine",
        "tyramine",
        "putrescine",
        "cadaverine",
    ),
    "fatty_acids": (
        "acetic acid",
        "butanoic acid",
        "octanoic acid",
    ),
    "organic_acids": (
        "acetic acid",
        "lactic acid",
        "malic acid",
        "citric acid",
        "succinic acid",
        "tartaric acid",
        "pyruvic acid",
    ),
    "polyphenols": (
        "catechin",
        "epicatechin",
        "gallic acid",
        "ellagic acid",
        "quercetin",
        "resveratrol",
        "caffeic acid",
        "ferulic acid",
        "chlorogenic acid",
    ),
    "tannins": (
        "tannic acid",
        "catechin",
        "gallic acid",
        "ellagic acid",
    ),
    "terpenoids": (
        "limonene",
        "linalool",
        "geraniol",
        "nerol",
        "eucalyptol",
        "fenchone",
    ),
    "hop_terpenes": (
        "limonene",
        "linalool",
        "geraniol",
        "eucalyptol",
    ),
    "hop_acids": (
        "humulone",
        "cohumulone",
        "lupulone",
    ),
    "lactones": (
        "whisky lactone",
    ),
    "phenols": (
        "guaiacol",
        "syringaldehyde",
        "4-ethylphenol",
        "4-vinylphenol",
    ),
    "residual_sugars": (
        "glucose",
        "fructose",
        "sucrose",
    ),
    "smoke_compounds_equiv": (
        "guaiacol",
        "syringaldehyde",
    ),
}

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "beverage_id",
    "beverage_name",
    "category",
    "compound_name",
    "normalized_compound_name",
    "pubchem_cid",
    "chemical_category",
    "compound_role",
    "estimated_concentration",
    "concentration_unit",
    "digestion_effect",
    "metabolic_burden",
    "source_compound_class",
    "expansion_type",
    "source_dataset",
    "source_file",
    "source_row",
    "confidence_score",
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def input_matrix_path(root: Path) -> Path:
    return root / INPUT_MATRIX_RELATIVE


def input_report_path(root: Path) -> Path:
    return root / INPUT_REPORT_RELATIVE


def output_matrix_path(root: Path) -> Path:
    path = root / OUTPUT_MATRIX_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_report_path(root: Path) -> Path:
    path = root / OUTPUT_REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def class_key_for_row(row: Mapping[str, Any]) -> str:
    candidates = (
        normalize_compound_name(row.get("normalized_compound_name", "")),
        normalize_compound_name(row.get("compound_name", "")),
    )
    for candidate in candidates:
        key = CLASS_ALIAS_TO_KEY.get(candidate, "")
        if key:
            return key
    return ""


def is_resolved_cid(cid: str, pubchem_index: Mapping[str, Any]) -> bool:
    entry = pubchem_index.get(cid)
    return cid != UNKNOWN and entry is not None and bool(entry.rdkit_valid)


def resolve_representative(
    compound_name: str,
    resolver_index: Mapping[str, Tuple[str, ...]],
    pubchem_index: Mapping[str, Any],
) -> Optional[Dict[str, str]]:
    resolution = resolve_pubchem_cid(compound_name, resolver_index)
    if is_resolved_cid(resolution.cid, pubchem_index):
        return {
            "compound_name": compound_name,
            "normalized_compound_name": normalize_compound_name(compound_name),
            "pubchem_cid": resolution.cid,
            "resolution_strategy": resolution.strategy,
        }

    normalized = normalize_compound_name(compound_name)
    manual_cid = MANUAL_LOCAL_CID_MAP.get(normalized, UNKNOWN)
    if is_resolved_cid(manual_cid, pubchem_index):
        return {
            "compound_name": compound_name,
            "normalized_compound_name": normalized,
            "pubchem_cid": manual_cid,
            "resolution_strategy": "manual_local_cid",
        }

    return None


def build_resolved_ontology(
    resolver_index: Mapping[str, Tuple[str, ...]],
    pubchem_index: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    ontology: Dict[str, Dict[str, Any]] = {}
    for class_key, representatives in CLASS_ONTOLOGY.items():
        resolved: List[Dict[str, str]] = []
        unresolved: List[str] = []
        for representative in representatives:
            record = resolve_representative(representative, resolver_index, pubchem_index)
            if record is None:
                unresolved.append(representative)
                continue
            resolved.append(record)
        ontology[class_key] = {
            "class_key": class_key,
            "chemical_category": CLASS_TO_CATEGORY[class_key],
            "requested_representatives": list(representatives),
            "resolved_representatives": resolved,
            "unresolved_representatives": unresolved,
        }
    return ontology


def normalize_output_row(row: Mapping[str, Any]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for column in OUTPUT_COLUMNS:
        output[column] = clean_text(row.get(column, "")) or UNKNOWN
    return output


def append_output_row(rows: MutableSequence[Dict[str, str]], row: Mapping[str, Any]) -> None:
    rows.append(normalize_output_row(row))


def expand_matrix(
    matrix_df: pd.DataFrame,
    ontology: Mapping[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output_rows: List[Dict[str, str]] = []
    family_source_rows = 0
    representative_rows_added = 0
    expanded_family_rows = 0
    remaining_unexpanded_family_rows = 0
    class_row_counts: Dict[str, int] = {}
    representative_row_counts: Dict[str, int] = {}
    blocked_classes: set[str] = set()

    for _, series in matrix_df.iterrows():
        row = {key: clean_text(value) or UNKNOWN for key, value in series.to_dict().items()}
        class_key = class_key_for_row(row)
        is_family_source = class_key and row.get("pubchem_cid", UNKNOWN) == UNKNOWN

        if not is_family_source:
            row["source_compound_class"] = UNKNOWN
            row["expansion_type"] = "direct"
            append_output_row(output_rows, row)
            continue

        row["source_compound_class"] = row.get("normalized_compound_name", UNKNOWN) or class_key
        row["expansion_type"] = "family_expansion"
        append_output_row(output_rows, row)

        family_source_rows += 1
        class_row_counts[class_key] = class_row_counts.get(class_key, 0) + 1
        resolved_representatives = ontology[class_key]["resolved_representatives"]
        if not resolved_representatives:
            remaining_unexpanded_family_rows += 1
            blocked_classes.add(class_key)
            continue

        expanded_family_rows += 1
        for representative in resolved_representatives:
            expanded_row = dict(row)
            expanded_row["compound_name"] = representative["compound_name"]
            expanded_row["normalized_compound_name"] = representative["normalized_compound_name"]
            expanded_row["pubchem_cid"] = representative["pubchem_cid"]
            expanded_row["chemical_category"] = ontology[class_key]["chemical_category"]
            expanded_row["expansion_type"] = "representative_molecule"
            append_output_row(output_rows, expanded_row)
            representative_rows_added += 1
            representative_row_counts[class_key] = representative_row_counts.get(class_key, 0) + 1

    expanded_df = pd.DataFrame(output_rows, columns=list(OUTPUT_COLUMNS))
    expanded_df = expanded_df.sort_values(
        by=[
            "beverage_id",
            "source_compound_class",
            "normalized_compound_name",
            "expansion_type",
            "source_dataset",
            "source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "family_source_rows": family_source_rows,
        "expanded_family_rows": expanded_family_rows,
        "remaining_unexpanded_family_rows": remaining_unexpanded_family_rows,
        "representative_rows_added": representative_rows_added,
        "class_row_counts": dict(sorted(class_row_counts.items())),
        "representative_row_counts": dict(sorted(representative_row_counts.items())),
        "blocked_classes": sorted(blocked_classes),
    }
    return expanded_df, summary


def compute_metrics(matrix_df: pd.DataFrame, pubchem_index: Mapping[str, Any]) -> Dict[str, Any]:
    total_rows = int(len(matrix_df))
    unique_total = int(matrix_df["normalized_compound_name"].nunique()) if total_rows else 0
    resolved_mask = matrix_df["pubchem_cid"].map(lambda cid: is_resolved_cid(clean_text(cid) or UNKNOWN, pubchem_index))
    matched_compounds = int(matrix_df.loc[resolved_mask, "normalized_compound_name"].nunique()) if total_rows else 0
    unmatched_compounds = int(matrix_df.loc[~resolved_mask, "normalized_compound_name"].nunique()) if total_rows else 0
    resolved_rows = int(resolved_mask.sum()) if total_rows else 0
    pubchem_resolution_rate = round((matched_compounds / unique_total) * 100.0, 4) if unique_total else 0.0
    unknown_category_rate = (
        round(((matrix_df["chemical_category"] == UNKNOWN).sum() / total_rows) * 100.0, 4) if total_rows else 0.0
    )
    return {
        "matched_compounds": matched_compounds,
        "unmatched_compounds": unmatched_compounds,
        "resolved_rows": resolved_rows,
        "pubchem_resolution_rate": pubchem_resolution_rate,
        "unknown_category_rate": unknown_category_rate,
        "total_rows": total_rows,
        "unique_compounds_total": unique_total,
    }


def compute_metric_deltas(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    tracked = (
        "matched_compounds",
        "unmatched_compounds",
        "resolved_rows",
        "pubchem_resolution_rate",
        "unknown_category_rate",
        "total_rows",
    )
    deltas: Dict[str, Any] = {}
    for key in tracked:
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            deltas[key] = {
                "before": before_value,
                "after": after_value,
                "delta": round(float(after_value) - float(before_value), 4),
            }
        else:
            deltas[key] = {"before": before_value, "after": after_value, "delta": None}
    return deltas


def load_previous_etl03_metrics(root: Path) -> Dict[str, Any]:
    path = input_report_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding=ENCODING))
    except Exception:
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def safe_for_etl_04(metrics: Mapping[str, Any], expansion_summary: Mapping[str, Any]) -> bool:
    return (
        expansion_summary.get("remaining_unexpanded_family_rows", 0) == 0
        and float(metrics.get("unknown_category_rate", 100.0)) <= 10.0
    )


def to_relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    configure_logging()
    root = repo_root()
    matrix_path = input_matrix_path(root)
    report_path = output_report_path(root)
    expanded_matrix_path = output_matrix_path(root)

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing ETL_03 matrix: {matrix_path}")

    LOGGER.info("Loading ETL_03 matrix: %s", matrix_path)
    matrix_df = pd.read_csv(matrix_path, dtype=str, keep_default_na=False)
    for column in OUTPUT_COLUMNS:
        if column not in matrix_df.columns:
            matrix_df[column] = UNKNOWN

    chem_module, _ = load_rdkit()
    pubchem_index = load_pubchem_index(pubchem_library_root(root), chem_module)
    resolver_index = build_compound_resolver_index(pubchem_index)
    ontology = build_resolved_ontology(resolver_index, pubchem_index)

    base_metrics = compute_metrics(matrix_df, pubchem_index)
    expanded_df, expansion_summary = expand_matrix(matrix_df, ontology)
    expanded_metrics = compute_metrics(expanded_df, pubchem_index)
    delta_metrics = compute_metric_deltas(base_metrics, expanded_metrics)
    previous_etl03_metrics = load_previous_etl03_metrics(root)

    expanded_df.to_csv(expanded_matrix_path, index=False, encoding=ENCODING)

    ontology_report = {
        class_key: {
            "chemical_category": data["chemical_category"],
            "requested_representatives": data["requested_representatives"],
            "resolved_representatives": data["resolved_representatives"],
            "unresolved_representatives": data["unresolved_representatives"],
            "family_source_rows": expansion_summary["class_row_counts"].get(class_key, 0),
            "representative_rows_added": expansion_summary["representative_row_counts"].get(class_key, 0),
        }
        for class_key, data in sorted(ontology.items())
    }

    safe_flag = safe_for_etl_04(expanded_metrics, expansion_summary)
    report = {
        "metadata": {
            "script": "etl/etl_03c_chemical_class_expansion.py",
            "input_matrix_csv": to_relative(root, matrix_path),
            "input_report_json": str(INPUT_REPORT_RELATIVE),
            "pubchem_library_root": to_relative(root, pubchem_library_root(root)),
            "rdkit_available": chem_module is not None,
        },
        "ontology": ontology_report,
        "expansion_summary": expansion_summary,
        "metrics_before_expansion": base_metrics,
        "metrics_after_expansion": expanded_metrics,
        "resolution_delta": delta_metrics,
        "etl_03_reference_metrics": previous_etl03_metrics,
        "artifacts": {
            "beverage_compound_matrix_expanded_csv": to_relative(root, expanded_matrix_path),
            "chemical_class_expansion_report_json": to_relative(root, report_path),
        },
        "final_decision": {
            "safe_for_etl_04": safe_flag,
            "blocking_classes": expansion_summary["blocked_classes"],
        },
    }

    with report_path.open("w", encoding=ENCODING) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info(
        "Wrote expanded matrix rows=%d -> %s",
        len(expanded_df),
        expanded_matrix_path,
    )
    LOGGER.info(
        "Wrote expansion report -> %s | safe_for_etl_04=%s",
        report_path,
        safe_flag,
    )


if __name__ == "__main__":
    main()
