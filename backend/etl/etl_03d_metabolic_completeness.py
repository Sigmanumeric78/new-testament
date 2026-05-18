"""ETL step 03d: metabolic completeness validator for beverage compound coverage."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from etl.etl_03_beverage_compounds import ENCODING, UNKNOWN, clean_text

LOGGER = logging.getLogger("etl_03d_metabolic_completeness")

EXPANDED_MATRIX_RELATIVE = Path("data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv")
EXPANSION_REPORT_RELATIVE = Path("data/interim/beverage/chemical_class_expansion_report.json")
OUTPUT_REPORT_RELATIVE = Path("data/interim/beverage/metabolic_completeness_report.json")

PRIORITY_ORDER: Tuple[str, ...] = (
    "critical_metabolism",
    "toxicity_relevant",
    "digestion_relevant",
    "sensory_only",
    "low_priority",
)

REPRESENTATIVE_MOLECULE = "representative molecule"


@dataclass(frozen=True)
class CriticalTarget:
    name: str
    aliases: Tuple[str, ...]
    weight: int
    pbpk_core: bool
    toxicity_weight: int
    priority_class: str
    notes: str


CRITICAL_TARGETS: Tuple[CriticalTarget, ...] = (
    CriticalTarget(
        name="ethanol",
        aliases=("ethanol",),
        weight=5,
        pbpk_core=True,
        toxicity_weight=1,
        priority_class="critical_metabolism",
        notes="Primary exposure driver for beverage metabolism simulation.",
    ),
    CriticalTarget(
        name="methanol",
        aliases=("methanol",),
        weight=4,
        pbpk_core=True,
        toxicity_weight=4,
        priority_class="critical_metabolism",
        notes="Key toxic congener with direct ADME relevance.",
    ),
    CriticalTarget(
        name="acetaldehyde",
        aliases=("acetaldehyde",),
        weight=4,
        pbpk_core=True,
        toxicity_weight=4,
        priority_class="critical_metabolism",
        notes="Primary oxidative ethanol metabolite and toxicity mediator.",
    ),
    CriticalTarget(
        name="acetate",
        aliases=("acetate", "acetic acid", "ethyl acetate"),
        weight=2,
        pbpk_core=True,
        toxicity_weight=1,
        priority_class="critical_metabolism",
        notes="Acetate-equivalent coverage for downstream metabolic handling.",
    ),
    CriticalTarget(
        name="fusel_alcohols",
        aliases=(
            "fusel alcohols",
            "1 propanol",
            "2 3 butanediol",
            "2 methyl 1 propanol",
            "3 methyl 1 butanol",
        ),
        weight=4,
        pbpk_core=True,
        toxicity_weight=2,
        priority_class="critical_metabolism",
        notes="Major higher-alcohol congener family affecting burden and clearance.",
    ),
    CriticalTarget(
        name="organic_acids",
        aliases=(
            "organic acids",
            "acetic acid",
            "citric acid",
            "lactic acid",
            "malic acid",
            "pyruvic acid",
            "succinic acid",
            "tartaric acid",
        ),
        weight=4,
        pbpk_core=True,
        toxicity_weight=1,
        priority_class="critical_metabolism",
        notes="Acid load and gut handling modifiers with broad digestion relevance.",
    ),
    CriticalTarget(
        name="histamine",
        aliases=("histamine",),
        weight=3,
        pbpk_core=True,
        toxicity_weight=3,
        priority_class="critical_metabolism",
        notes="Biogenic amine with direct symptom and gut-response relevance.",
    ),
    CriticalTarget(
        name="tyramine",
        aliases=("tyramine",),
        weight=3,
        pbpk_core=True,
        toxicity_weight=3,
        priority_class="critical_metabolism",
        notes="Biogenic amine relevant to adverse response burden.",
    ),
    CriticalTarget(
        name="sulfites",
        aliases=("sulfites", "sulfur dioxide", "potassium metabisulfite"),
        weight=3,
        pbpk_core=True,
        toxicity_weight=3,
        priority_class="critical_metabolism",
        notes="Preservative/toxicity cofactor with GI and intolerance relevance.",
    ),
    CriticalTarget(
        name="nitrosamines",
        aliases=("nitrosamines", "n nitrosamines", "ndma", "ndea", "dimethylnitrosamine", "diethylnitrosamine"),
        weight=3,
        pbpk_core=False,
        toxicity_weight=4,
        priority_class="toxicity_relevant",
        notes="Important toxicity sentinel, but not required for core beverage PBPK readiness.",
    ),
    CriticalTarget(
        name="sugars",
        aliases=("sugars", "residual sugars", "glucose", "fructose", "sucrose"),
        weight=4,
        pbpk_core=True,
        toxicity_weight=1,
        priority_class="critical_metabolism",
        notes="Gastric-emptying and ethanol-absorption modifiers.",
    ),
    CriticalTarget(
        name="polyphenols",
        aliases=(
            "polyphenols",
            "catechins",
            "anthocyanins",
            "flavonols",
            "tannins",
            "gallic acid",
            "ellagic acid",
            "quercetin",
            "resveratrol",
            "caffeic acid",
            "ferulic acid",
            "chlorogenic acid",
        ),
        weight=1,
        pbpk_core=True,
        toxicity_weight=1,
        priority_class="digestion_relevant",
        notes="Important modulators, but lower-order determinants than primary congeners.",
    ),
    CriticalTarget(
        name="diacetyl",
        aliases=("diacetyl",),
        weight=2,
        pbpk_core=True,
        toxicity_weight=2,
        priority_class="critical_metabolism",
        notes="Relevant fermentative congener with burden/toxicity significance.",
    ),
)

TARGET_BY_ALIAS: Dict[str, CriticalTarget] = {
    alias: target for target in CRITICAL_TARGETS for alias in target.aliases
}

EXPLICIT_CLASSIFICATION: Mapping[str, str] = {
    "4 ethylphenol": "sensory_only",
    "4 ethylguaiacol": "sensory_only",
    "4 vinylphenol": "sensory_only",
    "acetoin": "low_priority",
    "acrolein": "toxicity_relevant",
    "amino acids": "digestion_relevant",
    "anethole": "sensory_only",
    "anthocyanins": "digestion_relevant",
    "ascorbic acid": "digestion_relevant",
    "b vitamins": "low_priority",
    "benzaldehyde": "sensory_only",
    "beta glucans": "digestion_relevant",
    "cadaverine": "toxicity_relevant",
    "caffeine traces": "low_priority",
    "calcium": "low_priority",
    "caramel": "low_priority",
    "caramel compounds": "low_priority",
    "carbonation": "digestion_relevant",
    "carvacrol": "sensory_only",
    "catechins": "digestion_relevant",
    "cinnamaldehyde": "sensory_only",
    "copper": "toxicity_relevant",
    "congeners": "critical_metabolism",
    "coniferaldehyde": "sensory_only",
    "cresols": "toxicity_relevant",
    "dimethyl sulfide": "sensory_only",
    "esters": "sensory_only",
    "eucalyptol": "sensory_only",
    "fatty acids": "critical_metabolism",
    "fenchone": "sensory_only",
    "flavonols": "digestion_relevant",
    "fructose": "critical_metabolism",
    "fusel alcohols": "critical_metabolism",
    "gluten traces": "digestion_relevant",
    "glycerol": "digestion_relevant",
    "guaiacol": "sensory_only",
    "guanosine": "low_priority",
    "histamine": "critical_metabolism",
    "hop acids": "sensory_only",
    "hop terpenes": "sensory_only",
    "hops iso alpha acids": "sensory_only",
    "iron": "low_priority",
    "lactones": "sensory_only",
    "lead": "toxicity_relevant",
    "limonene": "sensory_only",
    "low molecular weight acids": "digestion_relevant",
    "maize congeners": "low_priority",
    "manganese": "low_priority",
    "melanoidins": "digestion_relevant",
    "n nitrosamines": "toxicity_relevant",
    "nitrogen compounds": "critical_metabolism",
    "organic acids": "critical_metabolism",
    "oxalic acid": "digestion_relevant",
    "phenols": "sensory_only",
    "polyphenols": "critical_metabolism",
    "potassium": "low_priority",
    "propionic acid": "digestion_relevant",
    "purines": "digestion_relevant",
    "putrescine": "toxicity_relevant",
    "pyrazines": "sensory_only",
    "pyrroles": "sensory_only",
    "residual sugars": "critical_metabolism",
    "safrole traces": "toxicity_relevant",
    "serotonin": "low_priority",
    "smoke compounds": "sensory_only",
    "sorbitol": "digestion_relevant",
    "sorbic acid": "digestion_relevant",
    "spermidine": "low_priority",
    "spermine": "low_priority",
    "sulfites": "critical_metabolism",
    "sunset yellow fcf": "low_priority",
    "syringaldehyde": "sensory_only",
    "tannins": "digestion_relevant",
    "terpenes": "sensory_only",
    "terpenoids": "sensory_only",
    "tetrahydro beta carboline": "toxicity_relevant",
    "theobromine": "low_priority",
    "thujone": "toxicity_relevant",
    "thymol": "sensory_only",
    "tryptophol": "low_priority",
    "tyramine": "critical_metabolism",
    "tyrosol": "low_priority",
    "uridine": "low_priority",
    "vanillic acid": "sensory_only",
    "whisky lactone": "sensory_only",
    "zinc": "low_priority",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return REPO_ROOT


def expanded_matrix_path(root: Path) -> Path:
    return root / EXPANDED_MATRIX_RELATIVE


def expansion_report_path(root: Path) -> Path:
    return root / EXPANSION_REPORT_RELATIVE


def output_report_path(root: Path) -> Path:
    path = root / OUTPUT_REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding=ENCODING))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text or UNKNOWN


def classify_unresolved_name(name: str) -> str:
    if name in EXPLICIT_CLASSIFICATION:
        return EXPLICIT_CLASSIFICATION[name]
    if name in TARGET_BY_ALIAS:
        return TARGET_BY_ALIAS[name].priority_class
    if "nitrosamine" in name or "amine" in name:
        return "toxicity_relevant"
    if any(token in name for token in ("acid", "sugar", "glucan", "gluten", "amino", "melanoidin")):
        return "digestion_relevant"
    if any(token in name for token in ("terp", "phenol", "aldehyde", "lactone", "hop", "smoke", "ester")):
        return "sensory_only"
    if any(token in name for token in ("lead", "copper", "safrole", "thujone", "acrolein")):
        return "toxicity_relevant"
    return "low_priority"


def build_target_coverage(resolved_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    resolved_names = set(resolved_df["normalized_compound_name"].tolist())
    resolved_sources = set(
        value for value in resolved_df["source_compound_class"].tolist() if value and value != UNKNOWN
    )
    coverage: Dict[str, Dict[str, Any]] = {}
    for target in CRITICAL_TARGETS:
        evidence_names = sorted(alias for alias in target.aliases if alias in resolved_names)
        evidence_source_classes = sorted(alias for alias in target.aliases if alias in resolved_sources)
        covered = bool(evidence_names or evidence_source_classes)
        coverage[target.name] = {
            "name": target.name,
            "aliases": list(target.aliases),
            "weight": target.weight,
            "pbpk_core": target.pbpk_core,
            "toxicity_weight": target.toxicity_weight,
            "priority_class": target.priority_class,
            "covered": covered,
            "evidence_names": evidence_names,
            "evidence_source_classes": evidence_source_classes,
            "notes": target.notes,
        }
    return coverage


def bool_to_text(value: bool) -> str:
    return "true" if value else "false"


def surrogate_target_name(row: pd.Series, target_coverage: Mapping[str, Dict[str, Any]]) -> str:
    normalized_name = normalize(row.get("normalized_compound_name", ""))
    source_class = normalize(row.get("source_compound_class", ""))
    if normalized_name in TARGET_BY_ALIAS:
        return TARGET_BY_ALIAS[normalized_name].name
    if source_class in TARGET_BY_ALIAS:
        return TARGET_BY_ALIAS[source_class].name

    for target_name, coverage in target_coverage.items():
        aliases = set(coverage["aliases"])
        if normalized_name in aliases or source_class in aliases:
            return target_name
    return UNKNOWN


def surrogate_coverage(
    normalized_name: str,
    source_class: str,
    resolved_names: set[str],
    resolved_representative_classes: set[str],
    target_coverage: Mapping[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    if normalized_name in resolved_names:
        return True, "same_name_resolved"
    if source_class and source_class != UNKNOWN and source_class in resolved_representative_classes:
        return True, "source_class_representative_resolved"
    target = TARGET_BY_ALIAS.get(normalized_name) or TARGET_BY_ALIAS.get(source_class)
    if target is not None and target_coverage[target.name]["covered"]:
        return True, f"critical_target_covered:{target.name}"
    return False, "no_resolved_surrogate"


def summarize_unresolved(
    unresolved_df: pd.DataFrame,
    resolved_df: pd.DataFrame,
    target_coverage: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    resolved_names = set(resolved_df["normalized_compound_name"].tolist())
    resolved_representative_classes = set(
        value
        for value in resolved_df.loc[resolved_df["expansion_type"] == REPRESENTATIVE_MOLECULE, "source_compound_class"].tolist()
        if value and value != UNKNOWN
    )

    records: List[Dict[str, Any]] = []
    grouped = unresolved_df.groupby("normalized_compound_name", sort=True)
    for normalized_name, group in grouped:
        source_classes = sorted(
            {
                normalize(value)
                for value in group["source_compound_class"].tolist()
                if normalize(value) != UNKNOWN
            }
        )
        chemical_categories = sorted(
            {
                normalize(value)
                for value in group["chemical_category"].tolist()
                if normalize(value) != UNKNOWN
            }
        )
        source_class = source_classes[0] if source_classes else UNKNOWN
        classification = classify_unresolved_name(normalized_name)
        covered, coverage_basis = surrogate_coverage(
            normalized_name=normalized_name,
            source_class=source_class,
            resolved_names=resolved_names,
            resolved_representative_classes=resolved_representative_classes,
            target_coverage=target_coverage,
        )
        related_target = surrogate_target_name(group.iloc[0], target_coverage)
        records.append(
            {
                "compound_name": clean_text(group.iloc[0].get("compound_name", "")) or normalized_name,
                "normalized_compound_name": normalized_name,
                "classification": classification,
                "row_count": int(len(group)),
                "chemical_categories": chemical_categories,
                "source_compound_classes": source_classes,
                "surrogate_covered": covered,
                "coverage_basis": coverage_basis,
                "related_critical_target": related_target,
            }
        )
    return sorted(records, key=lambda row: (-row["row_count"], row["normalized_compound_name"]))


def score_coverage(target_coverage: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    total_weight = sum(item["weight"] for item in target_coverage.values())
    covered_weight = sum(item["weight"] for item in target_coverage.values() if item["covered"])
    metabolic_coverage_score = round((covered_weight / total_weight) * 100.0, 4) if total_weight else 0.0

    toxicity_total = sum(item["toxicity_weight"] for item in target_coverage.values() if item["toxicity_weight"] > 0)
    toxicity_covered = sum(
        item["toxicity_weight"] for item in target_coverage.values() if item["toxicity_weight"] > 0 and item["covered"]
    )
    toxicity_coverage = round((toxicity_covered / toxicity_total) * 100.0, 4) if toxicity_total else 0.0

    pbpk_core = [item for item in target_coverage.values() if item["pbpk_core"]]
    pbpk_core_missing = [item["name"] for item in pbpk_core if not item["covered"]]
    pbpk_readiness = len(pbpk_core_missing) == 0

    return {
        "metabolic_coverage_score": metabolic_coverage_score,
        "toxicity_coverage": toxicity_coverage,
        "pbpk_readiness": pbpk_readiness,
        "pbpk_core_missing": pbpk_core_missing,
        "covered_weight": covered_weight,
        "total_weight": total_weight,
        "toxicity_weight_covered": toxicity_covered,
        "toxicity_weight_total": toxicity_total,
    }


def current_blocker_assessment(
    expansion_report: Mapping[str, Any],
    unresolved_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    record_by_name = {row["normalized_compound_name"]: row for row in unresolved_records}
    blocker_names = expansion_report.get("final_decision", {}).get("blocking_classes", [])
    assessed: List[Dict[str, Any]] = []
    for blocker in blocker_names:
        normalized = normalize(blocker)
        record = record_by_name.get(normalized)
        classification = record["classification"] if record else classify_unresolved_name(normalized)
        assessed.append(
            {
                "compound_or_class": normalized,
                "classification": classification,
                "surrogate_covered": bool(record["surrogate_covered"]) if record else False,
                "blocks_metabolism_simulation": classification in {"critical_metabolism", "toxicity_relevant", "digestion_relevant"},
            }
        )
    return assessed


def build_report_payload(
    root: Path,
    matrix_path: Path,
    expansion_report_path_value: Path,
    output_path: Path,
    unresolved_records: Sequence[Mapping[str, Any]],
    target_coverage: Mapping[str, Dict[str, Any]],
    coverage_scores: Mapping[str, Any],
    blocker_assessment: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    critical_compounds_missing = [
        {
            "compound_group": item["name"],
            "priority_class": item["priority_class"],
            "weight": item["weight"],
            "pbpk_core": item["pbpk_core"],
            "toxicity_weight": item["toxicity_weight"],
            "aliases": item["aliases"],
            "notes": item["notes"],
        }
        for item in target_coverage.values()
        if not item["covered"]
    ]

    noncritical_compounds_missing = [
        {
            "compound_name": row["compound_name"],
            "normalized_compound_name": row["normalized_compound_name"],
            "classification": row["classification"],
            "row_count": row["row_count"],
            "chemical_categories": row["chemical_categories"],
            "source_compound_classes": row["source_compound_classes"],
        }
        for row in unresolved_records
        if not row["surrogate_covered"] and row["normalized_compound_name"] not in TARGET_BY_ALIAS
    ]

    unresolved_by_priority: Dict[str, List[Dict[str, Any]]] = {key: [] for key in PRIORITY_ORDER}
    for row in unresolved_records:
        unresolved_by_priority[row["classification"]].append(
            {
                "compound_name": row["compound_name"],
                "normalized_compound_name": row["normalized_compound_name"],
                "row_count": row["row_count"],
                "surrogate_covered": row["surrogate_covered"],
                "coverage_basis": row["coverage_basis"],
                "related_critical_target": row["related_critical_target"],
                "chemical_categories": row["chemical_categories"],
                "source_compound_classes": row["source_compound_classes"],
            }
        )

    blocker_blocks_pbpk = any(item["blocks_metabolism_simulation"] for item in blocker_assessment)
    safe_for_etl_04 = (
        bool(coverage_scores["pbpk_readiness"])
        and float(coverage_scores["metabolic_coverage_score"]) >= 90.0
        and not blocker_blocks_pbpk
    )

    rationale: List[str] = []
    if coverage_scores["pbpk_readiness"]:
        rationale.append("All PBPK-core compound groups have resolved direct or representative coverage.")
    else:
        rationale.append("One or more PBPK-core compound groups remain uncovered.")
    if critical_compounds_missing:
        missing_names = ", ".join(item["compound_group"] for item in critical_compounds_missing)
        rationale.append(f"Important uncovered targets remain: {missing_names}.")
    if blocker_assessment:
        blocker_summary = ", ".join(
            f"{item['compound_or_class']}={item['classification']}" for item in blocker_assessment
        )
        rationale.append(f"ETL_03C blockers classify as: {blocker_summary}.")

    return {
        "metadata": {
            "script": "etl/etl_03d_metabolic_completeness.py",
            "input_matrix_csv": str(matrix_path.relative_to(root)),
            "input_expansion_report_json": str(expansion_report_path_value.relative_to(root)),
            "output_report_json": str(output_path.relative_to(root)),
        },
        "coverage": {
            "metabolic_coverage_score": coverage_scores["metabolic_coverage_score"],
            "toxicity_coverage": coverage_scores["toxicity_coverage"],
            "pbpk_readiness": coverage_scores["pbpk_readiness"],
            "pbpk_core_missing": coverage_scores["pbpk_core_missing"],
            "critical_targets_total": len(target_coverage),
            "critical_targets_covered": sum(1 for item in target_coverage.values() if item["covered"]),
            "weighted_targets_covered": coverage_scores["covered_weight"],
            "weighted_targets_total": coverage_scores["total_weight"],
        },
        "critical_target_coverage": {
            key: {
                "covered": value["covered"],
                "pbpk_core": value["pbpk_core"],
                "priority_class": value["priority_class"],
                "weight": value["weight"],
                "toxicity_weight": value["toxicity_weight"],
                "evidence_names": value["evidence_names"],
                "evidence_source_classes": value["evidence_source_classes"],
                "notes": value["notes"],
            }
            for key, value in sorted(target_coverage.items())
        },
        "critical_compounds_missing": critical_compounds_missing,
        "noncritical_compounds_missing": noncritical_compounds_missing,
        "unresolved_compound_classification": unresolved_by_priority,
        "current_blockers_assessment": blocker_assessment,
        "final_decision": {
            "safe_for_etl_04": safe_for_etl_04,
            "rationale": rationale,
        },
    }


def main() -> None:
    configure_logging()
    root = repo_root()
    matrix_path = expanded_matrix_path(root)
    expansion_report_path_value = expansion_report_path(root)
    output_path = output_report_path(root)

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing ETL_03C expanded matrix: {matrix_path}")
    if not expansion_report_path_value.exists():
        raise FileNotFoundError(f"Missing ETL_03C expansion report: {expansion_report_path_value}")

    LOGGER.info("Loading expanded compound matrix: %s", matrix_path)
    matrix_df = pd.read_csv(matrix_path, dtype=str, keep_default_na=False)
    for column in ("normalized_compound_name", "pubchem_cid", "source_compound_class", "expansion_type", "chemical_category", "compound_name"):
        if column not in matrix_df.columns:
            matrix_df[column] = UNKNOWN

    for column in ("normalized_compound_name", "source_compound_class", "chemical_category", "compound_name", "expansion_type"):
        matrix_df[column] = matrix_df[column].map(normalize)
    matrix_df["pubchem_cid"] = matrix_df["pubchem_cid"].map(lambda value: clean_text(value) or UNKNOWN)

    resolved_df = matrix_df.loc[matrix_df["pubchem_cid"] != UNKNOWN].copy()
    unresolved_df = matrix_df.loc[matrix_df["pubchem_cid"] == UNKNOWN].copy()

    target_coverage = build_target_coverage(resolved_df)
    unresolved_records = summarize_unresolved(unresolved_df, resolved_df, target_coverage)
    coverage_scores = score_coverage(target_coverage)
    expansion_report = load_json(expansion_report_path_value)
    blocker_assessment = current_blocker_assessment(expansion_report, unresolved_records)

    report = build_report_payload(
        root=root,
        matrix_path=matrix_path,
        expansion_report_path_value=expansion_report_path_value,
        output_path=output_path,
        unresolved_records=unresolved_records,
        target_coverage=target_coverage,
        coverage_scores=coverage_scores,
        blocker_assessment=blocker_assessment,
    )

    with output_path.open("w", encoding=ENCODING) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info(
        "Wrote metabolic completeness report -> %s | safe_for_etl_04=%s",
        output_path,
        report["final_decision"]["safe_for_etl_04"],
    )


if __name__ == "__main__":
    main()
