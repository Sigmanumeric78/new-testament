"""ETL step 4: merge LD50 data and compute toxicity math.

Loads organic chemicals, links LD50 endpoints via SUB -> REF_SUB join path,
extracts numeric LD50 values, computes exact molecular weight, and derives
normalized toxicity using -log10(Toxicity_Index_mmol).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import pandas as pd
import sympy as sp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGULATORY_XLSX = os.path.join(
    BASE_DIR, "data", "raw", "02_regulatory_toxicity", "OFT3.0 export repository.xlsx"
)
PUBCHEM_JSON_DIR = os.path.join(
    BASE_DIR, "data", "raw", "06_pubchem_cheminformatics", "json"
)
ORGANIC_CSV = os.path.join(BASE_DIR, "data", "processed", "organic_chemicals.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "standardized_toxicity.csv")

LD50_LABEL = "LD50"

NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?")
UNIT_RE = re.compile(r"(mg\s*/\s*kg\s*bw|mg\s*/\s*kg|g\s*/\s*kg\s*bw|g\s*/\s*kg)", re.I)
QUAL_RE = re.compile(r"(<=|>=|<|>|=)")


def load_pubchem_jsons(pubchem_dir: str) -> pd.DataFrame:
    """Load PubChem JSONs using pandas for traceability."""
    records = []
    for path in sorted(glob.glob(os.path.join(pubchem_dir, "*.json"))):
        try:
            df = pd.read_json(path)
            cid = None
            try:
                pc_compounds = df.get("PC_Compounds")
                if pc_compounds is not None and len(pc_compounds) > 0:
                    first = pc_compounds.iloc[0]
                    if isinstance(first, list) and first:
                        cid = first[0].get("id", {}).get("id", {}).get("cid")
            except Exception:
                cid = None
            records.append({"file": os.path.basename(path), "cid": cid, "loaded": True})
        except ValueError as exc:
            records.append(
                {
                    "file": os.path.basename(path),
                    "cid": None,
                    "loaded": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(records)


def read_sheet_with_usecols(sheet_name: str, required_cols: list[str]) -> pd.DataFrame:
    """Read an Excel sheet using only columns that exist."""
    header_df = pd.read_excel(
        REGULATORY_XLSX, sheet_name=sheet_name, engine="openpyxl", nrows=0
    )
    available = [c for c in required_cols if c in header_df.columns]
    missing = [c for c in required_cols if c not in header_df.columns]
    if missing:
        print(f"Warning: missing columns in {sheet_name}: {missing}")
    if not available:
        return pd.DataFrame()
    return pd.read_excel(
        REGULATORY_XLSX, sheet_name=sheet_name, engine="openpyxl", usecols=available
    )


def extract_ld50_value(
    row: pd.Series,
) -> Tuple[Optional[float], Optional[str], Optional[str], str]:
    """Extract LD50 value (mg/kg if possible), qualifier, unit, and raw text."""
    lower_val = row.get(
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.lowerValue"
    )
    upper_val = row.get(
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.upperValue"
    )
    lower_qual = row.get(
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.lowerQualifier"
    )
    upper_qual = row.get(
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.upperQualifier"
    )
    unit = row.get("ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.Unit")
    unit_other = row.get(
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.Unit.Other"
    )

    remarks_fields = [
        row.get("ResultsAndDiscussion.EffectLevels.Efflevel.RemarksOnResults.Other"),
        row.get("ResultsAndDiscussion.EffectLevels.Efflevel.RemarksOnResults.Remarks"),
        row.get("ResultsAndDiscussion.EffectLevels.RemarksOnResults.Other"),
        row.get("ResultsAndDiscussion.EffectLevels.RemarksOnResults.Remarks"),
    ]
    cleaned_fields = []
    for x in remarks_fields:
        if x is None:
            continue
        if isinstance(x, float) and pd.isna(x):
            continue
        text = str(x).strip()
        if text:
            cleaned_fields.append(text)
    remarks_text = " ".join(cleaned_fields)

    # Prefer numeric effect level fields if populated
    if pd.notna(lower_val) or pd.notna(upper_val):
        value = lower_val if pd.notna(lower_val) else upper_val
        qualifier = lower_qual if pd.notna(lower_qual) else upper_qual
        unit_value = unit if pd.notna(unit) else unit_other
        return (
            float(value),
            str(qualifier) if pd.notna(qualifier) else None,
            str(unit_value) if pd.notna(unit_value) else None,
            remarks_text,
        )

    # Parse from remarks
    qualifier = None
    if remarks_text:
        qual_match = QUAL_RE.search(remarks_text)
        if qual_match:
            qualifier = qual_match.group(1)

    numbers = (
        [float(n) for n in NUMBER_RE.findall(remarks_text)] if remarks_text else []
    )
    value = min(numbers) if numbers else None

    unit_match = UNIT_RE.search(remarks_text) if remarks_text else None
    unit_value = unit_match.group(1) if unit_match else None

    return value, qualifier, unit_value, remarks_text


def normalize_ld50(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or unit is None:
        return None
    if isinstance(unit, float) and pd.isna(unit):
        return None
    unit_norm = str(unit).lower().replace(" ", "")
    if unit_norm.startswith("mg/kg"):
        return value
    if unit_norm.startswith("g/kg"):
        return value * 1000.0
    return None


def calc_exact_mw(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(rdMolDescriptors.CalcExactMolWt(mol))


def compute_normalized_toxicity(tox_index: Optional[float]) -> Optional[float]:
    if tox_index is None or tox_index <= 0:
        return None
    return float(-sp.log(tox_index, 10))


def main() -> None:
    organic = pd.read_csv(ORGANIC_CSV)

    # Load regulatory sheets
    sub_cols = ["Document UUID", "ReferenceSubstance.ReferenceSubstance"]
    sub_df = read_sheet_with_usecols("SUB", sub_cols).rename(
        columns={
            "Document UUID": "substance_uuid",
            "ReferenceSubstance.ReferenceSubstance": "reference_substance_uuid",
        }
    )

    end_cols = [
        "Document UUID",
        "Parent UUID",
        "AdministrativeData.Endpoint",
        "ResultsAndDiscussion.EffectLevels.Endpoint",
        "ResultsAndDiscussion.EffectLevels.Endpoint.Other",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.lowerQualifier",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.lowerValue",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.upperQualifier",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.upperValue",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.Unit",
        "ResultsAndDiscussion.EffectLevels.Efflevel.EffectLevel.Unit.Other",
        "ResultsAndDiscussion.EffectLevels.Efflevel.RemarksOnResults.Other",
        "ResultsAndDiscussion.EffectLevels.Efflevel.RemarksOnResults.Remarks",
        "ResultsAndDiscussion.EffectLevels.RemarksOnResults.Other",
        "ResultsAndDiscussion.EffectLevels.RemarksOnResults.Remarks",
    ]

    human_df = read_sheet_with_usecols("END_STUDY_REC.HumanHealth", end_cols)
    human_df["source_sheet"] = "END_STUDY_REC.HumanHealth"

    terr_df = read_sheet_with_usecols("END_STUDY_REC.TerrestEcotox", end_cols)
    terr_df["source_sheet"] = "END_STUDY_REC.TerrestEcotox"

    # Load PubChem JSONs with pandas (metadata only)
    pubchem_meta = load_pubchem_jsons(PUBCHEM_JSON_DIR)

    # Filter LD50 rows
    def is_ld50(df: pd.DataFrame) -> pd.Series:
        endpoint_col = "ResultsAndDiscussion.EffectLevels.Endpoint"
        endpoint_other_col = "ResultsAndDiscussion.EffectLevels.Endpoint.Other"
        if df.empty:
            return pd.Series(dtype=bool)
        endpoint = (
            df[endpoint_col]
            if endpoint_col in df.columns
            else pd.Series([None] * len(df))
        )
        endpoint_other = (
            df[endpoint_other_col]
            if endpoint_other_col in df.columns
            else pd.Series([None] * len(df))
        )
        return (endpoint == LD50_LABEL) | (endpoint_other == LD50_LABEL)

    ld50_df = pd.concat(
        [human_df[is_ld50(human_df)], terr_df[is_ld50(terr_df)]], ignore_index=True
    )

    # Extract LD50 values
    extracted = ld50_df.apply(extract_ld50_value, axis=1, result_type="expand")
    extracted.columns = ["ld50_value", "ld50_qualifier", "ld50_unit", "ld50_raw_text"]
    ld50_df = pd.concat([ld50_df, extracted], axis=1)
    ld50_df["ld50_value_mg_per_kg"] = ld50_df.apply(
        lambda r: normalize_ld50(r["ld50_value"], r["ld50_unit"]), axis=1
    )

    # Join path: Parent UUID -> SUB.Document UUID -> ReferenceSubstance -> organic chemicals
    merged = ld50_df.merge(
        sub_df, left_on="Parent UUID", right_on="substance_uuid", how="left"
    ).merge(organic, on="reference_substance_uuid", how="inner")

    # Calculate exact molecular weight using RDKit
    merged["exact_molecular_weight"] = merged["rdkit_smiles"].map(calc_exact_mw)

    # Toxicity math
    merged["Toxicity_Index_mmol"] = merged.apply(
        lambda r: (
            r["ld50_value_mg_per_kg"] / r["exact_molecular_weight"]
            if pd.notna(r["ld50_value_mg_per_kg"])
            and pd.notna(r["exact_molecular_weight"])
            else None
        ),
        axis=1,
    )
    merged["Normalized_Toxicity"] = merged["Toxicity_Index_mmol"].map(
        compute_normalized_toxicity
    )

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)

    # Basic summary
    print(f"LD50 rows (raw): {len(ld50_df)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Output: {OUTPUT_CSV}")
    if not pubchem_meta.empty:
        print("PubChem JSON load summary:")
        print(pubchem_meta["loaded"].value_counts().to_string())


if __name__ == "__main__":
    main()
