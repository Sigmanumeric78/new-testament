"""ETL step 4 (refined): merge LD50 data, enhance parsing, and reconstruct SMILES.

- Loads organic chemicals CSV.
- Loads SUB + END_STUDY_REC.* sheets.
- Extracts LD50 values with improved text parsing coverage.
- Reconstructs PubChem SMILES from JSONs when available.
- Computes exact molecular weight and normalized toxicity.
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
UNIT_RE = re.compile(
    r"(mg\s*/\s*kg\s*bw|mg\s*/\s*kg|g\s*/\s*kg\s*bw|g\s*/\s*kg|mg\s*kg-1\s*bw|mg\s*kg-1|g\s*kg-1\s*bw|g\s*kg-1)",
    re.I,
)
QUAL_RE = re.compile(r"(<=|>=|<|>|=)")


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


def classify_smiles(smiles: Optional[str]) -> Tuple[Optional[str], str]:
    """Return canonical SMILES if valid organic, else status label."""
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None, "missing"

    smiles_str = str(smiles).strip()
    if not smiles_str:
        return None, "missing"

    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None, "invalid"

    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "salt"

    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if not has_carbon:
        return None, "inorganic"

    return Chem.MolToSmiles(mol, canonical=True), "ok"


def build_mol_from_pubchem(compound: dict) -> Optional[Chem.Mol]:
    atoms = compound.get("atoms", {})
    elements = atoms.get("element", [])
    aid = atoms.get("aid", [])
    if not elements or not aid or len(elements) != len(aid):
        return None

    id_to_idx = {aid[i]: i for i in range(len(aid))}
    rwmol = Chem.RWMol()
    for elem in elements:
        rwmol.AddAtom(Chem.Atom(int(elem)))

    bonds = compound.get("bonds", {})
    aid1 = bonds.get("aid1", [])
    aid2 = bonds.get("aid2", [])
    order = bonds.get("order", [])

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    for a1, a2, bo in zip(aid1, aid2, order):
        idx1 = id_to_idx.get(a1)
        idx2 = id_to_idx.get(a2)
        if idx1 is None or idx2 is None:
            continue
        bond_type = bond_type_map.get(int(bo), Chem.BondType.SINGLE)
        rwmol.AddBond(idx1, idx2, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            rwmol.GetAtomWithIdx(idx1).SetIsAromatic(True)
            rwmol.GetAtomWithIdx(idx2).SetIsAromatic(True)

    mol = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def reconstruct_pubchem_smiles(pubchem_dir: str) -> pd.DataFrame:
    """Load PubChem JSONs with pandas and reconstruct SMILES from atoms/bonds."""
    records = []
    for path in sorted(glob.glob(os.path.join(pubchem_dir, "*.json"))):
        df = pd.read_json(path)
        pc_compounds = df.get("PC_Compounds")
        compound = None
        if isinstance(pc_compounds, pd.Series) and len(pc_compounds) > 0:
            compound = pc_compounds.iloc[0]
        elif isinstance(pc_compounds, list) and pc_compounds:
            compound = pc_compounds[0]

        if not isinstance(compound, dict):
            records.append(
                {"cid": None, "pubchem_smiles": None, "file": os.path.basename(path)}
            )
            continue

        cid = compound.get("id", {}).get("id", {}).get("cid")
        mol = build_mol_from_pubchem(compound)
        if mol is None:
            records.append(
                {"cid": cid, "pubchem_smiles": None, "file": os.path.basename(path)}
            )
            continue

        smiles = Chem.MolToSmiles(mol, canonical=True)
        records.append(
            {"cid": cid, "pubchem_smiles": smiles, "file": os.path.basename(path)}
        )

    return pd.DataFrame(records)


def extract_ld50_value(
    row: pd.Series,
) -> Tuple[Optional[float], Optional[str], Optional[str], str, str]:
    """Extract LD50 value, qualifier, unit, raw text, and method."""
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
            "effect_level",
        )

    if not remarks_text:
        return None, None, None, "", "none"

    text = remarks_text
    text_lower = text.lower()
    unit_match = UNIT_RE.search(text)
    unit_value = unit_match.group(1) if unit_match else None

    # Range pattern: 0.56 < LD50 < 0.84
    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*<\s*ld50\s*<\s*(\d+(?:\.\d+)?)",
        text_lower,
    )
    if range_match:
        low = float(range_match.group(1))
        return low, "<", unit_value, remarks_text, "text_range"

    # Pattern: LD50 <= 2000
    ld50_match = re.search(
        r"ld50\s*(<=|>=|<|>|=)?\s*(\d+(?:\.\d+)?)",
        text_lower,
    )
    if ld50_match:
        qualifier = ld50_match.group(1)
        value = float(ld50_match.group(2))
        return value, qualifier, unit_value, remarks_text, "text_ld50"

    # Fallback: if LD50 mentioned, take min numeric in text
    if "ld50" in text_lower:
        numbers = [float(n) for n in NUMBER_RE.findall(text_lower)]
        value = min(numbers) if numbers else None
        return value, None, unit_value, remarks_text, "text_any"

    return None, None, unit_value, remarks_text, "none"


def normalize_ld50(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or unit is None:
        return None
    if isinstance(unit, float) and pd.isna(unit):
        return None
    unit_norm = str(unit).lower().replace(" ", "")
    if unit_norm.startswith("mg/kg") or unit_norm.startswith("mgkg-1"):
        return value
    if unit_norm.startswith("g/kg") or unit_norm.startswith("gkg-1"):
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

    extracted = ld50_df.apply(extract_ld50_value, axis=1, result_type="expand")
    extracted.columns = [
        "ld50_value",
        "ld50_qualifier",
        "ld50_unit",
        "ld50_raw_text",
        "ld50_parse_method",
    ]
    ld50_df = pd.concat([ld50_df, extracted], axis=1)
    ld50_df["ld50_value_mg_per_kg"] = ld50_df.apply(
        lambda r: normalize_ld50(r["ld50_value"], r["ld50_unit"]), axis=1
    )

    # Reconstruct PubChem SMILES and merge into organic
    pubchem_df = reconstruct_pubchem_smiles(PUBCHEM_JSON_DIR)
    pubchem_df = pubchem_df.dropna(subset=["cid"]).drop_duplicates(subset=["cid"])
    pubchem_df["cid"] = pubchem_df["cid"].astype(str)

    if "pubchem_cid" in organic.columns:
        organic["pubchem_cid"] = organic["pubchem_cid"].astype(str)
        organic = organic.merge(
            pubchem_df[["cid", "pubchem_smiles"]],
            left_on="pubchem_cid",
            right_on="cid",
            how="left",
        )
    else:
        organic["pubchem_smiles"] = None

    # Validate reconstructed SMILES and fill missing
    pubchem_parsed = organic["pubchem_smiles"].map(classify_smiles)
    organic["pubchem_rdkit_smiles"], organic["pubchem_parse_status"] = zip(
        *pubchem_parsed
    )

    organic["canonical_smiles"] = organic["rdkit_smiles"]
    missing_mask = organic["canonical_smiles"].isna() & (
        organic["pubchem_parse_status"] == "ok"
    )
    organic.loc[missing_mask, "canonical_smiles"] = organic.loc[
        missing_mask, "pubchem_rdkit_smiles"
    ]

    # Join path: Parent UUID -> SUB.Document UUID -> ReferenceSubstance -> organic chemicals
    merged = ld50_df.merge(
        sub_df, left_on="Parent UUID", right_on="substance_uuid", how="left"
    ).merge(organic, on="reference_substance_uuid", how="inner")

    # Calculate exact molecular weight using RDKit
    merged["exact_molecular_weight"] = merged["canonical_smiles"].map(calc_exact_mw)

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

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"LD50 rows (raw): {len(ld50_df)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Output: {OUTPUT_CSV}")
    print("LD50 parse method counts:")
    print(merged["ld50_parse_method"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
