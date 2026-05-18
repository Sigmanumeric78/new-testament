"""ETL parser: extract organic chemicals and save to CSV.

Loads regulatory Excel (REF_SUB) and PubChem JSONs (for audit/metadata),
parses SMILES with RDKit, filters invalid/inorganic/salts, and saves results.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import pandas as pd
from rdkit import Chem


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGULATORY_XLSX = os.path.join(
    BASE_DIR, "data", "raw", "02_regulatory_toxicity", "OFT3.0 export repository.xlsx"
)
PUBCHEM_JSON_DIR = os.path.join(
    BASE_DIR, "data", "raw", "06_pubchem_cheminformatics", "json"
)
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "organic_chemicals.csv")


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

    # Filter salts / mixtures (multiple fragments)
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "salt"

    # Filter inorganic (no carbon atoms)
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if not has_carbon:
        return None, "inorganic"

    return Chem.MolToSmiles(mol, canonical=True), "ok"


def main() -> None:
    # Load regulatory REF_SUB sheet
    ref_sub = pd.read_excel(REGULATORY_XLSX, sheet_name="REF_SUB", engine="openpyxl")

    # Load PubChem JSONs using pandas (metadata only)
    pubchem_meta = load_pubchem_jsons(PUBCHEM_JSON_DIR)

    # Select and rename columns
    columns = {
        "Document UUID": "reference_substance_uuid",
        "MolecularStructuralInfo.SmilesNotation": "smiles",
        "MolecularStructuralInfo.InChl": "inchi",
        "MolecularStructuralInfo.InChIKey": "inchikey",
        "MolecularStructuralInfo.MolecularFormula": "molecular_formula",
        "Inventory.CASNumber": "cas_number",
        "PUBCHEM CID": "pubchem_cid",
    }

    present_cols = [c for c in columns if c in ref_sub.columns]
    ref_sub = ref_sub[present_cols].rename(columns=columns)

    # Parse and filter SMILES
    parsed = ref_sub["smiles"].map(classify_smiles)
    ref_sub["rdkit_smiles"], ref_sub["parse_status"] = zip(*parsed)

    organic = ref_sub[ref_sub["parse_status"] == "ok"].copy()

    # Drop duplicate canonical SMILES
    organic = organic.drop_duplicates(subset=["rdkit_smiles"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    organic.to_csv(OUTPUT_CSV, index=False)

    # Print summary
    status_counts = ref_sub["parse_status"].value_counts(dropna=False)
    print("SMILES parse status counts:")
    print(status_counts.to_string())
    print(f"Saved organic molecules: {len(organic)} -> {OUTPUT_CSV}")

    # Print PubChem load summary (metadata only)
    if not pubchem_meta.empty:
        print("PubChem JSON load summary:")
        print(pubchem_meta["loaded"].value_counts().to_string())


if __name__ == "__main__":
    main()
