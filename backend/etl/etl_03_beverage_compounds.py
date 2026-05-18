"""ETL step 03: beverage compound ingestion.

Builds canonical beverage -> compound ontology matrix from:
- repaired beverage ontology
- beverage compound profile v3
- repaired alcohol compounds digestion dataset
- local PubChem cheminformatics assets
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_03_beverage_compounds")

ENCODING = "utf-8"
UNKNOWN = "unknown"

CANONICAL_CATEGORIES: Tuple[str, ...] = (
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

CHEMICAL_CATEGORIES: Tuple[str, ...] = (
    "active_alcohol",
    "metabolite",
    "congener",
    "fusel_alcohol",
    "ester",
    "polyphenol",
    "sulfite",
    "sugar",
    "organic_acid",
    "flavor_compound",
    "toxic_impurity",
    "unknown",
)
CHEMICAL_CATEGORY_SET: Set[str] = set(CHEMICAL_CATEGORIES)
FUZZY_MATCH_THRESHOLD = 0.9
PUBCHEM_CID_PATTERN = re.compile(r"(?:Conformer3D|Structure2D)_COMPOUND_CID_(\d+)\.(?:json|sdf)$", re.IGNORECASE)

PUBCHEM_FOLDER_TO_CATEGORY: Mapping[str, str] = {
    "congeners": "congener",
    "fusel_alcohols": "fusel_alcohol",
    "esters": "ester",
    "polyphenols": "polyphenol",
    "sulfites": "sulfite",
    "sugars": "sugar",
    "metabolism": "metabolite",
}

# Deterministic compound synonym normalization.
COMPOUND_SYNONYM_MAP: Mapping[str, str] = {
    "ethyl alcohol": "ethanol",
    "ethyl acetate precursor": "ethyl acetate",
    "isoamyl alcohol": "3-methyl-1-butanol",
    "isoamyl alcohol 3 methyl 1 butanol": "3-methyl-1-butanol",
    "isobutanol": "2-methyl-1-propanol",
    "isobutanol 2 methyl 1 propanol": "2-methyl-1-propanol",
    "1 propanol": "1-propanol",
    "2 3 butanediol": "2,3-butanediol",
    "sulphur dioxide": "sulfur dioxide",
    "sulphur dioxide so2": "sulfur dioxide",
    "potassium metabisulphite": "potassium metabisulfite",
    "sorbic acid preservative additive": "sorbic acid",
    "guaiacol 2 methoxyphenol": "guaiacol",
    "diacetyl 2 3 butanedione": "diacetyl",
    "acetic acid ethyl acetate precursor": "acetic acid",
    "cadaverine 1 5 pentanediamine": "cadaverine",
    "putrescine 1 4 butanediamine": "putrescine",
    "whisky lactone 5 butyl 4 methyldihydrofuranone": "whisky lactone",
    "resveratrol cinnamic acid derivatives": "resveratrol",
    "flavonols quercetin myricetin kaempferol glycosides": "flavonols",
    "tannins condensed": "tannins",
    "tannins hydrolyzable": "tannins",
    "carbonation co2": "carbon dioxide",
    "carbon dioxide co2": "carbon dioxide",
    "co2": "carbon dioxide",
    "benzaldehyde equiv": "benzaldehyde",
    "caprylic acid": "octanoic acid",
    "butyric acid": "butanoic acid",
    "caffeic_acid": "caffeic acid",
    "ferulic_acid": "ferulic acid",
    "4_ethylguaiacol": "4 ethylguaiacol",
}

# Deterministic local CID assignments for high-confidence names.
COMPOUND_TO_CID: Mapping[str, str] = {
    "ethanol": "702",
    "methanol": "887",
    "acetaldehyde": "177",
    "acetic acid": "176",
    "acetone": "180",
    "formaldehyde": "712",
    "furfural": "7362",
    "1-propanol": "1031",
    "2,3-butanediol": "263",
    "3-methyl-1-butanol": "31260",
    "2-methyl-1-propanol": "6568",
    "phenethyl alcohol": "6054",
    "ethyl acetate": "8857",
    "lactic acid": "612",
    "citric acid": "311",
    "malic acid": "525",
    "succinic acid": "1110",
    "tartaric acid": "439153",
    "pyruvic acid": "280",
    "carbon dioxide": "280",
    "butanoic acid": "264",
    "octanoic acid": "379",
    "caffeic acid": "689043",
    "ferulic acid": "445858",
    "chlorogenic acid": "1794427",
    "quercetin": "5280343",
    "resveratrol": "445154",
    "diacetyl": "650",
    "sulfur dioxide": "1119",
    "potassium metabisulfite": "24437",
    "vanillin": "1183",
    "gallic acid": "370",
}


@dataclass(frozen=True)
class PubChemEntry:
    cid: str
    local_group: str
    rdkit_valid: bool
    canonical_smiles: str
    source_json_file: str
    source_sdf_file: str
    aliases: Tuple[str, ...]


@dataclass(frozen=True)
class BeverageMatchResult:
    indices: Tuple[int, ...]
    strategy: str
    ambiguous: bool


@dataclass(frozen=True)
class CompoundResolutionResult:
    cid: str
    strategy: str
    confidence: float


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def beverage_reference_path(root: Path) -> Path:
    return root / "data" / "processed" / "beverage" / "reference_tables" / "master_beverage_reference_repaired.csv"


def beverage_compound_profile_path(root: Path) -> Path:
    return root / "data" / "raw" / "07_beverage_knowledge" / "beverage_compound_profile_v3.csv"


def digestion_repaired_candidates(root: Path) -> Tuple[Path, ...]:
    return (
        root / "data" / "interim" / "beverage" / "repaired" / "alcohol_compounds_digestion.csv",
        root
        / "data"
        / "interim"
        / "beverage"
        / "repaired"
        / "data"
        / "raw"
        / "07_beverage_knowledge"
        / "alcohol_compounds_digestion.csv",
    )


def pubchem_library_root(root: Path) -> Path:
    return root / "data" / "raw" / "06_pubchem_cheminformatics"


def matrix_output_path(root: Path) -> Path:
    out = root / "data" / "processed" / "beverage" / "compound_profiles"
    out.mkdir(parents=True, exist_ok=True)
    return out / "beverage_compound_matrix.csv"


def report_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "compound_ingestion_report.json"


def unresolved_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "unresolved_compounds.csv"


def manual_review_beverages_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "manual_review_beverages.csv"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s,()']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_compound_key(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return UNKNOWN
    # remove parenthetical qualifiers: diacetyl (2,3-butanedione) -> diacetyl
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace(",", " ")
    text = re.sub(r"\b(equiv|equivalent|equivalents)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else UNKNOWN


def normalize_compound_name(value: Any) -> str:
    key = normalize_compound_key(value)
    if key in COMPOUND_SYNONYM_MAP:
        canonical = normalize_compound_key(COMPOUND_SYNONYM_MAP[key])
        return canonical if canonical else UNKNOWN
    if key == UNKNOWN:
        return UNKNOWN
    return key


def canonicalize_category(value: Any) -> str:
    text = normalize_text(value).replace(" ", "_")
    if not text:
        return UNKNOWN
    return text if text in CANONICAL_CATEGORIES else UNKNOWN


def parse_beverage_mentions(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    text = re.sub(r"\([^)]*\)", "", text)
    mentions: List[str] = []
    for chunk in text.split(";"):
        for part in chunk.split(","):
            token = clean_text(part)
            if token:
                mentions.append(token)
    unique = sorted(set(mentions), key=lambda x: normalize_text(x))
    return unique


def split_aliases(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    aliases = [clean_text(part) for part in text.split(";")]
    aliases = [a for a in aliases if a]
    return aliases


def concentration_unit_from_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return UNKNOWN
    for pat in ("mg/l", "mg l", "g/l", "g l", "mg/kg", "g/kg", "%", "ppm"):
        if pat in text:
            return pat.replace(" ", "/") if " " in pat else pat
    return UNKNOWN


def boolean_to_text(value: bool) -> str:
    return "true" if value else "false"


def safe_float(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("%", "")
    text = text.replace(",", ".") if re.match(r"^\d+,\d+$", text) else text
    try:
        return float(text)
    except ValueError:
        return None


def resolve_digestion_path(root: Path) -> Path:
    for candidate in digestion_repaired_candidates(root):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No repaired digestion dataset found in candidates: {digestion_repaired_candidates(root)}")


def load_rdkit() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        return Chem, RDLogger
    except Exception:
        return None, None


def build_mol_from_pubchem_compound(compound: Mapping[str, Any], chem_module: Any) -> Optional[Any]:
    atoms = compound.get("atoms", {})
    elements = atoms.get("element", [])
    aid = atoms.get("aid", [])
    if not elements or not aid or len(elements) != len(aid):
        return None

    id_to_idx = {int(aid[i]): int(i) for i in range(len(aid))}
    rwmol = chem_module.RWMol()
    for elem in elements:
        rwmol.AddAtom(chem_module.Atom(int(elem)))

    bonds = compound.get("bonds", {})
    aid1 = bonds.get("aid1", [])
    aid2 = bonds.get("aid2", [])
    order = bonds.get("order", [])
    bond_type_map = {
        1: chem_module.BondType.SINGLE,
        2: chem_module.BondType.DOUBLE,
        3: chem_module.BondType.TRIPLE,
        4: chem_module.BondType.AROMATIC,
    }

    for a1, a2, bo in zip(aid1, aid2, order):
        idx1 = id_to_idx.get(int(a1))
        idx2 = id_to_idx.get(int(a2))
        if idx1 is None or idx2 is None:
            continue
        bond_type = bond_type_map.get(int(bo), chem_module.BondType.SINGLE)
        rwmol.AddBond(int(idx1), int(idx2), bond_type)
        if bond_type == chem_module.BondType.AROMATIC:
            rwmol.GetAtomWithIdx(int(idx1)).SetIsAromatic(True)
            rwmol.GetAtomWithIdx(int(idx2)).SetIsAromatic(True)

    mol = rwmol.GetMol()
    try:
        chem_module.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def extract_cid_from_pubchem_filename(path: Path) -> Optional[str]:
    match = PUBCHEM_CID_PATTERN.search(path.name)
    if match:
        return match.group(1)
    fallback = re.search(r"CID_(\d+)", path.name)
    if fallback:
        return fallback.group(1)
    return None


def extract_aliases_from_pubchem_json(payload: Mapping[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    compounds = payload.get("PC_Compounds", [])
    compound = compounds[0] if compounds and isinstance(compounds[0], dict) else None
    if compound is None:
        return aliases
    for prop in compound.get("props", []):
        if not isinstance(prop, dict):
            continue
        urn = prop.get("urn", {})
        value = prop.get("value", {})
        label = clean_text(urn.get("label", ""))
        sval = clean_text(value.get("sval", ""))
        if not sval:
            continue
        if label in {"IUPAC Name", "Compound"} and re.search(r"[A-Za-z]", sval):
            aliases.add(sval)
    return aliases


def extract_aliases_from_sdf_file(path: Path) -> Set[str]:
    aliases: Set[str] = set()
    try:
        text = path.read_text(encoding=ENCODING, errors="ignore")
    except Exception:
        return aliases

    for tag in ("PUBCHEM_IUPAC_NAME", "PUBCHEM_IUPAC_TRADITIONAL_NAME", "PUBCHEM_MOLECULAR_FORMULA"):
        match = re.search(rf">\s+<{tag}>\n([^\n]+)", text)
        if match:
            value = clean_text(match.group(1))
            if value:
                aliases.add(value)

    if any(normalize_text(alias) == "co2" for alias in aliases):
        aliases.add("carbon dioxide")
    return aliases


def load_pubchem_index(pubchem_root: Path, chem_module: Optional[Any]) -> Dict[str, PubChemEntry]:
    index_build: Dict[str, Dict[str, Any]] = {}
    if not pubchem_root.exists():
        return {}

    for json_path in sorted(pubchem_root.rglob("*.json")):
        cid = extract_cid_from_pubchem_filename(json_path)
        if not cid:
            continue
        entry = index_build.setdefault(
            cid,
            {
                "local_group": json_path.parent.name,
                "rdkit_valid": False,
                "canonical_smiles": UNKNOWN,
                "source_json_file": "",
                "source_sdf_file": "",
                "aliases": set(),
            },
        )
        entry["local_group"] = entry.get("local_group") or json_path.parent.name
        entry["source_json_file"] = json_path.as_posix()
        try:
            payload = json.loads(json_path.read_text(encoding=ENCODING))
            entry["aliases"].update(extract_aliases_from_pubchem_json(payload))
            compounds = payload.get("PC_Compounds", [])
            compound = compounds[0] if compounds and isinstance(compounds[0], dict) else None
            if chem_module is not None and compound is not None:
                mol = build_mol_from_pubchem_compound(compound, chem_module)
                if mol is not None:
                    entry["rdkit_valid"] = True
                    entry["canonical_smiles"] = str(chem_module.MolToSmiles(mol, canonical=True))
        except Exception:
            continue

    for sdf_path in sorted(pubchem_root.rglob("*.sdf")):
        cid = extract_cid_from_pubchem_filename(sdf_path)
        if not cid:
            continue
        entry = index_build.setdefault(
            cid,
            {
                "local_group": sdf_path.parent.name,
                "rdkit_valid": False,
                "canonical_smiles": UNKNOWN,
                "source_json_file": "",
                "source_sdf_file": "",
                "aliases": set(),
            },
        )
        entry["local_group"] = entry.get("local_group") or sdf_path.parent.name
        entry["source_sdf_file"] = sdf_path.as_posix()
        entry["aliases"].update(extract_aliases_from_sdf_file(sdf_path))

    index: Dict[str, PubChemEntry] = {}
    for cid, row in sorted(index_build.items(), key=lambda x: int(x[0])):
        normalized_aliases: Set[str] = set()
        for alias in row.get("aliases", set()):
            normalized = normalize_compound_key(alias)
            if normalized != UNKNOWN:
                normalized_aliases.add(normalized)
        index[cid] = PubChemEntry(
            cid=cid,
            local_group=clean_text(row.get("local_group", "")) or UNKNOWN,
            rdkit_valid=bool(row.get("rdkit_valid", False)),
            canonical_smiles=clean_text(row.get("canonical_smiles", "")) or UNKNOWN,
            source_json_file=clean_text(row.get("source_json_file", "")),
            source_sdf_file=clean_text(row.get("source_sdf_file", "")),
            aliases=tuple(sorted(normalized_aliases)),
        )
    return index


def build_compound_resolver_index(pubchem_index: Mapping[str, PubChemEntry]) -> Dict[str, Tuple[str, ...]]:
    alias_to_cids: Dict[str, Set[str]] = {}

    def add_alias(alias: str, cid: str) -> None:
        alias_key = normalize_compound_name(alias)
        if alias_key == UNKNOWN:
            return
        if not cid or cid == UNKNOWN:
            return
        alias_to_cids.setdefault(alias_key, set()).add(str(cid))

    for alias, cid in COMPOUND_TO_CID.items():
        add_alias(alias, cid)
    for source_alias, canonical_name in COMPOUND_SYNONYM_MAP.items():
        cid = COMPOUND_TO_CID.get(normalize_compound_name(canonical_name), UNKNOWN)
        if cid != UNKNOWN:
            add_alias(source_alias, cid)
    for cid, entry in pubchem_index.items():
        add_alias(f"cid {cid}", cid)
        for alias in entry.aliases:
            add_alias(alias, cid)

    return {key: tuple(sorted(values, key=lambda v: int(v))) for key, values in alias_to_cids.items()}


def resolve_pubchem_cid(
    raw_compound_name: str,
    resolver_index: Mapping[str, Tuple[str, ...]],
) -> CompoundResolutionResult:
    normalized = normalize_compound_name(raw_compound_name)
    if normalized == UNKNOWN:
        return CompoundResolutionResult(cid=UNKNOWN, strategy="empty_compound", confidence=0.0)

    direct = resolver_index.get(normalized, tuple())
    if len(direct) == 1:
        return CompoundResolutionResult(cid=direct[0], strategy="exact_alias", confidence=1.0)
    if len(direct) > 1:
        return CompoundResolutionResult(cid=UNKNOWN, strategy="ambiguous_exact_alias", confidence=0.0)

    best_key = ""
    best_score = 0.0
    tie_for_best = False
    for candidate_key in sorted(resolver_index.keys()):
        score = SequenceMatcher(None, normalized, candidate_key).ratio()
        if score > best_score:
            best_score = score
            best_key = candidate_key
            tie_for_best = False
        elif score == best_score:
            tie_for_best = True

    if best_key and not tie_for_best and best_score >= FUZZY_MATCH_THRESHOLD:
        cids = resolver_index.get(best_key, tuple())
        if len(cids) == 1:
            return CompoundResolutionResult(cid=cids[0], strategy="fuzzy_alias", confidence=round(best_score, 4))
        return CompoundResolutionResult(cid=UNKNOWN, strategy="ambiguous_fuzzy_alias", confidence=round(best_score, 4))

    return CompoundResolutionResult(cid=UNKNOWN, strategy="no_cid_match", confidence=round(best_score, 4))


def build_beverage_lookup(beverage_df: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    by_normalized_name: Dict[str, List[int]] = {}
    by_subcategory: Dict[str, List[int]] = {}
    by_alias: Dict[str, List[int]] = {}
    by_category: Dict[str, List[int]] = {}

    for idx, row in beverage_df.iterrows():
        normalized_name = normalize_text(row.get("normalized_name", ""))
        subcategory = normalize_text(row.get("subcategory", ""))
        category = canonicalize_category(row.get("category", ""))
        aliases = [normalize_text(alias) for alias in split_aliases(row.get("aliases", ""))]
        aliases = [alias for alias in aliases if alias]

        if normalized_name:
            by_normalized_name.setdefault(normalized_name, []).append(int(idx))
        if subcategory:
            by_subcategory.setdefault(subcategory, []).append(int(idx))
        for alias in aliases:
            by_alias.setdefault(alias, []).append(int(idx))
        if category:
            by_category.setdefault(category, []).append(int(idx))

    for mapping in (by_normalized_name, by_subcategory, by_alias, by_category):
        for key, values in mapping.items():
            mapping[key] = sorted(set(values))

    return {
        "by_normalized_name": by_normalized_name,
        "by_subcategory": by_subcategory,
        "by_alias": by_alias,
        "by_category": by_category,
    }


def broad_category_matches(hint_norm: str, lookup: Mapping[str, Dict[str, List[int]]]) -> Tuple[List[int], str]:
    category_map = lookup["by_category"]
    spirit_categories = ["whisky", "vodka", "rum", "gin", "tequila", "brandy", "liqueur"]
    fermented_categories = ["beer", "wine", "cider", "sake", "mead", "hard_seltzer"]

    if hint_norm in category_map:
        return list(category_map[hint_norm]), "category_exact"
    if hint_norm in {"spirit", "spirits"}:
        merged: List[int] = []
        for category in spirit_categories:
            merged.extend(category_map.get(category, []))
        return sorted(set(merged)), "category_spirits"
    if hint_norm in {"fermented", "fermented beverages"}:
        merged = []
        for category in fermented_categories:
            merged.extend(category_map.get(category, []))
        return sorted(set(merged)), "category_fermented"
    if hint_norm in {"all alcoholic beverages", "all beverages", "beverages"}:
        merged = []
        for values in category_map.values():
            merged.extend(values)
        return sorted(set(merged)), "category_all"
    if hint_norm in {"some wines"}:
        return sorted(set(category_map.get("wine", []) + category_map.get("fortified_wine", []))), "category_some_wines"
    if hint_norm in {"fruit brandies", "stone fruit brandies"}:
        return list(category_map.get("brandy", [])), "category_brandy_family"
    if hint_norm in {"some spirits"}:
        merged = []
        for category in spirit_categories:
            merged.extend(category_map.get(category, []))
        return sorted(set(merged)), "category_some_spirits"
    return [], "none"


def match_beverages_by_hint(hint: str, lookup: Mapping[str, Dict[str, List[int]]]) -> BeverageMatchResult:
    hint_norm = normalize_text(hint)
    if not hint_norm:
        return BeverageMatchResult(indices=tuple(), strategy="empty_hint", ambiguous=False)

    by_normalized_name = lookup["by_normalized_name"]
    by_subcategory = lookup["by_subcategory"]
    by_alias = lookup["by_alias"]

    if hint_norm in by_normalized_name:
        indices = tuple(by_normalized_name[hint_norm])
        return BeverageMatchResult(indices=indices, strategy="normalized_name_exact", ambiguous=len(indices) > 1)
    if hint_norm in by_subcategory:
        indices = tuple(by_subcategory[hint_norm])
        return BeverageMatchResult(indices=indices, strategy="subcategory_exact", ambiguous=len(indices) > 1)
    if hint_norm in by_alias:
        indices = tuple(by_alias[hint_norm])
        return BeverageMatchResult(indices=indices, strategy="alias_exact", ambiguous=len(indices) > 1)

    broad_indices, broad_strategy = broad_category_matches(hint_norm, lookup)
    if broad_indices:
        return BeverageMatchResult(indices=tuple(broad_indices), strategy=broad_strategy, ambiguous=True)

    hint_tokens = set(hint_norm.split(" "))
    partial_matches: List[int] = []
    for key, idxs in by_subcategory.items():
        key_tokens = set(key.split(" "))
        if hint_tokens and hint_tokens.issubset(key_tokens):
            partial_matches.extend(idxs)
    if partial_matches:
        indices = tuple(sorted(set(partial_matches)))
        return BeverageMatchResult(indices=indices, strategy="subcategory_token_subset", ambiguous=len(indices) > 1)

    return BeverageMatchResult(indices=tuple(), strategy="no_match", ambiguous=False)


def confidence_from_label(value: Any) -> str:
    label = normalize_text(value)
    if label in {"high", "medium", "low"}:
        return label
    return "medium"


def infer_chemical_category(
    normalized_compound_name: str,
    source_chemical_class: str,
    local_pubchem_group: str,
) -> str:
    if local_pubchem_group in PUBCHEM_FOLDER_TO_CATEGORY:
        mapped = PUBCHEM_FOLDER_TO_CATEGORY[local_pubchem_group]
        if mapped == "metabolite":
            # metabolism library contains mixed classes; specialize by keyword.
            if any(k in normalized_compound_name for k in ("ethanol", "methanol", "propanol", "butanol", "alcohol")):
                return "active_alcohol"
            if any(k in normalized_compound_name for k in ("acid",)):
                return "organic_acid"
            if any(k in normalized_compound_name for k in ("acetaldehyde", "formaldehyde", "nitrosamine")):
                return "toxic_impurity"
            return "metabolite"
        return mapped

    class_norm = normalize_text(source_chemical_class)
    compound = normalized_compound_name

    if any(k in class_norm for k in ("primary alcohol", "higher alcohol", "aromatic alcohol")):
        return "active_alcohol"
    if "fusel" in compound or "higher alcohol" in class_norm:
        return "fusel_alcohol"
    if "ester" in compound or "ester" in class_norm:
        return "ester"
    if any(k in compound for k in ("sulfite", "sulfur dioxide", "metabisulfite")):
        return "sulfite"
    if any(k in compound for k in ("fructose", "glucose", "sugar", "sorbitol", "mannitol", "residual sugars")):
        return "sugar"
    if "polyphenol" in class_norm or any(k in compound for k in ("tannin", "catechin", "resveratrol", "anthocyanin", "quercetin", "gallic")):
        return "polyphenol"
    if any(k in compound for k in ("acid",)):
        return "organic_acid"
    if any(k in compound for k in ("acetaldehyde", "methanol", "formaldehyde", "nitrosamine", "lead")):
        return "toxic_impurity"
    if any(k in class_norm for k in ("volatile", "aromatic", "phenol", "terpene", "flavouring")):
        return "flavor_compound"
    return "unknown"


def parse_compound_source_file(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.as_posix()


def build_base_output_record(
    beverage_row: Mapping[str, Any],
    compound_name: str,
    normalized_compound_name: str,
    pubchem_cid: str,
    chemical_category: str,
    compound_role: str,
    estimated_concentration: str,
    concentration_unit: str,
    digestion_effect: str,
    metabolic_burden: str,
    source_dataset: str,
    source_file: str,
    source_row: str,
    confidence_score: str,
) -> Dict[str, Any]:
    return {
        "beverage_id": clean_text(beverage_row.get("beverage_id", "")) or UNKNOWN,
        "beverage_name": clean_text(beverage_row.get("beverage_name", "")) or UNKNOWN,
        "category": canonicalize_category(beverage_row.get("category", "")),
        "compound_name": compound_name or UNKNOWN,
        "normalized_compound_name": normalized_compound_name or UNKNOWN,
        "pubchem_cid": pubchem_cid or UNKNOWN,
        "chemical_category": chemical_category if chemical_category in CHEMICAL_CATEGORY_SET else UNKNOWN,
        "compound_role": compound_role or UNKNOWN,
        "estimated_concentration": estimated_concentration or UNKNOWN,
        "concentration_unit": concentration_unit or UNKNOWN,
        "digestion_effect": digestion_effect or UNKNOWN,
        "metabolic_burden": metabolic_burden or UNKNOWN,
        "source_dataset": source_dataset,
        "source_file": source_file,
        "source_row": source_row,
        "confidence_score": confidence_score,
    }


def append_manual_review_beverage(
    rows: MutableSequence[Dict[str, Any]],
    source_dataset: str,
    source_file: str,
    source_row: int,
    beverage_hint: str,
    compound_name: str,
    match_strategy: str,
) -> None:
    rows.append(
        {
            "source_dataset": source_dataset,
            "source_file": source_file,
            "source_row": str(source_row),
            "beverage_hint": beverage_hint,
            "compound_name": compound_name,
            "match_strategy": match_strategy,
            "manual_review_required": "true",
        }
    )


def append_unresolved_compound(
    rows: MutableSequence[Dict[str, Any]],
    record: Mapping[str, Any],
    reason: str,
    rdkit_valid: str,
    canonical_smiles: str,
) -> None:
    rows.append(
        {
            "compound_name": clean_text(record.get("compound_name", "")),
            "normalized_compound_name": clean_text(record.get("normalized_compound_name", "")),
            "pubchem_cid": clean_text(record.get("pubchem_cid", "")) or UNKNOWN,
            "chemical_category": clean_text(record.get("chemical_category", "")) or UNKNOWN,
            "resolution_reason": reason,
            "rdkit_valid": rdkit_valid,
            "canonical_smiles": canonical_smiles,
            "source_dataset": clean_text(record.get("source_dataset", "")),
            "source_file": clean_text(record.get("source_file", "")),
            "source_row": clean_text(record.get("source_row", "")),
        }
    )


def process_profile_v3(
    root: Path,
    profile_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
    lookup: Mapping[str, Dict[str, List[int]]],
    pubchem_index: Mapping[str, PubChemEntry],
    resolver_index: Mapping[str, Tuple[str, ...]],
    output_rows: MutableSequence[Dict[str, Any]],
    manual_review_rows: MutableSequence[Dict[str, Any]],
) -> None:
    source_file = parse_compound_source_file(root, beverage_compound_profile_path(root))
    for idx, row in profile_df.iterrows():
        source_row = int(idx) + 2
        beverage_hint = clean_text(row.get("subcategory", ""))
        match = match_beverages_by_hint(beverage_hint, lookup)

        compound_name = clean_text(row.get("compound", ""))
        normalized_compound = normalize_compound_name(compound_name)
        compound_resolution = resolve_pubchem_cid(compound_name, resolver_index)
        pubchem_cid = compound_resolution.cid
        pubchem_entry = pubchem_index.get(pubchem_cid) if pubchem_cid != UNKNOWN else None
        chemical_category = infer_chemical_category(
            normalized_compound_name=normalized_compound,
            source_chemical_class=clean_text(row.get("physiological_role", "")),
            local_pubchem_group=pubchem_entry.local_group if pubchem_entry else "",
        )
        metabolic_burden = "yes" if "metabolic_burden" in normalize_text(row.get("physiological_role", "")) else UNKNOWN

        if not match.indices:
            append_manual_review_beverage(
                manual_review_rows,
                source_dataset="beverage_compound_profile_v3",
                source_file=source_file,
                source_row=source_row,
                beverage_hint=beverage_hint,
                compound_name=compound_name,
                match_strategy=match.strategy,
            )
            continue

        for beverage_idx in match.indices:
            beverage_row = beverage_df.loc[beverage_idx].to_dict()
            record = build_base_output_record(
                beverage_row=beverage_row,
                compound_name=compound_name,
                normalized_compound_name=normalized_compound,
                pubchem_cid=pubchem_cid,
                chemical_category=chemical_category,
                compound_role=clean_text(row.get("physiological_role", "")) or UNKNOWN,
                estimated_concentration=clean_text(row.get("abundance_level", "")) or UNKNOWN,
                concentration_unit="relative_abundance",
                digestion_effect=UNKNOWN,
                metabolic_burden=metabolic_burden,
                source_dataset="beverage_compound_profile_v3",
                source_file=source_file,
                source_row=str(source_row),
                confidence_score=confidence_from_label(row.get("confidence", "")),
            )
            output_rows.append(record)


def process_digestion(
    root: Path,
    digestion_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
    lookup: Mapping[str, Dict[str, List[int]]],
    pubchem_index: Mapping[str, PubChemEntry],
    resolver_index: Mapping[str, Tuple[str, ...]],
    output_rows: MutableSequence[Dict[str, Any]],
    manual_review_rows: MutableSequence[Dict[str, Any]],
) -> None:
    digestion_path = resolve_digestion_path(root)
    source_file = parse_compound_source_file(root, digestion_path)

    for idx, row in digestion_df.iterrows():
        source_row = int(idx) + 2
        mentions = parse_beverage_mentions(row.get("beverages_found_in", ""))
        if not mentions:
            mentions = [UNKNOWN]

        compound_name = clean_text(row.get("compound", ""))
        normalized_compound = normalize_compound_name(compound_name)
        compound_resolution = resolve_pubchem_cid(compound_name, resolver_index)
        pubchem_cid = compound_resolution.cid
        pubchem_entry = pubchem_index.get(pubchem_cid) if pubchem_cid != UNKNOWN else None
        chemical_category = infer_chemical_category(
            normalized_compound_name=normalized_compound,
            source_chemical_class=clean_text(row.get("chemical_class", "")),
            local_pubchem_group=pubchem_entry.local_group if pubchem_entry else "",
        )

        matched_any = False
        for mention in mentions:
            match = match_beverages_by_hint(mention, lookup)
            if not match.indices:
                append_manual_review_beverage(
                    manual_review_rows,
                    source_dataset="alcohol_compounds_digestion_repaired",
                    source_file=source_file,
                    source_row=source_row,
                    beverage_hint=mention,
                    compound_name=compound_name,
                    match_strategy=match.strategy,
                )
                continue

            matched_any = True
            for beverage_idx in match.indices:
                beverage_row = beverage_df.loc[beverage_idx].to_dict()
                mechanism_norm = normalize_text(row.get("mechanism", ""))
                metabolic_burden = "yes" if any(
                    token in mechanism_norm
                    for token in ("metabol", "toxic", "oxidative", "burden", "irritat")
                ) else UNKNOWN
                output_rows.append(
                    build_base_output_record(
                        beverage_row=beverage_row,
                        compound_name=compound_name,
                        normalized_compound_name=normalized_compound,
                        pubchem_cid=pubchem_cid,
                        chemical_category=chemical_category,
                        compound_role=clean_text(row.get("chemical_class", "")) or UNKNOWN,
                        estimated_concentration=clean_text(row.get("typical_concentration", "")) or UNKNOWN,
                        concentration_unit=concentration_unit_from_text(row.get("typical_concentration", "")),
                        digestion_effect=normalize_text(row.get("affects_digestion", "")) or UNKNOWN,
                        metabolic_burden=metabolic_burden,
                        source_dataset="alcohol_compounds_digestion_repaired",
                        source_file=source_file,
                        source_row=str(source_row),
                        confidence_score="high",
                    )
                )

        if not matched_any and mentions == [UNKNOWN]:
            append_manual_review_beverage(
                manual_review_rows,
                source_dataset="alcohol_compounds_digestion_repaired",
                source_file=source_file,
                source_row=source_row,
                beverage_hint=UNKNOWN,
                compound_name=compound_name,
                match_strategy="no_mentions_no_match",
            )


def deduplicate_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [
        "beverage_id",
        "normalized_compound_name",
        "source_dataset",
        "source_row",
    ]
    return df.sort_values(by=keys + ["compound_name"], kind="mergesort").drop_duplicates(subset=keys, keep="first")


def compute_metrics(
    matrix_df: pd.DataFrame,
    manual_review_df: pd.DataFrame,
    unresolved_df: pd.DataFrame,
    pubchem_index: Mapping[str, PubChemEntry],
) -> Dict[str, Any]:
    total_rows = int(len(matrix_df))
    matched_compounds = int(
        matrix_df.loc[matrix_df["pubchem_cid"] != UNKNOWN, "normalized_compound_name"].nunique()
    ) if total_rows else 0
    unmatched_compounds = int(
        matrix_df.loc[matrix_df["pubchem_cid"] == UNKNOWN, "normalized_compound_name"].nunique()
    ) if total_rows else 0

    matched_beverages = int(
        matrix_df.loc[matrix_df["beverage_id"] != UNKNOWN, "beverage_id"].nunique()
    ) if total_rows else 0
    unmatched_beverages = int(
        manual_review_df[["source_dataset", "source_file", "source_row", "beverage_hint"]].drop_duplicates().shape[0]
    ) if not manual_review_df.empty else 0

    unique_compounds_total = int(matrix_df["normalized_compound_name"].nunique()) if total_rows else 0
    pubchem_resolution_rate = round((matched_compounds / unique_compounds_total) * 100.0, 4) if unique_compounds_total else 0.0

    rdkit_resolved_rows = 0
    for cid in matrix_df["pubchem_cid"].tolist():
        if cid in pubchem_index and pubchem_index[cid].rdkit_valid:
            rdkit_resolved_rows += 1
    rdkit_validation_rate = round((rdkit_resolved_rows / total_rows) * 100.0, 4) if total_rows else 0.0

    unknown_category_rate = round(
        ((matrix_df["chemical_category"] == UNKNOWN).sum() / total_rows) * 100.0, 4
    ) if total_rows else 0.0

    safe_for_etl_04 = unmatched_beverages == 0 and unknown_category_rate <= 10.0

    return {
        "matched_compounds": matched_compounds,
        "unmatched_compounds": unmatched_compounds,
        "matched_beverages": matched_beverages,
        "unmatched_beverages": unmatched_beverages,
        "pubchem_resolution_rate": pubchem_resolution_rate,
        "rdkit_validation_rate": rdkit_validation_rate,
        "unknown_category_rate": unknown_category_rate,
        "unresolved_compounds_rows": int(len(unresolved_df)),
        "safe_for_etl_04": safe_for_etl_04,
    }


def serialize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): serialize_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [serialize_payload(v) for v in payload]
    if isinstance(payload, tuple):
        return [serialize_payload(v) for v in payload]
    return payload


def load_previous_metrics(report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding=ENCODING))
    except Exception:
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def compute_metric_deltas(previous: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    tracked = (
        "matched_compounds",
        "unmatched_compounds",
        "pubchem_resolution_rate",
        "rdkit_validation_rate",
        "unknown_category_rate",
    )
    deltas: Dict[str, Any] = {}
    for key in tracked:
        before = previous.get(key)
        after = current.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = round(float(after) - float(before), 4)
            deltas[key] = {"before": before, "after": after, "delta": delta}
        else:
            deltas[key] = {"before": before, "after": after, "delta": None}
    return deltas


def main() -> None:
    configure_logging()
    root = repo_root()

    beverage_path = beverage_reference_path(root)
    profile_path = beverage_compound_profile_path(root)
    digestion_path = resolve_digestion_path(root)
    pubchem_root = pubchem_library_root(root)

    if not beverage_path.exists():
        raise FileNotFoundError(f"Missing beverage reference: {beverage_path}")
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing beverage compound profile: {profile_path}")
    if not digestion_path.exists():
        raise FileNotFoundError(f"Missing repaired digestion dataset: {digestion_path}")

    LOGGER.info("Loading repaired beverage ontology: %s", beverage_path)
    beverage_df = pd.read_csv(beverage_path, dtype=str, keep_default_na=False)
    LOGGER.info("Loading beverage compound profile: %s", profile_path)
    profile_df = pd.read_csv(profile_path, dtype=str, keep_default_na=False)
    LOGGER.info("Loading repaired digestion data: %s", digestion_path)
    digestion_df = pd.read_csv(digestion_path, dtype=str, keep_default_na=False)

    chem_module, _ = load_rdkit()
    rdkit_available = chem_module is not None
    LOGGER.info("RDKit available: %s", rdkit_available)

    pubchem_index = load_pubchem_index(pubchem_root, chem_module)
    resolver_index = build_compound_resolver_index(pubchem_index)
    json_count = sum(1 for entry in pubchem_index.values() if entry.source_json_file)
    sdf_count = sum(1 for entry in pubchem_index.values() if entry.source_sdf_file)
    LOGGER.info(
        "Loaded local PubChem entries: %d (json=%d, sdf=%d, resolver_aliases=%d)",
        len(pubchem_index),
        json_count,
        sdf_count,
        len(resolver_index),
    )

    lookup = build_beverage_lookup(beverage_df)

    output_rows: List[Dict[str, Any]] = []
    manual_review_rows: List[Dict[str, Any]] = []

    process_profile_v3(
        root=root,
        profile_df=profile_df,
        beverage_df=beverage_df,
        lookup=lookup,
        pubchem_index=pubchem_index,
        resolver_index=resolver_index,
        output_rows=output_rows,
        manual_review_rows=manual_review_rows,
    )
    process_digestion(
        root=root,
        digestion_df=digestion_df,
        beverage_df=beverage_df,
        lookup=lookup,
        pubchem_index=pubchem_index,
        resolver_index=resolver_index,
        output_rows=output_rows,
        manual_review_rows=manual_review_rows,
    )

    matrix_df = pd.DataFrame(output_rows)
    if matrix_df.empty:
        raise RuntimeError("No compound links were generated; refusing to emit empty matrix.")

    matrix_df = deduplicate_output(matrix_df)
    matrix_df = matrix_df.sort_values(
        by=["beverage_id", "normalized_compound_name", "source_dataset", "source_row"],
        kind="mergesort",
    )

    unresolved_rows: List[Dict[str, Any]] = []
    for _, record in matrix_df.iterrows():
        cid = clean_text(record.get("pubchem_cid", "")) or UNKNOWN
        if cid == UNKNOWN:
            append_unresolved_compound(
                unresolved_rows,
                record=record.to_dict(),
                reason="missing_pubchem_cid_mapping",
                rdkit_valid=UNKNOWN,
                canonical_smiles=UNKNOWN,
            )
            continue
        pubchem_entry = pubchem_index.get(cid)
        if pubchem_entry is None:
            append_unresolved_compound(
                unresolved_rows,
                record=record.to_dict(),
                reason="cid_not_found_in_local_pubchem",
                rdkit_valid="false",
                canonical_smiles=UNKNOWN,
            )
            continue
        if not pubchem_entry.rdkit_valid:
            append_unresolved_compound(
                unresolved_rows,
                record=record.to_dict(),
                reason="rdkit_structure_unresolved",
                rdkit_valid="false",
                canonical_smiles=pubchem_entry.canonical_smiles,
            )
            continue

    unresolved_df = pd.DataFrame(unresolved_rows)
    if not unresolved_df.empty:
        unresolved_df = unresolved_df.drop_duplicates().sort_values(
            by=["normalized_compound_name", "source_dataset", "source_row"],
            kind="mergesort",
        )

    manual_review_df = pd.DataFrame(manual_review_rows)
    if not manual_review_df.empty:
        manual_review_df = manual_review_df.drop_duplicates().sort_values(
            by=["source_dataset", "source_file", "source_row", "beverage_hint", "compound_name"],
            kind="mergesort",
        )

    matrix_path = matrix_output_path(root)
    report_path = report_output_path(root)
    unresolved_path = unresolved_output_path(root)
    manual_review_path = manual_review_beverages_output_path(root)
    previous_metrics = load_previous_metrics(report_path)

    matrix_df.to_csv(matrix_path, index=False, encoding=ENCODING)
    unresolved_df.to_csv(unresolved_path, index=False, encoding=ENCODING)
    manual_review_df.to_csv(manual_review_path, index=False, encoding=ENCODING)

    metrics = compute_metrics(
        matrix_df=matrix_df,
        manual_review_df=manual_review_df,
        unresolved_df=unresolved_df,
        pubchem_index=pubchem_index,
    )
    resolution_delta = compute_metric_deltas(previous_metrics, metrics)

    report = {
        "metadata": {
            "script": "etl/etl_03_beverage_compounds.py",
            "beverage_reference_file": str(beverage_path.relative_to(root)),
            "compound_profile_file": str(profile_path.relative_to(root)),
            "digestion_file": str(digestion_path.relative_to(root)),
            "pubchem_library_root": str(pubchem_root.relative_to(root)),
            "recursive_pubchem_loading": True,
            "rdkit_available": rdkit_available,
            "total_matrix_rows": int(len(matrix_df)),
        },
        "metrics": metrics,
        "resolution_delta": resolution_delta,
        "artifacts": {
            "beverage_compound_matrix_csv": str(matrix_path.relative_to(root)),
            "compound_ingestion_report_json": str(report_path.relative_to(root)),
            "unresolved_compounds_csv": str(unresolved_path.relative_to(root)),
            "manual_review_beverages_csv": str(manual_review_path.relative_to(root)),
        },
        "final_decision": {
            "safe_for_etl_04": bool(metrics["safe_for_etl_04"]),
        },
    }

    with report_path.open("w", encoding=ENCODING) as fh:
        json.dump(serialize_payload(report), fh, indent=2, sort_keys=True)
        fh.write("\n")

    LOGGER.info("Wrote matrix rows=%d -> %s", len(matrix_df), matrix_path)
    LOGGER.info("Wrote unresolved compounds rows=%d -> %s", len(unresolved_df), unresolved_path)
    LOGGER.info("Wrote manual review beverages rows=%d -> %s", len(manual_review_df), manual_review_path)
    LOGGER.info("Wrote report -> %s | safe_for_etl_04=%s", report_path, metrics["safe_for_etl_04"])


if __name__ == "__main__":
    main()
