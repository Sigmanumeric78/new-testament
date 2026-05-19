"""Deterministic local chemical catalog for explorer endpoints."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.config import get_data_root, get_project_root

MATRIX_DATA_RELATIVE_PATH = Path("processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv")
PUBCHEM_JSON_DATA_RELATIVE_PATH = Path("raw/06_pubchem_cheminformatics/json")
PUBCHEM_SDF_DATA_RELATIVE_PATH = Path("raw/06_pubchem_cheminformatics/sdf")

CID_PATTERN = re.compile(r"CID_(\d+)")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "nan", "unknown"}:
        return ""
    return text


def _normalize_token(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value).lower()).strip()


def _safe_int(value: Any) -> Optional[int]:
    text = _clean_text(value)
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _slug(text: str) -> str:
    token = _normalize_token(text)
    if not token:
        return "unknown"
    token = re.sub(r"[^a-z0-9]+", "-", token).strip("-")
    return token or "unknown"


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalize_token(value)
    return text in {"1", "true", "yes", "y"}


@dataclass
class _PubChemAssets:
    json_3d: List[Path] = field(default_factory=list)
    json_2d: List[Path] = field(default_factory=list)
    sdf_3d: List[Path] = field(default_factory=list)
    sdf_2d: List[Path] = field(default_factory=list)

    def available_formats(self) -> List[str]:
        values: List[str] = []
        if self.json_3d:
            values.append("json_3d")
        if self.json_2d:
            values.append("json_2d")
        if self.sdf_3d:
            values.append("sdf_3d")
        if self.sdf_2d:
            values.append("sdf_2d")
        return values


@dataclass
class _CompoundAggregate:
    names: Counter = field(default_factory=Counter)
    classes: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    beverages: set = field(default_factory=set)
    cid: Optional[int] = None
    normalized_name: str = ""


class ChemicalCatalog:
    """In-memory deterministic catalog assembled from local processed artifacts."""

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        matrix_path: Optional[Path] = None,
        pubchem_json_root: Optional[Path] = None,
        pubchem_sdf_root: Optional[Path] = None,
    ) -> None:
        self.project_root = project_root or get_project_root()
        self.data_root = (self.project_root / "data") if project_root is not None else get_data_root()
        self.matrix_path = matrix_path or (self.data_root / MATRIX_DATA_RELATIVE_PATH)
        self.pubchem_json_root = pubchem_json_root or (self.data_root / PUBCHEM_JSON_DATA_RELATIVE_PATH)
        self.pubchem_sdf_root = pubchem_sdf_root or (self.data_root / PUBCHEM_SDF_DATA_RELATIVE_PATH)

        self._loaded = False
        self._records: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._pubchem_assets: Dict[int, _PubChemAssets] = {}

    @property
    def chemicals_count(self) -> int:
        self.load()
        return len(self._records)

    @property
    def compounds_with_3d_count(self) -> int:
        self.load()
        return sum(1 for item in self._records if bool(item.get("has_3d_conformer")))

    def load(self) -> None:
        if self._loaded:
            return

        self._pubchem_assets = self._scan_pubchem_assets()
        aggregates = self._read_matrix_aggregates()

        records: List[Dict[str, Any]] = []
        used_ids: set = set()

        for aggregate in aggregates:
            compound_name = aggregate.names.most_common(1)[0][0] if aggregate.names else aggregate.normalized_name
            normalized_name = aggregate.normalized_name or _normalize_token(compound_name)
            cid = aggregate.cid

            compound_id = self._make_compound_id(cid=cid, normalized_name=normalized_name, used_ids=used_ids)
            used_ids.add(compound_id)

            chemical_class = self._dominant_value(aggregate.classes) or self._dominant_value(aggregate.categories) or "unknown"
            source_compound_class = self._dominant_value(aggregate.classes) or "unknown"
            chemical_category = self._dominant_value(aggregate.categories) or "unknown"

            pubchem = self._pubchem_assets.get(cid) if cid is not None else None
            has_3d = bool(pubchem and (pubchem.sdf_3d or pubchem.json_3d))
            available_formats = pubchem.available_formats() if pubchem else []

            canonical_smiles = ""
            if cid is not None and pubchem:
                canonical_smiles = self._extract_smiles(pubchem)

            rdkit_valid = bool(canonical_smiles)
            beverages_sorted = sorted({_clean_text(item) for item in aggregate.beverages if _clean_text(item)})
            beverage_examples = beverages_sorted[:6]

            record = {
                "compound_id": compound_id,
                "compound_name": compound_name,
                "normalized_compound_name": normalized_name,
                "pubchem_cid": cid,
                "chemical_class": chemical_class,
                "canonical_smiles": canonical_smiles or None,
                "rdkit_valid": bool(rdkit_valid),
                "beverage_count": len(beverages_sorted),
                "beverage_examples": beverage_examples,
                "has_3d_conformer": bool(has_3d),
                "related_beverages": beverages_sorted,
                "source_compound_class": source_compound_class,
                "metabolism_relevance": self._infer_metabolism_relevance(chemical_class, chemical_category, compound_name),
                "toxicity_relevance": self._infer_toxicity_relevance(chemical_class, chemical_category, compound_name),
                "available_structure_formats": available_formats,
                "conformer_availability_summary": {
                    "has_3d_conformer": bool(has_3d),
                    "has_2d_structure": bool(pubchem and (pubchem.sdf_2d or pubchem.json_2d)),
                    "available_formats": available_formats,
                },
                "_cid": cid,
            }
            records.append(record)

        records.sort(key=lambda item: (_normalize_token(item.get("normalized_compound_name")), _normalize_token(item.get("compound_id"))))
        self._records = records
        self._by_id = {item["compound_id"]: item for item in records}
        self._loaded = True

    def list_chemicals(
        self,
        *,
        q: str = "",
        chemical_class: str = "",
        has_3d: Optional[bool] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> Dict[str, Any]:
        self.load()

        q_norm = _normalize_token(q)
        class_norm = _normalize_token(chemical_class)

        filtered: List[Dict[str, Any]] = []
        for record in self._records:
            if q_norm:
                haystack = " ".join(
                    [
                        _clean_text(record.get("compound_name")),
                        _clean_text(record.get("normalized_compound_name")),
                        str(record.get("pubchem_cid") or ""),
                    ]
                ).lower()
                if q_norm not in haystack:
                    continue
            if class_norm and class_norm != _normalize_token(record.get("chemical_class")):
                continue
            if has_3d is not None and bool(record.get("has_3d_conformer")) != bool(has_3d):
                continue
            filtered.append(self._public_summary(record))

        total = len(filtered)
        start = max(int(offset), 0)
        end = start + max(int(limit), 0)
        page = filtered[start:end]

        return {
            "items": page,
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
        }

    def get_compound(self, compound_id: str) -> Optional[Dict[str, Any]]:
        self.load()
        record = self._by_id.get(_clean_text(compound_id))
        if not record:
            return None

        detail = dict(self._public_summary(record))
        detail.update(
            {
                "related_beverages": list(record.get("related_beverages", [])),
                "source_compound_class": _clean_text(record.get("source_compound_class")) or "unknown",
                "metabolism_relevance": _clean_text(record.get("metabolism_relevance")) or "unknown",
                "toxicity_relevance": _clean_text(record.get("toxicity_relevance")) or "unknown",
                "available_structure_formats": list(record.get("available_structure_formats", [])),
                "conformer_availability_summary": dict(record.get("conformer_availability_summary", {})),
            }
        )
        return detail

    def get_conformer_payload(self, compound_id: str) -> Dict[str, Any]:
        self.load()
        record = self._by_id.get(_clean_text(compound_id))
        if not record:
            return {
                "compound_id": _clean_text(compound_id),
                "has_3d_conformer": False,
                "format": None,
                "sdf": None,
                "message": "Compound not found.",
            }

        cid = record.get("_cid")
        if cid is None:
            return {
                "compound_id": record.get("compound_id"),
                "has_3d_conformer": False,
                "format": None,
                "sdf": None,
                "message": "3D conformer not available for this compound.",
            }

        assets = self._pubchem_assets.get(int(cid))
        if not assets:
            return {
                "compound_id": record.get("compound_id"),
                "has_3d_conformer": False,
                "format": None,
                "sdf": None,
                "message": "3D conformer not available for this compound.",
            }

        if assets.sdf_3d:
            sdf_text = assets.sdf_3d[0].read_text(encoding="utf-8", errors="replace")
            return {
                "compound_id": record.get("compound_id"),
                "has_3d_conformer": True,
                "format": "sdf",
                "sdf": sdf_text,
                "message": "ok",
            }

        return {
            "compound_id": record.get("compound_id"),
            "has_3d_conformer": False,
            "format": None,
            "sdf": None,
            "message": "3D conformer not available for this compound.",
        }

    def _public_summary(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "compound_id": _clean_text(record.get("compound_id")),
            "compound_name": _clean_text(record.get("compound_name")),
            "normalized_compound_name": _clean_text(record.get("normalized_compound_name")),
            "pubchem_cid": record.get("pubchem_cid"),
            "chemical_class": _clean_text(record.get("chemical_class")) or "unknown",
            "canonical_smiles": record.get("canonical_smiles"),
            "rdkit_valid": bool(record.get("rdkit_valid")),
            "beverage_count": int(record.get("beverage_count") or 0),
            "beverage_examples": list(record.get("beverage_examples", [])),
            "has_3d_conformer": bool(record.get("has_3d_conformer")),
        }

    def _scan_pubchem_assets(self) -> Dict[int, _PubChemAssets]:
        asset_map: Dict[int, _PubChemAssets] = {}

        def register(path: Path, *, is_json: bool) -> None:
            match = CID_PATTERN.search(path.name)
            if not match:
                return
            cid = int(match.group(1))
            slot = asset_map.setdefault(cid, _PubChemAssets())
            is_3d = "Conformer3D_" in path.name
            if is_json and is_3d:
                slot.json_3d.append(path)
            elif is_json:
                slot.json_2d.append(path)
            elif is_3d:
                slot.sdf_3d.append(path)
            else:
                slot.sdf_2d.append(path)

        if self.pubchem_json_root.exists():
            for path in sorted(self.pubchem_json_root.rglob("*.json")):
                register(path, is_json=True)

        if self.pubchem_sdf_root.exists():
            for path in sorted(self.pubchem_sdf_root.rglob("*.sdf")):
                register(path, is_json=False)

        return asset_map

    def _read_matrix_aggregates(self) -> List[_CompoundAggregate]:
        if not self.matrix_path.exists():
            return []

        grouped: Dict[Tuple[str, Optional[int]], _CompoundAggregate] = {}
        with self.matrix_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_name = _normalize_token(row.get("normalized_compound_name") or row.get("compound_name"))
                if not normalized_name:
                    continue
                cid = _safe_int(row.get("pubchem_cid"))
                key = (normalized_name, cid)
                item = grouped.setdefault(key, _CompoundAggregate(cid=cid, normalized_name=normalized_name))

                compound_name = _clean_text(row.get("compound_name")) or normalized_name
                item.names[compound_name] += 1

                source_class = _clean_text(row.get("source_compound_class"))
                chem_category = _clean_text(row.get("chemical_category"))
                if source_class:
                    item.classes[source_class] += 1
                if chem_category:
                    item.categories[chem_category] += 1

                beverage_name = _clean_text(row.get("beverage_name"))
                if beverage_name:
                    item.beverages.add(beverage_name)

        return list(grouped.values())

    def _extract_smiles(self, assets: _PubChemAssets) -> str:
        candidate_paths = assets.json_2d + assets.json_3d
        for path in candidate_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            compounds = payload.get("PC_Compounds") if isinstance(payload, Mapping) else None
            if not isinstance(compounds, list) or not compounds:
                continue

            first = compounds[0]
            if not isinstance(first, Mapping):
                continue

            for prop in list(first.get("props", []) or []):
                if not isinstance(prop, Mapping):
                    continue
                urn = prop.get("urn") if isinstance(prop.get("urn"), Mapping) else {}
                if _normalize_token(urn.get("label")) != "smiles":
                    continue
                value = prop.get("value") if isinstance(prop.get("value"), Mapping) else {}
                smiles = _clean_text(value.get("sval"))
                if smiles:
                    return smiles

        return ""

    def _make_compound_id(self, *, cid: Optional[int], normalized_name: str, used_ids: set) -> str:
        if cid is not None:
            base = f"cmp-cid-{cid}"
        else:
            base = f"cmp-name-{_slug(normalized_name)}"

        if base not in used_ids:
            return base

        index = 2
        while True:
            candidate = f"{base}-{index}"
            if candidate not in used_ids:
                return candidate
            index += 1

    @staticmethod
    def _dominant_value(counter: Counter) -> str:
        if not counter:
            return ""
        return counter.most_common(1)[0][0]

    @staticmethod
    def _infer_metabolism_relevance(chemical_class: str, chemical_category: str, compound_name: str) -> str:
        text = " ".join([chemical_class, chemical_category, compound_name]).lower()
        if any(token in text for token in ("ethanol", "acetaldehyde", "fusel", "alcohol", "ester")):
            return "likely_relevant"
        if any(token in text for token in ("sugar", "polyphenol", "sulfite")):
            return "context_dependent"
        return "unknown"

    @staticmethod
    def _infer_toxicity_relevance(chemical_class: str, chemical_category: str, compound_name: str) -> str:
        text = " ".join([chemical_class, chemical_category, compound_name]).lower()
        if any(token in text for token in ("sulfite", "histamine", "tyramine", "congener", "acetaldehyde")):
            return "likely_relevant"
        if any(token in text for token in ("polyphenol", "fusel", "acid")):
            return "context_dependent"
        return "unknown"


_CACHED_CATALOG: Optional[ChemicalCatalog] = None


def get_chemical_catalog() -> ChemicalCatalog:
    global _CACHED_CATALOG
    if _CACHED_CATALOG is None:
        _CACHED_CATALOG = ChemicalCatalog()
        _CACHED_CATALOG.load()
    return _CACHED_CATALOG


def reset_chemical_catalog_cache() -> None:
    global _CACHED_CATALOG
    _CACHED_CATALOG = None
