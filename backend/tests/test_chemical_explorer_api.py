from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.chemical_routes import get_chemical_conformer, get_chemical_detail, list_chemicals
from services.chemical_catalog import ChemicalCatalog


def _required_list_fields(item: Dict[str, Any]) -> None:
    required = {
        "compound_id",
        "compound_name",
        "normalized_compound_name",
        "pubchem_cid",
        "chemical_class",
        "canonical_smiles",
        "rdkit_valid",
        "beverage_count",
        "beverage_examples",
        "has_3d_conformer",
    }
    assert required.issubset(set(item.keys()))


def test_chemical_catalog_builds_from_local_data() -> None:
    catalog = ChemicalCatalog()
    catalog.load()

    assert catalog.chemicals_count >= 1
    assert catalog.compounds_with_3d_count >= 1


def test_chemicals_list_route_returns_valid_payload() -> None:
    payload = list_chemicals(q="", chemical_class="", has_3d=None, limit=8, offset=0)

    assert isinstance(payload, dict)
    assert {"items", "total", "limit", "offset"}.issubset(set(payload.keys()))
    assert isinstance(payload["items"], list)
    assert payload["limit"] == 8
    assert payload["offset"] == 0

    if payload["items"]:
        _required_list_fields(payload["items"][0])


def test_chemical_detail_route_from_list_item() -> None:
    items = list_chemicals(q="", chemical_class="", has_3d=None, limit=1, offset=0).get("items", [])
    if not items:
        pytest.skip("No chemicals available in local catalog")

    compound_id = items[0]["compound_id"]
    detail = get_chemical_detail(compound_id)

    _required_list_fields(detail)
    assert "related_beverages" in detail
    assert "source_compound_class" in detail
    assert "metabolism_relevance" in detail
    assert "toxicity_relevance" in detail
    assert "available_structure_formats" in detail
    assert "conformer_availability_summary" in detail


def test_conformer_route_returns_json_and_handles_missing() -> None:
    items = list_chemicals(q="", chemical_class="", has_3d=True, limit=1, offset=0).get("items", [])

    if items:
        compound_id = items[0]["compound_id"]
        payload = get_chemical_conformer(compound_id)
        assert payload["compound_id"] == compound_id
        assert isinstance(payload.get("has_3d_conformer"), bool)
        if payload["has_3d_conformer"]:
            assert payload.get("format") == "sdf"
            assert isinstance(payload.get("sdf"), str) and len(payload.get("sdf", "")) > 20
        else:
            assert "not available" in str(payload.get("message", "")).lower()
    else:
        # Still verify endpoint contract for non-3D case.
        fallback_items = list_chemicals(q="", chemical_class="", has_3d=False, limit=1, offset=0).get("items", [])
        if not fallback_items:
            pytest.skip("No compounds available for conformer fallback test")
        payload = get_chemical_conformer(fallback_items[0]["compound_id"])
        assert payload["has_3d_conformer"] is False
        assert payload["sdf"] is None


def test_missing_conformer_fallback_with_stub(monkeypatch: Any) -> None:
    import api.chemical_routes as chemical_routes

    class _FakeCatalog:
        def list_chemicals(self, **_: Any) -> Dict[str, Any]:
            return {
                "items": [
                    {
                        "compound_id": "cmp-fake",
                        "compound_name": "fake",
                        "normalized_compound_name": "fake",
                        "pubchem_cid": 1,
                        "chemical_class": "unknown",
                        "canonical_smiles": None,
                        "rdkit_valid": False,
                        "beverage_count": 0,
                        "beverage_examples": [],
                        "has_3d_conformer": False,
                    }
                ],
                "total": 1,
                "limit": 1,
                "offset": 0,
            }

        def get_compound(self, compound_id: str) -> Dict[str, Any] | None:
            if compound_id != "cmp-fake":
                return None
            return {
                "compound_id": "cmp-fake",
                "compound_name": "fake",
                "normalized_compound_name": "fake",
                "pubchem_cid": 1,
                "chemical_class": "unknown",
                "canonical_smiles": None,
                "rdkit_valid": False,
                "beverage_count": 0,
                "beverage_examples": [],
                "has_3d_conformer": False,
                "related_beverages": [],
                "source_compound_class": "unknown",
                "metabolism_relevance": "unknown",
                "toxicity_relevance": "unknown",
                "available_structure_formats": [],
                "conformer_availability_summary": {"has_3d_conformer": False, "has_2d_structure": False, "available_formats": []},
            }

        def get_conformer_payload(self, compound_id: str) -> Dict[str, Any]:
            _ = compound_id
            return {
                "compound_id": "cmp-fake",
                "has_3d_conformer": False,
                "format": None,
                "sdf": None,
                "message": "3D conformer not available for this compound.",
            }

    monkeypatch.setattr(chemical_routes, "get_chemical_catalog", lambda: _FakeCatalog())

    payload = get_chemical_conformer("cmp-fake")
    assert payload["has_3d_conformer"] is False
    assert payload["sdf"] is None
    assert "not available" in payload["message"].lower()


def test_search_and_filter_behavior() -> None:
    items: List[Dict[str, Any]] = list_chemicals(q="", chemical_class="", has_3d=None, limit=5, offset=0).get("items", [])
    if not items:
        pytest.skip("No chemicals available in local catalog")

    first = items[0]
    query_token = first["compound_name"][:4]

    search_items = list_chemicals(q=query_token, chemical_class="", has_3d=None, limit=20, offset=0).get("items", [])
    for item in search_items:
        haystack = f"{item.get('compound_name','')} {item.get('normalized_compound_name','')} {item.get('pubchem_cid') or ''}".lower()
        assert query_token.lower() in haystack

    class_name = first.get("chemical_class") or ""
    if class_name:
        class_items = list_chemicals(q="", chemical_class=class_name, has_3d=None, limit=20, offset=0).get("items", [])
        for item in class_items:
            assert str(item.get("chemical_class", "")).lower() == str(class_name).lower()


def test_json_safe_payloads() -> None:
    list_payload = list_chemicals(q="", chemical_class="", has_3d=None, limit=3, offset=0)
    json.dumps(list_payload, sort_keys=True)

    items = list_payload.get("items", [])
    if not items:
        return

    compound_id = items[0]["compound_id"]
    detail_payload = get_chemical_detail(compound_id)
    conformer_payload = get_chemical_conformer(compound_id)

    json.dumps(detail_payload, sort_keys=True)
    json.dumps(conformer_payload, sort_keys=True)
