"""Chemical Explorer API routes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from api.logging_utils import structured_error
from services.chemical_catalog import get_chemical_catalog

router = APIRouter()


@router.get("/chemicals")
def list_chemicals(
    q: str = "",
    chemical_class: str = "",
    has_3d: Optional[bool] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    catalog = get_chemical_catalog()
    return catalog.list_chemicals(
        q=q,
        chemical_class=chemical_class,
        has_3d=has_3d,
        limit=limit,
        offset=offset,
    )


@router.get("/chemicals/{compound_id}")
def get_chemical_detail(compound_id: str) -> Dict[str, Any]:
    catalog = get_chemical_catalog()
    payload = catalog.get_compound(compound_id)
    if not payload:
        raise HTTPException(status_code=404, detail=structured_error("chemical not found", "chemicals_detail"))
    return payload


@router.get("/chemicals/{compound_id}/conformer")
def get_chemical_conformer(compound_id: str) -> Dict[str, Any]:
    catalog = get_chemical_catalog()
    detail = catalog.get_compound(compound_id)
    if not detail:
        raise HTTPException(status_code=404, detail=structured_error("chemical not found", "chemicals_conformer"))

    conformer = catalog.get_conformer_payload(compound_id)
    return {
        "compound_id": detail.get("compound_id"),
        "compound_name": detail.get("compound_name"),
        "pubchem_cid": detail.get("pubchem_cid"),
        "has_3d_conformer": bool(conformer.get("has_3d_conformer")),
        "format": conformer.get("format"),
        "sdf": conformer.get("sdf"),
        "message": conformer.get("message") or "ok",
    }
