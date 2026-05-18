"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("query must be non-empty")
        if len(text) > 2000:
            raise ValueError("query too long")
        return text


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    response_style: Optional[Literal["layman", "technical", "scientific"]] = None
    debug: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("query must be non-empty")
        if len(text) > 2000:
            raise ValueError("query too long")
        return text


class IntakeRequest(BaseModel):
    sex: Literal["male", "female", "unknown"]
    weight_kg: float = Field(..., gt=20.0, lt=400.0)
    age: Optional[int] = Field(default=None, ge=18, le=120)
    fed_state: Literal["fed", "fasted", "unknown"]
    drink_type: str = Field(..., min_length=1, max_length=64)
    amount_ml: float = Field(..., gt=1.0, lt=5000.0)
    duration_h: Optional[float] = Field(default=None, ge=0.0, le=72.0)
    goal: Literal["drive_check", "time_to_sober", "hangover_risk", "should_i_keep_drinking"]

    @field_validator("drink_type")
    @classmethod
    def normalize_drink_type(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("drink_type must be non-empty")
        return text


class ErrorResponse(BaseModel):
    error: bool = True
    message: str
    stage: str


class HealthComponent(BaseModel):
    ok: bool
    detail: str
    missing_required_count: Optional[int] = None
    missing_required: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    components: Dict[str, HealthComponent]


class AskResponse(BaseModel):
    query: str
    answer: str
    risk_level: str
    risk_summary: str
    estimated_peak_bac: Optional[float]
    estimated_time_to_sober_h: Optional[float]
    estimated_time_to_peak_h: Optional[float]
    driving_guidance: str
    continue_drinking_guidance: str
    hydration_guidance: str
    food_guidance: str
    medical_warning: str
    assumptions: List[str]
    missing_info: List[str]
    safe_for_display: bool
    advisor_fallback_used: bool = False
    synthesis_blocked: bool = False
    blocked_synthesis_reasons: List[str] = Field(default_factory=list)
    blocked_request_type: Optional[str]
    debug: Optional[Dict[str, Any]] = None
