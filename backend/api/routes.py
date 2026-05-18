"""API route handlers for Alcohol Intelligence pipeline."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Mapping, Optional

from fastapi import APIRouter, HTTPException

from api.logging_utils import log_request, structured_error
from api.schemas import AskRequest, IntakeRequest, QueryRequest
from reasoning.grounding_safety_guard import GroundingSafetyGuard
from reasoning.hybrid_orchestrator import orchestrate_query
from reasoning.query_router import route_query
from reasoning.response_synthesizer import OLLAMA_MODEL, ResponseSynthesizer
from reasoning.user_risk_advisor import build_user_risk_advice

router = APIRouter()

BANNED_INTERNAL_TERMS = (
    "PBPK",
    "Neo4j",
    "Weaviate",
    "causal path",
    "graph",
    "embedding",
    "vector",
    "simulator fallback",
    "confidence score",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return ""
    return text


def _strip_internal_terms(text: str) -> str:
    output = _clean_text(text)
    if not output:
        return ""
    for term in BANNED_INTERNAL_TERMS:
        output = re.sub(rf"\b{re.escape(term)}\b", "", output, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", output).strip()


def _is_service_unavailable_error(message: str) -> bool:
    m = _clean_text(message).lower()
    patterns = (
        "service unavailable",
        "connection refused",
        "failed to connect",
        "socket: operation not permitted",
        "timed out",
        "timeout",
        "not installed",
        "executable not found",
    )
    return any(pattern in m for pattern in patterns)


def _build_intake_query(payload: IntakeRequest) -> str:
    profile_bits = []
    if payload.sex != "unknown":
        profile_bits.append(payload.sex)
    profile_bits.append(f"{payload.weight_kg:g} kg")
    if payload.age is not None:
        profile_bits.append(f"{payload.age} years old")
    if payload.fed_state != "unknown":
        profile_bits.append(payload.fed_state)

    drinking_bits = [f"{payload.amount_ml:g} ml", payload.drink_type]
    if payload.duration_h is not None:
        drinking_bits.append(f"in {payload.duration_h:g} hour")

    if payload.goal == "drive_check":
        goal = "Can I drive now?"
    elif payload.goal == "time_to_sober":
        goal = "How long until I sober up?"
    elif payload.goal == "hangover_risk":
        goal = "What is my hangover risk?"
    else:
        goal = "How much more can I drink?"

    profile = " ".join(profile_bits).strip()
    drinking = " ".join(drinking_bits).strip()
    return f"I am {profile}, I drank {drinking}. {goal}".strip()


def _build_user_response(
    *,
    query: str,
    response_style: Optional[str],
    debug: bool,
    endpoint: str,
) -> Dict[str, Any]:
    stage = "route"
    start = time.perf_counter()

    route_result = route_query(query, enable_llm_fallback=False)
    route_payload = route_result.to_dict()

    stage = "orchestrate"
    orchestration = orchestrate_query(query, enable_llm_fallback=False)

    # Optional style override for API caller.
    if response_style and isinstance(orchestration.get("route"), Mapping):
        orchestration["route"]["response_style"] = response_style
    route_payload["response_style"] = response_style or route_payload.get("response_style")

    stage = "synthesis"
    synthesizer = ResponseSynthesizer(model=OLLAMA_MODEL, timeout_seconds=30)
    synthesis = synthesizer.synthesize_response(orchestration)

    stage = "guard"
    guard = GroundingSafetyGuard()
    guard_payload = guard.validate(synthesis)

    stage = "advice"
    advice = build_user_risk_advice(
        query=query,
        guarded_payload=guard_payload,
        synthesized_payload=synthesis,
        orchestrator_payload=orchestration,
    )

    guard_approved = bool(guard_payload.get("approved_for_display"))
    advice_safe = bool(advice.get("safe_for_display"))
    synthesis_blocked = not guard_approved

    blocked_request_type = _clean_text(advice.get("blocked_request_type")) or None
    blocked_reasons = [
        _clean_text(item)
        for item in list(guard_payload.get("blocked_reasons", []) or [])
        if _clean_text(item)
    ]
    blocked_synthesis_reasons = blocked_reasons if synthesis_blocked else []
    if synthesis_blocked and not blocked_synthesis_reasons:
        blocked_synthesis_reasons = ["Synthesis blocked by grounding/safety guard."]
    advisor_fallback_used = bool(synthesis_blocked and advice_safe)
    final_safe_for_display = bool(advice_safe)

    if not final_safe_for_display and blocked_request_type is None:
        blocked_request_type = blocked_synthesis_reasons[0] if blocked_synthesis_reasons else "safety_blocked"

    answer = _clean_text(advice.get("plain_answer"))
    if not final_safe_for_display:
        answer = "The system blocked this response for safety reasons."
        if blocked_request_type:
            answer = f"{answer} Request type: {blocked_request_type}."

    answer = _strip_internal_terms(answer)

    response: Dict[str, Any] = {
        "query": query,
        "answer": answer,
        "risk_level": _clean_text(advice.get("risk_level")) or "unknown",
        "risk_summary": _clean_text(advice.get("risk_summary")),
        "estimated_peak_bac": advice.get("estimated_peak_bac"),
        "estimated_time_to_sober_h": advice.get("estimated_time_to_sober_h"),
        "estimated_time_to_peak_h": advice.get("estimated_time_to_peak_h"),
        "ethanol_dose_g": advice.get("ethanol_dose_g"),
        "drink_abv_percent": advice.get("drink_abv_percent"),
        "drink_volume_ml": advice.get("drink_volume_ml"),
        "legal_limit_reference_bac": advice.get("legal_limit_reference_bac"),
        "is_estimated_below_0_08": advice.get("is_estimated_below_0_08"),
        "estimated_total_volume_for_0_08_ml": advice.get("estimated_total_volume_for_0_08_ml"),
        "estimated_additional_volume_to_0_08_ml": advice.get("estimated_additional_volume_to_0_08_ml"),
        "threshold_explanation": _clean_text(advice.get("threshold_explanation")) or None,
        "beverage_type": _clean_text(advice.get("beverage_type")) or None,
        "likely_compounds": [
            _clean_text(item)
            for item in list(advice.get("likely_compounds", []) or [])
            if _clean_text(item)
        ],
        "body_processes": [
            {
                "stage": _clean_text(item.get("stage")),
                "plain_explanation": _clean_text(item.get("plain_explanation")),
                "technical_explanation": _clean_text(item.get("technical_explanation")) or None,
            }
            for item in list(advice.get("body_processes", []) or [])
            if isinstance(item, Mapping)
            and _clean_text(item.get("stage"))
            and _clean_text(item.get("plain_explanation"))
        ],
        "detail_level": _clean_text(advice.get("detail_level")) or "layman",
        "driving_guidance": _clean_text(advice.get("driving_guidance")),
        "continue_drinking_guidance": _clean_text(advice.get("continue_drinking_guidance")),
        "hydration_guidance": _clean_text(advice.get("hydration_guidance")),
        "food_guidance": _clean_text(advice.get("food_guidance")),
        "medical_warning": _clean_text(advice.get("medical_warning")),
        "assumptions": [
            _clean_text(item)
            for item in list(advice.get("assumptions", []) or [])
            if _clean_text(item)
        ],
        "missing_info": [
            _clean_text(item)
            for item in list(advice.get("missing_info", []) or [])
            if _clean_text(item)
        ],
        "safe_for_display": final_safe_for_display,
        "advisor_fallback_used": advisor_fallback_used,
        "synthesis_blocked": synthesis_blocked,
        "blocked_synthesis_reasons": blocked_synthesis_reasons,
        "blocked_request_type": blocked_request_type,
    }

    if debug:
        response["debug"] = {
            "route": route_payload,
            "orchestration": orchestration,
            "synthesis": synthesis,
            "guard": guard_payload,
        }

    latency_ms = (time.perf_counter() - start) * 1000.0
    log_request(
        endpoint=endpoint,
        query=query,
        response_style=response_style,
        risk_level=response.get("risk_level"),
        safe_for_display=final_safe_for_display,
        latency_ms=latency_ms,
        error=None,
        stage="complete",
    )

    return response


@router.post("/route")
def route_endpoint(payload: QueryRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    query = payload.query
    try:
        result = route_query(query, enable_llm_fallback=False).to_dict()
        log_request(
            endpoint="/route",
            query=query,
            response_style=result.get("response_style"),
            risk_level=None,
            safe_for_display=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=None,
            stage="complete",
        )
        return result
    except Exception as exc:
        message = _clean_text(exc)
        log_request(
            endpoint="/route",
            query=query,
            response_style=None,
            risk_level=None,
            safe_for_display=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=message,
            stage="route",
        )
        code = 503 if _is_service_unavailable_error(message) else 500
        raise HTTPException(status_code=code, detail=structured_error(message or "routing failed", "route")) from exc


@router.post("/orchestrate")
def orchestrate_endpoint(payload: QueryRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    query = payload.query
    try:
        result = orchestrate_query(query, enable_llm_fallback=False)
        safe_for_response = bool(result.get("safe_for_response_synthesis"))
        log_request(
            endpoint="/orchestrate",
            query=query,
            response_style=result.get("route", {}).get("response_style") if isinstance(result.get("route"), Mapping) else None,
            risk_level=None,
            safe_for_display=safe_for_response,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=None,
            stage="complete",
        )
        return result
    except Exception as exc:
        message = _clean_text(exc)
        log_request(
            endpoint="/orchestrate",
            query=query,
            response_style=None,
            risk_level=None,
            safe_for_display=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=message,
            stage="orchestrate",
        )
        code = 503 if _is_service_unavailable_error(message) else 500
        raise HTTPException(status_code=code, detail=structured_error(message or "orchestration failed", "orchestrate")) from exc


@router.post("/ask")
def ask_endpoint(payload: AskRequest) -> Dict[str, Any]:
    query = payload.query
    start = time.perf_counter()
    try:
        return _build_user_response(
            query=query,
            response_style=payload.response_style,
            debug=bool(payload.debug),
            endpoint="/ask",
        )
    except HTTPException:
        raise
    except Exception as exc:
        message = _clean_text(exc)
        log_request(
            endpoint="/ask",
            query=query,
            response_style=payload.response_style,
            risk_level=None,
            safe_for_display=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=message,
            stage="pipeline",
        )
        code = 503 if _is_service_unavailable_error(message) else 500
        raise HTTPException(status_code=code, detail=structured_error(message or "ask pipeline failed", "ask")) from exc


@router.post("/intake")
def intake_endpoint(payload: IntakeRequest) -> Dict[str, Any]:
    query = _build_intake_query(payload)
    start = time.perf_counter()
    try:
        return _build_user_response(query=query, response_style="layman", debug=False, endpoint="/intake")
    except HTTPException:
        raise
    except Exception as exc:
        message = _clean_text(exc)
        log_request(
            endpoint="/intake",
            query=query,
            response_style="layman",
            risk_level=None,
            safe_for_display=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=message,
            stage="pipeline",
        )
        code = 503 if _is_service_unavailable_error(message) else 500
        raise HTTPException(status_code=code, detail=structured_error(message or "intake pipeline failed", "intake")) from exc
