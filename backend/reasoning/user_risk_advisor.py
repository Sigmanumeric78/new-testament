"""Phase 08F deterministic end-user alcohol risk advisor.

Consumes guarded/synthesized/orchestrator payloads and generates conservative,
plain-language guidance for end users.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BANNED_TECHNICAL_TERMS: Tuple[str, ...] = (
    "pbpk",
    "neo4j",
    "weaviate",
    "causal path",
    "adh",
    "aldh",
    "cyp2e1",
    "confidence score",
    "source dataset",
    "collection",
    "vector",
    "embedding",
    "graph",
    "simulator fallback",
    "body_fat_percent fallback",
    "retrieved evidence",
    "internal fallback",
)

BEVERAGE_TOKENS: Tuple[str, ...] = (
    "whisky",
    "whiskey",
    "vodka",
    "beer",
    "wine",
    "rum",
    "gin",
    "tequila",
    "brandy",
)

WORD_NUMBERS: Mapping[str, float] = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
}

UNSAFE_REQUEST_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "unsafe_continue_drinking": (
        r"\bhow\s+much\s+more\s+can\s+i\s+drink\b",
        r"\bshould\s+i\s+keep\s+drinking\b",
        r"\bkeep\s+drinking\b",
        r"\bsafe\s+amount\s+to\s+drink\s+more\b",
        r"\bhow\s+much\s+before\s+i\s+am\s+too\s+drunk\b",
        r"\bhow\s+much\s+before\s+i\s+am\s+too\s+hungover\b",
        r"\bhow\s+much\s+before\s+hangover\b",
        r"\bhow\s+much\s+before\s+driving\b",
        r"\bbefore\s+i\s+am\s+too\s+hungover\b",
        r"\bbefore\s+toxic\b",
    ),
    "unsafe_driving_check": (
        r"\bcan\s+i\s+drive\b",
        r"\bam\s+i\s+safe\s+to\s+drive\b",
        r"\bshould\s+i\s+drive\b",
        r"\bsafe\s+to\s+drive\b",
        r"\blegal\s+limit\b",
        r"\bdriving\b",
        r"\bcar\b",
        r"\bride\s+home\b",
    ),
    "unsafe_toxic_threshold": (
        r"\bhow\s+much\s+alcohol\s+will\s+be\s+toxic\b",
        r"\btoxic\s+amount\b",
    ),
}

DRIVING_BLOCKED_REFUSAL = (
    "I can’t tell you that you are safe to drive. "
    "Based on this estimate, you should not drive. "
    "This app cannot determine legal or actual driving safety."
)

CONTINUE_DRINKING_BLOCKED_REFUSAL = (
    "I can’t calculate a safe amount to keep drinking. "
    "You should not drink more right now."
)

EMERGENCY_PATTERNS: Tuple[str, ...] = (
    r"\bvomiting\s+repeatedly\b",
    r"\brepeated\s+vomiting\b",
    r"\bunconscious\b",
    r"\bcannot\s+wake\b",
    r"\bslow\s+breathing\b",
    r"\bconfusion\b",
    r"\bseizure\b",
    r"\bblue\s+lips\b",
    r"\balcohol\s+poisoning\b",
)

RISK_LEVELS: Tuple[str, ...] = (
    "unknown",
    "low",
    "moderate",
    "high",
    "very_high",
    "possible_medical_emergency",
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


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value).lower()).strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_beverage(token: str) -> str:
    token_n = _normalize_text(token)
    if token_n == "whiskey":
        return "whisky"
    return token_n


def _extract_weight_kg(query: str) -> Optional[float]:
    q = _normalize_text(query)
    match = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)\b", q)
    if match:
        return float(match.group(1))
    match = re.search(r"\bweigh\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)\b", q)
    if match:
        return float(match.group(1))
    return None


def _extract_sex(query: str) -> Optional[str]:
    q = _normalize_text(query)
    if re.search(r"\b(male|man)\b", q):
        return "male"
    if re.search(r"\b(female|woman)\b", q):
        return "female"
    return None


def _extract_age(query: str) -> Optional[int]:
    q = _normalize_text(query)
    match = re.search(r"\b(\d{1,3})\s*(years?\s*old|yo|y/o)\b", q)
    if match:
        value = int(match.group(1))
        if value > 0:
            return value
    match = re.search(r"\bage\s*(\d{1,3})\b", q)
    if match:
        value = int(match.group(1))
        if value > 0:
            return value
    return None


def _extract_fed_state(query: str) -> Optional[str]:
    q = _normalize_text(query)
    if re.search(r"\b(fasted|empty\s+stomach|without\s+food)\b", q):
        return "fasted"
    if re.search(r"\b(fed|ate|eaten|with\s+food|after\s+meal)\b", q):
        return "fed"
    return None


def _extract_beverage(query: str) -> Optional[str]:
    q = _normalize_text(query)
    for token in BEVERAGE_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", q):
            return _normalize_beverage(token)
    return None


def _extract_amount_ml(query: str) -> Optional[float]:
    q = _normalize_text(query)
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|oz|ounces?|shots?|glasses?|beers?)\b",
        q,
    )
    if match:
        amount = float(match.group(1))
        unit = match.group(2)
        if unit in {"ml", "milliliter", "milliliters"}:
            return round(amount, 6)
        if unit in {"l", "liter", "liters"}:
            return round(amount * 1000.0, 6)
        if unit in {"oz", "ounce", "ounces"}:
            return round(amount * 29.5735, 6)
        if unit in {"shot", "shots"}:
            return round(amount * 44.0, 6)
        if unit in {"glass", "glasses"}:
            return round(amount * 150.0, 6)
        if unit in {"beer", "beers"}:
            return round(amount * 355.0, 6)

    word_amount = re.search(r"\b(one|two|three|four|five)\s+(glass|glasses|beer|beers|shot|shots)\b", q)
    if word_amount:
        number = WORD_NUMBERS.get(word_amount.group(1), 0.0)
        unit = word_amount.group(2)
        if unit in {"glass", "glasses"}:
            return round(number * 150.0, 6)
        if unit in {"beer", "beers"}:
            return round(number * 355.0, 6)
        if unit in {"shot", "shots"}:
            return round(number * 44.0, 6)

    return None


def _extract_duration_h(query: str) -> Optional[float]:
    q = _normalize_text(query)

    match = re.search(r"\b(?:in|over)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?)\b", q)
    if match:
        return round(float(match.group(1)), 6)

    match = re.search(r"\b(?:in|over)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?)\b", q)
    if match:
        return round(float(match.group(1)) / 60.0, 6)

    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\s*(ago|since)\b", q)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("min"):
            return round(value / 60.0, 6)
        return round(value, 6)

    if re.search(r"\bjust\s+drank\b", q):
        return 0.0

    return None


def _detect_unsafe_request_type(query: str) -> Optional[str]:
    q = _normalize_text(query)
    for request_type, patterns in UNSAFE_REQUEST_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return request_type
    return None


def _detect_emergency(query: str) -> bool:
    q = _normalize_text(query)
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def extract_query_signals(query: str) -> Dict[str, Any]:
    text = _clean_text(query)
    q = _normalize_text(text)
    unsafe_type = _detect_unsafe_request_type(text)
    return {
        "weight_kg": _extract_weight_kg(text),
        "sex": _extract_sex(text),
        "age": _extract_age(text),
        "fed_state": _extract_fed_state(text),
        "drink_type": _extract_beverage(text),
        "amount_ml": _extract_amount_ml(text),
        "duration_h": _extract_duration_h(text),
        "unsafe_continue_drinking_request": unsafe_type == "unsafe_continue_drinking",
        "unsafe_driving_request": unsafe_type == "unsafe_driving_check",
        "unsafe_toxic_threshold_request": unsafe_type == "unsafe_toxic_threshold",
        "unsafe_request_type": unsafe_type,
        "mentions_hangover": bool(re.search(r"\bhangover|hungover\b", q)),
        "mentions_toxic": bool(re.search(r"\btoxic|toxicity\b", q)),
        "mentions_sober": bool(re.search(r"\bsober\b", q)),
        "mentions_time_to_clear": bool(re.search(r"\btime\s+to\s+(?:clear|sober)\b", q)),
        "mentions_drive": bool(re.search(r"\bdrive|driving|legal\s+limit|car|ride\s+home\b", q)),
        "emergency_symptoms_detected": _detect_emergency(text),
    }


def _extract_first_simulation(
    synthesized_payload: Optional[Mapping[str, Any]],
    orchestrator_payload: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if synthesized_payload:
        summary = synthesized_payload.get("simulation_summary")
        if isinstance(summary, Mapping):
            sims = summary.get("simulations")
            if isinstance(sims, list) and sims:
                first = sims[0]
                if isinstance(first, Mapping):
                    return first

    if orchestrator_payload:
        evidence_bundle = orchestrator_payload.get("evidence_bundle")
        if isinstance(evidence_bundle, Mapping):
            summary = evidence_bundle.get("simulation_summary")
            if isinstance(summary, Mapping):
                sims = summary.get("simulations")
                if isinstance(sims, list) and sims:
                    first = sims[0]
                    if isinstance(first, Mapping):
                        return first

    return None


def _extract_toxicity_summary(
    synthesized_payload: Optional[Mapping[str, Any]],
    orchestrator_payload: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if synthesized_payload and isinstance(synthesized_payload.get("toxicity_summary"), Mapping):
        return synthesized_payload["toxicity_summary"]
    if orchestrator_payload:
        evidence_bundle = orchestrator_payload.get("evidence_bundle")
        if isinstance(evidence_bundle, Mapping) and isinstance(evidence_bundle.get("toxicity_summary"), Mapping):
            return evidence_bundle.get("toxicity_summary")
    return None


def _risk_from_bac(peak_bac: Optional[float], *, emergency: bool) -> str:
    if emergency:
        return "possible_medical_emergency"
    if peak_bac is None:
        return "unknown"
    if peak_bac >= 0.30:
        return "possible_medical_emergency"
    if peak_bac >= 0.20:
        return "very_high"
    if peak_bac >= 0.08:
        return "high"
    if peak_bac >= 0.05:
        return "moderate"
    if peak_bac > 0.0:
        return "low"
    return "unknown"


def _risk_summary(risk_level: str, peak_bac: Optional[float]) -> str:
    if risk_level == "possible_medical_emergency":
        return "Your symptoms may indicate a medical emergency."
    if risk_level == "very_high":
        return "Your estimated alcohol level suggests very high impairment risk."
    if risk_level == "high":
        return "Your estimated alcohol level suggests high impairment risk."
    if risk_level == "moderate":
        return "Your estimated alcohol level suggests moderate impairment risk."
    if risk_level == "low":
        return "Your estimated alcohol level suggests some impairment risk."
    if peak_bac is None:
        return "I do not have enough data for a precise BAC estimate, so this is a conservative risk assessment."
    return "I do not have enough data for a precise risk estimate."


def _sanitize_plain_text(text: str) -> str:
    output = _clean_text(text)
    if not output:
        return ""

    for term in BANNED_TECHNICAL_TERMS:
        output = re.sub(rf"\b{re.escape(term)}\b", "", output, flags=re.IGNORECASE)

    output = re.sub(r"\s+", " ", output).strip()
    return output


def _display_bac(peak_bac: Optional[float]) -> Optional[str]:
    if peak_bac is None:
        return None
    return f"{peak_bac:.2f}"


def _display_hours(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    rounded = int(round(hours))
    if rounded <= 0:
        rounded = 1
    return str(rounded)


def _is_scientific_query(query: str) -> bool:
    q = _normalize_text(query)
    return bool(re.search(r"\b(research|study|studies|paper|papers|evidence|scientific)\b", q))


def _extract_evidence_signal(
    synthesized_payload: Optional[Mapping[str, Any]],
    orchestrator_payload: Optional[Mapping[str, Any]],
) -> str:
    if synthesized_payload:
        used_evidence = synthesized_payload.get("used_evidence")
        if isinstance(used_evidence, list):
            titles: List[str] = []
            for item in used_evidence:
                if not isinstance(item, Mapping):
                    continue
                title = _clean_text(item.get("title"))
                if title:
                    titles.append(title)
                if len(titles) >= 2:
                    break
            if titles:
                return ", ".join(titles)

    if orchestrator_payload:
        bundle = orchestrator_payload.get("evidence_bundle")
        if isinstance(bundle, Mapping):
            retrieved = bundle.get("retrieved_evidence")
            if isinstance(retrieved, list):
                titles = []
                for item in retrieved:
                    if not isinstance(item, Mapping):
                        continue
                    title = _clean_text(item.get("title"))
                    if title:
                        titles.append(title)
                    if len(titles) >= 2:
                        break
                if titles:
                    return ", ".join(titles)
    return ""


def _extract_guarded_or_synth_answer(
    guarded_payload: Mapping[str, Any],
    synthesized_payload: Optional[Mapping[str, Any]],
) -> str:
    if not bool(guarded_payload.get("approved_for_display")):
        return ""
    guarded_text = _clean_text(guarded_payload.get("final_answer"))
    if guarded_text and guarded_text.lower() != "response blocked by grounding/safety guard.":
        return guarded_text
    if synthesized_payload:
        return _clean_text(synthesized_payload.get("answer"))
    return ""


def _query_declares_abv(query: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*%\s*(?:abv)?\b", _normalize_text(query)))


def _extract_specific_assumptions(
    *,
    query: str,
    signals: Mapping[str, Any],
    first_sim: Optional[Mapping[str, Any]],
    synthesized_payload: Optional[Mapping[str, Any]],
) -> List[str]:
    assumptions: List[str] = []
    if first_sim:
        assumptions.append("Risk estimate is based on model assumptions and your provided details.")

    summary = synthesized_payload.get("simulation_summary") if synthesized_payload else None
    if not isinstance(summary, Mapping):
        return assumptions

    defaults_raw = list(summary.get("defaults_applied", []) or [])
    defaults = {_normalize_text(item) for item in defaults_raw if _clean_text(item)}
    safe_defaults = summary.get("safe_defaults") if isinstance(summary.get("safe_defaults"), Mapping) else {}

    if "age" in defaults:
        assumptions.append("Assumed adult age because age was not provided.")

    if "weight" in defaults:
        default_weight = _safe_float(safe_defaults.get("weight")) if isinstance(safe_defaults, Mapping) else None
        assumptions.append(
            f"Assumed {default_weight if default_weight is not None else 75.0:g} kg body weight because weight was not provided."
        )

    if "sex" in defaults:
        default_sex = _clean_text(safe_defaults.get("sex")) if isinstance(safe_defaults, Mapping) else ""
        assumptions.append(f"Assumed {default_sex or 'male'} sex because sex was not provided.")

    if "fed_state" in defaults:
        default_fed_state = _clean_text(safe_defaults.get("fed_state")) if isinstance(safe_defaults, Mapping) else ""
        assumptions.append(f"Assumed {default_fed_state or 'fed'} state because meal status was not provided.")

    if "liver_status" in defaults:
        default_liver = _normalize_text(safe_defaults.get("liver_status")) if isinstance(safe_defaults, Mapping) else ""
        if default_liver in {"", "healthy"}:
            assumptions.append("Assumed healthy adult metabolism.")
        else:
            assumptions.append(f"Assumed {default_liver} metabolism.")

    limitations = [_normalize_text(item) for item in list(synthesized_payload.get("limitations", []) or []) if _clean_text(item)]
    abv_default_used = any("default abv" in item for item in limitations)
    if (
        first_sim
        and abv_default_used
        and not _query_declares_abv(query)
        and signals.get("drink_type") is not None
    ):
        beverage = _clean_text(first_sim.get("beverage")) or _clean_text(signals.get("drink_type"))
        abv = _safe_float(first_sim.get("abv_percent"))
        if beverage and abv is not None:
            assumptions.append(f"Assumed standard {beverage} ABV of {abv:g}%.")

    return assumptions


def build_user_risk_advice(
    *,
    query: str,
    guarded_payload: Mapping[str, Any],
    synthesized_payload: Optional[Mapping[str, Any]] = None,
    orchestrator_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    signals = extract_query_signals(query)
    query_norm = _normalize_text(query)

    first_sim = _extract_first_simulation(synthesized_payload, orchestrator_payload)
    toxicity_summary = _extract_toxicity_summary(synthesized_payload, orchestrator_payload)

    estimated_peak_bac = _safe_float(first_sim.get("peak_bac_percent")) if first_sim else None
    estimated_time_to_sober_h = _safe_float(first_sim.get("time_to_sober_h")) if first_sim else None
    estimated_time_to_peak_h = _safe_float(first_sim.get("time_to_peak_h")) if first_sim else None

    emergency = bool(signals.get("emergency_symptoms_detected"))
    scientific_query = _is_scientific_query(query)
    comparison_harder_query = bool(
        re.search(r"\bwhisk(?:y|ey)\b", query_norm)
        and re.search(r"\bbeer\b", query_norm)
        and re.search(r"\bhit\s+harder\b|\bharder\s+than\b|\bwhich\s+hits\s+harder\b", query_norm)
    )
    wine_headache_query = bool(re.search(r"\bwine\b", query_norm) and re.search(r"\bheadache", query_norm))
    sulfite_research_query = bool(scientific_query and re.search(r"\bsulfite", query_norm))
    evidence_signal = _extract_evidence_signal(synthesized_payload, orchestrator_payload)
    source_answer = _extract_guarded_or_synth_answer(guarded_payload, synthesized_payload)
    for note in (
        "This is an estimate, not medical advice.",
        "Do not use this to decide whether it is safe to drive.",
        "Seek medical help for severe symptoms.",
    ):
        source_answer = re.sub(re.escape(note), " ", source_answer, flags=re.IGNORECASE)
    source_answer = _sanitize_plain_text(re.sub(r"\s+", " ", source_answer).strip())

    risk_level = _risk_from_bac(estimated_peak_bac, emergency=emergency)
    blocked_request_type = signals.get("unsafe_request_type")
    driving_query = bool(signals.get("mentions_drive")) or blocked_request_type == "unsafe_driving_check"

    driving_guidance = (
        "Do not drive based on this estimate. "
        "This app cannot determine legal or actual driving safety. "
        "Arrange a ride or wait."
    )
    if estimated_peak_bac is not None and estimated_peak_bac >= 0.08:
        driving_guidance = (
            "Do not drive based on this estimate. "
            "Your estimated alcohol level is in a high-risk range. "
            "This app cannot determine legal or actual driving safety. "
            "Arrange a ride or wait."
        )

    if blocked_request_type in {"unsafe_continue_drinking", "unsafe_toxic_threshold"}:
        continue_drinking_guidance = CONTINUE_DRINKING_BLOCKED_REFUSAL
    elif blocked_request_type == "unsafe_driving_check":
        continue_drinking_guidance = "You should not drink more right now."
    else:
        continue_drinking_guidance = "You should not drink more right now."

    if estimated_time_to_sober_h is not None:
        display_h = _display_hours(estimated_time_to_sober_h) or "unknown"
        time_guidance = (
            f"It may take about {display_h} hours for your body to clear most alcohol."
        )
    else:
        time_guidance = "I cannot give a precise sober-time estimate without more details, so use a conservative wait-and-rest plan."

    hydration_guidance = "Sip water to reduce dehydration. Water does not make alcohol leave your body faster."

    fed_state = signals.get("fed_state")
    if fed_state == "fasted":
        food_guidance = "Food may slow further absorption if alcohol is still being absorbed, but it will not instantly sober you up."
    else:
        food_guidance = "Food may help comfort and may slow further absorption, but it will not instantly sober you up."

    if emergency or risk_level == "possible_medical_emergency":
        medical_warning = "Seek emergency medical help immediately."
    else:
        medical_warning = "Seek medical help for severe or worsening symptoms."

    risk_summary = _risk_summary(risk_level, estimated_peak_bac)

    assumptions = _extract_specific_assumptions(
        query=query,
        signals=signals,
        first_sim=first_sim,
        synthesized_payload=synthesized_payload,
    )

    missing_info: List[str] = []
    for field, value in (
        ("weight", signals.get("weight_kg")),
        ("sex", signals.get("sex")),
        ("age", signals.get("age")),
        ("fed_state", signals.get("fed_state")),
        ("drink_type", signals.get("drink_type")),
        ("amount", signals.get("amount_ml")),
        ("duration", signals.get("duration_h")),
    ):
        if value is None:
            missing_info.append(field)

    plain_parts: List[str] = []
    if emergency:
        plain_parts.append("This may be an alcohol-related medical emergency.")
        plain_parts.append("Seek emergency medical help immediately.")
        plain_parts.append("Do not leave the person alone while waiting for help.")
        plain_parts.append("Do not give more alcohol.")
    else:
        if blocked_request_type == "unsafe_driving_check":
            plain_parts.append(DRIVING_BLOCKED_REFUSAL)
        elif blocked_request_type in {"unsafe_continue_drinking", "unsafe_toxic_threshold"}:
            plain_parts.append(CONTINUE_DRINKING_BLOCKED_REFUSAL)
        elif comparison_harder_query:
            plain_parts.append(
                "Whisky can hit harder than beer because it usually has higher alcohol concentration and faster early absorption."
            )
        elif wine_headache_query:
            plain_parts.append(
                "Wine headaches can be linked to compounds such as sulfites, histamine, tyramine, congeners, and polyphenols in sensitive people."
            )
            plain_parts.append("For your situation, you should not drink more right now.")
        elif sulfite_research_query:
            plain_parts.append(
                "Some research links sulfites with headache symptoms in sensitive people, but evidence quality can vary."
            )
            plain_parts.append("Some supporting evidence was unavailable.")
        elif source_answer:
            plain_parts.append(source_answer)
        else:
            plain_parts.append("For your situation, you should not drink more right now.")

        if scientific_query and evidence_signal:
            signal_norm = _normalize_text(evidence_signal)
            if (
                "sulfite" in signal_norm
                or "headache" in signal_norm
                or "histamine" in signal_norm
                or "tyramine" in signal_norm
            ):
                plain_parts.append(f"Relevant evidence includes: {evidence_signal}.")

        bac_display = _display_bac(estimated_peak_bac)
        if bac_display is not None:
            plain_parts.append(f"Your estimated peak BAC is about {bac_display}%.")

        if estimated_time_to_peak_h is not None:
            peak_h = _display_hours(estimated_time_to_peak_h) or "unknown"
            plain_parts.append(f"Your peak effect may be around {peak_h} hours after drinking.")

        if estimated_time_to_sober_h is not None:
            clear_h = _display_hours(estimated_time_to_sober_h) or "unknown"
            plain_parts.append(f"It may take about {clear_h} hours to clear alcohol.")

        if driving_query or estimated_peak_bac is not None:
            plain_parts.append(driving_guidance)

        if toxicity_summary and isinstance(toxicity_summary, Mapping):
            compounds = list(toxicity_summary.get("risk_compounds", []) or [])
            if compounds:
                compound_text = ", ".join([_clean_text(item) for item in compounds[:4] if _clean_text(item)])
                if compound_text:
                    plain_parts.append(f"Possible symptom-linked compounds include: {compound_text}.")

        plain_parts.append(risk_summary)
        plain_parts.append(hydration_guidance)
        plain_parts.append(food_guidance)

    plain_parts.append(medical_warning)

    plain_answer = _sanitize_plain_text(" ".join([_clean_text(item) for item in plain_parts if _clean_text(item)]))

    safe_for_display = bool(_clean_text(plain_answer))

    return {
        "plain_answer": plain_answer,
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "driving_guidance": driving_guidance,
        "continue_drinking_guidance": continue_drinking_guidance,
        "time_guidance": time_guidance,
        "hydration_guidance": hydration_guidance,
        "food_guidance": food_guidance,
        "medical_warning": medical_warning,
        "estimated_peak_bac": round(float(estimated_peak_bac), 6) if estimated_peak_bac is not None else None,
        "estimated_time_to_sober_h": round(float(estimated_time_to_sober_h), 6) if estimated_time_to_sober_h is not None else None,
        "estimated_time_to_peak_h": round(float(estimated_time_to_peak_h), 6) if estimated_time_to_peak_h is not None else None,
        "assumptions": sorted(set([_clean_text(item) for item in assumptions if _clean_text(item)])),
        "missing_info": sorted(set([_clean_text(item) for item in missing_info if _clean_text(item)])),
        "blocked_request_type": _clean_text(blocked_request_type) or None,
        "safe_for_display": safe_for_display,
    }


def to_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True)
