"""Deterministic query intent router for the alcohol physiology reasoning engine.

This module routes user questions to orchestration modules only.
It does not execute PBPK, retrieval, or graph reasoning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("query_router")

ENCODING = "utf-8"
LOW_CONFIDENCE_THRESHOLD = 0.75
OLLAMA_MODEL = "qwen2.5:3b"

INTENT_CLASSES: Tuple[str, ...] = (
    "simulation",
    "mechanistic_explanation",
    "toxicity_risk",
    "comparison",
    "scientific_evidence",
    "personalized_physiology",
    "retrieval_only",
)

INPUT_FIELDS: Tuple[str, ...] = (
    "body_weight",
    "sex",
    "age",
    "fed_state",
    "beverages",
    "drink_amount",
    "time_since_drinking",
)

INTENT_MODULES: Mapping[str, List[str]] = {
    "simulation": ["pbpk"],
    "mechanistic_explanation": ["neo4j", "weaviate"],
    "toxicity_risk": ["neo4j", "weaviate", "toxicity"],
    "comparison": ["pbpk", "neo4j", "weaviate"],
    "scientific_evidence": ["weaviate"],
    "personalized_physiology": ["pbpk", "neo4j"],
    "retrieval_only": ["weaviate"],
}

INTENT_REQUIRED_INPUTS: Mapping[str, Dict[str, bool]] = {
    "simulation": {
        "body_weight": True,
        "sex": True,
        "age": True,
        "fed_state": True,
        "beverages": True,
        "drink_amount": True,
        "time_since_drinking": True,
    },
    "mechanistic_explanation": {
        "body_weight": False,
        "sex": False,
        "age": False,
        "fed_state": False,
        "beverages": False,
        "drink_amount": False,
        "time_since_drinking": False,
    },
    "toxicity_risk": {
        "body_weight": False,
        "sex": False,
        "age": False,
        "fed_state": False,
        "beverages": False,
        "drink_amount": False,
        "time_since_drinking": False,
    },
    "comparison": {
        "body_weight": True,
        "sex": True,
        "age": True,
        "fed_state": True,
        "beverages": True,
        "drink_amount": True,
        "time_since_drinking": True,
    },
    "scientific_evidence": {
        "body_weight": False,
        "sex": False,
        "age": False,
        "fed_state": False,
        "beverages": False,
        "drink_amount": False,
        "time_since_drinking": False,
    },
    "personalized_physiology": {
        "body_weight": True,
        "sex": True,
        "age": True,
        "fed_state": True,
        "beverages": True,
        "drink_amount": True,
        "time_since_drinking": True,
    },
    "retrieval_only": {
        "body_weight": False,
        "sex": False,
        "age": False,
        "fed_state": False,
        "beverages": False,
        "drink_amount": False,
        "time_since_drinking": False,
    },
}

RESPONSE_STYLE_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "scientific": (
        "scientific",
        "peer reviewed",
        "study",
        "studies",
        "citations",
        "mechanistic detail",
        "biochemical",
    ),
    "technical": (
        "technical",
        "technically",
        "detailed",
        "deep dive",
        "expert",
        "advanced",
    ),
    "layman": (
        "simple words",
        "simple",
        "plain english",
        "easy to understand",
        "for beginners",
        "non technical",
    ),
}

BEVERAGE_TOKENS: Tuple[str, ...] = (
    "beer",
    "whisky",
    "whiskey",
    "vodka",
    "wine",
    "rum",
    "gin",
    "tequila",
    "cider",
    "lager",
    "ale",
    "champagne",
    "brandy",
    "bourbon",
    "mead",
    "cocktail",
)

SCORING_RULES: Mapping[str, Tuple[Tuple[str, float, str], ...]] = {
    "simulation": (
        (r"\bwhat\s+will\s+my\s+bac\b", 4.2, "explicit BAC forecast request"),
        (r"\b(?:bac|blood\s+alcohol)\b", 3.4, "BAC keyword"),
        (r"\bhow\s+drunk\b", 3.4, "intoxication forecast request"),
        (r"\buntil\s+sober\b", 3.5, "sobriety timing request"),
        (r"\btime\s+to\s+sober\b", 3.5, "sobriety timing phrase"),
        (r"\bhow\s+long\b.*\bsober\b", 3.2, "how-long sobriety request"),
        (r"\bpeak\s+bac\b", 3.0, "peak BAC request"),
        (r"\bwill\s+i\s+get\s+drunk\b", 3.0, "personal intoxication question"),
        (r"\bcan\s+i\s+drive\b|\bsafe\s+to\s+drive\b", 3.4, "driving safety check phrasing"),
        (r"\blegal\s+limit\b|\bcar\b|\bride\s+home\b", 2.8, "driving context phrasing"),
        (r"\btime\s+to\s+clear\b|\bclear\s+the\s+alcohol\b", 3.2, "alcohol clearance timing request"),
        (r"\bhow\s+much\s+more\s+can\s+i\s+drink\b|\bkeep\s+drinking\b", 3.0, "continued drinking threshold request"),
    ),
    "mechanistic_explanation": (
        (r"\bwhy\b", 1.4, "causal why-question marker"),
        (r"\bhow\s+does\b", 1.6, "mechanistic how-does marker"),
        (r"\bhit\s+harder\b", 2.6, "relative effect mechanism phrase"),
        (r"\bmetaboli[sz]e\b", 2.1, "metabolism mechanism keyword"),
        (r"\bmake\s+me\s+sleepy\b", 2.1, "effect mechanism question"),
        (r"\bmechanism\b", 2.2, "explicit mechanism keyword"),
    ),
    "toxicity_risk": (
        (r"\bhangover\b", 3.0, "hangover symptom marker"),
        (r"\bheadache(?:s)?\b", 3.2, "headache symptom marker"),
        (r"\bstomach\b", 2.4, "stomach symptom marker"),
        (r"\bupset\b", 2.2, "upset symptom marker"),
        (r"\bnausea\b", 2.6, "nausea symptom marker"),
        (r"\bvomit(?:ing)?\b", 2.5, "vomit symptom marker"),
        (r"\bmigraine\b", 2.4, "migraine symptom marker"),
        (r"\btoxic(?:ity)?\b", 2.5, "toxicity keyword"),
        (r"\brisk\b", 1.4, "risk framing"),
    ),
    "comparison": (
        (r"\bvs\.?\b", 3.5, "vs comparison marker"),
        (r"\bversus\b", 3.5, "versus comparison marker"),
        (r"\bcompare\b", 3.3, "compare keyword"),
        (r"\bcompared\s+to\b", 3.2, "compared-to phrase"),
        (r"\bharder\s+than\b", 3.2, "relative potency phrase"),
        (r"\bstronger\s+than\b", 3.0, "relative strength phrase"),
        (r"\bfaster\s+than\b", 2.8, "relative speed phrase"),
    ),
    "scientific_evidence": (
        (r"\bstud(?:y|ies)\b", 3.8, "study request marker"),
        (r"\bresearch\b", 3.8, "research request marker"),
        (r"\bpaper(?:s)?\b", 3.5, "paper request marker"),
        (r"\bliterature\b", 3.4, "literature request marker"),
        (r"\bevidence\b", 3.2, "evidence request marker"),
        (r"\bmeta[-\s]?analysis\b", 3.4, "meta-analysis request"),
        (r"\bshow\b.*\bstud", 3.4, "show studies phrase"),
    ),
    "personalized_physiology": (
        (r"\bi\s+am\b", 1.6, "first-person profile declaration"),
        (r"\bmy\b", 0.8, "first-person context marker"),
        (r"\bi\s+drank\b", 2.2, "personal drinking event marker"),
        (r"\bi\s+had\b", 1.8, "personal intake marker"),
        (r"\bmale\b|\bfemale\b", 1.8, "sex marker in profile"),
        (r"\bfasted\b|\bfed\b|\bempty\s+stomach\b", 1.8, "fed-state marker"),
        (r"\b\d{2,3}\s*(?:kg|kgs|kilograms?)\b", 2.1, "body weight marker"),
    ),
    "retrieval_only": (
        (r"\bwhat\s+is\b", 2.1, "definition query marker"),
        (r"\bwhat\s+are\b", 1.9, "plural definition query marker"),
        (r"\bdefine\b", 2.1, "define command"),
        (r"\btell\s+me\s+about\b", 2.0, "tell-me-about marker"),
        (r"\bmeaning\s+of\b", 2.0, "meaning-of marker"),
    ),
}


@dataclass(frozen=True)
class RouteQueryResult:
    intent: str
    sub_intents: List[str]
    required_modules: List[str]
    required_inputs: Dict[str, bool]
    missing_required_inputs: List[str]
    response_style: str
    confidence: float
    routing_reasoning: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(payload["confidence"]), 6)
        payload["sub_intents"] = list(payload["sub_intents"])
        payload["required_modules"] = list(payload["required_modules"])
        payload["missing_required_inputs"] = list(payload["missing_required_inputs"])
        payload["routing_reasoning"] = list(payload["routing_reasoning"])
        payload["required_inputs"] = {
            key: bool(payload["required_inputs"].get(key, False)) for key in INPUT_FIELDS
        }
        return payload


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text).lower()).strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _clean_text(text)
    if not candidate:
        return None

    # Try strict JSON first.
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Fall back to first {...} block.
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None
    block = match.group(0)
    try:
        parsed = json.loads(block)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _provided_inputs(query: str) -> Dict[str, bool]:
    q = _normalize_query(query)

    has_weight = bool(
        re.search(r"\b\d{2,3}(?:\.\d+)?\s*(?:kg|kgs|kilograms?)\b", q)
        or re.search(r"\bweigh\s*\d{2,3}(?:\.\d+)?\s*(?:kg|kgs|kilograms?)\b", q)
    )
    has_sex = bool(re.search(r"\b(male|female|man|woman)\b", q))
    has_age = bool(
        re.search(r"\b(?:age\s*)?\d{1,3}\s*(?:years?\s*old|yo|y/o)\b", q)
        or re.search(r"\bi\s*am\s*\d{1,3}\b", q)
    )
    has_fed_state = bool(
        re.search(r"\b(fed|ate|eaten|fasted|empty\s+stomach|with\s+food|without\s+food)\b", q)
    )

    beverage_hits = [token for token in BEVERAGE_TOKENS if re.search(rf"\b{re.escape(token)}\b", q)]
    has_beverages = bool(beverage_hits) or bool(re.search(r"\b(drank|drinking|drink)\b", q))

    has_amount = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:ml|l|oz|shots?|glasses?|beers?|drinks?)\b", q)
        or re.search(r"\b(one|two|three|four|five)\s*(?:glass|glasses|beer|beers|shot|shots)\b", q)
    )
    has_time_since = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\s*(?:ago|since)\b", q)
        or re.search(r"\b(?:in|over)\s*\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b", q)
        or re.search(r"\bjust\s+drank\b", q)
    )

    return {
        "body_weight": has_weight,
        "sex": has_sex,
        "age": has_age,
        "fed_state": has_fed_state,
        "beverages": has_beverages,
        "drink_amount": has_amount,
        "time_since_drinking": has_time_since,
    }


def _infer_response_style(query: str) -> str:
    q = _normalize_query(query)

    for marker in RESPONSE_STYLE_KEYWORDS["scientific"]:
        if marker in q:
            return "scientific"

    for marker in RESPONSE_STYLE_KEYWORDS["technical"]:
        if marker in q:
            return "technical"

    for marker in RESPONSE_STYLE_KEYWORDS["layman"]:
        if marker in q:
            return "layman"

    return "layman"


def _infer_sub_intents(intent: str, query: str) -> List[str]:
    q = _normalize_query(query)
    sub_intents: List[str] = []

    if intent == "simulation":
        if "bac" in q or "blood alcohol" in q:
            sub_intents.append("bac_forecast")
        if "sober" in q:
            sub_intents.append("sobriety_timing")
        if "time to clear" in q or "clear the alcohol" in q:
            sub_intents.append("clearance_timing")
        if "drunk" in q or "intox" in q:
            sub_intents.append("intoxication_level")
        if re.search(r"\bdrive|driving|legal\s+limit|car|ride\s+home\b", q):
            sub_intents.append("driving_safety_check")
        if re.search(r"\bhow\s+much\s+more\s+can\s+i\s+drink\s+before\b", q):
            sub_intents.append("unsafe_extra_amount_calculation")
        elif "how much more can i drink" in q or "keep drinking" in q or "drink more" in q:
            sub_intents.append("unsafe_continue_drinking_recommendation")
        if (
            "how drunk am i" in q
            or "how much alcohol is in my body" in q
            or "what is happening in my body" in q
            or "what will it do to me" in q
            or "how long to clear" in q
            or "what chemicals are in this drink" in q
        ):
            sub_intents.append("informational_current_risk")

    elif intent == "mechanistic_explanation":
        if "why" in q:
            sub_intents.append("causal_why")
        if "hit harder" in q:
            sub_intents.append("relative_potency_mechanism")
        if "sleepy" in q:
            sub_intents.append("sedation_mechanism")

    elif intent == "toxicity_risk":
        if "headache" in q:
            sub_intents.append("headache_risk")
        if "hangover" in q:
            sub_intents.append("hangover_risk")
        if "stomach" in q or "nausea" in q:
            sub_intents.append("gastrointestinal_risk")

    elif intent == "comparison":
        sub_intents.append("comparative_effects")
        if "harder" in q or "stronger" in q:
            sub_intents.append("potency_comparison")
        if "faster" in q:
            sub_intents.append("onset_comparison")

    elif intent == "scientific_evidence":
        sub_intents.append("literature_retrieval")

    elif intent == "personalized_physiology":
        sub_intents.append("profile_conditioned_analysis")
        if "drank" in q or "drink" in q:
            sub_intents.append("intake_event_context")

    elif intent == "retrieval_only":
        sub_intents.append("definition_lookup")

    if not sub_intents:
        sub_intents = [f"{intent}_general"]

    return sorted(set(sub_intents))


def _required_inputs_for_intent(intent: str) -> Dict[str, bool]:
    template = INTENT_REQUIRED_INPUTS.get(intent, {})
    return {field: bool(template.get(field, False)) for field in INPUT_FIELDS}


def _compute_confidence(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    base = top_score / (top_score + max(second_score, 0.0) + 1e-9)
    margin = max(top_score - max(second_score, 0.0), 0.0) / (top_score + 1e-9)
    absolute = min(1.0, top_score / 6.0)
    confidence = (0.45 * base) + (0.35 * margin) + (0.20 * absolute)
    return max(0.01, min(0.99, confidence))


def _canonical_intent(value: Any) -> Optional[str]:
    text = _normalize_query(value)
    if text in INTENT_CLASSES:
        return text
    alias_map = {
        "mechanistic": "mechanistic_explanation",
        "mechanistic explanation": "mechanistic_explanation",
        "toxicity": "toxicity_risk",
        "toxic": "toxicity_risk",
        "personalized": "personalized_physiology",
        "personalized physiology": "personalized_physiology",
        "evidence": "scientific_evidence",
        "science": "scientific_evidence",
        "retrieval": "retrieval_only",
    }
    return alias_map.get(text)


def _default_ollama_disambiguator(query: str, deterministic_payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if shutil.which("ollama") is None:
        return None

    prompt = (
        "You are a deterministic query intent disambiguator for an alcohol physiology system.\n"
        "Choose exactly one intent from:\n"
        "- simulation\n"
        "- mechanistic_explanation\n"
        "- toxicity_risk\n"
        "- comparison\n"
        "- scientific_evidence\n"
        "- personalized_physiology\n"
        "- retrieval_only\n\n"
        "Return JSON only, no markdown, with keys:\n"
        "intent (string), sub_intents (array of strings), response_style (technical|layman|scientific),\n"
        "confidence (0..1 float), routing_reasoning (array of short strings).\n\n"
        f"User query: {query}\n"
        f"Deterministic baseline: {json.dumps(dict(deterministic_payload), sort_keys=True)}\n"
    )

    try:
        completed = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, "--format", "json"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None

    raw = completed.stdout if completed.returncode == 0 else completed.stderr
    payload = _extract_json_object(raw)
    if payload is None:
        return None

    intent = _canonical_intent(payload.get("intent"))
    if intent is None:
        return None

    sub_intents_raw = payload.get("sub_intents", [])
    if not isinstance(sub_intents_raw, list):
        sub_intents_raw = []

    routing_reasoning_raw = payload.get("routing_reasoning", [])
    if not isinstance(routing_reasoning_raw, list):
        routing_reasoning_raw = []

    response_style = _normalize_query(payload.get("response_style"))
    if response_style not in {"technical", "layman", "scientific"}:
        response_style = "layman"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(0.99, confidence))

    return {
        "intent": intent,
        "sub_intents": [
            _clean_text(item) for item in sub_intents_raw if _clean_text(item)
        ],
        "response_style": response_style,
        "confidence": confidence,
        "routing_reasoning": [
            _clean_text(item) for item in routing_reasoning_raw if _clean_text(item)
        ],
    }


class QueryRouter:
    """Deterministic first-pass query router with optional local Ollama disambiguation."""

    def __init__(
        self,
        *,
        enable_llm_fallback: bool = True,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        disambiguator: Optional[
            Callable[[str, Mapping[str, Any]], Optional[Dict[str, Any]]]
        ] = None,
    ) -> None:
        self.enable_llm_fallback = bool(enable_llm_fallback)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.disambiguator = disambiguator or _default_ollama_disambiguator

    def _deterministic_route(self, query: str) -> Dict[str, Any]:
        q = _normalize_query(query)
        scores: Dict[str, float] = {intent: 0.0 for intent in INTENT_CLASSES}
        reasons: Dict[str, List[str]] = {intent: [] for intent in INTENT_CLASSES}

        for intent, rules in SCORING_RULES.items():
            for pattern, weight, reason in rules:
                if re.search(pattern, q):
                    scores[intent] += float(weight)
                    reasons[intent].append(f"+{weight:.1f}: {reason}")

        # Cross-intent contextual boosts.
        beverage_hits = [token for token in BEVERAGE_TOKENS if re.search(rf"\b{re.escape(token)}\b", q)]
        if len(beverage_hits) >= 2:
            scores["comparison"] += 1.8
            reasons["comparison"].append("+1.8: multiple beverage entities present")

        provided = _provided_inputs(q)
        provided_count = sum(1 for value in provided.values() if value)
        if provided_count >= 3:
            scores["personalized_physiology"] += 2.8
            reasons["personalized_physiology"].append(
                "+2.8: >=3 structured profile/input signals found"
            )
        elif provided_count >= 2:
            scores["personalized_physiology"] += 1.6
            reasons["personalized_physiology"].append(
                "+1.6: >=2 structured profile/input signals found"
            )

        if re.search(r"\b(why|how\s+does|mechanism)\b", q) and re.search(
            r"\b(headache|hangover|nausea|stomach|migraine)\b", q
        ):
            scores["toxicity_risk"] += 1.8
            reasons["toxicity_risk"].append("+1.8: symptom-focused causal question")

        if re.search(r"\b(show|find|give)\b", q) and re.search(
            r"\b(studies|research|papers|evidence)\b", q
        ):
            scores["scientific_evidence"] += 1.4
            reasons["scientific_evidence"].append("+1.4: explicit evidence retrieval verb")

        if re.search(r"\bwhat\s+is\b", q) and all(
            scores[intent] < 3.0
            for intent in (
                "simulation",
                "comparison",
                "toxicity_risk",
                "scientific_evidence",
                "personalized_physiology",
            )
        ):
            scores["retrieval_only"] += 1.4
            reasons["retrieval_only"].append("+1.4: definitional question with weak competing signals")

        # Select top two intents with deterministic tie-break by intent order.
        ranking = sorted(
            INTENT_CLASSES,
            key=lambda intent: (scores[intent], -INTENT_CLASSES.index(intent)),
            reverse=True,
        )
        top_intent = ranking[0]
        second_intent = ranking[1]

        top_score = float(scores[top_intent])
        second_score = float(scores[second_intent])
        confidence = _compute_confidence(top_score=top_score, second_score=second_score)

        if top_score <= 0:
            top_intent = "retrieval_only"
            second_intent = "scientific_evidence"
            top_score = 0.1
            second_score = 0.0
            confidence = 0.35
            reasons[top_intent].append("+0.1: defaulted to retrieval_only due to no rule matches")

        return {
            "intent": top_intent,
            "second_intent": second_intent,
            "scores": scores,
            "confidence": confidence,
            "reasoning": reasons[top_intent],
        }

    def route(self, query: str) -> RouteQueryResult:
        text = _clean_text(query)
        if not text:
            raise ValueError("Query must be a non-empty string.")

        provided_inputs = _provided_inputs(text)
        response_style = _infer_response_style(text)

        deterministic = self._deterministic_route(text)
        intent = str(deterministic["intent"])
        confidence = float(deterministic["confidence"])
        routing_reasoning = [
            "Deterministic first-pass routing with regex+keyword weighted scoring.",
            f"Top deterministic intent: {intent} (score={deterministic['scores'][intent]:.2f}).",
            f"Second intent: {deterministic['second_intent']} (score={deterministic['scores'][deterministic['second_intent']]:.2f}).",
        ]
        routing_reasoning.extend(deterministic["reasoning"])

        if self.enable_llm_fallback and confidence < self.low_confidence_threshold:
            fallback_payload = self.disambiguator(
                text,
                {
                    "deterministic_intent": intent,
                    "deterministic_confidence": round(confidence, 6),
                    "scores": {key: round(float(val), 6) for key, val in deterministic["scores"].items()},
                    "response_style": response_style,
                },
            )

            if fallback_payload is not None:
                candidate_intent = _canonical_intent(fallback_payload.get("intent"))
                if candidate_intent is not None:
                    intent = candidate_intent
                    try:
                        candidate_conf = float(fallback_payload.get("confidence", confidence))
                    except Exception:
                        candidate_conf = confidence
                    confidence = max(confidence, min(0.99, candidate_conf))
                    response_style_candidate = _normalize_query(fallback_payload.get("response_style"))
                    if response_style_candidate in {"technical", "layman", "scientific"}:
                        response_style = response_style_candidate
                    routing_reasoning.append(
                        f"Low-confidence fallback engaged via local Ollama model {OLLAMA_MODEL}."
                    )
                    fallback_reasoning = fallback_payload.get("routing_reasoning", [])
                    if isinstance(fallback_reasoning, list):
                        for line in fallback_reasoning:
                            clean = _clean_text(line)
                            if clean:
                                routing_reasoning.append(f"Fallback: {clean}")
                else:
                    routing_reasoning.append(
                        "Low-confidence fallback attempted but returned invalid intent; kept deterministic result."
                    )
            else:
                routing_reasoning.append(
                    "Low-confidence fallback attempted but unavailable/unparseable; kept deterministic result."
                )

        required_inputs = _required_inputs_for_intent(intent)
        missing_required_inputs = [
            field for field in INPUT_FIELDS if required_inputs[field] and not provided_inputs[field]
        ]
        sub_intents = _infer_sub_intents(intent=intent, query=text)

        # Preserve deterministic ordering for module orchestration.
        required_modules = list(INTENT_MODULES[intent])

        return RouteQueryResult(
            intent=intent,
            sub_intents=sub_intents,
            required_modules=required_modules,
            required_inputs=required_inputs,
            missing_required_inputs=missing_required_inputs,
            response_style=response_style,
            confidence=round(float(confidence), 6),
            routing_reasoning=routing_reasoning,
        )


def route_query(
    query: str,
    *,
    enable_llm_fallback: bool = True,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    disambiguator: Optional[Callable[[str, Mapping[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> RouteQueryResult:
    router = QueryRouter(
        enable_llm_fallback=enable_llm_fallback,
        low_confidence_threshold=low_confidence_threshold,
        disambiguator=disambiguator,
    )
    return router.route(query)


def validate_route_result_schema(result: RouteQueryResult) -> None:
    payload = result.to_dict()
    expected_keys = {
        "intent",
        "sub_intents",
        "required_modules",
        "required_inputs",
        "missing_required_inputs",
        "response_style",
        "confidence",
        "routing_reasoning",
    }
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"Invalid route schema keys: {sorted(actual_keys)}")
    if payload["intent"] not in INTENT_CLASSES:
        raise ValueError(f"Invalid intent: {payload['intent']}")
    if payload["response_style"] not in {"technical", "layman", "scientific"}:
        raise ValueError(f"Invalid response_style: {payload['response_style']}")
    if not isinstance(payload["confidence"], float):
        raise ValueError("confidence must be float")
    if payload["confidence"] < 0.0 or payload["confidence"] > 1.0:
        raise ValueError("confidence must be within [0, 1]")
    for key in INPUT_FIELDS:
        if key not in payload["required_inputs"]:
            raise ValueError(f"required_inputs missing key: {key}")


def _print_cli_output(result: RouteQueryResult, *, pretty: bool = True) -> None:
    payload = result.to_dict()
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic query intent router for alcohol physiology reasoning orchestration."
    )
    parser.add_argument("--query", type=str, default="", help="User query text to route.")
    parser.add_argument(
        "--disable-llm-fallback",
        action="store_true",
        help="Disable low-confidence local Ollama disambiguation.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=LOW_CONFIDENCE_THRESHOLD,
        help="Confidence threshold below which optional Ollama fallback is triggered.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON output instead of pretty-printed JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    query = _clean_text(args.query)

    if not query:
        raise SystemExit("Provide --query with non-empty text.")

    result = route_query(
        query,
        enable_llm_fallback=not bool(args.disable_llm_fallback),
        low_confidence_threshold=float(args.low_confidence_threshold),
    )
    validate_route_result_schema(result)
    _print_cli_output(result, pretty=not bool(args.compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
