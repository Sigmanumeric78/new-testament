"""Phase 08D deterministic grounding and safety guard.

This guard validates synthesized response payloads before user display.
It does not execute PBPK/Neo4j/Weaviate/Ollama directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from reasoning.response_synthesizer import OLLAMA_MODEL, synthesize_query
except ModuleNotFoundError:  # pragma: no cover - direct script execution path fix
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from reasoning.response_synthesizer import OLLAMA_MODEL, synthesize_query

MANDATORY_SAFETY_NOTES: Tuple[str, ...] = (
    "This is an estimate, not medical advice.",
    "Do not use this to decide whether it is safe to drive.",
)

TOXICITY_SAFETY_NOTE = "Seek medical help for severe symptoms."

BLOCKED_SAFETY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bsafe to drive\b", "Contains unsafe driving claim ('safe to drive')."),
    (r"\byou can drive\b", "Contains unsafe driving claim ('you can drive')."),
    (r"\bdrink more\b", "Encourages additional alcohol intake ('drink more')."),
    (r"\bsafe to drink more\b", "Encourages additional alcohol intake ('safe to drink more')."),
    (r"\bguaranteed sober\b", "Contains absolute sobriety guarantee ('guaranteed sober')."),
    (r"\bno risk\b", "Contains unsafe zero-risk claim ('no risk')."),
    (r"\bmedical diagnosis\b", "Contains prohibited diagnosis framing ('medical diagnosis')."),
    (r"\bthis proves you are healthy\b", "Contains unsupported health proof claim."),
    (r"\bignore symptoms\b", "Contains unsafe symptom-disregard instruction."),
)

DISEASE_TERMS: Tuple[str, ...] = (
    "disease",
    "disorder",
    "syndrome",
    "diabetes",
    "hypertension",
    "cancer",
    "cirrhosis",
    "hepatitis",
    "liver disease",
    "ulcer",
)

STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "why",
    "will",
    "with",
    "you",
    "your",
}

GENERIC_ALLOWED_TOKENS: Set[str] = {
    "absorption",
    "alcohol",
    "answer",
    "based",
    "beverage",
    "blood",
    "cause",
    "causal",
    "confidence",
    "difference",
    "drink",
    "drinking",
    "effect",
    "effects",
    "estimate",
    "evidence",
    "faster",
    "higher",
    "intoxication",
    "likely",
    "limited",
    "mechanism",
    "metabolism",
    "model",
    "path",
    "pbpk",
    "possible",
    "response",
    "risk",
    "route",
    "simulation",
    "style",
    "symptom",
    "symptoms",
    "time",
    "toxicity",
    "uncertainty",
    "current",
    "retrieval",
    "set",
    "records",
    "matching",
    "strongest",
    "source",
    "row",
    "containing",
    "compound",
    "interpretation",
    "present",
    "style",
    "beverages",
    "supports",
    "possible",
    "explanation",
    "sensitive",
    "individuals",
    "headaches",
    "limitations",
    "summarize",
    "observations",
    "establish",
    "universal",
    "causality",
    "safety",
    "boundary",
    "summary",
    "used",
    "decisions",
    "wine",
    "champagne",
    "dessert",
    "table",
    "sparkling",
}


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


def _tokenize(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _as_list_of_strings(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    values: List[str] = []
    for item in raw:
        text = _clean_text(item)
        if text:
            values.append(text)
    return values


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _json_safe(payload: Mapping[str, Any]) -> bool:
    try:
        json.dumps(payload, sort_keys=True)
        return True
    except Exception:
        return False


def _contains_case_insensitive(haystack: str, needle: str) -> bool:
    return _normalize_text(needle) in _normalize_text(haystack)


def _strip_standard_safety_notes(answer: str) -> str:
    cleaned = answer
    for note in list(MANDATORY_SAFETY_NOTES) + [TOXICITY_SAFETY_NOTE]:
        cleaned = re.sub(re.escape(note), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _overlap_ratio(source: str, answer_tokens: Set[str]) -> float:
    source_tokens = [token for token in _tokenize(source) if len(token) > 2 and token not in STOPWORDS]
    if not source_tokens:
        return 0.0
    unique = set(source_tokens)
    matched = sum(1 for token in unique if token in answer_tokens)
    return float(matched) / float(max(len(unique), 1))


def _build_evidence_tokens(response_payload: Mapping[str, Any]) -> Set[str]:
    parts: List[str] = []

    parts.append(_clean_text(response_payload.get("query")))

    for item in _as_list_of_strings(response_payload.get("used_facts")):
        parts.append(item)

    for item in _as_list_of_strings(response_payload.get("used_causal_paths")):
        parts.append(item)

    for item in list(response_payload.get("used_evidence", []) or []):
        if not isinstance(item, Mapping):
            continue
        parts.append(_clean_text(item.get("title")))
        parts.append(_clean_text(item.get("content_excerpt")))
        parts.append(_clean_text(item.get("source_dataset")))
        parts.append(_clean_text(item.get("source_file")))

    parts.append(json.dumps(response_payload.get("simulation_summary"), sort_keys=True, default=str))
    parts.append(json.dumps(response_payload.get("toxicity_summary"), sort_keys=True, default=str))

    for item in _as_list_of_strings(response_payload.get("limitations")):
        parts.append(item)

    evidence_tokens: Set[str] = set()
    for part in parts:
        evidence_tokens.update(_tokenize(part))
    return evidence_tokens


def _detect_unsafe_claims(answer: str) -> List[str]:
    findings: List[str] = []
    normalized_answer = _normalize_text(answer)
    for note in list(MANDATORY_SAFETY_NOTES) + [TOXICITY_SAFETY_NOTE]:
        normalized_answer = re.sub(re.escape(_normalize_text(note)), " ", normalized_answer)
    normalized_answer = re.sub(r"\s+", " ", normalized_answer).strip()

    negative_drink_more = bool(
        re.search(r"\b(?:do\s+not|don't|not|should\s+not|cannot|can't)\s+drink\s+more\b", normalized_answer)
    )

    for pattern, reason in BLOCKED_SAFETY_PATTERNS:
        if pattern == r"\bdrink more\b":
            if re.search(pattern, normalized_answer) and not negative_drink_more:
                findings.append(reason)
            continue
        if re.search(pattern, normalized_answer):
            findings.append(reason)

    if re.search(r"\byou\s+have\b", normalized_answer):
        for disease in DISEASE_TERMS:
            if disease in normalized_answer:
                findings.append("Contains unsupported medical diagnosis claim ('you have [disease]').")
                break

    return sorted(set(findings))


def _required_safety_notes(response_payload: Mapping[str, Any]) -> List[str]:
    notes = list(MANDATORY_SAFETY_NOTES)
    toxicity_summary = response_payload.get("toxicity_summary")
    if toxicity_summary is not None:
        notes.append(TOXICITY_SAFETY_NOTE)
    return notes


def _notes_present_map(
    *,
    answer: str,
    safety_notes: Sequence[str],
    required_notes: Sequence[str],
) -> Dict[str, bool]:
    note_text = "\n".join([_clean_text(item) for item in safety_notes if _clean_text(item)])
    mapping: Dict[str, bool] = {}
    for note in required_notes:
        present = _contains_case_insensitive(note_text, note) or _contains_case_insensitive(answer, note)
        mapping[note] = bool(present)
    return mapping


def _compute_grounding(
    *,
    answer: str,
    response_payload: Mapping[str, Any],
) -> Tuple[float, bool, List[str]]:
    warnings: List[str] = []
    clean_answer = _strip_standard_safety_notes(answer)

    answer_tokens = [token for token in _tokenize(clean_answer) if len(token) > 2 and token not in STOPWORDS]
    unique_answer_tokens = sorted(set(answer_tokens))

    if not unique_answer_tokens:
        return 0.0, True, ["Answer has no meaningful grounded content after removing safety notes."]

    evidence_tokens = _build_evidence_tokens(response_payload)

    known_tokens = [
        token
        for token in unique_answer_tokens
        if token in evidence_tokens or token in GENERIC_ALLOWED_TOKENS
    ]
    unknown_tokens = [
        token
        for token in unique_answer_tokens
        if token not in evidence_tokens and token not in GENERIC_ALLOWED_TOKENS
    ]

    token_overlap = float(len(known_tokens)) / float(max(len(unique_answer_tokens), 1))
    unknown_ratio = float(len(unknown_tokens)) / float(max(len(unique_answer_tokens), 1))

    source_phrases = _as_list_of_strings(response_payload.get("used_facts")) + _as_list_of_strings(
        response_payload.get("used_causal_paths")
    )

    if source_phrases:
        matched_source = sum(1 for phrase in source_phrases if _overlap_ratio(phrase, set(unique_answer_tokens)) >= 0.30)
        phrase_coverage = float(matched_source) / float(max(len(source_phrases), 1))
    else:
        phrase_coverage = 0.65

    used_evidence = list(response_payload.get("used_evidence", []) or [])
    evidence_mention = 1.0
    if used_evidence:
        evidence_mention = 0.0
        for item in used_evidence:
            if not isinstance(item, Mapping):
                continue
            title = _clean_text(item.get("title"))
            if not title:
                continue
            if _overlap_ratio(title, set(unique_answer_tokens)) >= 0.25:
                evidence_mention = 1.0
                break

    external_unsupported_flag = bool(response_payload.get("unsupported_claims_detected"))

    unsupported = False
    if external_unsupported_flag:
        unsupported = True
        warnings.append("Response synthesizer flagged unsupported claims.")

    if len(unique_answer_tokens) >= 12 and unknown_ratio >= 0.45:
        unsupported = True
        warnings.append("High proportion of major answer tokens are not supported by evidence.")

    score = (0.65 * token_overlap) + (0.25 * phrase_coverage) + (0.10 * evidence_mention)
    if unsupported:
        score = min(score, 0.69)

    score = max(0.0, min(1.0, score))
    return round(score, 6), unsupported, sorted(set(warnings))


def _compute_safety_score(
    *,
    unsafe_claims_detected: bool,
    missing_note_count_before_repair: int,
) -> float:
    if unsafe_claims_detected:
        return 0.0
    penalty = 0.01 * float(max(missing_note_count_before_repair, 0))
    score = max(0.0, 1.0 - penalty)
    return round(min(score, 1.0), 6)


class GroundingSafetyGuard:
    """Deterministic grounding + safety validator for synthesized responses."""

    def validate(self, response_payload: Mapping[str, Any]) -> Dict[str, Any]:
        query = _clean_text(response_payload.get("query"))
        answer = _clean_text(response_payload.get("answer"))
        safety_notes_input = _as_list_of_strings(response_payload.get("safety_notes"))

        required_notes = _required_safety_notes(response_payload)
        notes_map_before = _notes_present_map(
            answer=answer,
            safety_notes=safety_notes_input,
            required_notes=required_notes,
        )
        missing_notes_before = [note for note in required_notes if not notes_map_before.get(note, False)]

        unsafe_findings = _detect_unsafe_claims(answer)
        unsafe_claims_detected = bool(unsafe_findings)

        required_edits: List[str] = []
        final_answer = answer
        warnings: List[str] = []

        # Auto-repair only for missing safety notes and only when no blocked unsafe claims are present.
        if not unsafe_claims_detected and missing_notes_before:
            append_block = "\n\n" + "\n".join(missing_notes_before)
            final_answer = (answer + append_block).strip()
            for note in missing_notes_before:
                required_edits.append(f"Appended missing safety note: {note}")
            warnings.append("Missing safety notes were auto-repaired.")

        notes_map_after = _notes_present_map(
            answer=final_answer,
            safety_notes=safety_notes_input,
            required_notes=required_notes,
        )

        safety_notes_present = all(notes_map_after.values()) if notes_map_after else False
        medical_disclaimer_present = bool(notes_map_after.get(MANDATORY_SAFETY_NOTES[0], False))
        driving_warning_present = bool(notes_map_after.get(MANDATORY_SAFETY_NOTES[1], False))

        grounding_score, unsupported_claims_detected, grounding_warnings = _compute_grounding(
            answer=final_answer,
            response_payload=response_payload,
        )
        warnings.extend(grounding_warnings)

        safety_score = _compute_safety_score(
            unsafe_claims_detected=unsafe_claims_detected,
            missing_note_count_before_repair=len(missing_notes_before),
        )

        blocked_reasons: List[str] = []
        if unsafe_claims_detected:
            blocked_reasons.extend(unsafe_findings)

        if grounding_score < 0.70:
            blocked_reasons.append("Grounding score below threshold (0.70).")

        if safety_score < 0.95:
            blocked_reasons.append("Safety score below threshold (0.95).")

        if not _clean_text(final_answer):
            blocked_reasons.append("Final answer is empty.")

        approved_for_display = bool(
            safety_score >= 0.95
            and grounding_score >= 0.70
            and not unsafe_claims_detected
            and bool(_clean_text(final_answer))
        )

        if not approved_for_display and unsafe_claims_detected:
            final_answer = "Response blocked by grounding/safety guard."

        payload = {
            "query": query,
            "approved_for_display": approved_for_display,
            "final_answer": final_answer,
            "grounding_score": grounding_score,
            "safety_score": safety_score,
            "blocked_reasons": sorted(set([_clean_text(item) for item in blocked_reasons if _clean_text(item)])),
            "warnings": sorted(set([_clean_text(item) for item in warnings if _clean_text(item)])),
            "required_edits": sorted(set([_clean_text(item) for item in required_edits if _clean_text(item)])),
            "unsupported_claims_detected": bool(unsupported_claims_detected),
            "unsafe_claims_detected": bool(unsafe_claims_detected),
            "safety_notes_present": bool(safety_notes_present),
            "medical_disclaimer_present": bool(medical_disclaimer_present),
            "driving_warning_present": bool(driving_warning_present),
        }

        return payload

    def validate_query(
        self,
        query: str,
        *,
        model: str = OLLAMA_MODEL,
        timeout_seconds: int = 30,
        enable_router_llm_fallback: bool = False,
    ) -> Dict[str, Any]:
        synthesized = synthesize_query(
            query,
            model=model,
            timeout_seconds=timeout_seconds,
            enable_router_llm_fallback=enable_router_llm_fallback,
        )
        return self.validate(synthesized)


def validate_response(response_payload: Mapping[str, Any]) -> Dict[str, Any]:
    guard = GroundingSafetyGuard()
    return guard.validate(response_payload)


def guard_query(
    query: str,
    *,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = 30,
    enable_router_llm_fallback: bool = False,
) -> Dict[str, Any]:
    guard = GroundingSafetyGuard()
    return guard.validate_query(
        query,
        model=model,
        timeout_seconds=timeout_seconds,
        enable_router_llm_fallback=enable_router_llm_fallback,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08D grounding and safety guard")
    parser.add_argument("--query", type=str, default="", help="User query text")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Local Ollama model for synthesizer step")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for synthesizer model call.",
    )
    parser.add_argument(
        "--enable-router-llm-fallback",
        action="store_true",
        help="Enable optional router fallback in the synthesizer chain.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON output instead of pretty JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    query = _clean_text(args.query)
    if not query:
        raise SystemExit("Provide --query with non-empty text.")

    payload = guard_query(
        query,
        model=_clean_text(args.model) or OLLAMA_MODEL,
        timeout_seconds=int(args.timeout_seconds),
        enable_router_llm_fallback=bool(args.enable_router_llm_fallback),
    )

    if bool(args.compact):
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
