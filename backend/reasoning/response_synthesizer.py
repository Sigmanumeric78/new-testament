"""Phase 08C grounded response synthesizer.

This module only consumes structured output from the Phase 08B hybrid orchestrator.
It does not directly execute PBPK, Neo4j, or Weaviate queries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from reasoning.hybrid_orchestrator import orchestrate_query
    from utils.config import get_ollama_config
except ModuleNotFoundError:  # pragma: no cover - direct script execution path fix
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from reasoning.hybrid_orchestrator import orchestrate_query
    from utils.config import get_ollama_config

LOGGER = logging.getLogger("response_synthesizer")

OLLAMA_MODEL = "qwen2.5:3b"

RESPONSE_STYLES: Tuple[str, ...] = ("layman", "technical", "scientific")

MANDATORY_SAFETY_NOTES: Tuple[str, ...] = (
    "This is an estimate, not medical advice.",
    "Do not use this to decide whether it is safe to drive.",
)

TOXICITY_SAFETY_NOTE = "Seek medical help for severe symptoms."

SUPPORTED_INTENTS_FOR_TOXICITY_NOTE: Tuple[str, ...] = ("toxicity_risk",)

FALLBACK_UNCERTAINTY_NOTE = "Evidence is limited, so this response has uncertainty."

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
    "cause",
    "confidence",
    "difference",
    "drink",
    "drinking",
    "effect",
    "effects",
    "evidence",
    "faster",
    "higher",
    "intoxication",
    "likely",
    "limited",
    "mechanism",
    "metabolism",
    "machinery",
    "model",
    "operate",
    "path",
    "pbpk",
    "peak",
    "possible",
    "response",
    "risk",
    "route",
    "sober",
    "simulation",
    "style",
    "symptom",
    "symptoms",
    "time",
    "water",
    "dehydration",
    "clear",
    "hours",
    "drive",
    "driving",
    "drink",
    "drinking",
    "alcohol",
    "uncertainty",
}

UNSAFE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bsafe to drive\b", "Contains unsafe driving guidance."),
    (r"\bdrive safely\b", "Contains unsafe driving guidance."),
    (r"\byou should drink\b", "Encourages alcohol intake."),
    (r"\bdrink more\b", "Encourages alcohol intake."),
    (r"\bdiagnos(?:e|is|ed)\b", "Contains diagnostic claim."),
)

LAYMAN_BANNED_TERMS: Tuple[str, ...] = (
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


def _tokenize(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _clean_text(text)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None

    block = match.group(0)
    try:
        parsed = json.loads(block)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


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


def _strip_layman_banned_terms(text: str) -> str:
    output = _clean_text(text)
    if not output:
        return ""
    for term in LAYMAN_BANNED_TERMS:
        output = re.sub(rf"\b{re.escape(term)}\b", "", output, flags=re.IGNORECASE)
    output = re.sub(r"\s+", " ", output).strip()
    return output


class ResponseSynthesizer:
    """Grounded response synthesizer over hybrid orchestrator evidence bundles."""

    def __init__(
        self,
        *,
        model: str = OLLAMA_MODEL,
        timeout_seconds: int = 30,
    ) -> None:
        ollama_config = get_ollama_config()
        self.model = _clean_text(model) or OLLAMA_MODEL
        self.ollama_host = _clean_text(ollama_config.get("host"))
        self.timeout_seconds = int(timeout_seconds)
        self._model_fallback_used = False

    def _determine_response_style(self, route_payload: Mapping[str, Any]) -> str:
        intent = _clean_text(route_payload.get("intent"))
        style = _clean_text(route_payload.get("response_style"))

        if intent == "scientific_evidence":
            return "scientific"

        if style in RESPONSE_STYLES:
            return style
        return "layman"

    def _build_grounded_prompt(
        self,
        *,
        query: str,
        response_style: str,
        route_payload: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
    ) -> str:
        key_facts = list(evidence_bundle.get("key_facts", []) or [])
        causal_paths = list(evidence_bundle.get("causal_paths", []) or [])
        retrieved_evidence = list(evidence_bundle.get("retrieved_evidence", []) or [])
        simulation_summary = evidence_bundle.get("simulation_summary")
        toxicity_summary = evidence_bundle.get("toxicity_summary")
        limitations = list(evidence_bundle.get("limitations", []) or [])

        compact_evidence = []
        for item in retrieved_evidence[:8]:
            compact_evidence.append(
                {
                    "object_id": _clean_text(item.get("object_id")),
                    "collection": _clean_text(item.get("collection")),
                    "title": _clean_text(item.get("title")),
                    "content_excerpt": _clean_text(item.get("content_excerpt"))[:280],
                    "source_dataset": _clean_text(item.get("source_dataset")),
                    "source_file": _clean_text(item.get("source_file")),
                }
            )

        prompt = (
            "You are a strict grounded response synthesizer.\n"
            "You must only use the provided evidence payload.\n"
            "Do not introduce external facts, assumptions, or unsupported claims.\n"
            "Do not provide medical diagnosis.\n"
            "Do not say it is safe to drive.\n"
            "Do not encourage additional alcohol intake.\n"
            "If confidence is limited, explicitly mention uncertainty.\n"
            "Return JSON only (no markdown).\n\n"
            "Output JSON schema:\n"
            "{\n"
            "  \"answer\": string,\n"
            "  \"used_facts\": [string],\n"
            "  \"used_causal_paths\": [string],\n"
            "  \"used_evidence_ids\": [string],\n"
            "  \"limitations\": [string]\n"
            "}\n\n"
            f"User query: {query}\n"
            f"Response style: {response_style}\n"
            f"Route intent: {_clean_text(route_payload.get('intent'))}\n"
            f"Route confidence: {_safe_float(route_payload.get('confidence')):.6f}\n"
            f"Key facts: {json.dumps(key_facts, ensure_ascii=True)}\n"
            f"Causal paths: {json.dumps(causal_paths, ensure_ascii=True)}\n"
            f"Retrieved evidence: {json.dumps(compact_evidence, ensure_ascii=True)}\n"
            f"Simulation summary: {json.dumps(simulation_summary, ensure_ascii=True)}\n"
            f"Toxicity summary: {json.dumps(toxicity_summary, ensure_ascii=True)}\n"
            f"Limitations: {json.dumps(limitations, ensure_ascii=True)}\n"
        )
        return prompt

    def _invoke_ollama(self, prompt: str) -> str:
        if shutil.which("ollama") is None:
            raise RuntimeError("ollama executable not found.")

        env = os.environ.copy()
        if self.ollama_host:
            env["OLLAMA_HOST"] = self.ollama_host

        completed = subprocess.run(
            ["ollama", "run", self.model, "--format", "json"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
            env=env,
        )

        if completed.returncode != 0:
            raise RuntimeError(_clean_text(completed.stderr) or "ollama returned non-zero exit code")

        output = _clean_text(completed.stdout)
        if not output:
            raise RuntimeError("ollama returned empty output")
        return output

    def _select_used_evidence(
        self,
        requested_ids: Sequence[str],
        retrieved_evidence: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in retrieved_evidence:
            object_id = _clean_text(item.get("object_id"))
            if not object_id:
                continue
            by_id[object_id] = {
                "object_id": object_id,
                "collection": _clean_text(item.get("collection")),
                "title": _clean_text(item.get("title")),
                "content_excerpt": _clean_text(item.get("content_excerpt")),
                "score": item.get("score"),
                "distance": item.get("distance"),
                "source_dataset": _clean_text(item.get("source_dataset")),
                "source_file": _clean_text(item.get("source_file")),
            }

        selected: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for evidence_id in requested_ids:
            if evidence_id in seen:
                continue
            if evidence_id not in by_id:
                continue
            seen.add(evidence_id)
            selected.append(by_id[evidence_id])
        return selected

    def _parse_model_payload(
        self,
        raw: str,
        evidence_bundle: Mapping[str, Any],
    ) -> Dict[str, Any]:
        parsed = _extract_json_object(raw)
        if parsed is None:
            raise ValueError("Model output did not contain valid JSON object.")

        answer = _clean_text(parsed.get("answer"))
        if not answer:
            raise ValueError("Model output missing non-empty 'answer'.")

        available_facts = list(evidence_bundle.get("key_facts", []) or [])
        available_paths = list(evidence_bundle.get("causal_paths", []) or [])
        available_evidence = list(evidence_bundle.get("retrieved_evidence", []) or [])

        requested_facts = _as_list_of_strings(parsed.get("used_facts"))
        requested_paths = _as_list_of_strings(parsed.get("used_causal_paths"))
        requested_evidence_ids = _as_list_of_strings(parsed.get("used_evidence_ids"))
        requested_limitations = _as_list_of_strings(parsed.get("limitations"))

        used_facts = [item for item in requested_facts if item in available_facts]
        used_causal_paths = [item for item in requested_paths if item in available_paths]
        used_evidence = self._select_used_evidence(requested_evidence_ids, available_evidence)

        return {
            "answer": answer,
            "used_facts": used_facts,
            "used_causal_paths": used_causal_paths,
            "used_evidence": used_evidence,
            "limitations": requested_limitations,
        }

    def _rule_based_response(
        self,
        *,
        query: str,
        response_style: str,
        route_payload: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
    ) -> Dict[str, Any]:
        key_facts = list(evidence_bundle.get("key_facts", []) or [])
        causal_paths = list(evidence_bundle.get("causal_paths", []) or [])
        retrieved_evidence = list(evidence_bundle.get("retrieved_evidence", []) or [])
        simulation_summary = evidence_bundle.get("simulation_summary")
        toxicity_summary = evidence_bundle.get("toxicity_summary")
        limitations = _as_list_of_strings(evidence_bundle.get("limitations"))

        used_facts = key_facts[:3]
        used_causal_paths = causal_paths[:2]
        used_evidence = []
        for item in retrieved_evidence[:3]:
            used_evidence.append(
                {
                    "object_id": _clean_text(item.get("object_id")),
                    "collection": _clean_text(item.get("collection")),
                    "title": _clean_text(item.get("title")),
                    "content_excerpt": _clean_text(item.get("content_excerpt")),
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                    "source_dataset": _clean_text(item.get("source_dataset")),
                    "source_file": _clean_text(item.get("source_file")),
                }
            )

        answer_parts: List[str] = []

        if response_style == "scientific":
            answer_parts.append("Evidence-based summary from the current retrieval and reasoning bundle:")
        elif response_style == "technical":
            answer_parts.append("Technical summary from routed evidence modules:")
        else:
            answer_parts.append("Based on the evidence collected for your question:")

        if simulation_summary and isinstance(simulation_summary, dict):
            sims = simulation_summary.get("simulations", [])
            if isinstance(sims, list) and sims:
                first = sims[0]
                beverage = _clean_text(first.get("beverage")) or "the beverage"
                peak = first.get("peak_bac_percent")
                sober = first.get("time_to_sober_h")
                answer_parts.append(
                    f"PBPK estimated {beverage} peak BAC around {peak} with time-to-sober around {sober} hours."
                )

        if toxicity_summary and isinstance(toxicity_summary, dict):
            compounds = list(toxicity_summary.get("risk_compounds", []) or [])
            risk_types = list(toxicity_summary.get("risk_types", []) or [])
            if compounds or risk_types:
                compound_text = ", ".join(compounds[:4]) if compounds else "none specified"
                risk_text = ", ".join(risk_types[:3]) if risk_types else "none specified"
                answer_parts.append(
                    f"Toxicity-linked evidence points to compounds ({compound_text}) and risk types ({risk_text})."
                )

        if used_causal_paths:
            answer_parts.append(f"Representative causal path: {used_causal_paths[0]}")

        if used_evidence:
            titles = [item["title"] for item in used_evidence if _clean_text(item.get("title"))]
            if titles:
                answer_parts.append(f"Retrieved evidence includes: {', '.join(titles[:2])}.")

        if not simulation_summary and not toxicity_summary and not used_causal_paths and not used_evidence:
            answer_parts.append("The current bundle has limited direct evidence for a strong conclusion.")

        if _safe_float(evidence_bundle.get("confidence_score"), 0.0) < 0.55:
            answer_parts.append(FALLBACK_UNCERTAINTY_NOTE)

        if limitations:
            answer_parts.append(f"Known limitations: {limitations[0]}")

        answer = " ".join([part for part in answer_parts if _clean_text(part)])

        return {
            "answer": answer,
            "used_facts": used_facts,
            "used_causal_paths": used_causal_paths,
            "used_evidence": used_evidence,
            "limitations": limitations,
        }

    def _collect_evidence_tokens(
        self,
        query: str,
        route_payload: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
    ) -> Set[str]:
        token_source_parts: List[str] = [query, _clean_text(route_payload.get("intent"))]

        for item in list(evidence_bundle.get("key_facts", []) or []):
            token_source_parts.append(_clean_text(item))

        for item in list(evidence_bundle.get("causal_paths", []) or []):
            token_source_parts.append(_clean_text(item))

        for item in list(evidence_bundle.get("retrieved_evidence", []) or []):
            token_source_parts.append(_clean_text(item.get("title")))
            token_source_parts.append(_clean_text(item.get("content_excerpt")))
            token_source_parts.append(_clean_text(item.get("source_dataset")))
            token_source_parts.append(_clean_text(item.get("source_file")))

        token_source_parts.append(json.dumps(evidence_bundle.get("simulation_summary"), sort_keys=True, default=str))
        token_source_parts.append(json.dumps(evidence_bundle.get("toxicity_summary"), sort_keys=True, default=str))
        token_source_parts.append(json.dumps(evidence_bundle.get("limitations"), sort_keys=True, default=str))

        evidence_tokens: Set[str] = set()
        for chunk in token_source_parts:
            evidence_tokens.update(_tokenize(chunk))
        return evidence_tokens

    def _detect_unsupported_claims(
        self,
        answer: str,
        *,
        query: str,
        route_payload: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
        strict_token_check: bool = True,
    ) -> Tuple[bool, List[str]]:
        findings: List[str] = []
        normalized_answer = _normalize_text(answer)
        for note in list(MANDATORY_SAFETY_NOTES) + [TOXICITY_SAFETY_NOTE]:
            normalized_answer = re.sub(re.escape(_normalize_text(note)), " ", normalized_answer)
        normalized_answer = re.sub(r"\s+", " ", normalized_answer).strip()

        negative_drink_more = bool(
            re.search(r"\b(?:do\s+not|don't|not|should\s+not|cannot|can't)\s+drink\s+more\b", normalized_answer)
        )

        for pattern, reason in UNSAFE_PATTERNS:
            if pattern == r"\bdrink more\b":
                if re.search(pattern, normalized_answer) and not negative_drink_more:
                    findings.append(reason)
                continue
            if re.search(pattern, normalized_answer):
                findings.append(reason)

        evidence_tokens = self._collect_evidence_tokens(query, route_payload, evidence_bundle)
        answer_tokens = [
            token
            for token in _tokenize(answer)
            if len(token) > 2 and token not in STOPWORDS
        ]

        if not answer_tokens:
            findings.append("Answer has no meaningful tokens.")
            return True, findings

        unique_answer_tokens = sorted(set(answer_tokens))
        unknown_tokens = [
            token
            for token in unique_answer_tokens
            if token not in evidence_tokens and token not in GENERIC_ALLOWED_TOKENS
        ]

        unknown_ratio = float(len(unknown_tokens)) / float(max(len(unique_answer_tokens), 1))
        if strict_token_check:
            if len(unique_answer_tokens) >= 14 and len(unknown_tokens) >= 10 and unknown_ratio >= 0.68:
                findings.append(
                    "High unknown-token ratio relative to provided evidence (possible unsupported content)."
                )

        return (len(findings) > 0), findings

    def _build_safety_notes(self, intent: str) -> List[str]:
        notes = list(MANDATORY_SAFETY_NOTES)
        if intent in SUPPORTED_INTENTS_FOR_TOXICITY_NOTE:
            notes.append(TOXICITY_SAFETY_NOTE)
        return notes

    def _extract_primary_simulation(self, evidence_bundle: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        summary = evidence_bundle.get("simulation_summary")
        if not isinstance(summary, Mapping):
            return None
        simulations = summary.get("simulations")
        if not isinstance(simulations, list) or not simulations:
            return None
        first = simulations[0]
        if not isinstance(first, Mapping):
            return None
        return first

    def _is_unsafe_intent_query(self, query: str) -> bool:
        q = _normalize_text(query)
        patterns = (
            r"\bhow\s+much\s+more\s+can\s+i\s+drink\b",
            r"\bshould\s+i\s+keep\s+drinking\b",
            r"\bkeep\s+drinking\b",
            r"\bcan\s+i\s+drive\b",
            r"\bam\s+i\s+safe\s+to\s+drive\b",
            r"\bsafe\s+to\s+drive\b",
            r"\bhow\s+much\s+before\s+i\s+am\s+too\s+drunk\b",
            r"\bhow\s+much\s+alcohol\s+will\s+be\s+toxic\b",
        )
        return any(re.search(pattern, q) for pattern in patterns)

    def _build_default_layman_answer(
        self,
        *,
        query: str,
        intent: str,
        evidence_bundle: Mapping[str, Any],
        safety_notes: Sequence[str],
    ) -> str:
        parts: List[str] = []
        parts.append("For your situation, you should not drink more right now.")

        if self._is_unsafe_intent_query(query):
            parts.append(
                "I can’t help calculate a safe amount to keep drinking or drive. "
                "I can estimate your current risk and suggest safer next steps."
            )

        simulation = self._extract_primary_simulation(evidence_bundle)
        if simulation is not None:
            peak = _safe_float(simulation.get("peak_bac_percent"), -1.0)
            t_sober = _safe_float(simulation.get("time_to_sober_h"), -1.0)
            t_peak = _safe_float(simulation.get("time_to_peak_h"), -1.0)
            if peak >= 0:
                parts.append(f"Your estimated peak BAC is around {peak:.3f}%.")
            if t_sober >= 0:
                parts.append(f"It may take around {t_sober:.1f} hours for your body to clear the alcohol.")
            if t_peak >= 0:
                parts.append(f"Your peak effect may be around {t_peak:.1f} hours after drinking.")
        else:
            parts.append("I do not have enough details for a precise BAC estimate, so this is conservative guidance.")

        parts.append("Do not drive or operate machinery right now.")
        parts.append("Water can help dehydration, but it will not make alcohol leave your body faster.")

        toxicity_summary = evidence_bundle.get("toxicity_summary")
        if isinstance(toxicity_summary, Mapping):
            compounds = list(toxicity_summary.get("risk_compounds", []) or [])
            if compounds:
                clean_compounds = [_clean_text(item) for item in compounds if _clean_text(item)]
                if clean_compounds:
                    parts.append(f"Possible symptom-linked compounds include {', '.join(clean_compounds[:4])}.")

        if _safe_float(evidence_bundle.get("confidence_score"), 0.0) < 0.55:
            parts.append("This estimate has higher uncertainty because evidence is limited.")

        if intent == "toxicity_risk":
            parts.append("Seek medical help for severe symptoms.")

        for note in safety_notes:
            clean_note = _clean_text(note)
            if clean_note:
                parts.append(clean_note)

        answer = " ".join([_clean_text(item) for item in parts if _clean_text(item)])
        return _strip_layman_banned_terms(answer)

    def synthesize_response(self, orchestrator_payload: Mapping[str, Any]) -> Dict[str, Any]:
        query = _clean_text(orchestrator_payload.get("query"))
        if not query:
            raise ValueError("orchestrator_payload must include non-empty 'query'.")

        route_payload = orchestrator_payload.get("route")
        if not isinstance(route_payload, Mapping):
            raise ValueError("orchestrator_payload missing 'route' mapping.")

        evidence_bundle = orchestrator_payload.get("evidence_bundle")
        if not isinstance(evidence_bundle, Mapping):
            raise ValueError("orchestrator_payload missing 'evidence_bundle' mapping.")

        response_style = self._determine_response_style(route_payload)
        intent = _clean_text(route_payload.get("intent"))

        safety_notes = self._build_safety_notes(intent)

        parsed_payload: Dict[str, Any]
        model_limitations: List[str] = []
        self._model_fallback_used = False

        prompt = self._build_grounded_prompt(
            query=query,
            response_style=response_style,
            route_payload=route_payload,
            evidence_bundle=evidence_bundle,
        )

        try:
            raw_output = self._invoke_ollama(prompt)
            parsed_payload = self._parse_model_payload(raw_output, evidence_bundle)
        except Exception as exc:
            self._model_fallback_used = True
            model_limitations.append(f"Model generation fallback used: {exc}")
            parsed_payload = self._rule_based_response(
                query=query,
                response_style=response_style,
                route_payload=route_payload,
                evidence_bundle=evidence_bundle,
            )

        if response_style == "layman":
            parsed_payload["answer"] = self._build_default_layman_answer(
                query=query,
                intent=intent,
                evidence_bundle=evidence_bundle,
                safety_notes=safety_notes,
            )

        answer = _clean_text(parsed_payload.get("answer"))

        unsupported_claims_detected, grounding_findings = self._detect_unsupported_claims(
            answer,
            query=query,
            route_payload=route_payload,
            evidence_bundle=evidence_bundle,
            strict_token_check=response_style != "layman",
        )

        limitations = _as_list_of_strings(evidence_bundle.get("limitations"))
        limitations.extend(_as_list_of_strings(parsed_payload.get("limitations")))
        limitations.extend(model_limitations)
        limitations.extend(grounding_findings)
        limitations = sorted(set([_clean_text(item) for item in limitations if _clean_text(item)]))

        if unsupported_claims_detected:
            limitations.append("Unsupported or unsafe claims were detected; block user display.")

        used_facts = [
            item
            for item in _as_list_of_strings(parsed_payload.get("used_facts"))
            if item in list(evidence_bundle.get("key_facts", []) or [])
        ]
        used_causal_paths = [
            item
            for item in _as_list_of_strings(parsed_payload.get("used_causal_paths"))
            if item in list(evidence_bundle.get("causal_paths", []) or [])
        ]

        used_evidence: List[Dict[str, Any]] = []
        for item in list(parsed_payload.get("used_evidence", []) or []):
            if not isinstance(item, Mapping):
                continue
            used_evidence.append(
                {
                    "object_id": _clean_text(item.get("object_id")),
                    "collection": _clean_text(item.get("collection")),
                    "title": _clean_text(item.get("title")),
                    "content_excerpt": _clean_text(item.get("content_excerpt")),
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                    "source_dataset": _clean_text(item.get("source_dataset")),
                    "source_file": _clean_text(item.get("source_file")),
                }
            )

        if not used_facts:
            used_facts = list(evidence_bundle.get("key_facts", []) or [])[:3]
        if not used_causal_paths:
            used_causal_paths = list(evidence_bundle.get("causal_paths", []) or [])[:2]
        if not used_evidence:
            source_evidence = list(evidence_bundle.get("retrieved_evidence", []) or [])[:3]
            for item in source_evidence:
                used_evidence.append(
                    {
                        "object_id": _clean_text(item.get("object_id")),
                        "collection": _clean_text(item.get("collection")),
                        "title": _clean_text(item.get("title")),
                        "content_excerpt": _clean_text(item.get("content_excerpt")),
                        "score": item.get("score"),
                        "distance": item.get("distance"),
                        "source_dataset": _clean_text(item.get("source_dataset")),
                        "source_file": _clean_text(item.get("source_file")),
                    }
                )

        confidence_score = round(_safe_float(evidence_bundle.get("confidence_score"), 0.0), 6)

        safe_for_user_display = bool(
            answer
            and not unsupported_claims_detected
            and _json_safe({"answer": answer, "safety_notes": safety_notes})
        )

        response = {
            "query": query,
            "answer": answer,
            "response_style": response_style,
            "used_facts": used_facts,
            "used_causal_paths": used_causal_paths,
            "used_evidence": used_evidence,
            "simulation_summary": evidence_bundle.get("simulation_summary"),
            "toxicity_summary": evidence_bundle.get("toxicity_summary"),
            "limitations": sorted(set(limitations)),
            "safety_notes": safety_notes,
            "confidence_score": confidence_score,
            "unsupported_claims_detected": bool(unsupported_claims_detected),
            "safe_for_user_display": safe_for_user_display,
        }

        if not response["safe_for_user_display"]:
            response["answer"] = (
                "I could not produce a safely grounded answer from the current evidence bundle. "
                "Please review limitations and evidence details."
            )

        return response

    def synthesize_from_query(
        self,
        query: str,
        *,
        enable_router_llm_fallback: bool = False,
    ) -> Dict[str, Any]:
        orchestration = orchestrate_query(
            query,
            enable_llm_fallback=bool(enable_router_llm_fallback),
        )
        return self.synthesize_response(orchestration)


def synthesize_response(
    orchestrator_payload: Mapping[str, Any],
    *,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    synthesizer = ResponseSynthesizer(model=model, timeout_seconds=timeout_seconds)
    return synthesizer.synthesize_response(orchestrator_payload)


def synthesize_query(
    query: str,
    *,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = 30,
    enable_router_llm_fallback: bool = False,
) -> Dict[str, Any]:
    synthesizer = ResponseSynthesizer(model=model, timeout_seconds=timeout_seconds)
    return synthesizer.synthesize_from_query(
        query,
        enable_router_llm_fallback=enable_router_llm_fallback,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08C grounded response synthesizer")
    parser.add_argument("--query", type=str, default="", help="User query text")
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help="Local Ollama model name.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for local Ollama generation.",
    )
    parser.add_argument(
        "--enable-router-llm-fallback",
        action="store_true",
        help="Enable optional low-confidence router fallback through Ollama in orchestrator step.",
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

    payload = synthesize_query(
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
