"""Phase 08B hybrid orchestrator.

This module executes routing + evidence collection only.
It does not synthesize final user-facing explanations.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    GraphDatabase = None

try:
    from reasoning.query_router import LOW_CONFIDENCE_THRESHOLD, route_query, validate_route_result_schema
    from simulation.pbpk.pbpk_master_simulator import (
        beverage_modifiers_path,
        parameter_library_path,
        population_modifiers_path,
        repo_root,
        run_simulation,
    )
    from utils.config import get_data_root, get_neo4j_config, get_vector_backend
    from vectorstores.pinecone_store import PineconeVectorStore
except ModuleNotFoundError:  # pragma: no cover - direct script execution path fix
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from reasoning.query_router import LOW_CONFIDENCE_THRESHOLD, route_query, validate_route_result_schema
    from simulation.pbpk.pbpk_master_simulator import (
        beverage_modifiers_path,
        parameter_library_path,
        population_modifiers_path,
        repo_root,
        run_simulation,
    )
    from utils.config import get_data_root, get_neo4j_config, get_vector_backend
    from vectorstores.pinecone_store import PineconeVectorStore

LOGGER = logging.getLogger("hybrid_orchestrator")

TOP_K = 8
EPS = 1e-12

SEMANTIC_RETRIEVAL_MODULE = "semantic_retrieval"
MODULE_KEYS: Tuple[str, ...] = ("pbpk", "neo4j", SEMANTIC_RETRIEVAL_MODULE, "toxicity")

SAFE_PBPK_DEFAULTS: Mapping[str, Any] = {
    "sex": "male",
    "weight": 75.0,
    "fed_state": "fed",
    "age": 30,
    "body_fat_percent": None,
    "liver_status": "healthy",
}

DEFAULT_HEIGHT_BY_SEX: Mapping[str, float] = {
    "male": 178.0,
    "female": 165.0,
}

DEFAULT_ABV_BY_BEVERAGE: Mapping[str, float] = {
    "beer": 5.0,
    "whisky": 40.0,
    "vodka": 40.0,
    "wine": 12.0,
    "rum": 40.0,
    "gin": 40.0,
    "tequila": 40.0,
    "cider": 5.0,
    "lager": 5.0,
    "ale": 6.0,
    "champagne": 12.0,
    "brandy": 40.0,
    "bourbon": 40.0,
    "mead": 12.0,
    "cocktail": 18.0,
}

DEFAULT_VOLUME_ML_BY_BEVERAGE: Mapping[str, float] = {
    "beer": 355.0,
    "whisky": 45.0,
    "vodka": 45.0,
    "wine": 150.0,
    "rum": 45.0,
    "gin": 45.0,
    "tequila": 45.0,
    "cider": 355.0,
    "lager": 355.0,
    "ale": 355.0,
    "champagne": 150.0,
    "brandy": 45.0,
    "bourbon": 45.0,
    "mead": 150.0,
    "cocktail": 150.0,
}

BEVERAGE_ALIASES: Mapping[str, str] = {
    "beer": "beer",
    "whisky": "whisky",
    "whiskey": "whisky",
    "vodka": "vodka",
    "wine": "wine",
    "rum": "rum",
    "gin": "gin",
    "tequila": "tequila",
    "cider": "cider",
    "lager": "lager",
    "ale": "ale",
    "champagne": "champagne",
    "brandy": "brandy",
    "bourbon": "bourbon",
    "mead": "mead",
    "cocktail": "cocktail",
}

SEMANTIC_COLLECTIONS: Tuple[str, ...] = (
    "BeverageKnowledge",
    "CompoundKnowledge",
    "MetabolismKnowledge",
    "PBPKKnowledge",
    "ToxicityKnowledge",
    "PopulationKnowledge",
    "ScientificEvidence",
)

SEMANTIC_EMBEDDED_PARQUET: Mapping[str, str] = {
    "BeverageKnowledge": "beverage_embeddings.parquet",
    "CompoundKnowledge": "compound_embeddings.parquet",
    "MetabolismKnowledge": "metabolism_embeddings.parquet",
    "PBPKKnowledge": "pbpk_embeddings.parquet",
    "ToxicityKnowledge": "toxicity_embeddings.parquet",
    "PopulationKnowledge": "population_embeddings.parquet",
    "ScientificEvidence": "scientific_evidence_embeddings.parquet",
}

INTENT_COLLECTION_SCOPE: Mapping[str, Tuple[str, ...]] = {
    "scientific_evidence": ("ScientificEvidence",),
    "retrieval_only": (
        "BeverageKnowledge",
        "CompoundKnowledge",
        "MetabolismKnowledge",
        "PBPKKnowledge",
        "ToxicityKnowledge",
        "PopulationKnowledge",
    ),
    "toxicity_risk": ("ToxicityKnowledge", "CompoundKnowledge", "ScientificEvidence"),
    "mechanistic_explanation": ("CompoundKnowledge", "MetabolismKnowledge", "ScientificEvidence"),
    "comparison": ("BeverageKnowledge", "PBPKKnowledge", "ScientificEvidence"),
}

TOXICITY_SIGNAL_TERMS: Tuple[str, ...] = (
    "headache",
    "hangover",
    "nausea",
    "migraine",
    "stomach",
    "histamine",
    "sulfite",
    "sulfites",
    "tyramine",
    "acetaldehyde",
    "congener",
    "congeners",
)

HIGH_PRIORITY_VECTOR_QUERY_TERMS: Set[str] = {
    "acetaldehyde",
    "congener",
    "congeners",
    "hangover",
    "headache",
    "headaches",
    "histamine",
    "migraine",
    "migraines",
    "polyphenol",
    "polyphenols",
    "sulfide",
    "sulfides",
    "sulfite",
    "sulfites",
    "sulphite",
    "sulphites",
    "tyramine",
}

GENERIC_VECTOR_QUERY_TERMS: Set[str] = {
    "alcohol",
    "and",
    "evidence",
    "research",
    "show",
    "study",
    "studies",
    "the",
}

PERSONALIZATION_MARKERS: Tuple[str, ...] = (
    "for my body",
    "my body",
    "my physiology",
    "personalized",
    "my profile",
    "for me personally",
)

WORD_NUMBERS: Mapping[str, float] = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
}


@dataclass(frozen=True)
class ParsedQueryInputs:
    beverages: List[str]
    drink_amount_ml: Optional[float]
    declared_abv_percent: Optional[float]
    body_weight_kg: Optional[float]
    sex: Optional[str]
    age_years: Optional[int]
    fed_state: Optional[str]
    body_fat_percent: Optional[float]
    liver_status: Optional[str]
    time_since_drinking_h: Optional[float]
    personalized_request: bool


@dataclass(frozen=True)
class EmbeddedVectorRow:
    object_id: str
    chunk_id: str
    collection: str
    search_text: str
    token_set: Set[str]
    vector: np.ndarray


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


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(text)))


def _parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return default
    try:
        number = float(text)
    except Exception:
        return default
    return number


def _extract_beverages(query: str) -> List[str]:
    normalized = _normalize_text(query)
    hits: List[str] = []
    for raw, canonical in BEVERAGE_ALIASES.items():
        if re.search(rf"\b{re.escape(raw)}\b", normalized):
            hits.append(canonical)
    ordered: List[str] = []
    seen: Set[str] = set()
    for beverage in hits:
        if beverage in seen:
            continue
        seen.add(beverage)
        ordered.append(beverage)
    return ordered


def _extract_drink_amount_ml(query: str) -> Optional[float]:
    normalized = _normalize_text(query)
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|oz|ounces?|shots?|glasses?|beers?|drinks?)\b",
        normalized,
    )
    if match is None:
        return None

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
    if unit in {"drink", "drinks"}:
        return round(amount * 45.0, 6)

    word_amount = re.search(r"\b(one|two|three|four|five)\s+(glass|glasses|beer|beers|shot|shots)\b", normalized)
    if word_amount is not None:
        number = WORD_NUMBERS.get(word_amount.group(1), 0.0)
        word_unit = word_amount.group(2)
        if word_unit in {"glass", "glasses"}:
            return round(number * 150.0, 6)
        if word_unit in {"beer", "beers"}:
            return round(number * 355.0, 6)
        if word_unit in {"shot", "shots"}:
            return round(number * 44.0, 6)
    return None


def _extract_abv_percent(query: str) -> Optional[float]:
    normalized = _normalize_text(query)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:abv)?\b", normalized)
    if match is None:
        return None
    value = float(match.group(1))
    if value <= 0 or value > 100:
        return None
    return round(value, 6)


def _extract_weight_kg(query: str) -> Optional[float]:
    normalized = _normalize_text(query)
    match = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilogram|kilograms)\b", normalized)
    if match is None:
        match = re.search(r"\bweigh\s*(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilogram|kilograms)\b", normalized)
        if match is None:
            return None
    value = float(match.group(1))
    return value if value > 0 else None


def _extract_sex(query: str) -> Optional[str]:
    normalized = _normalize_text(query)
    if re.search(r"\b(male|man)\b", normalized):
        return "male"
    if re.search(r"\b(female|woman)\b", normalized):
        return "female"
    return None


def _extract_age_years(query: str) -> Optional[int]:
    normalized = _normalize_text(query)
    match = re.search(r"\b(\d{1,3})\s*(years?\s*old|yo|y/o)\b", normalized)
    if match is None:
        match = re.search(r"\bage\s*(\d{1,3})\b", normalized)
    if match is None:
        return None
    value = int(match.group(1))
    if value <= 0:
        return None
    return value


def _extract_fed_state(query: str) -> Optional[str]:
    normalized = _normalize_text(query)
    if re.search(r"\b(fasted|empty\s+stomach|without\s+food)\b", normalized):
        return "fasted"
    if re.search(r"\b(fed|ate|eaten|with\s+food|after\s+meal)\b", normalized):
        return "fed"
    return None


def _extract_body_fat_percent(query: str) -> Optional[float]:
    normalized = _normalize_text(query)
    match = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:body\s*fat|fat)\b", normalized)
    if match is None:
        return None
    value = float(match.group(1))
    if value < 0 or value > 80:
        return None
    return value


def _extract_liver_status(query: str) -> Optional[str]:
    normalized = _normalize_text(query)
    if re.search(r"\b(cirrhosis|liver\s*impairment|impaired\s*liver|hepatic\s*impairment)\b", normalized):
        return "liver_impairment"
    if re.search(r"\b(healthy\s*liver|normal\s*liver)\b", normalized):
        return "healthy"
    return None


def _extract_time_since_h(query: str) -> Optional[float]:
    normalized = _normalize_text(query)
    match = re.search(r"\b(?:in|over)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?)\b", normalized)
    if match is not None:
        return round(float(match.group(1)), 6)

    match = re.search(r"\b(?:in|over)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?)\b", normalized)
    if match is not None:
        return round(float(match.group(1)) / 60.0, 6)

    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\s*(ago|since)\b", normalized)
    if match is None:
        if re.search(r"\bjust\s+drank\b", normalized):
            return 0.0
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if value < 0:
        return None
    if unit.startswith("min"):
        return round(value / 60.0, 6)
    return round(value, 6)


def _is_personalized_request(query: str) -> bool:
    normalized = _normalize_text(query)
    if any(marker in normalized for marker in PERSONALIZATION_MARKERS):
        return True
    if re.search(r"\b(my\s+weight|my\s+age|my\s+sex|my\s+body)\b", normalized):
        return True
    if re.search(r"\b(i\s+am|i'm)\b", normalized):
        return True
    return False


def parse_query_inputs(query: str) -> ParsedQueryInputs:
    return ParsedQueryInputs(
        beverages=_extract_beverages(query),
        drink_amount_ml=_extract_drink_amount_ml(query),
        declared_abv_percent=_extract_abv_percent(query),
        body_weight_kg=_extract_weight_kg(query),
        sex=_extract_sex(query),
        age_years=_extract_age_years(query),
        fed_state=_extract_fed_state(query),
        body_fat_percent=_extract_body_fat_percent(query),
        liver_status=_extract_liver_status(query),
        time_since_drinking_h=_extract_time_since_h(query),
        personalized_request=_is_personalized_request(query),
    )


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_json_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "model_dump"):
        try:
            dumped = raw.model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    if hasattr(raw, "to_dict"):
        try:
            dumped = raw.to_dict()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    return {}


def _safe_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _parse_json_blob(raw: Any) -> Dict[str, Any]:
    text = _clean_text(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_source_fields(*sources: Mapping[str, Any]) -> Tuple[str, str]:
    for source in sources:
        source_dataset = _clean_text(source.get("source_dataset"))
        source_file = _clean_text(source.get("source_file"))
        if source_dataset or source_file:
            return source_dataset, source_file
    return "", ""


def _score_from_metadata(metadata: Any) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    raw_score = _to_float(getattr(metadata, "score", None)) if metadata is not None else None
    distance = _to_float(getattr(metadata, "distance", None)) if metadata is not None else None
    certainty = _to_float(getattr(metadata, "certainty", None)) if metadata is not None else None

    if raw_score is not None:
        score = raw_score
    elif certainty is not None:
        score = certainty
    elif distance is not None:
        score = 1.0 / (1.0 + max(distance, 0.0))
    else:
        score = 0.0
    return raw_score, distance, certainty, float(score)


def _parse_vector_value(value: Any) -> List[float]:
    data = value
    if isinstance(data, str):
        text = _clean_text(data)
        if not text:
            return []
        try:
            data = json.loads(text)
        except Exception:
            return []
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, (list, tuple)):
        return []

    vector: List[float] = []
    for item in data:
        try:
            number = float(item)
        except Exception:
            return []
        if np.isnan(number):
            return []
        vector.append(number)
    return vector


def _expand_query_terms(terms: Set[str]) -> Set[str]:
    expanded = set(terms)
    for term in list(terms):
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    if "sulfite" in expanded or "sulfites" in expanded:
        expanded.update({"sulphite", "sulphites", "sulfide", "sulfides"})
    return expanded


def _query_vector_term_weight(term: str) -> float:
    if not term:
        return 0.0
    if " " in term:
        return 3.0
    if term in HIGH_PRIORITY_VECTOR_QUERY_TERMS:
        return 8.0
    if term in GENERIC_VECTOR_QUERY_TERMS:
        return 0.25
    if len(term) <= 2:
        return 0.0
    return 1.0


def _collection_scope(intent: str) -> List[str]:
    if intent in INTENT_COLLECTION_SCOPE:
        return list(INTENT_COLLECTION_SCOPE[intent])
    return list(SEMANTIC_COLLECTIONS)


def _rewrite_semantic_query(intent: str, query: str) -> str:
    q = _clean_text(query)
    q_norm = _normalize_text(query)

    if intent == "comparison" and "whisky" in q_norm and "beer" in q_norm and "non-alcoholic" not in q_norm:
        return f"{q} ethanol concentration absorption impairment comparison"

    if intent == "toxicity_risk" and re.search(r"\bwine\b", q_norm) and re.search(r"\bheadache", q_norm):
        return f"{q} sulfites histamine tyramine congeners polyphenols headache risk"

    if intent == "scientific_evidence" and "sulfite" in q_norm:
        return f"{q} sulfites headache alcohol scientific evidence"

    return q


def _focus_tokens_for_retrieval(intent: str, query: str) -> Set[str]:
    tokens = _tokenize(query)
    if intent == "comparison":
        tokens.update({"whisky", "whiskey", "beer", "ethanol", "absorption", "abv", "harder"})
    elif intent == "toxicity_risk":
        tokens.update({"headache", "hangover", "sulfites", "histamine", "tyramine", "congeners", "polyphenols"})
    elif intent == "scientific_evidence":
        tokens.update({"research", "study", "studies", "evidence", "sulfites"})
    elif intent == "mechanistic_explanation":
        tokens.update({"whisky", "beer", "absorption", "ethanol", "metabolism", "harder"})
    return set(token for token in tokens if token)


def _retrieval_relevance_score(
    *,
    intent: str,
    query: str,
    hit: Mapping[str, Any],
    focus_tokens: Set[str],
) -> float:
    title = _normalize_text(hit.get("title"))
    excerpt = _normalize_text(hit.get("content_excerpt"))
    text = f"{title} {excerpt}".strip()
    if not text:
        return 0.0

    doc_tokens = _tokenize(text)
    overlap = len(doc_tokens & focus_tokens)
    base = float(overlap) / float(max(len(focus_tokens), 1))

    if intent == "comparison":
        query_norm = _normalize_text(query)
        if "non-alcoholic" not in query_norm and ("non alcoholic" in text or "non-alcoholic" in text):
            base -= 0.35
        if "whisky" in query_norm and "beer" in query_norm:
            if ("whisky" in doc_tokens or "whiskey" in doc_tokens) and "beer" in doc_tokens:
                base += 0.20
            elif ("whisky" in doc_tokens or "whiskey" in doc_tokens or "beer" in doc_tokens):
                base += 0.08

    if intent == "toxicity_risk":
        preferred = {"sulfites", "histamine", "tyramine", "congeners", "polyphenols"}
        if doc_tokens & preferred:
            base += 0.20
        if "headache" in doc_tokens:
            base += 0.12

    if intent == "scientific_evidence" and "sulfite" in _normalize_text(query):
        if "sulfite" in text or "sulfites" in text:
            base += 0.25

    return round(base, 6)


def _filter_relevant_hits(
    *,
    intent: str,
    query: str,
    hits: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    focus_tokens = _focus_tokens_for_retrieval(intent, query)
    if not hits:
        return [], True

    scored: List[Tuple[float, float, Dict[str, Any]]] = []
    for item in hits:
        hit = dict(item)
        relevance = _retrieval_relevance_score(intent=intent, query=query, hit=hit, focus_tokens=focus_tokens)
        backend_score = _to_float(hit.get("score")) or 0.0
        scored.append((relevance, backend_score, hit))

    scored.sort(key=lambda row: (-row[0], -row[1], _clean_text(row[2].get("collection")), _clean_text(row[2].get("object_id"))))

    filtered: List[Dict[str, Any]] = []
    for relevance, _, hit in scored:
        if relevance >= 0.03:
            filtered.append(hit)
        if len(filtered) >= top_k:
            break

    low_relevance = False
    if not filtered:
        low_relevance = True
        filtered = [row[2] for row in scored[: min(3, len(scored))]]

    return filtered, low_relevance


def _json_safe(payload: Mapping[str, Any]) -> bool:
    try:
        json.dumps(payload, sort_keys=True)
        return True
    except Exception:
        return False


class HybridOrchestrator:
    """Query-router-driven module orchestrator for evidence packaging."""

    def __init__(
        self,
        *,
        enable_llm_fallback: bool = False,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.enable_llm_fallback = bool(enable_llm_fallback)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self._pbpk_sources: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None
        self._embedded_cache: Dict[str, pd.DataFrame] = {}
        self._embedded_vector_cache: Dict[str, List[EmbeddedVectorRow]] = {}
        self._query_vector_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[List[float], Dict[str, Any]]] = {}

    def _load_pbpk_sources(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._pbpk_sources is None:
            root = repo_root()
            library_df = pd.read_csv(parameter_library_path(root), dtype=str, keep_default_na=False)
            population_df = pd.read_csv(population_modifiers_path(root), dtype=str, keep_default_na=False)
            beverage_df = pd.read_csv(beverage_modifiers_path(root), dtype=str, keep_default_na=False)
            self._pbpk_sources = (library_df, population_df, beverage_df)
        return self._pbpk_sources

    def _get_default_abv(self, beverage: str) -> float:
        return float(DEFAULT_ABV_BY_BEVERAGE.get(beverage, 12.0))

    def _get_default_volume_ml(self, beverage: str) -> float:
        return float(DEFAULT_VOLUME_ML_BY_BEVERAGE.get(beverage, 150.0))

    def _execute_pbpk(
        self,
        query: str,
        route: Mapping[str, Any],
        parsed_inputs: ParsedQueryInputs,
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        route_intent = _clean_text(route.get("intent"))
        personalized_mode = bool(parsed_inputs.personalized_request or route_intent == "personalized_physiology")

        missing_inputs: List[str] = []
        if not parsed_inputs.beverages:
            missing_inputs.append("beverages")

        if route_intent != "comparison" and parsed_inputs.drink_amount_ml is None:
            missing_inputs.append("drink_amount")

        if personalized_mode:
            if parsed_inputs.body_weight_kg is None:
                missing_inputs.append("body_weight")
            if parsed_inputs.sex is None:
                missing_inputs.append("sex")
            if parsed_inputs.fed_state is None:
                missing_inputs.append("fed_state")

        if missing_inputs:
            missing_sorted = sorted(set(missing_inputs))
            limitations = [
                "PBPK simulation skipped: missing critical PBPK inputs for this query.",
                f"Missing PBPK inputs: {', '.join(missing_sorted)}.",
            ]
            return (
                {
                    "status": "skipped_missing_inputs",
                    "personalized_mode": personalized_mode,
                    "missing_inputs": missing_sorted,
                    "safe_defaults": dict(SAFE_PBPK_DEFAULTS),
                },
                missing_sorted,
                limitations,
            )

        try:
            library_df, population_df, beverage_df = self._load_pbpk_sources()
        except Exception as exc:
            limitations = [f"PBPK source loading failed: {exc}"]
            return (
                {
                    "status": "error",
                    "error": str(exc),
                    "personalized_mode": personalized_mode,
                    "safe_defaults": dict(SAFE_PBPK_DEFAULTS),
                },
                [],
                limitations,
            )

        sex = parsed_inputs.sex or _clean_text(SAFE_PBPK_DEFAULTS["sex"])
        weight = parsed_inputs.body_weight_kg if parsed_inputs.body_weight_kg is not None else float(SAFE_PBPK_DEFAULTS["weight"])
        age = parsed_inputs.age_years if parsed_inputs.age_years is not None else int(SAFE_PBPK_DEFAULTS["age"])
        fed_state = parsed_inputs.fed_state or _clean_text(SAFE_PBPK_DEFAULTS["fed_state"])
        liver_status = parsed_inputs.liver_status or _clean_text(SAFE_PBPK_DEFAULTS["liver_status"])

        defaults_applied: List[str] = []
        if parsed_inputs.sex is None:
            defaults_applied.append("sex")
        if parsed_inputs.body_weight_kg is None:
            defaults_applied.append("weight")
        if parsed_inputs.age_years is None:
            defaults_applied.append("age")
        if parsed_inputs.fed_state is None:
            defaults_applied.append("fed_state")
        if parsed_inputs.liver_status is None:
            defaults_applied.append("liver_status")

        body_fat_inferred: Optional[float] = parsed_inputs.body_fat_percent
        if body_fat_inferred is None:
            defaults_applied.append("body_fat_percent")
            body_fat_for_pbpk = 20.0 if sex == "male" else 28.0
        else:
            body_fat_for_pbpk = body_fat_inferred

        height_cm = DEFAULT_HEIGHT_BY_SEX.get(sex, 178.0)
        if sex not in DEFAULT_HEIGHT_BY_SEX:
            defaults_applied.append("height")

        if route_intent == "comparison":
            beverages = parsed_inputs.beverages[:2]
        else:
            beverages = parsed_inputs.beverages[:1]

        simulations: List[Dict[str, Any]] = []
        limitations: List[str] = []
        if personalized_mode and parsed_inputs.age_years is None:
            limitations.append("Assumed adult age because age was not provided.")

        for beverage in beverages:
            volume_ml = (
                parsed_inputs.drink_amount_ml
                if parsed_inputs.drink_amount_ml is not None
                else self._get_default_volume_ml(beverage)
            )
            if parsed_inputs.drink_amount_ml is None:
                limitations.append(
                    f"PBPK used default drink volume for '{beverage}': {self._get_default_volume_ml(beverage)} ml."
                )

            abv = parsed_inputs.declared_abv_percent if parsed_inputs.declared_abv_percent is not None else self._get_default_abv(beverage)
            if parsed_inputs.declared_abv_percent is None:
                limitations.append(
                    f"PBPK used default ABV for '{beverage}': {self._get_default_abv(beverage)}%."
                )

            user_payload: Dict[str, Any] = {
                "sex": sex,
                "weight": float(weight),
                "height": float(height_cm),
                "age": int(age),
                "body_fat_percent": float(body_fat_for_pbpk),
                "fed_or_fasted": fed_state,
                "liver_status": liver_status,
            }
            drink_payload: Dict[str, Any] = {
                "beverage": beverage,
                "volume_ml": float(volume_ml),
                "abv": float(abv),
                "serving_time": 0.0,
            }

            result = run_simulation(
                user_payload=user_payload,
                drink_payload=drink_payload,
                library_df=library_df,
                population_df=population_df,
                beverage_df=beverage_df,
            )
            summary = result["summary"]
            simulations.append(
                {
                    "beverage": beverage,
                    "volume_ml": round(float(volume_ml), 6),
                    "abv_percent": round(float(abv), 6),
                    "peak_bac_percent": _safe_round(_to_float(summary.get("peak_bac_percent"))),
                    "time_to_peak_h": _safe_round(_to_float(summary.get("time_to_peak_h"))),
                    "time_to_sober_h": _safe_round(_to_float(summary.get("time_to_sober_h"))),
                    "ethanol_auc_mg_h_l": _safe_round(
                        _to_float(summary.get("compound_burden", {}).get("ethanol_auc_mg_h_l"))
                    ),
                    "acetaldehyde_auc_mg_h_l": _safe_round(
                        _to_float(summary.get("compound_burden", {}).get("acetaldehyde_auc_mg_h_l"))
                    ),
                }
            )

        if SAFE_PBPK_DEFAULTS["body_fat_percent"] is None and "body_fat_percent" in defaults_applied:
            limitations.append("PBPK internal fallback used for body_fat_percent to satisfy simulator input contract.")

        return (
            {
                "status": "success",
                "personalized_mode": personalized_mode,
                "safe_defaults": dict(SAFE_PBPK_DEFAULTS),
                "defaults_applied": sorted(set(defaults_applied)),
                "simulations": simulations,
            },
            [],
            sorted(set(limitations)),
        )

    def _neo4j_template_queries(self, intent: str) -> List[str]:
        if intent == "mechanistic_explanation":
            return ["A"]
        if intent == "toxicity_risk":
            return ["B", "D"]
        if intent == "personalized_physiology":
            return ["C", "D"]
        if intent == "comparison":
            return ["A", "B", "C"]
        return ["A", "C"]

    def _execute_neo4j(
        self,
        query: str,
        route: Mapping[str, Any],
        parsed_inputs: ParsedQueryInputs,
    ) -> Tuple[Dict[str, Any], List[str]]:
        if GraphDatabase is None:
            return (
                {
                    "status": "unavailable",
                    "error": "neo4j driver is not installed.",
                    "query_templates_used": [],
                    "path_count": 0,
                    "paths": [],
                    "node_names": [],
                    "relationship_types": [],
                },
                ["Neo4j module unavailable: neo4j driver is not installed."],
            )

        try:
            config = get_neo4j_config()
        except Exception as exc:
            return (
                {
                    "status": "unavailable",
                    "error": str(exc),
                    "query_templates_used": [],
                    "path_count": 0,
                    "paths": [],
                    "node_names": [],
                    "relationship_types": [],
                },
                [f"Neo4j module unavailable: {exc}"],
            )

        beverages = parsed_inputs.beverages
        groups = sorted(
            set(
                item
                for item in [parsed_inputs.sex, parsed_inputs.fed_state, "general_population"]
                if _clean_text(item)
            )
        )
        conditions = sorted(
            set(
                token
                for token in [parsed_inputs.fed_state, "fasted", "fed", "food", "empty stomach"]
                if _clean_text(token)
            )
        )

        query_map: Mapping[str, str] = {
            "A": """
                MATCH (b:Beverage)-[r1:CONTAINS]->(c:Compound)-[r2:METABOLIZED_BY]->(e:Enzyme)
                WHERE size($beverages) = 0
                   OR any(token IN $beverages WHERE
                        toLower(coalesce(b.name,'')) CONTAINS token
                        OR toLower(coalesce(b.category,'')) CONTAINS token)
                RETURN b.name AS beverage,
                       c.name AS compound,
                       e.name AS enzyme,
                       type(r1) AS rel_1,
                       type(r2) AS rel_2,
                       coalesce(r2.confidence_score, r1.confidence_score, e.confidence_score, c.confidence_score, 0.0) AS confidence
                ORDER BY beverage, compound, enzyme
                LIMIT 40
            """,
            "B": """
                MATCH (b:Beverage)-[r1:CONTAINS]->(c:Compound)-[r2:CONTRIBUTES_TO]->(t:ToxicityRisk)
                WHERE size($beverages) = 0
                   OR any(token IN $beverages WHERE
                        toLower(coalesce(b.name,'')) CONTAINS token
                        OR toLower(coalesce(b.category,'')) CONTAINS token)
                RETURN b.name AS beverage,
                       c.name AS compound,
                       t.risk_type AS risk_type,
                       type(r1) AS rel_1,
                       type(r2) AS rel_2,
                       coalesce(t.confidence_score, r2.confidence_score, r1.confidence_score, 0.0) AS confidence
                ORDER BY beverage, compound, risk_type
                LIMIT 60
            """,
            "C": """
                MATCH (g:PopulationGroup)-[r1:MODIFIES]->(p:PBPKParameter)-[r2:AFFECTS]->(bc:BodyCompartment)
                WHERE size($groups) = 0 OR g.group_name IN $groups
                RETURN g.group_name AS group_name,
                       p.parameter_name AS parameter_name,
                       bc.name AS compartment,
                       type(r1) AS rel_1,
                       type(r2) AS rel_2,
                       coalesce(r1.confidence_score, 0.0) AS confidence
                ORDER BY group_name, parameter_name, compartment
                LIMIT 60
            """,
            "D": """
                MATCH (pc:PhysiologyCondition)-[r:INCREASES|DECREASES]->(p:PBPKParameter)
                WHERE size($conditions) = 0
                   OR any(token IN $conditions WHERE toLower(coalesce(pc.condition,'')) CONTAINS token)
                RETURN pc.condition AS condition,
                       p.parameter_name AS parameter_name,
                       type(r) AS effect,
                       coalesce(r.confidence_score, 0.0) AS confidence
                ORDER BY condition, parameter_name
                LIMIT 60
            """,
        }

        template_ids = self._neo4j_template_queries(_clean_text(route.get("intent")))
        params = {
            "beverages": beverages,
            "groups": groups,
            "conditions": conditions,
        }

        path_entries: List[Dict[str, Any]] = []
        node_names: Set[str] = set()
        rel_types: Set[str] = set()

        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))

        try:
            with driver.session(database=config["database"]) as session:
                for template_id in template_ids:
                    statement = query_map[template_id]
                    rows = list(session.run(statement, **params))
                    for row in rows:
                        data = dict(row)
                        if template_id == "A":
                            beverage = _clean_text(data.get("beverage"))
                            compound = _clean_text(data.get("compound"))
                            enzyme = _clean_text(data.get("enzyme"))
                            rel_1 = _clean_text(data.get("rel_1"))
                            rel_2 = _clean_text(data.get("rel_2"))
                            confidence = _to_float(data.get("confidence")) or 0.0
                            path_text = f"{beverage} -[{rel_1}]-> {compound} -[{rel_2}]-> {enzyme}"
                            nodes = [beverage, compound, enzyme]
                            rels = [rel_1, rel_2]
                            extra = {"beverage": beverage, "compound": compound, "enzyme": enzyme}
                        elif template_id == "B":
                            beverage = _clean_text(data.get("beverage"))
                            compound = _clean_text(data.get("compound"))
                            risk_type = _clean_text(data.get("risk_type"))
                            rel_1 = _clean_text(data.get("rel_1"))
                            rel_2 = _clean_text(data.get("rel_2"))
                            confidence = _to_float(data.get("confidence")) or 0.0
                            path_text = f"{beverage} -[{rel_1}]-> {compound} -[{rel_2}]-> {risk_type}"
                            nodes = [beverage, compound, risk_type]
                            rels = [rel_1, rel_2]
                            extra = {"beverage": beverage, "compound": compound, "risk_type": risk_type}
                        elif template_id == "C":
                            group_name = _clean_text(data.get("group_name"))
                            parameter_name = _clean_text(data.get("parameter_name"))
                            compartment = _clean_text(data.get("compartment"))
                            rel_1 = _clean_text(data.get("rel_1"))
                            rel_2 = _clean_text(data.get("rel_2"))
                            confidence = _to_float(data.get("confidence")) or 0.0
                            path_text = f"{group_name} -[{rel_1}]-> {parameter_name} -[{rel_2}]-> {compartment}"
                            nodes = [group_name, parameter_name, compartment]
                            rels = [rel_1, rel_2]
                            extra = {
                                "population_group": group_name,
                                "parameter_name": parameter_name,
                                "body_compartment": compartment,
                            }
                        else:
                            condition = _clean_text(data.get("condition"))
                            parameter_name = _clean_text(data.get("parameter_name"))
                            effect = _clean_text(data.get("effect"))
                            confidence = _to_float(data.get("confidence")) or 0.0
                            path_text = f"{condition} -[{effect}]-> {parameter_name}"
                            nodes = [condition, parameter_name]
                            rels = [effect]
                            extra = {"condition": condition, "parameter_name": parameter_name, "effect": effect}

                        path_entries.append(
                            {
                                "template": template_id,
                                "path": path_text,
                                "nodes": nodes,
                                "relationship_types": rels,
                                "confidence": _safe_round(confidence),
                                **extra,
                            }
                        )
                        node_names.update([item for item in nodes if item])
                        rel_types.update([item for item in rels if item])
        except Exception as exc:
            return (
                {
                    "status": "error",
                    "error": str(exc),
                    "query_templates_used": list(template_ids),
                    "path_count": 0,
                    "paths": [],
                    "node_names": [],
                    "relationship_types": [],
                },
                [f"Neo4j query execution failed: {exc}"],
            )
        finally:
            driver.close()

        ordered_paths = sorted(path_entries, key=lambda item: (item["template"], item["path"]))
        limitations: List[str] = []
        if not ordered_paths:
            limitations.append("Neo4j returned zero matching causal paths for this query.")

        return (
            {
                "status": "success",
                "query_templates_used": list(template_ids),
                "path_count": len(ordered_paths),
                "paths": ordered_paths,
                "node_names": sorted(node_names),
                "relationship_types": sorted(rel_types),
            },
            limitations,
        )

    def _embedded_corpus_path(self, collection: str) -> Path:
        data_root = get_data_root()
        return data_root / "processed" / "weaviate" / "embedded" / SEMANTIC_EMBEDDED_PARQUET[collection]

    def _load_embedded_collection_df(self, collection: str) -> pd.DataFrame:
        if collection in self._embedded_cache:
            return self._embedded_cache[collection]

        path = self._embedded_corpus_path(collection)
        if not path.exists():
            raise FileNotFoundError(f"Embedded corpus file not found: {path}")

        df = pd.read_parquet(path)
        columns = [
            col
            for col in [
                "object_id",
                "chunk_id",
                "collection",
                "title",
                "content",
                "metadata",
                "provenance",
                "source_dataset",
                "source_file",
            ]
            if col in df.columns
        ]
        compact = df[columns].copy()
        if "collection" not in compact.columns:
            compact["collection"] = collection
        compact = compact.sort_values(by=["object_id", "chunk_id"], kind="mergesort", na_position="last")
        self._embedded_cache[collection] = compact
        return compact

    def _load_embedded_vector_rows(self, collection: str) -> List[EmbeddedVectorRow]:
        if collection in self._embedded_vector_cache:
            return self._embedded_vector_cache[collection]

        path = self._embedded_corpus_path(collection)
        if not path.exists():
            raise FileNotFoundError(f"Embedded corpus file not found: {path}")

        df = pd.read_parquet(path)
        if "embedding" not in df.columns:
            raise ValueError(f"Embedded corpus file has no embedding column: {path}")

        rows: List[EmbeddedVectorRow] = []
        expected_dimension = 0
        for _, row in df.iterrows():
            vector_values = _parse_vector_value(row.get("embedding"))
            if not vector_values:
                continue
            vector = np.asarray(vector_values, dtype=float)
            if expected_dimension == 0:
                expected_dimension = int(vector.shape[0])
            elif int(vector.shape[0]) != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch in {collection}: expected {expected_dimension}, got {vector.shape[0]}"
                )

            title = _clean_text(row.get("title"))
            content = _clean_text(row.get("content"))
            metadata_raw = _clean_text(row.get("metadata"))
            provenance_raw = _clean_text(row.get("provenance"))
            search_text = " ".join([title, content, metadata_raw, provenance_raw]).strip()
            rows.append(
                EmbeddedVectorRow(
                    object_id=_clean_text(row.get("object_id")),
                    chunk_id=_clean_text(row.get("chunk_id")),
                    collection=_clean_text(row.get("collection")) or collection,
                    search_text=_normalize_text(search_text),
                    token_set=_tokenize(search_text),
                    vector=vector,
                )
            )

        if not rows:
            raise ValueError(f"No usable embedded vectors found in {path}")

        self._embedded_vector_cache[collection] = rows
        return rows

    def _available_vector_rows(self, collections: Sequence[str]) -> Tuple[List[EmbeddedVectorRow], int]:
        rows: List[EmbeddedVectorRow] = []
        for collection in collections:
            try:
                rows.extend(self._load_embedded_vector_rows(collection))
            except FileNotFoundError:
                continue

        if not rows and "ScientificEvidence" not in collections:
            try:
                rows.extend(self._load_embedded_vector_rows("ScientificEvidence"))
            except FileNotFoundError:
                pass

        if not rows:
            raise FileNotFoundError("No embedded vector corpus available for query-vector generation.")

        dimensions = {int(row.vector.shape[0]) for row in rows}
        if len(dimensions) != 1:
            raise ValueError(f"Embedded vector dimension mismatch across collections: {sorted(dimensions)}")
        dimension = int(next(iter(dimensions)))
        if dimension <= 0:
            raise ValueError("Embedded vector dimension must be positive.")
        return rows, dimension

    def _build_query_vector(
        self,
        search_query: str,
        collections: Sequence[str],
    ) -> Tuple[List[float], Dict[str, Any]]:
        cache_key = (_normalize_text(search_query), tuple(sorted(collections)))
        if cache_key in self._query_vector_cache:
            return self._query_vector_cache[cache_key]

        rows, dimension = self._available_vector_rows(collections)
        query_tokens = _expand_query_terms(_tokenize(search_query))
        query_phrase = _normalize_text(search_query)
        terms = {term for term in query_tokens if term}
        if query_phrase:
            terms.add(query_phrase)

        candidates: List[Tuple[float, str, str, str, np.ndarray]] = []
        for row in rows:
            overlap = 0.0
            for term in terms:
                weight = _query_vector_term_weight(term)
                if weight <= 0.0:
                    continue
                if " " in term:
                    if term in row.search_text:
                        overlap += weight
                elif term in row.token_set:
                    overlap += weight
            if overlap > 0:
                candidates.append((overlap, row.collection, row.chunk_id, row.object_id, row.vector))

        if candidates:
            ordered = sorted(candidates, key=lambda item: (-item[0], item[1], item[2], item[3]))
            max_score = float(ordered[0][0])
            if max_score >= 4.0:
                focused = [item for item in ordered if float(item[0]) >= max_score - EPS]
                selected = focused[:300]
            else:
                selected = ordered[:3000]
            weights = np.asarray([float(item[0]) for item in selected], dtype=float)
            matrix = np.vstack([item[4] for item in selected])
            vector = np.average(matrix, axis=0, weights=weights)
            source = "embedded_corpus_weighted_average"
            seed_rows = len(selected)
            max_overlap = float(selected[0][0])
        else:
            selected_rows = sorted(rows, key=lambda item: (item.collection, item.chunk_id, item.object_id))[:2000]
            matrix = np.vstack([item.vector for item in selected_rows])
            vector = np.mean(matrix, axis=0)
            source = "embedded_corpus_fallback_average"
            seed_rows = len(selected_rows)
            max_overlap = 0.0

        if int(vector.shape[0]) != dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {dimension}, got {vector.shape[0]}")

        norm = float(np.linalg.norm(vector))
        if norm > EPS:
            vector = vector / norm

        result = (
            [float(value) for value in vector.tolist()],
            {
                "query_vector_source": source,
                "query_vector_dimension": dimension,
                "query_vector_seed_rows": int(seed_rows),
                "query_vector_seed_overlap_max": round(float(max_overlap), 6),
                "embedding_model_reference": "nomic-ai/nomic-embed-text-v1",
            },
        )
        self._query_vector_cache[cache_key] = result
        return result

    def _embedded_fallback_search(
        self,
        query: str,
        collections: Sequence[str],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_tokens = _tokenize(query)
        query_token_count = max(len(query_tokens), 1)

        candidates: List[Dict[str, Any]] = []

        for collection in collections:
            try:
                df = self._load_embedded_collection_df(collection)
            except FileNotFoundError:
                continue
            for _, row in df.iterrows():
                object_id = _clean_text(row.get("object_id"))
                title = _clean_text(row.get("title"))
                content = _clean_text(row.get("content"))
                metadata_raw = row.get("metadata")
                provenance_raw = row.get("provenance")

                metadata = _parse_json_blob(metadata_raw)
                provenance = _parse_json_blob(provenance_raw)

                source_dataset, source_file = _extract_source_fields(
                    _coerce_json_dict(row),
                    metadata,
                    provenance,
                )

                search_text = " ".join([title, content, _clean_text(metadata_raw), _clean_text(provenance_raw)]).strip()
                doc_tokens = _tokenize(search_text)

                overlap = len(query_tokens & doc_tokens)
                lexical = float(overlap) / float(query_token_count)
                if query and _normalize_text(query) in _normalize_text(search_text):
                    lexical += 0.2
                score = lexical

                candidates.append(
                    {
                        "object_id": object_id,
                        "collection": _clean_text(row.get("collection")) or collection,
                        "title": title,
                        "content_excerpt": content[:320],
                        "score": _safe_round(score),
                        "distance": None,
                        "source_dataset": source_dataset,
                        "source_file": source_file,
                    }
                )

        ordered = sorted(
            candidates,
            key=lambda item: (
                -(item["score"] if item["score"] is not None else 0.0),
                item["collection"],
                item["object_id"],
                item["title"],
            ),
        )

        deduped: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for item in ordered:
            key = (item["collection"], item["object_id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= top_k:
                break

        return deduped

    def _execute_pinecone(
        self,
        query: str,
        route: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        intent = _clean_text(route.get("intent"))
        collections = _collection_scope(intent)
        search_query = _rewrite_semantic_query(intent, query)
        limitations: List[str] = []
        query_vector_details: Dict[str, Any] = {}

        try:
            query_vector, query_vector_details = self._build_query_vector(search_query, collections)
            store = PineconeVectorStore()
            try:
                hits = store.query_by_vector(query_vector, top_k=TOP_K, collections=collections)
            finally:
                store.close()

            if not hits:
                limitations.append("Pinecone returned zero hits; used embedded fallback retrieval.")
                try:
                    hits = self._embedded_fallback_search(search_query, collections, top_k=TOP_K)
                except Exception as embedded_exc:
                    limitations.append(f"Embedded fallback retrieval unavailable: {embedded_exc}")
                    hits = []
                backend = "embedded_fallback"
            else:
                backend = "pinecone"

            hits, low_relevance = _filter_relevant_hits(intent=intent, query=query, hits=hits, top_k=TOP_K)
            if low_relevance:
                limitations.append("Some supporting evidence was unavailable.")

            return (
                {
                    "status": "success",
                    "retrieval_backend": backend,
                    "top_k": TOP_K,
                    "collections_searched": collections,
                    "hit_count": len(hits),
                    "hits": hits,
                    **query_vector_details,
                    "query_vector_dimension": 768,
                    "embedding_model_reference": "nomic-ai/nomic-embed-text-v1",
                },
                limitations,
            )
        except Exception:
            limitations.append("Pinecone query failed; used embedded fallback retrieval.")
            try:
                fallback_hits = self._embedded_fallback_search(search_query, collections, top_k=TOP_K)
            except Exception as embedded_exc:
                limitations.append(f"Embedded fallback retrieval unavailable: {embedded_exc}")
                fallback_hits = []
            fallback_hits, low_relevance = _filter_relevant_hits(
                intent=intent,
                query=query,
                hits=fallback_hits,
                top_k=TOP_K,
            )
            if low_relevance:
                limitations.append("Some supporting evidence was unavailable.")
            return (
                {
                    "status": "success",
                    "retrieval_backend": "embedded_fallback",
                    "top_k": TOP_K,
                    "collections_searched": collections,
                    "hit_count": len(fallback_hits),
                    "hits": fallback_hits,
                    **query_vector_details,
                },
                limitations,
            )

    def _execute_semantic_retrieval(
        self,
        query: str,
        route: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        _ = get_vector_backend()
        return self._execute_pinecone(query=query, route=route)

    def _execute_toxicity(
        self,
        query: str,
        neo4j_result: Optional[Mapping[str, Any]],
        semantic_retrieval_result: Optional[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        if neo4j_result is None and semantic_retrieval_result is None:
            return (
                {
                    "status": "unavailable",
                    "risk_compounds": [],
                    "risk_types": [],
                    "symptom_modifiers": [],
                    "confidence": 0.0,
                },
                ["Toxicity module unavailable: missing Neo4j and semantic retrieval evidence inputs."],
            )

        risk_compounds: Set[str] = set()
        risk_types: Set[str] = set()
        symptom_modifiers: Set[str] = set()

        for term in TOXICITY_SIGNAL_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", _normalize_text(query)):
                symptom_modifiers.add(term)

        if neo4j_result and isinstance(neo4j_result.get("paths"), list):
            for item in neo4j_result["paths"]:
                compound = _clean_text(item.get("compound"))
                risk_type = _clean_text(item.get("risk_type"))
                if compound:
                    risk_compounds.add(compound)
                if risk_type:
                    risk_types.add(risk_type)

        if semantic_retrieval_result and isinstance(semantic_retrieval_result.get("hits"), list):
            for hit in semantic_retrieval_result["hits"]:
                title = _normalize_text(hit.get("title"))
                excerpt = _normalize_text(hit.get("content_excerpt"))
                collection = _clean_text(hit.get("collection"))

                if collection == "CompoundKnowledge" and title.endswith("compound profile"):
                    candidate = title.split("compound profile")[0].strip()
                    if candidate:
                        risk_compounds.add(candidate)

                for term in TOXICITY_SIGNAL_TERMS:
                    if term in excerpt or term in title:
                        symptom_modifiers.add(term)

        has_neo4j = bool(neo4j_result and neo4j_result.get("status") == "success")
        has_semantic_retrieval = bool(
            semantic_retrieval_result and semantic_retrieval_result.get("status") == "success"
        )
        confidence = (
            0.35 * (1.0 if has_neo4j else 0.0)
            + 0.30 * (1.0 if has_semantic_retrieval else 0.0)
            + 0.20 * min(len(risk_compounds) / 5.0, 1.0)
            + 0.15 * min(len(risk_types) / 3.0, 1.0)
        )

        return (
            {
                "status": "success",
                "risk_compounds": sorted(risk_compounds),
                "risk_types": sorted(risk_types),
                "symptom_modifiers": sorted(symptom_modifiers),
                "confidence": _safe_round(confidence),
            },
            [] if risk_compounds or risk_types else ["Toxicity module found no explicit risk compounds/types in evidence."],
        )

    def _compute_confidence_score(
        self,
        route: Mapping[str, Any],
        module_results: Mapping[str, Optional[Mapping[str, Any]]],
    ) -> float:
        route_conf = _to_float(route.get("confidence")) or 0.0
        required_modules = [item for item in route.get("required_modules", []) if item in MODULE_KEYS]
        required_count = max(len(required_modules), 1)

        success_count = 0
        for module in required_modules:
            result = module_results.get(module)
            if result and _clean_text(result.get("status")) == "success":
                success_count += 1

        semantic_retrieval_hits = 0
        if module_results.get(SEMANTIC_RETRIEVAL_MODULE):
            semantic_retrieval_hits = int(module_results[SEMANTIC_RETRIEVAL_MODULE].get("hit_count") or 0)

        neo4j_paths = 0
        if module_results.get("neo4j"):
            neo4j_paths = int(module_results["neo4j"].get("path_count") or 0)

        if "pbpk" in required_modules:
            pbpk_ok = bool(module_results.get("pbpk") and module_results["pbpk"].get("status") == "success")
            pbpk_component = 1.0 if pbpk_ok else 0.0
        else:
            pbpk_component = 1.0

        score = (
            0.45 * float(route_conf)
            + 0.20 * (float(success_count) / float(required_count))
            + 0.15 * min(float(semantic_retrieval_hits) / float(TOP_K), 1.0)
            + 0.10 * min(float(neo4j_paths) / 12.0, 1.0)
            + 0.10 * pbpk_component
        )
        score = max(0.0, min(1.0, score))
        return round(score, 6)

    def _build_evidence_bundle(
        self,
        query: str,
        route: Mapping[str, Any],
        module_results: Mapping[str, Optional[Mapping[str, Any]]],
        limitations: Sequence[str],
    ) -> Dict[str, Any]:
        key_facts: List[str] = []
        causal_paths: List[str] = []
        retrieved_evidence: List[Dict[str, Any]] = []

        pbpk_result = module_results.get("pbpk")
        if pbpk_result and pbpk_result.get("status") == "success":
            simulations = pbpk_result.get("simulations", [])
            if isinstance(simulations, list):
                for item in simulations[:3]:
                    beverage = _clean_text(item.get("beverage"))
                    peak = item.get("peak_bac_percent")
                    t_peak = item.get("time_to_peak_h")
                    key_facts.append(
                        f"PBPK simulation: {beverage} peak_bac_percent={peak}, time_to_peak_h={t_peak}."
                    )

        neo4j_result = module_results.get("neo4j")
        if neo4j_result and neo4j_result.get("status") == "success":
            paths = neo4j_result.get("paths", [])
            if isinstance(paths, list):
                for item in paths:
                    path_text = _clean_text(item.get("path"))
                    if path_text:
                        causal_paths.append(path_text)
                if paths:
                    key_facts.append(f"Neo4j returned {len(paths)} causal paths.")

        semantic_retrieval_result = module_results.get(SEMANTIC_RETRIEVAL_MODULE)
        if semantic_retrieval_result and semantic_retrieval_result.get("status") == "success":
            hits = semantic_retrieval_result.get("hits", [])
            if isinstance(hits, list):
                for item in hits:
                    retrieved_evidence.append(
                        {
                            "object_id": _clean_text(item.get("object_id")),
                            "chunk_id": _clean_text(item.get("chunk_id")),
                            "collection": _clean_text(item.get("collection")),
                            "title": _clean_text(item.get("title")),
                            "content_excerpt": _clean_text(item.get("content_excerpt")),
                            "content": _clean_text(item.get("content")),
                            "score": item.get("score"),
                            "distance": item.get("distance"),
                            "retrieval_backend": _clean_text(item.get("retrieval_backend"))
                            or _clean_text(semantic_retrieval_result.get("retrieval_backend")),
                            "source_dataset": _clean_text(item.get("source_dataset")),
                            "source_file": _clean_text(item.get("source_file")),
                        }
                    )
                if hits:
                    key_facts.append(
                        f"Evidence retrieval returned {len(hits)} objects across {len(set([_clean_text(i.get('collection')) for i in hits]))} collections."
                    )

        toxicity_result = module_results.get("toxicity")
        if toxicity_result and toxicity_result.get("status") == "success":
            compounds = toxicity_result.get("risk_compounds", [])
            risk_types = toxicity_result.get("risk_types", [])
            if compounds or risk_types:
                key_facts.append(
                    f"Toxicity evidence identified {len(compounds)} risk compounds and {len(risk_types)} risk types."
                )

        confidence = self._compute_confidence_score(route, module_results)

        simulation_summary: Optional[Dict[str, Any]]
        if pbpk_result and pbpk_result.get("status") == "success":
            simulation_summary = {
                "simulations": pbpk_result.get("simulations", []),
                "defaults_applied": pbpk_result.get("defaults_applied", []),
                "personalized_mode": bool(pbpk_result.get("personalized_mode")),
            }
        else:
            simulation_summary = None

        toxicity_summary: Optional[Dict[str, Any]]
        if toxicity_result and toxicity_result.get("status") == "success":
            toxicity_summary = {
                "risk_compounds": list(toxicity_result.get("risk_compounds", [])),
                "risk_types": list(toxicity_result.get("risk_types", [])),
                "symptom_modifiers": list(toxicity_result.get("symptom_modifiers", [])),
                "confidence": toxicity_result.get("confidence"),
            }
        else:
            toxicity_summary = None

        return {
            "key_facts": sorted(set(key_facts)),
            "causal_paths": sorted(set(causal_paths)),
            "retrieved_evidence": retrieved_evidence,
            "simulation_summary": simulation_summary,
            "toxicity_summary": toxicity_summary,
            "confidence_score": confidence,
            "limitations": sorted(set([_clean_text(item) for item in limitations if _clean_text(item)])),
        }

    def orchestrate(self, query: str) -> Dict[str, Any]:
        text = _clean_text(query)
        if not text:
            raise ValueError("Query must be a non-empty string.")

        route_result = route_query(
            text,
            enable_llm_fallback=self.enable_llm_fallback,
            low_confidence_threshold=self.low_confidence_threshold,
        )
        validate_route_result_schema(route_result)
        route_payload = route_result.to_dict()

        parsed_inputs = parse_query_inputs(text)
        module_results: Dict[str, Optional[Dict[str, Any]]] = {key: None for key in MODULE_KEYS}

        missing_inputs: List[str] = []
        limitations: List[str] = []

        for module in route_payload["required_modules"]:
            if module == "pbpk":
                pbpk_result, pbpk_missing, pbpk_limits = self._execute_pbpk(
                    query=text,
                    route=route_payload,
                    parsed_inputs=parsed_inputs,
                )
                module_results["pbpk"] = pbpk_result
                missing_inputs.extend(pbpk_missing)
                limitations.extend(pbpk_limits)

            elif module == "neo4j":
                neo4j_result, neo4j_limits = self._execute_neo4j(
                    query=text,
                    route=route_payload,
                    parsed_inputs=parsed_inputs,
                )
                module_results["neo4j"] = neo4j_result
                limitations.extend(neo4j_limits)

            elif module == SEMANTIC_RETRIEVAL_MODULE:
                semantic_retrieval_result, semantic_retrieval_limits = self._execute_semantic_retrieval(
                    query=text,
                    route=route_payload,
                )
                module_results[SEMANTIC_RETRIEVAL_MODULE] = semantic_retrieval_result
                limitations.extend(semantic_retrieval_limits)

            elif module == "toxicity":
                toxicity_result, toxicity_limits = self._execute_toxicity(
                    query=text,
                    neo4j_result=module_results.get("neo4j"),
                    semantic_retrieval_result=module_results.get(SEMANTIC_RETRIEVAL_MODULE),
                )
                module_results["toxicity"] = toxicity_result
                limitations.extend(toxicity_limits)

        missing_inputs_sorted = sorted(set([_clean_text(item) for item in missing_inputs if _clean_text(item)]))

        evidence_bundle = self._build_evidence_bundle(
            query=text,
            route=route_payload,
            module_results=module_results,
            limitations=limitations,
        )

        required_modules = list(route_payload.get("required_modules", []))
        required_success = True
        for module in required_modules:
            result = module_results.get(module)
            if result is None or _clean_text(result.get("status")) != "success":
                required_success = False
                break

        payload: Dict[str, Any] = {
            "query": text,
            "route": route_payload,
            "module_results": {
                "pbpk": module_results.get("pbpk"),
                "neo4j": module_results.get("neo4j"),
                SEMANTIC_RETRIEVAL_MODULE: module_results.get(SEMANTIC_RETRIEVAL_MODULE),
                "toxicity": module_results.get("toxicity"),
            },
            "evidence_bundle": evidence_bundle,
            "missing_inputs": missing_inputs_sorted,
            "safe_for_response_synthesis": False,
        }

        payload["safe_for_response_synthesis"] = bool(
            required_success
            and not missing_inputs_sorted
            and evidence_bundle["confidence_score"] >= 0.35
            and _json_safe(payload)
        )

        return payload


def orchestrate_query(
    query: str,
    *,
    enable_llm_fallback: bool = False,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    orchestrator = HybridOrchestrator(
        enable_llm_fallback=enable_llm_fallback,
        low_confidence_threshold=low_confidence_threshold,
    )
    return orchestrator.orchestrate(query)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 08B hybrid orchestrator")
    parser.add_argument("--query", type=str, default="", help="User query text")
    parser.add_argument(
        "--enable-llm-fallback",
        action="store_true",
        help="Enable low-confidence router fallback through local Ollama.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=LOW_CONFIDENCE_THRESHOLD,
        help="Router fallback threshold.",
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

    payload = orchestrate_query(
        query,
        enable_llm_fallback=bool(args.enable_llm_fallback),
        low_confidence_threshold=float(args.low_confidence_threshold),
    )

    if bool(args.compact):
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
