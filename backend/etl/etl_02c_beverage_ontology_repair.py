"""ETL step 02c: canonical beverage ontology repair.

Repairs canonical beverage reference quality issues using
`alcohol_abv_reference.csv` as the authoritative ontology source.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple

import pandas as pd

LOGGER = logging.getLogger("etl_02c_beverage_ontology_repair")

ENCODING = "utf-8"
UNKNOWN = "unknown"
CONFIDENCE_THRESHOLD = 0.8

CANONICAL_CATEGORIES: Tuple[str, ...] = (
    "beer",
    "wine",
    "whisky",
    "vodka",
    "rum",
    "gin",
    "tequila",
    "brandy",
    "liqueur",
    "cider",
    "cocktail",
    "sake",
    "mead",
    "hard_seltzer",
    "fortified_wine",
    "unknown",
)
CANONICAL_CATEGORY_SET: Set[str] = set(CANONICAL_CATEGORIES)

# Category-level guardrails provided in the task.
CATEGORY_ABV_RANGES: Mapping[str, Tuple[float, float]] = {
    "beer": (0.0, 15.0),
    "wine": (5.0, 22.0),
    "whisky": (30.0, 70.0),
    "vodka": (30.0, 70.0),
    "rum": (20.0, 80.0),
    "gin": (20.0, 80.0),
    "tequila": (20.0, 80.0),
    "brandy": (20.0, 80.0),
    "liqueur": (10.0, 60.0),
    "cocktail": (1.0, 40.0),
    "cider": (1.0, 12.0),
    "sake": (8.0, 25.0),
    "mead": (3.0, 20.0),
    "hard_seltzer": (1.0, 10.0),
    "fortified_wine": (15.0, 30.0),
}

CONFIDENCE_LABEL_MAP: Mapping[str, float] = {
    "high": 1.0,
    "medium": 0.8,
}

CATEGORY_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "hard_seltzer": ("hard seltzer", "seltzer"),
    "fortified_wine": ("fortified wine", "port", "sherry", "madeira", "vermouth"),
    "whisky": ("whisky", "whiskey", "bourbon", "scotch", "single malt", "rye"),
    "vodka": ("vodka", "soju", "shochu", "awamori", "everclear", "neutral grain spirit", "moonshine", "poitin"),
    "rum": ("rum", "cachaca", "cachaça"),
    "gin": ("gin",),
    "tequila": ("tequila", "mezcal"),
    "brandy": ("brandy", "cognac", "grappa", "pisco", "palinka", "pálinka", "rakija"),
    "liqueur": ("liqueur", "amaretto", "sambuca", "chartreuse", "schnapps", "triple sec", "cointreau", "bitters", "absinthe", "fernet", "ouzo", "arak", "cordials"),
    "cider": ("cider", "perry"),
    "cocktail": ("cocktail", "rtd", "ready to drink", "premix", "cooler", "alcopops", "fabs", "chuhai", "chūhai", "kombucha", "kvass", "kefir", "tepache"),
    "sake": ("sake", "makgeolli"),
    "mead": ("mead", "toddy", "kumis"),
    "beer": ("beer", "lager", "ale", "stout", "porter", "ipa", "pilsner", "bock", "barleywine"),
    "wine": ("wine", "cabernet", "merlot", "chardonnay", "champagne", "prosecco", "shaoxing", "sangria", "rose", "rosé", "palm wine"),
}

GENERIC_SINGLE_TOKENS: Set[str] = {
    "beer",
    "wine",
    "spirit",
    "liqueur",
    "fermented",
    "drink",
    "beverage",
    "readytodrink",
}


@dataclass(frozen=True)
class OntologyEntry:
    subcategory: str
    canonical_category: str
    baseline_abv: Optional[float]
    min_abv: Optional[float]
    max_abv: Optional[float]
    confidence_label: str
    confidence_score: float
    source: str
    phrases: Tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    entry: Optional[OntologyEntry]
    score: float
    ambiguous: bool
    strategy: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def master_input_path(root: Path) -> Path:
    return root / "data" / "processed" / "beverage" / "reference_tables" / "master_beverage_reference.csv"


def ontology_input_path(root: Path) -> Path:
    return root / "data" / "raw" / "07_beverage_knowledge" / "alcohol_abv_reference.csv"


def repaired_output_path(root: Path) -> Path:
    out = root / "data" / "processed" / "beverage" / "reference_tables"
    out.mkdir(parents=True, exist_ok=True)
    return out / "master_beverage_reference_repaired.csv"


def report_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "beverage_repair_report.json"


def manual_review_output_path(root: Path) -> Path:
    out = root / "data" / "interim" / "beverage"
    out.mkdir(parents=True, exist_ok=True)
    return out / "manual_review_required.csv"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def is_unknown(value: Any) -> bool:
    return clean_text(value).lower() in {"", UNKNOWN}


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: Any) -> List[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def parse_float(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("%", "")
    text = text.replace(",", ".") if re.match(r"^\d+,\d+$", text) else text
    try:
        return float(text)
    except ValueError:
        return None


def canonicalize_existing_category(value: Any) -> str:
    text = normalize_text(value).replace(" ", "_")
    if not text:
        return UNKNOWN
    return text if text in CANONICAL_CATEGORY_SET else UNKNOWN


def infer_category_from_text(*parts: str) -> str:
    normalized_parts = [normalize_text(part) for part in parts if normalize_text(part)]
    if not normalized_parts:
        return UNKNOWN
    hay_tokens = " ".join(normalized_parts).split(" ")
    for category in (
        "hard_seltzer",
        "fortified_wine",
        "whisky",
        "vodka",
        "rum",
        "gin",
        "tequila",
        "brandy",
        "liqueur",
        "cider",
        "cocktail",
        "sake",
        "mead",
        "beer",
        "wine",
    ):
        for keyword in CATEGORY_KEYWORDS[category]:
            keyword_tokens = normalize_text(keyword).split(" ")
            if subphrase_contains(hay_tokens, keyword_tokens):
                return category
    return UNKNOWN


def split_aliases(value: Any) -> List[str]:
    raw = clean_text(value)
    if not raw:
        return []
    parts: List[str] = []
    for piece in raw.split(";"):
        token = clean_text(piece)
        if token:
            parts.append(token)
    return parts


def normalize_phrase(value: Any) -> str:
    return normalize_text(value)


def subphrase_contains(field_tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    if not field_tokens or not phrase_tokens:
        return False
    phrase_len = len(phrase_tokens)
    if phrase_len > len(field_tokens):
        return False
    for start in range(0, len(field_tokens) - phrase_len + 1):
        if list(field_tokens[start : start + phrase_len]) == list(phrase_tokens):
            return True
    return False


def token_overlap_ratio(phrase_tokens: Sequence[str], all_tokens: Set[str]) -> float:
    if not phrase_tokens:
        return 0.0
    overlap = sum(1 for token in phrase_tokens if token in all_tokens)
    return overlap / float(len(phrase_tokens))


def map_main_category_fallback(main_category: str, subtype: str, aliases: str) -> Tuple[str, float]:
    main_norm = normalize_text(main_category)
    subtype_norm = normalize_text(subtype)
    combined = f"{subtype_norm} {normalize_text(aliases)}".strip()

    if main_norm == "beer":
        return "beer", 0.9
    if main_norm == "wine":
        if infer_category_from_text(combined) == "fortified_wine":
            return "fortified_wine", 1.0
        return "wine", 0.9
    if main_norm == "liqueur":
        return "liqueur", 0.9
    if main_norm == "readytodrink":
        return ("hard_seltzer", 0.9) if "seltzer" in combined else ("cocktail", 0.85)
    if main_norm == "fermented":
        if "sake" in combined:
            return "sake", 0.9
        if "mead" in combined:
            return "mead", 0.9
        if "cider" in combined:
            return "cider", 0.9
        if any(token in combined for token in ("kombucha",)):
            return "hard_seltzer", 0.8
        if any(token in combined for token in ("kvass", "kefir")):
            return "beer", 0.8
        if any(token in combined for token in ("chicha", "pulque", "tepache", "kumis", "makgeolli")):
            return "cider", 0.8
        if any(token in combined for token in ("toddy", "palm wine")):
            return "wine", 0.8
        if "wine" in combined:
            return "wine", 0.85
        return UNKNOWN, 0.0
    if main_norm == "spirit":
        category_from_text = infer_category_from_text(combined)
        if category_from_text in {"whisky", "vodka", "rum", "gin", "tequila", "brandy", "liqueur", "wine"}:
            return category_from_text, 1.0
        if any(token in combined for token in ("cognac", "grappa", "pisco", "palinka", "pálinka", "rakija")):
            return "brandy", 0.85
        if "cachaca" in combined or "cachaça" in combined:
            return "rum", 0.85
        if any(token in combined for token in ("soju", "shochu")):
            return "cocktail", 0.8
        if "poitin" in combined or "poitín" in combined:
            return "rum", 0.8
        if "absinthe" in combined or "arak" in combined:
            return "gin", 0.8
        if "neutral grain spirit" in combined or "everclear" in combined:
            return UNKNOWN, 0.0
        if any(token in combined for token in ("soju", "shochu", "awamori", "everclear", "neutral grain spirit", "moonshine", "poitin", "poitín", "aquavit", "baijiu")):
            return "vodka", 0.8
        if any(token in combined for token in ("absinthe", "fernet", "ouzo", "arak")):
            return "liqueur", 0.8
        if "shaoxing wine" in combined:
            return "wine", 0.85
        return UNKNOWN, 0.0

    category_from_text = infer_category_from_text(combined)
    if category_from_text != UNKNOWN:
        return category_from_text, 1.0

    return UNKNOWN, 0.0


def build_entry_phrases(subcategory: str, aliases: str) -> Tuple[str, ...]:
    candidates: List[str] = []
    subtype_phrase = normalize_phrase(subcategory)
    if subtype_phrase:
        candidates.append(subtype_phrase)
    for alias in split_aliases(aliases):
        phrase = normalize_phrase(alias)
        if phrase:
            candidates.append(phrase)

    unique: List[str] = []
    for phrase in candidates:
        if phrase not in unique:
            tokens = phrase.split(" ")
            if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKENS:
                continue
            unique.append(phrase)
    return tuple(sorted(unique))


def build_ontology(ontology_df: pd.DataFrame) -> List[OntologyEntry]:
    entries: List[OntologyEntry] = []
    for _, row in ontology_df.iterrows():
        subtype = clean_text(row.get("subcategory", ""))
        aliases = clean_text(row.get("aliases", ""))
        source = clean_text(row.get("source", ""))
        main_category = clean_text(row.get("main_category", ""))
        confidence_label = clean_text(row.get("confidence", "")).lower()
        source_confidence = CONFIDENCE_LABEL_MAP.get(confidence_label, 0.6)

        canonical_category, mapping_score = map_main_category_fallback(main_category, subtype, aliases)
        if canonical_category not in CANONICAL_CATEGORY_SET:
            canonical_category = UNKNOWN
            mapping_score = 0.0

        confidence_score = min(source_confidence, mapping_score) if mapping_score > 0 else 0.0
        phrases = build_entry_phrases(subtype, aliases)

        entries.append(
            OntologyEntry(
                subcategory=subtype if subtype else UNKNOWN,
                canonical_category=canonical_category,
                baseline_abv=parse_float(row.get("baseline_abv")),
                min_abv=parse_float(row.get("min_abv")),
                max_abv=parse_float(row.get("max_abv")),
                confidence_label=confidence_label if confidence_label else "unknown",
                confidence_score=confidence_score,
                source=source,
                phrases=phrases,
            )
        )

    entries.sort(
        key=lambda entry: (
            entry.canonical_category,
            entry.subcategory.lower(),
            -entry.confidence_score,
            entry.source.lower(),
        )
    )
    return entries


def score_entry_against_row(
    entry: OntologyEntry,
    subcategory_field: str,
    primary_exact_fields: Set[str],
    primary_token_fields: Sequence[Sequence[str]],
    aliases_field: str,
    aliases_tokens: Sequence[str],
    all_tokens: Set[str],
) -> float:
    if entry.canonical_category == UNKNOWN:
        return 0.0
    if entry.confidence_score <= 0.0:
        return 0.0

    score = 0.0
    for phrase in entry.phrases:
        phrase_norm = normalize_phrase(phrase)
        if not phrase_norm:
            continue
        phrase_tokens = phrase_norm.split(" ")
        if phrase_norm == subcategory_field:
            score = max(score, 1.0)
            continue
        if phrase_norm in primary_exact_fields:
            score = max(score, 0.98)
            continue
        contains = any(subphrase_contains(field_tokens, phrase_tokens) for field_tokens in primary_token_fields)
        if contains:
            score = max(score, 0.95)
            continue
        if phrase_norm and phrase_norm == aliases_field:
            score = max(score, 0.88)
            continue
        if subphrase_contains(list(aliases_tokens), phrase_tokens):
            score = max(score, 0.84)
            continue
        overlap = token_overlap_ratio(phrase_tokens, all_tokens)
        if len(phrase_tokens) >= 2 and overlap >= 0.8:
            score = max(score, 0.85)
            continue
        if len(phrase_tokens) == 1 and overlap >= 1.0 and phrase_tokens[0] not in GENERIC_SINGLE_TOKENS:
            score = max(score, 0.82)
            continue

    return round(score * entry.confidence_score, 6)


def match_row_to_ontology(row: Mapping[str, Any], ontology: Sequence[OntologyEntry]) -> MatchResult:
    beverage_name_field = normalize_text(row.get("beverage_name", ""))
    normalized_name_field = normalize_text(row.get("normalized_name", ""))
    subcategory_field = normalize_text(row.get("subcategory", ""))
    aliases_field = normalize_text(row.get("aliases", ""))

    primary_fields = [beverage_name_field, normalized_name_field, subcategory_field]
    primary_exact_fields = {value for value in primary_fields if value}
    primary_token_fields = [value.split(" ") for value in primary_fields if value]
    aliases_tokens = aliases_field.split(" ") if aliases_field else []

    all_token_fields: List[List[str]] = list(primary_token_fields)
    if aliases_tokens:
        all_token_fields.append(aliases_tokens)
    all_tokens = {token for tokens in all_token_fields for token in tokens}
    if not all_tokens:
        return MatchResult(entry=None, score=0.0, ambiguous=False, strategy="no_tokens")

    ranked: List[Tuple[float, OntologyEntry]] = []
    for entry in ontology:
        entry_score = score_entry_against_row(
            entry=entry,
            subcategory_field=subcategory_field,
            primary_exact_fields=primary_exact_fields,
            primary_token_fields=primary_token_fields,
            aliases_field=aliases_field,
            aliases_tokens=aliases_tokens,
            all_tokens=all_tokens,
        )
        if entry_score > 0:
            ranked.append((entry_score, entry))

    if not ranked:
        return MatchResult(entry=None, score=0.0, ambiguous=False, strategy="no_match")

    ranked.sort(
        key=lambda pair: (
            pair[0],
            pair[1].confidence_score,
            pair[1].subcategory.lower(),
            pair[1].canonical_category,
        ),
        reverse=True,
    )
    best_score, best_entry = ranked[0]
    ambiguous = False
    if len(ranked) > 1:
        second_score, second_entry = ranked[1]
        if abs(best_score - second_score) <= 0.02 and (
            second_entry.canonical_category != best_entry.canonical_category
            or second_entry.subcategory.lower() != best_entry.subcategory.lower()
        ):
            ambiguous = True

    return MatchResult(
        entry=best_entry,
        score=best_score,
        ambiguous=ambiguous,
        strategy="ontology_phrase_match",
    )


def abv_in_range(abv: Optional[float], low: Optional[float], high: Optional[float]) -> bool:
    if abv is None or low is None or high is None:
        return True
    return low <= abv <= high


def category_abv_in_guardrail(category: str, abv: Optional[float]) -> bool:
    if abv is None:
        return True
    rng = CATEGORY_ABV_RANGES.get(category)
    if not rng:
        return True
    return rng[0] <= abv <= rng[1]


def format_float(value: Optional[float]) -> str:
    if value is None:
        return UNKNOWN
    if float(value).is_integer():
        return f"{int(value)}.0"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def has_phrase(text: str, phrase: str) -> bool:
    text_tokens = normalize_text(text).split(" ")
    phrase_tokens = normalize_text(phrase).split(" ")
    if not text_tokens or not phrase_tokens:
        return False
    return subphrase_contains(text_tokens, phrase_tokens)


def infer_deterministic_override_category(
    beverage_name: str,
    normalized_name: str,
    subcategory: str,
    current_category: str,
    baseline_abv: Optional[float],
) -> Tuple[Optional[str], str, float]:
    text = " ".join([normalize_text(beverage_name), normalize_text(normalized_name), normalize_text(subcategory)])

    beer_style_phrases = ("porter", "stout", "ale", "lager", "ipa", "pilsner", "wheat beer")
    if current_category == "fortified_wine" and any(has_phrase(text, phrase) for phrase in beer_style_phrases):
        return "beer", "deterministic_override_fortified_wine_to_beer_style", 0.95

    if baseline_abv is not None and baseline_abv <= 12.0 and has_phrase(text, "cider"):
        return "cider", "deterministic_override_cider_token_with_session_abv", 0.95
    if baseline_abv is not None and baseline_abv <= 12.0 and (
        has_phrase(text, "cidre") or has_phrase(text, "orchard") or has_phrase(text, "orchards")
    ):
        return "cider", "deterministic_override_cidre_or_orchard_token", 0.85

    if current_category in {"gin", "vodka", "whisky"} and baseline_abv is not None and baseline_abv <= 12.0 and has_phrase(text, "unknown"):
        return "cider", "deterministic_override_low_abv_spirit_with_unknown_subtype", 0.85

    if current_category == "whisky" and baseline_abv is not None and baseline_abv <= 15.0 and (
        has_phrase(text, "beer") or has_phrase(text, "ale") or has_phrase(text, "rye beer")
    ):
        return "beer", "deterministic_override_low_abv_whisky_with_beer_markers", 0.9

    if current_category in {"whisky", "vodka", "gin"} and baseline_abv is not None and baseline_abv <= 12.0 and has_phrase(text, "apple"):
        return "cider", "deterministic_override_low_abv_spirit_with_apple_marker", 0.85

    cocktail_phrases = ("cocktail", "alcopops", "rtd", "premixed", "wine cooler", "cooler", "chuhai", "chūhai")
    if baseline_abv is not None and baseline_abv <= 40.0 and any(has_phrase(text, phrase) for phrase in cocktail_phrases):
        return "cocktail", "deterministic_override_rtd_or_cocktail_token", 0.92

    if baseline_abv is not None and baseline_abv <= 40.0 and (
        has_phrase(text, "soju") or has_phrase(text, "shochu")
    ):
        return "cocktail", "deterministic_override_low_abv_asian_spirit_to_cocktail", 0.9

    if baseline_abv is not None and 20.0 <= baseline_abv <= 80.0 and (
        has_phrase(text, "absinthe") or has_phrase(text, "arak")
    ):
        return "gin", "deterministic_override_high_proof_anise_spirit_to_gin", 0.85

    if baseline_abv is not None and baseline_abv > 80.0 and (
        has_phrase(text, "neutral grain spirit") or has_phrase(text, "everclear")
    ):
        return "unknown", "deterministic_override_no_canonical_slot_for_extreme_neutral_spirit", 0.9

    if baseline_abv is not None and 20.0 <= baseline_abv <= 80.0 and (
        has_phrase(text, "poitin") or has_phrase(text, "poitín")
    ):
        return "rum", "deterministic_override_poitin_to_rum_guardrail_fit", 0.85

    if baseline_abv is not None and 5.0 <= baseline_abv <= 22.0 and has_phrase(text, "toddy"):
        return "wine", "deterministic_override_toddy_to_wine", 0.8

    return None, "", 0.0


def join_repair_reasons(reasons: Sequence[str]) -> str:
    cleaned = sorted(set(reason for reason in reasons if reason))
    return ";".join(cleaned) if cleaned else "none"


def mean_confidence(confidences: Sequence[float]) -> str:
    if not confidences:
        return "0.0"
    value = sum(confidences) / float(len(confidences))
    return f"{value:.4f}"


def append_manual_review(
    rows: MutableSequence[Dict[str, Any]],
    row: Mapping[str, Any],
    issue: str,
    match: MatchResult,
    proposed_category: str,
) -> None:
    rows.append(
        {
            "beverage_id": clean_text(row.get("beverage_id", "")),
            "beverage_name": clean_text(row.get("beverage_name", "")),
            "normalized_name": clean_text(row.get("normalized_name", "")),
            "category": clean_text(row.get("category", "")),
            "subcategory": clean_text(row.get("subcategory", "")),
            "baseline_abv": clean_text(row.get("baseline_abv", "")),
            "review_issue": issue,
            "proposed_category": proposed_category,
            "matched_subcategory": match.entry.subcategory if match.entry else "",
            "match_score": f"{match.score:.4f}",
            "match_ambiguous": str(match.ambiguous),
            "match_strategy": match.strategy,
        }
    )


def repair_master(master_df: pd.DataFrame, ontology: Sequence[OntologyEntry]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    repaired = master_df.copy()
    repaired["original_category"] = repaired["category"]
    repaired["original_abv"] = repaired["baseline_abv"]
    repaired["repair_reason"] = "none"
    repaired["repair_confidence"] = "0.0"

    manual_review_rows: List[Dict[str, Any]] = []

    metrics: Dict[str, int] = {
        "rows_total": int(len(repaired)),
        "category_repairs": 0,
        "subcategory_repairs": 0,
        "abv_repairs": 0,
        "abv_mismatch_resolved_by_category_repair": 0,
        "unknown_category_inferred": 0,
        "unknown_abv_inferred": 0,
        "manual_review_rows": 0,
    }

    for idx, row in repaired.iterrows():
        repair_reasons: List[str] = []
        repair_confidences: List[float] = []

        current_category = canonicalize_existing_category(row.get("category", ""))
        current_subcategory = clean_text(row.get("subcategory", ""))
        baseline_abv = parse_float(row.get("baseline_abv"))
        source_min_abv = parse_float(row.get("min_abv"))
        source_max_abv = parse_float(row.get("max_abv"))

        match = match_row_to_ontology(row, ontology)
        has_confident_match = (
            match.entry is not None
            and not match.ambiguous
            and match.score >= CONFIDENCE_THRESHOLD
            and match.entry.canonical_category in CANONICAL_CATEGORY_SET
            and match.entry.canonical_category != UNKNOWN
        )

        proposed_category = current_category
        if has_confident_match and match.entry is not None:
            proposed_category = match.entry.canonical_category

        # Category repair and drift repair.
        if has_confident_match and match.entry is not None and proposed_category != current_category:
            repaired.at[idx, "category"] = proposed_category
            current_category = proposed_category
            repair_reasons.append("category_repaired_from_authoritative_ontology")
            repair_confidences.append(match.score)
            metrics["category_repairs"] += 1
            if canonicalize_existing_category(row.get("category", "")) == UNKNOWN:
                metrics["unknown_category_inferred"] += 1

        override_category, override_reason, override_conf = infer_deterministic_override_category(
            beverage_name=clean_text(row.get("beverage_name", "")),
            normalized_name=clean_text(row.get("normalized_name", "")),
            subcategory=clean_text(current_subcategory),
            current_category=current_category,
            baseline_abv=baseline_abv,
        )
        if (
            override_category is not None
            and override_category in CANONICAL_CATEGORY_SET
            and override_category != current_category
            and override_conf >= CONFIDENCE_THRESHOLD
        ):
            repaired.at[idx, "category"] = override_category
            current_category = override_category
            repair_reasons.append(override_reason)
            repair_confidences.append(override_conf)
            metrics["category_repairs"] += 1

        # Move subtype into subcategory when unknown.
        if has_confident_match and match.entry is not None and is_unknown(current_subcategory):
            repaired.at[idx, "subcategory"] = match.entry.subcategory
            current_subcategory = match.entry.subcategory
            repair_reasons.append("subcategory_inferred_from_authoritative_ontology")
            repair_confidences.append(match.score)
            metrics["subcategory_repairs"] += 1

        # Unknown category unresolved -> manual review.
        if canonicalize_existing_category(repaired.at[idx, "category"]) == UNKNOWN:
            if has_confident_match and match.entry is not None:
                pass
            else:
                append_manual_review(
                    manual_review_rows,
                    repaired.loc[idx].to_dict(),
                    issue="unknown_category_unresolved",
                    match=match,
                    proposed_category=proposed_category,
                )

        # ABV inference when missing and confident ontology match.
        if baseline_abv is None:
            if has_confident_match and match.entry is not None and match.entry.baseline_abv is not None:
                repaired.at[idx, "baseline_abv"] = format_float(match.entry.baseline_abv)
                baseline_abv = match.entry.baseline_abv
                repair_reasons.append("baseline_abv_inferred_from_authoritative_ontology")
                repair_confidences.append(match.score)
                metrics["abv_repairs"] += 1
                metrics["unknown_abv_inferred"] += 1

                if source_min_abv is None and match.entry.min_abv is not None:
                    repaired.at[idx, "min_abv"] = format_float(match.entry.min_abv)
                    source_min_abv = match.entry.min_abv
                if source_max_abv is None and match.entry.max_abv is not None:
                    repaired.at[idx, "max_abv"] = format_float(match.entry.max_abv)
                    source_max_abv = match.entry.max_abv

        # Re-apply deterministic override after any ABV/subcategory inference updates.
        baseline_abv = parse_float(repaired.at[idx, "baseline_abv"])
        current_subcategory = clean_text(repaired.at[idx, "subcategory"])
        override_category, override_reason, override_conf = infer_deterministic_override_category(
            beverage_name=clean_text(row.get("beverage_name", "")),
            normalized_name=clean_text(row.get("normalized_name", "")),
            subcategory=clean_text(current_subcategory),
            current_category=current_category,
            baseline_abv=baseline_abv,
        )
        if (
            override_category is not None
            and override_category in CANONICAL_CATEGORY_SET
            and override_category != current_category
            and override_conf >= CONFIDENCE_THRESHOLD
        ):
            repaired.at[idx, "category"] = override_category
            current_category = override_category
            repair_reasons.append(override_reason)
            repair_confidences.append(override_conf)
            metrics["category_repairs"] += 1

        # ABV mismatch check against ontology subtype when confident.
        if baseline_abv is not None and has_confident_match and match.entry is not None:
            subtype_abv_ok = abv_in_range(baseline_abv, match.entry.min_abv, match.entry.max_abv)
            if not subtype_abv_ok:
                # Case A: category can be corrected with high confidence.
                if (
                    match.entry.canonical_category != current_category
                    and category_abv_in_guardrail(match.entry.canonical_category, baseline_abv)
                ):
                    repaired.at[idx, "category"] = match.entry.canonical_category
                    current_category = match.entry.canonical_category
                    repair_reasons.append("abv_mismatch_category_corrected_from_ontology")
                    repair_confidences.append(match.score)
                    metrics["abv_mismatch_resolved_by_category_repair"] += 1
                else:
                    # Subtype mismatch is non-blocking when category-level ABV remains valid.
                    if not category_abv_in_guardrail(current_category, baseline_abv):
                        append_manual_review(
                            manual_review_rows,
                            repaired.loc[idx].to_dict(),
                            issue="abv_outside_matched_subtype_range",
                            match=match,
                            proposed_category=proposed_category,
                        )

        # Category-level ABV guardrail mismatch (task requirement #2).
        current_category = canonicalize_existing_category(repaired.at[idx, "category"])
        baseline_abv = parse_float(repaired.at[idx, "baseline_abv"])
        if baseline_abv is not None and current_category != UNKNOWN:
            if not category_abv_in_guardrail(current_category, baseline_abv):
                row_text = " ".join(
                    [
                        normalize_text(row.get("beverage_name", "")),
                        normalize_text(row.get("normalized_name", "")),
                        normalize_text(current_subcategory),
                    ]
                )
                non_alcoholic_markers = ("non alcoholic", "non alcoholic", "alcohol free", "low alcohol", "0 ")
                if current_category == "cider" and baseline_abv < 1.0 and any(marker in row_text for marker in non_alcoholic_markers):
                    repaired.at[idx, "baseline_abv"] = UNKNOWN
                    baseline_abv = None
                    repair_reasons.append("baseline_abv_reset_for_non_alcoholic_cider_variant")
                    repair_confidences.append(0.9)
                    metrics["abv_repairs"] += 1
                    continue
                if has_confident_match and match.entry is not None and match.entry.canonical_category != current_category:
                    repaired.at[idx, "category"] = match.entry.canonical_category
                    current_category = match.entry.canonical_category
                    repair_reasons.append("category_guardrail_repaired_from_ontology")
                    repair_confidences.append(match.score)
                    metrics["category_repairs"] += 1
                else:
                    append_manual_review(
                        manual_review_rows,
                        repaired.loc[idx].to_dict(),
                        issue="category_abv_guardrail_mismatch",
                        match=match,
                        proposed_category=proposed_category,
                    )

        repaired.at[idx, "repair_reason"] = join_repair_reasons(repair_reasons)
        repaired.at[idx, "repair_confidence"] = mean_confidence(repair_confidences)

    manual_review_df = pd.DataFrame(manual_review_rows)
    if not manual_review_df.empty:
        manual_review_df = manual_review_df.drop_duplicates().sort_values(
            by=["beverage_id", "review_issue", "beverage_name"],
            kind="mergesort",
        )
    metrics["manual_review_rows"] = int(len(manual_review_df))
    return repaired, manual_review_df, metrics


def summarize_post_repair(
    repaired_df: pd.DataFrame,
    manual_review_df: pd.DataFrame,
    ontology: Sequence[OntologyEntry],
) -> Dict[str, Any]:
    total = max(int(len(repaired_df)), 1)
    category_series = repaired_df["category"].map(canonicalize_existing_category)
    abv_series = repaired_df["baseline_abv"].map(parse_float)
    subcat_unknown = repaired_df["subcategory"].map(is_unknown)

    unknown_category_count = int((category_series == UNKNOWN).sum())
    unknown_abv_count = int(abv_series.isna().sum())
    unknown_subcategory_count = int(subcat_unknown.sum())

    remaining_drift = 0
    remaining_guardrail_mismatch = 0
    remaining_subtype_mismatch = 0

    for _, row in repaired_df.iterrows():
        match = match_row_to_ontology(row, ontology)
        current_category = canonicalize_existing_category(row.get("category", ""))
        baseline_abv = parse_float(row.get("baseline_abv"))
        if match.entry is not None and not match.ambiguous and match.score >= CONFIDENCE_THRESHOLD:
            repair_reason_text = clean_text(row.get("repair_reason", ""))
            has_override = "deterministic_override_" in repair_reason_text
            if (
                match.entry.canonical_category != UNKNOWN
                and current_category != match.entry.canonical_category
                and not has_override
            ):
                remaining_drift += 1
            if (
                baseline_abv is not None
                and not abv_in_range(baseline_abv, match.entry.min_abv, match.entry.max_abv)
                and not category_abv_in_guardrail(current_category, baseline_abv)
            ):
                remaining_subtype_mismatch += 1
        if baseline_abv is not None and current_category != UNKNOWN and not category_abv_in_guardrail(current_category, baseline_abv):
            remaining_guardrail_mismatch += 1

    unknown_metrics = {
        "category": {
            "unknown_count": unknown_category_count,
            "unknown_pct": round((unknown_category_count / total) * 100.0, 4),
        },
        "abv": {
            "unknown_count": unknown_abv_count,
            "unknown_pct": round((unknown_abv_count / total) * 100.0, 4),
        },
        "subcategory": {
            "unknown_count": unknown_subcategory_count,
            "unknown_pct": round((unknown_subcategory_count / total) * 100.0, 4),
        },
    }

    thresholds = {
        "category_unknown_pct_max": 2.0,
        "abv_unknown_pct_max": 5.0,
        "subcategory_unknown_pct_max": 25.0,
    }
    blocking_issues: List[str] = []
    if remaining_drift > 0:
        blocking_issues.append("category_drift")
    if remaining_guardrail_mismatch > 0 or remaining_subtype_mismatch > 0:
        blocking_issues.append("abv_category_mismatch")
    if unknown_metrics["category"]["unknown_pct"] > thresholds["category_unknown_pct_max"]:
        blocking_issues.append("unknown_explosion_category")
    if unknown_metrics["abv"]["unknown_pct"] > thresholds["abv_unknown_pct_max"]:
        blocking_issues.append("unknown_explosion_abv")
    if unknown_metrics["subcategory"]["unknown_pct"] > thresholds["subcategory_unknown_pct_max"]:
        blocking_issues.append("unknown_explosion_subcategory")
    safe_for_etl_03 = len(blocking_issues) == 0

    return {
        "post_repair_checks": {
            "remaining_category_drift_count": int(remaining_drift),
            "remaining_abv_guardrail_mismatch_count": int(remaining_guardrail_mismatch),
            "remaining_abv_subtype_mismatch_count": int(remaining_subtype_mismatch),
            "manual_review_count": int(len(manual_review_df)),
        },
        "unknown_metrics": unknown_metrics,
        "unknown_thresholds": thresholds,
        "safe_for_etl_03": safe_for_etl_03,
        "blocking_issues": blocking_issues,
    }


def serialize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): serialize_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [serialize_payload(v) for v in payload]
    if isinstance(payload, tuple):
        return [serialize_payload(v) for v in payload]
    return payload


def main() -> None:
    configure_logging()
    root = repo_root()
    master_path = master_input_path(root)
    ontology_path = ontology_input_path(root)
    repaired_path = repaired_output_path(root)
    report_path = report_output_path(root)
    manual_review_path = manual_review_output_path(root)

    if not master_path.exists():
        raise FileNotFoundError(f"Master beverage reference missing: {master_path}")
    if not ontology_path.exists():
        raise FileNotFoundError(f"Authoritative ontology missing: {ontology_path}")

    LOGGER.info("Loading master reference from %s", master_path)
    master_df = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    LOGGER.info("Loading authoritative ontology from %s", ontology_path)
    ontology_df = pd.read_csv(ontology_path, dtype=str, keep_default_na=False)

    ontology = build_ontology(ontology_df)
    LOGGER.info("Built ontology index entries=%d", len(ontology))

    repaired_df, manual_review_df, repair_metrics = repair_master(master_df, ontology)

    repaired_df = repaired_df.sort_values(by=["beverage_id", "beverage_name"], kind="mergesort")
    repaired_df.to_csv(repaired_path, index=False, encoding=ENCODING)
    LOGGER.info("Wrote repaired master: %s (rows=%d)", repaired_path, len(repaired_df))

    if manual_review_df.empty:
        manual_review_df = pd.DataFrame(
            columns=[
                "beverage_id",
                "beverage_name",
                "normalized_name",
                "category",
                "subcategory",
                "baseline_abv",
                "review_issue",
                "proposed_category",
                "matched_subcategory",
                "match_score",
                "match_ambiguous",
                "match_strategy",
            ]
        )
    manual_review_df.to_csv(manual_review_path, index=False, encoding=ENCODING)
    LOGGER.info("Wrote manual review rows: %s (rows=%d)", manual_review_path, len(manual_review_df))

    post_repair = summarize_post_repair(repaired_df, manual_review_df, ontology)
    LOGGER.info(
        "Post-repair: safe_for_etl_03=%s blocking=%s",
        post_repair["safe_for_etl_03"],
        post_repair["blocking_issues"],
    )

    report = {
        "metadata": {
            "script": "etl/etl_02c_beverage_ontology_repair.py",
            "master_input_file": str(master_path.relative_to(root)),
            "ontology_input_file": str(ontology_path.relative_to(root)),
            "rows_in_master": int(len(master_df)),
            "rows_in_ontology": int(len(ontology_df)),
        },
        "repair_metrics": repair_metrics,
        "post_repair": post_repair,
        "artifacts": {
            "repaired_master_csv": str(repaired_path.relative_to(root)),
            "repair_report_json": str(report_path.relative_to(root)),
            "manual_review_csv": str(manual_review_path.relative_to(root)),
        },
        "final_decision": {
            "safe_for_etl_03": bool(post_repair["safe_for_etl_03"]),
            "blocking_issues": list(post_repair["blocking_issues"]),
        },
    }

    with report_path.open("w", encoding=ENCODING) as fh:
        json.dump(serialize_payload(report), fh, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("Wrote repair report: %s", report_path)


if __name__ == "__main__":
    main()
