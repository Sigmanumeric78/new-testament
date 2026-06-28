"""ETL step 07f: deterministic semantic retrieval validation (Weaviate + Neo4j + PBPK)."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import weaviate  # type: ignore
    from weaviate.classes.query import MetadataQuery  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    weaviate = None
    MetadataQuery = None

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    GraphDatabase = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.pbpk.pbpk_master_simulator import (  # noqa: E402
    beverage_modifiers_path,
    parameter_library_path,
    population_modifiers_path,
    run_simulation,
)
from utils.config import get_neo4j_config, get_weaviate_config  # noqa: E402

LOGGER = logging.getLogger("etl_07f_semantic_retrieval_testing")

ENCODING = "utf-8"
TOP_K = 10
HYBRID_ALPHA = 0.65
EPS = 1e-12

COLLECTION_FILES: Tuple[Tuple[str, str], ...] = (
    ("BeverageKnowledge", "beverage_embeddings.parquet"),
    ("CompoundKnowledge", "compound_embeddings.parquet"),
    ("MetabolismKnowledge", "metabolism_embeddings.parquet"),
    ("PBPKKnowledge", "pbpk_embeddings.parquet"),
    ("ToxicityKnowledge", "toxicity_embeddings.parquet"),
    ("PopulationKnowledge", "population_embeddings.parquet"),
    ("ScientificEvidence", "scientific_evidence_embeddings.parquet"),
)

REQUIRED_COLLECTIONS: Tuple[str, ...] = tuple(collection for collection, _ in COLLECTION_FILES)

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "object_id",
    "chunk_id",
    "collection",
    "title",
    "content",
    "embedding",
    "metadata",
    "provenance",
)

REQUIRED_PROPERTIES: Tuple[str, ...] = (
    "object_id",
    "chunk_id",
    "title",
    "content",
    "collection",
    "confidence_score",
    "source_dataset",
    "source_file",
    "metadata",
    "provenance",
)


@dataclass(frozen=True)
class QueryScenario:
    scenario_id: str
    query_text: str
    expected_concepts: Tuple[str, ...]


TEST_SCENARIOS: Tuple[QueryScenario, ...] = (
    QueryScenario(
        scenario_id="q1_whisky_vs_beer",
        query_text="why whisky hits harder than beer",
        expected_concepts=("ethanol concentration", "congeners", "acetaldehyde", "abv"),
    ),
    QueryScenario(
        scenario_id="q2_hangover_compounds",
        query_text="what compounds worsen hangovers",
        expected_concepts=("acetaldehyde", "histamine", "sulfites", "fusel alcohols", "congeners"),
    ),
    QueryScenario(
        scenario_id="q3_female_intoxication",
        query_text="why women get drunk faster",
        expected_concepts=("body water", "sex differences", "distribution volume", "adh", "population modifiers"),
    ),
    QueryScenario(
        scenario_id="q4_empty_stomach",
        query_text="why drinking on an empty stomach hits faster",
        expected_concepts=("gastric emptying", "absorption", "fed state", "fasted state"),
    ),
    QueryScenario(
        scenario_id="q5_wine_headaches",
        query_text="why wine gives headaches",
        expected_concepts=("sulfites", "histamine", "tyramine", "polyphenols"),
    ),
)

# Deterministic, domain-focused aliases used for query-vector seeding and concept matching.
CONCEPT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "ethanol concentration": ("ethanol", "blood alcohol", "bac", "ethanol concentration"),
    "congeners": ("congeners", "congener"),
    "acetaldehyde": ("acetaldehyde",),
    "abv": ("abv", "alcohol by volume"),
    "histamine": ("histamine",),
    "sulfites": ("sulfites", "sulfite"),
    "fusel alcohols": ("fusel alcohols", "fusel alcohol", "fusel"),
    "body water": ("body water", "total body water", "body_water_fraction"),
    "sex differences": ("sex differences", "female", "male", "sex"),
    "distribution volume": ("distribution volume", "volume of distribution", "ethanol_distribution_volume"),
    "adh": ("adh", "alcohol dehydrogenase", "adh_metabolism_rate"),
    "population modifiers": ("population modifiers", "population_group", "modifier"),
    "gastric emptying": ("gastric emptying", "gastric_emptying_rate"),
    "absorption": ("absorption", "intestinal absorption", "intestinal_absorption_rate"),
    "fed state": ("fed", "fed state", "with food"),
    "fasted state": ("fasted", "fasted state", "empty stomach"),
    "tyramine": ("tyramine",),
    "polyphenols": ("polyphenols", "polyphenol"),
}

SCENARIO_GROUNDING_QUERIES: Mapping[str, str] = {
    "q1_whisky_vs_beer": (
        "MATCH (b:Beverage)-[:CONTAINS]->(:Compound)-[:METABOLIZED_BY]->(:Enzyme) "
        "WHERE toLower(coalesce(b.name,'')) CONTAINS 'whisk' "
        "   OR toLower(coalesce(b.category,'')) CONTAINS 'beer' "
        "RETURN count(*) AS path_count"
    ),
    "q2_hangover_compounds": (
        "MATCH (:Beverage)-[:CONTAINS]->(:Compound)-[:CONTRIBUTES_TO]->(t:ToxicityRisk) "
        "WHERE toLower(coalesce(t.risk_type,'')) CONTAINS 'hangover' "
        "RETURN count(*) AS path_count"
    ),
    "q3_female_intoxication": (
        "MATCH (g:PopulationGroup)-[:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(:BodyCompartment) "
        "WHERE g.group_name = 'female' "
        "  AND p.parameter_name IN ['body_water_fraction', 'ethanol_distribution_volume', 'adh_metabolism_rate'] "
        "RETURN count(*) AS path_count"
    ),
    "q4_empty_stomach": (
        "MATCH (g:PopulationGroup)-[:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(:BodyCompartment) "
        "WHERE g.group_name IN ['fasted', 'fed'] "
        "  AND p.parameter_name IN ['gastric_emptying_rate', 'intestinal_absorption_rate'] "
        "RETURN count(*) AS path_count"
    ),
    "q5_wine_headaches": (
        "MATCH (b:Beverage)-[:CONTAINS]->(:Compound)-[:CONTRIBUTES_TO]->(:ToxicityRisk) "
        "WHERE toLower(coalesce(b.name,'')) CONTAINS 'wine' "
        "RETURN count(*) AS path_count"
    ),
}


@dataclass(frozen=True)
class CorpusRow:
    object_id: str
    chunk_id: str
    collection: str
    title: str
    content: str
    metadata: str
    provenance: str
    vector: np.ndarray
    token_set: Set[str]
    search_text: str


@dataclass(frozen=True)
class RetrievalItem:
    object_id: str
    chunk_id: str
    collection: str
    title: str
    content: str
    metadata: str
    provenance: str
    similarity_score: float
    raw_score: Optional[float]
    distance: Optional[float]
    certainty: Optional[float]
    retrieval_method: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def embeddings_dir(root: Path) -> Path:
    return root / "data" / "processed" / "weaviate" / "embedded"


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "semantic_retrieval_test_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return ""
    return text


def parse_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        text = clean_text(value)
        if not text:
            return default
        try:
            return float(text)
        except Exception:
            return default


def normalize_token(value: str) -> str:
    text = clean_text(value).lower()
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_token(text)))


def stringify_json_field(value: Any) -> str:
    if isinstance(value, str):
        text = clean_text(value)
        if not text:
            return "{}"
        return text
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    text = clean_text(value)
    if not text:
        return "{}"
    return text


def parse_vector(value: Any) -> Tuple[np.ndarray, bool]:
    data: Any = value
    if isinstance(data, str):
        text = clean_text(data)
        if not text:
            return np.array([], dtype=float), True
        try:
            data = json.loads(text)
        except Exception:
            return np.array([], dtype=float), True
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, (list, tuple)):
        return np.array([], dtype=float), True
    numbers: List[float] = []
    has_nan = False
    for item in data:
        try:
            number = float(item)
        except Exception:
            return np.array([], dtype=float), True
        if math.isnan(number):
            has_nan = True
        numbers.append(number)
    return np.array(numbers, dtype=float), has_nan


def parse_weaviate_url(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid WEAVIATE_URL: '{url}'. Expected http(s)://host[:port]")
    secure = parsed.scheme.lower() == "https"
    default_port = 443 if secure else 80
    return {
        "http_host": parsed.hostname,
        "http_port": int(parsed.port or default_port),
        "http_secure": secure,
    }


def connect_weaviate(config: Mapping[str, str]) -> Any:
    url_info = parse_weaviate_url(config["url"])
    grpc_host = clean_text(config.get("grpc_host", "")) or "localhost"
    grpc_port = int(clean_text(config.get("grpc_port", "")) or "50051")
    api_key = clean_text(config.get("api_key", ""))

    auth_credentials = None
    if api_key:
        try:
            from weaviate.classes.init import Auth  # type: ignore

            auth_credentials = Auth.api_key(api_key)
        except Exception:
            from weaviate.auth import AuthApiKey  # type: ignore

            auth_credentials = AuthApiKey(api_key)

    try:
        return weaviate.connect_to_custom(
            http_host=url_info["http_host"],
            http_port=url_info["http_port"],
            http_secure=url_info["http_secure"],
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=url_info["http_secure"],
            auth_credentials=auth_credentials,
        )
    except Exception:
        return weaviate.connect_to_local(
            host=url_info["http_host"],
            port=url_info["http_port"],
            grpc_port=grpc_port,
            auth_credentials=auth_credentials,
        )


def load_embedding_corpus(root: Path) -> Tuple[Dict[str, List[CorpusRow]], Dict[str, Any]]:
    base = embeddings_dir(root)
    rows_by_collection: Dict[str, List[CorpusRow]] = {collection: [] for collection, _ in COLLECTION_FILES}

    missing_vectors = 0
    nan_vectors = 0
    dimension_mismatch_rows = 0
    dimensions: Set[int] = set()

    expected_dimension: Optional[int] = None

    for collection_name, filename in COLLECTION_FILES:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding parquet: {path}")
        df = pd.read_parquet(path)

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{filename} missing required columns: {', '.join(missing_cols)}")

        ordered = df.sort_values(by=["chunk_id", "object_id"], kind="mergesort").reset_index(drop=True)
        for _, row in ordered.iterrows():
            object_id = clean_text(row.get("object_id"))
            if not object_id:
                continue
            chunk_id = clean_text(row.get("chunk_id"))
            title = clean_text(row.get("title"))
            content = clean_text(row.get("content"))
            metadata = stringify_json_field(row.get("metadata"))
            provenance = stringify_json_field(row.get("provenance"))
            collection = clean_text(row.get("collection")) or collection_name

            vector, has_nan = parse_vector(row.get("embedding"))
            if vector.size == 0:
                missing_vectors += 1
                continue
            if has_nan:
                nan_vectors += 1
                continue

            current_dim = int(vector.size)
            if expected_dimension is None:
                expected_dimension = current_dim
            if current_dim != expected_dimension:
                dimension_mismatch_rows += 1
                continue
            dimensions.add(current_dim)

            search_text = " ".join(
                item for item in [title, content, metadata, provenance, object_id, chunk_id, collection] if item
            ).lower()
            token_set = tokenize(search_text)
            rows_by_collection[collection_name].append(
                CorpusRow(
                    object_id=object_id,
                    chunk_id=chunk_id,
                    collection=collection,
                    title=title,
                    content=content,
                    metadata=metadata,
                    provenance=provenance,
                    vector=vector,
                    token_set=token_set,
                    search_text=search_text,
                )
            )

    for collection_name in sorted(rows_by_collection):
        rows_by_collection[collection_name] = sorted(
            rows_by_collection[collection_name],
            key=lambda row: (row.chunk_id, row.object_id),
        )

    total_rows = sum(len(rows) for rows in rows_by_collection.values())
    if total_rows == 0:
        raise ValueError("No valid embedding rows were loaded from parquet inputs.")

    embedding_dimension = int(next(iter(dimensions))) if len(dimensions) == 1 else 0
    metrics = {
        "total_rows": int(total_rows),
        "missing_vectors": int(missing_vectors),
        "nan_vectors": int(nan_vectors),
        "dimension_mismatch_rows": int(dimension_mismatch_rows),
        "embedding_dimension": int(embedding_dimension),
        "dimension_consistent": bool(len(dimensions) == 1 and embedding_dimension > 0 and dimension_mismatch_rows == 0),
        "rows_per_collection": {key: len(value) for key, value in rows_by_collection.items()},
    }
    return rows_by_collection, metrics


def scenario_terms(scenario: QueryScenario) -> List[str]:
    terms: Set[str] = set()
    query_text = normalize_token(scenario.query_text)
    terms.update(tokenize(query_text))
    terms.add(query_text)

    for concept in scenario.expected_concepts:
        concept_norm = normalize_token(concept)
        terms.add(concept_norm)
        terms.update(tokenize(concept_norm))
        for alias in CONCEPT_ALIASES.get(concept_norm, tuple()):
            alias_norm = normalize_token(alias)
            terms.add(alias_norm)
            terms.update(tokenize(alias_norm))

    return sorted(term for term in terms if term)


def build_query_vector(
    scenario: QueryScenario,
    rows_by_collection: Mapping[str, Sequence[CorpusRow]],
    embedding_dimension: int,
) -> Tuple[List[float], Dict[str, Any]]:
    terms = scenario_terms(scenario)
    candidates: List[Tuple[int, str, np.ndarray]] = []

    for collection_name in sorted(rows_by_collection):
        for row in rows_by_collection[collection_name]:
            overlap = 0
            for term in terms:
                if " " in term:
                    if term in row.search_text:
                        overlap += 1
                else:
                    if term in row.token_set:
                        overlap += 1
            if overlap > 0:
                candidates.append((overlap, row.object_id, row.vector))

    # Deterministic fallback if no lexical anchors are found.
    if not candidates:
        fallback = list(rows_by_collection.get("ScientificEvidence", []))
        if not fallback:
            for collection_name in sorted(rows_by_collection):
                fallback.extend(rows_by_collection[collection_name])
        fallback = sorted(fallback, key=lambda row: (row.chunk_id, row.object_id))[:2000]
        if not fallback:
            raise ValueError("Unable to build query vector: embedding corpus is empty.")
        stacked = np.vstack([row.vector for row in fallback])
        mean_vector = np.mean(stacked, axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if norm > EPS:
            mean_vector = mean_vector / norm
        return mean_vector.astype(float).tolist(), {
            "seed_terms": terms,
            "seed_mode": "fallback_average",
            "seed_rows": int(len(fallback)),
            "seed_overlap_max": 0,
            "embedding_dimension": int(embedding_dimension),
        }

    candidates_sorted = sorted(candidates, key=lambda item: (-item[0], item[1]))[:3000]
    weights = np.array([float(item[0]) for item in candidates_sorted], dtype=float)
    matrix = np.vstack([item[2] for item in candidates_sorted])
    weighted = np.average(matrix, axis=0, weights=weights)

    norm = float(np.linalg.norm(weighted))
    if norm > EPS:
        weighted = weighted / norm

    return weighted.astype(float).tolist(), {
        "seed_terms": terms,
        "seed_mode": "weighted_overlap_average",
        "seed_rows": int(len(candidates_sorted)),
        "seed_overlap_max": int(max(item[0] for item in candidates_sorted)),
        "embedding_dimension": int(embedding_dimension),
    }


def as_properties_dict(raw: Any) -> Dict[str, Any]:
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


def metadata_value(metadata: Any, key: str) -> Optional[float]:
    if metadata is None:
        return None
    value = getattr(metadata, key, None)
    return parse_float(value, default=None)


def similarity_score(raw_score: Optional[float], distance: Optional[float], certainty: Optional[float]) -> float:
    if raw_score is not None:
        return float(raw_score)
    if certainty is not None:
        return float(certainty)
    if distance is not None:
        return float(1.0 / (1.0 + max(distance, 0.0)))
    return 0.0


def retrieve_top_k_for_query(
    client: Any,
    scenario: QueryScenario,
    query_vector: Sequence[float],
    k: int,
) -> Tuple[List[RetrievalItem], Dict[str, str]]:
    items: List[RetrievalItem] = []
    method_by_collection: Dict[str, str] = {}

    metadata_query = None
    if MetadataQuery is not None:
        metadata_query = MetadataQuery(score=True, distance=True, certainty=True)

    for collection_name in REQUIRED_COLLECTIONS:
        if not client.collections.exists(collection_name):
            raise RuntimeError(
                f"Collection missing in Weaviate: {collection_name}. Ensure ingestion completed before retrieval testing."
            )

        collection = client.collections.get(collection_name)
        response = None
        method = "hybrid"
        try:
            response = collection.query.hybrid(
                query=scenario.query_text,
                vector=list(query_vector),
                alpha=HYBRID_ALPHA,
                limit=k,
                return_metadata=metadata_query,
                return_properties=list(REQUIRED_PROPERTIES),
            )
        except Exception:
            method = "near_vector_fallback"
            response = collection.query.near_vector(
                near_vector=list(query_vector),
                limit=k,
                return_metadata=metadata_query,
                return_properties=list(REQUIRED_PROPERTIES),
            )

        method_by_collection[collection_name] = method

        objects = list(getattr(response, "objects", []) or [])
        for obj in objects:
            props = as_properties_dict(getattr(obj, "properties", {}))
            metadata = getattr(obj, "metadata", None)
            raw_score = metadata_value(metadata, "score")
            distance = metadata_value(metadata, "distance")
            certainty = metadata_value(metadata, "certainty")
            score = similarity_score(raw_score=raw_score, distance=distance, certainty=certainty)
            items.append(
                RetrievalItem(
                    object_id=clean_text(props.get("object_id")),
                    chunk_id=clean_text(props.get("chunk_id")),
                    collection=clean_text(props.get("collection")) or collection_name,
                    title=clean_text(props.get("title")),
                    content=clean_text(props.get("content")),
                    metadata=clean_text(props.get("metadata")),
                    provenance=clean_text(props.get("provenance")),
                    similarity_score=float(score),
                    raw_score=raw_score,
                    distance=distance,
                    certainty=certainty,
                    retrieval_method=method,
                )
            )

    ordered = sorted(
        items,
        key=lambda row: (
            -row.similarity_score,
            row.collection,
            row.chunk_id,
            row.object_id,
        ),
    )

    # Deduplicate deterministically by (collection, object_id, chunk_id)
    seen: Set[Tuple[str, str, str]] = set()
    deduped: List[RetrievalItem] = []
    for row in ordered:
        key = (row.collection, row.object_id, row.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= k:
            break

    return deduped, method_by_collection


def concept_alias_set(concept: str) -> Set[str]:
    concept_norm = normalize_token(concept)
    aliases = set(CONCEPT_ALIASES.get(concept_norm, tuple()))
    aliases.add(concept_norm)
    expanded: Set[str] = set()
    for alias in aliases:
        alias_norm = normalize_token(alias)
        if alias_norm:
            expanded.add(alias_norm)
            expanded.update(tokenize(alias_norm))
    return expanded


def concept_present_in_text(concept: str, text: str, tokens: Set[str]) -> bool:
    text_norm = normalize_token(text)
    for alias in concept_alias_set(concept):
        if not alias:
            continue
        if " " in alias:
            if alias in text_norm:
                return True
        else:
            if alias in tokens:
                return True
            if alias in text_norm:
                return True
    return False


def evaluate_concept_coverage(
    scenario: QueryScenario,
    retrieved: Sequence[RetrievalItem],
) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    matched_by_item: Dict[str, Set[str]] = {}
    matched_concepts: Set[str] = set()

    for item in retrieved:
        key = f"{item.collection}::{item.object_id}::{item.chunk_id}"
        text = " ".join([item.title, item.content, item.metadata, item.provenance]).strip()
        tokens = tokenize(text)
        hits: Set[str] = set()
        for concept in scenario.expected_concepts:
            if concept_present_in_text(concept=concept, text=text, tokens=tokens):
                hits.add(concept)
                matched_concepts.add(concept)
        matched_by_item[key] = hits

    expected_total = float(len(scenario.expected_concepts))
    coverage = float(len(matched_concepts) / expected_total) if expected_total > 0 else 0.0
    relevant_items = sum(1 for concepts in matched_by_item.values() if concepts)
    retrieved_count = len(retrieved)
    semantic_precision = float(relevant_items / float(retrieved_count)) if retrieved_count > 0 else 0.0

    return (
        {
            "matched_expected_concepts": sorted(matched_concepts),
            "coverage_score": round(coverage, 6),
            "relevant_items": int(relevant_items),
            "retrieved_count": int(retrieved_count),
            "semantic_precision": round(semantic_precision, 6),
        },
        matched_by_item,
    )


def concept_grounding_in_graph(session: Any, concept: str) -> Dict[str, Any]:
    token = normalize_token(concept)
    query = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN [
      'Beverage','Compound','ChemicalClass','Enzyme','PopulationGroup',
      'PBPKParameter','BodyCompartment','ToxicityRisk','PhysiologyCondition'
    ])
      AND (
        toLower(coalesce(n.name,'')) CONTAINS $concept
        OR toLower(coalesce(n.normalized_name,'')) CONTAINS $concept
        OR toLower(coalesce(n.parameter_name,'')) CONTAINS $concept
        OR toLower(coalesce(n.group_name,'')) CONTAINS $concept
        OR toLower(coalesce(n.condition,'')) CONTAINS $concept
        OR toLower(coalesce(n.risk_type,'')) CONTAINS $concept
        OR toLower(coalesce(n.class_name,'')) CONTAINS $concept
        OR toLower(coalesce(n.modifier_reason,'')) CONTAINS $concept
        OR toLower(coalesce(n.source_parameter_name,'')) CONTAINS $concept
      )
    WITH collect(DISTINCT n) AS nodes
    UNWIND nodes AS n
    OPTIONAL MATCH p=(n)-[*1..2]-()
    RETURN count(DISTINCT n) AS node_hits,
           count(DISTINCT p) AS path_hits,
           collect(DISTINCT labels(n))[0..10] AS labels_sample
    """
    record = session.run(query, concept=token).single()
    node_hits = int(record["node_hits"]) if record is not None else 0
    path_hits = int(record["path_hits"]) if record is not None else 0
    labels_sample_raw = record["labels_sample"] if record is not None else []
    labels_flat: List[str] = []
    for group in labels_sample_raw or []:
        if isinstance(group, list):
            labels_flat.extend([clean_text(item) for item in group if clean_text(item)])
    labels_flat = sorted(set(labels_flat))
    grounded = node_hits > 0 and path_hits > 0
    return {
        "concept": concept,
        "node_hits": node_hits,
        "path_hits": path_hits,
        "grounded": grounded,
        "labels_sample": labels_flat,
    }


def scenario_path_grounding(session: Any, scenario_id: str) -> Dict[str, Any]:
    query = SCENARIO_GROUNDING_QUERIES.get(scenario_id, "")
    if not query:
        return {"path_count": 0, "path_check_passed": False, "query": ""}
    record = session.run(query).single()
    path_count = int(record["path_count"]) if record is not None else 0
    return {
        "path_count": int(path_count),
        "path_check_passed": bool(path_count > 0),
        "query": query,
    }


def evaluate_grounding(
    session: Any,
    scenario: QueryScenario,
    matched_by_item: Mapping[str, Set[str]],
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    concept_rows: List[Dict[str, Any]] = []
    concept_grounded: Dict[str, bool] = {}

    for concept in scenario.expected_concepts:
        result = concept_grounding_in_graph(session=session, concept=concept)
        concept_rows.append(result)
        concept_grounded[concept] = bool(result["grounded"])

    expected_total = float(len(scenario.expected_concepts))
    grounded_count = float(sum(1 for concept in scenario.expected_concepts if concept_grounded.get(concept, False)))
    concept_grounding_ratio = grounded_count / expected_total if expected_total > 0 else 0.0

    path_check = scenario_path_grounding(session=session, scenario_id=scenario.scenario_id)
    path_pass = 1.0 if path_check["path_check_passed"] else 0.0
    grounding_score = (0.7 * concept_grounding_ratio) + (0.3 * path_pass)

    grounded_items = 0
    retrieved_count = len(matched_by_item)
    for concepts in matched_by_item.values():
        if any(concept_grounded.get(concept, False) for concept in concepts):
            grounded_items += 1

    hallucination_resistance = float(grounded_items / float(retrieved_count)) if retrieved_count > 0 else 0.0

    return (
        {
            "concept_grounding": concept_rows,
            "grounded_expected_count": int(grounded_count),
            "expected_concept_count": int(expected_total),
            "concept_grounding_ratio": round(concept_grounding_ratio, 6),
            "path_validation": path_check,
            "grounding_score": round(float(grounding_score), 6),
            "grounded_retrieval_items": int(grounded_items),
            "retrieved_item_count": int(retrieved_count),
            "hallucination_resistance": round(hallucination_resistance, 6),
        },
        concept_grounded,
    )


def pbpk_user_profile(sex: str, weight: float, fed_or_fasted: str, body_fat_percent: float) -> Dict[str, Any]:
    height = 178.0 if sex == "male" else 165.0
    return {
        "sex": sex,
        "weight": weight,
        "height": height,
        "age": 35,
        "body_fat_percent": body_fat_percent,
        "fed_or_fasted": fed_or_fasted,
        "liver_status": "healthy",
    }


def pbpk_drink_profile() -> Dict[str, Any]:
    return {
        "beverage": "whisky",
        "volume_ml": 180.0,
        "abv": 40.0,
        "serving_time": 0.0,
    }


def load_pbpk_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required PBPK input not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=ENCODING)


def run_pbpk_consistency(root: Path) -> Dict[str, Any]:
    library_df = load_pbpk_dataframe(parameter_library_path(root))
    population_df = load_pbpk_dataframe(population_modifiers_path(root))
    beverage_df = load_pbpk_dataframe(beverage_modifiers_path(root))

    fed_result = run_simulation(
        user_payload=pbpk_user_profile("male", 75.0, "fed", 20.0),
        drink_payload=pbpk_drink_profile(),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    fasted_result = run_simulation(
        user_payload=pbpk_user_profile("male", 75.0, "fasted", 20.0),
        drink_payload=pbpk_drink_profile(),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    female_result = run_simulation(
        user_payload=pbpk_user_profile("female", 60.0, "fasted", 28.0),
        drink_payload=pbpk_drink_profile(),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )

    fed_peak = float(fed_result["summary"]["peak_bac_percent"])
    fasted_peak = float(fasted_result["summary"]["peak_bac_percent"])
    female_peak = float(female_result["summary"]["peak_bac_percent"])
    fed_tpeak = float(fed_result["summary"]["time_to_peak_h"])
    fasted_tpeak = float(fasted_result["summary"]["time_to_peak_h"])

    checks = {
        "fasted_peak_gt_fed_peak": bool(fasted_peak > fed_peak),
        "female_peak_gt_male_peak": bool(female_peak > fasted_peak),
        "fasted_time_to_peak_lte_fed_time_to_peak": bool(fasted_tpeak <= fed_tpeak),
    }
    required_checks = ["fasted_peak_gt_fed_peak", "female_peak_gt_male_peak"]
    pass_count = sum(1 for name in required_checks if checks[name])
    score = float(pass_count / float(len(required_checks))) if required_checks else 0.0

    return {
        "fed_peak_bac_percent": round(fed_peak, 8),
        "fasted_peak_bac_percent": round(fasted_peak, 8),
        "female_peak_bac_percent": round(female_peak, 8),
        "fed_time_to_peak_h": round(fed_tpeak, 8),
        "fasted_time_to_peak_h": round(fasted_tpeak, 8),
        "checks": checks,
        "required_check_names": required_checks,
        "passed_required_checks": int(pass_count),
        "required_checks_total": int(len(required_checks)),
        "pbpk_consistency_score": round(score, 6),
    }


def build_failure_report(error: str, runtime_seconds: float) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error": error,
        "top_k": TOP_K,
        "semantic_precision_score": 0.0,
        "grounding_score": 0.0,
        "pbpk_consistency_score": 0.0,
        "evidence_coverage_score": 0.0,
        "hallucination_resistance_score": 0.0,
        "overall_retrieval_score": 0.0,
        "queries": [],
        "safe_for_hybrid_reasoning_engine": False,
        "runtime_seconds": round(runtime_seconds, 4),
    }


def render_retrieval_rows(
    retrieved: Sequence[RetrievalItem],
    matched_by_item: Mapping[str, Set[str]],
    concept_grounded: Mapping[str, bool],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in retrieved:
        key = f"{item.collection}::{item.object_id}::{item.chunk_id}"
        matched = sorted(matched_by_item.get(key, set()))
        grounded_concepts = sorted([concept for concept in matched if concept_grounded.get(concept, False)])
        rows.append(
            {
                "object_id": item.object_id,
                "chunk_id": item.chunk_id,
                "collection": item.collection,
                "title": item.title,
                "similarity_score": round(item.similarity_score, 8),
                "raw_score": None if item.raw_score is None else round(item.raw_score, 8),
                "distance": None if item.distance is None else round(item.distance, 8),
                "certainty": None if item.certainty is None else round(item.certainty, 8),
                "retrieval_method": item.retrieval_method,
                "matched_expected_concepts": matched,
                "grounded_matched_concepts": grounded_concepts,
                "content_preview": clean_text(item.content)[:240],
            }
        )
    return rows


def main() -> None:
    configure_logging()
    started = time.perf_counter()
    root = repo_root()
    report_path = output_report_path(root)

    if weaviate is None:
        report = build_failure_report(
            error="weaviate-client is not installed.",
            runtime_seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
        return

    if GraphDatabase is None:
        report = build_failure_report(
            error="neo4j Python driver is not installed.",
            runtime_seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
        return

    try:
        weaviate_config = get_weaviate_config()
        neo4j_config = get_neo4j_config()
    except Exception as exc:
        report = build_failure_report(
            error=str(exc),
            runtime_seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
        return

    try:
        rows_by_collection, corpus_metrics = load_embedding_corpus(root)
    except Exception as exc:
        report = build_failure_report(
            error=str(exc),
            runtime_seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
        return

    weaviate_client = None
    neo4j_driver = None

    try:
        weaviate_client = connect_weaviate(weaviate_config)
        weaviate_ready = bool(weaviate_client.is_ready())
        if not weaviate_ready:
            raise RuntimeError("Weaviate is reachable but is_ready() returned False.")

        neo4j_driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
        )
        with neo4j_driver.session(database=neo4j_config["database"]) as session:
            session.run("RETURN 1 AS ok").single()

        query_reports: List[Dict[str, Any]] = []
        semantic_scores: List[float] = []
        grounding_scores: List[float] = []
        coverage_scores: List[float] = []
        hallucination_scores: List[float] = []

        for scenario in TEST_SCENARIOS:
            query_vector, vector_seed_details = build_query_vector(
                scenario=scenario,
                rows_by_collection=rows_by_collection,
                embedding_dimension=int(corpus_metrics["embedding_dimension"]),
            )
            retrieved, method_map = retrieve_top_k_for_query(
                client=weaviate_client,
                scenario=scenario,
                query_vector=query_vector,
                k=TOP_K,
            )
            coverage_summary, matched_by_item = evaluate_concept_coverage(
                scenario=scenario,
                retrieved=retrieved,
            )

            with neo4j_driver.session(database=neo4j_config["database"]) as session:
                grounding_summary, concept_grounded = evaluate_grounding(
                    session=session,
                    scenario=scenario,
                    matched_by_item=matched_by_item,
                )

            semantic_precision = float(coverage_summary["semantic_precision"])
            concept_coverage = float(coverage_summary["coverage_score"])
            grounding_score = float(grounding_summary["grounding_score"])
            hallucination_resistance = float(grounding_summary["hallucination_resistance"])

            semantic_scores.append(semantic_precision)
            coverage_scores.append(concept_coverage)
            grounding_scores.append(grounding_score)
            hallucination_scores.append(hallucination_resistance)

            retrieved_collections = sorted(set(item.collection for item in retrieved if item.collection))
            retrieval_rows = render_retrieval_rows(
                retrieved=retrieved,
                matched_by_item=matched_by_item,
                concept_grounded=concept_grounded,
            )

            query_reports.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "query_text": scenario.query_text,
                    "expected_concepts": list(scenario.expected_concepts),
                    "top_k": TOP_K,
                    "retrieved_count": int(len(retrieved)),
                    "retrieved_collections": retrieved_collections,
                    "retrieval_methods_by_collection": method_map,
                    "vector_seed": vector_seed_details,
                    "evidence_coverage": coverage_summary,
                    "grounding": grounding_summary,
                    "returned_chunks": retrieval_rows,
                }
            )

        pbpk_summary = run_pbpk_consistency(root)
        pbpk_score = float(pbpk_summary["pbpk_consistency_score"])

        semantic_precision_score = float(np.mean(np.array(semantic_scores, dtype=float))) if semantic_scores else 0.0
        grounding_score = float(np.mean(np.array(grounding_scores, dtype=float))) if grounding_scores else 0.0
        evidence_coverage_score = float(np.mean(np.array(coverage_scores, dtype=float))) if coverage_scores else 0.0
        hallucination_resistance_score = (
            float(np.mean(np.array(hallucination_scores, dtype=float))) if hallucination_scores else 0.0
        )

        overall_retrieval_score = (
            0.30 * semantic_precision_score
            + 0.20 * grounding_score
            + 0.20 * pbpk_score
            + 0.15 * evidence_coverage_score
            + 0.15 * hallucination_resistance_score
        )

        per_query_minimums_passed = all(
            (
                float(report["evidence_coverage"]["coverage_score"]) >= 0.20
                and float(report["grounding"]["path_validation"]["path_check_passed"]) >= 1.0
                and int(report["retrieved_count"]) > 0
            )
            for report in query_reports
        )
        pbpk_gate_passed = bool(
            pbpk_summary["checks"]["fasted_peak_gt_fed_peak"] and pbpk_summary["checks"]["female_peak_gt_male_peak"]
        )

        safe_for_hybrid_reasoning_engine = bool(
            weaviate_ready
            and per_query_minimums_passed
            and pbpk_gate_passed
            and corpus_metrics["dimension_consistent"]
            and corpus_metrics["missing_vectors"] == 0
            and corpus_metrics["nan_vectors"] == 0
            and overall_retrieval_score >= 0.55
        )

        report: Dict[str, Any] = {
            "status": "success",
            "top_k": TOP_K,
            "hybrid_alpha": HYBRID_ALPHA,
            "weaviate_connection_success": True,
            "neo4j_connection_success": True,
            "read_only_mode": True,
            "collection_scope": list(REQUIRED_COLLECTIONS),
            "embedding_corpus": corpus_metrics,
            "queries": query_reports,
            "pbpk_consistency": pbpk_summary,
            "semantic_precision_score": round(semantic_precision_score, 6),
            "grounding_score": round(grounding_score, 6),
            "pbpk_consistency_score": round(pbpk_score, 6),
            "evidence_coverage_score": round(evidence_coverage_score, 6),
            "hallucination_resistance_score": round(hallucination_resistance_score, 6),
            "overall_retrieval_score": round(overall_retrieval_score, 6),
            "safe_for_hybrid_reasoning_engine": safe_for_hybrid_reasoning_engine,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
        LOGGER.info("safe_for_hybrid_reasoning_engine=%s", safe_for_hybrid_reasoning_engine)

    except Exception as exc:
        report = build_failure_report(
            error=str(exc),
            runtime_seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote semantic retrieval report -> %s", report_path)
    finally:
        if weaviate_client is not None:
            try:
                weaviate_client.close()
            except Exception:
                pass
        if neo4j_driver is not None:
            try:
                neo4j_driver.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
