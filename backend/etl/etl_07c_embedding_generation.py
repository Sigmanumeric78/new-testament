"""ETL step 07c: deterministic embedding generation for Weaviate retrieval objects."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    SentenceTransformer = None

LOGGER = logging.getLogger("etl_07c_embedding_generation")

ENCODING = "utf-8"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE = 64

COLLECTION_FILE_MAP: Tuple[Tuple[str, str, str], ...] = (
    ("BeverageKnowledge", "beverage_knowledge.jsonl", "beverage_embeddings.parquet"),
    ("CompoundKnowledge", "compound_knowledge.jsonl", "compound_embeddings.parquet"),
    ("MetabolismKnowledge", "metabolism_knowledge.jsonl", "metabolism_embeddings.parquet"),
    ("PBPKKnowledge", "pbpk_knowledge.jsonl", "pbpk_embeddings.parquet"),
    ("ToxicityKnowledge", "toxicity_knowledge.jsonl", "toxicity_embeddings.parquet"),
    ("PopulationKnowledge", "population_knowledge.jsonl", "population_embeddings.parquet"),
    ("ScientificEvidence", "scientific_evidence.jsonl", "scientific_evidence_embeddings.parquet"),
)

METADATA_EMBED_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "BeverageKnowledge": (
        "beverage_id",
        "beverage_name",
        "category",
        "compound_count",
        "compound_classes",
        "chemical_categories",
        "modifier_count",
    ),
    "CompoundKnowledge": (
        "normalized_compound_name",
        "compound_name",
        "pubchem_cids",
        "chemical_categories",
        "compound_roles",
        "beverage_count",
    ),
    "MetabolismKnowledge": (
        "parameter_id",
        "parameter_name",
        "domain",
        "population_group",
        "condition",
        "effect_direction",
        "value",
        "unit",
    ),
    "PBPKKnowledge": (
        "parameter_id",
        "parameter_name",
        "compartment",
        "base_value",
        "unit",
        "population_group",
        "modifier",
    ),
    "ToxicityKnowledge": (
        "modifier_id",
        "beverage_id",
        "beverage_name",
        "category",
        "risk_type",
        "parameter_name",
        "modifier",
        "trigger_compounds",
    ),
    "PopulationKnowledge": (
        "population_group",
        "domain_count",
        "domains",
        "population_modifier_count",
        "pbpk_parameter_count",
        "effect_directions",
    ),
    "ScientificEvidence": (
        "evidence_type",
        "parameter_id",
        "parameter_name",
        "domain",
        "condition",
        "compound_name",
        "beverage_id",
    ),
}

ABBREVIATIONS: Tuple[str, ...] = ("ADH", "ALDH", "CYP2E1", "BAC", "PBPK")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_dir(root: Path) -> Path:
    return root / "data" / "processed" / "weaviate"


def output_dir(root: Path) -> Path:
    path = root / "data" / "processed" / "weaviate" / "embedded"
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_output_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "embedding_generation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL 07c deterministic embedding generation")
    parser.add_argument(
        "--collections",
        type=str,
        default="all",
        help="Comma-separated collection names to process, or 'all'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Fixed deterministic batch size for model encoding.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Deterministic row offset applied per selected collection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Deterministic row limit applied per selected collection; 0 means all remaining rows.",
    )
    return parser.parse_args()


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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preserve_abbreviations_after_lower(text: str) -> str:
    output = text
    for token in ABBREVIATIONS:
        output = re.sub(rf"\b{token.lower()}\b", token, output)
    return output


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", clean_text(text))
    normalized = normalize_whitespace(normalized)
    normalized = normalized.lower()
    normalized = preserve_abbreviations_after_lower(normalized)
    return normalized


def normalize_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        items = sorted({clean_text(item) for item in value if clean_text(item)})
        if not items:
            return "unknown"
        return ", ".join(items)
    text = clean_text(value)
    return text if text else "unknown"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input JSONL: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding=ENCODING) as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    rows.sort(key=lambda item: clean_text(item.get("chunk_id")))
    return rows


def build_embedding_text(obj: Mapping[str, Any]) -> str:
    collection = clean_text(obj.get("collection"))
    title = clean_text(obj.get("title"))
    content = clean_text(obj.get("content"))
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    selected_fields = METADATA_EMBED_FIELDS.get(collection, tuple())
    metadata_lines: List[str] = []
    for field in selected_fields:
        value = normalize_metadata_value(metadata.get(field))
        metadata_lines.append(f"{field}: {value}")
    parts = [
        f"title: {title or 'unknown'}",
        f"collection: {collection or 'unknown'}",
        "metadata: " + " | ".join(metadata_lines) if metadata_lines else "metadata: unknown",
        f"content: {content or 'unknown'}",
    ]
    combined = "\n".join(parts)
    return normalize_text(combined)


def chunked_indices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        yield start, end
        start = end


def encode_texts(
    model: Any,
    texts: Sequence[str],
    batch_size: int,
) -> Tuple[List[List[float]], List[int]]:
    vectors: List[List[float]] = [list() for _ in texts]
    failed_indices: List[int] = []
    for start, end in chunked_indices(len(texts), batch_size):
        batch = texts[start:end]
        try:
            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            for local_idx, vec in enumerate(embeddings):
                vectors[start + local_idx] = [float(x) for x in vec.tolist()]
        except Exception:
            for local_idx, text in enumerate(batch):
                absolute_idx = start + local_idx
                try:
                    vec = model.encode(
                        [text],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )[0]
                    vectors[absolute_idx] = [float(x) for x in vec.tolist()]
                except Exception:
                    failed_indices.append(absolute_idx)
    return vectors, failed_indices


def vector_has_nan(vector: Sequence[float]) -> bool:
    for value in vector:
        if math.isnan(float(value)):
            return True
    return False


def write_collection_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["chunk_id", "object_id"], kind="mergesort").reset_index(drop=True)
    df.to_parquet(path, index=False)


def merge_with_existing_parquet(path: Path, new_rows: Sequence[Mapping[str, Any]]) -> int:
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["chunk_id"], keep="last")
        combined = combined.sort_values(by=["chunk_id", "object_id"], kind="mergesort").reset_index(drop=True)
    else:
        combined = new_df.sort_values(by=["chunk_id", "object_id"], kind="mergesort").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return int(len(combined))


def zero_embeddings_per_collection() -> Dict[str, int]:
    return {collection: 0 for collection, _, _ in COLLECTION_FILE_MAP}


def build_failed_report(
    error: str,
    model_load_success: bool,
    runtime_seconds: float,
    batch_size: int,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "model_name": MODEL_NAME,
        "model_load_success": model_load_success,
        "batch_size": int(batch_size),
        "total_embeddings_generated": 0,
        "embeddings_per_collection": zero_embeddings_per_collection(),
        "embedding_dimension": 0,
        "failed_embeddings": 0,
        "missing_embeddings": 0,
        "nan_vectors": 0,
        "duplicate_chunk_ids": 0,
        "empty_content_rows": 0,
        "dimension_mismatch_rows": 0,
        "runtime_seconds": runtime_seconds,
        "error": error,
        "safe_for_weaviate_schema_init": False,
    }


def selected_collection_specs(selection: str) -> List[Tuple[str, str, str]]:
    if clean_text(selection).lower() in {"", "all"}:
        return list(COLLECTION_FILE_MAP)
    requested = {clean_text(item) for item in selection.split(",") if clean_text(item)}
    specs: List[Tuple[str, str, str]] = []
    for spec in COLLECTION_FILE_MAP:
        if spec[0] in requested:
            specs.append(spec)
    unknown = sorted(requested - {item[0] for item in COLLECTION_FILE_MAP})
    if unknown:
        raise ValueError(f"Unknown collection selection values: {', '.join(unknown)}")
    return specs


def count_existing_embeddings(out_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for collection, _, output_file in COLLECTION_FILE_MAP:
        path = out_dir / output_file
        if not path.exists():
            counts[collection] = 0
            continue
        try:
            df = pd.read_parquet(path)
            counts[collection] = int(len(df))
        except Exception:
            counts[collection] = 0
    return counts


def main() -> None:
    configure_logging()
    started = time.perf_counter()
    args = parse_args()
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    offset = max(0, int(args.offset))
    limit = int(args.limit)
    if limit < 0:
        raise ValueError("limit must be >= 0")

    root = repo_root()
    in_dir = input_dir(root)
    out_dir = output_dir(root)
    report_path = report_output_path(root)
    try:
        active_specs = selected_collection_specs(args.collections)
    except Exception as exc:
        runtime = round(time.perf_counter() - started, 4)
        report = build_failed_report(
            error=str(exc),
            model_load_success=False,
            runtime_seconds=runtime,
            batch_size=batch_size,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote embedding generation report -> %s", report_path)
        return

    if SentenceTransformer is None:
        runtime = round(time.perf_counter() - started, 4)
        report = build_failed_report(
            error="sentence-transformers is not available in this environment.",
            model_load_success=False,
            runtime_seconds=runtime,
            batch_size=batch_size,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote embedding generation report -> %s", report_path)
        return

    model_load_success = False
    model: Any = None
    model_dimension = 0
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        model_load_success = True
        model_dimension = int(model.get_sentence_embedding_dimension() or 0)
    except Exception as exc:
        runtime = round(time.perf_counter() - started, 4)
        report = build_failed_report(
            error=f"Model load failed: {exc}",
            model_load_success=False,
            runtime_seconds=runtime,
            batch_size=batch_size,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote embedding generation report -> %s", report_path)
        return

    all_rows_written: Dict[str, int] = {}
    failed_embeddings = 0
    nan_vectors = 0
    empty_content_rows = 0
    duplicate_chunk_ids = 0
    dimension_mismatch_rows = 0
    missing_embeddings = 0
    total_embeddings_generated = 0

    try:
        for collection, input_file, output_file in active_specs:
            objects = load_jsonl(in_dir / input_file)
            if offset > 0 or limit > 0:
                end_idx = None if limit == 0 else offset + limit
                objects = objects[offset:end_idx]
            chunk_ids = [clean_text(obj.get("chunk_id")) for obj in objects]
            duplicate_chunk_ids += int(pd.Series(chunk_ids).duplicated(keep=False).sum()) if chunk_ids else 0

            texts: List[str] = []
            prepared_rows: List[Dict[str, Any]] = []
            for obj in objects:
                content = clean_text(obj.get("content"))
                if not content:
                    empty_content_rows += 1
                embedding_text = build_embedding_text(obj)
                if not embedding_text:
                    empty_content_rows += 1
                texts.append(embedding_text)
                prepared_rows.append(
                    {
                        "object_id": clean_text(obj.get("object_id")),
                        "chunk_id": clean_text(obj.get("chunk_id")),
                        "collection": clean_text(obj.get("collection")),
                        "title": clean_text(obj.get("title")),
                        "content": content,
                        "metadata": json.dumps(obj.get("metadata", {}), sort_keys=True, ensure_ascii=True),
                        "provenance": json.dumps(obj.get("provenance", {}), sort_keys=True, ensure_ascii=True),
                    }
                )

            vectors, failed_indices = encode_texts(model=model, texts=texts, batch_size=batch_size)
            failed_embeddings += len(failed_indices)
            failed_set = set(failed_indices)

            output_rows: List[Dict[str, Any]] = []
            for idx, row in enumerate(prepared_rows):
                vector = vectors[idx]
                if idx in failed_set or not vector:
                    missing_embeddings += 1
                    continue
                if vector_has_nan(vector):
                    nan_vectors += 1
                if model_dimension > 0 and len(vector) != model_dimension:
                    dimension_mismatch_rows += 1
                output_rows.append(
                    {
                        "object_id": row["object_id"],
                        "chunk_id": row["chunk_id"],
                        "collection": row["collection"],
                        "title": row["title"],
                        "content": row["content"],
                        "embedding": vector,
                        "metadata": row["metadata"],
                        "provenance": row["provenance"],
                    }
                )

            merged_count = merge_with_existing_parquet(out_dir / output_file, output_rows)
            all_rows_written[collection] = merged_count
            total_embeddings_generated += len(output_rows)
    except Exception as exc:
        runtime = round(time.perf_counter() - started, 4)
        report = build_failed_report(
            error=str(exc),
            model_load_success=model_load_success,
            runtime_seconds=runtime,
            batch_size=batch_size,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote embedding generation report -> %s", report_path)
        return

    runtime_seconds = round(time.perf_counter() - started, 4)
    existing_counts = count_existing_embeddings(out_dir)
    embeddings_per_collection = {
        collection: int(existing_counts.get(collection, all_rows_written.get(collection, 0)))
        for collection, _, _ in COLLECTION_FILE_MAP
    }
    total_embeddings_generated = int(sum(embeddings_per_collection.values()))

    safe_for_weaviate_schema_init = (
        model_load_success
        and total_embeddings_generated > 0
        and failed_embeddings == 0
        and missing_embeddings == 0
        and nan_vectors == 0
        and duplicate_chunk_ids == 0
        and empty_content_rows == 0
        and dimension_mismatch_rows == 0
        and model_dimension > 0
    )

    report: Dict[str, Any] = {
        "status": "success",
        "model_name": MODEL_NAME,
        "model_load_success": model_load_success,
        "batch_size": int(batch_size),
        "total_embeddings_generated": int(total_embeddings_generated),
        "embeddings_per_collection": embeddings_per_collection,
        "embedding_dimension": int(model_dimension),
        "failed_embeddings": int(failed_embeddings),
        "missing_embeddings": int(missing_embeddings),
        "nan_vectors": int(nan_vectors),
        "duplicate_chunk_ids": int(duplicate_chunk_ids),
        "empty_content_rows": int(empty_content_rows),
        "dimension_mismatch_rows": int(dimension_mismatch_rows),
        "runtime_seconds": runtime_seconds,
        "artifacts": {
            collection: str((out_dir / output_file).relative_to(root))
            for collection, _, output_file in COLLECTION_FILE_MAP
        },
        "active_collections_this_run": [item[0] for item in active_specs],
        "safe_for_weaviate_schema_init": safe_for_weaviate_schema_init,
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
    LOGGER.info("Wrote embedding generation report -> %s", report_path)
    LOGGER.info("safe_for_weaviate_schema_init=%s", safe_for_weaviate_schema_init)


if __name__ == "__main__":
    main()
