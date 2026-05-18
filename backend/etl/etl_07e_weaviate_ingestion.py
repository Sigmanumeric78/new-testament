"""ETL step 07e: deterministic external-vector ingestion into local Weaviate."""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd

try:
    import weaviate  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    weaviate = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import get_weaviate_config

LOGGER = logging.getLogger("etl_07e_weaviate_ingestion")

ENCODING = "utf-8"
BATCH_SIZE = 100

COLLECTION_FILES: Tuple[Tuple[str, str], ...] = (
    ("BeverageKnowledge", "beverage_embeddings.parquet"),
    ("CompoundKnowledge", "compound_embeddings.parquet"),
    ("MetabolismKnowledge", "metabolism_embeddings.parquet"),
    ("PBPKKnowledge", "pbpk_embeddings.parquet"),
    ("ToxicityKnowledge", "toxicity_embeddings.parquet"),
    ("PopulationKnowledge", "population_embeddings.parquet"),
    ("ScientificEvidence", "scientific_evidence_embeddings.parquet"),
)

EXPECTED_COUNTS: Mapping[str, int] = {
    "BeverageKnowledge": 990,
    "CompoundKnowledge": 123,
    "MetabolismKnowledge": 44,
    "PBPKKnowledge": 13,
    "ToxicityKnowledge": 502,
    "PopulationKnowledge": 10,
    "ScientificEvidence": 11774,
}

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
class PreparedObject:
    object_id: str
    chunk_id: str
    collection: str
    weaviate_uuid: str
    properties: Dict[str, Any]
    vector: List[float]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Avoid emitting one line per HTTP request during large deterministic ingest runs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def embeddings_dir(root: Path) -> Path:
    return root / "data" / "processed" / "weaviate" / "embedded"


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "weaviate_ingestion_report.json"
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


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        text = clean_text(value)
        if not text:
            return 0.0
        try:
            return float(text)
        except Exception:
            return 0.0


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


def parse_vector(value: Any) -> Tuple[List[float], bool]:
    data: Any = value
    if isinstance(data, str):
        text = clean_text(data)
        if not text:
            return [], True
        try:
            data = json.loads(text)
        except Exception:
            return [], True
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, (list, tuple)):
        return [], True
    vector: List[float] = []
    has_nan = False
    for item in data:
        try:
            number = float(item)
        except Exception:
            return [], True
        if math.isnan(number):
            has_nan = True
        vector.append(number)
    return vector, has_nan


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


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    start = 0
    while start < len(seq):
        end = min(len(seq), start + size)
        yield seq[start:end]
        start = end


def collection_count(collection_obj: Any) -> int:
    result = collection_obj.aggregate.over_all(total_count=True)
    return int(getattr(result, "total_count", 0) or 0)


def prepare_objects_for_collection(
    collection: str,
    df: pd.DataFrame,
) -> Tuple[List[PreparedObject], Dict[str, int], int]:
    metrics = {
        "missing_vectors": 0,
        "nan_vectors": 0,
        "dimension_mismatch_rows": 0,
        "empty_content_rows": 0,
        "duplicate_uuid_count": 0,
    }
    prepared: List[PreparedObject] = []
    seen_uuids: set[str] = set()
    expected_dim = 0

    ordered = df.sort_values(by=["chunk_id", "object_id"], kind="mergesort").reset_index(drop=True)
    for _, row in ordered.iterrows():
        object_id = clean_text(row.get("object_id"))
        if not object_id:
            metrics["missing_vectors"] += 1
            continue
        chunk_id = clean_text(row.get("chunk_id"))
        title = clean_text(row.get("title"))
        content = clean_text(row.get("content"))
        if not content:
            metrics["empty_content_rows"] += 1

        vector, has_nan = parse_vector(row.get("embedding"))
        if not vector:
            metrics["missing_vectors"] += 1
            continue
        if has_nan:
            metrics["nan_vectors"] += 1
            continue
        if expected_dim == 0:
            expected_dim = len(vector)
        if len(vector) != expected_dim:
            metrics["dimension_mismatch_rows"] += 1
            continue

        deterministic_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, object_id))
        if deterministic_uuid in seen_uuids:
            metrics["duplicate_uuid_count"] += 1
        seen_uuids.add(deterministic_uuid)

        properties = {
            "object_id": object_id,
            "chunk_id": chunk_id,
            "title": title,
            "content": content,
            "collection": clean_text(row.get("collection")) or collection,
            "confidence_score": parse_float(row.get("confidence_score")),
            "source_dataset": clean_text(row.get("source_dataset")),
            "source_file": clean_text(row.get("source_file")),
            "metadata": stringify_json_field(row.get("metadata")),
            "provenance": stringify_json_field(row.get("provenance")),
        }
        prepared.append(
            PreparedObject(
                object_id=object_id,
                chunk_id=chunk_id,
                collection=collection,
                weaviate_uuid=deterministic_uuid,
                properties=properties,
                vector=vector,
            )
        )

    return prepared, metrics, expected_dim


def build_failure_report(
    error: str,
    runtime_seconds: float,
    connection_successful: bool,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "connection_success": connection_successful,
        "total_objects_uploaded": 0,
        "uploaded_per_collection": {name: 0 for name, _ in COLLECTION_FILES},
        "matched_existing_objects": 0,
        "matched_existing_per_collection": {name: 0 for name, _ in COLLECTION_FILES},
        "failed_uploads": 0,
        "missing_vectors": 0,
        "nan_vectors": 0,
        "duplicate_uuid_count": 0,
        "embedding_dimension": 0,
        "runtime_seconds": round(runtime_seconds, 4),
        "safe_for_semantic_retrieval_testing": False,
        "error": error,
    }


def main() -> None:
    configure_logging()
    started = time.perf_counter()
    root = repo_root()
    report_path = output_report_path(root)

    if weaviate is None:
        report = build_failure_report(
            error="weaviate-client is not installed.",
            runtime_seconds=time.perf_counter() - started,
            connection_successful=False,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
        return

    try:
        config = get_weaviate_config()
    except Exception as exc:
        report = build_failure_report(
            error=str(exc),
            runtime_seconds=time.perf_counter() - started,
            connection_successful=False,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
        return

    embed_dir = embeddings_dir(root)
    for _, filename in COLLECTION_FILES:
        file_path = embed_dir / filename
        if not file_path.exists():
            report = build_failure_report(
                error=f"Missing embedding parquet: {file_path}",
                runtime_seconds=time.perf_counter() - started,
                connection_successful=False,
            )
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
            LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
            return

    connection_success = False
    uploaded_per_collection: Dict[str, int] = {name: 0 for name, _ in COLLECTION_FILES}
    matched_existing_per_collection: Dict[str, int] = {name: 0 for name, _ in COLLECTION_FILES}
    failed_per_collection: Dict[str, int] = {name: 0 for name, _ in COLLECTION_FILES}
    collection_counts: Dict[str, int] = {}
    expected_count_mismatch: Dict[str, Dict[str, int]] = {}
    duplicate_uuid_count = 0
    missing_vectors = 0
    nan_vectors = 0
    dimension_mismatch_rows = 0
    empty_content_rows = 0
    embedding_dimensions: set[int] = set()

    client = None
    try:
        client = connect_weaviate(config)
        connection_success = bool(client.is_ready())
        if not connection_success:
            report = build_failure_report(
                error="Weaviate connection initialized but is_ready() returned False.",
                runtime_seconds=time.perf_counter() - started,
                connection_successful=False,
            )
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
            LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
            return

        for collection_name, filename in COLLECTION_FILES:
            if not client.collections.exists(collection_name):
                raise RuntimeError(
                    f"Collection missing in Weaviate schema: {collection_name}. Run etl_07d_weaviate_schema_init.py first."
                )
            collection = client.collections.get(collection_name)
            df = pd.read_parquet(embed_dir / filename)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{filename} missing required columns: {', '.join(missing_cols)}")

            prepared, per_metrics, dimension = prepare_objects_for_collection(collection_name, df)
            duplicate_uuid_count += per_metrics["duplicate_uuid_count"] if "duplicate_uuid_count" in per_metrics else 0
            missing_vectors += per_metrics["missing_vectors"]
            nan_vectors += per_metrics["nan_vectors"]
            dimension_mismatch_rows += per_metrics["dimension_mismatch_rows"]
            empty_content_rows += per_metrics["empty_content_rows"]
            if dimension > 0:
                embedding_dimensions.add(dimension)

            local_duplicate = len(prepared) - len({item.weaviate_uuid for item in prepared})
            duplicate_uuid_count += local_duplicate

            pending: List[PreparedObject] = []
            for obj in prepared:
                if collection.data.exists(obj.weaviate_uuid):
                    matched_existing_per_collection[collection_name] += 1
                else:
                    pending.append(obj)

            for batch_rows in chunked(pending, BATCH_SIZE):
                with collection.batch.fixed_size(batch_size=BATCH_SIZE, concurrent_requests=1) as batch:
                    for obj in batch_rows:
                        batch.add_object(
                            properties=obj.properties,
                            uuid=obj.weaviate_uuid,
                            vector=obj.vector,
                        )

                unresolved: List[PreparedObject] = []
                for obj in batch_rows:
                    if collection.data.exists(obj.weaviate_uuid):
                        uploaded_per_collection[collection_name] += 1
                    else:
                        unresolved.append(obj)

                for obj in unresolved:
                    inserted = False
                    for _ in range(2):
                        try:
                            collection.data.insert(
                                properties=obj.properties,
                                uuid=obj.weaviate_uuid,
                                vector=obj.vector,
                            )
                        except Exception:
                            pass
                        if collection.data.exists(obj.weaviate_uuid):
                            inserted = True
                            uploaded_per_collection[collection_name] += 1
                            break
                    if not inserted:
                        failed_per_collection[collection_name] += 1

            count = collection_count(collection)
            collection_counts[collection_name] = count
            expected = int(EXPECTED_COUNTS[collection_name])
            if count != expected:
                expected_count_mismatch[collection_name] = {"expected": expected, "actual": count}

        total_objects_uploaded = int(sum(uploaded_per_collection.values()))
        matched_existing_objects = int(sum(matched_existing_per_collection.values()))
        failed_uploads = int(sum(failed_per_collection.values()))
        ingestion_attempted = total_objects_uploaded + failed_uploads
        ingestion_success_rate = (
            round(total_objects_uploaded / float(ingestion_attempted), 6) if ingestion_attempted > 0 else 1.0
        )

        embedding_dimension = 0
        if len(embedding_dimensions) == 1:
            embedding_dimension = int(next(iter(embedding_dimensions)))

        dimension_consistency = len(embedding_dimensions) <= 1 and dimension_mismatch_rows == 0 and embedding_dimension > 0
        rerun_idempotency_passed = failed_uploads == 0 and not expected_count_mismatch

        safe_for_semantic_retrieval_testing = (
            connection_success
            and failed_uploads == 0
            and missing_vectors == 0
            and nan_vectors == 0
            and duplicate_uuid_count == 0
            and dimension_consistency
            and not expected_count_mismatch
            and rerun_idempotency_passed
        )

        report: Dict[str, Any] = {
            "status": "success",
            "connection_success": connection_success,
            "batch_size": BATCH_SIZE,
            "total_objects_uploaded": total_objects_uploaded,
            "uploaded_per_collection": uploaded_per_collection,
            "matched_existing_objects": matched_existing_objects,
            "matched_existing_per_collection": matched_existing_per_collection,
            "failed_uploads": failed_uploads,
            "failed_uploads_per_collection": failed_per_collection,
            "missing_vectors": int(missing_vectors),
            "nan_vectors": int(nan_vectors),
            "duplicate_uuid_count": int(duplicate_uuid_count),
            "embedding_dimension": int(embedding_dimension),
            "dimension_consistency": dimension_consistency,
            "dimension_mismatch_rows": int(dimension_mismatch_rows),
            "empty_content_rows": int(empty_content_rows),
            "collection_counts": collection_counts,
            "expected_counts": dict(EXPECTED_COUNTS),
            "expected_count_mismatch": expected_count_mismatch,
            "ingestion_success_rate": ingestion_success_rate,
            "rerun_idempotency_passed": rerun_idempotency_passed,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "safe_for_semantic_retrieval_testing": safe_for_semantic_retrieval_testing,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
        LOGGER.info("safe_for_semantic_retrieval_testing=%s", safe_for_semantic_retrieval_testing)
    except Exception as exc:
        report = build_failure_report(
            error=str(exc),
            runtime_seconds=time.perf_counter() - started,
            connection_successful=connection_success,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate ingestion report -> %s", report_path)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
