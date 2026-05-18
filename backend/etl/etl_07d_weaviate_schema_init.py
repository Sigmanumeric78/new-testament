"""ETL step 07d: initialize deterministic Weaviate schema (no ingestion)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

try:
    import weaviate  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    weaviate = None

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import get_weaviate_config

LOGGER = logging.getLogger("etl_07d_weaviate_schema_init")

ENCODING = "utf-8"

REQUIRED_COLLECTIONS: Tuple[str, ...] = (
    "BeverageKnowledge",
    "CompoundKnowledge",
    "MetabolismKnowledge",
    "PBPKKnowledge",
    "ToxicityKnowledge",
    "PopulationKnowledge",
    "ScientificEvidence",
)

REQUIRED_PROPERTIES: Tuple[Tuple[str, str], ...] = (
    ("object_id", "text"),
    ("chunk_id", "text"),
    ("title", "text"),
    ("content", "text"),
    ("collection", "text"),
    ("confidence_score", "number"),
    ("source_dataset", "text"),
    ("source_file", "text"),
    ("metadata", "text"),
    ("provenance", "text"),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def schema_design_path(root: Path) -> Path:
    return root / "rag" / "weaviate" / "weaviate_schema_design.md"


def schema_report_path(root: Path) -> Path:
    return root / "data" / "interim" / "weaviate" / "weaviate_schema_report.json"


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "weaviate" / "weaviate_schema_init_report.json"
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


def as_primitive(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): as_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_primitive(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return as_primitive(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return as_primitive(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return as_primitive(
                {
                    k: v
                    for k, v in vars(value).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        except Exception:
            pass
    return clean_text(value)


def parse_weaviate_url(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid WEAVIATE_URL: '{url}'. Expected form like http://localhost:8080")
    secure = parsed.scheme.lower() == "https"
    default_port = 443 if secure else 80
    return {
        "http_host": parsed.hostname,
        "http_port": int(parsed.port or default_port),
        "http_secure": secure,
    }


def load_and_validate_design_inputs(root: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    issues: List[str] = []
    design_file = schema_design_path(root)
    report_file = schema_report_path(root)
    snapshot: Dict[str, Any] = {"design_file_exists": design_file.exists(), "schema_report_exists": report_file.exists()}

    if not design_file.exists():
        issues.append(f"Missing design file: {design_file}")
    if not report_file.exists():
        issues.append(f"Missing schema report file: {report_file}")

    if design_file.exists():
        text = design_file.read_text(encoding=ENCODING)
        missing_in_design = [name for name in REQUIRED_COLLECTIONS if name not in text]
        snapshot["missing_collections_in_design_md"] = missing_in_design
        if missing_in_design:
            issues.append("Design markdown missing required collections: " + ", ".join(missing_in_design))

    if report_file.exists():
        report = json.loads(report_file.read_text(encoding=ENCODING))
        collection_rows = report.get("collections", [])
        observed = sorted(
            [clean_text(item.get("name")) for item in collection_rows if isinstance(item, dict) and clean_text(item.get("name"))]
        )
        missing_in_report = [name for name in REQUIRED_COLLECTIONS if name not in observed]
        snapshot["collections_in_schema_report"] = observed
        snapshot["missing_collections_in_schema_report"] = missing_in_report
        if missing_in_report:
            issues.append("Schema report missing required collections: " + ", ".join(missing_in_report))
    return len(issues) == 0, issues, snapshot


def build_collection_schema_dict(name: str) -> Dict[str, Any]:
    properties = [{"name": prop_name, "dataType": [dtype]} for prop_name, dtype in REQUIRED_PROPERTIES]
    return {
        "class": name,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": properties,
    }


def _build_typed_properties(config_module: Any) -> List[Any]:
    properties: List[Any] = []
    data_type = getattr(config_module, "DataType")
    prop_cls = getattr(config_module, "Property")
    for prop_name, dtype in REQUIRED_PROPERTIES:
        dt = getattr(data_type, "NUMBER") if dtype == "number" else getattr(data_type, "TEXT")
        properties.append(prop_cls(name=prop_name, data_type=dt))
    return properties


def create_collection_explicit(client: Any, collection_name: str) -> None:
    created = False
    try:
        schema_dict = build_collection_schema_dict(collection_name)
        client.collections.create_from_dict(schema_dict)
        created = True
    except Exception:
        created = False

    if created:
        return

    try:
        import weaviate.classes.config as wvcc  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to create collection with create_from_dict and typed API unavailable."
        ) from exc

    properties = _build_typed_properties(wvcc)
    configure = getattr(wvcc, "Configure")

    try:
        vector_config = configure.Vectors.self_provided()
        vector_index_config = configure.VectorIndex.hnsw(distance_metric=wvcc.VectorDistances.COSINE)
        client.collections.create(
            name=collection_name,
            properties=properties,
            vector_config=vector_config,
            vector_index_config=vector_index_config,
        )
        return
    except Exception:
        pass

    # Old client fallback
    vectorizer_config = configure.Vectorizer.none()
    vector_index_config = configure.VectorIndex.hnsw(distance_metric=wvcc.VectorDistances.COSINE)
    client.collections.create(
        name=collection_name,
        properties=properties,
        vectorizer_config=vectorizer_config,
        vector_index_config=vector_index_config,
    )


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
            try:
                from weaviate.auth import AuthApiKey  # type: ignore

                auth_credentials = AuthApiKey(api_key)
            except Exception as exc:
                raise RuntimeError("WEAVIATE_API_KEY provided but auth class is unavailable in client.") from exc

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


def list_collection_names(client: Any) -> List[str]:
    names: List[str] = []
    listing = client.collections.list_all()
    listing = as_primitive(listing)
    if isinstance(listing, dict):
        names = sorted([clean_text(k) for k in listing.keys() if clean_text(k)])
    elif isinstance(listing, list):
        for item in listing:
            if isinstance(item, dict):
                name = clean_text(item.get("name") or item.get("class"))
                if name:
                    names.append(name)
            else:
                name = clean_text(item)
                if name:
                    names.append(name)
        names = sorted(names)
    return names


def extract_property_names(collection_config: Any) -> List[str]:
    cfg = as_primitive(collection_config)
    names: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "properties" in node and isinstance(node["properties"], list):
                for item in node["properties"]:
                    if isinstance(item, dict):
                        name = clean_text(item.get("name"))
                        if name:
                            names.append(name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(cfg)
    return sorted(set(names))


def validate_vectorizer_config(collection_config: Any) -> bool:
    cfg = as_primitive(collection_config)
    text = json.dumps(cfg, sort_keys=True).lower()
    has_external_vectorizer = ("self_provided" in text) or ('"vectorizer": "none"' in text) or ('"vectorizer":"none"' in text)
    has_hnsw = "hnsw" in text
    has_cosine = "cosine" in text
    return bool(has_external_vectorizer and has_hnsw and has_cosine)


def build_failure_report(
    runtime_seconds: float,
    error: str,
    connection_successful: bool,
    created: Sequence[str],
    existing: Sequence[str],
    design_snapshot: Mapping[str, Any],
    design_issues: Sequence[str],
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "connection_successful": connection_successful,
        "collections_created": list(created),
        "collections_existing": list(existing),
        "schema_validation_passed": False,
        "missing_required_properties": {},
        "vectorizer_validation": False,
        "runtime_seconds": round(runtime_seconds, 4),
        "safe_for_weaviate_ingestion": False,
        "design_input_validation": {
            "passed": len(design_issues) == 0,
            "issues": list(design_issues),
            "snapshot": dict(design_snapshot),
        },
        "error": error,
    }


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    root = repo_root()
    report_path = output_report_path(root)

    created: List[str] = []
    existing: List[str] = []
    connection_successful = False
    schema_validation_passed = False
    missing_required_properties: Dict[str, List[str]] = {}
    vectorizer_by_collection: Dict[str, bool] = {}

    design_ok, design_issues, design_snapshot = load_and_validate_design_inputs(root)
    if weaviate is None:
        report = build_failure_report(
            runtime_seconds=time.perf_counter() - start,
            error="weaviate-client is not installed in this environment.",
            connection_successful=False,
            created=created,
            existing=existing,
            design_snapshot=design_snapshot,
            design_issues=design_issues,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate schema init report -> %s", report_path)
        return

    try:
        config = get_weaviate_config()
    except Exception as exc:
        report = build_failure_report(
            runtime_seconds=time.perf_counter() - start,
            error=str(exc),
            connection_successful=False,
            created=created,
            existing=existing,
            design_snapshot=design_snapshot,
            design_issues=design_issues,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate schema init report -> %s", report_path)
        return

    client = None
    try:
        client = connect_weaviate(config)
        connection_successful = bool(client.is_ready())
        if not connection_successful:
            report = build_failure_report(
                runtime_seconds=time.perf_counter() - start,
                error="Weaviate client connected but is_ready() returned False.",
                connection_successful=False,
                created=created,
                existing=existing,
                design_snapshot=design_snapshot,
                design_issues=design_issues,
            )
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
            LOGGER.info("Wrote Weaviate schema init report -> %s", report_path)
            return

        for collection in REQUIRED_COLLECTIONS:
            if client.collections.exists(collection):
                existing.append(collection)
                continue
            create_collection_explicit(client, collection)
            created.append(collection)

        collection_names = list_collection_names(client)
        duplicate_collection_prevention = len(collection_names) == len(set(collection_names))

        for collection in REQUIRED_COLLECTIONS:
            if not client.collections.exists(collection):
                missing_required_properties[collection] = [prop for prop, _ in REQUIRED_PROPERTIES]
                vectorizer_by_collection[collection] = False
                continue

            cfg_obj = client.collections.get(collection).config.get()
            property_names = extract_property_names(cfg_obj)
            missing_props = [prop_name for prop_name, _ in REQUIRED_PROPERTIES if prop_name not in property_names]
            missing_required_properties[collection] = missing_props
            vectorizer_by_collection[collection] = validate_vectorizer_config(cfg_obj)

        all_missing_props = sum(len(v) for v in missing_required_properties.values())
        vectorizer_validation = all(bool(v) for v in vectorizer_by_collection.values())

        schema_validation_passed = (
            design_ok
            and duplicate_collection_prevention
            and all_missing_props == 0
            and vectorizer_validation
            and all(client.collections.exists(collection) for collection in REQUIRED_COLLECTIONS)
        )

        runtime_seconds = round(time.perf_counter() - start, 4)
        report: Dict[str, Any] = {
            "status": "success",
            "connection_successful": connection_successful,
            "collections_created": sorted(created),
            "collections_existing": sorted(existing),
            "schema_validation_passed": schema_validation_passed,
            "missing_required_properties": missing_required_properties,
            "vectorizer_validation": vectorizer_validation,
            "vectorizer_validation_by_collection": vectorizer_by_collection,
            "runtime_seconds": runtime_seconds,
            "duplicate_collection_prevention": duplicate_collection_prevention,
            "design_input_validation": {
                "passed": len(design_issues) == 0,
                "issues": list(design_issues),
                "snapshot": dict(design_snapshot),
            },
            "safe_for_weaviate_ingestion": bool(connection_successful and schema_validation_passed),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate schema init report -> %s", report_path)
        LOGGER.info("safe_for_weaviate_ingestion=%s", report["safe_for_weaviate_ingestion"])
    except Exception as exc:
        report = build_failure_report(
            runtime_seconds=time.perf_counter() - start,
            error=str(exc),
            connection_successful=connection_successful,
            created=created,
            existing=existing,
            design_snapshot=design_snapshot,
            design_issues=design_issues,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=ENCODING)
        LOGGER.info("Wrote Weaviate schema init report -> %s", report_path)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
