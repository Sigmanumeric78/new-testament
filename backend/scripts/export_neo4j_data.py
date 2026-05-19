#!/usr/bin/env python3
"""Read-only Neo4j runtime export foundation.

Exports graph metadata snapshots to:
data/exports/runtime/<release>/neo4j/
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - environment without neo4j driver
    GraphDatabase = None

from utils.config import get_neo4j_config, project_root

DEFAULT_RELEASE = "v0.6-chemical-explorer"


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


def runtime_export_root(release: str, output_root: Optional[str] = None) -> Path:
    if output_root:
        base = Path(output_root)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / release
    return project_root() / "data" / "exports" / "runtime" / release


def neo4j_export_dir(release: str, output_root: Optional[str] = None) -> Path:
    return runtime_export_root(release, output_root) / "neo4j"


def expected_neo4j_files() -> List[str]:
    return [
        "cypher_counts.json",
        "node_counts.json",
        "relationship_counts.json",
        "schema_snapshot.json",
        "graph_export_manifest.json",
        "neo4j_dump_instructions.md",
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_json_safe(dict(payload))
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_query(session: Any, query: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in session.run(query):
        rows.append(_to_json_safe(dict(record)))
    return rows


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date, time)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    # Neo4j temporal types often expose iso_format() or isoformat().
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        try:
            return iso_format()
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    # Neo4j Path-like
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        nodes = getattr(value, "nodes", [])
        relationships = getattr(value, "relationships", [])
        return {
            "type": "Path",
            "nodes": [_to_json_safe(node) for node in list(nodes)],
            "relationships": [_to_json_safe(rel) for rel in list(relationships)],
        }

    # Neo4j Node-like
    if hasattr(value, "labels") and hasattr(value, "items"):
        node_props: Dict[str, Any] = {}
        try:
            node_props = {str(k): _to_json_safe(v) for k, v in dict(value.items()).items()}
        except Exception:
            node_props = {}
        return {
            "type": "Node",
            "element_id": _clean_text(getattr(value, "element_id", "")) or None,
            "labels": sorted([_clean_text(label) for label in list(getattr(value, "labels", [])) if _clean_text(label)]),
            "properties": node_props,
        }

    # Neo4j Relationship-like
    if hasattr(value, "start_node") and hasattr(value, "end_node"):
        rel_props: Dict[str, Any] = {}
        try:
            rel_props = {str(k): _to_json_safe(v) for k, v in dict(value.items()).items()}
        except Exception:
            rel_props = {}
        start_node = getattr(value, "start_node", None)
        end_node = getattr(value, "end_node", None)
        return {
            "type": "Relationship",
            "element_id": _clean_text(getattr(value, "element_id", "")) or None,
            "relationship_type": _clean_text(getattr(value, "type", "")) or None,
            "start_node_element_id": _clean_text(getattr(start_node, "element_id", "")) or None,
            "end_node_element_id": _clean_text(getattr(end_node, "element_id", "")) or None,
            "properties": rel_props,
        }

    # Neo4j Record/Data classes may expose .data()
    data_method = getattr(value, "data", None)
    if callable(data_method):
        try:
            return _to_json_safe(data_method())
        except Exception:
            pass

    # Generic object fallback
    if hasattr(value, "__dict__"):
        try:
            obj_vars = vars(value)
            if obj_vars:
                return _to_json_safe(obj_vars)
        except Exception:
            pass

    return str(value)


def _export_unavailable(target_dir: Path, release: str, reason: str) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    node_counts = {"status": "unavailable", "reason": reason, "generated_at_utc": generated_at, "rows": []}
    relationship_counts = {"status": "unavailable", "reason": reason, "generated_at_utc": generated_at, "rows": []}
    cypher_counts = {"status": "unavailable", "reason": reason, "generated_at_utc": generated_at, "summary": {}}
    schema_snapshot = {"status": "unavailable", "reason": reason, "generated_at_utc": generated_at}

    _write_json(target_dir / "node_counts.json", node_counts)
    _write_json(target_dir / "relationship_counts.json", relationship_counts)
    _write_json(target_dir / "cypher_counts.json", cypher_counts)
    _write_json(target_dir / "schema_snapshot.json", schema_snapshot)

    manifest = {
        "release": release,
        "generated_at_utc": generated_at,
        "status": "unavailable",
        "read_only": True,
        "reason": reason,
        "files": expected_neo4j_files(),
    }
    _write_json(target_dir / "graph_export_manifest.json", manifest)
    _write_text(target_dir / "neo4j_dump_instructions.md", _neo4j_dump_instructions())
    return _to_json_safe(manifest)


def _neo4j_dump_instructions() -> str:
    return (
        "# Neo4j Native Dump Instructions\n\n"
        "This export script is read-only and does not run native dump commands.\n\n"
        "Use manual dump commands from a Neo4j host when appropriate:\n\n"
        "```bash\n"
        "neo4j-admin database dump neo4j --to-path=/path/to/backup\n"
        "```\n\n"
        "For restore (on target host):\n\n"
        "```bash\n"
        "neo4j-admin database load neo4j --from-path=/path/to/backup --overwrite-destination=true\n"
        "```\n\n"
        "Alternative restore path:\n"
        "- rebuild graph from processed CSV artifacts using existing ETL scripts.\n"
    )


def export_neo4j_data(release: str = DEFAULT_RELEASE, output_root: Optional[str] = None) -> Dict[str, Any]:
    target_dir = neo4j_export_dir(release, output_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    if GraphDatabase is None:
        return _export_unavailable(target_dir, release, "neo4j Python driver is not installed.")

    try:
        config = get_neo4j_config()
    except Exception as exc:
        return _export_unavailable(target_dir, release, str(exc))

    driver = None
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            node_rows = _run_query(
                session,
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label",
            )
            rel_rows = _run_query(
                session,
                "MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count ORDER BY relationship_type",
            )

            totals = _to_json_safe(
                {
                "total_nodes": _run_query(session, "MATCH (n) RETURN count(n) AS value")[0]["value"],
                "total_relationships": _run_query(session, "MATCH ()-[r]->() RETURN count(r) AS value")[0]["value"],
                "distinct_labels": _run_query(
                    session, "MATCH (n) UNWIND labels(n) AS label RETURN count(DISTINCT label) AS value"
                )[0]["value"],
                "distinct_relationship_types": _run_query(
                    session, "MATCH ()-[r]->() RETURN count(DISTINCT type(r)) AS value"
                )[0]["value"],
                }
            )

            schema_snapshot: Dict[str, Any] = {
                "status": "ok",
                "generated_at_utc": generated_at,
                "constraints": [],
                "indexes": [],
                "schema_queries_read_only": True,
            }
            try:
                schema_snapshot["constraints"] = _run_query(session, "SHOW CONSTRAINTS")
            except Exception as exc:
                schema_snapshot["constraints_error"] = _clean_text(exc)
            try:
                schema_snapshot["indexes"] = _run_query(session, "SHOW INDEXES")
            except Exception as exc:
                schema_snapshot["indexes_error"] = _clean_text(exc)

            _write_json(
                target_dir / "node_counts.json",
                {
                    "status": "ok",
                    "generated_at_utc": generated_at,
                    "query": "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label",
                    "rows": node_rows,
                },
            )
            _write_json(
                target_dir / "relationship_counts.json",
                {
                    "status": "ok",
                    "generated_at_utc": generated_at,
                    "query": (
                        "MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count "
                        "ORDER BY relationship_type"
                    ),
                    "rows": rel_rows,
                },
            )
            _write_json(
                target_dir / "cypher_counts.json",
                {
                    "status": "ok",
                    "generated_at_utc": generated_at,
                    "summary": totals,
                    "validation_queries_read_only": True,
                },
            )
            _write_json(target_dir / "schema_snapshot.json", _to_json_safe(schema_snapshot))
    except Exception as exc:
        return _export_unavailable(target_dir, release, f"neo4j export error: {exc}")
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    _write_text(target_dir / "neo4j_dump_instructions.md", _neo4j_dump_instructions())
    manifest = {
        "release": release,
        "generated_at_utc": generated_at,
        "status": "ok",
        "read_only": True,
        "database": _clean_text(config.get("database")) or "neo4j",
        "files": expected_neo4j_files(),
    }
    _write_json(target_dir / "graph_export_manifest.json", manifest)
    return _to_json_safe(manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Neo4j runtime metadata (read-only)")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Release name")
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional root directory for runtime exports. Defaults to data/exports/runtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = export_neo4j_data(release=_clean_text(args.release) or DEFAULT_RELEASE, output_root=args.output_root or None)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
