#!/usr/bin/env python3
"""Repository hygiene checks for GitHub push readiness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "data" / "interim" / "reasoning" / "repo_hygiene_report.json"
MAX_TRACKED_SIZE_BYTES = 50 * 1024 * 1024

DISALLOWED_PREFIXES = (
    "data/raw/01_food_composition/",
    "data/raw/04_biological_pathways/",
    "data/raw/06_pubchem_cheminformatics/",
    "data/processed/weaviate/embedded/",
)

REQUIRED_FILES = (
    ".gitignore",
    ".env.example",
    "ARTIFACTS.md",
    "README.md",
)


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_paths_from_nul_output(args: Sequence[str]) -> List[str]:
    completed = _run_git(args)
    if completed.returncode != 0:
        return []
    raw = completed.stdout
    if not raw:
        return []
    return sorted({item for item in raw.split("\0") if item})


def _is_ignored(path: str) -> bool:
    completed = _run_git(["check-ignore", path])
    return completed.returncode == 0


def _collect_pycache_dirs() -> List[str]:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("__pycache__"))


def _collect_pyc_files() -> List[str]:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("*.pyc"))


def _tracked_and_staged_paths() -> List[str]:
    tracked = set(_git_paths_from_nul_output(["ls-files", "-z"]))
    staged = set(_git_paths_from_nul_output(["diff", "--cached", "--name-only", "-z"]))
    return sorted(tracked | staged)


def _large_tracked_or_staged(paths: Sequence[str]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for rel in paths:
        abs_path = REPO_ROOT / rel
        if not abs_path.exists() or not abs_path.is_file():
            continue
        size = abs_path.stat().st_size
        if size > MAX_TRACKED_SIZE_BYTES:
            results.append({"path": rel, "size_bytes": size})
    return sorted(results, key=lambda item: str(item["path"]))


def _find_disallowed_paths(paths: Sequence[str]) -> List[str]:
    hits: List[str] = []
    for rel in paths:
        lowered = rel.replace("\\", "/")
        if any(lowered.startswith(prefix) for prefix in DISALLOWED_PREFIXES):
            hits.append(rel)
    return sorted(set(hits))


def _find_pdf_paths(paths: Sequence[str]) -> List[str]:
    return sorted({rel for rel in paths if rel.lower().endswith(".pdf")})


def main() -> int:
    env_path = REPO_ROOT / ".env"
    env_exists = env_path.exists()
    env_ignored = _is_ignored(".env")

    required_file_exists = {name: (REPO_ROOT / name).exists() for name in REQUIRED_FILES}

    pycache_dirs = _collect_pycache_dirs()
    pyc_files = _collect_pyc_files()

    tracked_or_staged = _tracked_and_staged_paths()
    large_files = _large_tracked_or_staged(tracked_or_staged)
    disallowed_paths = _find_disallowed_paths(tracked_or_staged)
    staged_or_tracked_pdfs = _find_pdf_paths(tracked_or_staged)

    checks = {
        "env_exists_locally": env_exists,
        "env_is_git_ignored": env_ignored,
        "env_example_exists": required_file_exists[".env.example"],
        "gitignore_exists": required_file_exists[".gitignore"],
        "artifacts_doc_exists": required_file_exists["ARTIFACTS.md"],
        "readme_exists": required_file_exists["README.md"],
        "no_pycache_dirs": len(pycache_dirs) == 0,
        "no_pyc_files": len(pyc_files) == 0,
        "no_large_tracked_or_staged_files": len(large_files) == 0,
        "raw_data_not_tracked_or_staged": len(disallowed_paths) == 0,
        "pdfs_not_tracked_or_staged": len(staged_or_tracked_pdfs) == 0,
    }

    safe_for_git_push = all(checks.values())

    report = {
        "checks": checks,
        "pycache_dirs_found": pycache_dirs,
        "pyc_files_found": pyc_files,
        "tracked_or_staged_count": len(tracked_or_staged),
        "large_files_detected": large_files,
        "disallowed_paths_detected": disallowed_paths,
        "pdfs_tracked_or_staged": staged_or_tracked_pdfs,
        "safe_for_git_push": safe_for_git_push,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if safe_for_git_push else 1


if __name__ == "__main__":
    raise SystemExit(main())
