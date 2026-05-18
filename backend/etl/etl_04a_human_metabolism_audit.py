"""ETL step 04a: audit human metabolism corpus readiness for ETL_04 ingestion."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PDF_BACKEND = "unavailable"
try:
    from pypdf import PdfReader  # type: ignore

    PDF_BACKEND = "pypdf"
except Exception:
    from PyPDF2 import PdfReader  # type: ignore

    PDF_BACKEND = "PyPDF2_fallback"

ENCODING = "utf-8"
UNKNOWN = "unknown"
SUPPORTED_SUFFIXES: Tuple[str, ...] = (".pdf", ".csv", ".xlsx", ".txt", ".md")

DOMAIN_ORDER: Tuple[str, ...] = (
    "gastric_emptying",
    "alcohol_absorption",
    "food_effects",
    "body_water_distribution",
    "sex_differences",
    "age_effects",
    "body_mass_effects",
    "lean_body_mass",
    "enzyme_variation",
    "liver_function",
    "ethanol_elimination_rate",
    "bac_kinetics",
    "distribution_volume",
    "metabolic_modifiers",
)

PBPK_CRITICAL_DOMAINS: Tuple[str, ...] = (
    "gastric_emptying",
    "alcohol_absorption",
    "body_water_distribution",
    "enzyme_variation",
    "ethanol_elimination_rate",
    "bac_kinetics",
    "body_mass_effects",
)

DOMAIN_SCORE_WEIGHTS: Mapping[str, int] = {
    "strong": 3,
    "adequate": 2,
    "weak": 1,
    "missing": 0,
}

DOMAIN_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "gastric_emptying": (
        "gastric emptying",
        "stomach emptying",
        "emptying rate",
        "gastric transit",
        "gastric retention",
    ),
    "alcohol_absorption": (
        "alcohol absorption",
        "ethanol absorption",
        "intestinal absorption",
        "first pass metabolism",
        "absorption rate",
        "absorptive phase",
    ),
    "food_effects": (
        "food effect",
        "fed state",
        "fasted state",
        "meal effect",
        "with food",
        "empty stomach",
    ),
    "body_water_distribution": (
        "body water",
        "total body water",
        "water distribution",
        "intracellular water",
        "extracellular water",
        "hydration status",
    ),
    "sex_differences": (
        "sex difference",
        "male female",
        "women",
        "men",
        "female",
        "male",
        "gender difference",
    ),
    "age_effects": (
        "age effect",
        "aging",
        "older adult",
        "elderly",
        "adolescent",
        "age related",
    ),
    "body_mass_effects": (
        "body mass",
        "body weight",
        "weight based",
        "obesity",
        "bmi",
        "mass effect",
    ),
    "lean_body_mass": (
        "lean body mass",
        "fat free mass",
        "body composition",
        "adipose tissue",
        "muscle mass",
    ),
    "enzyme_variation": (
        "adh",
        "aldehyde dehydrogenase",
        "alcohol dehydrogenase",
        "enzyme variation",
        "genetic polymorphism",
        "metabolic enzyme",
        "aldh",
        "cyp2e1",
    ),
    "liver_function": (
        "liver function",
        "hepatic function",
        "liver disease",
        "cirrhosis",
        "hepatic clearance",
        "portal blood flow",
        "liver blood flow",
    ),
    "ethanol_elimination_rate": (
        "ethanol elimination",
        "alcohol elimination",
        "elimination rate",
        "clearance rate",
        "zero order",
        "metabolic clearance",
    ),
    "bac_kinetics": (
        "bac",
        "blood alcohol concentration",
        "brac",
        "kinetic profile",
        "time to peak",
        "widmark",
        "blood alcohol curve",
    ),
    "distribution_volume": (
        "volume of distribution",
        "distribution volume",
        "widmark factor",
        "r factor",
        "apparent volume",
    ),
    "metabolic_modifiers": (
        "metabolic modifier",
        "cofactor",
        "modifier",
        "co ingestion",
        "drug interaction",
        "absorption modifier",
        "clearance modifier",
    ),
}

LOGGER = logging.getLogger("etl_04a_human_metabolism_audit")


@dataclass(frozen=True)
class FileAudit:
    relative_path: str
    filename: str
    filetype: str
    size_bytes: int
    page_count: int
    extractable_text_length: int
    readability_score: Optional[float]
    extracted_text: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def corpus_root(root: Path) -> Path:
    return root / "data" / "raw" / "08_human_metabolism"


def output_report_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "human_metabolism_audit_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_domain_csv_path(root: Path) -> Path:
    path = root / "data" / "interim" / "human" / "human_domain_coverage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s.%/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [part.strip() for part in parts if clean_text(part)]


def split_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def count_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    groups = re.findall(r"[aeiouy]+", token)
    count = len(groups)
    if token.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_score(text: str) -> Optional[float]:
    words = split_words(text)
    sentences = split_sentences(text)
    if len(words) < 30 or len(sentences) < 2:
        return None
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = len(words) / max(len(sentences), 1)
    syllables_per_word = syllables / max(len(words), 1)
    score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return round(score, 4)


def extract_pdf_text(path: Path) -> Tuple[int, str]:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return 0, ""
    texts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text)
    return len(reader.pages), "\n".join(texts)


def extract_csv_text(path: Path) -> str:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return ""
    if df.empty and not list(df.columns):
        return ""
    lines: List[str] = []
    if list(df.columns):
        lines.append(" ".join(clean_text(column) for column in df.columns if clean_text(column)))
    for row in df.astype(str).itertuples(index=False, name=None):
        line = " ".join(clean_text(value) for value in row if clean_text(value))
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_xlsx_text(path: Path) -> str:
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception:
        return ""
    lines: List[str] = []
    for sheet_name in sorted(sheets.keys()):
        df = sheets[sheet_name].fillna("")
        lines.append(clean_text(sheet_name))
        if list(df.columns):
            lines.append(" ".join(clean_text(column) for column in df.columns if clean_text(column)))
        for row in df.astype(str).itertuples(index=False, name=None):
            line = " ".join(clean_text(value) for value in row if clean_text(value))
            if line:
                lines.append(line)
    return "\n".join(line for line in lines if line)


def extract_plain_text(path: Path) -> str:
    try:
        return path.read_text(encoding=ENCODING, errors="ignore")
    except Exception:
        return ""


def audit_file(root: Path, path: Path) -> FileAudit:
    suffix = path.suffix.lower()
    page_count = 0
    text = ""
    if suffix == ".pdf":
        page_count, text = extract_pdf_text(path)
    elif suffix == ".csv":
        text = extract_csv_text(path)
    elif suffix == ".xlsx":
        text = extract_xlsx_text(path)
    elif suffix in {".txt", ".md"}:
        text = extract_plain_text(path)

    extractable_length = len(text)
    score = readability_score(text)
    return FileAudit(
        relative_path=str(path.relative_to(root)),
        filename=path.name,
        filetype=suffix.lstrip("."),
        size_bytes=path.stat().st_size,
        page_count=page_count,
        extractable_text_length=extractable_length,
        readability_score=score,
        extracted_text=text,
    )


def iter_supported_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def keyword_hits(text: str, keyword: str) -> int:
    if not text or not keyword:
        return 0
    pattern = re.escape(keyword)
    return len(re.findall(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text))


def domain_score(total_hits: int, supporting_files: int, text_length: int) -> str:
    if total_hits == 0 or supporting_files == 0:
        return "missing"
    if total_hits >= 8 and supporting_files >= 2 and text_length >= 4000:
        return "strong"
    if total_hits >= 3 and text_length >= 1200:
        return "adequate"
    return "weak"


def audit_domains(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    domain_rows: List[Dict[str, Any]] = []
    normalized_texts: List[Tuple[FileAudit, str]] = [
        (audit, normalize_text(audit.extracted_text)) for audit in file_audits
    ]

    for domain in DOMAIN_ORDER:
        keywords = DOMAIN_KEYWORDS[domain]
        total_hits = 0
        matched_keywords: List[str] = []
        supporting_files: List[str] = []
        supporting_text_length = 0

        for audit, normalized in normalized_texts:
            file_hits = 0
            for keyword in keywords:
                hits = keyword_hits(normalized, keyword)
                if hits > 0:
                    file_hits += hits
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
            if file_hits > 0:
                total_hits += file_hits
                supporting_files.append(audit.relative_path)
                supporting_text_length += audit.extractable_text_length

        score = domain_score(
            total_hits=total_hits,
            supporting_files=len(supporting_files),
            text_length=supporting_text_length,
        )
        domain_rows.append(
            {
                "domain": domain,
                "score": score,
                "supporting_file_count": len(supporting_files),
                "supporting_files": supporting_files,
                "matched_keyword_count": len(matched_keywords),
                "matched_keywords": matched_keywords,
                "total_keyword_hits": total_hits,
                "supporting_extractable_text_length": supporting_text_length,
            }
        )
    return domain_rows


def build_file_rows(file_audits: Sequence[FileAudit]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for audit in file_audits:
        rows.append(
            {
                "filename": audit.filename,
                "relative_path": audit.relative_path,
                "filetype": audit.filetype,
                "size_bytes": audit.size_bytes,
                "page_count": audit.page_count,
                "extractable_text_length": audit.extractable_text_length,
                "readability_score": audit.readability_score,
            }
        )
    return rows


def additional_data_required(domain_rows: Sequence[Mapping[str, Any]], file_count: int) -> bool:
    readiness = evaluate_readiness(domain_rows, file_count)
    return not readiness["safe_for_etl_04_ingestion"]


def corpus_quality_score(domain_rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(DOMAIN_SCORE_WEIGHTS.get(clean_text(row.get("score", "")).lower(), 0) for row in domain_rows)


def evaluate_readiness(domain_rows: Sequence[Mapping[str, Any]], file_count: int) -> Dict[str, Any]:
    score_by_domain = {
        clean_text(row.get("domain", "")): clean_text(row.get("score", "")).lower() or "missing"
        for row in domain_rows
    }
    missing_domains = sorted(domain for domain, score in score_by_domain.items() if score == "missing")
    strong_or_adequate_domains = sorted(
        domain for domain, score in score_by_domain.items() if score in {"strong", "adequate"}
    )
    pbpk_critical_weak_or_missing = sorted(
        domain
        for domain in PBPK_CRITICAL_DOMAINS
        if score_by_domain.get(domain, "missing") not in {"strong", "adequate"}
    )

    safe = (
        file_count > 0
        and len(missing_domains) == 0
        and len(strong_or_adequate_domains) >= 10
        and len(pbpk_critical_weak_or_missing) == 0
    )

    reasoning: List[str] = []
    reasoning.append(f"Missing domains: {len(missing_domains)}.")
    reasoning.append(f"Domains scored strong or adequate: {len(strong_or_adequate_domains)}.")
    reasoning.append(
        f"PBPK-critical domains at adequate-or-strong coverage: {len(PBPK_CRITICAL_DOMAINS) - len(pbpk_critical_weak_or_missing)} of {len(PBPK_CRITICAL_DOMAINS)}."
    )
    if file_count == 0:
        reasoning.append("No supported corpus files were found, so ingestion is blocked.")
    if missing_domains:
        reasoning.append(f"Ingestion is blocked because missing domains remain: {', '.join(missing_domains)}.")
    if len(strong_or_adequate_domains) < 10:
        reasoning.append("Ingestion is blocked because fewer than 10 domains are scored strong or adequate.")
    if pbpk_critical_weak_or_missing:
        reasoning.append(
            "Ingestion is blocked because PBPK-critical domains are below adequate: "
            + ", ".join(pbpk_critical_weak_or_missing)
            + "."
        )
    if safe:
        weak_noncritical = sorted(
            domain
            for domain, score in score_by_domain.items()
            if score == "weak" and domain not in PBPK_CRITICAL_DOMAINS
        )
        reasoning.append("Ingestion is allowed because there are no missing domains, at least 10 domains are strong or adequate, and all PBPK-critical domains are adequate or strong.")
        if weak_noncritical:
            reasoning.append(
                "Weak noncritical domains are non-blocking under the ETL_04 gate: "
                + ", ".join(weak_noncritical)
                + "."
            )

    return {
        "safe_for_etl_04_ingestion": safe,
        "missing_domains": missing_domains,
        "strong_or_adequate_domains": strong_or_adequate_domains,
        "pbpk_critical_domains": list(PBPK_CRITICAL_DOMAINS),
        "pbpk_critical_weak_or_missing": pbpk_critical_weak_or_missing,
        "readiness_reasoning": reasoning,
    }


def serialize_report(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): serialize_report(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [serialize_report(value) for value in payload]
    return payload


def main() -> None:
    configure_logging()
    root = repo_root()
    corpus = corpus_root(root)
    report_path = output_report_path(root)
    coverage_csv_path = output_domain_csv_path(root)

    if not corpus.exists():
        raise FileNotFoundError(f"Missing human metabolism corpus directory: {corpus}")

    LOGGER.info("Auditing human metabolism corpus: %s", corpus)
    file_audits = [audit_file(root, path) for path in iter_supported_files(corpus)]
    domain_rows = audit_domains(file_audits)

    domain_df = pd.DataFrame(
        [
            {
                "domain": row["domain"],
                "score": row["score"],
                "supporting_file_count": row["supporting_file_count"],
                "matched_keyword_count": row["matched_keyword_count"],
                "total_keyword_hits": row["total_keyword_hits"],
                "supporting_extractable_text_length": row["supporting_extractable_text_length"],
                "matched_keywords": "; ".join(row["matched_keywords"]),
                "supporting_files": "; ".join(row["supporting_files"]),
            }
            for row in domain_rows
        ]
    )
    domain_df.to_csv(coverage_csv_path, index=False, encoding=ENCODING)

    file_rows = build_file_rows(file_audits)
    total_extractable_text = sum(audit.extractable_text_length for audit in file_audits)
    readable_files = sum(1 for audit in file_audits if audit.readability_score is not None)
    score_counts: Dict[str, int] = {label: 0 for label in ("missing", "weak", "adequate", "strong")}
    for row in domain_rows:
        score_counts[row["score"]] += 1

    readiness = evaluate_readiness(domain_rows, len(file_audits))
    quality_score = corpus_quality_score(domain_rows)
    max_quality_score = len(domain_rows) * DOMAIN_SCORE_WEIGHTS["strong"]
    need_more_data = additional_data_required(domain_rows, len(file_audits))
    safe = readiness["safe_for_etl_04_ingestion"]

    report = {
        "metadata": {
            "script": "etl/etl_04a_human_metabolism_audit.py",
            "corpus_root": str(corpus.relative_to(root)),
            "supported_filetypes": [suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES],
            "pdf_backend": PDF_BACKEND,
            "files_found": len(file_audits),
            "total_extractable_text_length": total_extractable_text,
            "readable_files": readable_files,
        },
        "documents": file_rows,
        "domain_coverage": domain_rows,
        "domain_score_counts": score_counts,
        "corpus_quality_score": quality_score,
        "corpus_quality_score_max": max_quality_score,
        "recommendation": {
            "additional_data_required": need_more_data,
        },
        "readiness_reasoning": readiness["readiness_reasoning"],
        "final_decision": {
            "safe_for_etl_04_ingestion": safe,
            "missing_domains": readiness["missing_domains"],
            "pbpk_critical_domains": readiness["pbpk_critical_domains"],
            "pbpk_critical_weak_or_missing": readiness["pbpk_critical_weak_or_missing"],
            "strong_or_adequate_domain_count": len(readiness["strong_or_adequate_domains"]),
        },
    }

    with report_path.open("w", encoding=ENCODING) as handle:
        json.dump(serialize_report(report), handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info(
        "Wrote human metabolism audit report -> %s | safe_for_etl_04_ingestion=%s",
        report_path,
        safe,
    )
    LOGGER.info("Wrote domain coverage CSV -> %s", coverage_csv_path)


if __name__ == "__main__":
    main()
