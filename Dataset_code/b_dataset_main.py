#!/usr/bin/env python3
"""Build company-level intermediates and final CLOVA tuning datasets.

Pipeline:
  source JSON/XML -> document-level split -> company JSONL intermediates
  -> train.csv / validation.csv / test.csv

All common paths and generation settings live in the project-root Config.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Config as cfg  # noqa: E402
from a_dataset_try1 import load_input_records, normalize_text  # noqa: E402
from a_dataset_try2 import (  # noqa: E402
    ClovaClient,
    build_rows,
    make_bundles,
    write_rows,
)


SPLIT_NAMES = ("train", "validation", "test")


def record_identity(record: dict[str, Any]) -> str:
    source = normalize_text(record.get("_source_path"))
    receipt = normalize_text(record.get("rcept_no") or record.get("receipt_no"))
    return f"{source}#{receipt}" if receipt else source or normalize_text(record.get("title") or record.get("question"))


def is_index_record(record: dict[str, Any]) -> bool:
    source_name = Path(str(record.get("_source_path") or "")).name.lower()
    metadata_keys = {"corp_code", "corp_name", "report_nm", "rcept_no", "rcept_dt"}
    return source_name.startswith("list_") or metadata_keys.issubset(record.keys())


def record_year(record: dict[str, Any]) -> int | None:
    candidates = (
        record.get("title"),
        record.get("question"),
        record.get("rcept_dt"),
        record.get("rcept_no"),
        record.get("_source_path"),
    )
    for value in candidates:
        match = re.search(r"(?<!\d)(20\d{2})", normalize_text(value))
        if match:
            return int(match.group(1))
    return None


def record_company(record: dict[str, Any]) -> str:
    title = normalize_text(record.get("title") or record.get("question"))
    if "/" in title:
        return title.split("/", 1)[0].strip() or "unknown"
    source = Path(str(record.get("_source_path") or ""))
    # Expected layout: exchange/company/receipt/file.xml
    if len(source.parts) >= 3:
        return source.parts[-3]
    return "unknown"


def hash_split(identity: str) -> str:
    train_ratio, validation_ratio, test_ratio = cfg.HASH_SPLIT_RATIOS
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        raise ValueError("HASH_SPLIT_RATIOS must sum to a positive value")
    value = int(
        hashlib.sha256((cfg.SPLIT_SEED + identity).encode("utf-8")).hexdigest()[:12],
        16,
    ) / float(0xFFFFFFFFFFFF)
    train_edge = train_ratio / total
    validation_edge = (train_ratio + validation_ratio) / total
    if value < train_edge:
        return "train"
    if value < validation_edge:
        return "validation"
    return "test"


def choose_split(record: dict[str, Any]) -> str:
    identity = record_identity(record)
    if cfg.DATASET_SPLIT_MODE == "hash":
        return hash_split(identity)
    if cfg.DATASET_SPLIT_MODE != "time":
        raise ValueError("DATASET_SPLIT_MODE must be 'time' or 'hash'")
    year = record_year(record)
    if year is None:
        return hash_split(identity)
    if year <= cfg.TRAIN_END_YEAR:
        return "train"
    if year in cfg.VALIDATION_YEARS:
        return "validation"
    if year >= cfg.TEST_START_YEAR:
        return "test"
    return hash_split(identity)


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in SPLIT_NAMES}
    seen_sources: dict[str, str] = {}
    for record in records:
        source = record_identity(record)
        split = choose_split(record)
        previous = seen_sources.get(source)
        if previous is not None and previous != split:
            raise RuntimeError(f"Document leakage detected: {source} is in {previous} and {split}")
        seen_sources[source] = split
        result[split].append(record)
    source_sets = {
        name: {record_identity(record) for record in items}
        for name, items in result.items()
    }
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                raise RuntimeError(f"Document leakage between {left} and {right}: {next(iter(overlap))}")
    return result


def limit_by_document(records: list[dict[str, Any]], max_documents: int) -> list[dict[str, Any]]:
    if max_documents <= 0:
        return records
    selected_sources: set[str] = set()
    limited: list[dict[str, Any]] = []
    for record in records:
        source = record_identity(record)
        if source not in selected_sources and len(selected_sources) >= max_documents:
            continue
        selected_sources.add(source)
        limited.append(record)
    return limited


def safe_filename(company: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", company).strip("._") or "unknown"
    suffix = hashlib.sha1(company.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:70]}_{suffix}.jsonl"


def group_by_company(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record_company(record)].append(record)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def reindex_rows(rows: list[dict[str, Any]]) -> None:
    for cid, row in enumerate(rows):
        row["C_ID"] = cid
        row["T_ID"] = 0


def write_manifest(
    path: Path,
    split_map: dict[str, list[dict[str, Any]]],
    row_counts: dict[str, int],
    rejected_counts: dict[str, int],
) -> None:
    manifest = {
        "split_mode": cfg.DATASET_SPLIT_MODE,
        "settings": {
            "questions_per_document": cfg.QUESTIONS_PER_DOCUMENT,
            "questions_per_multi_context": cfg.QUESTIONS_PER_MULTI_CONTEXT,
            "multi_document_size": cfg.MULTI_DOCUMENT_SIZE,
            "include_multi_document": cfg.INCLUDE_MULTI_DOCUMENT,
            "model": cfg.CLOVA_MODEL,
        },
        "splits": {},
    }
    for name, records in split_map.items():
        manifest["splits"][name] = {
            "documents": len({record_identity(record) for record in records}),
            "companies": len({record_company(record) for record in records}),
            "rows": row_counts.get(name, 0),
            "rejected": rejected_counts.get(name, 0),
            "years": dict(sorted(Counter(record_year(record) for record in records).items(), key=lambda x: str(x[0]))),
            "sources": sorted({record_identity(record) for record in records}),
        }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(split_map: dict[str, list[dict[str, Any]]]) -> None:
    print(f"Split mode: {cfg.DATASET_SPLIT_MODE}")
    for name in SPLIT_NAMES:
        records = split_map[name]
        documents = len({record_identity(record) for record in records})
        companies = len({record_company(record) for record in records})
        years = Counter(record_year(record) for record in records)
        print(f"{name}: documents={documents}, companies={companies}, years={dict(years)}")


def generate_all(
    split_map: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    limit_companies: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "generated"
    rejected_dir = output_dir / "rejected"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    client = ClovaClient(
        cfg.CLOVA_STUDIO_API_KEY,
        cfg.CLOVA_BASE_URL,
        cfg.CLOVA_MODEL,
        cfg.API_TIMEOUT_SECONDS,
        cfg.API_RETRIES,
    )
    row_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    pipeline_errors: list[str] = []

    for split_name in SPLIT_NAMES:
        final_rows: list[dict[str, Any]] = []
        rejected_total = 0
        companies = list(group_by_company(split_map[split_name]).items())
        if limit_companies > 0:
            companies = companies[:limit_companies]
        for company_index, (company, records) in enumerate(companies, start=1):
            print(f"[{split_name}] {company_index}/{len(companies)} {company}: {len(records)} records")
            bundles = make_bundles(
                records,
                max_context_chars=cfg.MAX_CONTEXT_CHARS,
                multi_doc_size=cfg.MULTI_DOCUMENT_SIZE,
                include_multi=cfg.INCLUDE_MULTI_DOCUMENT,
            )
            try:
                rows, errors = build_rows(
                    bundles,
                    client,
                    single_count=cfg.QUESTIONS_PER_DOCUMENT,
                    multi_count=cfg.QUESTIONS_PER_MULTI_CONTEXT,
                    max_tokens=cfg.MAX_GENERATION_TOKENS,
                    temperature=cfg.GENERATION_TEMPERATURE,
                    include_context=True,
                    system_prompt=cfg.TUNING_SYSTEM_PROMPT,
                    cache_path=cfg.DATASET_CACHE_PATH,
                    limit=0,
                )
            except Exception as exc:  # continue other companies and report at the end
                pipeline_errors.append(f"[{split_name}] {company}: {exc}")
                continue

            company_path = intermediate_dir / split_name / safe_filename(company)
            write_rows(rows, company_path, "jsonl", cfg.TUNING_SYSTEM_PROMPT)
            if errors:
                error_path = rejected_dir / split_name / (safe_filename(company) + ".errors.log")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text("\n".join(errors), encoding="utf-8")
            rejected_total += len(errors)
            final_rows.extend(rows)

        reindex_rows(final_rows)
        final_path = output_dir / f"{split_name}.csv"
        write_rows(final_rows, final_path, "csv", cfg.TUNING_SYSTEM_PROMPT)
        row_counts[split_name] = len(final_rows)
        rejected_counts[split_name] = rejected_total
        print(f"Wrote {final_path}: {len(final_rows)} rows")

    write_manifest(output_dir / "manifest.json", split_map, row_counts, rejected_counts)
    if pipeline_errors:
        (rejected_dir / "pipeline.errors.log").write_text("\n".join(pipeline_errors), encoding="utf-8")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if pipeline_errors:
        print(f"Pipeline errors: {len(pipeline_errors)} (see rejected/pipeline.errors.log)")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate and split HyperCLOVA X tuning datasets.")
    parser.add_argument("--input", type=Path, default=cfg.DATASET_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATASET_OUTPUT_DIR)
    parser.add_argument("--max-documents", type=int, default=cfg.MAX_DOCUMENTS)
    parser.add_argument("--limit-companies", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only inspect document splits; no API calls")
    args = parser.parse_args()

    records = load_input_records(args.input)
    if not cfg.INCLUDE_INDEX_RECORDS:
        records = [record for record in records if not is_index_record(record)]
    records = limit_by_document(records, args.max_documents)
    if not records:
        raise RuntimeError(f"No usable records found below {args.input}")
    split_map = split_records(records)
    print(f"Loaded records: {len(records)}")
    print_summary(split_map)
    if args.dry_run:
        return
    if not cfg.CLOVA_STUDIO_API_KEY:
        raise RuntimeError("CLOVA_STUDIO_API_KEY is empty in Config.py")
    generate_all(split_map, args.output_dir, max(0, args.limit_companies))


if __name__ == "__main__":
    main()
