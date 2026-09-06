from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .search_pipeline import CompanyScope, PERIODIC_REPORT_TYPES
from .z_search_task import OperationalSearchExecutor, SearchRuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "회사·사업연도·보고서 유형을 제한한 뒤 실제 Chroma 원문에서 "
            "문자열의 존재 여부와 예시를 확인합니다."
        )
    )
    parser.add_argument("--company", required=True)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--terms", required=True, nargs="+")
    parser.add_argument(
        "--report-types",
        nargs="+",
        default=list(PERIODIC_REPORT_TYPES),
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="각 표현의 개별 결과와 함께 모든 표현이 한 청크에 있는 결과를 계산합니다.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _snippet(text: str, term: str, width: int = 180) -> str:
    index = text.lower().find(term.lower())
    if index < 0:
        return text[: width * 2]
    start = max(0, index - width)
    end = min(len(text), index + len(term) + width)
    return text[start:end].replace("\n", " ").strip()


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_year > args.end_year:
        raise ValueError("start-year must not be after end-year")
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")

    config = SearchRuntimeConfig.from_environment(args.project_root)
    executor = OperationalSearchExecutor(config, debug=args.debug)
    catalog = executor.catalog
    company_codes = set(
        catalog.resolve_company_scope(
            CompanyScope("direct", (args.company,))
        )
    )
    report_types = set(args.report_types)
    documents = [
        document
        for document in catalog.documents
        if document.corp_code in company_codes
        and document.normalized_report_type in report_types
        and document.base_year is not None
        and args.start_year <= document.base_year <= args.end_year
    ]
    doc_ids = tuple(document.doc_id for document in documents)
    rows = executor.store.get_document_rows(doc_ids) if doc_ids else []
    documents_by_id = {document.doc_id: document for document in documents}

    document_counts: dict[str, int] = defaultdict(int)
    for document in documents:
        document_counts[str(document.base_year)] += 1

    term_results: dict[str, Any] = {}
    terms = tuple(dict.fromkeys(value.strip() for value in args.terms if value.strip()))
    if not terms:
        raise ValueError("terms must contain at least one non-empty value")
    for term in terms:
        by_year: dict[str, dict[str, Any]] = {}
        matches_by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            metadata = row.get("metadata") or {}
            document = documents_by_id.get(str(metadata.get("doc_id") or "").strip())
            if document is None:
                continue
            text = str(row.get("text") or "")
            title = " ".join(
                str(metadata.get(key) or "")
                for key in ("table_title", "section_path", "subtitle")
            )
            if term.lower() not in f"{title}\n{text}".lower():
                continue
            matches_by_year[str(document.base_year)].append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": document.doc_id,
                    "report_type": document.normalized_report_type,
                    "report_name": document.report_nm,
                    "table_title": metadata.get("table_title"),
                    "section_path": metadata.get("section_path"),
                    "snippet": _snippet(f"{title}\n{text}", term),
                }
            )
        for year in map(str, range(args.start_year, args.end_year + 1)):
            matches = matches_by_year.get(year, [])
            by_year[year] = {
                "matching_chunks": len(matches),
                "matching_documents": len(
                    {match["doc_id"] for match in matches}
                ),
                "samples": matches[: args.max_samples],
            }
        term_results[term] = {
            "total_matching_chunks": sum(
                value["matching_chunks"] for value in by_year.values()
            ),
            "years": by_year,
        }

    combined_result: dict[str, Any] | None = None
    if args.require_all:
        combined_by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            metadata = row.get("metadata") or {}
            document = documents_by_id.get(
                str(metadata.get("doc_id") or "").strip()
            )
            if document is None:
                continue
            text = str(row.get("text") or "")
            title = " ".join(
                str(metadata.get(key) or "")
                for key in ("table_title", "section_path", "subtitle")
            )
            haystack = f"{title}\n{text}"
            if not all(term.lower() in haystack.lower() for term in terms):
                continue
            combined_by_year[str(document.base_year)].append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": document.doc_id,
                    "report_type": document.normalized_report_type,
                    "report_name": document.report_nm,
                    "table_title": metadata.get("table_title"),
                    "section_path": metadata.get("section_path"),
                    "snippet": _snippet(haystack, terms[0]) if terms else "",
                }
            )
        combined_years = {}
        for year in map(str, range(args.start_year, args.end_year + 1)):
            matches = combined_by_year.get(year, [])
            combined_years[year] = {
                "matching_chunks": len(matches),
                "matching_documents": len(
                    {match["doc_id"] for match in matches}
                ),
                "samples": matches[: args.max_samples],
            }
        combined_result = {
            "required_terms": list(terms),
            "total_matching_chunks": sum(
                value["matching_chunks"] for value in combined_years.values()
            ),
            "years": combined_years,
        }

    return {
        "company": args.company,
        "period": {"start_year": args.start_year, "end_year": args.end_year},
        "report_types": list(args.report_types),
        "documents_considered": {
            "total": len(documents),
            "by_year": dict(sorted(document_counts.items())),
        },
        "chunks_considered": len(rows),
        "terms": term_results,
        "all_terms_together": combined_result,
    }


def main() -> int:
    args = parse_args()
    result = inspect(args)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = args.project_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"[SEARCH][INSPECT] saved={output}")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
