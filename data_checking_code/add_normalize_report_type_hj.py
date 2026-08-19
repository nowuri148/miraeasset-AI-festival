#!/usr/bin/env python3
"""Add ``normalized_report_type`` to the DART corpus manifest.

The original ``manifest.jsonl`` is never modified.  This script writes a new
JSON Lines file and fails when a document cannot be mapped, unless
``--allow-unknown`` is explicitly supplied.

Mapping strategy
----------------
1. periodic/exchange: map the existing ``doc_subtype``.
2. holding: ``doc_subtype`` is too coarse, so distinguish 일반/약식 using
   ``report_nm``.
3. major: ``doc_subtype`` is null in the corpus, so map the normalized
   ``report_nm``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


# doc_subtype alone identifies these report types.
SUBTYPE_REPORT_TYPE: dict[tuple[str, str], str] = {
    ("periodic", "annual"): "annual_report",
    ("periodic", "half"): "semiannual_report",
    ("periodic", "quarter"): "quarterly_report",
    ("exchange", "단일판매공급계약체결"): "supply_contract",
    ("exchange", "단일판매공급계약해지"): "supply_contract_termination",
    ("exchange", "신규시설투자등"): "facility_investment",
    ("exchange", "투자판단관련주요경영사항"): "material_management_matter",
}


# holding has one doc_subtype but two semantically distinct report forms.
HOLDING_REPORT_NAME_MAP: dict[str, str] = {
    "주식등의대량보유상황보고서(일반)": "large_shareholding_general",
    "주식등의대량보유상황보고서(약식)": "large_shareholding_summary",
}


# All major-report titles currently present in the 4,204-row corpus.
# Aliases that mean the same report type intentionally map to the same value.
MAJOR_REPORT_NAME_MAP: dict[str, str] = {
    "주요사항보고서(자기주식처분결정)":
        "treasury_stock_disposal_decision",
    "주요사항보고서(자기주식취득결정)":
        "treasury_stock_acquisition_decision",
    "주요사항보고서(상각형조건부자본증권발행결정)":
        "write_down_contingent_capital_security_issuance_decision",
    "주요사항보고서(유상증자결정)":
        "paid_in_capital_increase_decision",
    "유상증자결정":
        "paid_in_capital_increase_decision",
    "주요사항보고서(자기주식취득신탁계약체결결정)":
        "treasury_stock_trust_agreement_conclusion_decision",
    "주요사항보고서(자기주식취득신탁계약해지결정)":
        "treasury_stock_trust_agreement_termination_decision",
    "주요사항보고서(회사합병결정)":
        "merger_decision",
    "주요사항보고서(전환사채권발행결정)":
        "convertible_bond_issuance_decision",
    "주요사항보고서(주식교환ㆍ이전결정)":
        "stock_exchange_or_transfer_decision",
    "주요사항보고서(회사분할결정)":
        "company_split_decision",
    "주요사항보고서(회사분할합병결정)":
        "split_merger_decision",
    "주요사항보고서(타법인주식및출자증권양수결정)":
        "other_company_shares_acquisition_decision",
    "주요사항보고서(타법인주식및출자증권양도결정)":
        "other_company_shares_disposal_decision",
    "주요사항보고서(감자결정)":
        "capital_reduction_decision",
    "주요사항보고서(자본으로인정되는채무증권발행결정)":
        "regulatory_capital_debt_security_issuance_decision",
    "주요사항보고서(무상증자결정)":
        "bonus_issue_decision",
    "주요사항보고서(소송등의제기)":
        "lawsuit_filing",
    "주요사항보고서(교환사채권발행결정)":
        "exchangeable_bond_issuance_decision",
    "주요사항보고서(해외증권시장주권등상장폐지)":
        "overseas_securities_delisting_decision",
    "주요사항보고서(해외증권시장주권등상장폐지결정)":
        "overseas_securities_delisting_decision",
    "주요사항보고서(해외증권시장주권등상장)":
        "overseas_securities_listing_decision",
    "주요사항보고서(해외증권시장주권등상장결정)":
        "overseas_securities_listing_decision",
    "주요사항보고서(유형자산양도결정)":
        "tangible_asset_disposal_decision",
    "주요사항보고서(유형자산양수결정)":
        "tangible_asset_acquisition_decision",
    "주요사항보고서(영업양수결정)":
        "business_acquisition_decision",
    "주요사항보고서(자기전환사채매도결정)":
        "own_convertible_bond_sale_decision",
    "주요사항보고서(영업정지)":
        "business_suspension",
    "주요사항보고서(제3자의전환사채매수선택권행사)":
        "third_party_convertible_bond_call_option_exercise",
}


LEADING_DISCLOSURE_TAGS = re.compile(r"^(?:\s*\[[^\]]+\]\s*)+")
ALL_WHITESPACE = re.compile(r"\s+")


def canonical_report_name(report_name: str) -> str:
    """Make report titles stable without removing semantic parentheses.

    Examples:
        [기재정정]주요사항보고서(자기주식 취득 결정)
        -> 주요사항보고서(자기주식취득결정)
    """

    value = unicodedata.normalize("NFKC", report_name or "")
    value = LEADING_DISCLOSURE_TAGS.sub("", value)
    # NFKC turns the compatibility character U+318D (ㆍ) into the
    # conjoining jamo U+119E (ᆞ), so fold both forms back to one delimiter.
    value = (
        value.replace("·", "ㆍ")
        .replace("・", "ㆍ")
        .replace("ᆞ", "ㆍ")
    )
    value = ALL_WHITESPACE.sub("", value)
    return value.strip()


def normalize_report_type(document: dict[str, Any]) -> str | None:
    """Return a canonical report type, or ``None`` when not mapped."""

    doc_group = document.get("doc_group")
    doc_subtype = document.get("doc_subtype")
    report_name = canonical_report_name(document.get("report_nm", ""))

    subtype_key = (doc_group, doc_subtype)
    if subtype_key in SUBTYPE_REPORT_TYPE:
        return SUBTYPE_REPORT_TYPE[subtype_key]

    if doc_group == "holding":
        return HOLDING_REPORT_NAME_MAP.get(report_name)

    if doc_group == "major":
        return MAJOR_REPORT_NAME_MAP.get(report_name)

    return None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error


def enrich_manifest(
    input_path: Path,
    output_path: Path,
    *,
    allow_unknown: bool = False,
) -> Counter[str]:
    """Write a new manifest and return the normalized-type distribution."""

    documents = list(read_jsonl(input_path))
    enriched_documents: list[dict[str, Any]] = []
    unknown_documents: list[dict[str, Any]] = []
    distribution: Counter[str] = Counter()

    for document in documents:
        report_type = normalize_report_type(document)

        if report_type is None:
            report_type = "unknown"
            unknown_documents.append(
                {
                    "doc_id": document.get("doc_id"),
                    "doc_group": document.get("doc_group"),
                    "doc_subtype": document.get("doc_subtype"),
                    "report_nm": document.get("report_nm"),
                }
            )

        # Dict insertion order preserves all original fields and appends the
        # new field at the end of each JSON object.
        enriched = dict(document)
        enriched["normalized_report_type"] = report_type
        enriched_documents.append(enriched)
        distribution[report_type] += 1

    if unknown_documents and not allow_unknown:
        preview = json.dumps(
            unknown_documents[:20], ensure_ascii=False, indent=2
        )
        raise ValueError(
            f"{len(unknown_documents)} document(s) could not be mapped. "
            "No output was written. First cases:\n"
            f"{preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for document in enriched_documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    return distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add normalized_report_type to corpus/manifest.jsonl"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the original manifest.jsonl",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for the new manifest_v1.jsonl",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Write unknown instead of failing for unmapped documents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distribution = enrich_manifest(
        args.input,
        args.output,
        allow_unknown=args.allow_unknown,
    )

    print(f"Created: {args.output}")
    print(f"Documents: {sum(distribution.values()):,}")
    print("normalized_report_type distribution:")
    for report_type, count in sorted(distribution.items()):
        print(f"  {report_type}: {count:,}")


if __name__ == "__main__":
    main()