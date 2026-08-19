#!/usr/bin/env python3
"""Add ``event_date`` to the enriched DART corpus manifest.

``event_date`` means the date on which the disclosed event or decision
occurred.  It is deliberately not replaced with ``rcept_dt`` when extraction
fails.  Periodic reports have no single event date and therefore receive
``null``.

The input manifest is never modified.  Dates are written as ``YYYYMMDD`` to
match the format of the existing ``rcept_dt`` field.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "3.0.0-form-variants"


PERIODIC_REPORT_TYPES = {
    "annual_report",
    "semiannual_report",
    "quarterly_report",
}


# DART major/holding forms store dates in TU elements.  The AUNIT code is a
# stable form-field identifier; AUNITVALUE contains the machine-readable date.
XML_DATE_UNIT_BY_REPORT_TYPE: dict[str, tuple[str, ...]] = {
    "large_shareholding_general": ("RPT_RSP_DT",),
    "large_shareholding_summary": ("RPT_RSP_DT",),
    # Some stock-option or employee-plan filings have no separate decision
    # date.  In those forms the planned transaction start date is the most
    # specific disclosed event date.
    "treasury_stock_disposal_decision": ("SEL_DATE", "SEL_BGN"),
    "treasury_stock_acquisition_decision": ("ACQ_DATE", "ACQ_BGN"),
    "write_down_contingent_capital_security_issuance_decision": ("DRC_DT",),
    "paid_in_capital_increase_decision": ("DRC_DT",),
    "treasury_stock_trust_agreement_conclusion_decision": ("DEC_DATE",),
    # This form exposes the scheduled termination date, not a separate
    # machine-readable board-decision date.
    "treasury_stock_trust_agreement_termination_decision": ("CNL_SCH",),
    "merger_decision": ("DRT_RSLT_DT",),
    "convertible_bond_issuance_decision": ("DRC_DT",),
    "stock_exchange_or_transfer_decision": ("DRT_RSLT_DT",),
    "company_split_decision": ("DRT_RSLT_DT",),
    "split_merger_decision": ("DRT_RSLT_DT",),
    "other_company_shares_acquisition_decision": ("DRT_RSLT_DT",),
    "other_company_shares_disposal_decision": ("DRT_RSLT_DT",),
    "capital_reduction_decision": ("DRC_DT",),
    "regulatory_capital_debt_security_issuance_decision": ("DRC_DT",),
    "bonus_issue_decision": ("DRC_DT",),
    "lawsuit_filing": ("APPL_DT",),
    "exchangeable_bond_issuance_decision": ("DRC_DT",),
    # DART has both decision and completion/confirmation variants under the
    # same normalized report type.
    "overseas_securities_delisting_decision": (
        "DRC_DT",
        "CONF_DT",
        "TRD_END_DT",
    ),
    "overseas_securities_listing_decision": (
        "DRC_DT",
        "CONF_DT",
        "TRD_BGN_DT",
    ),
    "tangible_asset_disposal_decision": ("DRT_RSLT_DT",),
    "tangible_asset_acquisition_decision": ("DRT_RSLT_DT",),
    "business_acquisition_decision": ("DRT_RSLT_DT",),
    "own_convertible_bond_sale_decision": ("SELD_DT",),
    "business_suspension": ("OSM_234_01",),
    "third_party_convertible_bond_call_option_exercise": ("EXC_DT",),
}


# KRX exchange disclosures are HTML tables rather than DART TU-form XML.
# Labels are normalized before matching, so spacing differences are harmless.
EXCHANGE_LABEL_BY_REPORT_TYPE: dict[str, tuple[str, ...]] = {
    "supply_contract": ("계약(수주)일자", "계약(수주)일"),
    "supply_contract_termination": ("해지일자", "계약해지일", "해지일"),
    "facility_investment": ("이사회결의일(결정일)",),
    "material_management_matter": (
        "이사회결의일(결정일)또는사실확인일",
        "사실발생(확인)일",
        "사실발생(확인일)",
        "사실확인일",
        "결정일",
        "이사회결의일(결정일)",
    ),
}


TU_TAG = re.compile(r"<TU\b[^>]*>", re.IGNORECASE)
ATTRIBUTE = re.compile(
    r"([A-Za-z_:][\w:.-]*)\s*=\s*(['\"])(.*?)\2",
    re.DOTALL,
)
DATE_PATTERNS = (
    re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"),
)
WHITESPACE = re.compile(r"\s+")


class TableRowParser(HTMLParser):
    """Collect visible text from each cell of each HTML table row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None:
            assert self._row is not None
            text = WHITESPACE.sub(" ", "".join(self._cell_parts)).strip()
            self._row.append(text)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
            self._cell_parts = None


def normalize_date(value: str) -> str | None:
    """Return a valid calendar date as YYYYMMDD, otherwise ``None``."""

    for pattern in DATE_PATTERNS:
        match = pattern.search(value or "")
        if not match:
            continue
        year, month, day = (int(part) for part in match.groups())
        compact = f"{year:04d}{month:02d}{day:02d}"
        try:
            datetime.strptime(compact, "%Y%m%d")
        except ValueError:
            continue
        return compact
    return None


def normalize_label(value: str) -> str:
    return WHITESPACE.sub("", value).replace("ㆍ", "").replace("·", "")


def extract_tu_date(text: str, unit_names: tuple[str, ...]) -> str | None:
    """Extract the last valid matching TU value (the corrected main form)."""

    values_by_unit: dict[str, list[str]] = {name: [] for name in unit_names}
    for tag in TU_TAG.findall(text):
        attrs = {
            name.upper(): value.strip()
            for name, _, value in ATTRIBUTE.findall(tag)
        }
        unit_name = attrs.get("AUNIT")
        if unit_name in values_by_unit:
            value = attrs.get("AUNITVALUE", "")
            normalized = normalize_date(value)
            if normalized:
                values_by_unit[unit_name].append(normalized)

    for unit_name in unit_names:
        if values_by_unit[unit_name]:
            return values_by_unit[unit_name][-1]
    return None


def extract_exchange_date(text: str, labels: tuple[str, ...]) -> str | None:
    """Extract a date from the value cell following a KRX form label."""

    parser = TableRowParser()
    parser.feed(text)
    normalized_labels = tuple(normalize_label(label) for label in labels)
    # Labels are ordered by semantic preference.  For example, a factual
    # occurrence date is preferred to a generic decision date.  Within the
    # same label, the last occurrence is the complete/current corrected form.
    for label in normalized_labels:
        candidates: list[str] = []
        for row in parser.rows:
            for index, cell in enumerate(row):
                normalized_cell = normalize_label(cell)
                if label not in normalized_cell:
                    continue
                # The event date is in a value cell after the label cell.  Do
                # not scan cells before it: correction tables may contain
                # unrelated dates there.
                for value_cell in row[index + 1 :]:
                    date = normalize_date(value_cell)
                    if date:
                        candidates.append(date)
                        break
        if candidates:
            return candidates[-1]
    return None


def contains_tu_unit(text: str, unit_names: tuple[str, ...]) -> bool:
    """Return whether the form contains one of the mapped date fields."""

    wanted = set(unit_names)
    for tag in TU_TAG.findall(text):
        attrs = {
            name.upper(): value.strip()
            for name, _, value in ATTRIBUTE.findall(tag)
        }
        if attrs.get("AUNIT") in wanted:
            return True
    return False


def contains_exchange_label(text: str, labels: tuple[str, ...]) -> bool:
    parser = TableRowParser()
    parser.feed(text)
    wanted = tuple(normalize_label(label) for label in labels)
    return any(
        any(label in normalize_label(cell) for label in wanted)
        for row in parser.rows
        for cell in row
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error


def document_files(document: dict[str, Any], corpus_root: Path) -> list[Path]:
    relative_path = document.get("file_path")
    if not relative_path:
        return []

    location = resolve_normalized_path(corpus_root, Path(relative_path))
    if location.is_file():
        return [location]

    receipt_number = str(document.get("rcept_no", ""))
    exact = location / f"{receipt_number}.xml"
    if exact.is_file():
        return [exact]
    return sorted(location.glob("*.xml")) if location.is_dir() else []


def resolve_normalized_path(root: Path, relative_path: Path) -> Path:
    """Resolve a corpus path even when Git stored Korean names as NFD.

    macOS-originated Git paths can contain decomposed Hangul (NFD), while the
    manifest contains composed Hangul (NFC).  They look identical in VS Code
    but compare as different strings on filesystems that preserve both forms.
    """

    current = root
    for part in relative_path.parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue

        if not current.is_dir():
            return direct

        normalized_part = unicodedata.normalize("NFC", part).casefold()
        matches = [
            child
            for child in current.iterdir()
            if unicodedata.normalize("NFC", child.name).casefold()
            == normalized_part
        ]
        if len(matches) != 1:
            return direct
        current = matches[0]

    return current


def extract_event_date(
    document: dict[str, Any], corpus_root: Path
) -> tuple[str | None, str]:
    """Return ``(event_date, status)`` for one manifest document."""

    report_type = document.get("normalized_report_type")
    if report_type in PERIODIC_REPORT_TYPES:
        return None, "not_applicable"

    unit_names = XML_DATE_UNIT_BY_REPORT_TYPE.get(report_type)
    labels = EXCHANGE_LABEL_BY_REPORT_TYPE.get(report_type)
    if unit_names is None and labels is None:
        return None, "unsupported_report_type"

    files = document_files(document, corpus_root)
    if not files:
        return None, "file_not_found"

    candidates: list[str] = []
    found_mapped_field = False
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if unit_names is not None:
            found_mapped_field |= contains_tu_unit(text, unit_names)
            value = extract_tu_date(text, unit_names)
        else:
            exchange_labels = labels or ()
            found_mapped_field |= contains_exchange_label(text, exchange_labels)
            value = extract_exchange_date(text, exchange_labels)
        if value:
            candidates.append(value)

    if candidates:
        return candidates[-1], "extracted"
    if found_mapped_field:
        # The XML explicitly contains the correct field but its value is '-'
        # or blank.  Preserve that source fact as null instead of inventing a
        # receipt-date fallback.
        return None, "not_disclosed"
    return None, "date_not_found"


def enrich_manifest(
    input_path: Path,
    corpus_root: Path,
    output_path: Path,
    *,
    allow_unresolved: bool = False,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    documents = list(read_jsonl(input_path))
    enriched_documents: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []

    for document in documents:
        event_date, status = extract_event_date(document, corpus_root)
        status_counts[status] += 1

        enriched = dict(document)
        enriched["event_date"] = event_date
        enriched_documents.append(enriched)

        if status not in {"extracted", "not_applicable", "not_disclosed"}:
            unresolved.append(
                {
                    "doc_id": document.get("doc_id"),
                    "normalized_report_type": document.get(
                        "normalized_report_type"
                    ),
                    "report_nm": document.get("report_nm"),
                    "file_path": document.get("file_path"),
                    "status": status,
                }
            )

    if unresolved and not allow_unresolved:
        review_path = output_path.with_name(
            f"{output_path.stem}_event_date_review.jsonl"
        )
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8", newline="\n") as file:
            for item in unresolved:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        preview = json.dumps(unresolved[:20], ensure_ascii=False, indent=2)
        raise ValueError(
            f"{len(unresolved)} non-periodic document(s) have no event_date. "
            "No output was written. First cases:\n"
            f"{preview}\n"
            f"Full review list: {review_path}\n"
            "Review the extraction rules, or use --allow-unresolved only for "
            "an exploratory output."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for document in enriched_documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    return status_counts, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add event_date to manifest_v1.jsonl from source XML"
    )
    parser.add_argument("input", type=Path, help="Input manifest_v1.jsonl")
    parser.add_argument(
        "corpus_root",
        type=Path,
        help="Corpus directory containing raw/ (normally: corpus)",
    )
    parser.add_argument("output", type=Path, help="Output manifest_v2.jsonl")
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write null for unresolved non-periodic documents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}")
    status_counts, unresolved = enrich_manifest(
        args.input,
        args.corpus_root,
        args.output,
        allow_unresolved=args.allow_unresolved,
    )

    print(f"Created: {args.output}")
    print(f"Documents: {sum(status_counts.values()):,}")
    print("event_date extraction status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count:,}")
    if unresolved:
        print(f"WARNING: unresolved non-periodic documents: {len(unresolved):,}")


if __name__ == "__main__":
    main()