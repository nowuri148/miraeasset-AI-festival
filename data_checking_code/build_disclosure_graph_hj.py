#!/usr/bin/env python3
"""Build disclosure-chain metadata and graph edges from the DART corpus.

Outputs
-------
1. ``manifest_v3.jsonl``
   Preserves every field in ``manifest_v2.jsonl`` and appends
   ``disclosure_chain_id`` and ``relation_status``.  The latter is null for
   ordinary documents and is either ``resolved`` or ``external_source`` for
   correction documents in a successful strict-mode build.
2. ``graph_edges_v1.jsonl``
   Materializes Company-[:FILED]->Document and
   CorrectionDocument-[:CORRECTS]->PreviousDocument triples.

Matching policy
---------------
* ``is_correction`` only says that a document is a correction; it never
  selects the target by itself.
* Exact receipt numbers referenced in the XML are preferred.
* Otherwise, the correction header date is matched with company, report
  group/type and group-specific identity fields.
* A correction whose referenced source is outside this corpus receives a
  stable external chain ID, but no dangling CORRECTS edge is written.
* Ambiguous or structurally unparseable corrections fail in strict mode and
  are written to a review JSONL instead of being guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "1.2.1-scoped-receipt-alias"

LEADING_DISCLOSURE_TAGS = re.compile(r"^(?:\s*\[[^\]]+\]\s*)+")
HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")
RECEIPT_IN_HREF = re.compile(
    r"(?:acptno|rcpno)=([0-9]{14})",
    re.IGNORECASE,
)
ANCHOR = re.compile(
    r"<a\b(?P<attrs>[^>]*)>(?P<label>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)

DATE_PATTERNS = (
    re.compile(
        r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*"
        r"(\d{1,2})(?!\d)"
    ),
    re.compile(
        # A few source filings contain the typo '08년 28일' instead of
        # '08월 28일', so accept either marker for the month component.
        r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*(?:월|년)\s*"
        r"(\d{1,2})\s*일?(?!\d)"
    ),
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"),
)

INITIAL_SUBMISSION_LABELS = (
    "정정대상 공시서류의 최초제출일",
    "정정대상 공시서류의 최초 제출일",
    "정정대상공시서류의최초제출일",
)
PREVIOUS_SUBMISSION_LABELS = (
    "정정관련 공시서류제출일",
    "정정관련공시서류제출일",
)


@dataclass(frozen=True)
class CorrectionReference:
    kind: str | None
    date: str | None
    receipt_numbers: tuple[str, ...]
    dated_receipts: tuple[tuple[str, str | None], ...]


@dataclass
class Resolution:
    target_doc_id: str | None
    chain_id: str
    status: str
    match_method: str | None
    confidence: str | None
    reference: CorrectionReference
    candidate_doc_ids: list[str]


def normalize_date(value: str) -> str | None:
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


def plain_text(source: str) -> str:
    value = HTML_TAG.sub(" ", source)
    value = html.unescape(value)
    return WHITESPACE.sub(" ", value).strip()


def canonical_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = LEADING_DISCLOSURE_TAGS.sub("", normalized)
    normalized = (
        normalized.replace("·", "ㆍ")
        .replace("・", "ㆍ")
        .replace("ᆞ", "ㆍ")
    )
    return WHITESPACE.sub("", normalized).strip().casefold()


def find_date_after_label(text: str, labels: tuple[str, ...]) -> str | None:
    compact_text = WHITESPACE.sub(" ", text)
    for label in labels:
        start = compact_text.find(label)
        if start < 0:
            # Some KRX forms omit spaces inside the label.
            no_space_text = WHITESPACE.sub("", compact_text)
            no_space_label = WHITESPACE.sub("", label)
            start = no_space_text.find(no_space_label)
            if start >= 0:
                return normalize_date(
                    no_space_text[start + len(no_space_label) : start + 120]
                )
            continue
        return normalize_date(
            compact_text[start + len(label) : start + len(label) + 160]
        )
    return None


def extract_anchor_receipts(source: str) -> tuple[tuple[str, str | None], ...]:
    found: list[tuple[str, str | None]] = []
    for match in ANCHOR.finditer(source):
        attrs = html.unescape(match.group("attrs"))
        numbers = RECEIPT_IN_HREF.findall(attrs)
        if not numbers:
            continue
        label = plain_text(match.group("label"))
        label_date = normalize_date(label)
        # acptno and rcpno usually repeat the same number in one link.
        for receipt_number in dict.fromkeys(numbers):
            found.append((receipt_number, label_date))
    return tuple(dict.fromkeys(found))


def extract_correction_reference(source: str) -> CorrectionReference:
    text = plain_text(source)
    previous_date = find_date_after_label(text, PREVIOUS_SUBMISSION_LABELS)
    initial_date = find_date_after_label(text, INITIAL_SUBMISSION_LABELS)

    if previous_date:
        kind = "previous_submission_date"
        reference_date = previous_date
    elif initial_date:
        kind = "initial_submission_date"
        reference_date = initial_date
    else:
        kind = None
        reference_date = None

    dated_receipts = extract_anchor_receipts(source)
    if reference_date:
        matching = [
            receipt
            for receipt, date in dated_receipts
            if date == reference_date
        ]
    else:
        matching = [receipt for receipt, _ in dated_receipts]

    return CorrectionReference(
        kind=kind,
        date=reference_date,
        receipt_numbers=tuple(dict.fromkeys(matching)),
        dated_receipts=dated_receipts,
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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_normalized_path(root: Path, relative_path: Path) -> Path:
    """Resolve NFC manifest paths against NFC/NFD filesystem names."""

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


def load_document_source(
    document: dict[str, Any], corpus_root: Path
) -> tuple[str | None, str | None]:
    files = document_files(document, corpus_root)
    if not files:
        return None, "file_not_found"
    parts = [path.read_text(encoding="utf-8-sig") for path in files]
    return "\n".join(parts), None


def document_order(document: dict[str, Any]) -> tuple[str, str]:
    return (
        str(document.get("rcept_dt") or ""),
        str(document.get("rcept_no") or ""),
    )


def receipt_aliases(document: dict[str, Any]) -> tuple[str, ...]:
    """Return corpus and KRX-link representations of a receipt number.

    Exchange documents use an 8/9 market marker at position 9 in the corpus
    (for example 20241125800033), while KRX hyperlinks can expose the same
    receipt as 20241125000033.  Both are exact identifiers for one filing.
    """

    receipt = str(document.get("rcept_no") or "")
    aliases = [receipt] if receipt else []
    if document.get("doc_group") == "exchange" and len(receipt) == 14:
        aliases.append(f"{receipt[:8]}0{receipt[9:]}")
    return tuple(dict.fromkeys(aliases))


def receipt_lookup_key(
    document: dict[str, Any], receipt_number: str
) -> tuple[str, str, str]:
    """Scope receipt aliases to one filing system and one company.

    A KRX alias such as 20250627000228 can legitimately equal a DART/holding
    receipt owned by another document.  Receipt aliases are therefore unique
    only within doc_group and corp_code, not across the entire corpus.
    """

    return (
        str(document.get("doc_group") or ""),
        str(document.get("corp_code") or ""),
        receipt_number,
    )


def extract_tag_value(source: str, code: str) -> str | None:
    pattern = re.compile(
        rf"<(?:TE|TU)\b[^>]*(?:ACODE|AUNIT)=[\"']{re.escape(code)}"
        rf"[\"'][^>]*>(.*?)</(?:TE|TU)>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(source)
    return canonical_text(plain_text(match.group(1))) if match else None


def event_identity_fingerprint(
    document: dict[str, Any],
    corpus_root: Path,
    cache: dict[str, tuple[str, ...] | None],
) -> tuple[str, ...] | None:
    """Extract type-specific identity fields only when metadata is ambiguous."""

    doc_id = str(document.get("doc_id"))
    if doc_id in cache:
        return cache[doc_id]

    source, error = load_document_source(document, corpus_root)
    if error or source is None:
        cache[doc_id] = None
        return None

    report_type = document.get("normalized_report_type")
    if report_type == "stock_exchange_or_transfer_decision":
        target_company = extract_tag_value(source, "EXCH_NM")
        result = ("exchange_target", target_company) if target_company else None
        cache[doc_id] = result
        return result

    if report_type == "supply_contract":
        text = plain_text(source)

        def capture(pattern: str) -> str:
            match = re.search(pattern, text, re.IGNORECASE)
            return canonical_text(match.group(1)) if match else ""

        contract_name = capture(r"체결계약명\s+(.+?)\s+2\.\s*계약내역")
        counterparty = capture(r"3\.\s*계약상대\s+(.+?)\s+-\s*회사와의\s*관계")
        period = re.search(
            r"5\.\s*계약기간\s+시작일\s+(.+?)\s+종료일\s+(.+?)\s+"
            r"6\.\s*주요\s*계약조건",
            text,
            re.IGNORECASE,
        )
        start_date = normalize_date(period.group(1)) if period else None
        end_date = normalize_date(period.group(2)) if period else None
        result = (
            "supply_contract",
            contract_name,
            counterparty,
            start_date or "",
            end_date or "",
            str(document.get("event_date") or ""),
        )
        cache[doc_id] = result
        return result

    if report_type == "material_management_matter":
        text = plain_text(source)
        # report_nm is sometimes only the generic form name.  The numbered
        # title inside the KRX form identifies which event is being corrected.
        match = re.search(
            r"1\.\s*(?:제목|공시내용)\s+(.+?)\s+2\.\s*(?:주요내용|주요사항)",
            text,
            re.IGNORECASE,
        )
        event_title = unicodedata.normalize("NFKC", match.group(1)) if match else ""
        stopwords = {
            "투자판단", "관련", "주요경영사항", "변경", "안내", "계약",
            "정정", "공시", "사항", "주요내용",
        }
        keywords = tuple(
            sorted(
                {
                    token.casefold()
                    for token in re.findall(r"[0-9A-Za-z가-힣]{3,}", event_title)
                    if token not in stopwords
                }
            )
        )
        result = (
            ("material_management_matter", *keywords)
            if keywords
            else None
        )
        cache[doc_id] = result
        return result

    cache[doc_id] = None
    return None


def is_earlier(candidate: dict[str, Any], correction: dict[str, Any]) -> bool:
    return document_order(candidate) < document_order(correction)


def compatible_document(
    correction: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    if correction.get("corp_code") != candidate.get("corp_code"):
        return False
    if correction.get("doc_group") != candidate.get("doc_group"):
        return False
    if (
        correction.get("normalized_report_type")
        != candidate.get("normalized_report_type")
    ):
        return False

    group = correction.get("doc_group")
    if group == "periodic":
        return (
            correction.get("base_year") == candidate.get("base_year")
            and correction.get("base_month") == candidate.get("base_month")
        )

    if group == "holding":
        correction_filer = canonical_text(correction.get("flr_nm"))
        candidate_filer = canonical_text(candidate.get("flr_nm"))
        if (
            correction_filer
            and candidate_filer
            and correction_filer != candidate_filer
        ):
            return False

    if group == "exchange":
        correction_title = canonical_text(correction.get("report_nm"))
        candidate_title = canonical_text(candidate.get("report_nm"))
        # Detailed material-management titles distinguish different events.
        if correction.get("normalized_report_type") == "material_management_matter":
            if correction_title != candidate_title:
                return False

    correction_event_date = correction.get("event_date")
    candidate_event_date = candidate.get("event_date")
    if correction_event_date and candidate_event_date:
        return correction_event_date == candidate_event_date
    return True


def identity_signature(
    document: dict[str, Any], reference: CorrectionReference
) -> tuple[str, ...]:
    group = str(document.get("doc_group") or "")
    parts = [
        group,
        str(document.get("corp_code") or ""),
        str(document.get("normalized_report_type") or ""),
        str(reference.date or "unknown-date"),
    ]

    if group == "periodic":
        parts.extend(
            [
                str(document.get("base_year") or ""),
                str(document.get("base_month") or ""),
            ]
        )
    elif group == "holding":
        parts.append(canonical_text(document.get("flr_nm")))
    elif group == "exchange" and document.get(
        "normalized_report_type"
    ) == "material_management_matter":
        parts.append(canonical_text(document.get("report_nm")))

    event_date = document.get("event_date")
    if event_date:
        parts.append(str(event_date))
    return tuple(parts)


def external_chain_id(
    document: dict[str, Any],
    reference: CorrectionReference,
    external_receipt: str | None = None,
) -> str:
    group = str(document.get("doc_group") or "document")
    if external_receipt:
        return f"chain_external_{group}_{external_receipt}"

    signature = "|".join(identity_signature(document, reference))
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    corp_code = str(document.get("corp_code") or "unknown")
    date = str(reference.date or "unknown")
    return f"chain_external_{group}_{corp_code}_{date}_{digest}"


def select_unique_candidate(
    correction: dict[str, Any],
    candidates: list[dict[str, Any]],
    chain_by_doc_id: dict[str, str],
    corpus_root: Path,
    fingerprint_cache: dict[str, tuple[str, ...] | None],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = [
        item
        for item in candidates
        if is_earlier(item, correction)
        and compatible_document(correction, item)
    ]
    if len(candidates) <= 1:
        return (candidates[0] if candidates else None), candidates

    correction_title = canonical_text(correction.get("report_nm"))
    exact_title = [
        item
        for item in candidates
        if canonical_text(item.get("report_nm")) == correction_title
    ]
    if len(exact_title) == 1:
        return exact_title[0], exact_title
    if exact_title:
        candidates = exact_title

    correction_fingerprint = event_identity_fingerprint(
        correction, corpus_root, fingerprint_cache
    )
    if correction_fingerprint is not None:
        matching_fingerprints = [
            item
            for item in candidates
            if event_identity_fingerprint(item, corpus_root, fingerprint_cache)
            == correction_fingerprint
        ]
        if len(matching_fingerprints) == 1:
            return matching_fingerprints[0], matching_fingerprints
        if matching_fingerprints:
            candidates = matching_fingerprints

        if (
            len(candidates) > 1
            and correction_fingerprint[0] == "material_management_matter"
        ):
            correction_terms = set(correction_fingerprint[1:])
            scored: list[tuple[int, dict[str, Any]]] = []
            for candidate in candidates:
                candidate_fingerprint = event_identity_fingerprint(
                    candidate, corpus_root, fingerprint_cache
                )
                candidate_terms = (
                    set(candidate_fingerprint[1:])
                    if candidate_fingerprint
                    and candidate_fingerprint[0]
                    == "material_management_matter"
                    else set()
                )
                scored.append(
                    (len(correction_terms & candidate_terms), candidate)
                )
            scored.sort(key=lambda pair: pair[0], reverse=True)
            if scored[0][0] > 0 and scored[0][0] > scored[1][0]:
                return scored[0][1], [pair[1] for pair in scored]

    known_chains = {
        chain_by_doc_id.get(str(item.get("doc_id"))) for item in candidates
    }
    known_chains.discard(None)
    if len(known_chains) == 1:
        return max(candidates, key=document_order), candidates
    return None, candidates


def periodic_metadata_candidates(
    correction: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find the same periodic filing without depending on an XML header.

    Some periodic corrections in the supplied corpus contain only PDF/viewer
    files.  Company, report type, base year and base month jointly identify
    their disclosure series.  The candidate must still predate the correction.
    """

    if correction.get("doc_group") != "periodic":
        return []
    return [
        candidate
        for candidate in documents
        if candidate is not correction
        and is_earlier(candidate, correction)
        and compatible_document(correction, candidate)
    ]


def build_graph_artifacts(
    input_manifest: Path,
    corpus_root: Path,
    output_manifest: Path,
    output_edges: Path,
    *,
    allow_unresolved: bool = False,
) -> tuple[Counter[str], Path]:
    documents = list(read_jsonl(input_manifest))
    if not documents:
        raise ValueError(f"No documents found in {input_manifest}")

    doc_by_id = {str(item["doc_id"]): item for item in documents}
    doc_by_receipt: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in documents:
        for alias in receipt_aliases(item):
            lookup_key = receipt_lookup_key(item, alias)
            existing = doc_by_receipt.get(lookup_key)
            if existing is not None and existing["doc_id"] != item["doc_id"]:
                raise ValueError(
                    f"Receipt alias collision within group/company: "
                    f"{lookup_key} maps to both "
                    f"{existing['doc_id']} and {item['doc_id']}"
                )
            doc_by_receipt[lookup_key] = item
    if len(doc_by_id) != len(documents):
        raise ValueError("Duplicate doc_id values exist in the input manifest")

    by_date: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in documents:
        by_date[str(item.get("rcept_dt") or "")].append(item)

    chain_by_doc_id: dict[str, str] = {
        str(item["doc_id"]): str(item["doc_id"])
        for item in documents
        if not item.get("is_correction")
    }
    relation_status_by_doc_id: dict[str, str | None] = {
        str(item["doc_id"]): None for item in documents
    }
    latest_correction_by_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    review: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    fingerprint_cache: dict[str, tuple[str, ...] | None] = {}

    for item in documents:
        edge = {
            "source_type": "Company",
            "source_id": str(item.get("corp_code") or ""),
            "edge_type": "FILED",
            "target_type": "Document",
            "target_id": str(item["doc_id"]),
            "match_method": "manifest_corp_code",
            "confidence": "exact",
        }
        key = (edge["source_id"], edge["edge_type"], edge["target_id"])
        if key not in edge_keys:
            edges.append(edge)
            edge_keys.add(key)

    corrections = sorted(
        (item for item in documents if item.get("is_correction")),
        key=document_order,
    )

    for correction in corrections:
        doc_id = str(correction["doc_id"])
        source, source_error = load_document_source(correction, corpus_root)
        if source_error and correction.get("doc_group") != "periodic":
            reference = CorrectionReference(None, None, (), ())
            resolution = Resolution(
                target_doc_id=None,
                chain_id=external_chain_id(correction, reference),
                status=source_error,
                match_method=None,
                confidence=None,
                reference=reference,
                candidate_doc_ids=[],
            )
        else:
            reference = (
                extract_correction_reference(source)
                if source is not None
                else CorrectionReference(None, None, (), ())
            )
            signature = identity_signature(correction, reference)
            target: dict[str, Any] | None = None
            considered: list[dict[str, Any]] = []
            match_method: str | None = None
            confidence: str | None = None

            exact_candidates = [
                doc_by_receipt[
                    receipt_lookup_key(correction, number)
                ]
                for number in reference.receipt_numbers
                if receipt_lookup_key(correction, number) in doc_by_receipt
            ]
            target, considered = select_unique_candidate(
                correction,
                exact_candidates,
                chain_by_doc_id,
                corpus_root,
                fingerprint_cache,
            )
            if target:
                match_method = "referenced_receipt_number"
                confidence = "exact"

            if target is None and reference.date:
                date_candidates = list(by_date.get(reference.date, []))
                if reference.kind == "initial_submission_date":
                    roots = [
                        candidate
                        for candidate in date_candidates
                        if not candidate.get("is_correction")
                    ]
                    target, considered = select_unique_candidate(
                        correction,
                        roots,
                        chain_by_doc_id,
                        corpus_root,
                        fingerprint_cache,
                    )
                else:
                    target, considered = select_unique_candidate(
                        correction,
                        date_candidates,
                        chain_by_doc_id,
                        corpus_root,
                        fingerprint_cache,
                    )
                if target:
                    match_method = reference.kind
                    confidence = "high"

            if target is None and correction.get("doc_group") == "periodic":
                target, considered = select_unique_candidate(
                    correction,
                    periodic_metadata_candidates(correction, documents),
                    chain_by_doc_id,
                    corpus_root,
                    fingerprint_cache,
                )
                if target:
                    match_method = "periodic_report_period"
                    confidence = "high"

            if target is None and reference.kind == "initial_submission_date":
                previous_correction = latest_correction_by_signature.get(signature)
                if previous_correction is not None:
                    target = previous_correction
                    considered = [previous_correction]
                    match_method = "same_external_chain_sequence"
                    confidence = "medium"

            if target is not None:
                target_doc_id = str(target["doc_id"])
                chain_id = chain_by_doc_id.get(target_doc_id, target_doc_id)
                resolution = Resolution(
                    target_doc_id=target_doc_id,
                    chain_id=chain_id,
                    status="resolved",
                    match_method=match_method,
                    confidence=confidence,
                    reference=reference,
                    candidate_doc_ids=[
                        str(candidate["doc_id"]) for candidate in considered
                    ],
                )
            else:
                external_numbers = [
                    number
                    for number in reference.receipt_numbers
                    if receipt_lookup_key(correction, number)
                    not in doc_by_receipt
                    and number != str(correction.get("rcept_no"))
                ]
                external_number = (
                    external_numbers[0] if len(external_numbers) == 1 else None
                )

                if considered:
                    status = "ambiguous_candidates"
                elif correction.get("doc_group") == "periodic":
                    status = "external_source"
                elif reference.date or external_number:
                    status = "external_source"
                else:
                    status = "missing_correction_reference"

                resolution = Resolution(
                    target_doc_id=None,
                    chain_id=external_chain_id(
                        correction, reference, external_number
                    ),
                    status=status,
                    match_method=(
                        "external_receipt_number"
                        if external_number
                        else reference.kind
                    ),
                    confidence=("exact" if external_number else None),
                    reference=reference,
                    candidate_doc_ids=[
                        str(candidate["doc_id"]) for candidate in considered
                    ],
                )

        chain_by_doc_id[doc_id] = resolution.chain_id
        relation_status_by_doc_id[doc_id] = (
            resolution.status
            if resolution.status in {"resolved", "external_source"}
            else None
        )
        status_counts[resolution.status] += 1

        if source_error is None:
            latest_correction_by_signature[
                identity_signature(correction, resolution.reference)
            ] = correction

        if resolution.target_doc_id:
            edge = {
                "source_type": "Document",
                "source_id": doc_id,
                "edge_type": "CORRECTS",
                "target_type": "Document",
                "target_id": resolution.target_doc_id,
                "match_method": resolution.match_method,
                "confidence": resolution.confidence,
                "reference_date": resolution.reference.date,
            }
            key = (edge["source_id"], edge["edge_type"], edge["target_id"])
            if key not in edge_keys:
                edges.append(edge)
                edge_keys.add(key)

        if resolution.status != "resolved":
            review.append(
                {
                    "doc_id": doc_id,
                    "corp_code": correction.get("corp_code"),
                    "corp_name": correction.get("corp_name"),
                    "doc_group": correction.get("doc_group"),
                    "normalized_report_type": correction.get(
                        "normalized_report_type"
                    ),
                    "rcept_dt": correction.get("rcept_dt"),
                    "status": resolution.status,
                    "reference_kind": resolution.reference.kind,
                    "reference_date": resolution.reference.date,
                    "referenced_receipt_numbers": list(
                        resolution.reference.receipt_numbers
                    ),
                    "candidate_doc_ids": resolution.candidate_doc_ids,
                    "assigned_chain_id": resolution.chain_id,
                }
            )

    blocking = [
        item
        for item in review
        if item["status"]
        in {
            "ambiguous_candidates",
            "missing_correction_reference",
            "file_not_found",
        }
    ]
    review_path = output_edges.with_name(
        f"{output_edges.stem}_review.jsonl"
    )
    if review:
        write_jsonl(review_path, review)

    if blocking and not allow_unresolved:
        preview = json.dumps(blocking[:20], ensure_ascii=False, indent=2)
        raise ValueError(
            f"{len(blocking)} correction document(s) require review. "
            "Primary outputs were not written.\n"
            f"Full review list: {review_path}\n"
            f"First cases:\n{preview}\n"
            "Fix the matching rules, or use --allow-unresolved only for an "
            "exploratory output."
        )

    enriched_documents: list[dict[str, Any]] = []
    for item in documents:
        enriched = dict(item)
        enriched["disclosure_chain_id"] = chain_by_doc_id[str(item["doc_id"])]
        enriched["relation_status"] = relation_status_by_doc_id[
            str(item["doc_id"])
        ]
        enriched_documents.append(enriched)

    write_jsonl(output_manifest, enriched_documents)
    write_jsonl(output_edges, edges)

    status_counts["filed_edges"] = sum(
        1 for edge in edges if edge["edge_type"] == "FILED"
    )
    status_counts["corrects_edges"] = sum(
        1 for edge in edges if edge["edge_type"] == "CORRECTS"
    )
    return status_counts, review_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add disclosure_chain_id/relation_status and create "
            "FILED/CORRECTS graph edges"
        )
    )
    parser.add_argument("input", type=Path, help="Input manifest_v2.jsonl")
    parser.add_argument(
        "corpus_root",
        type=Path,
        help="Corpus directory containing raw/ (normally: corpus)",
    )
    parser.add_argument("manifest_output", type=Path)
    parser.add_argument("edges_output", type=Path)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write exploratory outputs even when blocking cases remain",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}")
    counts, review_path = build_graph_artifacts(
        args.input,
        args.corpus_root,
        args.manifest_output,
        args.edges_output,
        allow_unresolved=args.allow_unresolved,
    )

    print(f"Created manifest: {args.manifest_output}")
    print(f"Created edges: {args.edges_output}")
    print("graph build status:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count:,}")
    if review_path.exists():
        print(f"Review list: {review_path}")


if __name__ == "__main__":
    main()