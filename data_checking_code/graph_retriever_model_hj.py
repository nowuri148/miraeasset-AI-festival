#!/usr/bin/env python3
r"""Model-plan-only multi-hop retriever for the disclosure graph.

This module reads the finalized manifest and graph edge JSONL files, builds
in-memory indexes, traverses FILED/CORRECTS relationships, and optionally
extracts relevant table rows from the source XML/HTML files.  It does not call
an LLM.  The returned Evidence Bundle is intended to be inspected directly or
passed to a separate HyperCLOVA X client.

It deliberately does not infer company, report type, year, or date from a
natural-language question.  HCX-005 must create a search-plan object, and the
caller must pass that object to ``GraphRetriever.retrieve_from_plan``.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


SCRIPT_VERSION = "2.3.0-string-sentinel-contract"

DEFAULT_MANIFEST = Path("corpus/derived/manifest_v3.jsonl")
DEFAULT_EDGES = Path("corpus/derived/graph_edges_v1.jsonl")
DEFAULT_CORPUS_ROOT = Path("corpus")

DATE_RE = re.compile(r"^\d{8}$")
NOT_APPLICABLE = "not_applicable"

PERIODIC_REPORT_TYPES = frozenset(
    {"annual_report", "semiannual_report", "quarterly_report"}
)


class RetrievalError(RuntimeError):
    """Raised when deterministic retrieval cannot safely continue."""


class PipelineDebugLogger:
    """Append one complete HCX-RAG trace per line to a UTF-8 JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str = "Asia/Seoul",
    ) -> None:
        self.path = Path(path)
        self.timezone = ZoneInfo(timezone_name)

    def timestamp(self) -> str:
        return datetime.now(self.timezone).isoformat(timespec="milliseconds")

    def new_record(
        self,
        *,
        question: str,
        model: str,
        pipeline_version: str,
    ) -> dict[str, Any]:
        return {
            "log_schema_version": "clova_graph_rag_debug_v1",
            "trace_id": str(uuid.uuid4()),
            "question_timestamp": self.timestamp(),
            "pipeline_version": pipeline_version,
            "retriever_version": SCRIPT_VERSION,
            "model": model,
            "original_question": question,
            "hcx_function_arguments": None,
            "python_validation": {
                "status": "not_started",
                "canonical_plan": None,
                "error": None,
            },
            "evidence_bundle": None,
            "final_answer": None,
            "usage": {"planner": None, "answer": None},
            "pipeline_status": "running",
            "failed_stage": None,
            "error": None,
            "finished_timestamp": None,
        }

    def append(self, record: dict[str, Any]) -> None:
        record["finished_timestamp"] = record.get("finished_timestamp") or self.timestamp()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")
            file.flush()


@dataclass(frozen=True)
class CompanyRecord:
    corp_code: str
    corp_name: str
    listed_name: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "listed_name": self.listed_name,
        }


class DisclosureTableParser(HTMLParser):
    """Collect visible cell text from HTML-like DART disclosure tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            if self._row:
                self._finish_row()
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
            self._row.append(clean_text("".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            self._finish_row()

    def close(self) -> None:
        super().close()
        if self._row:
            self._finish_row()

    def _finish_row(self) -> None:
        assert self._row is not None
        row = [cell for cell in self._row if cell]
        if row:
            self.rows.append(row)
        self._row = None
        self._cell_parts = None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"(?:주식회사|㈜|\(주\))", "", value)
    return re.sub(r"[\s()·ㆍ._-]+", "", value).casefold()


def normalize_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RetrievalError(
            f"{field_name} must be a YYYYMMDD string or {NOT_APPLICABLE!r}"
        )
    if value == NOT_APPLICABLE:
        return None
    digits = value.strip()
    if not DATE_RE.fullmatch(digits):
        raise RetrievalError(
            f"{field_name} must be YYYYMMDD or {NOT_APPLICABLE!r}: {value!r}"
        )
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError as exc:
        raise RetrievalError(
            f"{field_name} is not a valid calendar date: {value!r}"
        ) from exc
    return digits


def filing_date_of(doc: dict[str, Any]) -> str:
    """Return the receipt-number date, falling back to manifest rcept_dt."""
    rcept_no = str(doc.get("rcept_no") or "")
    if len(rcept_no) >= 8 and DATE_RE.fullmatch(rcept_no[:8]):
        return rcept_no[:8]
    rcept_dt = str(doc.get("rcept_dt") or "")
    return rcept_dt if DATE_RE.fullmatch(rcept_dt) else ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RetrievalError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RetrievalError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def decode_disclosure(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


class GraphRetriever:
    """Load, validate, and traverse the disclosure graph in memory."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        edges_path: str | Path = DEFAULT_EDGES,
        corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.edges_path = Path(edges_path)
        self.corpus_root = Path(corpus_root)

        self.documents = read_jsonl(self.manifest_path)
        self.edges = read_jsonl(self.edges_path)

        self.docs_by_id: dict[str, dict[str, Any]] = {}
        self.companies_by_code: dict[str, CompanyRecord] = {}
        self.company_alias_to_codes: dict[str, set[str]] = defaultdict(set)
        self.company_to_docs: dict[str, list[str]] = defaultdict(list)
        self.chain_to_docs: dict[str, list[str]] = defaultdict(list)
        self.corrects_target: dict[str, str] = {}
        self.corrections_by_target: dict[str, list[str]] = defaultdict(list)
        self.corrects_edges: list[dict[str, Any]] = []
        self.filed_edge_keys: set[tuple[str, str]] = set()
        self.report_types: set[str] = set()

        self._build_indexes()
        self._validate_graph()

    def _build_indexes(self) -> None:
        for doc in self.documents:
            doc_id = doc.get("doc_id")
            corp_code = doc.get("corp_code")
            if not doc_id or not corp_code:
                raise RetrievalError("Every manifest row requires doc_id and corp_code")
            if doc_id in self.docs_by_id:
                raise RetrievalError(f"Duplicate doc_id: {doc_id}")
            self.docs_by_id[doc_id] = doc

            company = CompanyRecord(
                corp_code=corp_code,
                corp_name=doc.get("corp_name") or doc.get("listed_name") or corp_code,
                listed_name=doc.get("listed_name"),
            )
            self.companies_by_code.setdefault(corp_code, company)
            for alias in (corp_code, doc.get("corp_name"), doc.get("listed_name")):
                if alias:
                    self.company_alias_to_codes[normalize_name(str(alias))].add(corp_code)

            chain_id = doc.get("disclosure_chain_id") or doc_id
            self.chain_to_docs[str(chain_id)].append(doc_id)
            report_type = doc.get("normalized_report_type")
            if report_type:
                self.report_types.add(str(report_type))

        for edge in self.edges:
            edge_type = edge.get("edge_type")
            if edge_type == "FILED":
                corp_code = str(edge.get("source_id"))
                doc_id = str(edge.get("target_id"))
                self.company_to_docs[corp_code].append(doc_id)
                self.filed_edge_keys.add((corp_code, doc_id))
            elif edge_type == "CORRECTS":
                source_id = str(edge.get("source_id"))
                target_id = str(edge.get("target_id"))
                if source_id in self.corrects_target:
                    raise RetrievalError(
                        f"Correction has multiple outgoing CORRECTS edges: {source_id}"
                    )
                self.corrects_target[source_id] = target_id
                self.corrections_by_target[target_id].append(source_id)
                self.corrects_edges.append(edge)

        for mapping in (
            self.company_to_docs,
            self.chain_to_docs,
            self.corrections_by_target,
        ):
            for key, values in mapping.items():
                mapping[key] = sorted(set(values), key=self._doc_sort_key)

    def _validate_graph(self) -> None:
        for corp_code, doc_ids in self.company_to_docs.items():
            for doc_id in doc_ids:
                if doc_id not in self.docs_by_id:
                    raise RetrievalError(f"FILED target is missing: {doc_id}")
                if self.docs_by_id[doc_id]["corp_code"] != corp_code:
                    raise RetrievalError(f"FILED corp_code mismatch: {corp_code} -> {doc_id}")

        for source_id, target_id in self.corrects_target.items():
            if source_id not in self.docs_by_id or target_id not in self.docs_by_id:
                raise RetrievalError(f"CORRECTS endpoint is missing: {source_id} -> {target_id}")
            source = self.docs_by_id[source_id]
            target = self.docs_by_id[target_id]
            if not source.get("is_correction"):
                raise RetrievalError(f"CORRECTS source is not marked correction: {source_id}")
            if source.get("corp_code") != target.get("corp_code"):
                raise RetrievalError(f"CORRECTS company mismatch: {source_id} -> {target_id}")
            if source.get("normalized_report_type") != target.get("normalized_report_type"):
                raise RetrievalError(f"CORRECTS report type mismatch: {source_id} -> {target_id}")

    def _doc_sort_key(self, doc_id: str) -> tuple[str, str]:
        doc = self.docs_by_id.get(doc_id, {})
        return filing_date_of(doc), doc_id

    def resolve_company(self, company: str) -> CompanyRecord:
        if company in self.companies_by_code:
            return self.companies_by_code[company]

        codes = self.company_alias_to_codes.get(normalize_name(company), set())
        if not codes:
            candidates = sorted(
                {
                    record.corp_name
                    for record in self.companies_by_code.values()
                    if normalize_name(company) in normalize_name(record.corp_name)
                }
            )
            hint = f" Similar names: {', '.join(candidates[:10])}" if candidates else ""
            raise RetrievalError(f"Company not found: {company}.{hint}")
        if len(codes) > 1:
            names = [self.companies_by_code[code].corp_name for code in sorted(codes)]
            raise RetrievalError(
                f"Company alias is ambiguous: {company} -> {', '.join(names)}. Use corp_code."
            )
        return self.companies_by_code[next(iter(codes))]

    def _retrieve_explicit(
        self,
        *,
        question: str,
        company: str,
        report_type: str | None = None,
        base_years: list[int],
        start_date: str | None = None,
        end_date: str | None = None,
        focus_terms: list[str],
        include_corrections: bool = True,
        max_chains: int = 10,
        include_xml: bool = True,
        max_evidence_rows: int = 12,
    ) -> dict[str, Any]:
        if max_chains < 1:
            raise RetrievalError("max_chains must be at least 1")
        if max_evidence_rows < 1:
            raise RetrievalError("max_evidence_rows must be at least 1")
        question = clean_text(question)
        company_record = self.resolve_company(company)
        if report_type is not None and report_type not in self.report_types:
            raise RetrievalError(
                f"Unknown normalized_report_type: {report_type}. "
                "The HCX plan must use a value present in the manifest."
            )

        base_years = sorted(set(base_years))

        start_date = normalize_date(start_date, "start_date")
        end_date = normalize_date(end_date, "end_date")
        if start_date and end_date and start_date > end_date:
            raise RetrievalError("start_date must be earlier than or equal to end_date")

        seed_ids = self._select_seeds(
            corp_code=company_record.corp_code,
            report_type=report_type,
            base_years=base_years,
            start_date=start_date,
            end_date=end_date,
            seed_doc_id=None,
            chain_id=None,
        )

        seed_chain_ids = unique_preserving_order(
            str(self.docs_by_id[doc_id].get("disclosure_chain_id") or doc_id)
            for doc_id in seed_ids
        )
        seed_chain_ids.sort(
            key=lambda current_chain_id: max(
                (
                    filing_date_of(self.docs_by_id[doc_id])
                    for doc_id in self.chain_to_docs.get(current_chain_id, [])
                ),
                default="",
            ),
            reverse=True,
        )
        truncated = len(seed_chain_ids) > max_chains
        selected_chain_ids = seed_chain_ids[:max_chains]

        selected_doc_ids: set[str] = set()
        if include_corrections:
            for selected_chain_id in selected_chain_ids:
                selected_doc_ids.update(self.chain_to_docs.get(selected_chain_id, []))
            for seed_id in seed_ids:
                if str(
                    self.docs_by_id[seed_id].get("disclosure_chain_id") or seed_id
                ) in selected_chain_ids:
                    selected_doc_ids.update(self._walk_correction_component(seed_id))
        else:
            selected_doc_ids.update(
                seed_id
                for seed_id in seed_ids
                if str(
                    self.docs_by_id[seed_id].get("disclosure_chain_id") or seed_id
                ) in selected_chain_ids
            )

        selected_doc_ids = {
            doc_id
            for doc_id in selected_doc_ids
            if self.docs_by_id[doc_id].get("corp_code") == company_record.corp_code
            and (
                report_type is None
                or self.docs_by_id[doc_id].get("normalized_report_type") == report_type
            )
        }
        ordered_doc_ids = sorted(selected_doc_ids, key=self._doc_sort_key)

        limitations: list[dict[str, str]] = []
        documents = []
        for doc_id in ordered_doc_ids:
            doc_result, doc_limitations = self._build_document_evidence(
                doc_id=doc_id,
                focus_terms=focus_terms or [],
                include_xml=include_xml,
                max_evidence_rows=max_evidence_rows,
            )
            documents.append(doc_result)
            limitations.extend(doc_limitations)

        graph_edges = self._selected_graph_edges(
            company_record.corp_code,
            set(ordered_doc_ids),
        )
        chains = self._build_chain_summaries(selected_chain_ids, set(ordered_doc_ids))
        comparison_groups = self._build_comparison_groups(
            base_years,
            set(ordered_doc_ids),
        )
        if len(base_years) >= 2:
            retrieval_mode = "temporal_comparison"
        elif len(chains) > 1:
            retrieval_mode = "multi_chain"
        elif include_corrections:
            retrieval_mode = "correction_chain"
        else:
            retrieval_mode = "document_lookup"

        if not seed_ids:
            limitations.append(
                {
                    "code": "no_matching_document",
                    "message": "No document matched the company, report type, and date filters.",
                }
            )
        if truncated:
            limitations.append(
                {
                    "code": "chain_limit_applied",
                    "message": (
                        f"{len(seed_chain_ids)} chains matched; only the first "
                        f"{max_chains} chains were returned."
                    ),
                }
            )

        limitations = self._deduplicate_limitations(limitations)
        return {
            "schema_version": "evidence_bundle_v1",
            "retriever_version": SCRIPT_VERSION,
            "request": {
                "question": question or None,
                "company_input": company,
                "normalized_report_type": report_type,
                "base_years": base_years,
                "start_date": start_date,
                "end_date": end_date,
                "focus_terms": focus_terms or [],
                "include_corrections": include_corrections,
                "max_chains": max_chains,
                "include_xml": include_xml,
            },
            "resolved_company": company_record.as_dict(),
            "retrieval_summary": {
                "retrieval_mode": retrieval_mode,
                "matched_seed_documents": len(seed_ids),
                "selected_chains": len(chains),
                "returned_documents": len(documents),
                "returned_graph_edges": len(graph_edges),
                "external_source_documents": sum(
                    1 for doc in documents if doc.get("relation_status") == "external_source"
                ),
            },
            "comparison_groups": comparison_groups,
            "chains": chains,
            "graph_edges": graph_edges,
            "documents": documents,
            "limitations": limitations,
        }

    def validate_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Validate only the Function Calling contract and canonicalize values.

        Natural-language interpretation belongs to HCX.  This method does not
        infer intent or impose question-type-specific semantic rules.  It only
        verifies required keys, JSON/Python types, corpus-controlled values,
        numeric ranges, and calendar-date syntax.
        """
        if not isinstance(plan, dict):
            raise RetrievalError("The model search plan must be a JSON object")

        required_fields = {
            "intent",
            "company",
            "normalized_report_type",
            "base_years",
            "start_date",
            "end_date",
            "include_corrections",
            "focus_terms",
        }
        missing_fields = sorted(required_fields - set(plan))
        if missing_fields:
            raise RetrievalError(
                "Missing required model search-plan field(s): "
                + ", ".join(missing_fields)
            )

        unknown_fields = sorted(set(plan) - required_fields)
        if unknown_fields:
            raise RetrievalError(
                "Unknown model search-plan field(s): " + ", ".join(unknown_fields)
            )

        company = plan.get("company")
        if not isinstance(company, str) or not clean_text(company):
            raise RetrievalError("Model search plan requires a non-empty company")
        company_record = self.resolve_company(clean_text(company))

        raw_report_type = plan["normalized_report_type"]
        if not isinstance(raw_report_type, str):
            raise RetrievalError("normalized_report_type must be a string")
        report_type = (
            None if raw_report_type == NOT_APPLICABLE else raw_report_type
        )
        if report_type is not None and report_type not in self.report_types:
            raise RetrievalError(
                f"Unknown normalized_report_type: {report_type}. "
                "The HCX plan must use a value present in the manifest."
            )

        intent = plan["intent"]
        allowed_intents = {
            "correction_history",
            "temporal_comparison",
            "disclosure_lookup",
        }
        if not isinstance(intent, str) or intent not in allowed_intents:
            raise RetrievalError(
                "intent must be correction_history, temporal_comparison, "
                "or disclosure_lookup"
            )

        raw_years = plan["base_years"]
        if not isinstance(raw_years, list) or any(
            isinstance(year, bool) or not isinstance(year, int) for year in raw_years
        ):
            raise RetrievalError("base_years must be an array of integers")
        if any(year < 2000 or year > 2100 for year in raw_years):
            raise RetrievalError("base_years must be between 2000 and 2100")

        raw_focus_terms = plan["focus_terms"]
        if not isinstance(raw_focus_terms, list) or any(
            not isinstance(term, str) for term in raw_focus_terms
        ):
            raise RetrievalError("focus_terms must be an array of strings")
        focus_terms = unique_preserving_order(
            clean_text(term) for term in raw_focus_terms if clean_text(term)
        )
        if len(focus_terms) > 8:
            raise RetrievalError("focus_terms must contain at most 8 unique terms")

        include_corrections = plan["include_corrections"]
        if not isinstance(include_corrections, bool):
            raise RetrievalError("include_corrections must be boolean")

        start_date = plan["start_date"]
        end_date = plan["end_date"]
        for field_name, value in (("start_date", start_date), ("end_date", end_date)):
            if not isinstance(value, str):
                raise RetrievalError(
                    f"{field_name} must be a YYYYMMDD string or "
                    f"{NOT_APPLICABLE!r}"
                )
        start_date = normalize_date(start_date, "start_date")
        end_date = normalize_date(end_date, "end_date")
        if start_date and end_date and start_date > end_date:
            raise RetrievalError("start_date must be earlier than or equal to end_date")

        return {
            "intent": intent,
            "company": company_record.corp_code,
            "normalized_report_type": report_type,
            "base_years": sorted(set(raw_years)),
            "start_date": start_date,
            "end_date": end_date,
            "include_corrections": include_corrections,
            "focus_terms": focus_terms,
        }

    def retrieve_from_plan(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        max_chains: int = 10,
        include_xml: bool = True,
        max_evidence_rows: int = 12,
    ) -> dict[str, Any]:
        """Validate an HCX-created search plan and retrieve without rule inference."""
        canonical_plan = self.validate_plan(plan)
        bundle = self._retrieve_explicit(
            question=question,
            company=canonical_plan["company"],
            report_type=canonical_plan["normalized_report_type"],
            base_years=canonical_plan["base_years"],
            start_date=canonical_plan["start_date"],
            end_date=canonical_plan["end_date"],
            focus_terms=canonical_plan["focus_terms"],
            include_corrections=canonical_plan["include_corrections"],
            max_chains=max_chains,
            include_xml=include_xml,
            max_evidence_rows=max_evidence_rows,
        )
        bundle["search_plan"] = canonical_plan
        return bundle

    def _select_seeds(
        self,
        *,
        corp_code: str,
        report_type: str | None,
        base_years: list[int],
        start_date: str | None,
        end_date: str | None,
        seed_doc_id: str | None,
        chain_id: str | None,
    ) -> list[str]:
        if seed_doc_id:
            candidates = [seed_doc_id]
        elif chain_id:
            candidates = list(self.chain_to_docs.get(chain_id, []))
        else:
            candidates = list(self.company_to_docs.get(corp_code, []))

        output = []
        for doc_id in candidates:
            doc = self.docs_by_id[doc_id]
            if doc.get("corp_code") != corp_code:
                continue
            if report_type and doc.get("normalized_report_type") != report_type:
                continue
            if base_years:
                try:
                    base_year = int(doc.get("base_year"))
                except (TypeError, ValueError):
                    continue
                if base_year not in base_years:
                    continue
            filing_date = filing_date_of(doc)
            if start_date and filing_date < start_date:
                continue
            if end_date and filing_date > end_date:
                continue
            output.append(doc_id)
        return sorted(set(output), key=self._doc_sort_key)

    def _build_comparison_groups(
        self,
        base_years: list[int],
        selected_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Group selected periodic documents by reporting-period year."""
        if not base_years:
            return []
        output: list[dict[str, Any]] = []
        for base_year in base_years:
            document_ids = sorted(
                [
                    doc_id
                    for doc_id in selected_doc_ids
                    if self.docs_by_id[doc_id].get("base_year") == base_year
                    or str(self.docs_by_id[doc_id].get("base_year")) == str(base_year)
                ],
                key=self._doc_sort_key,
            )
            chain_ids = unique_preserving_order(
                str(self.docs_by_id[doc_id].get("disclosure_chain_id") or doc_id)
                for doc_id in document_ids
            )
            output.append(
                {
                    "base_year": base_year,
                    "document_ids": document_ids,
                    "disclosure_chain_ids": chain_ids,
                }
            )
        return output

    def _walk_correction_component(self, start_doc_id: str) -> set[str]:
        found: set[str] = set()
        queue: deque[str] = deque([start_doc_id])
        while queue:
            doc_id = queue.popleft()
            if doc_id in found:
                continue
            found.add(doc_id)
            target = self.corrects_target.get(doc_id)
            if target:
                queue.append(target)
            queue.extend(self.corrections_by_target.get(doc_id, []))
        return found

    def _selected_graph_edges(
        self,
        corp_code: str,
        selected_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for doc_id in sorted(selected_doc_ids, key=self._doc_sort_key):
            if (corp_code, doc_id) in self.filed_edge_keys:
                output.append(
                    {
                        "source_type": "Company",
                        "source_id": corp_code,
                        "edge_type": "FILED",
                        "target_type": "Document",
                        "target_id": doc_id,
                    }
                )
        for edge in self.corrects_edges:
            if edge["source_id"] in selected_doc_ids and edge["target_id"] in selected_doc_ids:
                output.append(dict(edge))
        return output

    def _build_chain_summaries(
        self,
        selected_chain_ids: list[str],
        selected_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        output = []
        for chain_id in selected_chain_ids:
            doc_ids = sorted(
                [doc_id for doc_id in self.chain_to_docs.get(chain_id, []) if doc_id in selected_doc_ids],
                key=self._doc_sort_key,
            )
            if not doc_ids:
                continue
            output.append(
                {
                    "disclosure_chain_id": chain_id,
                    "document_ids": doc_ids,
                    "first_filing_date": filing_date_of(self.docs_by_id[doc_ids[0]]),
                    "last_filing_date": filing_date_of(self.docs_by_id[doc_ids[-1]]),
                    "has_external_source": any(
                        self.docs_by_id[doc_id].get("relation_status") == "external_source"
                        for doc_id in doc_ids
                    ),
                }
            )
        return output

    def _build_document_evidence(
        self,
        *,
        doc_id: str,
        focus_terms: list[str],
        include_xml: bool,
        max_evidence_rows: int,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        doc = self.docs_by_id[doc_id]
        result = {
            key: doc.get(key)
            for key in (
                "doc_id",
                "corp_code",
                "corp_name",
                "listed_name",
                "stock_code",
                "doc_group",
                "doc_subtype",
                "normalized_report_type",
                "report_nm",
                "is_correction",
                "rcept_no",
                "rcept_dt",
                "event_date",
                "base_year",
                "base_month",
                "flr_nm",
                "disclosure_chain_id",
                "relation_status",
                "file_path",
            )
        }
        filing_date = filing_date_of(doc)
        result["filing_date"] = filing_date or None
        result["rcept_date_consistent"] = (
            not filing_date
            or not doc.get("rcept_dt")
            or filing_date == str(doc.get("rcept_dt"))
        )
        result["corrects_doc_id"] = self.corrects_target.get(doc_id)
        result["corrected_by_doc_ids"] = list(self.corrections_by_target.get(doc_id, []))

        limitations: list[dict[str, str]] = []
        if result["rcept_date_consistent"] is False:
            limitations.append(
                {
                    "code": "receipt_date_mismatch",
                    "doc_id": doc_id,
                    "message": (
                        f"{doc_id} has rcept_dt={doc.get('rcept_dt')} but its receipt "
                        f"number encodes {filing_date}; retrieval used {filing_date}."
                    ),
                }
            )
        if doc.get("relation_status") == "external_source":
            limitations.append(
                {
                    "code": "external_source",
                    "doc_id": doc_id,
                    "message": (
                        f"{doc_id} refers to a correction source outside the corpus; "
                        "the complete before/after history cannot be reconstructed."
                    ),
                }
            )

        if not include_xml:
            result["source_files"] = []
            result["evidence_rows"] = []
            result["source_status"] = "skipped"
            return result, limitations

        files = self._document_files(doc)
        if not files:
            result["source_files"] = []
            result["evidence_rows"] = []
            result["source_status"] = "file_not_found"
            limitations.append(
                {
                    "code": "source_file_not_found",
                    "doc_id": doc_id,
                    "message": f"Source XML/HTML file was not found for {doc_id}.",
                }
            )
            return result, limitations

        keywords = self._evidence_keywords(focus_terms)
        evidence_rows: list[dict[str, Any]] = []
        source_files = []
        for path in files:
            raw = path.read_bytes()
            text, encoding = decode_disclosure(raw)
            rows = self._extract_table_rows(text)
            selected = self._select_relevant_rows(rows, keywords, max_evidence_rows)
            try:
                relative_path = str(path.relative_to(self.corpus_root))
            except ValueError:
                relative_path = str(path)
            source_files.append(
                {
                    "path": relative_path.replace("\\", "/"),
                    "encoding_used": encoding,
                    "table_rows_found": len(rows),
                }
            )
            for row in selected:
                row["source_file"] = relative_path.replace("\\", "/")
                evidence_rows.append(row)

        evidence_rows.sort(key=lambda item: (-item["score"], item["text"]))
        deduplicated: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for row in evidence_rows:
            if row["text"] in seen_text:
                continue
            seen_text.add(row["text"])
            deduplicated.append(row)
            if len(deduplicated) >= max_evidence_rows:
                break

        result["source_files"] = source_files
        result["evidence_rows"] = deduplicated
        result["source_status"] = "loaded"
        if not deduplicated:
            limitations.append(
                {
                    "code": "no_relevant_table_row",
                    "doc_id": doc_id,
                    "message": (
                        f"Source file was loaded for {doc_id}, but no table row matched "
                        "the HCX search plan's focus_terms."
                    ),
                }
            )
        return result, limitations

    def _evidence_keywords(
        self,
        focus_terms: list[str],
    ) -> list[str]:
        return unique_preserving_order(clean_text(term) for term in focus_terms)

    @staticmethod
    def _extract_table_rows(text: str) -> list[list[str]]:
        parser = DisclosureTableParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception:
            # DART files occasionally contain malformed HTML.  Rows parsed
            # before the malformed fragment remain useful and deterministic.
            pass
        return parser.rows

    @staticmethod
    def _select_relevant_rows(
        rows: list[list[str]],
        keywords: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for row_index, cells in enumerate(rows):
            text = " | ".join(cells)
            normalized_text = normalize_name(text)
            matched_terms = [
                term for term in keywords if normalize_name(term) in normalized_text
            ]
            if not matched_terms:
                continue
            score = len(matched_terms)
            scored.append(
                {
                    "row_index": row_index,
                    "cells": cells,
                    "text": text,
                    "matched_terms": matched_terms,
                    "score": score,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["row_index"]))
        return scored[:limit]

    def _document_files(self, doc: dict[str, Any]) -> list[Path]:
        raw_path = str(doc.get("file_path") or "").strip()
        if not raw_path:
            return []
        location = self._resolve_unicode_path(self.corpus_root, Path(raw_path))
        if location is None:
            return []
        if location.is_file():
            return [location]

        preferred_extensions = {".xml", ".html", ".htm", ".xhtml"}
        files = sorted(
            [path for path in location.rglob("*") if path.is_file()],
            key=lambda path: str(path),
        )
        preferred = [path for path in files if path.suffix.casefold() in preferred_extensions]
        return preferred or files

    @staticmethod
    def _resolve_unicode_path(root: Path, relative: Path) -> Path | None:
        candidate = root / relative
        if candidate.exists():
            return candidate

        current = root
        if not current.exists():
            return None
        for part in relative.parts:
            normalized_part = unicodedata.normalize("NFC", part).casefold()
            matches = [
                child
                for child in current.iterdir()
                if unicodedata.normalize("NFC", child.name).casefold() == normalized_part
            ]
            if len(matches) != 1:
                return None
            current = matches[0]
        return current if current.exists() else None

    @staticmethod
    def _deduplicate_limitations(
        limitations: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for item in limitations:
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output