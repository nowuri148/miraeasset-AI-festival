from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import warnings
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, TypeVar

import numpy as np


DATE_BASES = {"base_year", "event_date", "rcept_dt"}
SCOPE_TYPES = {"direct", "group", "sector", "industry", "all"}
ALLOWED_REPORT_TYPES = frozenset(
    {
        "annual_report",
        "semiannual_report",
        "quarterly_report",
        "facility_investment",
        "supply_contract",
        "other_company_shares_acquisition_decision",
        "material_management_matter",
        "treasury_stock_acquisition_decision",
        "lawsuit_filing",
        "merger_decision",
        "treasury_stock_disposal_decision",
        "paid_in_capital_increase_decision",
        "large_shareholding_general",
        "supply_contract_termination",
        "other_company_shares_disposal_decision",
        "convertible_bond_issuance_decision",
        "tangible_asset_disposal_decision",
        "large_shareholding_summary",
        "tangible_asset_acquisition_decision",
        "company_split_decision",
        "write_down_contingent_capital_security_issuance_decision",
        "treasury_stock_trust_agreement_conclusion_decision",
        "treasury_stock_trust_agreement_termination_decision",
        "stock_exchange_or_transfer_decision",
        "split_merger_decision",
        "overseas_securities_delisting_decision",
        "regulatory_capital_debt_security_issuance_decision",
        "capital_reduction_decision",
        "bonus_issue_decision",
        "exchangeable_bond_issuance_decision",
        "own_convertible_bond_sale_decision",
        "overseas_securities_listing_decision",
        "business_acquisition_decision",
        "third_party_convertible_bond_call_option_exercise",
        "business_suspension",
    }
)
ALLOWED_INDUSTRIES = frozenset(
    {
        "IT",
        "건강관리",
        "경기관련소비재",
        "금융",
        "산업재",
        "소재",
        "커뮤니케이션서비스",
        "필수소비재",
    }
)
ALLOWED_SECTORS = frozenset(
    {
        "2차전지",
        "AI소프트웨어·플랫폼",
        "건설",
        "게임",
        "금융·보험",
        "로봇",
        "바이오·제약",
        "반도체·전자부품",
        "방산·항공우주",
        "비철금속",
        "소비재·유통",
        "신재생에너지",
        "엔터테인먼트",
        "운송·물류",
        "원전",
        "자동차·모빌리티",
        "전력기기",
        "조선",
        "철강",
        "통신",
    }
)
PLAN_FIELDS = {"retrieval_requests"}
REQUEST_REQUIRED_FIELDS = {
    "query",
    "normalized_report_types",
    "period",
    "date_basis",
    "use_correction_graph",
    "company_scope",
    "exact_keywords",
}
PERIOD_RE = re.compile(
    r"^(?P<year>\d{4})(?:-(?:(?P<month>\d{2})-(?P<day>\d{2})|Q(?P<quarter>[1-4])|H(?P<half>[12])))?$"
)

DEFAULT_DENSE_K = 24
DEFAULT_KEYWORD_DENSE_K = 12
DEFAULT_RERANK_K = 10
DEFAULT_CONTEXT_CHAR_BUDGET = 30_000
# 기간이 정확히 일치하는 요청×연도 셀을 최대 12개까지 다룰 때,
# 각 셀의 적격 상위 청크 2개를 모두 담을 수 있는 기본 상한이다.
DEFAULT_CONTEXT_MAX_CHUNKS = 24
DEFAULT_EXACT_CHUNKS_PER_CELL = 2
DEFAULT_FALLBACK_CHUNKS_PER_CELL = 1
MAX_REPORT_TYPES_PER_REQUEST = 3
PERIODIC_REPORT_TYPES = (
    "annual_report",
    "semiannual_report",
    "quarterly_report",
)
DEFAULT_QUERY_WEIGHT = 0.80
DEFAULT_KEYWORD_WEIGHT = 0.20
DEFAULT_QUERY_SCORE_THRESHOLD = 0.55
DEFAULT_MIN_QUERY_RELATIVE_RELEVANCE = 0.85
DEFAULT_MIN_RELATIVE_RELEVANCE = 0.90
KEYWORD_RESULT_FIELDS = ("metrics", "topic_keywords")
GENERIC_GROUNDED_KEYWORDS = frozenset(
    {
        "비교",
        "분석",
        "설명",
        "확인",
        "조회",
        "검색",
        "요약",
        "추출",
        "계산",
    }
)
RRF_K = 60
DEFAULT_EVENT_DATE_FALLBACK = "rcept_dt"
ALLOWED_EVENT_DATE_FALLBACKS = {None, "rcept_dt"}

K = TypeVar("K")
V = TypeVar("V")


def _debug_print(enabled: bool, stage: str, message: str) -> None:
    if enabled:
        print(f"[SEARCH][{stage}] {message}", flush=True)


@dataclass(frozen=True)
class CompanyScope:
    scope_type: str
    scope_values: tuple[str, ...]


@dataclass(frozen=True)
class PeriodSpec:
    start: str
    end: str
    date_basis: str
    lower_key: int
    upper_key: int


@dataclass(frozen=True)
class PeriodIntent:
    kind: str
    report_type: str
    base_month: int | None
    label: str


@dataclass(frozen=True)
class RetrievalRequest:
    request_id: str
    query: str
    normalized_report_types: tuple[str, ...]
    period: PeriodSpec
    use_correction_graph: bool
    company_scope: CompanyScope
    exact_keywords: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSearchPlan:
    retrieval_requests: tuple[RetrievalRequest, ...]


@dataclass(frozen=True)
class DocumentRef:
    doc_id: str
    corp_code: str
    corp_name: str
    normalized_report_type: str
    report_nm: str
    base_year: int | None
    base_month: int | None
    event_date: str
    rcept_dt: str
    rcept_no: str
    is_correction: bool
    disclosure_chain_id: str


@dataclass(frozen=True)
class ResolvedRequest:
    request: RetrievalRequest
    scope: CompanyScope
    seed_documents: tuple[DocumentRef, ...]
    documents: tuple[DocumentRef, ...]
    graph_expanded_doc_ids: tuple[str, ...]
    date_basis_by_doc_id: tuple[tuple[str, str], ...]
    fallback_seed_doc_ids: tuple[str, ...]
    report_type_fallback_applied: bool = False
    report_type_fallback_types: tuple[str, ...] = ()
    report_type_fallback_document_ids: tuple[str, ...] = ()

    def date_basis_for(self, document: DocumentRef) -> str:
        return dict(self.date_basis_by_doc_id).get(
            document.doc_id,
            self.request.period.date_basis,
        )


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    request_ids: set[str] = field(default_factory=set)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    request_fusion_scores: dict[str, float] = field(default_factory=dict)
    request_scores: dict[str, float] = field(default_factory=dict)
    request_evidence_strengths: dict[str, float] = field(default_factory=dict)
    request_match_signals: dict[str, dict[str, float]] = field(default_factory=dict)
    request_keyword_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    is_expanded: bool = False
    parent_chunk_id: str | None = None

    @property
    def doc_id(self) -> str:
        return _clean(self.metadata.get("doc_id"))


@dataclass(frozen=True)
class ContextBundle:
    question: str
    documents: tuple[DocumentRef, ...]
    chunks: tuple[dict[str, Any], ...]
    total_chars: int
    covered_request_ids: tuple[str, ...]
    covered_period_buckets: tuple[str, ...]
    request_diagnostics: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        request_period_coverage = [
            {
                "request_id": diagnostic.get("request_id"),
                "expected_period_buckets": list(
                    diagnostic.get("expected_period_buckets", [])
                ),
                "qualified_period_buckets": list(
                    diagnostic.get("qualified_period_buckets", [])
                ),
                "covered_period_buckets": list(
                    diagnostic.get("covered_period_buckets", [])
                ),
                "missing_period_buckets": list(
                    diagnostic.get("missing_period_buckets", [])
                ),
            }
            for diagnostic in self.request_diagnostics
        ]
        missing_period_cells = [
            {
                "request_id": coverage["request_id"],
                "period_bucket": period_bucket,
            }
            for coverage in request_period_coverage
            for period_bucket in coverage["missing_period_buckets"]
        ]
        covered_request_year_cells = [
            {
                "request_id": coverage["request_id"],
                "period_bucket": period_bucket,
            }
            for coverage in request_period_coverage
            for period_bucket in coverage["covered_period_buckets"]
        ]
        period_fallbacks = [
            {
                "request_id": diagnostic.get("request_id"),
                **fallback,
            }
            for diagnostic in self.request_diagnostics
            for fallback in diagnostic.get("period_fallbacks", [])
            if isinstance(fallback, dict)
        ]
        return {
            "question": self.question,
            "documents": [asdict(document) for document in self.documents],
            "chunks": list(self.chunks),
            "total_chars": self.total_chars,
            "covered_request_ids": list(self.covered_request_ids),
            "covered_period_buckets": list(self.covered_period_buckets),
            "coverage_unit": "request_year",
            "covered_request_year_cells": covered_request_year_cells,
            "request_period_coverage": request_period_coverage,
            "missing_period_cells": missing_period_cells,
            "request_year_coverage_complete": not missing_period_cells,
            "period_fallbacks": period_fallbacks,
            "request_diagnostics": list(self.request_diagnostics),
        }


class HybridChunkStore(Protocol):
    def dense_search(
        self, *, query: str, doc_ids: tuple[str, ...], top_k: int
    ) -> list[dict[str, Any]]: ...

    def dense_scores(
        self,
        *,
        queries: tuple[str, ...],
        candidates: tuple[Candidate, ...],
    ) -> dict[str, dict[str, float]]: ...

    def get_related_chunks(
        self, *, chunk: Candidate, window: int
    ) -> list[dict[str, Any]]: ...


class ManifestGraphCatalog:
    """문서 메타데이터 필터와 정정 관계 탐색을 담당한다."""

    def __init__(
        self,
        manifest_path: str | Path,
        correction_edges_path: str | Path | None = None,
        universe_path: str | Path | None = None,
        event_date_fallback: str | None = DEFAULT_EVENT_DATE_FALLBACK,
        debug: bool = False,
    ):
        if event_date_fallback not in ALLOWED_EVENT_DATE_FALLBACKS:
            raise ValueError(
                "event_date_fallback must be None or 'rcept_dt'"
            )
        self.event_date_fallback = event_date_fallback
        self.debug = debug
        self.manifest_path = Path(manifest_path)
        rows = _load_jsonl(self.manifest_path)
        self.documents: tuple[DocumentRef, ...] = tuple(
            _document_from_manifest(row) for row in rows
        )
        self.by_id = {document.doc_id: document for document in self.documents}
        self.company_codes: dict[str, set[str]] = defaultdict(set)
        self.company_alias_codes: dict[str, set[str]] = defaultdict(set)
        self.company_names_by_code: dict[str, str] = {}
        self.document_company_codes: set[str] = set()
        self.sector_codes: dict[str, set[str]] = defaultdict(set)
        self.industry_codes: dict[str, set[str]] = defaultdict(set)
        self.chain_docs: dict[str, set[str]] = defaultdict(set)
        self.correction_neighbors: dict[str, set[str]] = defaultdict(set)

        for raw, document in zip(rows, self.documents):
            if not document.doc_id:
                continue
            self.company_codes[document.corp_name].add(document.corp_code)
            self.company_alias_codes[document.corp_name].add(document.corp_code)
            self.company_alias_codes[document.corp_code].add(document.corp_code)
            self.company_names_by_code[document.corp_code] = document.corp_name
            self.document_company_codes.add(document.corp_code)
            for alias_key in ("listed_name", "stock_code", "corp_eng_name"):
                alias = _clean(raw.get(alias_key))
                if alias:
                    self.company_alias_codes[alias].add(document.corp_code)
            sector = _clean(raw.get("sector"))
            industry = _clean(raw.get("industry"))
            if sector:
                self.sector_codes[sector].add(document.corp_code)
            if industry:
                self.industry_codes[industry].add(document.corp_code)
            if document.disclosure_chain_id:
                self.chain_docs[document.disclosure_chain_id].add(document.doc_id)

        self.universe_path = Path(universe_path) if universe_path else None
        if self.universe_path:
            for raw in _load_csv(self.universe_path):
                corp_code = _clean(raw.get("corp_code"))
                corp_name = _clean(raw.get("corp_name"))
                if not corp_code or not corp_name:
                    continue
                self.company_codes[corp_name].add(corp_code)
                self.company_names_by_code.setdefault(corp_code, corp_name)
                for alias_key in (
                    "corp_code",
                    "corp_name",
                    "listed_name",
                    "stock_code",
                    "corp_eng_name",
                ):
                    alias = _clean(raw.get(alias_key))
                    if alias:
                        self.company_alias_codes[alias].add(corp_code)
                sector = _clean(raw.get("sector"))
                industry = _clean(raw.get("industry"))
                if sector:
                    self.sector_codes[sector].add(corp_code)
                if industry:
                    self.industry_codes[industry].add(corp_code)

        self.correction_edges_path = (
            Path(correction_edges_path) if correction_edges_path else None
        )
        if self.correction_edges_path:
            if not self.correction_edges_path.exists():
                raise FileNotFoundError(self.correction_edges_path)
            for edge in _load_jsonl(self.correction_edges_path):
                if _clean(edge.get("edge_type")).upper() != "CORRECTS":
                    continue
                source = _clean(edge.get("source_id"))
                target = _clean(edge.get("target_id"))
                if source in self.by_id and target in self.by_id:
                    self.correction_neighbors[source].add(target)
                    self.correction_neighbors[target].add(source)

        _debug_print(
            self.debug,
            "CATALOG",
            f"manifest={self.manifest_path} documents={len(self.documents):,} "
            f"companies={len(self.document_company_codes):,} "
            f"report_types={len(self.report_types):,} "
            f"correction_edges={'on' if self.has_correction_graph else 'off'} "
            f"event_date_fallback={self.event_date_fallback or 'strict'}",
        )

    @property
    def report_types(self) -> set[str]:
        return {
            document.normalized_report_type
            for document in self.documents
            if document.normalized_report_type
        }

    @property
    def has_correction_graph(self) -> bool:
        return self.correction_edges_path is not None

    def resolve_company_scope(
        self,
        scope: CompanyScope,
        group_members: Mapping[str, Iterable[str]] | None = None,
    ) -> tuple[str, ...]:
        if scope.scope_type == "all":
            return tuple(sorted(self.document_company_codes))

        if not scope.scope_values:
            raise ValueError(f"scope_values is required for {scope.scope_type}")

        if scope.scope_type in {"sector", "industry"}:
            source = self.sector_codes if scope.scope_type == "sector" else self.industry_codes
            codes: set[str] = set()
            for value in scope.scope_values:
                if value not in source:
                    raise LookupError(f"unknown {scope.scope_type}: {value}")
                codes.update(source[value])
            return tuple(sorted(codes & self.document_company_codes))

        if scope.scope_type == "group":
            if not group_members:
                raise LookupError(
                    "group scope requires target_companies expansion from the keyword extractor"
                )
            member_names: list[str] = []
            for value in scope.scope_values:
                members = group_members.get(value)
                if not members:
                    raise LookupError(f"unresolved group: {value}")
                member_names.extend(members)
            return self._resolve_direct_values(member_names)

        return self._resolve_direct_values(scope.scope_values)

    def _resolve_direct_values(self, values: Iterable[str]) -> tuple[str, ...]:
        codes = set()
        for raw_value in values:
            value = _clean(raw_value)
            matches = self.company_alias_codes.get(value, set())
            if not matches:
                raise LookupError(f"unknown company: {value}")
            if len(matches) > 1:
                names = sorted(
                    self.company_names_by_code.get(code, code) for code in matches
                )
                raise LookupError(f"ambiguous company identifier: {value} -> {names}")
            codes.update(matches)
        resolved = codes & self.document_company_codes
        if not resolved:
            raise LookupError("resolved companies have no documents in the manifest")
        return tuple(sorted(resolved))

    def resolve_request(
        self,
        request: RetrievalRequest,
        group_members: Mapping[str, Iterable[str]] | None = None,
        *,
        debug: bool | None = None,
    ) -> ResolvedRequest:
        debug_enabled = self.debug if debug is None else debug
        scope = request.company_scope
        company_codes = set(self.resolve_company_scope(scope, group_members))
        report_types = set(request.normalized_report_types)
        requested_eligible = [
            document
            for document in self.documents
            if document.corp_code in company_codes
            and document.normalized_report_type in report_types
        ]
        requested_seeds: list[DocumentRef] = []
        fallback_seed_ids: set[str] = set()
        date_basis_by_doc_id: dict[str, str] = {}
        for document in requested_eligible:
            basis = _effective_document_date_basis(
                document,
                requested_basis=request.period.date_basis,
                event_date_fallback=self.event_date_fallback,
            )
            if basis is None:
                continue
            date_basis_by_doc_id[document.doc_id] = basis
            if _document_matches_period(document, request.period, date_basis=basis):
                requested_seeds.append(document)
                if basis != request.period.date_basis:
                    fallback_seed_ids.add(document.doc_id)

        requested_eligible_count = len(requested_eligible)
        requested_seed_count = len(requested_seeds)
        requested_event_fallback_count = len(fallback_seed_ids)

        _debug_print(
            debug_enabled,
            "SCOPE",
            f"{request.request_id} scope={scope.scope_type}:{list(scope.scope_values)} "
            f"companies={len(company_codes):,} report_types={list(request.normalized_report_types)} "
            f"eligible_documents={requested_eligible_count:,}",
        )
        _debug_print(
            debug_enabled,
            "PERIOD",
            f"{request.request_id} requested_basis={request.period.date_basis} "
            f"period={request.period.start}..{request.period.end} "
            f"matched={requested_seed_count:,} "
            f"fallback_to_rcept_dt={requested_event_fallback_count:,}",
        )

        # 정정 그래프는 검색계획 LLM이 지정한 문서유형과 시드에만 적용한다.
        # 실행기가 기본으로 추가하는 정기공시 전체까지 그래프를 확장하지 않는다.
        expanded_ids: set[str] = set()
        requested_documents = list(requested_seeds)
        if request.use_correction_graph:
            if not self.has_correction_graph:
                raise RuntimeError(
                    f"{request.request_id}: correction graph requested but no edge file was configured"
                )
            expanded_ids = self._expand_correction_components(
                {document.doc_id for document in requested_seeds}
            )
            for seed in requested_seeds:
                if seed.disclosure_chain_id:
                    expanded_ids.update(self.chain_docs[seed.disclosure_chain_id])
            requested_documents = [
                document
                for document in requested_eligible
                if document.doc_id in expanded_ids
            ]

        for document in requested_documents:
            if document.doc_id not in date_basis_by_doc_id:
                basis = _effective_document_date_basis(
                    document,
                    requested_basis=request.period.date_basis,
                    event_date_fallback=self.event_date_fallback,
                )
                if basis is not None:
                    date_basis_by_doc_id[document.doc_id] = basis

        # 정기공시 3종의 문서 메타데이터 후보를 미리 확보한다. 실제 벡터
        # 검색은 execute_search_plan에서 정확한 기간 유형을 먼저 실행하고,
        # 비어 있는 요청×연도 셀에 대해서만 3종 fallback을 실행한다.
        periodic_period = _as_base_year_period(request.period)
        periodic_eligible = [
            document
            for document in self.documents
            if document.corp_code in company_codes
            and document.normalized_report_type in PERIODIC_REPORT_TYPES
        ]
        periodic_seeds = [
            document
            for document in periodic_eligible
            if _document_matches_period(
                document,
                periodic_period,
                date_basis="base_year",
            )
        ]
        for document in periodic_seeds:
            date_basis_by_doc_id.setdefault(document.doc_id, "base_year")

        # 기존 필드명은 외부 진단 결과와의 호환성을 위해 유지한다.
        report_type_fallback_applied = bool(periodic_seeds)
        report_type_fallback_types = tuple(
            report_type
            for report_type in PERIODIC_REPORT_TYPES
            if any(
                document.normalized_report_type == report_type
                for document in periodic_seeds
            )
        )
        report_type_fallback_document_ids = tuple(
            sorted(document.doc_id for document in periodic_seeds)
        )

        _debug_print(
            debug_enabled,
            "PERIODIC_CATALOG",
            f"{request.request_id} types={list(report_type_fallback_types)} "
            f"period={periodic_period.start}..{periodic_period.end} "
            f"matched={len(periodic_seeds):,} date_basis=base_year",
        )

        seeds = list(
            {
                document.doc_id: document
                for document in [*requested_seeds, *periodic_seeds]
            }.values()
        )
        documents = [*requested_documents, *periodic_seeds]
        documents = sorted(
            {document.doc_id: document for document in documents}.values(),
            key=_document_sort_key,
        )
        resolved_doc_ids = {document.doc_id for document in documents}
        date_basis_by_doc_id = {
            doc_id: basis
            for doc_id, basis in date_basis_by_doc_id.items()
            if doc_id in resolved_doc_ids
        }
        seed_ids = {document.doc_id for document in seeds}
        _debug_print(
            debug_enabled,
            "GRAPH",
            f"{request.request_id} enabled={request.use_correction_graph} "
            f"planned_seeds={len(requested_seeds):,} "
            f"periodic_seeds={len(periodic_seeds):,} "
            f"resolved_documents={len(documents):,} "
            f"expanded={sum(document.doc_id not in seed_ids for document in documents):,}",
        )
        return ResolvedRequest(
            request=request,
            scope=scope,
            seed_documents=tuple(sorted(seeds, key=_document_sort_key)),
            documents=tuple(documents),
            graph_expanded_doc_ids=tuple(
                sorted(document.doc_id for document in documents if document.doc_id not in seed_ids)
            ),
            date_basis_by_doc_id=tuple(sorted(date_basis_by_doc_id.items())),
            fallback_seed_doc_ids=tuple(sorted(fallback_seed_ids)),
            report_type_fallback_applied=report_type_fallback_applied,
            report_type_fallback_types=report_type_fallback_types,
            report_type_fallback_document_ids=report_type_fallback_document_ids,
        )

    def _expand_correction_components(self, seed_ids: set[str]) -> set[str]:
        visited = set(seed_ids)
        queue = deque(seed_ids)
        while queue:
            current = queue.popleft()
            for neighbor in self.correction_neighbors.get(current, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited


def company_scope_from_keyword_result(result: dict[str, Any]) -> CompanyScope:
    """키워드 추출 결과의 정규화 범위를 그대로 변환한다."""
    scope_type = _clean(result.get("scope_type"))
    scope_values = _unique_strings(result.get("scope_values") or [])
    if not scope_type:
        raise ValueError("keyword result has no scope_type")
    return _normalize_company_scope(
        {"scope_type": scope_type, "scope_values": list(scope_values)}
    )


def group_members_from_keyword_result(
    result: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """group 범위일 때만 실행용 구성원 목록을 만든다.

    target_companies는 LLM 출력에 복사하지 않고, Python 실행 단계에서만
    CompanyScopeResolver가 확정한 기업집단 구성원으로 사용한다.
    """
    scope = company_scope_from_keyword_result(result)
    if scope.scope_type != "group":
        return {}
    if len(scope.scope_values) != 1:
        raise ValueError("group scope currently requires exactly one scope value")
    targets = _unique_strings(result.get("target_companies") or [])
    if not targets:
        raise ValueError("group scope has no resolved target_companies")
    return {scope.scope_values[0]: targets}


def grounded_keywords_by_request(
    question: str,
    plan: PreparedSearchPlan,
    *,
    keyword_result: Mapping[str, Any] | None,
    debug: bool = False,
) -> dict[str, tuple[str, ...]]:
    """검색계획 LLM이 요청별로 배정한 exact_keywords를 검증한다.

    요청 간 키워드 배정은 검색계획 LLM의 책임이다. Python 실행기는
    keyword_result가 있으면 metrics와 topic_keywords에 실제 존재하는
    표현인지만 확인하고, 검색문과의 유사도로 다시 배정하지 않는다.
    """
    del question  # 함수 호출 호환성을 위해 인자는 유지

    validate_against_keyword_result = keyword_result is not None
    allowed_by_key: dict[str, str] = {}
    blocked_keys: set[str] = set()
    if keyword_result:
        for field_name in (
            "scope_values",
            "companies",
            "target_companies",
            "doc_types",
        ):
            blocked_keys.update(
                _compact_match_key(value)
                for value in _iter_string_values(keyword_result.get(field_name))
            )
        for field_name in KEYWORD_RESULT_FIELDS:
            for value in _iter_string_values(keyword_result.get(field_name)):
                keyword = _clean(value)
                keyword_key = _compact_match_key(keyword)
                if (
                    len(keyword_key) >= 2
                    and keyword_key not in blocked_keys
                    and keyword_key not in GENERIC_GROUNDED_KEYWORDS
                ):
                    allowed_by_key.setdefault(keyword_key, keyword)

    result: dict[str, tuple[str, ...]] = {}
    for request in plan.retrieval_requests:
        selected: list[str] = []
        rejected: list[str] = []
        seen: set[str] = set()

        for value in request.exact_keywords:
            keyword = _clean(value)
            keyword_key = _compact_match_key(keyword)
            if len(keyword_key) < 2 or keyword_key in seen:
                continue
            if keyword_key in GENERIC_GROUNDED_KEYWORDS:
                rejected.append(keyword)
                continue
            if (
                validate_against_keyword_result
                and keyword_key not in allowed_by_key
            ):
                rejected.append(keyword)
                continue
            seen.add(keyword_key)
            selected.append(allowed_by_key.get(keyword_key, keyword))

        result[request.request_id] = tuple(selected)
        _debug_print(
            debug,
            "GROUND_KEYWORDS",
            f"{request.request_id} selected={selected} rejected={rejected}",
        )

    _debug_print(
        debug,
        "GROUND_KEYWORDS",
        f"allowed_pool={list(allowed_by_key.values())}",
    )
    return result


def execute_search_plan(
    plan: dict[str, Any],
    catalog: ManifestGraphCatalog,
    store: HybridChunkStore,
    *,
    question: str,
    keyword_result: Mapping[str, Any] | None = None,
    group_members: Mapping[str, Iterable[str]] | None = None,
    dense_k: int = DEFAULT_DENSE_K,
    keyword_dense_k: int = DEFAULT_KEYWORD_DENSE_K,
    lexical_k: int | None = None,
    rerank_k_per_document_request: int = DEFAULT_RERANK_K,
    context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    context_max_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    query_score_threshold: float = DEFAULT_QUERY_SCORE_THRESHOLD,
    min_query_relative_relevance: float = (
        DEFAULT_MIN_QUERY_RELATIVE_RELEVANCE
    ),
    min_relative_relevance: float = DEFAULT_MIN_RELATIVE_RELEVANCE,
    debug: bool = False,
) -> ContextBundle:
    question = _clean(question)
    if not question:
        raise ValueError("question is required as an executor runtime input")

    _debug_print(debug, "START", f"question={question!r}")
    if lexical_k is not None:
        _debug_print(
            debug,
            "CONFIG",
            f"lexical_k={lexical_k} ignored because BM25 retrieval is disabled",
        )
    prepared = validate_and_normalize_search_plan(plan)
    period_intent = _infer_period_intent(question)
    _debug_print(
        debug,
        "VALIDATE",
        f"retrieval_requests={len(prepared.retrieval_requests):,}",
    )
    _debug_print(
        debug,
        "PERIOD_INTENT",
        (
            f"kind={period_intent.kind if period_intent else 'unspecified'} "
            f"report_type={period_intent.report_type if period_intent else '-'} "
            f"base_month={period_intent.base_month if period_intent else '-'}"
        ),
    )
    resolved = [
        catalog.resolve_request(request, group_members, debug=debug)
        for request in prepared.retrieval_requests
    ]
    if not any(item.documents for item in resolved):
        raise LookupError("no documents matched any retrieval request")

    keywords_by_request = grounded_keywords_by_request(
        question,
        prepared,
        keyword_result=keyword_result,
        debug=debug,
    )

    # 1차 검색은 질문의 기간 분류와 정확히 일치하는 정기공시만 사용한다.
    # 검색계획에 포함된 비정기공시는 그대로 유지하지만, 정기공시 3종을
    # 처음부터 한꺼번에 벡터 검색하지 않는다.
    primary_resolved = _primary_resolved_requests(
        resolved,
        period_intent,
    )
    primary_raw = retrieve_candidates(
        primary_resolved,
        store,
        keywords_by_request=keywords_by_request,
        dense_k=dense_k,
        keyword_dense_k=keyword_dense_k,
        debug=debug,
    )
    primary_fused = fuse_and_deduplicate_candidates(
        primary_raw,
        debug=debug,
    )
    primary_reranked = rerank_candidates(
        prepared,
        primary_fused,
        store=store,
        keywords_by_request=keywords_by_request,
        per_document_request_limit=rerank_k_per_document_request,
        query_weight=query_weight,
        keyword_weight=keyword_weight,
        debug=debug,
    )
    primary_expanded = expand_context(
        primary_reranked,
        store,
        debug=debug,
    )

    missing_exact_cells = _missing_exact_period_cells(
        primary_resolved,
        primary_expanded,
        period_intent=period_intent,
        query_score_threshold=query_score_threshold,
        min_query_relative_relevance=min_query_relative_relevance,
        min_relative_relevance=min_relative_relevance,
    )

    # 정확한 기간 청크로 채우지 못한 요청×연도 셀만 정기공시 3종을
    # 검색한다. fallback 후보에는 상대 임계값을 다시 적용하지 않고,
    # pack_context에서 질문 임베딩 점수가 가장 높은 청크 1개를 고른다.
    fallback_candidates: list[Candidate] = []
    if any(missing_exact_cells.values()):
        fallback_resolved = _fallback_resolved_requests(
            resolved,
            missing_exact_cells,
        )
        fallback_raw = retrieve_candidates(
            fallback_resolved,
            store,
            keywords_by_request=keywords_by_request,
            dense_k=dense_k,
            keyword_dense_k=keyword_dense_k,
            debug=debug,
        )
        fallback_fused = fuse_and_deduplicate_candidates(
            fallback_raw,
            debug=debug,
        )
        fallback_candidates = rerank_candidates(
            prepared,
            fallback_fused,
            store=store,
            keywords_by_request=keywords_by_request,
            per_document_request_limit=rerank_k_per_document_request,
            query_weight=query_weight,
            keyword_weight=keyword_weight,
            debug=debug,
        )
        _debug_print(
            debug,
            "PERIOD_FALLBACK",
            (
                f"missing_cells={sum(len(value) for value in missing_exact_cells.values()):,} "
                f"candidates={len(fallback_candidates):,}"
            ),
        )

    return pack_context(
        question,
        resolved,
        primary_expanded,
        char_budget=context_char_budget,
        max_chunks=context_max_chunks,
        keywords_by_request=keywords_by_request,
        query_score_threshold=query_score_threshold,
        min_query_relative_relevance=min_query_relative_relevance,
        min_relative_relevance=min_relative_relevance,
        period_intent=period_intent,
        fallback_candidates=fallback_candidates,
        fallback_cells=missing_exact_cells,
        debug=debug,
    )


def validate_and_normalize_search_plan(plan: dict[str, Any]) -> PreparedSearchPlan:
    if not isinstance(plan, dict):
        raise TypeError("search plan must be a dict")
    unknown_top = set(plan) - PLAN_FIELDS
    if unknown_top:
        raise ValueError(f"unsupported top-level fields: {sorted(unknown_top)}")
    raw_requests = plan.get("retrieval_requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("retrieval_requests must be a non-empty list")

    requests = []
    for index, raw in enumerate(raw_requests, start=1):
        request_id = f"request_{index:03d}"
        if not isinstance(raw, dict):
            raise TypeError(f"{request_id} must be an object")
        missing = REQUEST_REQUIRED_FIELDS - set(raw)
        unknown = set(raw) - REQUEST_REQUIRED_FIELDS
        if missing:
            raise ValueError(f"{request_id} missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"{request_id} unsupported fields: {sorted(unknown)}")

        query = _clean(raw.get("query"))
        if not query:
            raise ValueError(f"{request_id}.query must not be empty")
        report_types = _normalize_string_list(
            raw.get("normalized_report_types"),
            f"{request_id}.normalized_report_types",
        )
        if not 1 <= len(report_types) <= MAX_REPORT_TYPES_PER_REQUEST:
            raise ValueError(
                f"{request_id}.normalized_report_types must contain 1 to "
                f"{MAX_REPORT_TYPES_PER_REQUEST} values"
            )
        invalid = sorted(set(report_types) - ALLOWED_REPORT_TYPES)
        if invalid:
            raise ValueError(
                f"{request_id} has unsupported normalized_report_types: {invalid}"
            )

        date_basis = _clean(raw.get("date_basis"))
        if date_basis not in DATE_BASES:
            raise ValueError(f"{request_id}.date_basis must be one of {sorted(DATE_BASES)}")
        period = _normalize_period(raw.get("period"), date_basis, request_id)
        use_graph = raw.get("use_correction_graph")
        if not isinstance(use_graph, bool):
            raise TypeError(f"{request_id}.use_correction_graph must be boolean")

        scope = _normalize_company_scope(raw.get("company_scope"), request_id)
        keywords = _normalize_string_list(
            raw.get("exact_keywords"), f"{request_id}.exact_keywords"
        )

        requests.append(
            RetrievalRequest(
                request_id=request_id,
                query=query,
                normalized_report_types=report_types,
                period=period,
                use_correction_graph=use_graph,
                company_scope=scope,
                exact_keywords=keywords,
            )
        )
    return PreparedSearchPlan(tuple(requests))


def retrieve_candidates(
    resolved: list[ResolvedRequest],
    store: HybridChunkStore,
    *,
    keywords_by_request: Mapping[str, tuple[str, ...]],
    dense_k: int,
    keyword_dense_k: int,
    debug: bool = False,
) -> list[Candidate]:
    if dense_k <= 0:
        raise ValueError("dense_k must be positive")
    if keyword_dense_k <= 0:
        raise ValueError("keyword_dense_k must be positive")
    candidates: list[Candidate] = []
    for item in resolved:
        request = item.request
        query = request.query
        keywords = keywords_by_request.get(request.request_id, ())
        _debug_print(
            debug,
            "QUERY",
            f"{request.request_id} dense_query={query!r} "
            f"grounded_keywords={list(keywords)}",
        )
        groups: dict[
            tuple[str, str, str],
            list[DocumentRef],
        ] = defaultdict(list)
        for document in item.documents:
            source_type = (
                "periodic"
                if document.normalized_report_type in PERIODIC_REPORT_TYPES
                else "planned"
            )
            groups[
                (
                    document.corp_code,
                    _period_bucket(document, item.date_basis_for(document)),
                    source_type,
                )
            ].append(document)
        for group_index, (group_key, documents) in enumerate(
            groups.items(),
            start=1,
        ):
            _, period_bucket, source_type = group_key
            doc_ids = tuple(document.doc_id for document in documents)
            query_dense = store.dense_search(
                query=query,
                doc_ids=doc_ids,
                top_k=dense_k,
            )
            keyword_dense_rows: list[tuple[int, str, list[dict[str, Any]]]] = []
            for keyword_index, keyword in enumerate(keywords, start=1):
                rows = store.dense_search(
                    query=keyword,
                    doc_ids=doc_ids,
                    top_k=keyword_dense_k,
                )
                keyword_dense_rows.append((keyword_index, keyword, rows))
                _debug_print(
                    debug,
                    "KEYWORD_DENSE",
                    f"{request.request_id} group={group_index}/{len(groups)} "
                    f"keyword={keyword!r} results={len(rows):,}",
                )
            _debug_print(
                debug,
                "RETRIEVE",
                f"{request.request_id} group={group_index}/{len(groups)} "
                f"company={documents[0].corp_name if documents else '-'} "
                f"source={source_type} period_bucket={period_bucket} "
                f"documents={len(documents):,} query_dense={len(query_dense):,} "
                f"keyword_dense={sum(len(rows) for _, _, rows in keyword_dense_rows):,}",
            )
            candidates.extend(
                _ranked_rows_to_candidates(
                    query_dense,
                    request_id=request.request_id,
                    channel=f"{request.request_id}:g{group_index}:query_dense",
                )
            )
            for keyword_index, _, rows in keyword_dense_rows:
                candidates.extend(
                    _ranked_rows_to_candidates(
                        rows,
                        request_id=request.request_id,
                        channel=(
                            f"{request.request_id}:g{group_index}:"
                            f"keyword_dense:{keyword_index}"
                        ),
                    )
                )
    _debug_print(debug, "RETRIEVE", f"raw_candidates={len(candidates):,}")
    return candidates


def fuse_and_deduplicate_candidates(
    candidates: list[Candidate], *, debug: bool = False
) -> list[Candidate]:
    by_chunk: dict[str, Candidate] = {}
    for candidate in candidates:
        existing = by_chunk.setdefault(
            candidate.chunk_id,
            Candidate(candidate.chunk_id, candidate.text, dict(candidate.metadata)),
        )
        existing.request_ids.update(candidate.request_ids)
        for channel, rank in candidate.channel_ranks.items():
            existing.channel_ranks[channel] = min(
                rank, existing.channel_ranks.get(channel, rank)
            )
        for channel, score in candidate.channel_scores.items():
            existing.channel_scores[channel] = max(
                score, existing.channel_scores.get(channel, -math.inf)
            )

    content_seen: dict[tuple[str, str], Candidate] = {}
    kept: list[Candidate] = []
    for candidate in by_chunk.values():
        normalized = re.sub(r"\s+", " ", candidate.text).strip()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        key = (candidate.doc_id, digest)
        original = content_seen.get(key)
        if original:
            original.request_ids.update(candidate.request_ids)
            for channel, rank in candidate.channel_ranks.items():
                original.channel_ranks[channel] = min(
                    rank, original.channel_ranks.get(channel, rank)
                )
            for channel, score in candidate.channel_scores.items():
                original.channel_scores[channel] = max(
                    score, original.channel_scores.get(channel, -math.inf)
                )
            continue
        content_seen[key] = candidate
        kept.append(candidate)

    for candidate in kept:
        candidate.request_fusion_scores = {
            request_id: sum(
                1.0 / (RRF_K + rank)
                for channel, rank in candidate.channel_ranks.items()
                if channel.startswith(f"{request_id}:")
            )
            for request_id in candidate.request_ids
        }
        candidate.fusion_score = max(candidate.request_fusion_scores.values(), default=0.0)
    result = sorted(kept, key=lambda item: item.fusion_score, reverse=True)
    _debug_print(
        debug,
        "FUSE",
        f"input={len(candidates):,} unique_chunks={len(result):,}",
    )
    return result


def rerank_candidates(
    plan: PreparedSearchPlan,
    candidates: list[Candidate],
    *,
    store: HybridChunkStore,
    keywords_by_request: Mapping[str, tuple[str, ...]],
    per_document_request_limit: int,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    debug: bool = False,
) -> list[Candidate]:
    if per_document_request_limit <= 0:
        raise ValueError("per_document_request_limit must be positive")
    if query_weight < 0.0 or keyword_weight < 0.0:
        raise ValueError("dense score weights must be non-negative")
    if query_weight + keyword_weight <= 0.0:
        raise ValueError("at least one dense score weight must be positive")

    requests = {request.request_id: request for request in plan.retrieval_requests}
    request_candidates = {
        request_id: [
            candidate for candidate in candidates if request_id in candidate.request_ids
        ]
        for request_id in requests
    }

    rescored_by_request: dict[str, dict[str, dict[str, float]]] = {}
    dense_scorer = getattr(store, "dense_scores", None)
    for request_id, request in requests.items():
        query_texts = tuple(
            dict.fromkeys(
                (request.query, *keywords_by_request.get(request_id, ()))
            )
        )
        values: dict[str, dict[str, float]] = {}
        if callable(dense_scorer) and request_candidates[request_id]:
            try:
                values = dense_scorer(
                    queries=query_texts,
                    candidates=tuple(request_candidates[request_id]),
                )
            except Exception as exc:
                _debug_print(
                    debug,
                    "DENSE_RESCORE",
                    f"{request_id} failed={type(exc).__name__}; "
                    "using retrieval-channel scores",
                )
        rescored_by_request[request_id] = values

    for candidate in candidates:
        candidate.request_scores = {}
        candidate.request_evidence_strengths = {}
        candidate.request_match_signals = {}
        candidate.request_keyword_scores = {}
        for request_id in candidate.request_ids:
            request = requests[request_id]
            keywords = keywords_by_request.get(request_id, ())
            score_map = rescored_by_request.get(request_id, {})
            query_score = score_map.get(request.query, {}).get(
                candidate.chunk_id,
                _retrieval_channel_score(
                    candidate,
                    request_id=request_id,
                    channel_fragment=":query_dense",
                ),
            )
            keyword_scores = {
                keyword: score_map.get(keyword, {}).get(
                    candidate.chunk_id,
                    _retrieval_channel_score(
                        candidate,
                        request_id=request_id,
                        channel_fragment=f":keyword_dense:{index}",
                    ),
                )
                for index, keyword in enumerate(keywords, start=1)
            }
            keyword_score = max(keyword_scores.values(), default=0.0)
            if keywords:
                weight_total = query_weight + keyword_weight
                effective_query_weight = query_weight / weight_total
                effective_keyword_weight = keyword_weight / weight_total
            else:
                effective_query_weight = 1.0
                effective_keyword_weight = 0.0
            final_score = (
                effective_query_weight * query_score
                + effective_keyword_weight * keyword_score
            )
            candidate.request_scores[request_id] = final_score
            candidate.request_evidence_strengths[request_id] = query_score
            candidate.request_keyword_scores[request_id] = keyword_scores
            candidate.request_match_signals[request_id] = {
                "query_dense_similarity": query_score,
                "keyword_dense_similarity": keyword_score,
                "query_weight": effective_query_weight,
                "keyword_weight": effective_keyword_weight,
                "final_dense_score": final_score,
            }
        candidate.rerank_score = max(candidate.request_scores.values(), default=float("-inf"))

    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        for request_id in candidate.request_ids:
            grouped[(candidate.doc_id, request_id)].append(candidate)
    keep_ids: set[str] = set()
    for (_, request_id), values in grouped.items():
        values.sort(
            key=lambda item: item.request_scores.get(request_id, float("-inf")),
            reverse=True,
        )
        keep_ids.update(item.chunk_id for item in values[:per_document_request_limit])
    result = sorted(
        (candidate for candidate in candidates if candidate.chunk_id in keep_ids),
        key=lambda item: item.rerank_score,
        reverse=True,
    )
    _debug_print(
        debug,
        "RERANK",
        f"input={len(candidates):,} retained={len(result):,} "
        f"per_document_request_limit={per_document_request_limit} "
        f"query_weight={query_weight:.2f} keyword_weight={keyword_weight:.2f}",
    )
    if debug:
        for request_id in requests:
            top = sorted(
                (
                    candidate
                    for candidate in result
                    if request_id in candidate.request_ids
                ),
                key=lambda candidate: candidate.request_scores.get(
                    request_id, float("-inf")
                ),
                reverse=True,
            )[:5]
            for rank, candidate in enumerate(top, start=1):
                signals = candidate.request_match_signals.get(request_id, {})
                _debug_print(
                    True,
                    "RERANK_SCORE",
                    f"{request_id} rank={rank} chunk={candidate.chunk_id} "
                    f"query={signals.get('query_dense_similarity', 0.0):.4f} "
                    f"keyword={signals.get('keyword_dense_similarity', 0.0):.4f} "
                    f"final={signals.get('final_dense_score', 0.0):.4f}",
                )
    return result


def expand_context(
    reranked: list[Candidate],
    store: HybridChunkStore,
    *,
    seeds_per_document_request: int = 2,
    neighbor_window: int = 1,
    debug: bool = False,
) -> list[Candidate]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in reranked:
        for request_id in candidate.request_ids:
            grouped[(candidate.doc_id, request_id)].append(candidate)
    seed_ids: set[str] = set()
    for (_, request_id), values in grouped.items():
        values.sort(
            key=lambda item: item.request_scores.get(request_id, float("-inf")),
            reverse=True,
        )
        seed_ids.update(item.chunk_id for item in values[:seeds_per_document_request])

    by_id = {candidate.chunk_id: candidate for candidate in reranked}
    for seed_id in sorted(seed_ids):
        seed = by_id[seed_id]
        for row in store.get_related_chunks(chunk=seed, window=neighbor_window):
            chunk_id = _clean(row.get("chunk_id"))
            text = _clean(row.get("text"))
            if not chunk_id or not text or chunk_id in by_id:
                continue
            by_id[chunk_id] = Candidate(
                chunk_id=chunk_id,
                text=text,
                metadata=dict(row.get("metadata") or {}),
                request_ids=set(seed.request_ids),
                request_fusion_scores={key: value * 0.6 for key, value in seed.request_fusion_scores.items()},
                request_scores={key: value * 0.78 for key, value in seed.request_scores.items()},
                request_evidence_strengths=dict(seed.request_evidence_strengths),
                request_match_signals={
                    key: dict(value)
                    for key, value in seed.request_match_signals.items()
                },
                request_keyword_scores={
                    key: dict(value)
                    for key, value in seed.request_keyword_scores.items()
                },
                fusion_score=seed.fusion_score * 0.6,
                rerank_score=seed.rerank_score * 0.78,
                is_expanded=True,
                parent_chunk_id=seed.chunk_id,
            )
    result = sorted(by_id.values(), key=lambda item: item.rerank_score, reverse=True)
    _debug_print(
        debug,
        "EXPAND",
        f"seeds={len(seed_ids):,} before={len(reranked):,} after={len(result):,}",
    )
    return result


def _subset_resolved_request(
    item: ResolvedRequest,
    documents: Iterable[DocumentRef],
) -> ResolvedRequest:
    selected_documents = tuple(
        sorted(
            {document.doc_id: document for document in documents}.values(),
            key=_document_sort_key,
        )
    )
    selected_ids = {document.doc_id for document in selected_documents}
    selected_periodic = [
        document
        for document in selected_documents
        if document.normalized_report_type in PERIODIC_REPORT_TYPES
    ]
    return ResolvedRequest(
        request=item.request,
        scope=item.scope,
        seed_documents=tuple(
            document
            for document in item.seed_documents
            if document.doc_id in selected_ids
        ),
        documents=selected_documents,
        graph_expanded_doc_ids=tuple(
            doc_id
            for doc_id in item.graph_expanded_doc_ids
            if doc_id in selected_ids
        ),
        date_basis_by_doc_id=tuple(
            (doc_id, basis)
            for doc_id, basis in item.date_basis_by_doc_id
            if doc_id in selected_ids
        ),
        fallback_seed_doc_ids=tuple(
            doc_id
            for doc_id in item.fallback_seed_doc_ids
            if doc_id in selected_ids
        ),
        report_type_fallback_applied=bool(selected_periodic),
        report_type_fallback_types=tuple(
            report_type
            for report_type in PERIODIC_REPORT_TYPES
            if any(
                document.normalized_report_type == report_type
                for document in selected_periodic
            )
        ),
        report_type_fallback_document_ids=tuple(
            sorted(document.doc_id for document in selected_periodic)
        ),
    )


def _document_matches_period_intent(
    document: DocumentRef,
    period_intent: PeriodIntent | None,
) -> bool:
    if period_intent is None:
        return True
    if document.normalized_report_type != period_intent.report_type:
        return False
    return (
        period_intent.base_month is None
        or document.base_month == period_intent.base_month
    )


def _primary_resolved_requests(
    resolved: list[ResolvedRequest],
    period_intent: PeriodIntent | None,
) -> list[ResolvedRequest]:
    if period_intent is None:
        return resolved
    primary: list[ResolvedRequest] = []
    for item in resolved:
        documents = [
            document
            for document in item.documents
            if (
                document.normalized_report_type not in PERIODIC_REPORT_TYPES
                or _document_matches_period_intent(document, period_intent)
            )
        ]
        primary.append(_subset_resolved_request(item, documents))
    return primary


def _fallback_resolved_requests(
    resolved: list[ResolvedRequest],
    fallback_cells: Mapping[str, set[str]],
) -> list[ResolvedRequest]:
    fallback: list[ResolvedRequest] = []
    for item in resolved:
        years = fallback_cells.get(item.request.request_id, set())
        documents = [
            document
            for document in item.documents
            if document.normalized_report_type in PERIODIC_REPORT_TYPES
            and _period_bucket(
                document,
                item.date_basis_for(document),
            )
            in years
        ]
        fallback.append(_subset_resolved_request(item, documents))
    return fallback


def _qualification_state(
    resolved: list[ResolvedRequest],
    candidates: list[Candidate],
    *,
    query_score_threshold: float,
    min_query_relative_relevance: float,
    min_relative_relevance: float,
    debug: bool = False,
) -> tuple[
    dict[str, list[Candidate]],
    dict[str, float],
    dict[str, float],
]:
    best_request_scores: dict[str, float] = {}
    best_query_scores: dict[str, float] = {}
    for item in resolved:
        request_id = item.request.request_id
        best_request_scores[request_id] = max(
            (
                candidate.request_scores.get(request_id, -math.inf)
                for candidate in candidates
                if request_id in candidate.request_ids
            ),
            default=-math.inf,
        )
        best_query_scores[request_id] = max(
            (
                candidate.request_match_signals.get(request_id, {}).get(
                    "query_dense_similarity",
                    -math.inf,
                )
                for candidate in candidates
                if request_id in candidate.request_ids
            ),
            default=-math.inf,
        )

    def qualifies(candidate: Candidate, request_id: str) -> bool:
        final_score = candidate.request_scores.get(request_id, -math.inf)
        best_final = best_request_scores.get(request_id, -math.inf)
        query_score = candidate.request_match_signals.get(request_id, {}).get(
            "query_dense_similarity",
            -math.inf,
        )
        best_query = best_query_scores.get(request_id, -math.inf)
        if not all(
            math.isfinite(value)
            for value in (final_score, best_final, query_score, best_query)
        ):
            return False
        effective_query_threshold = max(
            query_score_threshold,
            best_query * min_query_relative_relevance,
        )
        return (
            query_score >= effective_query_threshold
            and final_score >= best_final * min_relative_relevance
        )

    qualified_pools: dict[str, list[Candidate]] = {}
    for item in resolved:
        request_id = item.request.request_id
        pool = [
            candidate
            for candidate in candidates
            if request_id in candidate.request_ids
            and qualifies(candidate, request_id)
        ]
        pool.sort(
            key=lambda candidate: candidate.request_scores.get(
                request_id,
                -math.inf,
            ),
            reverse=True,
        )
        qualified_pools[request_id] = pool
        if debug:
            best_query = best_query_scores[request_id]
            effective_query_threshold = (
                max(
                    query_score_threshold,
                    best_query * min_query_relative_relevance,
                )
                if math.isfinite(best_query)
                else query_score_threshold
            )
            _debug_print(
                True,
                "QUALIFY",
                f"{request_id} candidates={sum(request_id in candidate.request_ids for candidate in candidates):,} "
                f"qualified={len(pool):,} best_query={best_query:.4f} "
                f"best_final={best_request_scores[request_id]:.4f} "
                f"query_threshold={effective_query_threshold:.4f} "
                f"final_relative={min_relative_relevance:.2f}",
            )
    return qualified_pools, best_request_scores, best_query_scores


def _missing_exact_period_cells(
    resolved: list[ResolvedRequest],
    candidates: list[Candidate],
    *,
    period_intent: PeriodIntent | None,
    query_score_threshold: float,
    min_query_relative_relevance: float,
    min_relative_relevance: float,
) -> dict[str, set[str]]:
    if period_intent is None:
        return {
            item.request.request_id: set()
            for item in resolved
        }
    qualified_pools, _, _ = _qualification_state(
        resolved,
        candidates,
        query_score_threshold=query_score_threshold,
        min_query_relative_relevance=min_query_relative_relevance,
        min_relative_relevance=min_relative_relevance,
    )
    missing: dict[str, set[str]] = {}
    for item in resolved:
        request_id = item.request.request_id
        documents = {
            document.doc_id: document
            for document in item.documents
        }
        covered = {
            _period_bucket(document, item.date_basis_for(document))
            for candidate in qualified_pools.get(request_id, [])
            if (document := documents.get(candidate.doc_id)) is not None
        }
        missing[request_id] = set(
            _requested_year_buckets(item.request.period)
        ) - covered
    return missing


def _candidate_query_score(
    candidate: Candidate,
    request_id: str,
) -> float:
    return candidate.request_match_signals.get(request_id, {}).get(
        "query_dense_similarity",
        -math.inf,
    )


def _document_period_label(document: DocumentRef) -> str:
    labels = {
        ("annual_report", 12): "사업보고서(연간)",
        ("semiannual_report", 6): "반기보고서(상반기 누적)",
        ("quarterly_report", 3): "1분기보고서",
        ("quarterly_report", 9): "3분기보고서(누적)",
    }
    return labels.get(
        (document.normalized_report_type, document.base_month),
        document.report_nm or document.normalized_report_type,
    )


def pack_context(
    question: str,
    resolved: list[ResolvedRequest],
    candidates: list[Candidate],
    *,
    char_budget: int,
    max_chunks: int,
    keywords_by_request: Mapping[str, tuple[str, ...]],
    query_score_threshold: float = DEFAULT_QUERY_SCORE_THRESHOLD,
    min_query_relative_relevance: float = (
        DEFAULT_MIN_QUERY_RELATIVE_RELEVANCE
    ),
    min_relative_relevance: float = DEFAULT_MIN_RELATIVE_RELEVANCE,
    period_intent: PeriodIntent | None = None,
    fallback_candidates: list[Candidate] | None = None,
    fallback_cells: Mapping[str, set[str]] | None = None,
    debug: bool = False,
) -> ContextBundle:
    if char_budget <= 0 or max_chunks <= 0:
        raise ValueError("context limits must be positive")
    if not 0.0 <= query_score_threshold <= 1.0:
        raise ValueError("query_score_threshold must be between 0 and 1")
    if not 0.0 <= min_query_relative_relevance <= 1.0:
        raise ValueError(
            "min_query_relative_relevance must be between 0 and 1"
        )
    if not 0.0 <= min_relative_relevance <= 1.0:
        raise ValueError("min_relative_relevance must be between 0 and 1")
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    used_chars = 0

    def add(candidate: Candidate) -> bool:
        nonlocal used_chars
        if candidate.chunk_id in selected_ids or len(selected) >= max_chunks:
            return False
        size = len(candidate.text)
        if used_chars + size > char_budget:
            return False
        selected.append(candidate)
        selected_ids.add(candidate.chunk_id)
        used_chars += size
        return True

    fallback_candidates = list(fallback_candidates or [])
    fallback_cells = {
        request_id: set(years)
        for request_id, years in (fallback_cells or {}).items()
    }
    qualified_pools, best_request_scores, best_query_scores = (
        _qualification_state(
            resolved,
            candidates,
            query_score_threshold=query_score_threshold,
            min_query_relative_relevance=min_query_relative_relevance,
            min_relative_relevance=min_relative_relevance,
            debug=debug,
        )
    )

    documents_by_request = {
        item.request.request_id: {
            document.doc_id: document
            for document in item.documents
        }
        for item in resolved
    }
    items_by_request = {
        item.request.request_id: item
        for item in resolved
    }
    cell_pools: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for item in resolved:
        request_id = item.request.request_id
        documents = documents_by_request[request_id]
        for candidate in qualified_pools.get(request_id, []):
            document = documents.get(candidate.doc_id)
            if document is None:
                continue
            period_bucket = _period_bucket(
                document,
                item.date_basis_for(document),
            )
            cell_pools[(request_id, period_bucket)].append(candidate)

    for (request_id, _), pool in cell_pools.items():
        pool.sort(
            key=lambda candidate: (
                candidate.request_scores.get(request_id, -math.inf),
                candidate.rerank_score,
            ),
            reverse=True,
        )

    cell_selected_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    selected_request_ids: dict[str, set[str]] = defaultdict(set)
    fallback_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def assign_to_cell(
        candidate: Candidate,
        request_id: str,
        period_bucket: str,
    ) -> bool:
        cell = (request_id, period_bucket)
        if candidate.chunk_id in cell_selected_ids[cell]:
            return True
        if candidate.chunk_id not in selected_ids and not add(candidate):
            return False
        cell_selected_ids[cell].append(candidate.chunk_id)
        selected_request_ids[candidate.chunk_id].add(request_id)
        return True

    expected_cells = [
        (item.request.request_id, period_bucket)
        for item in resolved
        for period_bucket in _requested_year_buckets(item.request.period)
    ]

    # 모든 셀에 1개씩 먼저 배정해 앞쪽 셀이 2개를 차지하면서 뒤쪽 셀을
    # 밀어내지 않도록 한다.
    for request_id, period_bucket in expected_cells:
        for candidate in cell_pools.get((request_id, period_bucket), []):
            if assign_to_cell(candidate, request_id, period_bucket):
                break

    # 1차 검색으로 비어 있는 셀에만 fallback 후보 1개를 배정한다.
    fallback_pools: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for item in resolved:
        request_id = item.request.request_id
        requested_fallback_years = fallback_cells.get(request_id, set())
        if not requested_fallback_years:
            continue
        documents = documents_by_request[request_id]
        for candidate in fallback_candidates:
            if request_id not in candidate.request_ids:
                continue
            document = documents.get(candidate.doc_id)
            if document is None:
                continue
            period_bucket = _period_bucket(
                document,
                item.date_basis_for(document),
            )
            if period_bucket in requested_fallback_years:
                fallback_pools[(request_id, period_bucket)].append(candidate)

    for (request_id, _), pool in fallback_pools.items():
        pool.sort(
            key=lambda candidate: (
                _candidate_query_score(candidate, request_id),
                candidate.request_scores.get(request_id, -math.inf),
                candidate.rerank_score,
            ),
            reverse=True,
        )

    for request_id, period_bucket in expected_cells:
        cell = (request_id, period_bucket)
        if len(cell_selected_ids[cell]) >= DEFAULT_FALLBACK_CHUNKS_PER_CELL:
            continue
        for candidate in fallback_pools.get(cell, []):
            if not assign_to_cell(candidate, request_id, period_bucket):
                continue
            item = items_by_request[request_id]
            document = documents_by_request[request_id].get(candidate.doc_id)
            if document is not None:
                fallback_records[request_id].append(
                    {
                        "period_bucket": period_bucket,
                        "chunk_id": candidate.chunk_id,
                        "requested_period": (
                            period_intent.label
                            if period_intent is not None
                            else "지정 기간"
                        ),
                        "used_period": _document_period_label(document),
                        "used_report_name": document.report_nm,
                        "used_report_type": document.normalized_report_type,
                        "used_base_month": document.base_month,
                        "query_score": round(
                            _candidate_query_score(candidate, request_id),
                            6,
                        ),
                    }
                )
            break

    # 모든 셀의 첫 번째 청크를 확보한 뒤, 1차 기간 필터를 통과한 같은
    # 셀의 적격 후보 중 차순위 청크를 하나 더 배정한다. 두 번째 후보는
    # 셀별 상위 후보이며 셀당 최대 2개를 넘기지 않는다.
    second_options: list[tuple[float, float, str, str, Candidate]] = []
    for request_id, period_bucket in expected_cells:
        cell = (request_id, period_bucket)
        if not cell_selected_ids[cell] or cell in fallback_pools:
            continue
        for candidate in cell_pools.get(cell, []):
            if candidate.chunk_id in cell_selected_ids[cell]:
                continue
            second_options.append(
                (
                    candidate.request_scores.get(request_id, -math.inf),
                    candidate.rerank_score,
                    request_id,
                    period_bucket,
                    candidate,
                )
            )
            break
    second_options.sort(key=lambda value: (value[0], value[1]), reverse=True)
    for _, _, request_id, period_bucket, candidate in second_options:
        if len(selected) >= max_chunks:
            break
        if len(cell_selected_ids[(request_id, period_bucket)]) >= (
            DEFAULT_EXACT_CHUNKS_PER_CELL
        ):
            continue
        assign_to_cell(candidate, request_id, period_bucket)

    chunks = tuple(
        {
            "chunk_id": candidate.chunk_id,
            "text": candidate.text,
            "metadata": candidate.metadata,
            "retrieval": {
                "request_ids": sorted(
                    selected_request_ids.get(candidate.chunk_id, set())
                ),
                "fusion_score": round(candidate.fusion_score, 6),
                "rerank_score": round(candidate.rerank_score, 6),
                "request_scores": {
                    key: round(value, 6) for key, value in sorted(candidate.request_scores.items())
                },
                "evidence_strengths": {
                    key: round(value, 6)
                    for key, value in sorted(
                        candidate.request_evidence_strengths.items()
                    )
                },
                "match_signals": {
                    request_id: {
                        key: round(value, 6)
                        for key, value in sorted(signals.items())
                    }
                    for request_id, signals in sorted(
                        candidate.request_match_signals.items()
                    )
                },
                "keyword_scores": {
                    request_id: {
                        keyword: round(value, 6)
                        for keyword, value in sorted(scores.items())
                    }
                    for request_id, scores in sorted(
                        candidate.request_keyword_scores.items()
                    )
                },
                "is_expanded": candidate.is_expanded,
                "parent_chunk_id": candidate.parent_chunk_id,
            },
        }
        for candidate in selected
    )
    selected_doc_ids = {candidate.doc_id for candidate in selected}
    all_documents = {
        document.doc_id: document
        for item in resolved
        for document in item.documents
        if document.doc_id in selected_doc_ids
    }
    diagnostics = []
    for item in resolved:
        request_id = item.request.request_id
        selected_count = sum(
            request_id in selected_request_ids.get(candidate.chunk_id, set())
            for candidate in selected
        )
        candidate_count = len(
            {
                candidate.chunk_id
                for candidate in [*candidates, *fallback_candidates]
                if request_id in candidate.request_ids
            }
        )
        qualified_count = len(qualified_pools[request_id])
        expected_periods = set(_requested_year_buckets(item.request.period))
        documents_by_id = {
            document.doc_id: document for document in item.documents
        }
        available_periods = {
            _period_bucket(document, item.date_basis_for(document))
            for document in item.documents
        }
        qualified_periods = {
            _period_bucket(document, item.date_basis_for(document))
            for candidate in qualified_pools[request_id]
            if (document := documents_by_id.get(candidate.doc_id)) is not None
        }
        selected_periods = {
            period_bucket
            for (cell_request_id, period_bucket), chunk_ids in cell_selected_ids.items()
            if cell_request_id == request_id and chunk_ids
        }
        missing_qualified_periods = sorted(expected_periods - qualified_periods)
        packing_omitted_periods = sorted(
            (expected_periods & qualified_periods) - selected_periods
        )
        missing_periods = sorted(expected_periods - selected_periods)
        if not item.documents:
            status = "no_document"
        elif not qualified_count:
            status = "insufficient_relevance"
        elif not selected_count:
            status = "no_chunk_selected"
        elif missing_periods:
            status = "partial"
        else:
            status = "covered"
        diagnostics.append(
            {
                "request_id": request_id,
                "query": item.request.query,
                "grounded_keywords": list(
                    keywords_by_request.get(request_id, ())
                ),
                "status": status,
                "seed_document_ids": [document.doc_id for document in item.seed_documents],
                "resolved_document_ids": [document.doc_id for document in item.documents],
                "graph_expanded_document_ids": list(item.graph_expanded_doc_ids),
                "requested_date_basis": item.request.period.date_basis,
                "date_basis_used": sorted(
                    set(dict(item.date_basis_by_doc_id).values())
                ),
                "event_date_fallback_applied": bool(item.fallback_seed_doc_ids),
                "event_date_fallback_seed_document_ids": list(item.fallback_seed_doc_ids),
                "period_intent": (
                    {
                        "kind": period_intent.kind,
                        "report_type": period_intent.report_type,
                        "base_month": period_intent.base_month,
                    }
                    if period_intent is not None
                    else None
                ),
                "periodic_baseline_applied": period_intent is not None,
                "periodic_baseline_types": (
                    [period_intent.report_type]
                    if period_intent is not None
                    else list(item.report_type_fallback_types)
                ),
                "periodic_baseline_document_ids": list(
                    document.doc_id
                    for document in item.documents
                    if document.normalized_report_type in PERIODIC_REPORT_TYPES
                    and (
                        period_intent is None
                        or _document_matches_period_intent(
                            document,
                            period_intent,
                        )
                    )
                ),
                # 이전 결과 소비 코드와의 호환성을 위해 기존 키도 유지
                "report_type_fallback_applied": bool(
                    fallback_records.get(request_id)
                ),
                "report_type_fallback_types": sorted(
                    {
                        str(record.get("used_report_type") or "")
                        for record in fallback_records.get(request_id, [])
                        if record.get("used_report_type")
                    }
                ),
                "report_type_fallback_document_ids": sorted(
                    {
                        candidate.doc_id
                        for candidate in fallback_candidates
                        if candidate.chunk_id
                        in {
                            str(record.get("chunk_id") or "")
                            for record in fallback_records.get(request_id, [])
                        }
                    }
                ),
                "period_fallbacks": fallback_records.get(request_id, []),
                "fallback_relative_thresholds_applied": False,
                "candidate_chunks": candidate_count,
                "qualified_candidate_chunks": qualified_count,
                "selected_chunks": selected_count,
                "expected_period_buckets": sorted(expected_periods),
                "available_period_buckets": sorted(available_periods),
                "qualified_period_buckets": sorted(qualified_periods),
                "selected_period_buckets": sorted(selected_periods),
                "covered_period_buckets": sorted(
                    expected_periods & selected_periods
                ),
                "missing_period_buckets": missing_periods,
                "missing_qualified_period_buckets": missing_qualified_periods,
                "packing_omitted_period_buckets": packing_omitted_periods,
                "relevance_threshold": {
                    "query_absolute": query_score_threshold,
                    "query_relative_to_request_best": (
                        min_query_relative_relevance
                    ),
                    "final_relative_to_request_best": min_relative_relevance,
                    "request_best_query_score": (
                        round(best_query_scores[request_id], 6)
                        if math.isfinite(best_query_scores[request_id])
                        else None
                    ),
                    "request_effective_query_threshold": (
                        round(
                            max(
                                query_score_threshold,
                                best_query_scores[request_id]
                                * min_query_relative_relevance,
                            ),
                            6,
                        )
                        if math.isfinite(best_query_scores[request_id])
                        else None
                    ),
                    "request_best_final_score": (
                        round(best_request_scores[request_id], 6)
                        if math.isfinite(best_request_scores[request_id])
                        else None
                    ),
                },
            }
        )
    covered_buckets: set[str] = set()
    for item in resolved:
        documents = {document.doc_id: document for document in item.documents}
        for candidate in selected:
            if item.request.request_id not in selected_request_ids.get(
                candidate.chunk_id,
                set(),
            ):
                continue
            document = documents.get(candidate.doc_id)
            if document:
                covered_buckets.add(
                    f"{document.corp_name}|{_period_bucket(document, item.date_basis_for(document))}"
                )
    bundle = ContextBundle(
        question=question,
        documents=tuple(sorted(all_documents.values(), key=_document_sort_key)),
        chunks=chunks,
        total_chars=sum(len(chunk["text"]) for chunk in chunks),
        covered_request_ids=tuple(
            sorted(
                {
                    request_id
                    for request_ids in selected_request_ids.values()
                    for request_id in request_ids
                }
            )
        ),
        covered_period_buckets=tuple(sorted(covered_buckets)),
        request_diagnostics=tuple(diagnostics),
    )
    _debug_print(
        debug,
        "PACK",
        f"selected_chunks={len(bundle.chunks):,} characters={bundle.total_chars:,} "
        f"covered_requests={len(bundle.covered_request_ids):,}/{len(resolved):,} "
        f"missing_request_year_cells={sum(len(item['missing_period_buckets']) for item in diagnostics):,}",
    )
    _debug_print(debug, "DONE", "context bundle ready")
    return bundle


class JsonlHybridChunkStore:
    """소규모 검증용 JSONL 검색 저장소. 운영에서는 Chroma 저장소를 쓴다."""

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.rows = _load_jsonl(self.path)
        self._dense_cache: dict[tuple[str, ...], tuple[Any, Any, list[int]]] = {}

    def dense_search(self, *, query: str, doc_ids: tuple[str, ...], top_k: int) -> list[dict[str, Any]]:
        from sklearn.feature_extraction.text import HashingVectorizer

        key = tuple(sorted(doc_ids))
        cached = self._dense_cache.get(key)
        if cached is None:
            indices = [
                index for index, row in enumerate(self.rows)
                if _clean((row.get("metadata") or {}).get("doc_id")) in doc_ids
            ]
            if not indices:
                return []
            vectorizer = HashingVectorizer(
                analyzer="char_wb", ngram_range=(2, 3), n_features=2**18,
                alternate_sign=False, norm="l2",
            )
            matrix = vectorizer.transform([self.rows[index].get("text") or "" for index in indices])
            cached = (vectorizer, matrix, indices)
            self._dense_cache[key] = cached
        vectorizer, matrix, indices = cached
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        return [
            {**self.rows[indices[position]], "score": float(scores[position])}
            for position in np.argsort(-scores)[:top_k]
            if scores[position] > 0
        ]

    def dense_scores(
        self,
        *,
        queries: tuple[str, ...],
        candidates: tuple[Candidate, ...],
    ) -> dict[str, dict[str, float]]:
        """검증용 저장소에서 합쳐진 후보를 같은 Dense 공간으로 재점수화한다."""
        if not queries or not candidates:
            return {}
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
        )
        candidate_matrix = vectorizer.transform(
            [candidate.text for candidate in candidates]
        )
        query_matrix = vectorizer.transform(list(queries))
        similarity = (query_matrix @ candidate_matrix.T).toarray()
        return {
            query: {
                # 문자 n-gram 검증 백엔드의 코사인은 BGE-M3보다 수치가
                # 작게 형성되므로 순서를 보존하는 단조 보정만 적용한다.
                candidate.chunk_id: math.sqrt(
                    max(float(similarity[query_index, candidate_index]), 0.0)
                )
                for candidate_index, candidate in enumerate(candidates)
            }
            for query_index, query in enumerate(queries)
        }

    def lexical_search(
        self,
        *,
        query: str,
        exact_keywords: tuple[str, ...],
        doc_ids: tuple[str, ...],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """구형 호출부 호환용 Dense 별칭이며 BM25를 실행하지 않는다."""
        del exact_keywords
        return self.dense_search(query=query, doc_ids=doc_ids, top_k=top_k)

    def get_related_chunks(self, *, chunk: Candidate, window: int) -> list[dict[str, Any]]:
        return _find_related_rows(self.rows, chunk=chunk, window=window)


class CompanyChromaHybridChunkStore:
    """기존 회사별 Chroma DB를 사용하는 운영용 Dense 저장소.

    Dense 검색은 확정된 doc_id를 Chroma ``where`` 조건으로 먼저 제한한다.
    설치된 Chroma 버전이나 기존 메타데이터가 필터 검색을 지원하지 않는
    경우에만 회사 컬렉션 전체 HNSW 후보를 단계적으로 넓히는 방식으로
    안전하게 되돌아간다. 전체 질의와 원 질문에서 검증된 키워드는 모두
    동일한 BGE-M3 임베딩 공간에서 검색하고 재점수화한다.
    """

    def __init__(
        self,
        catalog: ManifestGraphCatalog,
        db_path: str | Path,
        company_map_path: str | Path,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_seq_length: int = 512,
        candidate_sizes: Iterable[int] | None = None,
        cache_max_entries: int = 128,
        chroma_get_batch_size: int = 100,
        debug: bool = False,
        retriever: Any | None = None,
    ):
        if cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be positive")
        if chroma_get_batch_size <= 0:
            raise ValueError("chroma_get_batch_size must be positive")

        self.catalog = catalog
        self.debug = debug
        self.cache_max_entries = cache_max_entries
        self.chroma_get_batch_size = chroma_get_batch_size
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._document_cache: OrderedDict[
            tuple[str, ...], list[dict[str, Any]]
        ] = OrderedDict()

        if retriever is None:
            try:
                from .vector_retriever import VectorRetriever
            except ImportError as exc:
                raise RuntimeError(
                    "search_info/vector_retriever.py is required for the company Chroma backend"
                ) from exc
            kwargs: dict[str, Any] = {
                "db_path": db_path,
                "company_map_path": company_map_path,
                "model_name": model_name,
                "device": device,
                "max_seq_length": max_seq_length,
            }
            if candidate_sizes is not None:
                kwargs["candidate_sizes"] = candidate_sizes
            retriever = VectorRetriever(**kwargs)

        self.retriever = retriever
        configured_sizes = (
            candidate_sizes
            if candidate_sizes is not None
            else getattr(retriever, "candidate_sizes", (100, 300, 1000, 3000, 10000))
        )
        self.candidate_sizes = tuple(
            sorted({int(size) for size in configured_sizes if int(size) > 0})
        )
        if not self.candidate_sizes:
            raise ValueError("candidate_sizes must not be empty")

        _debug_print(
            self.debug,
            "STORE",
            f"backend=company_chroma db={Path(db_path)} map={Path(company_map_path)} "
            f"model={model_name} device={device}",
        )

    def dense_search(
        self, *, query: str, doc_ids: tuple[str, ...], top_k: int
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not doc_ids:
            return []
        allowed_by_company = self._group_doc_ids_by_company(doc_ids)
        embedding = _lru_get(self._query_embedding_cache, query)
        if embedding is None:
            embedding = self.retriever.encode_query(query)
            _lru_put(
                self._query_embedding_cache,
                query,
                embedding,
                self.cache_max_entries,
            )
            _debug_print(self.debug, "DENSE", "query embedding created")
        else:
            _debug_print(self.debug, "DENSE", "query embedding cache hit")

        merged: list[dict[str, Any]] = []
        for corp_name, allowed_ids in allowed_by_company.items():
            merged.extend(
                self._dense_search_company(
                    corp_name=corp_name,
                    allowed_doc_ids=allowed_ids,
                    query_embedding=embedding,
                    top_k=top_k,
                )
            )
        merged.sort(key=lambda row: float(row.get("distance", math.inf)))
        return merged[:top_k]

    def dense_scores(
        self,
        *,
        queries: tuple[str, ...],
        candidates: tuple[Candidate, ...],
    ) -> dict[str, dict[str, float]]:
        """합쳐진 후보 전체에 query/keyword 코사인 점수를 다시 계산한다."""
        if not queries or not candidates:
            return {}

        query_vectors: dict[str, np.ndarray] = {}
        for query in queries:
            embedding = _lru_get(self._query_embedding_cache, query)
            if embedding is None:
                embedding = self.retriever.encode_query(query)
                _lru_put(
                    self._query_embedding_cache,
                    query,
                    embedding,
                    self.cache_max_entries,
                )
            query_vectors[query] = _unit_vector(embedding)

        grouped: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            document = self.catalog.by_id.get(candidate.doc_id)
            if document is not None:
                grouped[document.corp_name].append(candidate)

        result: dict[str, dict[str, float]] = {
            query: {} for query in queries
        }
        embedded_count = 0
        for corp_name, company_candidates in grouped.items():
            collection = self.retriever.get_collection(corp_name)
            unique_candidates = {
                candidate.chunk_id: candidate for candidate in company_candidates
            }
            ordered_ids = sorted(unique_candidates)
            for start in range(0, len(ordered_ids), self.chroma_get_batch_size):
                batch_ids = ordered_ids[
                    start : start + self.chroma_get_batch_size
                ]
                try:
                    raw = collection.get(ids=batch_ids, include=["embeddings"])
                except Exception as exc:
                    _debug_print(
                        self.debug,
                        "DENSE_RESCORE",
                        f"company={corp_name} embedding_get_failed="
                        f"{type(exc).__name__}",
                    )
                    continue
                ids = list(raw.get("ids") or [])
                embeddings = raw.get("embeddings")
                if embeddings is None:
                    continue
                for chunk_id, embedding in zip(ids, embeddings):
                    if embedding is None:
                        continue
                    document_vector = _unit_vector(embedding)
                    if document_vector.size == 0:
                        continue
                    embedded_count += 1
                    for query, query_vector in query_vectors.items():
                        if query_vector.size != document_vector.size:
                            continue
                        result[query][_clean(chunk_id)] = float(
                            np.dot(query_vector, document_vector)
                        )

        _debug_print(
            self.debug,
            "DENSE_RESCORE",
            f"queries={len(queries):,} candidates={len(candidates):,} "
            f"embedded_candidates={embedded_count:,}",
        )
        return result

    def lexical_search(
        self,
        *,
        query: str,
        exact_keywords: tuple[str, ...],
        doc_ids: tuple[str, ...],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """구형 호출부 호환용 Dense 별칭이며 BM25를 실행하지 않는다."""
        del exact_keywords
        return self.dense_search(query=query, doc_ids=doc_ids, top_k=top_k)

    def get_related_chunks(
        self, *, chunk: Candidate, window: int
    ) -> list[dict[str, Any]]:
        return _find_related_rows(
            self._get_document_rows((chunk.doc_id,)),
            chunk=chunk,
            window=window,
        )

    def get_document_rows(
        self, doc_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        """진단 도구에서 확정 문서의 원문 청크를 읽는 공개 진입점."""
        return list(self._get_document_rows(tuple(doc_ids)))

    def _dense_search_company(
        self,
        *,
        corp_name: str,
        allowed_doc_ids: set[str],
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self.retriever.has_company(corp_name):
            raise KeyError(f"company is missing from Chroma map: {corp_name}")
        collection = self.retriever.get_collection(corp_name)
        collection_count = int(self.retriever.get_collection_count(corp_name))
        if collection_count <= 0:
            return []

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                where=_chroma_value_filter("doc_id", sorted(allowed_doc_ids)),
                n_results=min(top_k, collection_count),
                include=["documents", "metadatas", "distances"],
            )
            prefiltered = [
                row
                for row in self._parse_query_result(result)
                if _clean((row.get("metadata") or {}).get("doc_id"))
                in allowed_doc_ids
            ]
            _debug_print(
                self.debug,
                "DENSE",
                f"company={corp_name} prefiltered=true "
                f"allowed_documents={len(allowed_doc_ids):,} "
                f"matched={len(prefiltered):,}",
            )
            if prefiltered:
                return prefiltered[:top_k]
        except Exception as exc:
            _debug_print(
                self.debug,
                "DENSE",
                f"company={corp_name} prefilter_failed={type(exc).__name__}; "
                "using adaptive fallback",
            )

        sizes = sorted(
            {
                min(size, collection_count)
                for size in (*self.candidate_sizes, max(top_k, top_k * 5))
                if size > 0
            }
        )
        filtered: list[dict[str, Any]] = []
        last_candidate_k = 0
        for candidate_k in sizes:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                include=["documents", "metadatas", "distances"],
            )
            filtered = [
                row
                for row in self._parse_query_result(result)
                if _clean((row.get("metadata") or {}).get("doc_id")) in allowed_doc_ids
            ]
            _debug_print(
                self.debug,
                "DENSE",
                f"company={corp_name} candidate_k={candidate_k:,} "
                f"allowed_documents={len(allowed_doc_ids):,} matched={len(filtered):,}",
            )
            last_candidate_k = candidate_k
            if len(filtered) >= top_k or candidate_k >= collection_count:
                break

        if len(filtered) < top_k and last_candidate_k < collection_count:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=collection_count,
                include=["documents", "metadatas", "distances"],
            )
            filtered = [
                row
                for row in self._parse_query_result(result)
                if _clean((row.get("metadata") or {}).get("doc_id")) in allowed_doc_ids
            ]
            _debug_print(
                self.debug,
                "DENSE",
                f"company={corp_name} candidate_k={collection_count:,} "
                f"allowed_documents={len(allowed_doc_ids):,} matched={len(filtered):,} "
                "full_collection=true",
            )
        return filtered[:top_k]

    @staticmethod
    def _parse_query_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            distance_value = float(distance)
            rows.append(
                {
                    "chunk_id": _clean(chunk_id),
                    "text": _clean(text),
                    "metadata": dict(metadata or {}),
                    "distance": distance_value,
                    "score": 1.0 - distance_value,
                }
            )
        return rows

    def _group_doc_ids_by_company(
        self, doc_ids: Iterable[str]
    ) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        unknown: list[str] = []
        for doc_id in dict.fromkeys(doc_ids):
            document = self.catalog.by_id.get(doc_id)
            if document is None:
                unknown.append(doc_id)
                continue
            grouped[document.corp_name].add(document.doc_id)
        if unknown:
            raise LookupError(
                "document IDs are missing from the manifest: " + ", ".join(unknown[:10])
            )
        return grouped

    def _get_document_rows(self, doc_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        key = tuple(sorted(set(doc_ids)))
        cached = _lru_get(self._document_cache, key)
        if cached is not None:
            return cached

        rows: list[dict[str, Any]] = []
        allowed_by_company = self._group_doc_ids_by_company(key)
        for corp_name, allowed_ids in allowed_by_company.items():
            collection = self.retriever.get_collection(corp_name)
            company_rows: list[dict[str, Any]] = []
            try:
                ordered_ids = sorted(allowed_ids)
                for start in range(0, len(ordered_ids), self.chroma_get_batch_size):
                    batch = ordered_ids[start : start + self.chroma_get_batch_size]
                    result = collection.get(
                        where=_chroma_value_filter("doc_id", batch),
                        include=["documents", "metadatas"],
                    )
                    company_rows.extend(self._parse_get_result(result))
            except Exception as exc:
                _debug_print(
                    self.debug,
                    "STORE_GET",
                    f"company={corp_name} filtered get failed={type(exc).__name__}; "
                    "falling back to collection get + Python filter",
                )
                result = collection.get(include=["documents", "metadatas"])
                company_rows = self._parse_get_result(result)

            rows.extend(
                row
                for row in company_rows
                if _clean((row.get("metadata") or {}).get("doc_id")) in allowed_ids
            )

        unique = {
            _clean(row.get("chunk_id")): row
            for row in rows
            if _clean(row.get("chunk_id")) and _clean(row.get("text"))
        }
        cached = list(unique.values())
        _lru_put(self._document_cache, key, cached, self.cache_max_entries)
        return cached

    @staticmethod
    def _parse_get_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": _clean(chunk_id),
                "text": _clean(text),
                "metadata": dict(metadata or {}),
            }
            for chunk_id, text, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]


class ChromaBgeHybridChunkStore:
    """단일 Chroma 컬렉션을 사용하는 BGE-M3 Dense 저장소."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        collection_name: str = "disclosure_chunks",
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_batch_size: int = 32,
        cache_max_entries: int = 128,
    ):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("chromadb and sentence-transformers are required") from exc
        self.client = chromadb.PersistentClient(path=str(Path(db_path)))
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_batch_size = embedding_batch_size
        if cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be positive")
        self.cache_max_entries = cache_max_entries
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._document_cache: OrderedDict[
            tuple[str, ...], list[dict[str, Any]]
        ] = OrderedDict()

    def dense_search(self, *, query: str, doc_ids: tuple[str, ...], top_k: int) -> list[dict[str, Any]]:
        available = len(self._get_document_rows(doc_ids))
        if available == 0:
            return []
        embedding = _lru_get(self._query_embedding_cache, query)
        if embedding is None:
            vector = self.model.encode(
                [query], batch_size=self.embedding_batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
            )[0]
            embedding = vector.tolist()
            _lru_put(
                self._query_embedding_cache,
                query,
                embedding,
                self.cache_max_entries,
            )
        result = self.collection.query(
            query_embeddings=[embedding],
            where=_chroma_value_filter("doc_id", doc_ids),
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        texts = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata or {},
                "score": 1.0 - float(distance),
            }
            for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances)
        ]

    def dense_scores(
        self,
        *,
        queries: tuple[str, ...],
        candidates: tuple[Candidate, ...],
    ) -> dict[str, dict[str, float]]:
        if not queries or not candidates:
            return {}
        vectors = self.model.encode(
            list(queries),
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vectors = {
            query: _unit_vector(vector)
            for query, vector in zip(queries, vectors)
        }
        result: dict[str, dict[str, float]] = {
            query: {} for query in queries
        }
        ordered_ids = list(dict.fromkeys(candidate.chunk_id for candidate in candidates))
        raw = self.collection.get(ids=ordered_ids, include=["embeddings"])
        embeddings = raw.get("embeddings")
        if embeddings is None:
            return result
        for chunk_id, embedding in zip(raw.get("ids") or [], embeddings):
            if embedding is None:
                continue
            document_vector = _unit_vector(embedding)
            for query, query_vector in query_vectors.items():
                if query_vector.size == document_vector.size:
                    result[query][_clean(chunk_id)] = float(
                        np.dot(query_vector, document_vector)
                    )
        return result

    def lexical_search(
        self,
        *,
        query: str,
        exact_keywords: tuple[str, ...],
        doc_ids: tuple[str, ...],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """구형 호출부 호환용 Dense 별칭이며 BM25를 실행하지 않는다."""
        del exact_keywords
        return self.dense_search(query=query, doc_ids=doc_ids, top_k=top_k)

    def get_related_chunks(self, *, chunk: Candidate, window: int) -> list[dict[str, Any]]:
        return _find_related_rows(self._get_document_rows((chunk.doc_id,)), chunk=chunk, window=window)

    def _get_document_rows(self, doc_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        key = tuple(sorted(doc_ids))
        cached = _lru_get(self._document_cache, key)
        if cached is None:
            result = self.collection.get(
                where=_chroma_value_filter("doc_id", doc_ids),
                include=["documents", "metadatas"],
            )
            cached = [
                {"chunk_id": chunk_id, "text": text, "metadata": metadata or {}}
                for chunk_id, text, metadata in zip(
                    result.get("ids") or [],
                    result.get("documents") or [],
                    result.get("metadatas") or [],
                )
            ]
            _lru_put(
                self._document_cache,
                key,
                cached,
                self.cache_max_entries,
            )
        return cached


def _normalize_period(raw: Any, date_basis: str, request_id: str) -> PeriodSpec:
    if not isinstance(raw, dict) or set(raw) != {"start", "end"}:
        raise ValueError(f"{request_id}.period must contain only start and end")
    start = _clean(raw.get("start"))
    end = _clean(raw.get("end"))
    if date_basis == "base_year":
        if not _is_base_year_value(start) or not _is_base_year_value(end):
            raise ValueError(
                f"{request_id}: base_year period must use YYYY, YYYY-Qn, or YYYY-Hn"
            )
    elif not _is_day_value(start) or not _is_day_value(end):
        raise ValueError(
            f"{request_id}: {date_basis} period must use YYYY-MM-DD"
        )
    start_parts = _parse_period_value(start, is_end=False, request_id=request_id)
    end_parts = _parse_period_value(end, is_end=True, request_id=request_id)
    if date_basis == "base_year":
        lower = start_parts[0] * 100 + start_parts[1]
        upper = end_parts[0] * 100 + end_parts[1]
    else:
        lower = start_parts[0] * 10_000 + start_parts[1] * 100 + start_parts[2]
        upper = end_parts[0] * 10_000 + end_parts[1] * 100 + end_parts[2]
    if lower > upper:
        raise ValueError(f"{request_id}.period start must not be after end")
    return PeriodSpec(start, end, date_basis, lower, upper)


def _as_base_year_period(period: PeriodSpec) -> PeriodSpec:
    """날짜 기반 요청을 정기보고서 fallback용 사업연도 범위로 바꾼다."""
    if period.date_basis == "base_year":
        return period
    start_year = int(period.start[:4])
    end_year = int(period.end[:4])
    return PeriodSpec(
        start=str(start_year),
        end=str(end_year),
        date_basis="base_year",
        lower_key=start_year * 100 + 1,
        upper_key=end_year * 100 + 12,
    )


def _is_day_value(value: str) -> bool:
    match = PERIOD_RE.fullmatch(value)
    return bool(match and match.group("day"))


def _is_base_year_value(value: str) -> bool:
    match = PERIOD_RE.fullmatch(value)
    return bool(match and not match.group("day"))


def _parse_period_value(value: str, *, is_end: bool, request_id: str) -> tuple[int, int, int | None]:
    match = PERIOD_RE.fullmatch(value)
    if not match:
        raise ValueError(
            f"{request_id}: unsupported period value {value!r}; use YYYY, YYYY-Qn, YYYY-Hn, or YYYY-MM-DD"
        )
    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")
    if month_text and day_text:
        parsed = date(year, int(month_text), int(day_text))
        return parsed.year, parsed.month, parsed.day
    quarter = match.group("quarter")
    if quarter:
        number = int(quarter)
        month = number * 3 if is_end else (number - 1) * 3 + 1
        day = _last_day(year, month) if is_end else 1
        return year, month, day
    half = match.group("half")
    if half:
        number = int(half)
        month = (6 if number == 1 else 12) if is_end else (1 if number == 1 else 7)
        day = _last_day(year, month) if is_end else 1
        return year, month, day
    return year, (12 if is_end else 1), (_last_day(year, 12) if is_end else 1)


def _normalize_company_scope(raw: Any, request_id: str = "company_scope") -> CompanyScope:
    if not isinstance(raw, dict) or set(raw) != {"scope_type", "scope_values"}:
        raise ValueError(f"{request_id}.company_scope must contain only scope_type and scope_values")
    scope_type = _clean(raw.get("scope_type"))
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"{request_id}.scope_type must be one of {sorted(SCOPE_TYPES)}")
    values = _normalize_string_list(
        raw.get("scope_values"), f"{request_id}.company_scope.scope_values"
    )
    if scope_type == "all" and values:
        raise ValueError(f"{request_id}.scope_values must be empty for all")
    if scope_type != "all" and not values:
        raise ValueError(f"{request_id}.scope_values must not be empty")
    if scope_type == "sector":
        invalid = sorted(set(values) - ALLOWED_SECTORS)
        if invalid:
            raise ValueError(f"{request_id} has unsupported sectors: {invalid}")
    if scope_type == "industry":
        invalid = sorted(set(values) - ALLOWED_INDUSTRIES)
        if invalid:
            raise ValueError(f"{request_id} has unsupported industries: {invalid}")
    return CompanyScope(scope_type, values)


def _effective_document_date_basis(
    document: DocumentRef,
    *,
    requested_basis: str,
    event_date_fallback: str | None,
) -> str | None:
    """문서별 실제 날짜 기준을 결정한다.

    event_date가 있는 문서는 반드시 event_date로 판정한다. 해당 값이 없거나
    유효하지 않은 문서만 rcept_dt를 대체 기준으로 허용한다. base_year와
    rcept_dt 요청은 의미가 다르므로 다른 날짜 기준으로 자동 전환하지 않는다.
    """
    if requested_basis == "base_year":
        return "base_year" if document.base_year is not None else None
    if requested_basis == "rcept_dt":
        return "rcept_dt" if _date_key(document.rcept_dt) is not None else None
    if _date_key(document.event_date) is not None:
        return "event_date"
    if event_date_fallback == "rcept_dt" and _date_key(document.rcept_dt) is not None:
        return "rcept_dt"
    return None


def _document_matches_period(
    document: DocumentRef,
    period: PeriodSpec,
    *,
    date_basis: str | None = None,
) -> bool:
    basis = date_basis or period.date_basis
    if basis == "base_year":
        if document.base_year is None:
            return False
        if document.base_month is None and (
            "-Q" in period.start
            or "-H" in period.start
            or "-Q" in period.end
            or "-H" in period.end
        ):
            return False
        key = document.base_year * 100 + (document.base_month or 12)
    else:
        value = document.event_date if basis == "event_date" else document.rcept_dt
        key = _date_key(value)
        if key is None:
            return False
    return period.lower_key <= key <= period.upper_key


def _date_key(value: Any) -> int | None:
    digits = re.sub(r"\D", "", _clean(value))
    if len(digits) != 8:
        return None
    try:
        parsed = date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
    except ValueError:
        return None
    return parsed.year * 10_000 + parsed.month * 100 + parsed.day


def _document_from_manifest(raw: dict[str, Any]) -> DocumentRef:
    return DocumentRef(
        doc_id=_clean(raw.get("doc_id")),
        corp_code=_clean(raw.get("corp_code")),
        corp_name=_clean(raw.get("corp_name")),
        normalized_report_type=_canonical_report_type(raw),
        report_nm=_clean(raw.get("report_nm")),
        base_year=_to_int_or_none(raw.get("base_year")),
        base_month=_to_int_or_none(raw.get("base_month")),
        event_date=_clean(raw.get("event_date")),
        rcept_dt=_clean(raw.get("rcept_dt")),
        rcept_no=_clean(raw.get("rcept_no")),
        is_correction=_to_bool(raw.get("is_correction")),
        disclosure_chain_id=_clean(raw.get("disclosure_chain_id") or raw.get("chain_id")),
    )


def _canonical_report_type(raw: dict[str, Any]) -> str:
    value = _clean(raw.get("normalized_report_type"))
    aliases = {
        "사업보고서": "annual_report",
        "반기보고서": "semiannual_report",
        "분기보고서": "quarterly_report",
        "annual": "annual_report",
        "semiannual": "semiannual_report",
        "quarterly": "quarterly_report",
    }
    return aliases.get(value, aliases.get(_clean(raw.get("doc_subtype")), value))


def _period_bucket(document: DocumentRef, date_basis: str) -> str:
    if date_basis == "base_year":
        return str(document.base_year or "unknown")
    value = document.event_date if date_basis == "event_date" else document.rcept_dt
    key = _date_key(value)
    return str(key // 10_000) if key else "unknown"


def _requested_year_buckets(period: PeriodSpec) -> tuple[str, ...]:
    """요청 기간을 요청 × 연도 패킹에 사용할 연도 버킷으로 변환한다."""
    try:
        start_year = int(period.start[:4])
        end_year = int(period.end[:4])
    except (TypeError, ValueError):
        return ()
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    return tuple(str(year) for year in range(start_year, end_year + 1))


def _infer_period_intent(question: str) -> PeriodIntent | None:
    """원 질문에서 정기공시 기간 분류와 base_month를 결정한다."""
    text = _clean(question)
    if not text:
        return None

    quarter_match = re.search(
        r"(?:제\s*)?([1-4])\s*분기|\bQ([1-4])\b|\b([1-4])Q\b",
        text,
        flags=re.IGNORECASE,
    )
    if quarter_match:
        quarter = int(next(value for value in quarter_match.groups() if value))
        by_quarter = {
            1: PeriodIntent("q1", "quarterly_report", 3, "1분기"),
            2: PeriodIntent("q2", "semiannual_report", 6, "2분기"),
            3: PeriodIntent("q3", "quarterly_report", 9, "3분기"),
            4: PeriodIntent("q4", "annual_report", 12, "4분기"),
        }
        return by_quarter[quarter]

    if re.search(
        r"(?:상반기|(?<!하)반기|\bH1\b|\b1H\b)",
        text,
        re.IGNORECASE,
    ):
        return PeriodIntent(
            "half_year",
            "semiannual_report",
            6,
            "반기",
        )

    if re.search(r"(?:연간|연도별|연말|온기|사업연도|\bFY\b|사업보고서)", text, re.IGNORECASE):
        return PeriodIntent(
            "annual",
            "annual_report",
            12,
            "연간",
        )

    mentioned_months = {
        int(value)
        for value in re.findall(r"(?<!\d)(\d{1,2})\s*월", text)
        if 1 <= int(value) <= 12
    }
    if len(mentioned_months) == 1:
        month = next(iter(mentioned_months))
        by_month = {
            3: PeriodIntent("q1", "quarterly_report", 3, "3월 말"),
            6: PeriodIntent("half_year", "semiannual_report", 6, "6월 말"),
            9: PeriodIntent("q3", "quarterly_report", 9, "9월 말"),
            12: PeriodIntent("annual", "annual_report", 12, "12월 말"),
        }
        return by_month.get(month)

    if re.search(r"(?:분기별|분기)", text):
        return PeriodIntent(
            "quarterly",
            "quarterly_report",
            None,
            "분기",
        )

    # 반기·분기·월 표현 없이 연도만 있으면 연간 비교로 해석한다.
    if re.search(r"(?:19|20)\d{2}\s*년?", text) and not mentioned_months:
        return PeriodIntent(
            "annual",
            "annual_report",
            12,
            "연간",
        )
    return None


def _ranked_rows_to_candidates(
    rows: list[dict[str, Any]], *, request_id: str, channel: str
) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=_clean(row.get("chunk_id")),
            text=_clean(row.get("text")),
            metadata=dict(row.get("metadata") or {}),
            request_ids={request_id},
            channel_ranks={channel: rank},
            channel_scores={channel: float(row.get("score") or 0.0)},
        )
        for rank, row in enumerate(rows, start=1)
        if _clean(row.get("chunk_id")) and _clean(row.get("text"))
    ]


def _find_related_rows(
    rows: list[dict[str, Any]], *, chunk: Candidate, window: int
) -> list[dict[str, Any]]:
    metadata = chunk.metadata
    doc_id = chunk.doc_id
    chunk_type = _clean(metadata.get("chunk_type"))
    result = []
    if chunk_type == "narrative":
        index = _to_int_or_none(metadata.get("narrative_index"))
        if index is None:
            return []
        for row in rows:
            row_meta = row.get("metadata") or {}
            row_index = _to_int_or_none(row_meta.get("narrative_index"))
            if (
                _clean(row_meta.get("doc_id")) == doc_id
                and _clean(row_meta.get("chunk_type")) == "narrative"
                and _clean(row_meta.get("section_path")) == _clean(metadata.get("section_path"))
                and _clean(row_meta.get("subtitle")) == _clean(metadata.get("subtitle"))
                and row_index is not None
                and 0 < abs(row_index - index) <= window
            ):
                result.append(row)
        return result
    table_index = _to_int_or_none(metadata.get("table_index"))
    part_index = _to_int_or_none(metadata.get("table_part_index"))
    if table_index is None or part_index is None:
        return []
    for row in rows:
        row_meta = row.get("metadata") or {}
        row_part = _to_int_or_none(row_meta.get("table_part_index"))
        if (
            _clean(row_meta.get("doc_id")) == doc_id
            and _to_int_or_none(row_meta.get("table_index")) == table_index
            and row_part is not None
            and 0 < abs(row_part - part_index) <= window
        ):
            result.append(row)
    return result


def _chroma_value_filter(field: str, values: Iterable[Any]) -> dict[str, Any]:
    items = list(values)
    if not items:
        raise ValueError(f"empty Chroma filter: {field}")
    return {field: items[0]} if len(items) == 1 else {field: {"$in": items}}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    data = path.read_bytes()
    raw_lines = data.splitlines()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_incomplete_tail = line_number == len(raw_lines) and not data.endswith(b"\n")
            if is_incomplete_tail:
                warnings.warn(
                    f"ignored incomplete final JSONL row at {path}:{line_number}",
                    RuntimeWarning,
                )
                break
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _lru_get(cache: OrderedDict[K, V], key: K) -> V | None:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_put(
    cache: OrderedDict[K, V], key: K, value: V, max_entries: int
) -> None:
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _unit_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return vector
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _iter_string_values(value: Any) -> tuple[str, ...]:
    """중첩된 추출기 결과에서 문자열 값만 순서대로 꺼낸다."""
    if isinstance(value, str):
        cleaned = _clean(value)
        return (cleaned,) if cleaned else ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_iter_string_values(nested))
        return tuple(result)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_iter_string_values(nested))
        return tuple(result)
    return ()


def _compact_match_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", _clean(value).lower())


def _match_keys_overlap(left: str, right: str) -> bool:
    return bool(left and right and (left in right or right in left))


def _retrieval_channel_score(
    candidate: Candidate,
    *,
    request_id: str,
    channel_fragment: str,
) -> float:
    values = [
        float(score)
        for channel, score in candidate.channel_scores.items()
        if channel.startswith(f"{request_id}:")
        and channel_fragment in channel
        and math.isfinite(float(score))
    ]
    return max(values, default=0.0)


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("list value must not be a string")
    return tuple(dict.fromkeys(cleaned for value in values if (cleaned := _clean(value))))


def _normalize_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must contain only strings")
    return _unique_strings(value)


def _document_sort_key(document: DocumentRef) -> tuple[Any, ...]:
    return (document.corp_name, document.base_year or 0, document.rcept_dt, document.doc_id)


def _last_day(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date.resolution).day


def _clean(value: Any) -> str:
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


def _to_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y"}
