from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
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
DEFAULT_LEXICAL_K = 24
DEFAULT_RERANK_K = 10
DEFAULT_CONTEXT_CHAR_BUDGET = 30_000
DEFAULT_CONTEXT_MAX_CHUNKS = 18
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
    request_fusion_scores: dict[str, float] = field(default_factory=dict)
    request_scores: dict[str, float] = field(default_factory=dict)
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
        return {
            "question": self.question,
            "documents": [asdict(document) for document in self.documents],
            "chunks": list(self.chunks),
            "total_chars": self.total_chars,
            "covered_request_ids": list(self.covered_request_ids),
            "covered_period_buckets": list(self.covered_period_buckets),
            "request_diagnostics": list(self.request_diagnostics),
        }


class HybridChunkStore(Protocol):
    def dense_search(
        self, *, query: str, doc_ids: tuple[str, ...], top_k: int
    ) -> list[dict[str, Any]]: ...

    def lexical_search(
        self,
        *,
        query: str,
        exact_keywords: tuple[str, ...],
        doc_ids: tuple[str, ...],
        top_k: int,
    ) -> list[dict[str, Any]]: ...

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
        eligible = [
            document
            for document in self.documents
            if document.corp_code in company_codes
            and document.normalized_report_type in report_types
        ]
        seeds: list[DocumentRef] = []
        fallback_seed_ids: set[str] = set()
        date_basis_by_doc_id: dict[str, str] = {}
        for document in eligible:
            basis = _effective_document_date_basis(
                document,
                requested_basis=request.period.date_basis,
                event_date_fallback=self.event_date_fallback,
            )
            if basis is None:
                continue
            date_basis_by_doc_id[document.doc_id] = basis
            if _document_matches_period(document, request.period, date_basis=basis):
                seeds.append(document)
                if basis != request.period.date_basis:
                    fallback_seed_ids.add(document.doc_id)

        _debug_print(
            debug_enabled,
            "SCOPE",
            f"{request.request_id} scope={scope.scope_type}:{list(scope.scope_values)} "
            f"companies={len(company_codes):,} report_types={list(request.normalized_report_types)} "
            f"eligible_documents={len(eligible):,}",
        )
        _debug_print(
            debug_enabled,
            "PERIOD",
            f"{request.request_id} requested_basis={request.period.date_basis} "
            f"period={request.period.start}..{request.period.end} "
            f"matched={len(seeds):,} fallback_to_rcept_dt={len(fallback_seed_ids):,}",
        )

        expanded_ids: set[str] = set()
        documents = list(seeds)
        if request.use_correction_graph:
            if not self.has_correction_graph:
                raise RuntimeError(
                    f"{request.request_id}: correction graph requested but no edge file was configured"
                )
            expanded_ids = self._expand_correction_components(
                {document.doc_id for document in seeds}
            )
            for seed in seeds:
                if seed.disclosure_chain_id:
                    expanded_ids.update(self.chain_docs[seed.disclosure_chain_id])
            documents = [
                document
                for document in eligible
                if document.doc_id in expanded_ids
            ]

        for document in documents:
            if document.doc_id not in date_basis_by_doc_id:
                basis = _effective_document_date_basis(
                    document,
                    requested_basis=request.period.date_basis,
                    event_date_fallback=self.event_date_fallback,
                )
                if basis is not None:
                    date_basis_by_doc_id[document.doc_id] = basis

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
            f"seed_documents={len(seed_ids):,} resolved_documents={len(documents):,} "
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


def execute_search_plan(
    plan: dict[str, Any],
    catalog: ManifestGraphCatalog,
    store: HybridChunkStore,
    *,
    question: str,
    group_members: Mapping[str, Iterable[str]] | None = None,
    dense_k: int = DEFAULT_DENSE_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    rerank_k_per_document_request: int = DEFAULT_RERANK_K,
    context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    context_max_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    debug: bool = False,
) -> ContextBundle:
    question = _clean(question)
    if not question:
        raise ValueError("question is required as an executor runtime input")

    _debug_print(debug, "START", f"question={question!r}")
    prepared = validate_and_normalize_search_plan(plan)
    _debug_print(
        debug,
        "VALIDATE",
        f"retrieval_requests={len(prepared.retrieval_requests):,}",
    )
    resolved = [
        catalog.resolve_request(request, group_members, debug=debug)
        for request in prepared.retrieval_requests
    ]
    if not any(item.documents for item in resolved):
        raise LookupError("no documents matched any retrieval request")

    raw = retrieve_candidates(
        resolved,
        store,
        dense_k=dense_k,
        lexical_k=lexical_k,
        debug=debug,
    )
    fused = fuse_and_deduplicate_candidates(raw, debug=debug)
    reranked = rerank_candidates(
        prepared,
        fused,
        per_document_request_limit=rerank_k_per_document_request,
        debug=debug,
    )
    expanded = expand_context(reranked, store, debug=debug)
    return pack_context(
        question,
        resolved,
        expanded,
        char_budget=context_char_budget,
        max_chunks=context_max_chunks,
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
        if not 1 <= len(report_types) <= 5:
            raise ValueError(
                f"{request_id}.normalized_report_types must contain 1 to 5 values"
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
    dense_k: int,
    lexical_k: int,
    debug: bool = False,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for item in resolved:
        request = item.request
        groups: dict[tuple[str, str], list[DocumentRef]] = defaultdict(list)
        for document in item.documents:
            groups[
                (
                    document.corp_code,
                    _period_bucket(document, item.date_basis_for(document)),
                )
            ].append(document)
        for group_index, documents in enumerate(groups.values(), start=1):
            doc_ids = tuple(document.doc_id for document in documents)
            dense = store.dense_search(query=request.query, doc_ids=doc_ids, top_k=dense_k)
            lexical = store.lexical_search(
                query=request.query,
                exact_keywords=request.exact_keywords,
                doc_ids=doc_ids,
                top_k=lexical_k,
            )
            _debug_print(
                debug,
                "RETRIEVE",
                f"{request.request_id} group={group_index}/{len(groups)} "
                f"company={documents[0].corp_name if documents else '-'} "
                f"documents={len(documents):,} dense={len(dense):,} lexical={len(lexical):,}",
            )
            candidates.extend(
                _ranked_rows_to_candidates(
                    dense,
                    request_id=request.request_id,
                    channel=f"{request.request_id}:g{group_index}:dense",
                )
            )
            candidates.extend(
                _ranked_rows_to_candidates(
                    lexical,
                    request_id=request.request_id,
                    channel=f"{request.request_id}:g{group_index}:lexical",
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
    per_document_request_limit: int,
    debug: bool = False,
) -> list[Candidate]:
    requests = {request.request_id: request for request in plan.retrieval_requests}
    maxima = {
        request_id: max(
            (candidate.request_fusion_scores.get(request_id, 0.0) for candidate in candidates),
            default=1.0,
        ) or 1.0
        for request_id in requests
    }
    for candidate in candidates:
        text = candidate.text.lower()
        metadata = candidate.metadata
        title = " ".join(
            _clean(metadata.get(key)).lower()
            for key in ("table_title", "section_path", "subtitle")
        )
        quality = 0.0
        priority = _clean(metadata.get("search_priority")).lower()
        if priority == "high":
            quality += 0.18
        elif priority == "low":
            quality -= 0.18
        chunk_type = _clean(metadata.get("chunk_type")).lower()
        if chunk_type == "document_admin":
            quality -= 0.75
        elif chunk_type == "governance_table":
            quality -= 0.5

        candidate.request_scores = {}
        for request_id in candidate.request_ids:
            request = requests[request_id]
            score = candidate.request_fusion_scores.get(request_id, 0.0) / maxima[request_id]
            query_terms = tuple(dict.fromkeys(_tokenize(request.query)))
            if query_terms:
                score += 0.28 * sum(term in text for term in query_terms) / len(query_terms)
                score += 0.12 * sum(term in title for term in query_terms) / len(query_terms)
            if request.exact_keywords:
                keywords = tuple(keyword.lower() for keyword in request.exact_keywords)
                score += 0.55 * sum(keyword in text for keyword in keywords) / len(keywords)
                score += 0.22 * sum(keyword in title for keyword in keywords) / len(keywords)
            candidate.request_scores[request_id] = score + quality
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
        f"per_document_request_limit={per_document_request_limit}",
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


def pack_context(
    question: str,
    resolved: list[ResolvedRequest],
    candidates: list[Candidate],
    *,
    char_budget: int,
    max_chunks: int,
    debug: bool = False,
) -> ContextBundle:
    if char_budget <= 0 or max_chunks <= 0:
        raise ValueError("context limits must be positive")
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

    # 검색 요청별 최소 한 개의 근거를 먼저 확보한다.
    for item in resolved:
        pool = [candidate for candidate in candidates if item.request.request_id in candidate.request_ids]
        if pool:
            add(max(pool, key=lambda candidate: candidate.request_scores.get(item.request.request_id, -math.inf)))

    # 요청별 시점 버킷을 먼저 확보해 비교 연도·반기 누락을 막는다.
    period_pools: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    company_period_pools: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for item in resolved:
        docs = {document.doc_id: document for document in item.documents}
        for candidate in candidates:
            document = docs.get(candidate.doc_id)
            if document and item.request.request_id in candidate.request_ids:
                period_bucket = _period_bucket(document, item.date_basis_for(document))
                period_pools[(item.request.request_id, period_bucket)].append(candidate)
                company_period_pools[
                    (item.request.request_id, document.corp_name, period_bucket)
                ].append(candidate)
    for pool in period_pools.values():
        add(max(pool, key=lambda candidate: candidate.rerank_score))

    # 범위가 넓을 때는 점수가 높은 서로 다른 회사·시점 근거 일부를 확보한다.
    diversity_candidates = sorted(
        (
            max(pool, key=lambda candidate: candidate.rerank_score)
            for pool in company_period_pools.values()
        ),
        key=lambda candidate: candidate.rerank_score,
        reverse=True,
    )
    diversity_slots = min(len(diversity_candidates), max_chunks // 3)
    diversity_added = 0
    for candidate in diversity_candidates:
        if diversity_added >= diversity_slots:
            break
        if add(candidate):
            diversity_added += 1

    # 남은 예산은 요청별 순환 방식으로 채운다.
    request_pools = {}
    for item in resolved:
        request_id = item.request.request_id
        pool = [
            candidate
            for candidate in candidates
            if request_id in candidate.request_ids
        ]
        pool.sort(
            key=lambda candidate: candidate.request_scores.get(
                request_id, float("-inf")
            ),
            reverse=True,
        )
        request_pools[request_id] = pool
    positions = {request_id: 0 for request_id in request_pools}
    while len(selected) < max_chunks:
        progressed = False
        for request_id, pool in request_pools.items():
            while positions[request_id] < len(pool):
                candidate = pool[positions[request_id]]
                positions[request_id] += 1
                if add(candidate):
                    progressed = True
                    break
            if len(selected) >= max_chunks:
                break
        if not progressed:
            break

    if not selected:
        raise LookupError("no chunks could be packed into context")

    chunks = tuple(
        {
            "chunk_id": candidate.chunk_id,
            "text": candidate.text,
            "metadata": candidate.metadata,
            "retrieval": {
                "request_ids": sorted(candidate.request_ids),
                "fusion_score": round(candidate.fusion_score, 6),
                "rerank_score": round(candidate.rerank_score, 6),
                "request_scores": {
                    key: round(value, 6) for key, value in sorted(candidate.request_scores.items())
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
        selected_count = sum(request_id in candidate.request_ids for candidate in selected)
        candidate_count = sum(request_id in candidate.request_ids for candidate in candidates)
        diagnostics.append(
            {
                "request_id": request_id,
                "query": item.request.query,
                "status": "covered" if selected_count else "no_chunk_selected" if item.documents else "no_document",
                "seed_document_ids": [document.doc_id for document in item.seed_documents],
                "resolved_document_ids": [document.doc_id for document in item.documents],
                "graph_expanded_document_ids": list(item.graph_expanded_doc_ids),
                "requested_date_basis": item.request.period.date_basis,
                "date_basis_used": sorted(
                    set(dict(item.date_basis_by_doc_id).values())
                ),
                "event_date_fallback_applied": bool(item.fallback_seed_doc_ids),
                "event_date_fallback_seed_document_ids": list(item.fallback_seed_doc_ids),
                "candidate_chunks": candidate_count,
                "selected_chunks": selected_count,
            }
        )
    covered_buckets: set[str] = set()
    for item in resolved:
        documents = {document.doc_id: document for document in item.documents}
        for candidate in selected:
            if item.request.request_id not in candidate.request_ids:
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
        covered_request_ids=tuple(sorted({request_id for candidate in selected for request_id in candidate.request_ids})),
        covered_period_buckets=tuple(sorted(covered_buckets)),
        request_diagnostics=tuple(diagnostics),
    )
    _debug_print(
        debug,
        "PACK",
        f"selected_chunks={len(bundle.chunks):,} characters={bundle.total_chars:,} "
        f"covered_requests={len(bundle.covered_request_ids):,}/{len(resolved):,}",
    )
    _debug_print(debug, "DONE", "context bundle ready")
    return bundle


class JsonlHybridChunkStore:
    """소규모 검증용 JSONL 검색 저장소. 운영에서는 Chroma 저장소를 쓴다."""

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.rows = _load_jsonl(self.path)
        self._dense_cache: dict[tuple[str, ...], tuple[Any, Any, list[int]]] = {}
        self._lexical_cache: dict[tuple[str, ...], _Bm25Index] = {}

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

    def lexical_search(
        self, *, query: str, exact_keywords: tuple[str, ...], doc_ids: tuple[str, ...], top_k: int
    ) -> list[dict[str, Any]]:
        key = tuple(sorted(doc_ids))
        index = self._lexical_cache.get(key)
        if index is None:
            index = _Bm25Index([
                row for row in self.rows
                if _clean((row.get("metadata") or {}).get("doc_id")) in doc_ids
            ])
            self._lexical_cache[key] = index
        return index.search(query=query, exact_keywords=exact_keywords, top_k=top_k)

    def get_related_chunks(self, *, chunk: Candidate, window: int) -> list[dict[str, Any]]:
        return _find_related_rows(self.rows, chunk=chunk, window=window)


class CompanyChromaHybridChunkStore:
    """기존 회사별 Chroma DB를 사용하는 운영용 하이브리드 저장소.

    Dense 검색은 저장소의 ``VectorRetriever``와 동일하게 회사 컬렉션에서
    where 없이 HNSW 검색한 뒤, 문서 범위를 Python에서 필터링한다. 후보가
    부족하면 검색 개수를 단계적으로 늘린다. 키워드 검색은 문서 범위의
    청크만 읽어 로컬 BM25로 수행한다.
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
        self._lexical_cache: OrderedDict[tuple[str, ...], _Bm25Index] = OrderedDict()

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

    def lexical_search(
        self,
        *,
        query: str,
        exact_keywords: tuple[str, ...],
        doc_ids: tuple[str, ...],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not doc_ids:
            return []
        key = tuple(sorted(set(doc_ids)))
        index = _lru_get(self._lexical_cache, key)
        if index is None:
            rows = self._get_document_rows(key)
            index = _Bm25Index(rows)
            _lru_put(self._lexical_cache, key, index, self.cache_max_entries)
            _debug_print(
                self.debug,
                "LEXICAL",
                f"BM25 index created documents={len(key):,} chunks={len(rows):,}",
            )
        else:
            _debug_print(self.debug, "LEXICAL", "BM25 index cache hit")
        results = index.search(
            query=query,
            exact_keywords=exact_keywords,
            top_k=top_k,
        )
        _debug_print(self.debug, "LEXICAL", f"results={len(results):,}")
        return results

    def get_related_chunks(
        self, *, chunk: Candidate, window: int
    ) -> list[dict[str, Any]]:
        return _find_related_rows(
            self._get_document_rows((chunk.doc_id,)),
            chunk=chunk,
            window=window,
        )

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
                    "LEXICAL",
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
    """운영용 Chroma + BGE-M3 벡터 검색과 로컬 BM25 키워드 검색."""

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
        self._lexical_cache: OrderedDict[tuple[str, ...], _Bm25Index] = OrderedDict()

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

    def lexical_search(
        self, *, query: str, exact_keywords: tuple[str, ...], doc_ids: tuple[str, ...], top_k: int
    ) -> list[dict[str, Any]]:
        key = tuple(sorted(doc_ids))
        index = _lru_get(self._lexical_cache, key)
        if index is None:
            index = _Bm25Index(self._get_document_rows(doc_ids))
            _lru_put(
                self._lexical_cache, key, index, self.cache_max_entries
            )
        return index.search(query=query, exact_keywords=exact_keywords, top_k=top_k)

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


class _Bm25Index:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.documents = [_tokenize(row.get("text") or "") for row in rows]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequency.update(set(tokens))
        size = len(self.documents)
        self.idf = {
            term: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, *, query: str, exact_keywords: tuple[str, ...], top_k: int) -> list[dict[str, Any]]:
        query_terms = _tokenize(" ".join((query, *exact_keywords)))
        if not query_terms or not self.rows:
            return []
        scores = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / (self.average_length or 1)
                )
                score += self.idf.get(term, 0.0) * frequency * 2.5 / denominator
            text = _clean(self.rows[index].get("text")).lower()
            score += 1.5 * sum(keyword.lower() in text for keyword in exact_keywords)
            scores.append(score)
        return [
            {**self.rows[index], "score": float(scores[index])}
            for index in np.argsort(-np.asarray(scores))[:top_k]
            if scores[index] > 0
        ]


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


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[가-힣A-Za-z0-9%.-]+", value) if len(token) > 1]


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
