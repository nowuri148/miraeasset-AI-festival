from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from main import (
    initialize_components,
    run_agent,
    build_user_output,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Mirae Asset AI Festival Disclosure Agent",
    version="1.0.0",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    """
    서버 시작 시 Keyword Extractor / Company Resolver 초기화.
    """

    initialize_components()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root() -> dict[str, str]:

    return {
        "status": "ok",
        "service": "disclosure-agent",
    }


@app.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok",
    }


# ============================================================
# TEXT UTILS
# ============================================================

def normalize_text(
    value: Any,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            value
            or ""
        ).strip(),
    )


def normalize_number_text(
    value: Any,
) -> str:

    return (
        str(
            value
            or ""
        )
        .replace(
            ",",
            "",
        )
        .replace(
            " ",
            "",
        )
    )


def get_item_document(
    item: dict[str, Any],
) -> str:

    return str(
        item.get(
            "document"
        )
        or item.get(
            "text"
        )
        or ""
    ).strip()


# ============================================================
# ANSWER NUMBER EXTRACTION
# ============================================================

def extract_answer_numbers(
    text: str,
) -> set[str]:
    """
    최종 답변에서 실제 결과값으로 사용된 숫자를 추출한다.

    연도, 분기 등은 가능한 한 제외한다.

    예:
        삼성전자의 2023년 전체 설비투자 금액은
        531,139억 원입니다.

    반환:
        {"531139"}
    """

    text = str(
        text
        or ""
    )

    numbers: set[str] = set()

    # --------------------------------------------------------
    # 금액 / 주식 수
    # --------------------------------------------------------

    financial_pattern = (
        r"(?<!\d)"
        r"("
        r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?"
        r")"
        r"\s*"
        r"(?:"
        r"백만원"
        r"|천만원"
        r"|억원"
        r"|조원"
        r"|천원"
        r"|만원"
        r"|백만주"
        r"|천주"
        r"|원"
        r"|주"
        r")"
    )

    for match in re.finditer(
        financial_pattern,
        text,
    ):

        value = (
            match.group(1)
            .replace(
                ",",
                "",
            )
        )

        if value:
            numbers.add(
                value
            )

    # --------------------------------------------------------
    # 비율
    # --------------------------------------------------------

    percent_pattern = (
        r"(?<!\d)"
        r"("
        r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?"
        r")"
        r"\s*"
        r"(?:%|퍼센트)"
    )

    for match in re.finditer(
        percent_pattern,
        text,
    ):

        value = (
            match.group(1)
            .replace(
                ",",
                "",
            )
        )

        if value:

            numbers.add(
                value
                + "%"
            )

    # --------------------------------------------------------
    # 단위가 없는 답변의 fallback
    # --------------------------------------------------------

    if not numbers:

        for match in re.findall(
            r"\d[\d,\.]*",
            text,
        ):

            value = (
                match.replace(
                    ",",
                    "",
                )
            )

            try:

                integer_value = int(
                    value
                    .split(
                        ".",
                        1,
                    )[0]
                )

            except ValueError:

                integer_value = -1

            # 연도 제거
            if (
                "." not in value
                and 1900
                <= integer_value
                <= 2100
            ):
                continue

            if len(
                value
            ) < 2:
                continue

            numbers.add(
                value
            )

    return numbers


# ============================================================
# EVIDENCE TERMS
# ============================================================

def build_evidence_terms(
    result: dict[str, Any],
) -> list[str]:
    """
    Keyword Extractor 결과 중
    metric/topic/company를 evidence 탐색에 활용한다.
    """

    extracted = (
        result.get(
            "extracted"
        )
        or {}
    )

    terms: list[str] = []

    for key in (
        "metrics",
        "topic_keywords",
    ):

        values = (
            extracted.get(
                key
            )
            or []
        )

        for value in values:

            value = (
                normalize_text(
                    value
                )
            )

            if (
                value
                and value not in terms
            ):

                terms.append(
                    value
                )

    companies = (
        extracted.get(
            "target_companies"
        )
        or extracted.get(
            "companies"
        )
        or []
    )

    for company in companies:

        company = (
            normalize_text(
                company
            )
        )

        if (
            company
            and company not in terms
        ):

            terms.append(
                company
            )

    return terms


# ============================================================
# NUMBER MATCH
# ============================================================

def text_contains_answer_number(
    text: str,
    answer_numbers: set[str],
) -> bool:
    """
    text 안에 최종 답변의 핵심 숫자가 있는지 확인.
    """

    normalized_text = (
        normalize_number_text(
            text
        )
    )

    for number in (
        answer_numbers
    ):

        normalized_number = (
            normalize_number_text(
                number
            )
        )

        if (
            normalized_number
            and normalized_number
            in normalized_text
        ):

            return True

    return False


# ============================================================
# TERM COVERAGE
# ============================================================

def calculate_term_coverage(
    text: str,
    terms: list[str],
) -> float:

    if not terms:
        return 0.0

    normalized = (
        normalize_text(
            text
        ).lower()
    )

    matched = 0

    for term in terms:

        term = str(
            term
            or ""
        ).strip().lower()

        if (
            term
            and term in normalized
        ):

            matched += 1

    return (
        matched
        / len(
            terms
        )
    )


# ============================================================
# SELECT USED RESULTS
# ============================================================

def select_used_results(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Answer Generator의 used_source_ids에 해당하는
    rerank 결과를 가져온다.

    used_source_ids가 없으면 TOP1을 사용한다.
    """

    reranked_results = (
        result.get(
            "reranked_results"
        )
        or []
    )

    if not reranked_results:
        return []

    used_source_ids = (
        result.get(
            "used_source_ids"
        )
        or []
    )

    selected: list[
        dict[str, Any]
    ] = []

    for source_id in (
        used_source_ids
    ):

        try:

            index = (
                int(
                    source_id
                )
                - 1
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if (
            0
            <= index
            < len(
                reranked_results
            )
        ):

            selected.append(
                reranked_results[
                    index
                ]
            )

    if not selected:

        selected = [
            reranked_results[
                0
            ]
        ]

    return selected


# ============================================================
# CHUNK DEDUP
# ============================================================

def deduplicate_chunks(
    results: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    완전히 동일한 chunk만 제거한다.

    동일 rcept_no의 서로 다른 chunk는
    이 단계에서는 제거하지 않는다.
    """

    output: list[
        dict[str, Any]
    ] = []

    seen: set[
        str
    ] = set()

    for index, item in enumerate(
        results
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        chunk_id = str(
            item.get(
                "chunk_id"
            )
            or metadata.get(
                "chunk_id"
            )
            or ""
        ).strip()

        if chunk_id:

            key = (
                f"chunk:{chunk_id}"
            )

        else:

            key = (
                f"index:{index}"
            )

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            item
        )

    return output


# ============================================================
# DOCUMENT KEY
# ============================================================

def get_document_key(
    item: dict[str, Any],
    fallback_index: int = 0,
) -> str:

    metadata = (
        item.get(
            "metadata"
        )
        or {}
    )

    return str(
        metadata.get(
            "rcept_no"
        )
        or item.get(
            "doc_id"
        )
        or metadata.get(
            "doc_id"
        )
        or item.get(
            "chunk_id"
        )
        or metadata.get(
            "chunk_id"
        )
        or f"unknown_{fallback_index}"
    )


# ============================================================
# CHUNK SCORE
# ============================================================

def score_context_chunk(
    item: dict[str, Any],
    *,
    answer_numbers: set[str],
    terms: list[str],
    rerank_position: int,
) -> float:
    """
    같은 공시의 여러 chunk 중
    API retrieved_context에 가장 적합한 대표 chunk를 고른다.

    우선순위:
    1. 답변 숫자가 실제로 존재
    2. 질문 metric/topic coverage
    3. rerank 상위
    """

    document = (
        get_item_document(
            item
        )
    )

    if not document:
        return float(
            "-inf"
        )

    score = 0.0

    # --------------------------------------------------------
    # 답변 실제 숫자 존재
    # 가장 중요한 신호
    # --------------------------------------------------------

    if text_contains_answer_number(
        document,
        answer_numbers,
    ):

        score += 100.0

    # --------------------------------------------------------
    # 질문 표현 coverage
    # --------------------------------------------------------

    score += (
        10.0
        * calculate_term_coverage(
            document,
            terms,
        )
    )

    # --------------------------------------------------------
    # rerank 순위
    # 앞에 있을수록 조금 가점
    # --------------------------------------------------------

    score += max(
        0.0,
        5.0
        - (
            rerank_position
            * 0.1
        ),
    )

    return score


# ============================================================
# ANSWER SOURCE RECOVERY
# ============================================================

def ensure_answer_number_source(
    result: dict[str, Any],
    selected_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    선택된 source 어디에도 최종 답변 숫자가 없다면
    reranked_results에서 그 숫자가 실제로 존재하는
    가장 상위 chunk 하나만 추가한다.

    중요:
    모든 matching chunk를 추가하지 않는다.

    따라서 Q-002처럼 동일 숫자가 여러 표에 반복되어도
    retrieved_context가 불필요하게 길어지지 않는다.
    """

    answer = str(
        result.get(
            "answer"
        )
        or ""
    )

    answer_numbers = (
        extract_answer_numbers(
            answer
        )
    )

    if not answer_numbers:

        return (
            selected_results
        )

    # --------------------------------------------------------
    # 이미 선택된 chunk 중 답변 숫자가 있으면
    # 추가할 필요 없음
    # --------------------------------------------------------

    for item in (
        selected_results
    ):

        if text_contains_answer_number(
            get_item_document(
                item
            ),
            answer_numbers,
        ):

            return (
                selected_results
            )

    # --------------------------------------------------------
    # 없을 때만 rerank에서 첫 matching chunk 1개 추가
    # --------------------------------------------------------

    reranked_results = (
        result.get(
            "reranked_results"
        )
        or []
    )

    output = list(
        selected_results
    )

    for item in (
        reranked_results
    ):

        document = (
            get_item_document(
                item
            )
        )

        if not document:
            continue

        if text_contains_answer_number(
            document,
            answer_numbers,
        ):

            output.append(
                item
            )

            break

    return output


# ============================================================
# REPRESENTATIVE CHUNK PER DOCUMENT
# ============================================================

def select_best_chunk_per_document(
    result: dict[str, Any],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    동일 공시의 여러 chunk가 선택되어 있으면
    retrieved_context에는 가장 좋은 대표 chunk 하나만 사용한다.

    다만 서로 다른 공시라면 각각 하나씩 유지한다.
    """

    answer = str(
        result.get(
            "answer"
        )
        or ""
    )

    answer_numbers = (
        extract_answer_numbers(
            answer
        )
    )

    terms = (
        build_evidence_terms(
            result
        )
    )

    reranked_results = (
        result.get(
            "reranked_results"
        )
        or []
    )

    # rerank position lookup
    positions: dict[
        str,
        int
    ] = {}

    for index, item in enumerate(
        reranked_results,
        start=1,
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        chunk_id = str(
            item.get(
                "chunk_id"
            )
            or metadata.get(
                "chunk_id"
            )
            or ""
        ).strip()

        if chunk_id:

            positions[
                chunk_id
            ] = index

    grouped: OrderedDict[
        str,
        list[dict[str, Any]]
    ] = OrderedDict()

    for index, item in enumerate(
        results
    ):

        key = (
            get_document_key(
                item,
                fallback_index=index,
            )
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            item
        )

    selected: list[
        dict[str, Any]
    ] = []

    for items in (
        grouped.values()
    ):

        best_item = None
        best_score = float(
            "-inf"
        )

        for item in items:

            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            chunk_id = str(
                item.get(
                    "chunk_id"
                )
                or metadata.get(
                    "chunk_id"
                )
                or ""
            ).strip()

            position = (
                positions.get(
                    chunk_id,
                    999,
                )
            )

            score = (
                score_context_chunk(
                    item,
                    answer_numbers=answer_numbers,
                    terms=terms,
                    rerank_position=position,
                )
            )

            if (
                best_item is None
                or score > best_score
            ):

                best_item = (
                    item
                )

                best_score = (
                    score
                )

        if best_item is not None:

            selected.append(
                best_item
            )

    return selected


# ============================================================
# EVIDENCE LINE SCORE
# ============================================================

def score_evidence_line(
    line: str,
    *,
    terms: list[str],
    answer_numbers: set[str],
) -> float:

    normalized_line = (
        normalize_text(
            line
        ).lower()
    )

    if not normalized_line:
        return -1.0

    score = 0.0

    # --------------------------------------------------------
    # 답변 숫자
    # --------------------------------------------------------

    if text_contains_answer_number(
        line,
        answer_numbers,
    ):

        score += 20.0

    # --------------------------------------------------------
    # metric/topic
    # --------------------------------------------------------

    for term in terms:

        term = str(
            term
            or ""
        ).strip().lower()

        if (
            term
            and term in normalized_line
        ):

            score += 2.0

            if len(
                term
            ) >= 4:

                score += 0.5

    # --------------------------------------------------------
    # 표
    # --------------------------------------------------------

    if "|" in line:
        score += 0.5

    return score


# ============================================================
# TABLE HEADER DETECTION
# ============================================================

def is_likely_table_header(
    line: str,
) -> bool:
    """
    일반적인 공시 표의 헤더 행 여부를 보수적으로 판단한다.
    """

    if "|" not in line:
        return False

    normalized = (
        normalize_text(
            line
        ).lower()
    )

    header_terms = (
        "구분",
        "사업부문",
        "항목",
        "내용",
        "투자액",
        "매출액",
        "비율",
        "단위",
        "품목",
        "주요회사",
        "기간",
        "제40기",
        "제39기",
        "제38기",
    )

    matches = sum(
        term in normalized
        for term in header_terms
    )

    return (
        matches
        >= 1
    )


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def extract_relevant_evidence(
    document: str,
    *,
    terms: list[str],
    answer: str,
    max_lines: int = 5,
) -> str:
    """
    대표 chunk에서 API에 노출할 핵심 근거만 추출한다.

    원칙:
    - 답변 숫자가 포함된 행은 반드시 유지
    - 표 헤더는 최대 2개 유지
    - 관련도가 높은 보조 행만 최소한 추가
    - 동일 chunk 전체를 그대로 노출하지 않음
    """

    document = str(
        document
        or ""
    ).strip()

    if not document:
        return ""

    lines = [
        line.strip()
        for line in (
            document
            .splitlines()
        )
        if line.strip()
    ]

    if not lines:
        return ""

    answer_numbers = (
        extract_answer_numbers(
            answer
        )
    )

    direct_indices = [
        index
        for index, line in enumerate(
            lines
        )
        if (
            answer_numbers
            and text_contains_answer_number(
                line,
                answer_numbers,
            )
        )
    ]

    scored = [
        (
            score_evidence_line(
                line,
                terms=terms,
                answer_numbers=answer_numbers,
            ),
            index,
            line,
        )
        for index, line in enumerate(
            lines
        )
    ]

    scored = [
        item
        for item in scored
        if item[0] > 0
    ]

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    selected: list[
        int
    ] = []

    # ========================================================
    # 1. 직접 답변 행
    # ========================================================

    for index in (
        direct_indices
    ):

        if index not in selected:

            selected.append(
                index
            )

    # ========================================================
    # 2. 표 헤더
    # ========================================================

    if direct_indices:

        # 문서 안에서 답변 행 이전의
        # 가장 가까운 헤더 최대 2개
        for direct_index in (
            direct_indices
        ):

            header_candidates = [
                index
                for index in range(
                    0,
                    direct_index,
                )
                if is_likely_table_header(
                    lines[
                        index
                    ]
                )
            ]

            for index in (
                header_candidates[
                    -2:
                ]
            ):

                if index not in selected:

                    selected.append(
                        index
                    )

    # ========================================================
    # 3. 관련 문장 보충
    # ========================================================

    for _, index, _ in (
        scored
    ):

        if index in selected:
            continue

        if len(
            selected
        ) >= max_lines:
            break

        selected.append(
            index
        )

    # ========================================================
    # 4. 아무 근거도 못 찾으면 앞부분 fallback
    # ========================================================

    if not selected:

        selected = list(
            range(
                min(
                    max_lines,
                    len(
                        lines
                    ),
                )
            )
        )

    # --------------------------------------------------------
    # 직접 답변 행은 max_lines 때문에 삭제하지 않는다.
    # --------------------------------------------------------

    must_keep = set(
        direct_indices
    )

    selected = list(
        dict.fromkeys(
            selected
        )
    )

    if (
        len(
            selected
        )
        > max_lines
    ):

        final_indices = [
            index
            for index in selected
            if index in must_keep
        ]

        for index in selected:

            if index in final_indices:
                continue

            if len(
                final_indices
            ) >= max_lines:
                break

            final_indices.append(
                index
            )

        selected = (
            final_indices
        )

    selected = sorted(
        set(
            selected
        )
    )

    return "\n".join(
        lines[
            index
        ]
        for index in selected
    )


# ============================================================
# API RETRIEVED CONTEXT
# ============================================================

def build_api_retrieved_context(
    result: dict[str, Any],
    max_sources: int = 5,
) -> str:
    """
    평가 API용 retrieved_context.

    흐름:
    1. Answer Generator selected source
    2. 답변 숫자 source가 빠졌으면 1개 보완
    3. 동일 chunk 제거
    4. 동일 공시에서는 대표 chunk 1개 선택
    5. 대표 chunk에서 핵심 행만 추출

    따라서:
    - 근거 숫자가 빠지지 않음
    - 같은 숫자가 여러 표에 반복되어도
      retrieved_context가 과도하게 길어지지 않음
    """

    # --------------------------------------------------------
    # Answer Generator 선택 source
    # --------------------------------------------------------

    results = (
        select_used_results(
            result
        )
    )

    # --------------------------------------------------------
    # 답변 숫자 source가 선택되지 않은 경우에만 보완
    # --------------------------------------------------------

    results = (
        ensure_answer_number_source(
            result,
            results,
        )
    )

    # --------------------------------------------------------
    # 동일 chunk 제거
    # --------------------------------------------------------

    results = (
        deduplicate_chunks(
            results
        )
    )

    if not results:

        return str(
            result.get(
                "retrieved_context"
            )
            or ""
        ).strip()

    # --------------------------------------------------------
    # 같은 공시에서는 대표 chunk 하나만 선택
    # --------------------------------------------------------

    results = (
        select_best_chunk_per_document(
            result,
            results,
        )
    )

    if not results:

        return str(
            result.get(
                "retrieved_context"
            )
            or ""
        ).strip()

    terms = (
        build_evidence_terms(
            result
        )
    )

    answer = str(
        result.get(
            "answer"
        )
        or ""
    )

    blocks: list[
        str
    ] = []

    for rank, item in enumerate(
        results[
            :max_sources
        ],
        start=1,
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        corp_name = (
            item.get(
                "corp_name"
            )
            or metadata.get(
                "corp_name"
            )
            or ""
        )

        report_nm = (
            metadata.get(
                "report_nm"
            )
            or ""
        )

        rcept_no = (
            metadata.get(
                "rcept_no"
            )
            or ""
        )

        document = (
            get_item_document(
                item
            )
        )

        evidence = (
            extract_relevant_evidence(
                document,
                terms=terms,
                answer=answer,
                max_lines=5,
            )
        )

        lines = [
            f"[근거 {rank}]",
        ]

        source_meta: list[
            str
        ] = []

        if corp_name:

            source_meta.append(
                str(
                    corp_name
                )
            )

        if report_nm:

            source_meta.append(
                str(
                    report_nm
                )
            )

        if rcept_no:

            source_meta.append(
                f"접수번호 {rcept_no}"
            )

        if source_meta:

            lines.append(
                " | ".join(
                    source_meta
                )
            )

        if evidence:

            lines.extend(
                [
                    "",
                    evidence,
                ]
            )

        blocks.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# THINK TRACE
# ============================================================

def build_think_trace(
    result: dict[str, Any],
) -> str:
    """
    내부 Chain-of-Thought가 아니라
    평가용 실행 과정 요약을 반환한다.
    """

    extracted = (
        result.get(
            "extracted"
        )
        or {}
    )

    task_type = (
        result.get(
            "task_type"
        )
        or extracted.get(
            "task_type"
        )
        or ""
    )

    companies = (
        extracted.get(
            "target_companies"
        )
        or extracted.get(
            "companies"
        )
        or []
    )

    doc_types = (
        extracted.get(
            "doc_types"
        )
        or []
    )

    metrics = (
        extracted.get(
            "metrics"
        )
        or []
    )

    company_text = (
        ", ".join(
            map(
                str,
                companies,
            )
        )
    )

    doc_text = (
        ", ".join(
            map(
                str,
                doc_types,
            )
        )
    )

    metric_text = (
        ", ".join(
            map(
                str,
                metrics,
            )
        )
    )

    if (
        task_type
        == "검색_정보추출"
    ):

        parts: list[
            str
        ] = []

        if company_text:

            parts.append(
                f"{company_text}의"
            )

        if doc_text:

            parts.append(
                f"{doc_text}를 검색하고,"
            )

        else:

            parts.append(
                "관련 공시를 검색하고,"
            )

        if metric_text:

            parts.append(
                f"{metric_text}이 포함된 "
                "근거를 재정렬한 뒤"
            )

        else:

            parts.append(
                "질의와 관련된 근거를 "
                "재정렬한 뒤"
            )

        parts.append(
            "선택된 공시 근거를 바탕으로 "
            "답변을 생성했습니다."
        )

        return " ".join(
            parts
        )

    return (
        "질의를 분석하고 관련 공시를 검색한 뒤, "
        "검색 결과를 재정렬하여 선택된 공시 근거를 "
        "바탕으로 답변을 생성했습니다."
    )


# ============================================================
# ANSWER API
# ============================================================

@app.get("/answer")
def answer(
    question_id: str = Query(
        ...,
        description="평가 질문 ID",
    ),
    question: str = Query(
        ...,
        description="평가 질의",
    ),
) -> JSONResponse:
    """
    주최 측 평가 API.

    Response:
        question_id
        question
        retrieved_context
        think_trace
        answer
    """

    # ========================================================
    # RUN AGENT
    # ========================================================

    try:

        result = (
            run_agent(
                question
            )
        )

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "question_id": (
                    question_id
                ),
                "question": (
                    question
                ),
                "retrieved_context": "",
                "think_trace": (
                    "질의를 처리하는 과정에서 "
                    "내부 오류가 발생했습니다."
                ),
                "answer": (
                    "답변을 생성할 수 없습니다."
                ),
            },
        )

    # ========================================================
    # FAILURE / CLARIFICATION
    # ========================================================

    if not result.get(
        "success"
    ):

        answer_text = str(
            result.get(
                "answer"
            )
            or (
                "제공된 공시 데이터에서 "
                "확인할 수 없습니다."
            )
        ).strip()

        return JSONResponse(
            status_code=200,
            content={
                "question_id": (
                    question_id
                ),
                "question": (
                    question
                ),
                "retrieved_context": (
                    str(
                        result.get(
                            "retrieved_context"
                        )
                        or ""
                    )
                ),
                "think_trace": (
                    "질의를 분석했으나 "
                    "현재 제공된 정보 또는 공시 데이터만으로 "
                    "답변을 확정할 수 없었습니다."
                ),
                "answer": (
                    answer_text
                ),
            },
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    final_answer = (
        build_user_output(
            result
        )
    )

    retrieved_context = (
        build_api_retrieved_context(
            result
        )
    )

    think_trace = (
        build_think_trace(
            result
        )
    )

    response = {
        "question_id": (
            question_id
        ),
        "question": (
            question
        ),
        "retrieved_context": (
            retrieved_context
        ),
        "think_trace": (
            think_trace
        ),
        "answer": (
            final_answer
        ),
    }

    return JSONResponse(
        status_code=200,
        content=response,
    )
