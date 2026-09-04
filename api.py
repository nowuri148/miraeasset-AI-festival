from __future__ import annotations

import re
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
        str(value or "").strip(),
    )


def extract_numbers(
    text: str,
) -> set[str]:
    """
    답변에 등장한 숫자를 근거 행 탐색에 활용한다.

    예:
    3,249,918백만원
    → 3249918
    """

    values: set[str] = set()

    for match in re.findall(
        r"\d[\d,\.]*",
        str(text or ""),
    ):

        normalized = (
            match.replace(",", "")
        )

        if normalized:
            values.add(
                normalized
            )

    return values


def normalize_number_text(
    text: str,
) -> str:

    return str(
        text or ""
    ).replace(
        ",",
        "",
    )


# ============================================================
# SEARCH TERMS
# ============================================================

def build_evidence_terms(
    result: dict[str, Any],
) -> list[str]:
    """
    Keyword Extractor가 찾은 metric/topic을 이용해
    근거 문장/행 검색용 키워드를 만든다.
    """

    extracted = (
        result.get("extracted")
        or {}
    )

    terms: list[str] = []

    for key in (
        "metrics",
        "topic_keywords",
    ):

        values = (
            extracted.get(key)
            or []
        )

        for value in values:

            value = normalize_text(
                value
            )

            if (
                value
                and value not in terms
            ):
                terms.append(
                    value
                )

    # 기업명도 보조적으로 사용
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

        company = normalize_text(
            company
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
# LINE SCORING
# ============================================================

def score_evidence_line(
    line: str,
    terms: list[str],
    answer_numbers: set[str],
) -> float:
    """
    질문 키워드와 최종 답변 숫자를 기준으로
    각 문장/표 행의 관련도를 계산한다.
    """

    normalized_line = (
        normalize_text(
            line
        ).lower()
    )

    if not normalized_line:
        return -1.0

    score = 0.0

    # --------------------------------------------------------
    # 질문 관련 키워드
    # --------------------------------------------------------

    for term in terms:

        term_lower = (
            term.lower()
        )

        if term_lower in normalized_line:

            score += 2.0

            # metric처럼 긴 표현일수록 조금 더 가중
            if len(term_lower) >= 4:
                score += 0.5

    # --------------------------------------------------------
    # 최종 답변에 사용된 숫자
    # --------------------------------------------------------

    number_line = (
        normalize_number_text(
            normalized_line
        )
    )

    for number in answer_numbers:

        if (
            len(number) >= 2
            and number in number_line
        ):
            score += 4.0

    # --------------------------------------------------------
    # 표 행
    # --------------------------------------------------------

    if "|" in line:
        score += 0.5

    return score


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def extract_relevant_evidence(
    document: str,
    *,
    terms: list[str],
    answer: str,
    max_lines: int = 4,
) -> str:
    """
    검색된 chunk 전체를 반환하지 않고,
    질문/답변과 직접 관련된 핵심 행 또는 문장만 뽑는다.

    표인 경우:
        헤더 + 핵심 행을 함께 반환

    서술형인 경우:
        관련도가 높은 문장/행 반환
    """

    document = str(
        document
        or ""
    ).strip()

    if not document:
        return ""

    lines = [
        line.strip()
        for line in document.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    answer_numbers = (
        extract_numbers(
            answer
        )
    )

    scored: list[
        tuple[float, int, str]
    ] = []

    for index, line in enumerate(
        lines
    ):

        score = (
            score_evidence_line(
                line,
                terms,
                answer_numbers,
            )
        )

        if score > 0:

            scored.append(
                (
                    score,
                    index,
                    line,
                )
            )

    # --------------------------------------------------------
    # 관련 문장을 못 찾은 경우
    # 너무 긴 원문 대신 앞부분만 fallback
    # --------------------------------------------------------

    if not scored:

        return "\n".join(
            lines[:max_lines]
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    best_score, best_index, best_line = (
        scored[0]
    )

    selected_indices: list[int] = []

    # --------------------------------------------------------
    # TABLE
    # 핵심 행 바로 앞의 헤더 최대 2줄 포함
    # --------------------------------------------------------

    if "|" in best_line:

        for offset in (
            -2,
            -1,
        ):

            index = (
                best_index
                + offset
            )

            if (
                0 <= index < len(lines)
                and "|" in lines[index]
            ):
                selected_indices.append(
                    index
                )

        selected_indices.append(
            best_index
        )

        # 다른 직접 근거 행을 최대 1개 추가
        for _, index, line in scored[1:]:

            if len(
                selected_indices
            ) >= max_lines:
                break

            if (
                index not in selected_indices
                and "|" in line
            ):
                selected_indices.append(
                    index
                )

    # --------------------------------------------------------
    # NARRATIVE
    # --------------------------------------------------------

    else:

        for _, index, _ in scored:

            if index not in selected_indices:
                selected_indices.append(
                    index
                )

            if (
                len(selected_indices)
                >= max_lines
            ):
                break

    selected_indices = sorted(
        set(
            selected_indices
        )
    )

    return "\n".join(
        lines[index]
        for index in selected_indices
    )


# ============================================================
# SELECT ACTUALLY USED SOURCES
# ============================================================

def select_used_results(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    HCX Answer Generator가 사용했다고 선택한
    used_source_ids만 retrieved_context에 사용한다.

    source_id는 rerank 결과의 1-based 순서와 대응한다.
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

    # --------------------------------------------------------
    # used_source_ids가 있으면 그것을 최우선 사용
    # --------------------------------------------------------

    for source_id in used_source_ids:

        try:

            index = (
                int(source_id)
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
            < len(reranked_results)
        ):

            selected.append(
                reranked_results[
                    index
                ]
            )

    # --------------------------------------------------------
    # 없으면 rerank TOP1 fallback
    # --------------------------------------------------------

    if not selected:

        selected = [
            reranked_results[0]
        ]

    return selected


# ============================================================
# DEDUP USED SOURCES
# ============================================================

def deduplicate_used_results(
    results: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    같은 공시에서 여러 chunk가 사용된 경우
    대표 chunk 하나만 남긴다.

    rcept_no → doc_id → chunk_id 순서로 식별.
    """

    unique: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for item in results:

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        key = str(
            metadata.get(
                "rcept_no"
            )
            or metadata.get(
                "doc_id"
            )
            or item.get(
                "chunk_id"
            )
            or ""
        )

        if key and key in seen:
            continue

        if key:
            seen.add(
                key
            )

        unique.append(
            item
        )

    return unique


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

def build_api_retrieved_context(
    result: dict[str, Any],
    max_sources: int = 5,
) -> str:
    """
    평가 API용 검색 근거.

    1. HCX가 실제 선택한 source만 사용
    2. 동일 공시 중복 제거
    3. 질문/답변에 직접 관련된 행·문장만 추출
    """

    used_results = (
        select_used_results(
            result
        )
    )

    used_results = (
        deduplicate_used_results(
            used_results
        )
    )

    if not used_results:

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

    blocks: list[str] = []

    for rank, item in enumerate(
        used_results[
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

        evidence = (
            extract_relevant_evidence(
                str(
                    item.get(
                        "document"
                    )
                    or ""
                ),
                terms=terms,
                answer=answer,
            )
        )

        lines = [
            f"[근거 {rank}]",
        ]

        # ----------------------------------------------------
        # SOURCE METADATA
        # ----------------------------------------------------

        source_meta: list[str] = []

        if corp_name:
            source_meta.append(
                str(corp_name)
            )

        if report_nm:
            source_meta.append(
                str(report_nm)
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

        # ----------------------------------------------------
        # CORE EVIDENCE
        # ----------------------------------------------------

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
    내부 Chain-of-Thought를 반환하는 것이 아니라,
    평가자가 확인할 수 있는 검색/도구 사용 과정의
    간결한 실행 요약을 반환한다.
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

    # --------------------------------------------------------
    # 검색 정보 추출
    # --------------------------------------------------------

    if task_type == (
        "검색_정보추출"
    ):

        parts: list[str] = []

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

    # --------------------------------------------------------
    # 향후 다른 task에서도 사용 가능한 fallback
    # --------------------------------------------------------

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
    주최 측 평가 API 스키마.

    GET /answer
        ?question_id=Q-001
        &question=평가질의

    Response:
        question_id
        question
        retrieved_context
        think_trace
        answer
    """

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

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

    # ========================================================
    # COMPETITION RESPONSE
    # ========================================================

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
