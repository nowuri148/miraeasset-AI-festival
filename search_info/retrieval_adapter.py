from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 20


def build_retrieval_request(
    question: str,
    extracted: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    HyperCLOVA X Keyword Extractor 결과를
    Vector DB가 사용할 Retrieval Request 형식으로 변환한다.

    역할:
    - 검색 가능 여부 확인
    - target_companies -> metadata filter
    - 기간 정보 -> metadata filter
    - doc_types -> metadata filter
    - metrics / topic_keywords -> semantic search hint
    - 원 질문은 그대로 semantic query로 유지

    반환 예:
    {
        "task_type": "검색_정보추출",
        "query": "현대자동차의 2025년 수출액은?",
        "filters": {
            "corp_name": ["현대자동차"],
            "event_year": 2025
        },
        "metrics": ["수출액"],
        "topic_keywords": ["현대자동차", "2025년", "수출액"],
        "top_k": 20
    }
    """

    # ========================================================
    # 1. 기본 검증
    # ========================================================

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "question must be a non-empty string"
        )

    if not isinstance(extracted, dict):
        raise TypeError(
            "extracted must be a dict"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )


    # ========================================================
    # 2. 검색 가능 여부 확인
    # ========================================================

    is_complete = bool(
        extracted.get("is_complete")
    )

    if not is_complete:
        return {
            "ready": False,
            "task_type": extracted.get(
                "task_type"
            ),
            "query": question.strip(),
            "missing_fields": list(
                extracted.get(
                    "missing_fields"
                )
                or []
            ),
            "clarification_question": (
                extracted.get(
                    "clarification_question"
                )
            ),
        }


    # ========================================================
    # 3. task type 확인
    # ========================================================

    task_type = str(
        extracted.get("task_type")
        or ""
    ).strip()

    if task_type != "검색_정보추출":
        return {
            "ready": False,
            "task_type": task_type,
            "query": question.strip(),
            "reason": (
                "This adapter is for "
                "검색_정보추출 retrieval only."
            ),
        }


    # ========================================================
    # 4. unresolved scope 확인
    # ========================================================

    unresolved_scope_values = list(
        extracted.get(
            "unresolved_scope_values"
        )
        or []
    )

    if unresolved_scope_values:
        return {
            "ready": False,
            "task_type": task_type,
            "query": question.strip(),
            "reason": "unresolved_scope_values",
            "unresolved_scope_values": (
                unresolved_scope_values
            ),
        }


    # ========================================================
    # 5. metadata filter 구성
    # ========================================================

    filters: dict[str, Any] = {}


    # --------------------------------------------------------
    # 5-1. 회사 필터
    # --------------------------------------------------------

    target_companies = [
        str(company).strip()
        for company in (
            extracted.get(
                "target_companies"
            )
            or []
        )
        if str(company).strip()
    ]

    scope_column = str(
        extracted.get("scope_column")
        or "corp_name"
    ).strip()

    if target_companies:
        filters[scope_column] = (
            target_companies
        )


    # --------------------------------------------------------
    # 5-2. 공시 유형 필터
    # --------------------------------------------------------

    doc_types = [
        str(doc_type).strip()
        for doc_type in (
            extracted.get("doc_types")
            or []
        )
        if str(doc_type).strip()
    ]

    if doc_types:
        filters[
            "normalized_report_type"
        ] = doc_types


    # --------------------------------------------------------
    # 5-3. 기간 필터
    # --------------------------------------------------------

    time_filter = build_time_filter(
        extracted
    )

    if time_filter:
        filters.update(
            time_filter
        )


    # ========================================================
    # 6. semantic search 정보
    # ========================================================

    metrics = [
        str(metric).strip()
        for metric in (
            extracted.get("metrics")
            or []
        )
        if str(metric).strip()
    ]

    topic_keywords = [
        str(keyword).strip()
        for keyword in (
            extracted.get(
                "topic_keywords"
            )
            or []
        )
        if str(keyword).strip()
    ]


    # ========================================================
    # 7. 최종 Retrieval Request
    # ========================================================

    return {
        "ready": True,

        "task_type": task_type,

        # Vector embedding 대상은 우선 원 질문 그대로 사용
        "query": question.strip(),

        # hard metadata filter
        "filters": filters,

        # reranker / semantic hint
        "metrics": metrics,

        "topic_keywords": (
            topic_keywords
        ),

        # 원 extractor 정보 일부도 남겨두면
        # 디버깅과 로그에 유용
        "scope_type": extracted.get(
            "scope_type"
        ),

        "time_type": extracted.get(
            "time_type"
        ),

        "top_k": top_k,
    }


def build_time_filter(
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    Extractor의 기간 정보를 Vector DB용 metadata filter로 변환한다.

    Vector DB metadata에 다음 값을 저장한다고 가정:
    - event_year
    - event_quarter
    - event_half

    point:
        2025
        2025 Q2
        2025 H1

    range:
        2023 Q3 ~ 2025 Q4
    """

    time_type = str(
        extracted.get("time_type")
        or ""
    ).strip()

    start_year = _to_int_or_none(
        extracted.get("start_year")
    )

    start_quarter = _to_int_or_none(
        extracted.get("start_quarter")
    )

    start_half = _to_int_or_none(
        extracted.get("start_half")
    )

    end_year = _to_int_or_none(
        extracted.get("end_year")
    )

    end_quarter = _to_int_or_none(
        extracted.get("end_quarter")
    )

    end_half = _to_int_or_none(
        extracted.get("end_half")
    )


    # ========================================================
    # point query
    # ========================================================

    if time_type == "point":

        result: dict[str, Any] = {}

        if start_year is not None:
            result["event_year"] = (
                start_year
            )

        if start_quarter is not None:
            result["event_quarter"] = (
                start_quarter
            )

        if start_half is not None:
            result["event_half"] = (
                start_half
            )

        return result


    # ========================================================
    # range query
    # ========================================================

    if time_type == "range":

        result = {}

        if (
            start_year is not None
            or end_year is not None
        ):
            result["event_year_range"] = {
                "gte": start_year,
                "lte": end_year,
            }

        if (
            start_quarter is not None
            or end_quarter is not None
        ):
            result[
                "event_quarter_range"
            ] = {
                "gte": start_quarter,
                "lte": end_quarter,
            }

        if (
            start_half is not None
            or end_half is not None
        ):
            result[
                "event_half_range"
            ] = {
                "gte": start_half,
                "lte": end_half,
            }

        return result


    return {}


def _to_int_or_none(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None