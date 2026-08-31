from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 20


# =============================================================================
# PUBLIC
# =============================================================================

def build_retrieval_request(
    question: str,
    extracted: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Keyword Extractor 결과를 VectorRetriever 호출용 request로 변환한다.

    현재 VectorRetriever 구조
    -----------------------
    - 회사별 Chroma collection 사용
    - corp_name은 metadata filter가 아니라 collection 선택에 사용
    - Chroma where filter는 사용하지 않음
    - metadata filtering은 VectorRetriever 내부 Python post-filter로 수행

    반환 예
    -------
    {
        "ready": True,
        "task_type": "검색_정보추출",
        "query": "SK텔레콤 2023년 1분기 무선통신사업 매출액은?",
        "companies": ["SK텔레콤"],

        "retriever_filters": {
            "base_year": 2023,
            "base_month": 3,
            "doc_group": "periodic",
            "doc_subtype": None,
            "search_priority": None,
            "is_correction": None,
            "rcept_no": None,
            "doc_id": None
        },

        "metrics": ["매출액"],
        "topic_keywords": ["무선통신사업"],
        "top_k": 20
    }
    """

    # =========================================================================
    # 1. 기본 검증
    # =========================================================================

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

    question = question.strip()

    # =========================================================================
    # 2. 검색 가능 여부 확인
    # =========================================================================

    is_complete = bool(
        extracted.get("is_complete")
    )

    if not is_complete:
        return {
            "ready": False,
            "task_type": extracted.get(
                "task_type"
            ),
            "query": question,
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

    # =========================================================================
    # 3. task_type 확인
    # =========================================================================

    task_type = str(
        extracted.get("task_type")
        or ""
    ).strip()

    if task_type != "검색_정보추출":
        return {
            "ready": False,
            "task_type": task_type,
            "query": question,
            "reason": (
                "This adapter supports "
                "검색_정보추출 only."
            ),
        }

    # =========================================================================
    # 4. unresolved scope 확인
    # =========================================================================

    unresolved_scope_values = [
        str(value).strip()
        for value in (
            extracted.get(
                "unresolved_scope_values"
            )
            or []
        )
        if str(value).strip()
    ]

    if unresolved_scope_values:
        return {
            "ready": False,
            "task_type": task_type,
            "query": question,
            "reason": (
                "unresolved_scope_values"
            ),
            "unresolved_scope_values": (
                unresolved_scope_values
            ),
        }

    # =========================================================================
    # 5. 대상 회사
    # =========================================================================

    companies = _normalize_string_list(
        extracted.get(
            "target_companies"
        )
    )

    if not companies:
        return {
            "ready": False,
            "task_type": task_type,
            "query": question,
            "reason": (
                "target_companies is empty"
            ),
        }

    # =========================================================================
    # 6. 기간 → VectorRetriever metadata 조건
    # =========================================================================

    time_request = build_time_request(
        extracted
    )

    if not time_request["supported"]:
        return {
            "ready": False,
            "task_type": task_type,
            "query": question,
            "reason": (
                time_request["reason"]
            ),
            "time_type": extracted.get(
                "time_type"
            ),
            "time_request": time_request,
        }

    # =========================================================================
    # 7. 문서 유형 → retriever filter
    # =========================================================================

    document_request = (
        build_document_request(
            extracted
        )
    )

    # =========================================================================
    # 8. Retriever filter
    # =========================================================================

    retriever_filters: dict[
        str,
        Any,
    ] = {
        "base_year": (
            time_request.get(
                "base_year"
            )
        ),
        "base_month": (
            time_request.get(
                "base_month"
            )
        ),
        "doc_group": (
            document_request.get(
                "doc_group"
            )
        ),
        "doc_subtype": (
            document_request.get(
                "doc_subtype"
            )
        ),
        "search_priority": None,
        "is_correction": None,
        "rcept_no": None,
        "doc_id": None,
    }

    # =========================================================================
    # 9. Semantic 정보
    # =========================================================================

    metrics = _normalize_string_list(
        extracted.get("metrics")
    )

    topic_keywords = (
        _normalize_string_list(
            extracted.get(
                "topic_keywords"
            )
        )
    )

    # =========================================================================
    # 10. 최종 request
    # =========================================================================

    return {
        "ready": True,

        "task_type": task_type,

        # embedding 대상
        "query": question,

        # 회사별 collection 선택
        "companies": companies,

        # VectorRetriever에 직접 전달
        "retriever_filters": (
            retriever_filters
        ),

        # 추후 reranker / HCX hint
        "metrics": metrics,

        "topic_keywords": (
            topic_keywords
        ),

        # debugging / logging
        "scope_type": extracted.get(
            "scope_type"
        ),

        "scope_column": extracted.get(
            "scope_column"
        ),

        "time_type": extracted.get(
            "time_type"
        ),

        "doc_types": _normalize_string_list(
            extracted.get(
                "doc_types"
            )
        ),

        "top_k": int(top_k),
    }


# =============================================================================
# TIME
# =============================================================================

def build_time_request(
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    Keyword Extractor의 기간 표현을
    현재 VectorRetriever metadata 구조로 변환한다.

    현재 VectorRetriever가 사용하는 주요 기간 metadata
    -----------------------------------------------------
    - base_year
    - base_month

    분기 → 월 변환
    -----------
    Q1 -> 3
    Q2 -> 6
    Q3 -> 9
    Q4 -> 12

    반기 → 월 변환
    -----------
    H1 -> 6
    H2 -> 12

    현재 검색_정보추출 adapter에서는
    point query를 직접 지원한다.

    range query는 여러 기간을 조회해야 하므로
    단일 VectorRetriever 호출로 변환하지 않고
    별도 multi-period 처리 대상으로 넘긴다.
    """

    time_type = str(
        extracted.get("time_type")
        or ""
    ).strip()

    start_year = _to_int_or_none(
        extracted.get("start_year")
    )

    start_quarter = _to_int_or_none(
        extracted.get(
            "start_quarter"
        )
    )

    start_half = _to_int_or_none(
        extracted.get("start_half")
    )

    end_year = _to_int_or_none(
        extracted.get("end_year")
    )

    end_quarter = _to_int_or_none(
        extracted.get(
            "end_quarter"
        )
    )

    end_half = _to_int_or_none(
        extracted.get("end_half")
    )

    # =========================================================================
    # time 없음
    # =========================================================================

    if not time_type:
        return {
            "supported": True,
            "base_year": None,
            "base_month": None,
        }

    # =========================================================================
    # POINT
    # =========================================================================

    if time_type == "point":

        base_year = start_year

        base_month: int | None = None

        # -----------------------------------------------------
        # quarter
        # -----------------------------------------------------

        if start_quarter is not None:

            base_month = (
                _quarter_to_month(
                    start_quarter
                )
            )

        # -----------------------------------------------------
        # half
        # -----------------------------------------------------

        elif start_half is not None:

            base_month = (
                _half_to_month(
                    start_half
                )
            )

        # -----------------------------------------------------
        # 연도만 있으면 month 제한 없음
        # -----------------------------------------------------

        return {
            "supported": True,
            "base_year": base_year,
            "base_month": base_month,
        }

    # =========================================================================
    # RANGE
    # =========================================================================

    if time_type == "range":

        return {
            "supported": False,
            "reason": (
                "range query requires "
                "multi-period retrieval"
            ),

            "start_year": start_year,
            "start_quarter": (
                start_quarter
            ),
            "start_half": start_half,

            "end_year": end_year,
            "end_quarter": (
                end_quarter
            ),
            "end_half": end_half,
        }

    # =========================================================================
    # 알 수 없는 time type
    # =========================================================================

    return {
        "supported": False,
        "reason": (
            f"unsupported time_type: "
            f"{time_type}"
        ),
    }


# =============================================================================
# DOCUMENT TYPE
# =============================================================================

def build_document_request(
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    Keyword Extractor의 doc_types를
    Vector DB metadata의 doc_group/doc_subtype으로 변환한다.

    현재 corpus의 상위 doc_group:
    - periodic
    - major
    - holding
    - exchange

    주의:
    extractor가 어떤 문자열을 실제로 내는지는
    keyword_extractor_ver2.py 정의와 맞춰야 한다.

    알 수 없는 doc_type은 강제 필터하지 않는다.
    잘못된 hard filter로 정답을 제거하는 것보다
    semantic retrieval에 맡기는 편이 안전하다.
    """

    doc_types = _normalize_string_list(
        extracted.get("doc_types")
    )

    if not doc_types:
        return {
            "doc_group": None,
            "doc_subtype": None,
        }

    # -------------------------------------------------------------------------
    # 하나의 doc_group으로 명확히 매핑되는 경우만 hard filter
    # -------------------------------------------------------------------------

    groups: set[str] = set()

    subtypes: list[str] = []

    for raw_type in doc_types:

        value = raw_type.strip()

        normalized = (
            value.lower()
            .replace(" ", "")
        )

        group = (
            _map_doc_type_to_group(
                normalized
            )
        )

        if group is not None:
            groups.add(group)
        else:
            # 현재 Vector DB의 실제 doc_subtype 값과
            # 일치한다고 보장할 수 없으므로
            # 우선 기록만 한다.
            subtypes.append(value)

    # 서로 다른 doc_group이 동시에 지정된 경우
    # 단일 hard filter는 걸지 않는다.
    if len(groups) == 1:
        doc_group = next(
            iter(groups)
        )
    else:
        doc_group = None

    return {
        "doc_group": doc_group,
        "doc_subtype": None,

        # debug 용
        "unmapped_doc_types": (
            subtypes
        ),
    }


def _map_doc_type_to_group(
    value: str,
) -> str | None:
    """
    extractor doc_type 문자열을
    corpus doc_group으로 보수적으로 매핑.
    """

    # =========================================================================
    # periodic
    # =========================================================================

    periodic_values = {
        "periodic",
        "정기공시",
        "사업보고서",
        "반기보고서",
        "분기보고서",
    }

    if value in periodic_values:
        return "periodic"

    # =========================================================================
    # major
    # =========================================================================

    major_values = {
        "major",
        "주요사항보고서",
        "주요사항",
        "주요경영사항",
    }

    if value in major_values:
        return "major"

    # =========================================================================
    # holding
    # =========================================================================

    holding_values = {
        "holding",
        "지분공시",
        "주식등의대량보유상황보고서",
        "대량보유",
    }

    if value in holding_values:
        return "holding"

    # =========================================================================
    # exchange
    # =========================================================================

    exchange_values = {
        "exchange",
        "거래소공시",
        "한국거래소",
        "krx",
    }

    if value in exchange_values:
        return "exchange"

    return None


# =============================================================================
# HELPERS
# =============================================================================

def _quarter_to_month(
    quarter: int,
) -> int:

    mapping = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
    }

    if quarter not in mapping:
        raise ValueError(
            f"Invalid quarter: "
            f"{quarter}"
        )

    return mapping[quarter]


def _half_to_month(
    half: int,
) -> int:

    mapping = {
        1: 6,
        2: 12,
    }

    if half not in mapping:
        raise ValueError(
            f"Invalid half: "
            f"{half}"
        )

    return mapping[half]


def _normalize_string_list(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, str):
        values = [value]

    elif isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        values = list(value)

    else:
        values = [value]

    result: list[str] = []

    for item in values:

        text = str(
            item
        ).strip()

        if text:
            result.append(
                text
            )

    # 중복 제거 / 순서 유지
    return list(
        dict.fromkeys(
            result
        )
    )


def _to_int_or_none(
    value: Any,
) -> int | None:

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        if not value:
            return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# =============================================================================
# LOCAL TEST
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # TEST 1
    # SK텔레콤 2023년 1분기
    # -------------------------------------------------------------------------

    extracted = {
        "is_complete": True,
        "task_type": (
            "검색_정보추출"
        ),

        "target_companies": [
            "SK텔레콤"
        ],

        "scope_type": (
            "company"
        ),

        "scope_column": (
            "corp_name"
        ),

        "unresolved_scope_values": [],

        "time_type": (
            "point"
        ),

        "start_year": 2023,
        "start_quarter": 1,

        "end_year": None,
        "end_quarter": None,

        "start_half": None,
        "end_half": None,

        "doc_types": [
            "분기보고서"
        ],

        "metrics": [
            "매출액"
        ],

        "topic_keywords": [
            "무선통신사업"
        ],
    }

    request = (
        build_retrieval_request(
            question=(
                "SK텔레콤 2023년 "
                "1분기 무선통신사업 "
                "매출액은?"
            ),
            extracted=extracted,
            top_k=10,
        )
    )

    from pprint import pprint

    pprint(
        request,
        sort_dicts=False,
    )


if __name__ == "__main__":
    main()