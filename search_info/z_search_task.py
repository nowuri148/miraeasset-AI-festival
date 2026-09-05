from __future__ import annotations

from typing import Any

from Config import (
    CLOVA_STUDIO_API_KEY,
    CLOVA_BASE_URL,
    CLOVA_MODEL,
)

from .retrieval_adapter import (
    build_retrieval_request,
)

from .vector_retriever import (
    VectorRetriever,
)

from .reranker import (
    HybridReranker,
)

from .answer_generator import (
    HCXAnswerGenerator,
    format_answer_with_sources,
)


# ============================================================
# CONFIG
# ============================================================

RETRIEVAL_TOP_K = 10
AUXILIARY_RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# GLOBAL COMPONENTS
# ============================================================

_vector_retriever: (
    VectorRetriever
    | None
) = None

_reranker: (
    HybridReranker
    | None
) = None

_answer_generator: (
    HCXAnswerGenerator
    | None
) = None


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_search_components() -> None:
    """
    검색_정보추출 경로에서 사용하는 컴포넌트를
    한 번만 초기화한다.

    특히 VectorRetriever는 BGE-M3 모델을 로드하므로
    요청마다 생성하면 안 된다.
    """

    global _vector_retriever
    global _reranker
    global _answer_generator

    if _vector_retriever is None:

        _vector_retriever = (
            VectorRetriever(
                device="cpu"
            )
        )

    if _reranker is None:

        _reranker = (
            HybridReranker()
        )

    if _answer_generator is None:

        _answer_generator = (
            HCXAnswerGenerator(
                api_key=(
                    CLOVA_STUDIO_API_KEY
                ),
                base_url=(
                    CLOVA_BASE_URL
                ),
                model_name=(
                    CLOVA_MODEL
                ),
                debug=False,
            )
        )


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def run_vector_retrieval(
    retrieval_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Retrieval Adapter 결과를 실제 VectorRetriever 호출로 연결한다.
    """

    initialize_search_components()

    assert (
        _vector_retriever
        is not None
    )

    companies = (
        retrieval_request.get(
            "companies"
        )
        or []
    )

    filters = (
        retrieval_request.get(
            "retriever_filters"
        )
        or {}
    )

    query = (
        retrieval_request.get(
            "query"
        )
        or ""
    )

    top_k = int(
        retrieval_request.get(
            "top_k",
            RETRIEVAL_TOP_K,
        )
    )

    if not companies:

        return []

    # --------------------------------------------------------
    # 단일 기업
    # --------------------------------------------------------

    if len(companies) == 1:

        return (
            _vector_retriever.search_company(
                question=query,
                corp_name=companies[0],
                top_k=top_k,

                base_year=filters.get(
                    "base_year"
                ),

                base_month=filters.get(
                    "base_month"
                ),

                doc_group=filters.get(
                    "doc_group"
                ),

                doc_subtype=filters.get(
                    "doc_subtype"
                ),

                search_priority=filters.get(
                    "search_priority"
                ),

                is_correction=filters.get(
                    "is_correction"
                ),

                rcept_no=filters.get(
                    "rcept_no"
                ),

                doc_id=filters.get(
                    "doc_id"
                ),

                debug=False,
            )
        )

    # --------------------------------------------------------
    # 복수 기업
    # --------------------------------------------------------

    return (
        _vector_retriever.search_companies(
            question=query,
            companies=companies,

            top_k_per_company=(
                top_k
            ),

            final_top_k=(
                top_k
            ),

            base_year=filters.get(
                "base_year"
            ),

            base_month=filters.get(
                "base_month"
            ),

            doc_group=filters.get(
                "doc_group"
            ),

            doc_subtype=filters.get(
                "doc_subtype"
            ),

            search_priority=filters.get(
                "search_priority"
            ),

            is_correction=filters.get(
                "is_correction"
            ),

            rcept_no=filters.get(
                "rcept_no"
            ),

            doc_id=filters.get(
                "doc_id"
            ),

            debug=False,
        )
    )


# ============================================================
# AUXILIARY RETRIEVAL
# ============================================================

def build_auxiliary_retrieval_request(
    retrieval_request: dict[str, Any],
) -> dict[str, Any] | None:
    """
    원 질문의 값/수치 검색과 별도로,
    topic_keywords가 실제 공시에서 어떤 사업부문/제품/서비스/
    세부 범위에 대응하는지 설명하는 근거를 찾기 위한
    보조 검색 request를 만든다.

    특정 기업이나 특정 산업을 하드코딩하지 않고
    Keyword Extractor가 추출한 topic_keywords를 사용한다.

    예)
    원 질문:
        삼성전자의 2023년 반도체 설비투자 금액은?

    topic_keywords:
        ["반도체"]

    보조 검색:
        삼성전자 반도체 사업 부문 사업 구성 주요 제품 서비스

    목적:
        "반도체 → DS 부문"과 같은 의미 연결 근거를
        공시 자체에서 찾는다.
    """

    companies = (
        retrieval_request.get(
            "companies"
        )
        or []
    )

    topic_keywords = (
        retrieval_request.get(
            "topic_keywords"
        )
        or []
    )

    if not companies:
        return None

    if not topic_keywords:
        return None

    normalized_companies = [
        str(value).strip()
        for value in companies
        if str(value).strip()
    ]

    normalized_topics = [
        str(value).strip()
        for value in topic_keywords
        if str(value).strip()
    ]

    if not normalized_companies:
        return None

    if not normalized_topics:
        return None

    company_text = " ".join(
        normalized_companies
    )

    topic_text = " ".join(
        normalized_topics
    )

    auxiliary_query = (
        f"{company_text} "
        f"{topic_text} "
        "사업 부문 사업 구성 "
        "주요 제품 서비스"
    ).strip()

    auxiliary_request = dict(
        retrieval_request
    )

    auxiliary_request[
        "query"
    ] = auxiliary_query

    auxiliary_request[
        "top_k"
    ] = min(
        int(
            retrieval_request.get(
                "top_k",
                AUXILIARY_RETRIEVAL_TOP_K,
            )
        ),
        AUXILIARY_RETRIEVAL_TOP_K,
    )

    auxiliary_request[
        "retrieval_mode"
    ] = "auxiliary_scope"

    return auxiliary_request


# ============================================================
# RETRIEVAL RESULT MERGE
# ============================================================

def _get_chunk_key(
    item: dict[str, Any],
    fallback_index: int,
) -> str:
    """
    검색 결과의 중복 제거에 사용할 key를 만든다.

    우선순위:
    1. item.chunk_id
    2. metadata.chunk_id
    3. fallback key
    """

    chunk_id = str(
        item.get(
            "chunk_id"
        )
        or ""
    ).strip()

    if chunk_id:
        return chunk_id

    metadata = (
        item.get(
            "metadata"
        )
        or {}
    )

    chunk_id = str(
        metadata.get(
            "chunk_id"
        )
        or ""
    ).strip()

    if chunk_id:
        return chunk_id

    return (
        f"__no_chunk_id_"
        f"{fallback_index}"
    )


def _safe_distance(
    item: dict[str, Any],
) -> float:
    """
    검색 결과의 distance를 안전하게 float로 변환한다.
    """

    try:
        return float(
            item.get(
                "distance",
                999999.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return 999999.0


def merge_retrieval_results(
    primary_results: list[dict[str, Any]],
    auxiliary_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    원 질문 검색 결과와 보조 의미 검색 결과를 합친다.

    - chunk_id 기준 중복 제거
    - 동일 chunk가 두 검색에 모두 등장하면
      distance가 더 작은 결과를 유지
    - retrieval_origin을 붙여 디버깅 가능하게 함
    """

    merged: dict[
        str,
        dict[str, Any],
    ] = {}

    fallback_index = 0

    # --------------------------------------------------------
    # PRIMARY
    # --------------------------------------------------------

    for item in primary_results:

        fallback_index += 1

        copied = dict(
            item
        )

        copied[
            "retrieval_origin"
        ] = "primary"

        key = _get_chunk_key(
            copied,
            fallback_index,
        )

        merged[key] = copied

    # --------------------------------------------------------
    # AUXILIARY
    # --------------------------------------------------------

    for item in auxiliary_results:

        fallback_index += 1

        copied = dict(
            item
        )

        copied[
            "retrieval_origin"
        ] = "auxiliary"

        key = _get_chunk_key(
            copied,
            fallback_index,
        )

        existing = (
            merged.get(
                key
            )
        )

        if existing is None:

            merged[key] = copied

            continue

        existing_distance = (
            _safe_distance(
                existing
            )
        )

        new_distance = (
            _safe_distance(
                copied
            )
        )

        # 두 검색에서 모두 검색된 chunk라면
        # 더 강하게 검색된 결과를 유지한다.
        if (
            new_distance
            < existing_distance
        ):

            copied[
                "retrieval_origin"
            ] = "primary+auxiliary"

            merged[key] = copied

        else:

            existing[
                "retrieval_origin"
            ] = "primary+auxiliary"

    return list(
        merged.values()
    )


# ============================================================
# SOURCE DEDUP
# ============================================================

def deduplicate_sources(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    같은 공시의 여러 chunk가 사용된 경우
    rcept_no 기준으로 출처를 하나로 합친다.
    """

    unique_sources = []
    seen = set()

    for source in sources:

        key = (
            source.get(
                "rcept_no"
            )
            or source.get(
                "doc_id"
            )
            or source.get(
                "chunk_id"
            )
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_sources.append(
            source
        )

    return unique_sources


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

def build_retrieved_context(
    reranked_results: list[
        dict[str, Any]
    ],
) -> str:
    """
    대회 API의 retrieved_context 등에 사용할 수 있도록
    reranker 최종 결과를 문자열로 구성한다.
    """

    blocks = []

    for rank, item in enumerate(
        reranked_results,
        start=1,
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        document = str(
            item.get(
                "document"
            )
            or ""
        ).strip()

        block = "\n".join(
            [
                f"[근거 {rank}]",
                (
                    "회사: "
                    f"{item.get('corp_name') or metadata.get('corp_name')}"
                ),
                (
                    "보고서: "
                    f"{metadata.get('report_nm')}"
                ),
                (
                    "접수번호: "
                    f"{metadata.get('rcept_no') or ''}"
                ),
                (
                    "chunk_id: "
                    f"{item.get('chunk_id')}"
                ),
                (
                    "retrieval_origin: "
                    f"{item.get('retrieval_origin') or ''}"
                ),
                "",
                document,
            ]
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# SEARCH TASK
# ============================================================

def run_search_task(
    question: str,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    검색_정보추출 전체 실행.

    흐름
    ----
    1. Retrieval Adapter
    2. Primary Vector Retrieval
    3. Auxiliary Scope Retrieval
    4. Retrieval Result Merge
    5. Hybrid Reranker
    6. HCX Answer Generator
    7. 답변 + 출처 반환

    Auxiliary Retrieval의 목적
    --------------------------
    원 질문 검색에서는 수치/표 근거는 잘 검색되지만,
    질문의 세부 대상과 공시의 사업부문 표현 사이의
    의미 관계를 설명하는 chunk가 누락될 수 있다.

    예:
        질문의 "반도체"
        ↕
        공시의 "DS 부문"

    따라서 topic_keywords를 기반으로
    사업/부문/제품 정의 근거를 추가 검색한 뒤
    원 검색 결과와 합쳐 최종 rerank한다.
    """

    initialize_search_components()

    assert (
        _reranker
        is not None
    )

    assert (
        _answer_generator
        is not None
    )

    # ========================================================
    # STEP 1. RETRIEVAL REQUEST
    # ========================================================

    retrieval_request = (
        build_retrieval_request(
            question=question,
            extracted=extracted,
            top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    if not retrieval_request.get(
        "ready"
    ):

        return {
            "success": False,

            "task_type": (
                "검색_정보추출"
            ),

            "status": (
                "retrieval_not_ready"
            ),

            "reason": (
                retrieval_request.get(
                    "reason"
                )
            ),

            "retrieval_request": (
                retrieval_request
            ),
        }

    # ========================================================
    # STEP 2. PRIMARY VECTOR RETRIEVAL
    # ========================================================

    primary_vector_results = (
        run_vector_retrieval(
            retrieval_request
        )
    )

    # ========================================================
    # STEP 3. AUXILIARY SCOPE RETRIEVAL
    # ========================================================

    auxiliary_request = (
        build_auxiliary_retrieval_request(
            retrieval_request
        )
    )

    auxiliary_vector_results: list[
        dict[str, Any]
    ] = []

    if auxiliary_request is not None:

        auxiliary_vector_results = (
            run_vector_retrieval(
                auxiliary_request
            )
        )

    # ========================================================
    # STEP 4. MERGE RETRIEVAL RESULTS
    # ========================================================

    vector_results = (
        merge_retrieval_results(
            primary_results=(
                primary_vector_results
            ),
            auxiliary_results=(
                auxiliary_vector_results
            ),
        )
    )

    if not vector_results:

        return {
            "success": False,

            "task_type": (
                "검색_정보추출"
            ),

            "status": (
                "no_retrieval_result"
            ),

            "answer": (
                "관련 공시 근거를 "
                "찾지 못했습니다."
            ),

            "sources": [],

            "retrieval_request": (
                retrieval_request
            ),

            "auxiliary_retrieval_request": (
                auxiliary_request
            ),
        }

    # ========================================================
    # STEP 5. RERANK
    # ========================================================

    query = (
        retrieval_request.get(
            "query"
        )
        or question
    )

    metrics = (
        retrieval_request.get(
            "metrics"
        )
        or []
    )

    topic_keywords = (
        retrieval_request.get(
            "topic_keywords"
        )
        or []
    )

    reranked_results = (
        _reranker.rerank(
            question=query,

            results=(
                vector_results
            ),

            metrics=(
                metrics
            ),

            topic_keywords=(
                topic_keywords
            ),

            top_k=(
                RERANK_TOP_K
            ),
        )
    )

    if not reranked_results:

        return {
            "success": False,

            "task_type": (
                "검색_정보추출"
            ),

            "status": (
                "no_rerank_result"
            ),

            "answer": (
                "관련 근거를 "
                "선별하지 못했습니다."
            ),

            "sources": [],

            "retrieval_request": (
                retrieval_request
            ),

            "auxiliary_retrieval_request": (
                auxiliary_request
            ),

            "vector_results": (
                vector_results
            ),
        }

    # ========================================================
    # STEP 6. ANSWER GENERATION
    # ========================================================

    answer_result = (
        _answer_generator.generate(
            question=query,

            reranked_results=(
                reranked_results
            ),
        )
    )

    answer = str(
        answer_result.get(
            "answer"
        )
        or ""
    ).strip()

    used_source_ids = (
        answer_result.get(
            "used_source_ids"
        )
        or []
    )

    sources = (
        answer_result.get(
            "sources"
        )
        or []
    )

    unique_sources = (
        deduplicate_sources(
            sources
        )
    )

    # ========================================================
    # STEP 7. CONTEXT
    # ========================================================

    retrieved_context = (
        build_retrieved_context(
            reranked_results
        )
    )

    # ========================================================
    # STEP 8. FINAL DISPLAY TEXT
    # ========================================================

    formatted_answer = (
        format_answer_with_sources(
            {
                **answer_result,

                "sources": (
                    unique_sources
                ),
            }
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "success": True,

        "task_type": (
            "검색_정보추출"
        ),

        "status": (
            "completed"
        ),

        "answer": (
            answer
        ),

        "answer_with_sources": (
            formatted_answer
        ),

        "sources": (
            unique_sources
        ),

        "used_source_ids": (
            used_source_ids
        ),

        "retrieved_context": (
            retrieved_context
        ),

        "retrieval_request": (
            retrieval_request
        ),

        # ----------------------------------------------------
        # 보조 검색 디버깅용
        # ----------------------------------------------------

        "auxiliary_retrieval_request": (
            auxiliary_request
        ),

        "primary_vector_results": (
            primary_vector_results
        ),

        "auxiliary_vector_results": (
            auxiliary_vector_results
        ),

        # ----------------------------------------------------
        # 기존 디버깅 / 평가용
        # ----------------------------------------------------

        "vector_results": (
            vector_results
        ),

        "reranked_results": (
            reranked_results
        ),
    }
