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
# RETRIEVAL
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
    2. Vector Retriever
    3. Hybrid Reranker
    4. HCX Answer Generator
    5. 답변 + 출처 반환
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
    # STEP 2. VECTOR RETRIEVAL
    # ========================================================

    vector_results = (
        run_vector_retrieval(
            retrieval_request
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
        }

    # ========================================================
    # STEP 3. RERANK
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
            results=vector_results,
            metrics=metrics,
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
        }

    # ========================================================
    # STEP 4. ANSWER GENERATION
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
    # STEP 5. CONTEXT
    # ========================================================

    retrieved_context = (
        build_retrieved_context(
            reranked_results
        )
    )

    # ========================================================
    # STEP 6. FINAL DISPLAY TEXT
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

        "answer": answer,

        "answer_with_sources": (
            formatted_answer
        ),

        "sources": (
            unique_sources
        ),

        "used_source_ids": (
            answer_result.get(
                "used_source_ids"
            )
            or []
        ),

        "retrieved_context": (
            retrieved_context
        ),

        "retrieval_request": (
            retrieval_request
        ),

        # 디버깅 / 평가용
        "vector_results": (
            vector_results
        ),

        "reranked_results": (
            reranked_results
        ),
    }