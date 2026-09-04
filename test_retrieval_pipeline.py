from __future__ import annotations

import os
import time
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from question.keyword_extractor_ver2 import (
    HyperClovaXKeywordExtractor,
    CompanyScopeResolver,
    extract_and_resolve,
)

from search_info.retrieval_adapter import (
    build_retrieval_request,
)

from search_info.vector_retriever import (
    VectorRetriever,
)

from search_info.reranker import (
    HybridReranker,
)

from search_info.answer_generator import (
    HCXAnswerGenerator,
    format_answer_with_sources,
)


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_PATH = (
    PROJECT_ROOT
    / ".env"
)

UNIVERSE_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "universe.csv"
)


# =============================================================================
# TEST CONFIG
# =============================================================================

QUESTION = (
    "SK텔레콤 2023년 1분기 "
    "무선통신사업 매출액은?"
)

# VectorRetriever가 가져올 후보 개수
RETRIEVAL_TOP_K = 10

# Reranker가 최종적으로 남길 개수
RERANK_TOP_K = 5


# =============================================================================
# ENV
# =============================================================================

load_dotenv(
    ENV_PATH,
    override=True,
)


# =============================================================================
# HELPER
# =============================================================================

def to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Extractor 반환값을 dict로 변환한다.
    """

    if isinstance(
        value,
        dict,
    ):
        return value

    if hasattr(
        value,
        "model_dump",
    ):
        return (
            value.model_dump()
        )

    if hasattr(
        value,
        "to_dict",
    ):
        return (
            value.to_dict()
        )

    if hasattr(
        value,
        "__dict__",
    ):
        return dict(
            value.__dict__
        )

    raise TypeError(
        "결과를 dict로 변환할 수 없습니다. "
        f"type={type(value)}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    total_start = (
        time.time()
    )

    extractor_elapsed = 0.0
    adapter_elapsed = 0.0
    retriever_load_elapsed = 0.0
    search_elapsed = 0.0
    rerank_elapsed = 0.0
    answer_elapsed = 0.0

    print()
    print("=" * 100)
    print("INTEGRATION TEST")
    print(
        "Keyword Extractor "
        "-> Retrieval Adapter "
        "-> Vector Retriever "
        "-> Reranker "
        "-> HCX Answer Generator"
    )
    print("=" * 100)

    print()
    print("QUESTION:")
    print(QUESTION)

    # =========================================================================
    # STEP 1. KEYWORD EXTRACTOR
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 1. KEYWORD EXTRACTOR"
    )
    print("=" * 100)

    api_key = os.getenv(
        "CLOVA_STUDIO_API_KEY",
        "",
    )

    if not api_key:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY가 "
            ".env에 없습니다."
        )

    from question import (
        keyword_extractor_ver2
        as keyword_module
    )

    base_url = os.getenv(
        "CLOVA_BASE_URL",
        getattr(
            keyword_module,
            "DEFAULT_BASE_URL",
            "https://clovastudio.stream.ntruss.com/v1/openai",
        ),
    )

    model_name = os.getenv(
        "CLOVA_MODEL",
        getattr(
            keyword_module,
            "DEFAULT_MODEL",
            "HCX-005",
        ),
    )

    print(
        "model    :",
        model_name,
    )

    print(
        "endpoint :",
        base_url,
    )

    extractor = (
        HyperClovaXKeywordExtractor(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            debug=False,
        )
    )

    resolver = (
        CompanyScopeResolver(
            UNIVERSE_PATH
        )
    )

    extractor_start = (
        time.time()
    )

    extracted_obj = (
        extract_and_resolve(
            QUESTION,
            extractor,
            resolver,
        )
    )

    extractor_elapsed = (
        time.time()
        - extractor_start
    )

    extracted = (
        to_dict(
            extracted_obj
        )
    )

    print()
    print(
        f"[TIME] extractor: "
        f"{extractor_elapsed:.3f}s"
    )

    print()
    print("[EXTRACTED]")

    pprint(
        extracted,
        sort_dicts=False,
    )

    # =========================================================================
    # COMPLETENESS CHECK
    # =========================================================================

    if not extracted.get(
        "is_complete"
    ):

        print()
        print("=" * 100)
        print(
            "CLARIFICATION REQUIRED"
        )
        print("=" * 100)

        print(
            "missing_fields:",
            extracted.get(
                "missing_fields"
            ),
        )

        print(
            "clarification_question:",
            extracted.get(
                "clarification_question"
            ),
        )

        return

    # =========================================================================
    # STEP 2. RETRIEVAL ADAPTER
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 2. RETRIEVAL ADAPTER"
    )
    print("=" * 100)

    adapter_start = (
        time.time()
    )

    retrieval_request = (
        build_retrieval_request(
            question=QUESTION,
            extracted=extracted,
            top_k=RETRIEVAL_TOP_K,
        )
    )

    adapter_elapsed = (
        time.time()
        - adapter_start
    )

    print()
    print(
        f"[TIME] adapter: "
        f"{adapter_elapsed:.6f}s"
    )

    print()
    print(
        "[RETRIEVAL REQUEST]"
    )

    pprint(
        retrieval_request,
        sort_dicts=False,
    )

    if not retrieval_request.get(
        "ready"
    ):

        print()
        print("=" * 100)
        print(
            "RETRIEVAL NOT READY"
        )
        print("=" * 100)

        pprint(
            retrieval_request,
            sort_dicts=False,
        )

        return

    # =========================================================================
    # STEP 3. VECTOR RETRIEVER
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 3. VECTOR RETRIEVER"
    )
    print("=" * 100)

    retriever_load_start = (
        time.time()
    )

    retriever = (
        VectorRetriever(
            device="cpu"
        )
    )

    retriever_load_elapsed = (
        time.time()
        - retriever_load_start
    )

    print(
        f"[TIME] VectorRetriever load: "
        f"{retriever_load_elapsed:.3f}s"
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

    retrieval_top_k = int(
        retrieval_request.get(
            "top_k",
            RETRIEVAL_TOP_K,
        )
    )

    query = (
        retrieval_request.get(
            "query"
        )
        or QUESTION
    )

    print()
    print("[VECTOR INPUT]")

    print(
        "companies:",
        companies,
    )

    print(
        "query:",
        query,
    )

    print(
        "filters:",
        filters,
    )

    print(
        "top_k:",
        retrieval_top_k,
    )

    if not companies:
        raise RuntimeError(
            "검색 대상 회사가 없습니다."
        )

    search_start = (
        time.time()
    )

    if len(companies) == 1:

        results = (
            retriever.search_company(
                question=query,
                corp_name=companies[0],
                top_k=retrieval_top_k,

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

                debug=True,
            )
        )

    else:

        results = (
            retriever.search_companies(
                question=query,
                companies=companies,

                top_k_per_company=(
                    retrieval_top_k
                ),

                final_top_k=(
                    retrieval_top_k
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

                debug=True,
            )
        )

    search_elapsed = (
        time.time()
        - search_start
    )

    # =========================================================================
    # VECTOR RESULT
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "VECTOR RETRIEVAL RESULT"
    )
    print("=" * 100)

    print(
        f"result count: "
        f"{len(results)}"
    )

    print(
        f"search time : "
        f"{search_elapsed:.3f}s"
    )

    for rank, item in enumerate(
        results,
        start=1,
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        print()
        print("-" * 100)

        print(
            f"[VECTOR TOP {rank}]"
        )

        print(
            "chunk_id :",
            item.get(
                "chunk_id"
            ),
        )

        print(
            "company  :",
            item.get(
                "corp_name"
            ),
        )

        print(
            "distance :",
            item.get(
                "distance"
            ),
        )

        print(
            "report   :",
            metadata.get(
                "report_nm"
            ),
        )

        print(
            "year     :",
            metadata.get(
                "base_year"
            ),
        )

        print(
            "month    :",
            metadata.get(
                "base_month"
            ),
        )

        print(
            "group    :",
            metadata.get(
                "doc_group"
            ),
        )

        print()
        print("[TEXT]")

        document = str(
            item.get(
                "document"
            )
            or ""
        )

        print(
            document[:700]
        )

    if not results:

        print()
        print(
            "FAIL: retrieval result "
            "is empty"
        )

        return

    # =========================================================================
    # STEP 4. RERANKER
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 4. RERANKER"
    )
    print("=" * 100)

    reranker = (
        HybridReranker()
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

    print()
    print("[RERANK INPUT]")

    print(
        "metrics:",
        metrics,
    )

    print(
        "topic_keywords:",
        topic_keywords,
    )

    print(
        "input chunks:",
        len(results),
    )

    print(
        "output top_k:",
        RERANK_TOP_K,
    )

    rerank_start = (
        time.time()
    )

    reranked_results = (
        reranker.rerank(
            question=query,
            results=results,
            metrics=metrics,
            topic_keywords=(
                topic_keywords
            ),
            top_k=RERANK_TOP_K,
        )
    )

    rerank_elapsed = (
        time.time()
        - rerank_start
    )

    print()
    print(
        f"[TIME] reranker: "
        f"{rerank_elapsed:.6f}s"
    )

    # =========================================================================
    # RERANK RESULT
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "RERANKED RESULT"
    )
    print("=" * 100)

    for item in (
        reranked_results
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        print()
        print("-" * 100)

        print(
            f"[RERANK TOP "
            f"{item.get('rerank_rank')}]"
        )

        print(
            "chunk_id        :",
            item.get(
                "chunk_id"
            ),
        )

        print(
            "original_rank   :",
            item.get(
                "original_rank"
            ),
        )

        print(
            "distance        :",
            item.get(
                "distance"
            ),
        )

        print(
            "rerank_score    :",
            item.get(
                "rerank_score"
            ),
        )

        print(
            "vector_score    :",
            item.get(
                "rerank_vector_score"
            ),
        )

        print(
            "keyword_score   :",
            item.get(
                "rerank_keyword_score"
            ),
        )

        print(
            "metric_score    :",
            item.get(
                "rerank_metric_score"
            ),
        )

        print(
            "evidence_bonus  :",
            item.get(
                "rerank_evidence_bonus"
            ),
        )

        print(
            "report          :",
            metadata.get(
                "report_nm"
            ),
        )

        print()
        print("[TEXT]")

        print(
            str(
                item.get(
                    "document"
                )
                or ""
            )[:1000]
        )

    # =========================================================================
    # RANK SUMMARY
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "RANK CHANGE SUMMARY"
    )
    print("=" * 100)

    for item in (
        reranked_results
    ):

        print(
            f"RERANK "
            f"{item.get('rerank_rank')}"
            f" <- VECTOR "
            f"{item.get('original_rank')}"
            f" | "
            f"{item.get('chunk_id')}"
            f" | score="
            f"{item.get('rerank_score')}"
        )

    if not reranked_results:

        print()
        print(
            "FAIL: reranker result "
            "is empty"
        )

        return

    # =========================================================================
    # STEP 5. HCX ANSWER GENERATOR
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 5. HCX ANSWER GENERATOR"
    )
    print("=" * 100)

    answer_generator = (
        HCXAnswerGenerator(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            debug=True,
        )
    )

    answer_start = (
        time.time()
    )

    answer_result = (
        answer_generator.generate(
            question=query,
            reranked_results=(
                reranked_results
            ),
        )
    )

    answer_elapsed = (
        time.time()
        - answer_start
    )

    # =========================================================================
    # RAW ANSWER RESULT
    # =========================================================================

    print()
    print(
        f"[TIME] answer generator: "
        f"{answer_elapsed:.3f}s"
    )

    print()
    print("=" * 100)
    print(
        "ANSWER RESULT"
    )
    print("=" * 100)

    print()
    print("[ANSWER]")

    print(
        answer_result.get(
            "answer"
        )
    )

    print()
    print(
        "[USED SOURCE IDS]"
    )

    print(
        answer_result.get(
            "used_source_ids"
        )
    )

    print()
    print(
        "[STRUCTURED SOURCES]"
    )

    pprint(
        answer_result.get(
            "sources"
        ),
        sort_dicts=False,
    )

    # =========================================================================
    # FINAL DISPLAY
    # =========================================================================

    final_answer = (
        format_answer_with_sources(
            answer_result
        )
    )

    print()
    print("=" * 100)
    print(
        "FINAL ANSWER WITH SOURCES"
    )
    print("=" * 100)

    print()
    print(
        final_answer
    )

    # =========================================================================
    # SOURCE DEDUP PREVIEW
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "SOURCE DEDUP PREVIEW"
    )
    print("=" * 100)

    sources = (
        answer_result.get(
            "sources"
        )
        or []
    )

    unique_sources = []
    seen_keys = set()

    for source in sources:

        dedup_key = (
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

        if dedup_key in seen_keys:
            continue

        seen_keys.add(
            dedup_key
        )

        unique_sources.append(
            source
        )

    print(
        f"HCX selected source chunks: "
        f"{len(sources)}"
    )

    print(
        f"Unique disclosure sources : "
        f"{len(unique_sources)}"
    )

    for source in (
        unique_sources
    ):

        print(
            "-",
            source.get(
                "corp_name"
            ),
            "|",
            source.get(
                "report_nm"
            ),
            "| rcept_no=",
            source.get(
                "rcept_no"
            ),
        )

    # =========================================================================
    # TIME SUMMARY
    # =========================================================================

    total_elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 100)
    print(
        "TIME SUMMARY"
    )
    print("=" * 100)

    print(
        f"keyword extractor : "
        f"{extractor_elapsed:.3f}s"
    )

    print(
        f"retrieval adapter : "
        f"{adapter_elapsed:.6f}s"
    )

    print(
        f"retriever loading : "
        f"{retriever_load_elapsed:.3f}s"
    )

    print(
        f"vector search     : "
        f"{search_elapsed:.3f}s"
    )

    print(
        f"reranker          : "
        f"{rerank_elapsed:.6f}s"
    )

    print(
        f"answer generator  : "
        f"{answer_elapsed:.3f}s"
    )

    print(
        f"total             : "
        f"{total_elapsed:.3f}s"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "VALIDATION"
    )
    print("=" * 100)

    errors: list[str] = []

    # -------------------------------------------------------------------------
    # Retriever result
    # -------------------------------------------------------------------------

    if not results:
        errors.append(
            "Vector retrieval result "
            "is empty"
        )

    # -------------------------------------------------------------------------
    # Reranker result
    # -------------------------------------------------------------------------

    if not reranked_results:
        errors.append(
            "Reranker result "
            "is empty"
        )

    # -------------------------------------------------------------------------
    # Answer
    # -------------------------------------------------------------------------

    answer_text = str(
        answer_result.get(
            "answer"
        )
        or ""
    ).strip()

    if not answer_text:
        errors.append(
            "Final answer is empty"
        )

    # -------------------------------------------------------------------------
    # Source
    # -------------------------------------------------------------------------

    used_source_ids = (
        answer_result.get(
            "used_source_ids"
        )
        or []
    )

    if not used_source_ids:
        errors.append(
            "HCX selected no sources"
        )

    if not sources:
        errors.append(
            "Resolved source list "
            "is empty"
        )

    # -------------------------------------------------------------------------
    # Metadata filter validation
    # -------------------------------------------------------------------------

    expected_year = (
        filters.get(
            "base_year"
        )
    )

    expected_month = (
        filters.get(
            "base_month"
        )
    )

    expected_group = (
        filters.get(
            "doc_group"
        )
    )

    for item in results:

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        if (
            expected_year
            is not None
            and metadata.get(
                "base_year"
            )
            != expected_year
        ):
            errors.append(
                "base_year mismatch: "
                f"{item.get('chunk_id')}"
            )

        if (
            expected_month
            is not None
            and metadata.get(
                "base_month"
            )
            != expected_month
        ):
            errors.append(
                "base_month mismatch: "
                f"{item.get('chunk_id')}"
            )

        if (
            expected_group
            is not None
            and metadata.get(
                "doc_group"
            )
            != expected_group
        ):
            errors.append(
                "doc_group mismatch: "
                f"{item.get('chunk_id')}"
            )

    # -------------------------------------------------------------------------
    # Used source id validation
    # -------------------------------------------------------------------------

    max_source_id = (
        len(
            reranked_results
        )
    )

    for source_id in (
        used_source_ids
    ):

        if not (
            1
            <= int(source_id)
            <= max_source_id
        ):
            errors.append(
                "Invalid source id: "
                f"{source_id}"
            )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    if errors:

        print()
        print(
            "FAIL: integration test "
            "has validation errors"
        )

        for error in errors:

            print(
                "-",
                error,
            )

    else:

        print()
        print(
            "PASS: "
            "extractor -> adapter -> "
            "vector retriever -> "
            "reranker -> "
            "HCX answer generator "
            "completed"
        )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
