
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any
import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# CONFIG
# ============================================================

from Config import (
    CLOVA_STUDIO_API_KEY,
    CLOVA_BASE_URL,
    CLOVA_MODEL,
)


# ============================================================
# KEYWORD EXTRACTOR
# ============================================================

from question.keyword_extractor_ver2 import (
    HyperClovaXKeywordExtractor,
    CompanyScopeResolver,
    extract_and_resolve,
)


# ============================================================
# TASK ROUTER
# ============================================================

from search_info.z_search_task import (
    run_search_task,
)


# 추후 구현
# from comparison_info.z_comparison_task import (
#     run_comparison_task,
# )

# from complex_info.z_complex_task import (
#     run_complex_task,
# )


# ============================================================
# COMPANY UNIVERSE
# ============================================================

UNIVERSE_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "universe.csv"
)


# ============================================================
# GLOBAL COMPONENTS
# ============================================================

_keyword_extractor: (
    HyperClovaXKeywordExtractor
    | None
) = None

_company_resolver: (
    CompanyScopeResolver
    | None
) = None


# ============================================================
# RESULT CONVERTER
# ============================================================

def result_to_dict(
    result: Any,
) -> dict[str, Any]:
    """
    extract_and_resolve() 반환값을 dict로 변환한다.
    """

    if isinstance(
        result,
        dict,
    ):
        return result

    if is_dataclass(
        result
    ):
        return asdict(
            result
        )

    if hasattr(
        result,
        "model_dump",
    ):
        return (
            result.model_dump()
        )

    if hasattr(
        result,
        "__dict__",
    ):
        return dict(
            result.__dict__
        )

    raise TypeError(
        "지원하지 않는 result 타입입니다: "
        f"{type(result)}"
    )


# ============================================================
# COMPONENT INITIALIZATION
# ============================================================

def initialize_components() -> None:
    """
    Keyword Extractor와 Company Resolver를
    최초 한 번만 초기화한다.
    """

    global _keyword_extractor
    global _company_resolver

    if (
        _keyword_extractor is not None
        and _company_resolver is not None
    ):
        return

    api_key = os.getenv(
        "CLOVA_STUDIO_API_KEY",
        CLOVA_STUDIO_API_KEY,
    )

    if not api_key:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY가 "
            "설정되어 있지 않습니다."
        )

    base_url = os.getenv(
        "CLOVA_BASE_URL",
        CLOVA_BASE_URL,
    )

    model_name = os.getenv(
        "CLOVA_MODEL",
        CLOVA_MODEL,
    )

    _keyword_extractor = (
        HyperClovaXKeywordExtractor(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            debug=False,
        )
    )

    _company_resolver = (
        CompanyScopeResolver(
            UNIVERSE_PATH
        )
    )


# ============================================================
# TASK ROUTING
# ============================================================

def route_task(
    question: str,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    task_type에 따라 각 실행 모듈로 분기한다.
    """

    task_type = (
        extracted.get(
            "task_type"
        )
    )

    # --------------------------------------------------------
    # 검색 정보 추출
    # --------------------------------------------------------

    if task_type == (
        "검색_정보추출"
    ):

        return (
            run_search_task(
                question=question,
                extracted=extracted,
            )
        )

    # --------------------------------------------------------
    # 다중조회 / 비교연산
    # --------------------------------------------------------

    if task_type == (
        "다중조회_비교연산"
    ):

        return {
            "success": False,
            "task_type": task_type,
            "status": (
                "not_implemented"
            ),
            "answer": (
                "다중조회/비교연산 기능은 "
                "아직 구현되지 않았습니다."
            ),
        }

    # --------------------------------------------------------
    # 복합문서추론
    # --------------------------------------------------------

    if task_type == (
        "복합문서추론"
    ):

        return {
            "success": False,
            "task_type": task_type,
            "status": (
                "not_implemented"
            ),
            "answer": (
                "복합문서추론 기능은 "
                "아직 구현되지 않았습니다."
            ),
        }

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return {
        "success": False,
        "task_type": task_type,
        "status": (
            "unsupported_task_type"
        ),
        "answer": (
            "지원하지 않는 질문 유형입니다."
        ),
    }


# ============================================================
# RUN TASK FROM EXTRACTED
# ============================================================

def run_task_from_extracted(
    *,
    question: str,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    이미 Keyword Extraction이 완료된 경우
    Extractor를 다시 호출하지 않고 Task만 실행한다.

    CLI clarification flow에서 사용한다.
    """

    try:

        task_result = (
            route_task(
                question=question,
                extracted=extracted,
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "task_type": (
                extracted.get(
                    "task_type"
                )
            ),
            "status": (
                "task_error"
            ),
            "answer": (
                "답변을 생성하는 중 "
                "오류가 발생했습니다."
            ),
            "error": str(
                exc
            ),
            "sources": [],
        }

    if isinstance(
        task_result,
        dict,
    ):

        # 내부 평가/디버깅용 정보
        task_result.setdefault(
            "question",
            question,
        )

        task_result.setdefault(
            "task_type",
            extracted.get(
                "task_type"
            ),
        )

        task_result.setdefault(
            "extracted",
            extracted,
        )

        return task_result

    return {
        "success": True,
        "question": question,
        "task_type": (
            extracted.get(
                "task_type"
            )
        ),
        "extracted": extracted,
        "result": task_result,
    }


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    question: str,
) -> dict[str, Any]:
    """
    외부에서 단일 질문 하나를 넣어 처리하는 핵심 함수.

    FastAPI에서는 이 함수 하나만 호출하면 된다.

    흐름:
    question
    → Keyword Extractor
    → Scope Resolver
    → Task Router
    → Answer
    """

    initialize_components()

    assert (
        _keyword_extractor
        is not None
    )

    assert (
        _company_resolver
        is not None
    )

    question = str(
        question
        or ""
    ).strip()

    if not question:

        return {
            "success": False,
            "status": (
                "empty_question"
            ),
            "answer": (
                "질문을 입력해주세요."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # Keyword Extraction
    # --------------------------------------------------------

    try:

        result = (
            extract_and_resolve(
                question,
                _keyword_extractor,
                _company_resolver,
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": (
                "extractor_error"
            ),
            "answer": (
                "질문을 분석하는 중 "
                "오류가 발생했습니다."
            ),
            "error": str(
                exc
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # 추가 정보 필요
    # --------------------------------------------------------

    if not result.is_complete:

        return {
            "success": False,
            "status": (
                "clarification_required"
            ),
            "answer": (
                result.clarification_question
                or "추가 정보가 필요합니다."
            ),
            "missing_fields": (
                result.missing_fields
            ),
            "clarification_question": (
                result.clarification_question
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # Result → dict
    # --------------------------------------------------------

    try:

        extracted = (
            result_to_dict(
                result
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": (
                "result_conversion_error"
            ),
            "answer": (
                "질문 분석 결과를 "
                "처리하지 못했습니다."
            ),
            "error": str(
                exc
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # 이미 추출했으므로 바로 Task 실행
    # --------------------------------------------------------

    return (
        run_task_from_extracted(
            question=question,
            extracted=extracted,
        )
    )


# ============================================================
# USER OUTPUT
# ============================================================

def build_user_output(
    task_result: dict[str, Any],
) -> str:
    """
    사용자에게는 최종 답변 + 출처만 보여준다.
    """

    # --------------------------------------------------------
    # Answer Generator가 완성한 문자열
    # --------------------------------------------------------

    answer_with_sources = str(
        task_result.get(
            "answer_with_sources"
        )
        or ""
    ).strip()

    if answer_with_sources:

        # 이전 formatter에서 [1], [2]가 남아 있어도
        # CLI에서는 제거해서 표시한다.
        lines = []

        for line in (
            answer_with_sources
            .splitlines()
        ):

            stripped = (
                line.strip()
            )

            if (
                stripped.startswith("[")
                and "] " in stripped
            ):

                prefix, rest = (
                    stripped.split(
                        "] ",
                        1,
                    )
                )

                number = (
                    prefix[1:]
                )

                if number.isdigit():
                    line = rest

            lines.append(
                line
            )

        return "\n".join(
            lines
        ).strip()

    # --------------------------------------------------------
    # 일반 answer fallback
    # --------------------------------------------------------

    answer = str(
        task_result.get(
            "answer"
        )
        or ""
    ).strip()

    sources = (
        task_result.get(
            "sources"
        )
        or []
    )

    if not sources:

        return (
            answer
            or "답변을 생성하지 못했습니다."
        )

    lines = [
        answer,
        "",
        "출처:",
    ]

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

        parts = []

        corp_name = (
            source.get(
                "corp_name"
            )
        )

        report_nm = (
            source.get(
                "report_nm"
            )
        )

        rcept_no = (
            source.get(
                "rcept_no"
            )
        )

        if corp_name:
            parts.append(
                str(corp_name)
            )

        if report_nm:
            parts.append(
                str(report_nm)
            )

        if rcept_no:
            parts.append(
                f"접수번호 {rcept_no}"
            )

        if parts:

            lines.append(
                ", ".join(
                    parts
                )
            )

    return "\n".join(
        lines
    )


# ============================================================
# CLI MAIN
# ============================================================

def main() -> None:

    initialize_components()

    assert (
        _keyword_extractor
        is not None
    )

    assert (
        _company_resolver
        is not None
    )

    print()
    print("=" * 70)
    print(
        "HyperCLOVA X QA Pipeline"
    )
    print("=" * 70)

    print(
        "종료하려면 "
        "q / quit / exit 입력"
    )

    while True:

        original_question = (
            input(
                "\n질문> "
            )
            .strip()
        )

        # ----------------------------------------------------
        # 종료
        # ----------------------------------------------------

        if (
            original_question.lower()
            in {
                "q",
                "quit",
                "exit",
            }
        ):

            print(
                "종료합니다."
            )

            break

        # ----------------------------------------------------
        # 빈 질문
        # ----------------------------------------------------

        if not original_question:
            continue

        conversation_question = (
            original_question
        )

        result = None

        # ====================================================
        # KEYWORD EXTRACTION + CLARIFICATION
        # ====================================================

        while True:

            try:

                result = (
                    extract_and_resolve(
                        conversation_question,
                        _keyword_extractor,
                        _company_resolver,
                    )
                )

            except Exception as exc:

                print()
                print(
                    "질문 분석 중 오류가 "
                    "발생했습니다."
                )

                print(
                    exc
                )

                result = None
                break

            # ------------------------------------------------
            # Complete
            # ------------------------------------------------

            if result.is_complete:
                break

            # ------------------------------------------------
            # 추가 정보 요청
            # ------------------------------------------------

            print()

            print(
                result.clarification_question
                or "추가 정보가 필요합니다."
            )

            additional = (
                input(
                    "답변> "
                )
                .strip()
            )

            if not additional:

                print(
                    "필요한 정보를 "
                    "입력해주세요."
                )

                continue

            # ------------------------------------------------
            # 취소
            # ------------------------------------------------

            if (
                additional.lower()
                in {
                    "취소",
                    "cancel",
                    "quit",
                    "q",
                }
            ):

                print(
                    "현재 질문을 "
                    "취소합니다."
                )

                result = None
                break

            # ------------------------------------------------
            # 기존 질문 + 추가 정보
            # ------------------------------------------------

            conversation_question = (
                conversation_question
                + "\n"
                + "[사용자 추가 정보] "
                + additional
            )

        # ====================================================
        # 실패 / 취소
        # ====================================================

        if result is None:
            continue

        # ====================================================
        # RESULT → DICT
        # ====================================================

        try:

            extracted = (
                result_to_dict(
                    result
                )
            )

        except Exception as exc:

            print()
            print(
                "질문 분석 결과를 "
                "처리하지 못했습니다."
            )

            print(
                exc
            )

            continue

        # ====================================================
        # IMPORTANT
        #
        # 여기서 run_agent()를 다시 호출하지 않는다.
        # 이미 얻은 extracted를 바로 사용한다.
        # ====================================================

        task_result = (
            run_task_from_extracted(
                question=(
                    conversation_question
                ),
                extracted=extracted,
            )
        )

        # ====================================================
        # USER OUTPUT ONLY
        # ====================================================

        print()

        print(
            build_user_output(
                task_result
            )
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
