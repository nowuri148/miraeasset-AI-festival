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

from Config import CLOVA_STUDIO_API_KEY


DEFAULT_BASE_URL = (
    "https://clovastudio.stream.ntruss.com/v1/openai"
)

DEFAULT_MODEL = "HCX-005"


# ============================================================
# KEYWORD EXTRACTOR
# ============================================================

from question.keyword_extractor_ver2 import (
    HyperClovaXKeywordExtractor,
    CompanyScopeResolver,
    extract_and_resolve,
    print_result,
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
# COMPANY UNIVERSE PATH
# ============================================================

UNIVERSE_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "universe.csv"
)

# 실제 universe 파일명이 다르면 위 경로만 수정하면 됨.


# ============================================================
# RESULT CONVERTER
# ============================================================

def result_to_dict(
    result: Any,
) -> dict[str, Any]:
    """
    extract_and_resolve()의 반환값을
    task 모듈에서 사용 가능한 dict로 변환한다.

    지원:
    - dict
    - dataclass
    - pydantic model
    - 일반 객체(__dict__)
    """

    if isinstance(result, dict):
        return result

    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "__dict__"):
        return dict(
            result.__dict__
        )

    raise TypeError(
        "지원하지 않는 result 타입입니다: "
        f"{type(result)}"
    )


# ============================================================
# TASK ROUTING
# ============================================================

def route_task(
    question: str,
    extracted: dict[str, Any],
) -> Any:
    """
    Keyword Extractor 결과의 task_type에 따라
    각 task 파일로 분기한다.
    """

    task_type = extracted.get(
        "task_type"
    )

    # --------------------------------------------------------
    # 1. 정보 검색
    # --------------------------------------------------------

    if task_type == "검색_정보추출":

        print()
        print("=" * 70)
        print("[Task Router] 검색_정보추출")
        print("=" * 70)

        return run_search_task(
            question=question,
            extracted=extracted,
        )


    # --------------------------------------------------------
    # 2. 다중조회 / 비교연산
    # --------------------------------------------------------

    if task_type == "다중조회_비교연산":

        print()
        print("=" * 70)
        print("[Task Router] 다중조회_비교연산")
        print("=" * 70)

        # 추후 구현 시 아래로 교체
        #
        # return run_comparison_task(
        #     question=question,
        #     extracted=extracted,
        # )

        return {
            "success": False,
            "task_type": task_type,
            "status": "not_implemented",
        }


    # --------------------------------------------------------
    # 3. 복합문서추론
    # --------------------------------------------------------

    if task_type == "복합문서추론":

        print()
        print("=" * 70)
        print("[Task Router] 복합문서추론")
        print("=" * 70)

        # 추후 구현 시 아래로 교체
        #
        # return run_complex_task(
        #     question=question,
        #     extracted=extracted,
        # )

        return {
            "success": False,
            "task_type": task_type,
            "status": "not_implemented",
        }


    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return {
        "success": False,
        "task_type": task_type,
        "status": "unsupported_task_type",
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # 1. HyperCLOVA X API KEY
    # ========================================================

    key = os.getenv(
        "CLOVA_STUDIO_API_KEY",
        CLOVA_STUDIO_API_KEY,
    )

    if not key:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY가 "
            "설정되어 있지 않습니다."
        )


    # ========================================================
    # 2. HyperCLOVA X ENDPOINT
    # ========================================================

    base_url = os.getenv(
        "HYPERCLOVA_X_ENDPOINT",
        DEFAULT_BASE_URL,
    )


    # ========================================================
    # 3. Keyword Extractor 초기화
    # ========================================================

    client = HyperClovaXKeywordExtractor(
        api_key=key,
        base_url=base_url,
        model_name=DEFAULT_MODEL,
        debug=False,
    )


    # ========================================================
    # 4. Company Scope Resolver 초기화
    # ========================================================

    resolver = CompanyScopeResolver(
        UNIVERSE_PATH
    )

    resolver.describe_columns()


    # ========================================================
    # 5. START
    # ========================================================

    print()
    print("=" * 70)
    print("HyperCLOVA X QA Pipeline")
    print("=" * 70)

    print(
        "종료하려면 "
        "q / quit / exit 입력"
    )


    # ========================================================
    # 6. 전체 질의 Loop
    # ========================================================

    while True:

        original_question = input(
            "\n질문> "
        ).strip()


        # ----------------------------------------------------
        # 종료
        # ----------------------------------------------------

        if original_question.lower() in {
            "q",
            "quit",
            "exit",
        }:
            print("종료합니다.")
            break


        # ----------------------------------------------------
        # 빈 입력
        # ----------------------------------------------------

        if not original_question:
            continue


        # ----------------------------------------------------
        # 추가질문을 누적하기 위한 질문
        # ----------------------------------------------------

        conversation_question = (
            original_question
        )

        result = None


        # ====================================================
        # 7. Keyword Extraction + Scope Resolution Loop
        # ====================================================

        while True:

            try:

                result = extract_and_resolve(
                    conversation_question,
                    client,
                    resolver,
                )

            except Exception as exc:

                print()
                print(
                    "추출/범위 해석 실패:"
                )

                print(
                    exc
                )

                result = None
                break


            # =================================================
            # 7-1. 질문 완성
            # =================================================

            if result.is_complete:

                print()
                print_result(
                    result
                )

                break


            # =================================================
            # 7-2. 추가 정보 필요
            # =================================================

            print()
            print("=" * 70)
            print("추가 정보 필요")
            print("=" * 70)

            print(
                "누락 슬롯:",
                result.missing_fields,
            )

            print(
                "확인 질문:",
                result.clarification_question,
            )


            additional = input(
                "답변> "
            ).strip()


            # -------------------------------------------------
            # 빈 답변
            # -------------------------------------------------

            if not additional:

                print(
                    "필요한 정보를 "
                    "입력해주세요."
                )

                continue


            # -------------------------------------------------
            # 현재 질문 취소
            # -------------------------------------------------

            if additional.lower() in {
                "취소",
                "cancel",
                "quit",
                "q",
            }:

                print(
                    "현재 질문을 "
                    "취소합니다."
                )

                result = None
                break


            # -------------------------------------------------
            # 추가 정보를 기존 질문에 누적
            # -------------------------------------------------

            conversation_question = (
                conversation_question
                + "\n[사용자 추가 정보] "
                + additional
            )


        # ====================================================
        # 8. Extraction 실패 / 취소
        # ====================================================

        if result is None:
            continue


        # ====================================================
        # 9. 아직 complete가 아닌 경우
        # ====================================================

        if not result.is_complete:
            continue


        # ====================================================
        # 10. Result -> dict
        # ====================================================

        try:

            extracted = result_to_dict(
                result
            )

        except Exception as exc:

            print()
            print(
                "Extractor 결과 변환 실패:"
            )

            print(
                exc
            )

            continue


        # ====================================================
        # 11. Task Routing
        # ====================================================

        try:

            task_result = route_task(
                question=conversation_question,
                extracted=extracted,
            )

        except Exception as exc:

            print()
            print(
                "Task 실행 실패:"
            )

            print(
                exc
            )

            continue


        # ====================================================
        # 12. 결과 출력
        # ====================================================

        print()
        print("=" * 70)
        print("Task 결과")
        print("=" * 70)

        print(
            task_result
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()