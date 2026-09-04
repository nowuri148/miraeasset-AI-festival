from __future__ import annotations

from typing import Any, Callable

from validation.answer_grounding_validator import (
    HCXGroundingValidator,
)

from validation.evidence_builder import (
    build_validation_evidence,
)


# ============================================================
# TYPE
# ============================================================

TaskExecutor = Callable[
    [],
    dict[str, Any],
]


# ============================================================
# GLOBAL VALIDATOR
# ============================================================

_validator: (
    HCXGroundingValidator
    | None
) = None


def get_validator() -> (
    HCXGroundingValidator
):

    global _validator

    if _validator is None:

        _validator = (
            HCXGroundingValidator(
                debug=False,
            )
        )

    return _validator


# ============================================================
# VALIDATED TASK RUNNER
# ============================================================

def run_with_grounding_validation(
    *,
    question: str,
    execute_task: TaskExecutor,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """
    Keyword Extraction 이후의 Task 결과를 검증한다.

    흐름:
    Task 실행
    → 최종 후보 답변
    → Evidence 구성
    → Grounding Validation
    → PASS면 반환
    → RETRY면 동일 Task 재실행

    Keyword Extractor는 이 함수에서 다시 실행하지 않는다.
    """

    validator = (
        get_validator()
    )

    validation_history: list[
        dict[str, Any]
    ] = []

    last_result: dict[
        str,
        Any,
    ] = {}

    # ========================================================
    # RETRY LOOP
    # ========================================================

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        # ----------------------------------------------------
        # 이미 추출된 조건으로 동일 Task 실행
        # ----------------------------------------------------

        result = (
            execute_task()
        )

        last_result = (
            result
        )

        # ----------------------------------------------------
        # Task 자체가 실패
        # ----------------------------------------------------

        if not result.get(
            "success"
        ):

            return result

        # ----------------------------------------------------
        # Task가 만든 최종 후보 답변
        # ----------------------------------------------------

        answer = str(
            result.get(
                "answer"
            )
            or ""
        ).strip()

        task_type = str(
            result.get(
                "task_type"
            )
            or ""
        ).strip()

        # ----------------------------------------------------
        # Task가 실제 사용한 최종 근거
        # ----------------------------------------------------

        evidence = (
            build_validation_evidence(
                result
            )
        )

        # ----------------------------------------------------
        # FINAL GROUNDING VALIDATION
        # ----------------------------------------------------

        validation = (
            validator.validate(
                question=question,
                answer=answer,
                evidence=evidence,
                task_type=task_type,
            )
        )

        validation_history.append(
            {
                "attempt": attempt,
                **validation,
            }
        )

        # ====================================================
        # PASS
        # ====================================================

        if validation.get(
            "is_grounded",
            False,
        ):

            result[
                "grounding_validation"
            ] = {
                "passed": True,
                "attempts": attempt,
                "verdict": "PASS",
                "reason": (
                    validation.get(
                        "reason"
                    )
                ),
            }

            result[
                "validation_history"
            ] = (
                validation_history
            )

            return result

        # ====================================================
        # RETRY
        # ====================================================

        print(
            "[GroundingValidator] "
            f"attempt={attempt} RETRY"
        )

        print(
            validation.get(
                "reason"
            )
        )

    # ========================================================
    # MAX RETRY FAILURE
    # ========================================================

    return {
        "success": False,

        "status": (
            "grounding_validation_failed"
        ),

        "task_type": (
            last_result.get(
                "task_type"
            )
        ),

        "question": (
            question
        ),

        "answer": (
            "제공된 공시 근거만으로 "
            "답변을 확정할 수 없습니다."
        ),

        "sources": [],

        "retrieved_context": (
            last_result.get(
                "retrieved_context"
            )
            or ""
        ),

        "grounding_validation": {
            "passed": False,
            "attempts": (
                max_attempts
            ),
            "verdict": "RETRY",
        },

        "validation_history": (
            validation_history
        ),
    }