from __future__ import annotations

from typing import Any, Callable

from .answer_grounding_validator import (
    HCXGroundingValidator,
)

from .evidence_builder import (
    build_validation_evidence,
)


# ============================================================
# TYPE
# ============================================================

TaskExecutor = Callable[
    [
        str,
        str | None,
    ],
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
# VERIFIED RUNNER
# ============================================================

def run_with_grounding_validation(
    *,
    question: str,
    execute_once: TaskExecutor,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """
    동일 질문을 최대 max_attempts회 실행한다.

    1. Task 실행
    2. 최종 답변 생성
    3. 실제 근거와 답변 비교
    4. PASS → 바로 return
    5. RETRY → 동일 질문 재실행
    6. 최대 횟수 실패 → 확인 불가 반환

    retry_feedback은 사용자 질문을 변경하지 않고
    내부 실행에만 전달한다.
    """

    validator = (
        get_validator()
    )

    retry_feedback: (
        str
        | None
    ) = None

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
        # Task 실행
        # ----------------------------------------------------

        result = (
            execute_once(
                question,
                retry_feedback,
            )
        )

        last_result = result

        # ----------------------------------------------------
        # Task 자체가 실패했다면
        # validator까지 갈 필요 없음
        # ----------------------------------------------------

        if not result.get(
            "success"
        ):

            return result

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
        )

        evidence = (
            build_validation_evidence(
                result
            )
        )

        # ----------------------------------------------------
        # Grounding Validation
        # ----------------------------------------------------

        validation = (
            validator.validate(
                question=question,
                answer=answer,
                evidence=evidence,
                task_type=task_type,
            )
        )

        validation_record = {
            "attempt": (
                attempt
            ),
            **validation,
        }

        validation_history.append(
            validation_record
        )

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        if validation.get(
            "is_grounded"
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
                "confidence": (
                    validation.get(
                        "confidence"
                    )
                ),
            }

            # 개발/평가 내부 확인용
            result[
                "validation_history"
            ] = (
                validation_history
            )

            return result

        # ----------------------------------------------------
        # RETRY
        # ----------------------------------------------------

        retry_feedback = str(
            validation.get(
                "retry_feedback"
            )
            or validation.get(
                "reason"
            )
            or (
                "답변과 검색 근거가 일치하지 않습니다. "
                "근거를 다시 확인하십시오."
            )
        )

    # ========================================================
    # MAX RETRY FAILURE
    # ========================================================

    return {
        "success": False,
        "task_type": (
            last_result.get(
                "task_type"
            )
        ),
        "status": (
            "grounding_validation_failed"
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