from __future__ import annotations

from typing import Any

from .retrieval_adapter import (
    build_retrieval_request,
)

def run_search_task(
    question: str,
    extracted: dict[str, Any],
) -> Any:

    retrieval_request = (
        build_retrieval_request(
            question=question,
            extracted=extracted,
            top_k=20,
        )
    )

    if not retrieval_request.get(
        "ready"
    ):
        return {
            "success": False,
            "reason": (
                retrieval_request.get(
                    "reason"
                )
            ),
        }

    # 아직 Vector DB 미연결
    return {
        "success": True,
        "stage": "retrieval_request",
        "retrieval_request": (
            retrieval_request
        ),
    }