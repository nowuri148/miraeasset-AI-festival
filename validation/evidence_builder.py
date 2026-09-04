from __future__ import annotations

from typing import Any


# ============================================================
# EVIDENCE BUILDER
# ============================================================

def build_validation_evidence(
    result: dict[str, Any],
    *,
    max_sources: int = 5,
    max_chars_per_source: int = 8000,
) -> str:
    """
    최종 답변 검증에 사용할 실제 근거를 구성한다.

    우선순위:
    1. validation_context
    2. HCX가 실제 선택한 used_source_ids
    3. rerank TOP-N
    4. retrieved_context fallback
    """

    # ========================================================
    # 이미 Task가 validation_context를 제공하는 경우
    # ========================================================

    validation_context = str(
        result.get(
            "validation_context"
        )
        or ""
    ).strip()

    if validation_context:

        return (
            validation_context
        )

    # ========================================================
    # RERANK RESULTS
    # ========================================================

    reranked_results = (
        result.get(
            "reranked_results"
        )
        or []
    )

    if not reranked_results:

        return str(
            result.get(
                "retrieved_context"
            )
            or ""
        ).strip()

    used_source_ids = (
        result.get(
            "used_source_ids"
        )
        or []
    )

    selected: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # ANSWER GENERATOR가 사용한 source 우선
    # ========================================================

    for source_id in used_source_ids:

        try:

            index = (
                int(source_id)
                - 1
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            0 <= index
            < len(reranked_results)
        ):

            selected.append(
                reranked_results[index]
            )

    # ========================================================
    # source ID가 없다면 TOP-N fallback
    # ========================================================

    if not selected:

        selected = (
            reranked_results[
                :max_sources
            ]
        )

    # ========================================================
    # 완전히 동일한 chunk 중복 제거
    #
    # 주의:
    # validation에서는 같은 공시라고 해서
    # rcept_no 기준으로 모두 제거하면 안 된다.
    # 같은 공시의 서로 다른 표가 각각 중요한 근거일 수 있다.
    # ========================================================

    unique: list[
        dict[str, Any]
    ] = []

    seen_chunk_ids: set[str] = set()

    for item in selected:

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        chunk_id = str(
            item.get(
                "chunk_id"
            )
            or metadata.get(
                "chunk_id"
            )
            or ""
        )

        if (
            chunk_id
            and chunk_id in seen_chunk_ids
        ):
            continue

        if chunk_id:

            seen_chunk_ids.add(
                chunk_id
            )

        unique.append(
            item
        )

    # ========================================================
    # CONTEXT BUILD
    # ========================================================

    blocks: list[str] = []

    for rank, item in enumerate(
        unique[
            :max_sources
        ],
        start=1,
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        corp_name = (
            item.get(
                "corp_name"
            )
            or metadata.get(
                "corp_name"
            )
            or ""
        )

        report_nm = (
            metadata.get(
                "report_nm"
            )
            or ""
        )

        rcept_no = (
            metadata.get(
                "rcept_no"
            )
            or ""
        )

        document = str(
            item.get(
                "document"
            )
            or ""
        ).strip()

        if (
            len(document)
            > max_chars_per_source
        ):

            document = (
                document[
                    :max_chars_per_source
                ]
                + "\n...[TRUNCATED]"
            )

        lines = [
            f"[SOURCE {rank}]",
        ]

        if corp_name:

            lines.append(
                f"회사: {corp_name}"
            )

        if report_nm:

            lines.append(
                f"공시: {report_nm}"
            )

        if rcept_no:

            lines.append(
                f"접수번호: {rcept_no}"
            )

        lines.extend(
            [
                "",
                document,
            ]
        )

        blocks.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        blocks
    )
