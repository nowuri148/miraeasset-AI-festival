from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DERIVED_DIR = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
)

PERIODIC_CHUNKS_PATH = (
    DERIVED_DIR
    / "periodic_vector_chunks.jsonl"
)

EVENT_CHUNKS_PATH = (
    DERIVED_DIR
    / "event_vector_chunks.jsonl"
)

OUTPUT_PATH = (
    DERIVED_DIR
    / "vector_chunks.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

# low priority chunk도 현재는 버리지 않는다.
# 최종 Vector DB retrieval 단계에서 우선순위를 활용한다.
INCLUDE_LOW_PRIORITY = True

# 비어 있거나 지나치게 짧은 chunk 제거 기준
MIN_TEXT_LENGTH = 10


# ============================================================
# JSONL
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    records: list[
        dict[str, Any]
    ] = []

    if not path.exists():

        print(
            "[WARN] 파일 없음:",
            path,
        )

        return records

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                print()
                print(
                    "[WARN] JSON decode 실패"
                )

                print(
                    "파일:",
                    path,
                )

                print(
                    "line:",
                    line_number,
                )

                print(
                    "error:",
                    exc,
                )

                continue

            records.append(
                record
            )

    return records


def save_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_chunk(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    chunk_id = str(
        chunk.get(
            "chunk_id"
        )
        or ""
    ).strip()

    text = str(
        chunk.get(
            "text"
        )
        or ""
    ).strip()

    metadata = (
        chunk.get(
            "metadata"
        )
        or {}
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    # --------------------------------------------------------
    # 공통 metadata key 보장
    # --------------------------------------------------------

    normalized_metadata = {
        # document
        "doc_id":
            metadata.get(
                "doc_id"
            ),

        "corp_code":
            metadata.get(
                "corp_code"
            ),

        "corp_name":
            metadata.get(
                "corp_name"
            ),

        "stock_code":
            metadata.get(
                "stock_code"
            ),

        "industry":
            metadata.get(
                "industry"
            ),

        "sector":
            metadata.get(
                "sector"
            ),

        # disclosure
        "doc_group":
            metadata.get(
                "doc_group"
            ),

        "doc_subtype":
            metadata.get(
                "doc_subtype"
            ),

        "report_nm":
            metadata.get(
                "report_nm"
            ),

        "normalized_report_type":
            metadata.get(
                "normalized_report_type"
            ),

        "rcept_no":
            metadata.get(
                "rcept_no"
            ),

        "rcept_dt":
            metadata.get(
                "rcept_dt"
            ),

        "event_date":
            metadata.get(
                "event_date"
            ),

        "base_year":
            metadata.get(
                "base_year"
            ),

        "base_month":
            metadata.get(
                "base_month"
            ),

        # correction
        "is_correction":
            metadata.get(
                "is_correction"
            ),

        "disclosure_chain_id":
            metadata.get(
                "disclosure_chain_id"
            ),

        "corrects_doc_id":
            metadata.get(
                "corrects_doc_id"
            ),

        "corrects_rcept_no":
            metadata.get(
                "corrects_rcept_no"
            ),

        # table
        "raw_file_index":
            metadata.get(
                "raw_file_index"
            ),

        "table_index":
            metadata.get(
                "table_index"
            ),

        "table_title":
            metadata.get(
                "table_title"
            ),

        "row_count":
            metadata.get(
                "row_count"
            ),

        "max_column_count":
            metadata.get(
                "max_column_count"
            ),

        # retrieval
        "chunk_type":
            metadata.get(
                "chunk_type"
            ),

        "search_priority":
            metadata.get(
                "search_priority"
            ),

        "chunk_strategy":
            metadata.get(
                "chunk_strategy"
            ),

        # field-value chunks
        "field_names":
            metadata.get(
                "field_names"
            )
            or [],

        "canonical_fields":
            metadata.get(
                "canonical_fields"
            )
            or [],

        # source
        "raw_file":
            metadata.get(
                "raw_file"
            ),
    }

    return {
        "chunk_id":
            chunk_id,

        "text":
            text,

        "metadata":
            normalized_metadata,
    }


# ============================================================
# VALIDATION
# ============================================================

def is_valid_chunk(
    chunk: dict[str, Any],
) -> tuple[
    bool,
    str,
]:

    chunk_id = str(
        chunk.get(
            "chunk_id"
        )
        or ""
    ).strip()

    text = str(
        chunk.get(
            "text"
        )
        or ""
    ).strip()

    metadata = (
        chunk.get(
            "metadata"
        )
        or {}
    )

    # --------------------------------------------------------
    # chunk id
    # --------------------------------------------------------

    if not chunk_id:

        return (
            False,
            "missing_chunk_id",
        )

    # --------------------------------------------------------
    # text
    # --------------------------------------------------------

    if not text:

        return (
            False,
            "empty_text",
        )

    if len(text) < MIN_TEXT_LENGTH:

        return (
            False,
            "text_too_short",
        )

    # --------------------------------------------------------
    # document identity
    # --------------------------------------------------------

    if not metadata.get(
        "doc_id"
    ):

        return (
            False,
            "missing_doc_id",
        )

    if not metadata.get(
        "corp_name"
    ):

        return (
            False,
            "missing_corp_name",
        )

    if not metadata.get(
        "doc_group"
    ):

        return (
            False,
            "missing_doc_group",
        )

    # --------------------------------------------------------
    # low priority
    # --------------------------------------------------------

    if (
        not INCLUDE_LOW_PRIORITY
        and metadata.get(
            "search_priority"
        )
        == "low"
    ):

        return (
            False,
            "low_priority",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_chunks(
    chunks: list[
        dict[str, Any]
    ],
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:

    seen: set[str] = set()

    unique_chunks: list[
        dict[str, Any]
    ] = []

    duplicate_ids: list[
        str
    ] = []

    for chunk in chunks:

        chunk_id = str(
            chunk.get(
                "chunk_id"
            )
            or ""
        )

        if chunk_id in seen:

            duplicate_ids.append(
                chunk_id
            )

            continue

        seen.add(
            chunk_id
        )

        unique_chunks.append(
            chunk
        )

    return (
        unique_chunks,
        duplicate_ids,
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    chunks: list[
        dict[str, Any]
    ],
    invalid_reasons: Counter,
    duplicate_ids: list[str],
) -> None:

    group_counts = Counter()

    type_counts = Counter()

    priority_counts = Counter()

    strategy_counts = Counter()

    company_counts = Counter()

    correction_count = 0

    document_ids: set[
        str
    ] = set()

    total_text_length = 0

    for chunk in chunks:

        text = str(
            chunk.get(
                "text"
            )
            or ""
        )

        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        doc_group = str(
            metadata.get(
                "doc_group"
            )
            or "unknown"
        )

        chunk_type = str(
            metadata.get(
                "chunk_type"
            )
            or "unknown"
        )

        priority = str(
            metadata.get(
                "search_priority"
            )
            or "unknown"
        )

        strategy = str(
            metadata.get(
                "chunk_strategy"
            )
            or "unknown"
        )

        corp_name = str(
            metadata.get(
                "corp_name"
            )
            or "unknown"
        )

        doc_id = str(
            metadata.get(
                "doc_id"
            )
            or ""
        )

        group_counts[
            doc_group
        ] += 1

        type_counts[
            chunk_type
        ] += 1

        priority_counts[
            priority
        ] += 1

        strategy_counts[
            strategy
        ] += 1

        company_counts[
            corp_name
        ] += 1

        if doc_id:

            document_ids.add(
                doc_id
            )

        if metadata.get(
            "is_correction"
        ):

            correction_count += 1

        total_text_length += len(
            text
        )

    print()
    print("=" * 100)
    print(
        "Final Vector Chunk Statistics"
    )
    print("=" * 100)

    print()
    print(
        "총 chunk:",
        len(chunks),
    )

    print(
        "고유 document:",
        len(document_ids),
    )

    print(
        "정정공시 chunk:",
        correction_count,
    )

    print(
        "중복 chunk_id:",
        len(
            duplicate_ids
        ),
    )

    if chunks:

        average_length = (
            total_text_length
            / len(chunks)
        )

        print(
            "평균 text 길이:",
            f"{average_length:.1f}",
        )

    # --------------------------------------------------------
    # DOC GROUP
    # --------------------------------------------------------

    print()
    print("[DOC GROUP]")

    for key, value in sorted(
        group_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    print()
    print("[STRATEGY]")

    for key, value in sorted(
        strategy_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    print()
    print("[TYPE]")

    for key, value in sorted(
        type_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    print()
    print("[PRIORITY]")

    for key, value in sorted(
        priority_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    print()
    print("[INVALID]")

    if invalid_reasons:

        for key, value in sorted(
            invalid_reasons.items()
        ):

            print(
                f"{key:30s}: "
                f"{value}"
            )

    else:

        print(
            "없음"
        )

    # --------------------------------------------------------
    # TOP COMPANIES
    # --------------------------------------------------------

    print()
    print("[TOP COMPANIES]")

    for corp_name, count in (
        company_counts.most_common(
            20
        )
    ):

        print(
            f"{corp_name:30s}: "
            f"{count}"
        )


# ============================================================
# PREVIEW
# ============================================================

def preview_chunks(
    chunks: list[
        dict[str, Any]
    ],
    limit: int = 5,
) -> None:

    print()
    print("=" * 100)
    print(
        "Final Vector Chunk Preview"
    )
    print("=" * 100)

    for index, chunk in enumerate(
        chunks[:limit],
        start=1,
    ):

        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        print()
        print("-" * 100)

        print(
            f"[CHUNK {index}]"
        )

        print(
            "chunk_id:",
            chunk.get(
                "chunk_id"
            ),
        )

        print(
            "doc_id:",
            metadata.get(
                "doc_id"
            ),
        )

        print(
            "corp_name:",
            metadata.get(
                "corp_name"
            ),
        )

        print(
            "doc_group:",
            metadata.get(
                "doc_group"
            ),
        )

        print(
            "report_nm:",
            metadata.get(
                "report_nm"
            ),
        )

        print(
            "chunk_type:",
            metadata.get(
                "chunk_type"
            ),
        )

        print(
            "search_priority:",
            metadata.get(
                "search_priority"
            ),
        )

        print(
            "chunk_strategy:",
            metadata.get(
                "chunk_strategy"
            ),
        )

        print()
        print(
            "[TEXT]"
        )

        text = str(
            chunk.get(
                "text"
            )
            or ""
        )

        if len(text) > 2000:

            text = (
                text[:2000]
                + "\n..."
            )

        print(
            text
        )


# ============================================================
# BUILD
# ============================================================

def build_final_chunks() -> list[
    dict[str, Any]
]:

    print()
    print("=" * 100)
    print(
        "Vector Chunk Builder"
    )
    print("=" * 100)

    # ========================================================
    # LOAD
    # ========================================================

    periodic_chunks = load_jsonl(
        PERIODIC_CHUNKS_PATH
    )

    event_chunks = load_jsonl(
        EVENT_CHUNKS_PATH
    )

    print()
    print(
        "periodic chunk:",
        len(
            periodic_chunks
        ),
    )

    print(
        "event chunk:",
        len(
            event_chunks
        ),
    )

    raw_chunks = (
        periodic_chunks
        + event_chunks
    )

    print(
        "통합 전 총 chunk:",
        len(
            raw_chunks
        ),
    )

    # ========================================================
    # NORMALIZE / VALIDATE
    # ========================================================

    valid_chunks: list[
        dict[str, Any]
    ] = []

    invalid_reasons: Counter = (
        Counter()
    )

    for raw_chunk in raw_chunks:

        chunk = normalize_chunk(
            raw_chunk
        )

        is_valid, reason = (
            is_valid_chunk(
                chunk
            )
        )

        if not is_valid:

            invalid_reasons[
                reason
            ] += 1

            continue

        valid_chunks.append(
            chunk
        )

    print(
        "validation 통과:",
        len(
            valid_chunks
        ),
    )

    # ========================================================
    # DEDUP
    # ========================================================

    final_chunks, duplicate_ids = (
        deduplicate_chunks(
            valid_chunks
        )
    )

    print(
        "dedup 후:",
        len(
            final_chunks
        ),
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_jsonl(
        final_chunks,
        OUTPUT_PATH,
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    preview_chunks(
        final_chunks,
        limit=5,
    )

    # ========================================================
    # STATS
    # ========================================================

    print_statistics(
        chunks=final_chunks,
        invalid_reasons=(
            invalid_reasons
        ),
        duplicate_ids=(
            duplicate_ids
        ),
    )

    return final_chunks


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    final_chunks = (
        build_final_chunks()
    )

    print()
    print("=" * 100)
    print(
        "최종 Vector DB 입력 파일 생성 완료"
    )
    print("=" * 100)

    print(
        "chunk:",
        len(
            final_chunks
        ),
    )

    print(
        "저장:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()