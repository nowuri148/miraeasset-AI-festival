from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from periodic_table_parser_split import (
    parse_periodic_tables,
)

from periodic_narrative_chunk_builder import (
    make_narrative_chunks,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "manifest.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
    / "periodic_vector_chunks.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_CORP_NAME = None

# 현재는 한 회사의 periodic 문서 5개로 테스트
MAX_DOCUMENTS = None


# ============================================================
# CLASSIFICATION KEYWORDS
# ============================================================

FINANCIAL_KEYWORDS = (
    "재무상태표",
    "손익계산서",
    "포괄손익",
    "현금흐름표",
    "재무제표",
    "매출액",
    "매출",
    "영업이익",
    "당기순이익",
    "순이익",
    "자산",
    "부채",
    "자본",
    "매출채권",
    "재고자산",
    "유형자산",
    "무형자산",
    "금융자산",
    "금융부채",
    "차입금",
    "영업활동",
    "투자활동",
    "재무활동",
)


BUSINESS_KEYWORDS = (
    "사업의 내용",
    "주요 제품",
    "주요제품",
    "제품",
    "수주",
    "계약",
    "생산능력",
    "생산실적",
    "가동률",
    "판매",
    "시장점유율",
    "연구개발",
    "설비투자",
    "원재료",
)


GOVERNANCE_KEYWORDS = (
    "이사회",
    "감사위원회",
    "감사제도",
    "주주총회",
    "임원",
    "보수",
    "최대주주",
    "주주",
)


COMPANY_KEYWORDS = (
    "회사명",
    "회 사 명",
    "대표이사",
    "본점 소재지",
    "본 점 소 재 지",
    "주소",
    "주 소",
    "전화번호",
    "홈페이지",
    "작성책임자",
    "작 성 책 임 자",
)


LOW_VALUE_TITLES = (
    "목 차",
    "목차",
)


SEPARATOR_PATTERN = re.compile(
    r"-{5,}"
)


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# RAW FILE FINDER
# ============================================================

def find_raw_files(
    record: dict[str, Any],
) -> list[Path]:

    relative_path = str(
        record.get("file_path")
        or ""
    ).strip()

    rcept_no = str(
        record.get("rcept_no")
        or ""
    ).strip()

    # --------------------------------------------------------
    # 1. manifest file_path 우선
    # --------------------------------------------------------

    if relative_path:

        base_path = (
            PROJECT_ROOT
            / "corpus"
            / relative_path
        )

        if base_path.is_file():
            return [base_path]

        if base_path.is_dir():

            files = [
                path
                for path in base_path.rglob("*")
                if path.is_file()
            ]

            if files:
                return sorted(files)

        parent = base_path.parent

        if parent.exists():

            prefix = base_path.name

            files = [
                path
                for path in parent.iterdir()
                if (
                    path.is_file()
                    and path.name.startswith(prefix)
                )
            ]

            if files:
                return sorted(files)

    # --------------------------------------------------------
    # 2. rcept_no fallback
    # --------------------------------------------------------

    if not rcept_no:
        return []

    raw_root = (
        PROJECT_ROOT
        / "corpus"
        / "raw"
    )

    matches: list[Path] = []

    for path in raw_root.rglob("*"):

        if not path.is_file():
            continue

        if rcept_no in path.name:
            matches.append(path)

    return sorted(matches)


# ============================================================
# TABLE BODY
# ============================================================

def table_rows_to_text(
    table: dict[str, Any],
) -> str:

    rows = (
        table.get("rows")
        or []
    )

    lines: list[str] = []

    for row in rows:

        line = " | ".join(
            str(cell or "").strip()
            for cell in row
        )

        if line.strip():
            lines.append(line)

    return "\n".join(lines).strip()


# ============================================================
# TABLE FILTER
# ============================================================

def is_searchable_table(
    table: dict[str, Any],
) -> bool:
    """
    명백하게 검색 가치가 없는 table만 제거한다.

    애매한 table은 삭제하지 않고,
    나중에 search_priority=low로 남긴다.
    """

    title = str(
        table.get("title")
        or ""
    ).strip()

    rows = (
        table.get("rows")
        or []
    )

    # --------------------------------------------------------
    # 너무 작은 table
    # --------------------------------------------------------

    if len(rows) < 2:
        return False

    body = table_rows_to_text(
        table
    )

    if len(body) < 10:
        return False

    # --------------------------------------------------------
    # 목차 제외
    # --------------------------------------------------------

    if title in LOW_VALUE_TITLES:
        return False

    separator_count = len(
        SEPARATOR_PATTERN.findall(
            body
        )
    )

    # 목차처럼
    # --------------------
    # 가 반복되는 경우 제거
    if separator_count >= 3:
        return False

    # --------------------------------------------------------
    # 사실상 내용이 없는 table
    # --------------------------------------------------------

    useful_text = re.sub(
        r"[-|\s]",
        "",
        body,
    )

    if len(useful_text) < 5:
        return False

    return True


# ============================================================
# CHUNK CLASSIFICATION
# ============================================================

def classify_chunk_type(
    table: dict[str, Any],
) -> str:
    """
    검색을 위한 보조 분류.

    이 결과를 정답처럼 사용하는 것이 아니라,
    retrieval/reranking 시 참고 metadata로 사용한다.
    """

    title = str(
        table.get("title")
        or ""
    )

    body = table_rows_to_text(
        table
    )

    combined = (
        title
        + "\n"
        + body
    )

    # --------------------------------------------------------
    # 재무
    # --------------------------------------------------------

    if any(
        keyword in combined
        for keyword in FINANCIAL_KEYWORDS
    ):
        return "financial_table"

    # --------------------------------------------------------
    # 사업
    # --------------------------------------------------------

    if any(
        keyword in combined
        for keyword in BUSINESS_KEYWORDS
    ):
        return "business_table"

    # --------------------------------------------------------
    # 지배구조
    # --------------------------------------------------------

    if any(
        keyword in combined
        for keyword in GOVERNANCE_KEYWORDS
    ):
        return "governance_table"

    # --------------------------------------------------------
    # 회사 기본정보
    # --------------------------------------------------------

    if any(
        keyword in combined
        for keyword in COMPANY_KEYWORDS
    ):
        return "document_admin"

    return "general_table"


# ============================================================
# SEARCH PRIORITY
# ============================================================

def determine_search_priority(
    table: dict[str, Any],
    chunk_type: str,
) -> str:
    """
    검색 우선순위:
        high
        medium
        low

    low라고 해서 삭제하지 않는다.
    """

    title = str(
        table.get("title")
        or ""
    ).strip()

    body = table_rows_to_text(
        table
    )

    # --------------------------------------------------------
    # 핵심 재무 / 사업 정보
    # --------------------------------------------------------

    if chunk_type in (
        "financial_table",
        "business_table",
    ):
        return "high"

    # --------------------------------------------------------
    # 지배구조
    # --------------------------------------------------------

    if chunk_type == "governance_table":
        return "medium"

    # --------------------------------------------------------
    # 회사 기본정보
    # --------------------------------------------------------

    if chunk_type == "document_admin":
        return "low"

    # --------------------------------------------------------
    # 기타 table
    # 숫자가 충분히 있으면 정보성 table일 가능성이 큼
    # --------------------------------------------------------

    number_count = len(
        re.findall(
            r"\d[\d,.]*",
            body,
        )
    )

    if number_count >= 3:
        return "medium"

    # --------------------------------------------------------
    # 지나치게 짧은 일반 table
    # --------------------------------------------------------

    if len(body) < 80:
        return "low"

    if title in LOW_VALUE_TITLES:
        return "low"

    return "medium"


# ============================================================
# CHUNK TEXT
# ============================================================

def build_chunk_text(
    record: dict[str, Any],
    table: dict[str, Any],
) -> str:
    """
    Vector embedding에 사용할 text.

    표의 원래 구조를 가능한 그대로 유지한다.
    """

    lines: list[str] = []

    corp_name = str(
        record.get("corp_name")
        or ""
    )

    report_nm = str(
        record.get("report_nm")
        or ""
    )

    base_year = record.get(
        "base_year"
    )

    base_month = record.get(
        "base_month"
    )

    doc_subtype = str(
        record.get("doc_subtype")
        or ""
    )

    title = str(
        table.get("title")
        or ""
    ).strip()

    # --------------------------------------------------------
    # 검색 context
    # --------------------------------------------------------

    if corp_name:
        lines.append(
            f"회사: {corp_name}"
        )

    if report_nm:
        lines.append(
            f"보고서: {report_nm}"
        )

    if base_year:

        if base_month:

            lines.append(
                f"기준기간: "
                f"{base_year}년 "
                f"{base_month}월"
            )

        else:

            lines.append(
                f"기준연도: "
                f"{base_year}년"
            )

    if doc_subtype:

        lines.append(
            f"보고서유형: "
            f"{doc_subtype}"
        )

    if title:

        lines.append(
            f"표 제목: "
            f"{title}"
        )

    lines.append("")

    # --------------------------------------------------------
    # 실제 table
    # --------------------------------------------------------

    body = table_rows_to_text(
        table
    )

    if body:
        lines.append(body)

    return "\n".join(
        lines
    ).strip()


# ============================================================
# BUILD CHUNK
# ============================================================

def build_chunk(
    record: dict[str, Any],
    table: dict[str, Any],
    raw_file: Path,
    raw_file_index: int,
) -> dict[str, Any]:

    doc_id = str(
        record.get("doc_id")
        or record.get("rcept_no")
        or ""
    )

    table_index = int(
        table.get("table_index")
        or 0
    )

    table_part_index = int(
        table.get("table_part_index")
        or 1
    )

    table_part_count = int(
        table.get("table_part_count")
        or 1
    )

    # --------------------------------------------------------
    # 같은 원본 table이 여러 part로 분할될 수 있으므로
    # part index까지 chunk_id에 포함한다.
    # --------------------------------------------------------

    chunk_id = (
        f"{doc_id}"
        f"_file_{raw_file_index:02d}"
        f"_table_{table_index:04d}"
        f"_part_{table_part_index:03d}"
    )

    text = build_chunk_text(
        record=record,
        table=table,
    )

    chunk_type = classify_chunk_type(
        table
    )

    search_priority = (
        determine_search_priority(
            table=table,
            chunk_type=chunk_type,
        )
    )

    return {
        "chunk_id": chunk_id,

        "text": text,

        "metadata": {

            # =================================================
            # DOCUMENT ID
            # =================================================

            "doc_id": record.get(
                "doc_id"
            ),

            # =================================================
            # COMPANY
            # =================================================

            "corp_code": record.get(
                "corp_code"
            ),

            "corp_name": record.get(
                "corp_name"
            ),

            "stock_code": record.get(
                "stock_code"
            ),

            "industry": record.get(
                "industry"
            ),

            "sector": record.get(
                "sector"
            ),

            # =================================================
            # DOCUMENT
            # =================================================

            "doc_group": record.get(
                "doc_group"
            ),

            "doc_subtype": record.get(
                "doc_subtype"
            ),

            "report_nm": record.get(
                "report_nm"
            ),

            "rcept_no": record.get(
                "rcept_no"
            ),

            "rcept_dt": record.get(
                "rcept_dt"
            ),

            "base_year": record.get(
                "base_year"
            ),

            "base_month": record.get(
                "base_month"
            ),

            "is_correction": record.get(
                "is_correction"
            ),

            # =================================================
            # TABLE
            # =================================================

            "raw_file_index": (
                raw_file_index
            ),

            "table_index": (
                table_index
            ),

            "table_part_index": (
                table_part_index
            ),

            "table_part_count": (
                table_part_count
            ),

            "is_split": table.get(
                "is_split",
                False,
            ),

            "header_row_count": table.get(
                "header_row_count"
            ),

            "source_row_start": table.get(
                "source_row_start"
            ),

            "source_row_end": table.get(
                "source_row_end"
            ),

            "oversized_single_row": table.get(
                "oversized_single_row",
                False,
            ),

            "table_title": table.get(
                "title"
            ),

            "row_count": table.get(
                "row_count"
            ),

            "max_column_count": table.get(
                "max_column_count"
            ),

            # =================================================
            # RETRIEVAL
            # =================================================

            "chunk_type": (
                chunk_type
            ),

            "chunk_strategy": (
                "table_row_split"
                if table_part_count > 1
                else "table"
            ),

            "search_priority": (
                search_priority
            ),

            # =================================================
            # SOURCE
            # =================================================

            "raw_file": str(
                raw_file
            ),
        },
    }


# ============================================================
# NARRATIVE CHUNKS
# ============================================================

def build_narrative_chunks_for_file(
    record: dict[str, Any],
    raw_file: Path,
    raw_file_index: int,
) -> list[dict[str, Any]]:
    """
    periodic_narrative_chunk_builder의 결과를
    최종 periodic_vector_chunks.jsonl 형식으로 맞춘다.

    한 doc_id에 raw file이 여러 개 있을 수 있으므로
    최종 chunk_id에 raw_file_index를 반드시 넣는다.
    """

    source_chunks = make_narrative_chunks(
        raw_file,
        extra_metadata=record,
    )

    chunks: list[dict[str, Any]] = []

    doc_id = str(
        record.get("doc_id")
        or record.get("rcept_no")
        or raw_file.stem
    )

    for narrative_index, source_chunk in enumerate(
        source_chunks,
        start=1,
    ):

        source_metadata = dict(
            source_chunk.get("metadata")
            or {}
        )

        # narrative builder 내부 index가 있으면 그것을 우선 사용
        source_narrative_index = int(
            source_metadata.get(
                "narrative_index"
            )
            or narrative_index
        )

        chunk_id = (
            f"{doc_id}"
            f"_file_{raw_file_index:02d}"
            f"_narrative_{source_narrative_index:04d}"
        )

        metadata = {
            **source_metadata,

            # manifest 값을 source of truth로 다시 덮어쓴다.
            "doc_id": record.get("doc_id"),
            "corp_code": record.get("corp_code"),
            "corp_name": record.get("corp_name"),
            "stock_code": record.get("stock_code"),
            "industry": record.get("industry"),
            "sector": record.get("sector"),

            "doc_group": record.get("doc_group"),
            "doc_subtype": record.get("doc_subtype"),
            "report_nm": record.get("report_nm"),
            "rcept_no": record.get("rcept_no"),
            "rcept_dt": record.get("rcept_dt"),
            "base_year": record.get("base_year"),
            "base_month": record.get("base_month"),
            "is_correction": record.get("is_correction"),

            "raw_file_index": raw_file_index,
            "raw_file": str(raw_file),

            "chunk_type": "narrative",
            "chunk_strategy": (
                source_metadata.get(
                    "chunk_strategy"
                )
                or "section_sentence"
            ),
            "search_priority": (
                source_metadata.get(
                    "search_priority"
                )
                or "medium"
            ),
        }

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": str(
                    source_chunk.get("text")
                    or ""
                ),
                "metadata": metadata,
            }
        )

    return chunks


# ============================================================
# BUILD DOCUMENT CHUNKS
# ============================================================

def build_document_chunks(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    raw_files = find_raw_files(
        record
    )

    chunks: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # 한 문서의 raw file 여러 개 처리
    # --------------------------------------------------------

    for raw_file_index, raw_file in enumerate(
        raw_files,
        start=1,
    ):

        # ====================================================
        # 1. TABLE
        # ====================================================

        tables = parse_periodic_tables(
            path=raw_file,
            corp_name=str(
                record.get("corp_name")
                or ""
            ),
            report_nm=str(
                record.get("report_nm")
                or ""
            ),
        )

        for table in tables:

            # 명백하게 필요 없는 table만 제거
            if not is_searchable_table(
                table
            ):
                continue

            chunk = build_chunk(
                record=record,
                table=table,
                raw_file=raw_file,
                raw_file_index=raw_file_index,
            )

            chunks.append(
                chunk
            )

        # ====================================================
        # 2. NARRATIVE
        # ====================================================

        narrative_chunks = (
            build_narrative_chunks_for_file(
                record=record,
                raw_file=raw_file,
                raw_file_index=raw_file_index,
            )
        )

        chunks.extend(
            narrative_chunks
        )

    return chunks


# ============================================================
# SAVE
# ============================================================

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
# PREVIEW
# ============================================================

def preview_chunks(
    chunks: list[dict[str, Any]],
    limit: int = 5,
) -> None:

    print()
    print("=" * 100)
    print(
        "Periodic Vector Chunk Preview"
    )
    print("=" * 100)

    print(
        "총 chunk:",
        len(chunks),
    )

    for index, chunk in enumerate(
        chunks[:limit],
        start=1,
    ):

        metadata = (
            chunk.get("metadata")
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
            "table_title:",
            metadata.get(
                "table_title"
            ),
        )

        print(
            "base_year:",
            metadata.get(
                "base_year"
            ),
        )

        print(
            "base_month:",
            metadata.get(
                "base_month"
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

        print()
        print(
            "[TEXT]"
        )

        text = str(
            chunk.get("text")
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
# STATISTICS
# ============================================================

def print_chunk_statistics(
    chunks: list[dict[str, Any]],
) -> None:

    type_counts: dict[
        str,
        int,
    ] = {}

    priority_counts: dict[
        str,
        int,
    ] = {}

    strategy_counts: dict[
        str,
        int,
    ] = {}

    # chunk_id 중복 검사용
    chunk_ids: set[str] = set()

    duplicate_chunk_ids: list[str] = []

    for chunk in chunks:

        chunk_id = str(
            chunk.get(
                "chunk_id"
            )
            or ""
        )

        if chunk_id in chunk_ids:

            duplicate_chunk_ids.append(
                chunk_id
            )

        else:

            chunk_ids.add(
                chunk_id
            )

        metadata = (
            chunk.get("metadata")
            or {}
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

        type_counts[
            chunk_type
        ] = (
            type_counts.get(
                chunk_type,
                0,
            )
            + 1
        )

        priority_counts[
            priority
        ] = (
            priority_counts.get(
                priority,
                0,
            )
            + 1
        )

        strategy = str(
            metadata.get(
                "chunk_strategy"
            )
            or "unknown"
        )

        strategy_counts[
            strategy
        ] = (
            strategy_counts.get(
                strategy,
                0,
            )
            + 1
        )

    print()
    print("=" * 100)
    print(
        "Chunk Statistics"
    )
    print("=" * 100)

    print()
    print("[TYPE]")

    for key, value in sorted(
        type_counts.items()
    ):

        print(
            f"{key:25s}: "
            f"{value}"
        )

    print()
    print("[PRIORITY]")

    for key, value in sorted(
        priority_counts.items()
    ):

        print(
            f"{key:25s}: "
            f"{value}"
        )

    print()
    print("[STRATEGY]")

    for key, value in sorted(
        strategy_counts.items()
    ):

        print(
            f"{key:25s}: "
            f"{value}"
        )

    print()
    print("[CHUNK ID]")

    print(
        "unique chunk_id:",
        len(chunk_ids),
    )

    print(
        "duplicate chunk_id:",
        len(
            duplicate_chunk_ids
        ),
    )

    if duplicate_chunk_ids:

        print()
        print(
            "중복 예시:"
        )

        for chunk_id in (
            duplicate_chunk_ids[:10]
        ):

            print(
                " -",
                chunk_id,
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    manifest = load_manifest(
        MANIFEST_PATH
    )

    # --------------------------------------------------------
    # 현재 테스트:
    # HD현대일렉트릭 periodic만
    # --------------------------------------------------------

    # ============================================================
    # PERIODIC DOCUMENT FILTER
    # ============================================================

    target_records = [
        record
        for record in manifest
        if (
            record.get("doc_group")
            == "periodic"
        )
    ]

    # 특정 회사 테스트일 때만 회사 필터 적용
    if TARGET_CORP_NAME:

        target_records = [
            record
            for record in target_records
            if (
                record.get("corp_name")
                == TARGET_CORP_NAME
            )
        ]

    # 시간 순 정렬
    target_records.sort(
        key=lambda record: (
            int(
                record.get("base_year")
                or 0
            ),
            int(
                record.get("base_month")
                or 0
            ),
            str(
                record.get("rcept_no")
                or ""
            ),
        )
    )

    # 테스트 개수 제한이 있을 때만 자르기
    if MAX_DOCUMENTS is not None:

        target_records = (
            target_records[
                :MAX_DOCUMENTS
            ]
        )

    print()
    print("=" * 100)
    print(
        "Periodic Chunk Builder"
    )
    print("=" * 100)

    print(
        "대상 회사:",
        TARGET_CORP_NAME,
    )

    print(
        "대상 문서:",
        len(
            target_records
        ),
    )

    all_chunks: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # DOCUMENT LOOP
    # ========================================================

    for document_index, record in enumerate(
        target_records,
        start=1,
    ):

        print()
        print(
            f"[{document_index}/"
            f"{len(target_records)}]"
        )

        print(
            record.get(
                "report_nm"
            ),
            record.get(
                "rcept_no"
            ),
        )

        chunks = build_document_chunks(
            record
        )

        print(
            "생성 chunk:",
            len(chunks),
        )

        all_chunks.extend(
            chunks
        )

    # ========================================================
    # SAVE
    # ========================================================

    save_jsonl(
        all_chunks,
        OUTPUT_PATH,
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    preview_chunks(
        all_chunks,
        limit=5,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print_chunk_statistics(
        all_chunks
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "저장 완료"
    )
    print("=" * 100)

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()