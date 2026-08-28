from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import warnings

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "manifest.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
    / "event_vector_chunks.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_CORP_NAME = None

# 테스트 단계
# 전체 처리 시 None
MAX_DOCUMENTS = None

# field-value chunk 하나에 넣을 최대 field 수
MAX_FIELDS_PER_CHUNK = 12

# 긴 서술형 value는 단독 chunk
LONG_VALUE_THRESHOLD = 700


# ============================================================
# XML WARNING
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


# ============================================================
# CELL TAGS
# ============================================================

CELL_TAGS = (
    "th",
    "td",
    "te",
)


# ============================================================
# TEXT UTILS
# ============================================================

def clean_text(
    text: Any,
) -> str:

    text = str(
        text or ""
    )

    text = text.replace(
        "\xa0",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text,
    )

    return text.strip()


def clean_field_name(
    text: Any,
) -> str:

    text = clean_text(
        text
    )

    # 앞쪽 bullet 제거
    text = re.sub(
        r"^[-ㆍ·]\s*",
        "",
        text,
    )

    # "1. 계약내역" 제거
    # 단 06.22 같은 날짜는 보존
    text = re.sub(
        r"^\d+\.\s+",
        "",
        text,
    )

    return text.strip()


def canonical_field_key(
    text: Any,
) -> str:

    text = clean_field_name(
        text
    )

    text = text.replace(
        "ㆍ",
        "·",
    )

    text = text.replace(
        "・",
        "·",
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    return text


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:

    records: list[
        dict[str, Any]
    ] = []

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
                for path
                in base_path.rglob("*")
                if path.is_file()
            ]

            if files:
                return sorted(files)

        parent = (
            base_path.parent
        )

        if parent.exists():

            prefix = (
                base_path.name
            )

            files = [
                path
                for path
                in parent.iterdir()
                if (
                    path.is_file()
                    and path.name.startswith(
                        prefix
                    )
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
            matches.append(
                path
            )

    return sorted(
        matches
    )


# ============================================================
# DOCUMENT READER
# ============================================================

def read_document(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


# ============================================================
# TABLE TITLE
# ============================================================

def find_table_title(
    table: Any,
) -> str:

    previous_nodes = (
        table.find_all_previous(
            [
                "p",
                "div",
                "span",
                "title",
            ],
            limit=15,
        )
    )

    for node in previous_nodes:

        text = clean_text(
            node.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        if len(text) > 150:
            continue

        if re.fullmatch(
            r"[\d\W_]+",
            text,
        ):
            continue

        return text

    return ""


# ============================================================
# TABLE GRID RESTORATION
# ============================================================

def extract_table_matrix(
    table: Any,
) -> list[list[str]]:

    rows = table.find_all(
        "tr"
    )

    if not rows:
        return []

    rowspan_state: dict[
        int,
        dict[str, Any],
    ] = {}

    matrix: list[
        list[str]
    ] = []

    for row in rows:

        cells = row.find_all(
            CELL_TAGS,
            recursive=False,
        )

        if not cells:
            continue

        current_row: list[str] = []

        col_index = 0

        def fill_rowspans() -> None:

            nonlocal col_index

            while (
                col_index
                in rowspan_state
            ):

                state = (
                    rowspan_state[
                        col_index
                    ]
                )

                current_row.append(
                    state["text"]
                )

                state[
                    "remaining"
                ] -= 1

                if (
                    state[
                        "remaining"
                    ]
                    <= 0
                ):
                    del rowspan_state[
                        col_index
                    ]

                col_index += 1

        fill_rowspans()

        for cell in cells:

            fill_rowspans()

            text = clean_text(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )

            try:
                rowspan = int(
                    cell.get(
                        "rowspan",
                        1,
                    )
                    or 1
                )

            except (
                TypeError,
                ValueError,
            ):
                rowspan = 1

            try:
                colspan = int(
                    cell.get(
                        "colspan",
                        1,
                    )
                    or 1
                )

            except (
                TypeError,
                ValueError,
            ):
                colspan = 1

            for offset in range(
                colspan
            ):

                target_col = (
                    col_index
                    + offset
                )

                current_row.append(
                    text
                )

                if rowspan > 1:

                    rowspan_state[
                        target_col
                    ] = {
                        "remaining":
                            rowspan - 1,

                        "text":
                            text,
                    }

            col_index += colspan

        fill_rowspans()

        if any(
            value.strip()
            for value
            in current_row
        ):
            matrix.append(
                current_row
            )

    if not matrix:
        return []

    max_columns = max(
        len(row)
        for row in matrix
    )

    for row in matrix:

        if len(row) < max_columns:

            row.extend(
                [""] * (
                    max_columns
                    - len(row)
                )
            )

    return matrix


# ============================================================
# TABLE TO TEXT
# ============================================================

# ============================================================
# TABLE TO TEXT
# ============================================================

def compress_repeated_cells(
    row: list[str],
) -> list[str]:
    """
    rowspan / colspan 복원 과정에서 생긴
    연속 중복 cell을 embedding용 text에서 제거한다.

    예:
        ["6. 신주 발행가액", "보통주 (원)", "보통주 (원)", "21,000"]

    ->
        ["6. 신주 발행가액", "보통주 (원)", "21,000"]

    주의:
        떨어져 있는 동일 값은 제거하지 않는다.

    예:
        ["비상장", "10", "-", "-", "10", "3"]

    ->
        ["비상장", "10", "-", "10", "3"]

    즉 첫 번째 10과 두 번째 10은 서로 떨어져 있으므로
    둘 다 유지된다.
    """

    compressed: list[str] = []

    previous_value: str | None = None

    for cell in row:

        value = clean_text(
            cell
        )

        # 빈 cell은 embedding text에서 제외
        if not value:
            continue

        # 직전 cell과 같은 값이면
        # colspan/rowspan 확장으로 인한 중복으로 보고 제거
        if (
            previous_value is not None
            and value == previous_value
        ):
            continue

        compressed.append(
            value
        )

        previous_value = (
            value
        )

    return compressed


def matrix_to_text(
    matrix: list[list[str]],
) -> str:
    """
    복원된 table matrix를
    Vector DB embedding용 문자열로 변환한다.

    원본 matrix는 수정하지 않고,
    문자열 직렬화 단계에서만 연속 중복을 제거한다.
    """

    lines: list[str] = []

    for row in matrix:

        compressed_row = (
            compress_repeated_cells(
                row
            )
        )

        if not compressed_row:
            continue

        line = " | ".join(
            compressed_row
        )

        if line.strip():

            lines.append(
                line
            )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# CHUNK STRATEGY
# ============================================================

def get_chunk_strategy(
    record: dict[str, Any],
) -> str:
    """
    문서 group별 기본 chunk 전략.

    table:
        구조를 그대로 보존

    field_value:
        field-value 변환 우선
    """

    doc_group = str(
        record.get(
            "doc_group"
        )
        or ""
    ).strip()

    # 복잡한 다차원 표
    if doc_group in (
        "periodic",
        "holding",
        "major"
    ):
        return "table"

    # 비교적 정형 공시
    if doc_group in (
        "exchange",
    ):
        return "field_value"

    # 알 수 없는 경우 정보 손실 방지
    return "table"


# ============================================================
# FIELD EXTRACTION
# ============================================================

def matrix_to_fields(
    matrix: list[list[str]],
) -> list[dict[str, Any]]:

    fields: list[
        dict[str, Any]
    ] = []

    for row_index, row in enumerate(
        matrix,
        start=1,
    ):

        values = [
            clean_text(
                value
            )
            for value
            in row
        ]

        non_empty = [
            value
            for value
            in values
            if value
        ]

        if len(
            non_empty
        ) < 2:
            continue

        # ----------------------------------------------------
        # 2열
        # field | value
        # ----------------------------------------------------

        if len(
            non_empty
        ) == 2:

            raw_field = (
                clean_field_name(
                    non_empty[0]
                )
            )

            value = (
                non_empty[1]
            )

            if not raw_field:
                continue

            fields.append(
                {
                    "row_index":
                        row_index,

                    "parent_field":
                        "",

                    "raw_field":
                        raw_field,

                    "canonical_field":
                        canonical_field_key(
                            raw_field
                        ),

                    "value":
                        value,
                }
            )

            continue

        # ----------------------------------------------------
        # 3열 이상
        # parent | child | value
        # ----------------------------------------------------

        parent = clean_field_name(
            non_empty[0]
        )

        child = clean_field_name(
            non_empty[-2]
        )

        value = non_empty[-1]

        if (
            parent
            and child
            and parent != child
        ):

            full_field = (
                f"{parent} > {child}"
            )

        else:

            full_field = (
                child
                or parent
            )

        if not full_field:
            continue

        fields.append(
            {
                "row_index":
                    row_index,

                "parent_field":
                    parent,

                "raw_field":
                    full_field,

                "canonical_field":
                    canonical_field_key(
                        full_field
                    ),

                "value":
                    value,
            }
        )

    return fields


# ============================================================
# FIELD QUALITY CHECK
# ============================================================

def fields_are_usable(
    fields: list[
        dict[str, Any]
    ],
) -> bool:
    """
    field-value 변환이 최소한 정상적인지 검사.

    실패 시 table 방식으로 fallback.
    """

    if not fields:
        return False

    valid_count = 0

    suspicious_count = 0

    for field in fields:

        name = clean_text(
            field.get(
                "raw_field"
            )
        )

        value = clean_text(
            field.get(
                "value"
            )
        )

        if (
            not name
            or not value
        ):
            suspicious_count += 1
            continue

        # field와 value가 완전히 동일
        if name == value:

            suspicious_count += 1
            continue

        # field가 지나치게 긴 문장
        if len(name) > 120:

            suspicious_count += 1
            continue

        # 숫자 하나가 field가 된 경우
        if re.fullmatch(
            r"[\d,.\-%]+",
            name,
        ):

            suspicious_count += 1
            continue

        valid_count += 1

    if valid_count < 2:
        return False

    total = (
        valid_count
        + suspicious_count
    )

    if total <= 0:
        return False

    valid_ratio = (
        valid_count
        / total
    )

    # 절반 이상이 정상이어야 field-value 사용
    return (
        valid_ratio
        >= 0.5
    )


# ============================================================
# PARSE EVENT DOCUMENT
# ============================================================

def parse_event_document(
    path: Path,
) -> list[dict[str, Any]]:
    """
    여기서는 table 구조만 복원한다.

    field-value 변환 여부는
    build 단계에서 문서 유형에 따라 결정한다.
    """

    raw = read_document(
        path
    )

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    tables = soup.find_all(
        "table"
    )

    results: list[
        dict[str, Any]
    ] = []

    for table_index, table in enumerate(
        tables,
        start=1,
    ):

        matrix = extract_table_matrix(
            table
        )

        if not matrix:
            continue

        title = find_table_title(
            table
        )

        table_text = matrix_to_text(
            matrix
        )

        results.append(
            {
                "table_index":
                    table_index,

                "title":
                    title,

                "rows":
                    matrix,

                "table_text":
                    table_text,

                "row_count":
                    len(matrix),

                "max_column_count":
                    max(
                        len(row)
                        for row
                        in matrix
                    ),
            }
        )

    return results


# ============================================================
# SEARCHABILITY
# ============================================================

def is_searchable_event_table(
    parsed_table: dict[str, Any],
) -> bool:

    rows = (
        parsed_table.get(
            "rows"
        )
        or []
    )

    text = clean_text(
        parsed_table.get(
            "table_text"
        )
    )

    if len(rows) < 2:
        return False

    if len(text) < 10:
        return False

    separator_count = len(
        re.findall(
            r"-{5,}",
            text,
        )
    )

    if separator_count >= 3:
        return False

    useful_text = re.sub(
        r"[-|\s]",
        "",
        text,
    )

    if len(useful_text) < 5:
        return False

    return True


# ============================================================
# CHUNK TYPE
# ============================================================

def classify_event_chunk(
    table_text: str,
    fields: list[
        dict[str, Any]
    ] | None = None,
) -> tuple[str, str]:

    fields = (
        fields
        or []
    )

    combined = (
        "\n".join(
            (
                f"{field.get('raw_field', '')} "
                f"{field.get('value', '')}"
            )
            for field
            in fields
        )
        + "\n"
        + table_text
    )

    if any(
        keyword in combined
        for keyword in (
            "계약",
            "수주",
            "공급",
        )
    ):
        return (
            "contract_event",
            "high",
        )

    if any(
        keyword in combined
        for keyword in (
            "자기주식",
            "취득예정",
            "처분예정",
        )
    ):
        return (
            "treasury_stock_event",
            "high",
        )

    if any(
        keyword in combined
        for keyword in (
            "유상증자",
            "무상증자",
            "증자",
        )
    ):
        return (
            "capital_event",
            "high",
        )

    if any(
        keyword in combined
        for keyword in (
            "시설투자",
            "신규시설",
            "투자금액",
        )
    ):
        return (
            "investment_event",
            "high",
        )

    if any(
        keyword in combined
        for keyword in (
            "보유주식",
            "보유비율",
            "대량보유",
        )
    ):
        return (
            "holding_report",
            "high",
        )

    return (
        "general_event",
        "medium",
    )


# ============================================================
# FIELD GROUPING
# ============================================================

def split_fields_into_groups(
    fields: list[
        dict[str, Any]
    ],
) -> list[
    list[dict[str, Any]]
]:

    groups: list[
        list[dict[str, Any]]
    ] = []

    current: list[
        dict[str, Any]
    ] = []

    for field in fields:

        value = clean_text(
            field.get(
                "value"
            )
        )

        # 긴 서술형 값
        if (
            len(value)
            >= LONG_VALUE_THRESHOLD
        ):

            if current:

                groups.append(
                    current
                )

                current = []

            groups.append(
                [
                    field
                ]
            )

            continue

        current.append(
            field
        )

        if (
            len(current)
            >= MAX_FIELDS_PER_CHUNK
        ):

            groups.append(
                current
            )

            current = []

    if current:

        groups.append(
            current
        )

    return groups


# ============================================================
# COMMON CHUNK HEADER
# ============================================================

def build_context_lines(
    record: dict[str, Any],
    table_title: str,
) -> list[str]:

    lines: list[str] = []

    corp_name = clean_text(
        record.get(
            "corp_name"
        )
    )

    report_nm = clean_text(
        record.get(
            "report_nm"
        )
    )

    report_type = clean_text(
        record.get(
            "normalized_report_type"
        )
    )

    event_date = clean_text(
        record.get(
            "event_date"
        )
    )

    if corp_name:

        lines.append(
            f"회사: {corp_name}"
        )

    if report_nm:

        lines.append(
            f"공시명: {report_nm}"
        )

    if report_type:

        lines.append(
            f"공시유형: {report_type}"
        )

    if event_date:

        lines.append(
            f"사건일자: {event_date}"
        )

    if table_title:

        lines.append(
            f"표 제목: {table_title}"
        )

    return lines


# ============================================================
# FIELD-VALUE CHUNK TEXT
# ============================================================

def build_field_value_text(
    record: dict[str, Any],
    table_title: str,
    fields: list[
        dict[str, Any]
    ],
) -> str:

    lines = build_context_lines(
        record,
        table_title,
    )

    lines.append("")

    for field in fields:

        name = clean_text(
            field.get(
                "raw_field"
            )
        )

        value = clean_text(
            field.get(
                "value"
            )
        )

        if (
            not name
            or not value
        ):
            continue

        lines.append(
            f"{name}: {value}"
        )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# TABLE CHUNK TEXT
# ============================================================

def build_table_text(
    record: dict[str, Any],
    table_title: str,
    table_text: str,
) -> str:

    lines = build_context_lines(
        record,
        table_title,
    )

    lines.append("")

    if table_text:

        lines.append(
            table_text
        )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# COMMON METADATA
# ============================================================

def build_common_metadata(
    record: dict[str, Any],
    parsed_table: dict[str, Any],
    raw_file: Path,
    raw_file_index: int,
    chunk_type: str,
    search_priority: str,
    chunk_strategy: str,
) -> dict[str, Any]:

    return {

        # =====================================================
        # DOCUMENT
        # =====================================================

        "doc_id":
            record.get(
                "doc_id"
            ),

        "corp_code":
            record.get(
                "corp_code"
            ),

        "corp_name":
            record.get(
                "corp_name"
            ),

        "stock_code":
            record.get(
                "stock_code"
            ),

        "industry":
            record.get(
                "industry"
            ),

        "sector":
            record.get(
                "sector"
            ),

        # =====================================================
        # DISCLOSURE
        # =====================================================

        "doc_group":
            record.get(
                "doc_group"
            ),

        "doc_subtype":
            record.get(
                "doc_subtype"
            ),

        "report_nm":
            record.get(
                "report_nm"
            ),

        "normalized_report_type":
            record.get(
                "normalized_report_type"
            ),

        "rcept_no":
            record.get(
                "rcept_no"
            ),

        "rcept_dt":
            record.get(
                "rcept_dt"
            ),

        "event_date":
            record.get(
                "event_date"
            ),

        "base_year":
            record.get(
                "base_year"
            ),

        "base_month":
            record.get(
                "base_month"
            ),

        # =====================================================
        # CORRECTION
        # =====================================================

        "is_correction":
            record.get(
                "is_correction"
            ),

        "disclosure_chain_id":
            record.get(
                "disclosure_chain_id"
            ),

        "corrects_doc_id":
            record.get(
                "corrects_doc_id"
            ),

        "corrects_rcept_no":
            record.get(
                "corrects_rcept_no"
            ),

        # =====================================================
        # TABLE
        # =====================================================

        "raw_file_index":
            raw_file_index,

        "table_index":
            parsed_table.get(
                "table_index"
            ),

        "table_title":
            parsed_table.get(
                "title"
            ),

        "row_count":
            parsed_table.get(
                "row_count"
            ),

        "max_column_count":
            parsed_table.get(
                "max_column_count"
            ),

        # =====================================================
        # RETRIEVAL
        # =====================================================

        "chunk_type":
            chunk_type,

        "search_priority":
            search_priority,

        "chunk_strategy":
            chunk_strategy,

        # =====================================================
        # SOURCE
        # =====================================================

        "raw_file":
            str(
                raw_file
            ),
    }


# ============================================================
# BUILD TABLE CHUNKS
# ============================================================

def build_table_chunks(
    record: dict[str, Any],
    parsed_table: dict[str, Any],
    raw_file: Path,
    raw_file_index: int,
) -> list[
    dict[str, Any]
]:

    doc_id = str(
        record.get(
            "doc_id"
        )
        or record.get(
            "rcept_no"
        )
        or ""
    )

    table_index = int(
        parsed_table.get(
            "table_index"
        )
        or 0
    )

    title = clean_text(
        parsed_table.get(
            "title"
        )
    )

    matrix = (
        parsed_table.get(
            "rows"
        )
        or []
    )

    table_text = clean_text(
        parsed_table.get(
            "table_text"
        )
    )

    # ========================================================
    # STRATEGY
    # ========================================================

    strategy = get_chunk_strategy(
        record
    )

    fields: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # major / exchange
    # → field-value 우선
    # --------------------------------------------------------

    if strategy == "field_value":

        candidate_fields = (
            matrix_to_fields(
                matrix
            )
        )

        if fields_are_usable(
            candidate_fields
        ):

            fields = (
                candidate_fields
            )

        else:

            # field-value 품질이 나쁘면
            # table fallback
            strategy = "table"

    # ========================================================
    # CLASSIFY
    # ========================================================

    chunk_type, search_priority = (
        classify_event_chunk(
            table_text=table_text,
            fields=fields,
        )
    )

    chunks: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # FIELD-VALUE
    # ========================================================

    if (
        strategy
        == "field_value"
    ):

        field_groups = (
            split_fields_into_groups(
                fields
            )
        )

        for group_index, group in enumerate(
            field_groups,
            start=1,
        ):

            chunk_id = (
                f"{doc_id}"
                f"_file_{raw_file_index:02d}"
                f"_table_{table_index:04d}"
                f"_part_{group_index:02d}"
            )

            text = (
                build_field_value_text(
                    record=record,
                    table_title=title,
                    fields=group,
                )
            )

            metadata = (
                build_common_metadata(
                    record=record,
                    parsed_table=parsed_table,
                    raw_file=raw_file,
                    raw_file_index=(
                        raw_file_index
                    ),
                    chunk_type=(
                        chunk_type
                    ),
                    search_priority=(
                        search_priority
                    ),
                    chunk_strategy=(
                        "field_value"
                    ),
                )
            )

            metadata[
                "field_names"
            ] = [
                clean_text(
                    field.get(
                        "raw_field"
                    )
                )
                for field
                in group
            ]

            metadata[
                "canonical_fields"
            ] = [
                clean_text(
                    field.get(
                        "canonical_field"
                    )
                )
                for field
                in group
            ]

            chunks.append(
                {
                    "chunk_id":
                        chunk_id,

                    "text":
                        text,

                    "metadata":
                        metadata,
                }
            )

        return chunks

    # ========================================================
    # TABLE
    # ========================================================

    chunk_id = (
        f"{doc_id}"
        f"_file_{raw_file_index:02d}"
        f"_table_{table_index:04d}"
    )

    text = build_table_text(
        record=record,
        table_title=title,
        table_text=table_text,
    )

    metadata = (
        build_common_metadata(
            record=record,
            parsed_table=parsed_table,
            raw_file=raw_file,
            raw_file_index=(
                raw_file_index
            ),
            chunk_type=(
                chunk_type
            ),
            search_priority=(
                search_priority
            ),
            chunk_strategy=(
                "table"
            ),
        )
    )

    metadata[
        "field_names"
    ] = []

    metadata[
        "canonical_fields"
    ] = []

    chunks.append(
        {
            "chunk_id":
                chunk_id,

            "text":
                text,

            "metadata":
                metadata,
        }
    )

    return chunks


# ============================================================
# BUILD DOCUMENT CHUNKS
# ============================================================

def build_document_chunks(
    record: dict[str, Any],
) -> list[
    dict[str, Any]
]:

    raw_files = find_raw_files(
        record
    )

    chunks: list[
        dict[str, Any]
    ] = []

    for raw_file_index, raw_file in enumerate(
        raw_files,
        start=1,
    ):

        parsed_tables = (
            parse_event_document(
                raw_file
            )
        )

        for parsed_table in parsed_tables:

            if not (
                is_searchable_event_table(
                    parsed_table
                )
            ):
                continue

            table_chunks = (
                build_table_chunks(
                    record=record,
                    parsed_table=parsed_table,
                    raw_file=raw_file,
                    raw_file_index=(
                        raw_file_index
                    ),
                )
            )

            chunks.extend(
                table_chunks
            )

    return chunks


# ============================================================
# SAVE
# ============================================================

def save_jsonl(
    records: list[
        dict[str, Any]
    ],
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
    chunks: list[
        dict[str, Any]
    ],
    limit: int = 10,
) -> None:

    print()
    print("=" * 100)
    print(
        "Event Vector Chunk Preview"
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
            "chunk_strategy:",
            metadata.get(
                "chunk_strategy"
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
            "is_correction:",
            metadata.get(
                "is_correction"
            ),
        )

        print(
            "field_names:",
            metadata.get(
                "field_names"
            ),
        )

        print()
        print("[TEXT]")

        text = clean_text(
            chunk.get(
                "text"
            )
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

def print_statistics(
    chunks: list[
        dict[str, Any]
    ],
) -> None:

    type_counts: dict[
        str,
        int,
    ] = {}

    strategy_counts: dict[
        str,
        int,
    ] = {}

    group_counts: dict[
        str,
        int,
    ] = {}

    correction_count = 0

    chunk_ids: set[
        str
    ] = set()

    duplicate_ids: list[
        str
    ] = []

    for chunk in chunks:

        chunk_id = clean_text(
            chunk.get(
                "chunk_id"
            )
        )

        if chunk_id in chunk_ids:

            duplicate_ids.append(
                chunk_id
            )

        else:

            chunk_ids.add(
                chunk_id
            )

        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        chunk_type = clean_text(
            metadata.get(
                "chunk_type"
            )
        ) or "unknown"

        strategy = clean_text(
            metadata.get(
                "chunk_strategy"
            )
        ) or "unknown"

        doc_group = clean_text(
            metadata.get(
                "doc_group"
            )
        ) or "unknown"

        type_counts[
            chunk_type
        ] = (
            type_counts.get(
                chunk_type,
                0,
            )
            + 1
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

        group_counts[
            doc_group
        ] = (
            group_counts.get(
                doc_group,
                0,
            )
            + 1
        )

        if metadata.get(
            "is_correction"
        ):

            correction_count += 1

    print()
    print("=" * 100)
    print(
        "Event Chunk Statistics"
    )
    print("=" * 100)

    print()
    print("[DOC GROUP]")

    for key, value in sorted(
        group_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    print()
    print("[STRATEGY]")

    for key, value in sorted(
        strategy_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    print()
    print("[TYPE]")

    for key, value in sorted(
        type_counts.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    print()
    print("[CORRECTION]")

    print(
        "correction chunks:",
        correction_count,
    )

    print()
    print("[CHUNK ID]")

    print(
        "unique chunk_id:",
        len(
            chunk_ids
        ),
    )

    print(
        "duplicate chunk_id:",
        len(
            duplicate_ids
        ),
    )

    if duplicate_ids:

        print()
        print(
            "중복 예시:"
        )

        for chunk_id in (
            duplicate_ids[:10]
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

    # ========================================================
    # NON-PERIODIC
    # ========================================================

    target_records = [
        record
        for record
        in manifest
        if (
            record.get(
                "doc_group"
            )
            != "periodic"
        )
    ]

    # 특정 회사 테스트
    if TARGET_CORP_NAME:

        target_records = [
            record
            for record
            in target_records
            if (
                record.get(
                    "corp_name"
                )
                == TARGET_CORP_NAME
            )
        ]

    # 접수일 순 정렬
    target_records.sort(
        key=lambda record: (
            str(
                record.get(
                    "rcept_dt"
                )
                or ""
            ),
            str(
                record.get(
                    "rcept_no"
                )
                or ""
            ),
        )
    )

    if MAX_DOCUMENTS is not None:

        target_records = (
            target_records[
                :MAX_DOCUMENTS
            ]
        )

    print()
    print("=" * 100)
    print(
        "Event Chunk Builder"
    )
    print("=" * 100)

    print(
        "대상 회사:",
        TARGET_CORP_NAME
        or "ALL",
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

        strategy = get_chunk_strategy(
            record
        )

        print()
        print(
            f"[{document_index}/"
            f"{len(target_records)}]"
        )

        print(
            record.get(
                "corp_name"
            ),
            "|",
            record.get(
                "doc_group"
            ),
            "|",
            strategy,
            "|",
            record.get(
                "report_nm"
            ),
            "|",
            record.get(
                "rcept_no"
            ),
        )

        chunks = (
            build_document_chunks(
                record
            )
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
        limit=10,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print_statistics(
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