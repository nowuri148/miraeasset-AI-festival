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

# event note chunking
EVENT_NOTE_MIN_CHARS = 40
EVENT_NOTE_HEADING_MIN_CHARS = 8
EVENT_NOTE_TARGET_CHARS = 600
EVENT_NOTE_MAX_CHARS = 900

TABLE_TARGET_CHARS = 1000
TABLE_MAX_CHARS = 1500
DEFAULT_HEADER_ROW_COUNT = 1


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
# TABLE SPLIT
# ============================================================

def detect_header_row_count(
    table: Any,
    matrix: list[list[str]],
) -> int:
    """
    <thead>가 있으면 실제 header row 수를 사용한다.

    없으면 기본적으로 첫 번째 row를 header로 본다.
    """

    if not matrix:
        return 0

    thead = table.find(
        "thead"
    )

    if thead is not None:

        header_rows = thead.find_all(
            "tr",
            recursive=False,
        )

        count = len(
            header_rows
        )

        if count > 0:

            return min(
                count,
                3,
            )

    return min(
        DEFAULT_HEADER_ROW_COUNT,
        len(matrix),
    )


def build_split_table_text(
    matrix: list[list[str]],
) -> str:
    """
    split된 matrix를 기존 event table과
    동일한 직렬화 방식으로 text로 만든다.
    """

    return matrix_to_text(
        matrix
    )


def estimate_split_table_length(
    matrix: list[list[str]],
) -> int:

    return len(
        build_split_table_text(
            matrix
        )
    )


def split_large_event_table(
    matrix: list[list[str]],
    header_row_count: int,
) -> list[dict[str, Any]]:
    """
    긴 table을 row 단위로 분할한다.

    - header row는 각 part에 반복
    - row 자체를 중간에서 자르지 않음
    - 한 row 자체가 1500자를 넘으면
      oversized_single_row=True로 그대로 보존
    """

    if not matrix:
        return []

    full_length = (
        estimate_split_table_length(
            matrix
        )
    )

    # ========================================================
    # 작은 표
    # ========================================================

    if (
        full_length
        <= TABLE_MAX_CHARS
    ):

        return [
            {
                "rows":
                    matrix,

                "source_row_start":
                    1,

                "source_row_end":
                    len(matrix),

                "oversized_single_row":
                    False,
            }
        ]

    # ========================================================
    # HEADER / DATA 분리
    # ========================================================

    header_row_count = max(
        0,
        min(
            header_row_count,
            len(matrix),
        ),
    )

    headers = matrix[
        :header_row_count
    ]

    data_rows = matrix[
        header_row_count:
    ]

    # header만 있는 이상한 표
    if not data_rows:

        return [
            {
                "rows":
                    matrix,

                "source_row_start":
                    1,

                "source_row_end":
                    len(matrix),

                "oversized_single_row":
                    (
                        full_length
                        > TABLE_MAX_CHARS
                    ),
            }
        ]

    parts: list[
        dict[str, Any]
    ] = []

    current_rows: list[
        list[str]
    ] = []

    current_start = (
        header_row_count
        + 1
    )

    for data_index, row in enumerate(
        data_rows,
        start=header_row_count + 1,
    ):

        candidate_rows = [
            *headers,
            *current_rows,
            row,
        ]

        candidate_length = (
            estimate_split_table_length(
                candidate_rows
            )
        )

        # ----------------------------------------------------
        # 현재 part가 비어 있는데
        # header + row 하나만으로 MAX 초과
        # ----------------------------------------------------

        if (
            not current_rows
            and candidate_length
            > TABLE_MAX_CHARS
        ):

            parts.append(
                {
                    "rows":
                        [
                            *headers,
                            row,
                        ],

                    "source_row_start":
                        data_index,

                    "source_row_end":
                        data_index,

                    "oversized_single_row":
                        True,
                }
            )

            current_start = (
                data_index
                + 1
            )

            continue

        # ----------------------------------------------------
        # 현재 part에 row를 추가하면 MAX 초과
        # ----------------------------------------------------

        if (
            current_rows
            and candidate_length
            > TABLE_MAX_CHARS
        ):

            parts.append(
                {
                    "rows":
                        [
                            *headers,
                            *current_rows,
                        ],

                    "source_row_start":
                        current_start,

                    "source_row_end":
                        data_index
                        - 1,

                    "oversized_single_row":
                        False,
                }
            )

            current_rows = [
                row
            ]

            current_start = (
                data_index
            )

            continue

        # ----------------------------------------------------
        # 아직 TARGET 이하
        # ----------------------------------------------------

        current_rows.append(
            row
        )

        current_length = (
            estimate_split_table_length(
                [
                    *headers,
                    *current_rows,
                ]
            )
        )

        # TARGET 도달 시 여기서 part 종료
        if (
            current_length
            >= TABLE_TARGET_CHARS
        ):

            parts.append(
                {
                    "rows":
                        [
                            *headers,
                            *current_rows,
                        ],

                    "source_row_start":
                        current_start,

                    "source_row_end":
                        data_index,

                    "oversized_single_row":
                        False,
                }
            )

            current_rows = []

            current_start = (
                data_index
                + 1
            )

    # ========================================================
    # 마지막 남은 row
    # ========================================================

    if current_rows:

        parts.append(
            {
                "rows":
                    [
                        *headers,
                        *current_rows,
                    ],

                "source_row_start":
                    current_start,

                "source_row_end":
                    header_row_count
                    + len(
                        data_rows
                    ),

                "oversized_single_row":
                    False,
            }
        )

    return parts

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
    여기서는 원본 table 구조만 복원한다.

    중요:
    - exchange는 원본 전체 table을 field-value로 변환해야 하므로
      이 단계에서 table을 분할하지 않는다.
    - major / holding의 대형 table 분할은
      build_document_chunks()에서 문서 유형을 확인한 뒤 수행한다.
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

        header_row_count = (
            detect_header_row_count(
                table=table,
                matrix=matrix,
            )
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

                # 원본 table의 실제 header 수를 보존
                "header_row_count":
                    header_row_count,

                # 아직 split 전
                "table_part_index":
                    1,

                "table_part_count":
                    1,

                "is_split":
                    False,

                "source_row_start":
                    1,

                "source_row_end":
                    len(matrix),

                "oversized_single_row":
                    False,

                "text_length":
                    len(
                        table_text
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

    if not rows:
        return False

    if len(text) < 10:
        return False

    # 1행짜리 표라도 긴 설명문이면 보존한다.
    # 단위표/짧은 장식성 표만 제거한다.
    if (
        len(rows) == 1
        and len(text) < 120
    ):
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

        "table_part_index":
            parsed_table.get(
                "table_part_index",
                1,
            ),

        "table_part_count":
            parsed_table.get(
                "table_part_count",
                1,
            ),

        "is_split":
            parsed_table.get(
                "is_split",
                False,
            ),

        "header_row_count":
            parsed_table.get(
                "header_row_count"
            ),

        "source_row_start":
            parsed_table.get(
                "source_row_start"
            ),

        "source_row_end":
            parsed_table.get(
                "source_row_end"
            ),

        "oversized_single_row":
            parsed_table.get(
                "oversized_single_row",
                False,
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

    table_part_index = int(
        parsed_table.get(
            "table_part_index"
        )
        or 1
    )

    table_part_count = int(
        parsed_table.get(
            "table_part_count"
        )
        or 1
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
    # exchange
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
        f"_part_{table_part_index:03d}"
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
                "table_row_split"
                if table_part_count > 1
                else "table"
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
# EVENT NOTE
# ============================================================

SECTION_TAG_PATTERN = re.compile(
    r"^section-\d+$",
    re.I,
)

NOTE_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\d+\.\s*"
    r"|[-※]\s*"
    r"|주\)\s*"
    r"|\([0-9]+\)\s*"
    r")"
)


def direct_section_title(
    section: Any,
) -> str:
    """
    현재 SECTION의 직계 TITLE만 반환한다.
    하위 SECTION의 TITLE은 섞지 않는다.
    """

    for child in section.children:

        name = str(
            getattr(
                child,
                "name",
                "",
            )
            or ""
        ).lower()

        if name != "title":
            continue

        title = clean_text(
            child.get_text(
                " ",
                strip=True,
            )
        )

        if title:
            return title

    return ""


def get_section_path(
    node: Any,
) -> str:
    """
    P가 속한 SECTION 계층을 상위 -> 하위 순서로 복원한다.
    """

    sections = []

    for parent in node.parents:

        name = str(
            getattr(
                parent,
                "name",
                "",
            )
            or ""
        )

        if SECTION_TAG_PATTERN.match(
            name
        ):
            sections.append(
                parent
            )

    sections.reverse()

    titles: list[str] = []

    for section in sections:

        title = direct_section_title(
            section
        )

        if (
            title
            and (
                not titles
                or titles[-1] != title
            )
        ):
            titles.append(
                title
            )

    return " > ".join(
        titles
    )


def is_event_note_heading(
    text: str,
) -> bool:
    """
    짧더라도 다음 긴 설명과 결합할 가치가 있는 P 제목인지 검사한다.
    """

    text = clean_text(
        text
    )

    if len(text) < EVENT_NOTE_HEADING_MIN_CHARS:
        return False

    if len(text) > 120:
        return False

    if NOTE_HEADING_PATTERN.match(
        text
    ):
        return True

    if any(
        keyword in text
        for keyword in (
            "기타 투자판단",
            "참고할 사항",
            "관련공시",
            "주요내용",
            "산정",
            "주의",
        )
    ):
        return True

    return False


def is_meaningful_event_note(
    text: str,
    allow_heading: bool = False,
) -> bool:

    text = clean_text(
        text
    )

    if not text:
        return False

    if (
        allow_heading
        and is_event_note_heading(
            text
        )
    ):
        return True

    if len(text) < EVENT_NOTE_MIN_CHARS:
        return False

    # "*해당사항 없음" 류 제거
    if re.fullmatch(
        r"[*※\s]*해당사항\s*없음[.]?",
        text,
    ):
        return False

    # 단위 표시 제거
    if re.fullmatch(
        r"\(?\s*단위\s*[:：].*?\)?",
        text,
    ):
        return False

    return True


def split_event_note_sentences(
    text: str,
) -> list[str]:
    """
    약식 문장 분리.
    variable-width lookbehind는 사용하지 않는다.
    """

    text = clean_text(
        text
    )

    if not text:
        return []

    # HTML/XML에서 줄바꿈이 소실된 경우를 조금 보완한다.
    text = re.sub(
        r"(?<=[.!?。])(?=[가-힣A-Za-z0-9\-※①②③④⑤⑥⑦⑧⑨])",
        "\n",
        text,
    )

    parts = re.split(
        r"(?<=[.!?。])\s+|\n+",
        text,
    )

    result: list[str] = []

    for part in parts:

        part = clean_text(
            part
        )

        if not part:
            continue

        if len(part) <= EVENT_NOTE_MAX_CHARS:

            result.append(
                part
            )

            continue

        # 한 문장 자체가 지나치게 긴 경우 fallback 분할
        start = 0

        while start < len(part):

            end = min(
                start + EVENT_NOTE_MAX_CHARS,
                len(part),
            )

            piece = part[
                start:end
            ]

            if end < len(part):

                candidates = [
                    piece.rfind(". "),
                    piece.rfind(", "),
                    piece.rfind("; "),
                    piece.rfind(" "),
                ]

                cut = max(
                    candidates
                )

                if cut > int(
                    EVENT_NOTE_MAX_CHARS
                    * 0.6
                ):

                    end = (
                        start
                        + cut
                        + 1
                    )

                    piece = part[
                        start:end
                    ]

            piece = clean_text(
                piece
            )

            if piece:
                result.append(
                    piece
                )

            start = end

    return result


def pack_event_note_text(
    text: str,
) -> list[str]:
    """
    약 600자를 목표로 묶고 900자를 넘지 않게 한다.
    event note는 overlap을 두지 않는다.
    """

    sentences = (
        split_event_note_sentences(
            text
        )
    )

    if not sentences:
        return []

    chunks: list[str] = []

    current: list[str] = []
    current_length = 0

    for sentence in sentences:

        additional = (
            len(sentence)
            + (
                1
                if current
                else 0
            )
        )

        if (
            current
            and (
                current_length
                + additional
                > EVENT_NOTE_MAX_CHARS
            )
        ):

            chunks.append(
                clean_text(
                    " ".join(
                        current
                    )
                )
            )

            current = []
            current_length = 0

        current.append(
            sentence
        )

        current_length += (
            len(sentence)
            + (
                1
                if len(current) > 1
                else 0
            )
        )

        if (
            current_length
            >= EVENT_NOTE_TARGET_CHARS
        ):

            chunks.append(
                clean_text(
                    " ".join(
                        current
                    )
                )
            )

            current = []
            current_length = 0

    if current:

        body = clean_text(
            " ".join(
                current
            )
        )

        if body:
            chunks.append(
                body
            )

    return chunks


def build_event_note_chunks(
    record: dict[str, Any],
    raw_file: Path,
    raw_file_index: int,
) -> list[dict[str, Any]]:
    """
    major / holding에 있는 TABLE 밖 P 설명을 보존한다.
    exchange에는 호출하지 않는다.
    """

    raw = read_document(
        raw_file
    )

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    doc_id = str(
        record.get("doc_id")
        or record.get("rcept_no")
        or raw_file.stem
    )

    candidates: list[
        dict[str, str]
    ] = []

    for p_tag in soup.find_all(
        "p"
    ):

        # table 내부 문장은 table chunk에서 이미 보존된다.
        if p_tag.find_parent(
            "table"
        ) is not None:
            continue

        p_text = clean_text(
            p_tag.get_text(
                " ",
                strip=True,
            )
        )

        if not is_meaningful_event_note(
            p_text,
            allow_heading=True,
        ):
            continue

        candidates.append(
            {
                "section_path":
                    get_section_path(
                        p_tag
                    ),

                "text":
                    p_text,
            }
        )

    # 같은 SECTION의 연속 P들을 먼저 합친다.
    grouped: list[
        dict[str, str]
    ] = []

    current_section: str | None = None
    current_texts: list[str] = []

    def flush_group() -> None:

        nonlocal current_section
        nonlocal current_texts

        if not current_texts:
            return

        body = clean_text(
            " ".join(
                current_texts
            )
        )

        if body:

            grouped.append(
                {
                    "section_path":
                        current_section
                        or "",

                    "text":
                        body,
                }
            )

        current_texts = []

    for candidate in candidates:

        section_path = (
            candidate[
                "section_path"
            ]
        )

        if (
            current_section
            is not None
            and section_path
            != current_section
        ):
            flush_group()

        current_section = (
            section_path
        )

        current_texts.append(
            candidate[
                "text"
            ]
        )

    flush_group()

    chunks: list[
        dict[str, Any]
    ] = []

    note_index = 0

    for group in grouped:

        section_path = (
            group[
                "section_path"
            ]
        )

        bodies = pack_event_note_text(
            group[
                "text"
            ]
        )

        for body in bodies:

            # 제목 하나만 남은 지나치게 짧은 결과는 최종적으로 제거
            if not is_meaningful_event_note(
                body,
                allow_heading=False,
            ):
                continue

            note_index += 1

            chunk_id = (
                f"{doc_id}"
                f"_file_{raw_file_index:02d}"
                f"_note_{note_index:04d}"
            )

            chunk_type, search_priority = (
                classify_event_chunk(
                    table_text=body,
                    fields=[],
                )
            )

            context_lines = (
                build_context_lines(
                    record=record,
                    table_title="",
                )
            )

            if section_path:

                context_lines.append(
                    f"섹션: {section_path}"
                )

            context_lines.append("")
            context_lines.append(body)

            text = "\n".join(
                context_lines
            ).strip()

            metadata = {

                # DOCUMENT / COMPANY
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

                # DISCLOSURE
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

                # CORRECTION
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

                # NOTE
                "raw_file_index":
                    raw_file_index,

                "note_index":
                    note_index,

                "section_path":
                    section_path,

                # RETRIEVAL
                "chunk_type":
                    chunk_type,

                "search_priority":
                    search_priority,

                "chunk_strategy":
                    "event_note",

                # SOURCE
                "raw_file":
                    str(
                        raw_file
                    ),
            }

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

        # ====================================================
        # ORIGINAL TABLE PARSE
        # ====================================================

        parsed_tables = (
            parse_event_document(
                raw_file
            )
        )

        doc_group = clean_text(
            record.get(
                "doc_group"
            )
        )

        expanded_tables: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # MAJOR / HOLDING:
        # 긴 table만 row 단위 분할
        #
        # EXCHANGE:
        # 원본 table 전체를 유지한 채 field-value 변환
        # ====================================================

        for parsed_table in parsed_tables:

            matrix = (
                parsed_table.get(
                    "rows"
                )
                or []
            )

            if not matrix:
                continue

            if doc_group in (
                "major",
                "holding",
            ):

                header_row_count = int(
                    parsed_table.get(
                        "header_row_count"
                    )
                    or 0
                )

                # <thead>가 없는 경우 최소 첫 row를
                # split context로 반복한다.
                if header_row_count <= 0:
                    header_row_count = min(
                        DEFAULT_HEADER_ROW_COUNT,
                        len(matrix),
                    )

                parts = (
                    split_large_event_table(
                        matrix=matrix,
                        header_row_count=(
                            header_row_count
                        ),
                    )
                )

                part_count = len(
                    parts
                )

                for part_index, part in enumerate(
                    parts,
                    start=1,
                ):

                    part_matrix = (
                        part[
                            "rows"
                        ]
                    )

                    part_text = (
                        matrix_to_text(
                            part_matrix
                        )
                    )

                    expanded_tables.append(
                        {
                            **parsed_table,

                            "rows":
                                part_matrix,

                            "table_text":
                                part_text,

                            "row_count":
                                len(
                                    part_matrix
                                ),

                            "max_column_count":
                                max(
                                    len(row)
                                    for row
                                    in part_matrix
                                ),

                            "table_part_index":
                                part_index,

                            "table_part_count":
                                part_count,

                            "is_split":
                                (
                                    part_count
                                    > 1
                                ),

                            "header_row_count":
                                header_row_count,

                            "source_row_start":
                                part.get(
                                    "source_row_start"
                                ),

                            "source_row_end":
                                part.get(
                                    "source_row_end"
                                ),

                            "oversized_single_row":
                                part.get(
                                    "oversized_single_row",
                                    False,
                                ),

                            "text_length":
                                len(
                                    part_text
                                ),
                        }
                    )

            else:

                # exchange는 절대 선분할하지 않는다.
                # 원본 전체 table을 matrix_to_fields()에 넘긴다.
                expanded_tables.append(
                    {
                        **parsed_table,

                        "table_part_index":
                            1,

                        "table_part_count":
                            1,

                        "is_split":
                            False,

                        "source_row_start":
                            1,

                        "source_row_end":
                            parsed_table.get(
                                "row_count"
                            ),

                        "oversized_single_row":
                            False,
                    }
                )

        # ====================================================
        # TABLE / FIELD-VALUE CHUNK BUILD
        # ====================================================

        for parsed_table in expanded_tables:

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

        # ====================================================
        # EVENT NOTE
        #
        # major / holding만 표 밖 P 설명 보존
        # exchange는 field-value table에 설명 포함
        # ====================================================

        if doc_group in (
            "major",
            "holding",
        ):

            note_chunks = (
                build_event_note_chunks(
                    record=record,
                    raw_file=raw_file,
                    raw_file_index=(
                        raw_file_index
                    ),
                )
            )

            chunks.extend(
                note_chunks
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