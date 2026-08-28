from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# TABLE CHUNK SETTINGS
# ============================================================
# 작은 표는 그대로 1 chunk로 유지하고,
# 큰 표만 "header + row 묶음"으로 분할한다.
#
# TARGET은 권장 크기, MAX는 가능한 한 넘지 않도록 하는 상한이다.
# 단, 단일 row 자체가 MAX보다 긴 경우에는 그 row를 별도 chunk로 보존한다.
TABLE_TARGET_CHARS = 1000
TABLE_MAX_CHARS = 1500
DEFAULT_HEADER_ROW_COUNT = 1

# ============================================================
# TEXT UTILS
# ============================================================

def clean_text(text: str) -> str:
    text = str(text or "")

    text = text.replace("\xa0", " ")

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


def normalize_cell_text(text: str) -> str:
    """
    표 셀 내부 텍스트만 가볍게 정리한다.

    중요한 점:
    숫자/날짜/단위/문구는 가능한 그대로 보존한다.
    """

    text = clean_text(text)

    # 지나친 공백만 정리
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# FILE READER
# ============================================================

def read_document(
    path: Path,
) -> str:
    """
    DART 파일을 읽는다.

    실제 문서는 html 형태의 xml일 수 있으므로
    BeautifulSoup html.parser로 후처리한다.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {path}"
        )

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


# ============================================================
# TABLE TITLE DETECTION
# ============================================================

def find_table_title(
    table: Any,
) -> str:
    """
    table 바로 앞의 문구에서 제목 후보를 찾는다.

    완벽한 제목 복원기가 아니라
    Vector DB 검색 시 표 의미를 보강하는 용도.
    """

    # 이전 요소들을 가까운 순서대로 탐색
    previous = table.find_all_previous(
        [
            "p",
            "div",
            "span",
            "td",
            "title",
        ],
        limit=15,
    )

    for node in previous:

        text = clean_text(
            node.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        # 너무 긴 본문은 제목으로 보지 않음
        if len(text) > 120:
            continue

        # 단순 숫자/기호는 제외
        if re.fullmatch(
            r"[\d\W_]+",
            text,
        ):
            continue

        return text

    return ""


# ============================================================
# TABLE ROW EXTRACTION
# ============================================================

def extract_table_matrix(
    table: Any,
) -> list[list[str]]:
    """
    HTML table의 rowspan / colspan을 반영해서
    실제 2차원 grid로 복원한다.

    예:
        <td rowspan="2">구분</td>
        <td colspan="4">연결대상회사수</td>

    를 실제 matrix에서 반복/확장해준다.
    """

    rows = table.find_all(
        "tr"
    )

    if not rows:
        return []

    # --------------------------------------------------------
    # 각 column에 rowspan으로 내려오는 값 저장
    #
    # key   = column index
    # value = {
    #     "remaining": 남은 row 수,
    #     "text": 값
    # }
    # --------------------------------------------------------

    rowspan_state: dict[
        int,
        dict[str, Any],
    ] = {}

    matrix: list[
        list[str]
    ] = []

    for row in rows:

        cells = row.find_all(
            [
                "th",
                "td",
                "te",
            ],
            recursive=False,
        )

        current_row: list[str] = []

        col_index = 0


        # ====================================================
        # helper
        # ====================================================

        def fill_active_rowspans() -> None:
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


        # ----------------------------------------------------
        # row 시작 시 기존 rowspan 반영
        # ----------------------------------------------------

        fill_active_rowspans()


        # ====================================================
        # 현재 row의 cell 처리
        # ====================================================

        for cell in cells:

            # 현재 위치에 rowspan이 있으면 먼저 채운다.
            fill_active_rowspans()

            text = normalize_cell_text(
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


            # ------------------------------------------------
            # colspan만큼 현재 row에 복제
            # ------------------------------------------------

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

                # --------------------------------------------
                # rowspan이면 다음 row에도 동일 값 전달
                # --------------------------------------------

                if rowspan > 1:

                    rowspan_state[
                        target_col
                    ] = {
                        "remaining": (
                            rowspan - 1
                        ),
                        "text": text,
                    }

            col_index += colspan


        # ----------------------------------------------------
        # row 뒤쪽에 남아 있는 rowspan도 반영
        # ----------------------------------------------------

        fill_active_rowspans()


        # ----------------------------------------------------
        # 완전히 빈 row 제외
        # ----------------------------------------------------

        if any(
            value.strip()
            for value in current_row
        ):
            matrix.append(
                current_row
            )


    # ========================================================
    # 열 길이 맞추기
    # ========================================================

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
# TABLE QUALITY CHECK
# ============================================================

def is_meaningful_table(
    matrix: list[list[str]],
) -> bool:
    """
    검색 대상으로 쓸 가치가 있는 표인지 최소 검사.
    """

    if len(matrix) < 2:
        return False

    non_empty_cells = 0

    total_chars = 0

    for row in matrix:

        for cell in row:

            if cell:
                non_empty_cells += 1
                total_chars += len(cell)

    if non_empty_cells < 4:
        return False

    if total_chars < 10:
        return False

    return True


# ============================================================
# HEADER DETECTION
# ============================================================

def guess_headers(
    matrix: list[list[str]],
) -> list[str]:
    """
    첫 번째 행을 header 후보로 사용한다.

    현재 목적은 표 구조 보존이므로
    억지로 header를 추론하지 않는다.
    """

    if not matrix:
        return []

    first_row = matrix[0]

    if len(first_row) < 2:
        return []

    return first_row


# ============================================================
# TABLE TEXT SERIALIZATION
# ============================================================

def table_to_text(
    title: str,
    matrix: list[list[str]],
    corp_name: str = "",
    report_nm: str = "",
) -> str:
    """
    표 구조를 그대로 보존해서
    Vector DB용 검색 텍스트로 변환한다.

    같은 값이 연속되어도 삭제하지 않는다.
    각 값은 서로 다른 column에 속할 수 있기 때문이다.
    """

    lines: list[str] = []

    if corp_name:
        lines.append(
            f"회사: {corp_name}"
        )

    if report_nm:
        lines.append(
            f"보고서: {report_nm}"
        )

    if title:
        lines.append(
            f"표 제목: {title}"
        )

    if lines:
        lines.append("")

    for row in matrix:

        cleaned_row = [
            clean_text(cell)
            for cell in row
        ]

        lines.append(
            " | ".join(
                cleaned_row
            )
        )

    return "\n".join(
        lines
    ).strip()



# ============================================================
# LARGE TABLE CHUNKING
# ============================================================

def detect_header_row_count(
    table: Any,
    matrix: list[list[str]],
) -> int:
    """
    표의 header row 수를 추정한다.

    우선순위:
    1) <thead> 내부의 <tr> 개수
    2) 없으면 기본 1행

    너무 많은 header 반복을 피하기 위해 최대 3행까지만 사용한다.
    """

    if not matrix:
        return 0

    thead = table.find("thead")

    if thead is not None:
        count = len(
            thead.find_all(
                "tr",
                recursive=False,
            )
        )

        if count <= 0:
            count = len(
                thead.find_all("tr")
            )

        if count > 0:
            return min(
                count,
                3,
                len(matrix),
            )

    return min(
        DEFAULT_HEADER_ROW_COUNT,
        len(matrix),
    )


def matrix_to_lines(
    matrix: list[list[str]],
) -> list[str]:
    """
    matrix의 각 row를 검색용 문자열 한 줄로 변환한다.
    """

    lines: list[str] = []

    for row in matrix:

        cleaned_row = [
            clean_text(cell)
            for cell in row
        ]

        lines.append(
            " | ".join(
                cleaned_row
            )
        )

    return lines


def build_table_text(
    *,
    title: str,
    matrix: list[list[str]],
    corp_name: str = "",
    report_nm: str = "",
    part_index: int | None = None,
    part_count: int | None = None,
) -> str:
    """
    표 일부(matrix)를 Vector DB 검색용 text로 직렬화한다.

    분할된 표라면 "(분할 x/y)" 문맥도 넣는다.
    """

    lines: list[str] = []

    if corp_name:
        lines.append(
            f"회사: {corp_name}"
        )

    if report_nm:
        lines.append(
            f"보고서: {report_nm}"
        )

    if title:
        lines.append(
            f"표 제목: {title}"
        )

    if (
        part_index is not None
        and part_count is not None
        and part_count > 1
    ):
        lines.append(
            f"표 분할: {part_index}/{part_count}"
        )

    if lines:
        lines.append("")

    lines.extend(
        matrix_to_lines(matrix)
    )

    return "\n".join(
        lines
    ).strip()


def estimate_table_text_length(
    *,
    title: str,
    matrix: list[list[str]],
    corp_name: str = "",
    report_nm: str = "",
) -> int:
    """
    현재 matrix를 text로 만들었을 때의 대략적인 길이를 계산한다.
    """

    return len(
        build_table_text(
            title=title,
            matrix=matrix,
            corp_name=corp_name,
            report_nm=report_nm,
        )
    )


def split_large_table_matrix(
    *,
    matrix: list[list[str]],
    header_row_count: int,
    title: str,
    corp_name: str = "",
    report_nm: str = "",
) -> list[dict[str, Any]]:
    """
    큰 표를 header + data row 묶음으로 분할한다.

    작은 표:
        전체 matrix -> part 1개

    큰 표:
        [header rows + data rows 1..N]
        [header rows + data rows N+1..M]
        ...

    각 part마다 header를 반복한다.

    주의:
    - 행 중간은 자르지 않는다.
    - 단일 row 자체가 TABLE_MAX_CHARS보다 긴 경우에는
      데이터 유실을 막기 위해 그 row 하나를 독립 part로 보존한다.
    """

    if not matrix:
        return []

    full_length = estimate_table_text_length(
        title=title,
        matrix=matrix,
        corp_name=corp_name,
        report_nm=report_nm,
    )

    # 작은 표는 기존처럼 그대로 사용
    if full_length <= TABLE_MAX_CHARS:
        return [
            {
                "matrix": matrix,
                "source_row_start": 1,
                "source_row_end": len(matrix),
                "oversized_single_row": False,
            }
        ]

    header_row_count = max(
        0,
        min(
            header_row_count,
            len(matrix),
        ),
    )

    header_rows = matrix[
        :header_row_count
    ]

    data_rows = matrix[
        header_row_count:
    ]

    # header만 있는 특수 케이스
    if not data_rows:
        return [
            {
                "matrix": matrix,
                "source_row_start": 1,
                "source_row_end": len(matrix),
                "oversized_single_row": (
                    full_length
                    > TABLE_MAX_CHARS
                ),
            }
        ]

    parts: list[dict[str, Any]] = []

    current_rows: list[list[str]] = []

    # 원본 matrix 기준 data row 시작 번호
    data_start_row_number = (
        header_row_count + 1
    )

    current_source_start = (
        data_start_row_number
    )

    for offset, row in enumerate(
        data_rows
    ):

        source_row_number = (
            data_start_row_number
            + offset
        )

        candidate_rows = (
            header_rows
            + current_rows
            + [row]
        )

        candidate_length = (
            estimate_table_text_length(
                title=title,
                matrix=candidate_rows,
                corp_name=corp_name,
                report_nm=report_nm,
            )
        )

        # 현재 part에 row를 넣으면 MAX 초과
        if (
            current_rows
            and candidate_length
            > TABLE_MAX_CHARS
        ):

            parts.append(
                {
                    "matrix": (
                        header_rows
                        + current_rows
                    ),
                    "source_row_start": (
                        current_source_start
                    ),
                    "source_row_end": (
                        source_row_number - 1
                    ),
                    "oversized_single_row": False,
                }
            )

            current_rows = []
            current_source_start = (
                source_row_number
            )

        # 새 part에서 현재 row 하나만 넣었을 때 길이 확인
        single_row_matrix = (
            header_rows
            + [row]
        )

        single_row_length = (
            estimate_table_text_length(
                title=title,
                matrix=single_row_matrix,
                corp_name=corp_name,
                report_nm=report_nm,
            )
        )

        # row 하나 자체가 MAX보다 큰 경우:
        # 행을 잘라 의미를 깨뜨리기보다 독립 part로 보존
        if single_row_length > TABLE_MAX_CHARS:

            if current_rows:

                parts.append(
                    {
                        "matrix": (
                            header_rows
                            + current_rows
                        ),
                        "source_row_start": (
                            current_source_start
                        ),
                        "source_row_end": (
                            source_row_number - 1
                        ),
                        "oversized_single_row": False,
                    }
                )

                current_rows = []

            parts.append(
                {
                    "matrix": (
                        header_rows
                        + [row]
                    ),
                    "source_row_start": (
                        source_row_number
                    ),
                    "source_row_end": (
                        source_row_number
                    ),
                    "oversized_single_row": True,
                }
            )

            current_source_start = (
                source_row_number + 1
            )

            continue

        current_rows.append(
            row
        )

        # TARGET을 넘었으면 여기서 자연스럽게 끊는다.
        current_matrix = (
            header_rows
            + current_rows
        )

        current_length = (
            estimate_table_text_length(
                title=title,
                matrix=current_matrix,
                corp_name=corp_name,
                report_nm=report_nm,
            )
        )

        if (
            current_length
            >= TABLE_TARGET_CHARS
        ):

            parts.append(
                {
                    "matrix": current_matrix,
                    "source_row_start": (
                        current_source_start
                    ),
                    "source_row_end": (
                        source_row_number
                    ),
                    "oversized_single_row": False,
                }
            )

            current_rows = []
            current_source_start = (
                source_row_number + 1
            )

    if current_rows:

        parts.append(
            {
                "matrix": (
                    header_rows
                    + current_rows
                ),
                "source_row_start": (
                    current_source_start
                ),
                "source_row_end": (
                    len(matrix)
                ),
                "oversized_single_row": False,
            }
        )

    return parts


# ============================================================
# PARSE ALL TABLES
# ============================================================

def parse_periodic_tables(
    path: Path,
    corp_name: str = "",
    report_nm: str = "",
) -> list[dict[str, Any]]:
    """
    periodic 문서 하나에서 모든 유의미한 table을 추출한다.

    작은 표:
        표 전체를 result 1개로 유지

    큰 표:
        header + row 묶음으로 여러 result로 분할

    따라서 하나의 원본 table_index에서
    table_part_index 1..N 이 생성될 수 있다.
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

        if not is_meaningful_table(
            matrix
        ):
            continue

        title = find_table_title(
            table
        )

        header_row_count = (
            detect_header_row_count(
                table,
                matrix,
            )
        )

        headers = flatten_headers(
            matrix,
            header_row_count=max(
                1,
                header_row_count,
            ),
        )

        parts = split_large_table_matrix(
            matrix=matrix,
            header_row_count=(
                header_row_count
            ),
            title=title,
            corp_name=corp_name,
            report_nm=report_nm,
        )

        part_count = len(parts)

        for table_part_index, part in enumerate(
            parts,
            start=1,
        ):

            part_matrix = part[
                "matrix"
            ]

            text = build_table_text(
                title=title,
                matrix=part_matrix,
                corp_name=corp_name,
                report_nm=report_nm,
                part_index=table_part_index,
                part_count=part_count,
            )

            result = {
                # 원본 XML table 번호
                "table_index": (
                    table_index
                ),

                # 같은 table이 여러 chunk로 갈린 경우 사용
                "table_part_index": (
                    table_part_index
                ),
                "table_part_count": (
                    part_count
                ),
                "is_split": (
                    part_count > 1
                ),

                "title": title,
                "headers": headers,

                # 이 part에 실제 저장되는 matrix
                # header는 각 part마다 반복됨
                "rows": part_matrix,

                "header_row_count": (
                    header_row_count
                ),

                # 원본 table의 어느 data row 범위인지
                "source_row_start": part[
                    "source_row_start"
                ],
                "source_row_end": part[
                    "source_row_end"
                ],

                "oversized_single_row": part[
                    "oversized_single_row"
                ],

                "row_count": len(
                    part_matrix
                ),

                "max_column_count": max(
                    len(row)
                    for row in part_matrix
                ),

                "text_length": len(
                    text
                ),

                "text": text,
            }

            results.append(
                result
            )

    return results


# ============================================================
# PREVIEW
# ============================================================

def preview_tables(
    tables: list[dict[str, Any]],
    limit: int = 5,
) -> None:
    """
    처음 몇 개 표만 사람이 확인할 수 있게 출력.
    """

    print()
    print("=" * 100)
    print("Periodic Table Parser Preview")
    print("=" * 100)

    print(
        "총 추출 table:",
        len(tables),
    )

    for index, table in enumerate(
        tables[:limit],
        start=1,
    ):

        print()
        print("-" * 100)

        print(
            f"[TABLE {index}]"
        )

        print(
            "table_index:",
            table[
                "table_index"
            ],
        )

        print(
            "table_part:",
            f"{table.get('table_part_index', 1)}/"
            f"{table.get('table_part_count', 1)}",
        )

        print(
            "text_length:",
            table.get(
                "text_length",
                len(table.get("text", "")),
            ),
        )

        print(
            "title:",
            table[
                "title"
            ],
        )

        print(
            "row_count:",
            table[
                "row_count"
            ],
        )

        print(
            "max_column_count:",
            table[
                "max_column_count"
            ],
        )

        print(
            "headers:",
            table[
                "headers"
            ],
        )

        print()
        print("[TEXT PREVIEW]")

        text = table[
            "text"
        ]

        if len(text) > 1500:

            text = (
                text[:1500]
                + "\n..."
            )

        print(
            text
        )


# ============================================================
# SAVE JSONL
# ============================================================

def save_tables_jsonl(
    tables: list[dict[str, Any]],
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

        for table in tables:

            f.write(
                json.dumps(
                    table,
                    ensure_ascii=False,
                )
                + "\n"
            )

def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:

    records = []

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


def find_raw_files(
    manifest_record: dict[str, Any],
) -> list[Path]:

    relative_path = str(
        manifest_record.get("file_path")
        or ""
    ).strip()

    rcept_no = str(
        manifest_record.get("rcept_no")
        or ""
    ).strip()

    # 1. manifest file_path
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
                if path.is_file()
                and path.name.startswith(prefix)
            ]

            if files:
                return sorted(files)

    # 2. rcept_no fallback
    if not rcept_no:
        return []

    raw_root = (
        PROJECT_ROOT
        / "corpus"
        / "raw"
    )

    matches = []

    for path in raw_root.rglob("*"):

        if not path.is_file():
            continue

        if rcept_no in path.name:
            matches.append(path)

    return sorted(matches)

def debug_table_html(
    path: Path,
    target_table_index: int,
) -> None:
    """
    특정 table의 실제 HTML/XML 구조를 출력한다.
    """

    raw = read_document(path)

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    tables = soup.find_all(
        "table"
    )

    if (
        target_table_index < 1
        or target_table_index > len(tables)
    ):
        print(
            f"잘못된 table index: "
            f"{target_table_index}"
        )
        return

    table = tables[
        target_table_index - 1
    ]

    print()
    print("=" * 100)
    print(
        f"RAW TABLE HTML "
        f"(table_index={target_table_index})"
    )
    print("=" * 100)

    print(
        table.prettify()
    )

def debug_table_cells(
    path: Path,
    target_table_index: int,
) -> None:

    raw = read_document(path)

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    tables = soup.find_all(
        "table"
    )

    table = tables[
        target_table_index - 1
    ]

    print()
    print("=" * 100)
    print(
        f"TABLE CELL DEBUG "
        f"(table_index={target_table_index})"
    )
    print("=" * 100)

    rows = table.find_all(
        "tr"
    )

    for row_index, row in enumerate(
        rows,
        start=1,
    ):

        print()
        print(
            f"[ROW {row_index}]"
        )

        cells = row.find_all(
            [
                "td",
                "th",
                "te",
            ],
            recursive=False,
        )

        for cell_index, cell in enumerate(
            cells,
            start=1,
        ):

            print(
                f"  CELL {cell_index}"
            )

            print(
                "    text   :",
                repr(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                ),
            )

            print(
                "    rowspan:",
                cell.get(
                    "rowspan"
                ),
            )

            print(
                "    colspan:",
                cell.get(
                    "colspan"
                ),
            )

            print(
                "    html   :",
                str(cell)[:500],
            )

def flatten_headers(
    matrix: list[list[str]],
    header_row_count: int = 2,
) -> list[str]:
    """
    여러 줄 header를 column별로 합쳐서
    하나의 의미 있는 header로 만든다.

    예:
    row 1:
        구분 | 연결대상회사수 | 연결대상회사수 | ... | 주요종속회사수

    row 2:
        구분 | 기초 | 증가 | 감소 | 기말 | 주요종속회사수

    결과:
        구분
        연결대상회사수_기초
        연결대상회사수_증가
        연결대상회사수_감소
        연결대상회사수_기말
        주요종속회사수
    """

    if not matrix:
        return []

    header_rows = matrix[
        :header_row_count
    ]

    max_columns = max(
        len(row)
        for row in header_rows
    )

    flattened: list[str] = []

    for col_index in range(
        max_columns
    ):

        parts: list[str] = []

        for row in header_rows:

            if col_index >= len(row):
                continue

            value = clean_text(
                row[col_index]
            )

            if not value:
                continue

            # rowspan 때문에 같은 값이 반복되는 경우 제거
            if (
                not parts
                or parts[-1] != value
            ):
                parts.append(
                    value
                )

        flattened.append(
            "_".join(parts)
        )

    return flattened

# ============================================================
# TEST MAIN
# ============================================================

def main() -> None:

    manifest_path = (
        PROJECT_ROOT
        / "corpus"
        / "manifest.jsonl"
    )

    manifest = load_manifest(
        manifest_path
    )

    # --------------------------------------------------------
    # 실제 periodic 문서만 선택
    # --------------------------------------------------------

    candidates = [
        record
        for record in manifest
        if record.get("corp_name")
        == "HD현대일렉트릭"
        and record.get("doc_group")
        == "periodic"
    ]

    if not candidates:
        raise RuntimeError(
            "periodic 문서를 찾지 못했습니다."
        )

    # 첫 번째 periodic 문서
    record = candidates[0]

    print()
    print("=" * 100)
    print("선택된 manifest 문서")
    print("=" * 100)

    print(
        "회사:",
        record.get("corp_name"),
    )

    print(
        "공시명:",
        record.get("report_nm"),
    )

    print(
        "subtype:",
        record.get("doc_subtype"),
    )

    print(
        "rcept_no:",
        record.get("rcept_no"),
    )

    print(
        "file_path:",
        record.get("file_path"),
    )

    raw_files = find_raw_files(
        record
    )

    if not raw_files:
        raise RuntimeError(
            "실제 raw 파일을 찾지 못했습니다."
        )

    print()
    print(
        "발견 raw files:"
    )

    for path in raw_files:
        print(
            " -",
            path,
        )

    # --------------------------------------------------------
    # annual은 여러 raw 파일이 있을 수 있으므로 전부 검사
    # --------------------------------------------------------

    all_tables = []

    for raw_file in raw_files:

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

        all_tables.extend(
            tables
        )

    preview_tables(
        all_tables,
        limit=5,
    )

    debug_table_html(
        raw_files[0],
        target_table_index=7,
    )

    debug_table_cells(
        raw_files[0],
        target_table_index=7,
    )

if __name__ == "__main__":
    main()