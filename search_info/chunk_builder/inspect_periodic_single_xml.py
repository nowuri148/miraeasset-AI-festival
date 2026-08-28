from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from periodic_narrative_chunk_builder import (
    make_narrative_chunks,
)

# 실제 production table parser 사용
from periodic_table_parser_split import (
    parse_periodic_tables,
)


# ============================================================
# TABLE PREVIEW
# ============================================================

def make_table_preview_chunks(
    xml_path: Path,
) -> list[dict[str, Any]]:
    """
    실제 periodic_table_parser.py 결과를 그대로 사용한다.

    따라서:
    - rowspan / colspan 복원
    - table title
    - header
    - 대형 table split
    - table_part_index
    - table_part_count

    모두 production 코드와 동일하게 확인할 수 있다.
    """

    # --------------------------------------------------------
    # narrative builder와 별개로
    # table parser에는 필요한 최소 context만 전달
    # --------------------------------------------------------

    from periodic_narrative_chunk_builder import (
        extract_document_meta,
    )

    meta = extract_document_meta(
        xml_path
    )

    corp_name = str(
        meta.get("corp_name")
        or ""
    )

    report_nm = str(
        meta.get("document_name")
        or ""
    )

    tables = parse_periodic_tables(
        path=xml_path,
        corp_name=corp_name,
        report_nm=report_nm,
    )

    rows: list[dict[str, Any]] = []

    for table in tables:

        table_index = int(
            table.get(
                "table_index",
                0,
            )
            or 0
        )

        table_part_index = int(
            table.get(
                "table_part_index",
                1,
            )
            or 1
        )

        table_part_count = int(
            table.get(
                "table_part_count",
                1,
            )
            or 1
        )

        title = str(
            table.get(
                "title",
                "",
            )
            or ""
        )

        text = str(
            table.get(
                "text",
                "",
            )
            or ""
        )

        # ----------------------------------------------------
        # preview용 chunk id
        # ----------------------------------------------------

        chunk_id = (
            f"periodic_"
            f"{meta.get('rcept_no') or xml_path.stem}"
            f"_table_{table_index:04d}"
            f"_part_{table_part_index:03d}"
        )

        rows.append(
            {
                "chunk_id": chunk_id,

                "chunk_type": "table",

                "section_or_title": title,

                "text": text,

                "text_length": len(
                    text
                ),

                "row_count": table.get(
                    "row_count",
                    "",
                ),

                # 원본 XML table index
                "source_index": table_index,

                # 새 table split 확인용
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
                    "header_row_count",
                    "",
                ),

                "source_row_start": table.get(
                    "source_row_start",
                    "",
                ),

                "source_row_end": table.get(
                    "source_row_end",
                    "",
                ),

                "oversized_single_row": table.get(
                    "oversized_single_row",
                    False,
                ),
            }
        )

    return rows


# ============================================================
# NARRATIVE PREVIEW
# ============================================================

def narrative_rows(
    xml_path: Path,
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for chunk in make_narrative_chunks(
        xml_path
    ):

        metadata = chunk[
            "metadata"
        ]

        rows.append(
            {
                "chunk_id": chunk[
                    "chunk_id"
                ],

                "chunk_type": (
                    "narrative"
                ),

                "section_or_title": (
                    metadata.get(
                        "section_path",
                        "",
                    )
                ),

                "text": chunk[
                    "text"
                ],

                "text_length": len(
                    chunk["text"]
                ),

                "row_count": "",

                "source_index": (
                    metadata.get(
                        "narrative_index",
                        "",
                    )
                ),

                # table 전용 컬럼은 빈 값
                "table_part_index": "",
                "table_part_count": "",
                "is_split": "",
                "header_row_count": "",
                "source_row_start": "",
                "source_row_end": "",
                "oversized_single_row": "",
            }
        )

    return rows


# ============================================================
# SAVE CSV
# ============================================================

def write_csv(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "chunk_no",
        "chunk_type",
        "chunk_id",
        "section_or_title",
        "text_length",
        "row_count",
        "source_index",

        # table split 확인용
        "table_part_index",
        "table_part_count",
        "is_split",
        "header_row_count",
        "source_row_start",
        "source_row_end",
        "oversized_single_row",

        "text",
    ]

    with output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i, row in enumerate(
            rows,
            start=1,
        ):

            writer.writerow(
                {
                    "chunk_no": i,
                    **row,
                }
            )


# ============================================================
# STATS
# ============================================================

def print_stats(
    table_rows: list[dict[str, Any]],
    narrative_rows_: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 100)
    print("CHUNK STATISTICS")
    print("=" * 100)

    print(
        "table chunks:",
        len(table_rows),
    )

    print(
        "narrative chunks:",
        len(narrative_rows_),
    )

    if table_rows:

        table_lengths = [
            int(
                row.get(
                    "text_length",
                    0,
                )
                or 0
            )
            for row in table_rows
        ]

        split_rows = [
            row
            for row in table_rows
            if row.get(
                "is_split"
            )
        ]

        oversized_rows = [
            row
            for row in table_rows
            if row.get(
                "oversized_single_row"
            )
        ]

        print()
        print("[TABLE]")

        print(
            "min text length:",
            min(table_lengths),
        )

        print(
            "max text length:",
            max(table_lengths),
        )

        print(
            "avg text length:",
            round(
                sum(table_lengths)
                / len(table_lengths),
                1,
            ),
        )

        print(
            "split table chunks:",
            len(split_rows),
        )

        print(
            "oversized single-row chunks:",
            len(oversized_rows),
        )

        over_1500 = [
            length
            for length in table_lengths
            if length > 1500
        ]

        print(
            "> 1500 chars:",
            len(over_1500),
        )

    if narrative_rows_:

        narrative_lengths = [
            int(
                row.get(
                    "text_length",
                    0,
                )
                or 0
            )
            for row in narrative_rows_
        ]

        print()
        print("[NARRATIVE]")

        print(
            "min text length:",
            min(narrative_lengths),
        )

        print(
            "max text length:",
            max(narrative_lengths),
        )

        print(
            "avg text length:",
            round(
                sum(narrative_lengths)
                / len(narrative_lengths),
                1,
            ),
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect one periodic XML "
            "using production table + narrative chunkers"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "periodic_chunk_preview.csv"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():

        raise FileNotFoundError(
            args.input
        )

    # --------------------------------------------------------
    # 실제 production table parser
    # --------------------------------------------------------

    table_rows = (
        make_table_preview_chunks(
            args.input
        )
    )

    # --------------------------------------------------------
    # narrative builder
    # --------------------------------------------------------

    narrative = narrative_rows(
        args.input
    )

    rows = [
        *table_rows,
        *narrative,
    ]

    write_csv(
        rows,
        args.output,
    )

    print()
    print("=" * 100)
    print(
        "Periodic Single XML Chunk Preview"
    )
    print("=" * 100)

    print(
        "input:",
        args.input,
    )

    print(
        "csv:",
        args.output.resolve(),
    )

    print_stats(
        table_rows,
        narrative,
    )

    # --------------------------------------------------------
    # 분할된 table sample
    # --------------------------------------------------------

    split_samples = [
        row
        for row in table_rows
        if row.get(
            "is_split"
        )
    ]

    print()
    print("[SPLIT TABLE SAMPLE]")

    for row in split_samples[:5]:

        print()
        print("-" * 100)

        print(
            "table:",
            row[
                "source_index"
            ],
        )

        print(
            "part:",
            f"{row['table_part_index']}"
            f"/{row['table_part_count']}",
        )

        print(
            "source rows:",
            f"{row['source_row_start']}"
            f"~{row['source_row_end']}",
        )

        print(
            "text length:",
            row[
                "text_length"
            ],
        )

        print()

        print(
            row["text"][:1500]
        )

    # --------------------------------------------------------
    # narrative sample
    # --------------------------------------------------------

    print()
    print("[NARRATIVE SAMPLE]")

    for row in narrative[:3]:

        print()
        print("-" * 100)

        print(
            "section:",
            row[
                "section_or_title"
            ],
        )

        print(
            row["text"][:1200]
        )


if __name__ == "__main__":
    main()