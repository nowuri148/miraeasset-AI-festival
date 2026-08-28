from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 같은 폴더의 event_chunk_builder_ver2.py를 import
from event_chunk_builder_ver2 import (
    MANIFEST_PATH,
    DEFAULT_HEADER_ROW_COUNT,
    build_event_note_chunks,
    build_table_chunks,
    clean_text,
    is_searchable_event_table,
    load_manifest,
    matrix_to_text,
    parse_event_document,
    split_large_event_table,
)

def find_manifest_record(
    manifest: list[dict[str, Any]],
    xml_path: Path,
    expected_group: str,
) -> dict[str, Any]:
    """
    파일명(stem)을 rcept_no로 간주해 manifest에서 찾는다.
    못 찾으면 최소 metadata record를 만들어 테스트는 계속한다.
    """

    stem = xml_path.stem

    for record in manifest:

        rcept_no = str(
            record.get("rcept_no")
            or ""
        ).strip()

        doc_group = str(
            record.get("doc_group")
            or ""
        ).strip()

        if (
            rcept_no == stem
            and doc_group == expected_group
        ):
            return record

    print(
        f"[WARN] manifest에서 {expected_group} / {stem} 레코드를 찾지 못했습니다."
    )
    print(
        "       최소 metadata로 구조 테스트만 진행합니다."
    )

    return {
        "doc_id": stem,
        "rcept_no": stem,
        "doc_group": expected_group,
        "corp_name": "",
        "report_nm": xml_path.name,
        "normalized_report_type": "",
        "event_date": "",
        "is_correction": False,
    }


def build_explicit_file_chunks(
    record: dict[str, Any],
    xml_path: Path,
) -> list[dict[str, Any]]:

    chunks: list[
        dict[str, Any]
    ] = []

    parsed_tables = (
        parse_event_document(
            xml_path
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

    # ========================================================
    # MAJOR / HOLDING TABLE SPLIT
    # ========================================================

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
                                part_count > 1
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

            # exchange는 원본 table 그대로
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

    # ========================================================
    # TABLE / FIELD-VALUE BUILD
    # ========================================================

    for parsed_table in expanded_tables:

        if not (
            is_searchable_event_table(
                parsed_table
            )
        ):
            continue

        chunks.extend(
            build_table_chunks(
                record=record,
                parsed_table=parsed_table,
                raw_file=xml_path,
                raw_file_index=1,
            )
        )

    # ========================================================
    # EVENT NOTE
    # ========================================================

    if doc_group in (
        "major",
        "holding",
    ):

        chunks.extend(
            build_event_note_chunks(
                record=record,
                raw_file=xml_path,
                raw_file_index=1,
            )
        )

    return chunks

def validate_one(
    label: str,
    record: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:

    ids = [
        str(
            chunk.get(
                "chunk_id"
            )
            or ""
        )
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count
        in Counter(ids).items()
        if count > 1
    ]

    strategy_counts = Counter(
        str(
            (
                chunk.get("metadata")
                or {}
            ).get(
                "chunk_strategy"
            )
            or "unknown"
        )
        for chunk in chunks
    )

    lengths = [
        len(
            str(
                chunk.get(
                    "text"
                )
                or ""
            )
        )
        for chunk in chunks
    ]

    note_chunks = [
        chunk
        for chunk in chunks
        if (
            (
                chunk.get("metadata")
                or {}
            ).get(
                "chunk_strategy"
            )
            == "event_note"
        )
    ]

    expected_note = (
        label
        in (
            "major",
            "holding",
        )
    )

    note_rule_ok = (
        bool(note_chunks)
        if expected_note
        else not note_chunks
    )

    result = {
        "label": label,
        "rcept_no": record.get(
            "rcept_no"
        ),
        "corp_name": record.get(
            "corp_name"
        ),
        "report_nm": record.get(
            "report_nm"
        ),
        "total_chunks": len(
            chunks
        ),
        "strategies": dict(
            strategy_counts
        ),
        "event_note_chunks": len(
            note_chunks
        ),
        "duplicate_chunk_ids": len(
            duplicate_ids
        ),
        "min_text_length": (
            min(lengths)
            if lengths
            else 0
        ),
        "max_text_length": (
            max(lengths)
            if lengths
            else 0
        ),
        "event_note_rule_ok": (
            note_rule_ok
        ),
        "pass": (
            len(duplicate_ids) == 0
            and note_rule_ok
        ),
    }

    return result


def print_preview(
    label: str,
    chunks: list[dict[str, Any]],
    limit: int,
) -> None:

    print()
    print("=" * 110)
    print(
        f"[{label.upper()}] CHUNK PREVIEW"
    )
    print("=" * 110)

    strategy_counts = Counter(
        str(
            (
                chunk.get("metadata")
                or {}
            ).get(
                "chunk_strategy"
            )
            or "unknown"
        )
        for chunk in chunks
    )

    print(
        "total:",
        len(chunks),
    )

    print(
        "strategies:",
        dict(strategy_counts),
    )

    print()
    print(
        "[EVENT NOTE]"
    )

    notes = [
        chunk
        for chunk in chunks
        if (
            (
                chunk.get("metadata")
                or {}
            ).get(
                "chunk_strategy"
            )
            == "event_note"
        )
    ]

    if not notes:

        print(
            "(none)"
        )

    for chunk in notes[:limit]:

        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        print()
        print(
            "-" * 110
        )

        print(
            "chunk_id:",
            chunk.get(
                "chunk_id"
            )
        )

        print(
            "section:",
            metadata.get(
                "section_path"
            )
        )

        print(
            "length:",
            len(
                str(
                    chunk.get(
                        "text"
                    )
                    or ""
                )
            )
        )

        print()

        print(
            str(
                chunk.get(
                    "text"
                )
                or ""
            )[:1600]
        )

    print()
    print(
        "[TABLE / FIELD-VALUE SAMPLE]"
    )

    normal = [
        chunk
        for chunk in chunks
        if (
            (
                chunk.get("metadata")
                or {}
            ).get(
                "chunk_strategy"
            )
            != "event_note"
        )
    ]

    for chunk in normal[:limit]:

        metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        print()
        print(
            "-" * 110
        )

        print(
            "strategy:",
            metadata.get(
                "chunk_strategy"
            )
        )

        print(
            "chunk_id:",
            chunk.get(
                "chunk_id"
            )
        )

        print(
            "title:",
            metadata.get(
                "table_title"
            )
        )

        print(
            "length:",
            len(
                str(
                    chunk.get(
                        "text"
                    )
                    or ""
                )
            )
        )

        print()

        print(
            str(
                chunk.get(
                    "text"
                )
                or ""
            )[:1200]
        )


def write_csv(
    all_rows: list[dict[str, Any]],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "doc_group",
        "chunk_no",
        "chunk_id",
        "chunk_strategy",
        "chunk_type",
        "search_priority",
        "section_path",
        "table_title",
        "text_length",
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

        for row in all_rows:
            writer.writerow(
                row
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Test major / holding / exchange "
            "with event_chunk_builder_ver2.py"
        )
    )

    parser.add_argument(
        "--major",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--holding",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--exchange",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "event_chunk_test_preview.csv"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    for path in (
        args.major,
        args.holding,
        args.exchange,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    manifest = load_manifest(
        MANIFEST_PATH
    )

    inputs = [
        (
            "major",
            args.major,
        ),
        (
            "holding",
            args.holding,
        ),
        (
            "exchange",
            args.exchange,
        ),
    ]

    all_csv_rows: list[
        dict[str, Any]
    ] = []

    validation_results = []

    for label, xml_path in inputs:

        record = find_manifest_record(
            manifest=manifest,
            xml_path=xml_path,
            expected_group=label,
        )

        chunks = build_explicit_file_chunks(
            record=record,
            xml_path=xml_path,
        )

        print_preview(
            label=label,
            chunks=chunks,
            limit=args.limit,
        )

        result = validate_one(
            label=label,
            record=record,
            chunks=chunks,
        )

        validation_results.append(
            result
        )

        for chunk_no, chunk in enumerate(
            chunks,
            start=1,
        ):

            metadata = (
                chunk.get(
                    "metadata"
                )
                or {}
            )

            text = str(
                chunk.get(
                    "text"
                )
                or ""
            )

            all_csv_rows.append(
                {
                    "doc_group":
                        label,

                    "chunk_no":
                        chunk_no,

                    "chunk_id":
                        chunk.get(
                            "chunk_id"
                        ),

                    "chunk_strategy":
                        metadata.get(
                            "chunk_strategy"
                        ),

                    "chunk_type":
                        metadata.get(
                            "chunk_type"
                        ),

                    "search_priority":
                        metadata.get(
                            "search_priority"
                        ),

                    "section_path":
                        metadata.get(
                            "section_path"
                        ),

                    "table_title":
                        metadata.get(
                            "table_title"
                        ),

                    "text_length":
                        len(text),

                    "text":
                        text,
                }
            )

    write_csv(
        all_csv_rows,
        args.output,
    )

    print()
    print("=" * 110)
    print(
        "VALIDATION SUMMARY"
    )
    print("=" * 110)

    for result in validation_results:

        print()
        print(
            f"[{result['label'].upper()}]"
        )

        print(
            "rcept_no:",
            result[
                "rcept_no"
            ],
        )

        print(
            "total_chunks:",
            result[
                "total_chunks"
            ],
        )

        print(
            "strategies:",
            result[
                "strategies"
            ],
        )

        print(
            "event_note_chunks:",
            result[
                "event_note_chunks"
            ],
        )

        print(
            "duplicate_chunk_ids:",
            result[
                "duplicate_chunk_ids"
            ],
        )

        print(
            "text length:",
            result[
                "min_text_length"
            ],
            "~",
            result[
                "max_text_length"
            ],
        )

        print(
            "event_note rule:",
            (
                "PASS"
                if result[
                    "event_note_rule_ok"
                ]
                else "FAIL"
            ),
        )

        print(
            "result:",
            (
                "PASS"
                if result[
                    "pass"
                ]
                else "FAIL"
            ),
        )

    overall_pass = all(
        result[
            "pass"
        ]
        for result
        in validation_results
    )

    print()
    print(
        "CSV:",
        args.output.resolve(),
    )

    print()
    print(
        "OVERALL:",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    )

    if not overall_pass:
        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
