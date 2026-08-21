from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import Config as cfg

from a_dataset_single import load_input_records
from a_dataset_single import (
    ClovaClient,
    ComparisonGroup,
    comparison_group_to_bundle,
    generate_comparison_dataset,
    record_company,
    record_identity,
    record_report_type,
    write_group_json,
)


# ============================================================
# USER SETTINGS
# ============================================================

# 비교할 파일을 직접 지정.
# 반드시 2~4개 파일을 넣는다.
FILE_PATHS = [
    r"C:\Users\User\.vscode\mirea_asset\corpus\raw\exchange\회사명\접수번호1\문서1.xml",
    r"C:\Users\User\.vscode\mirea_asset\corpus\raw\exchange\회사명\접수번호2\문서2.xml",
]

# 테스트용 비교 유형 라벨.
# pairing에는 사용하지 않고 prompt/context metadata 용도로만 사용.
COMPARE_TYPE = "manual_compare"

# 생성할 QA 개수
QUESTIONS_PER_GROUP = 4

# Context 최대 길이
MAX_CONTEXT_CHARS = cfg.MAX_CONTEXT_CHARS

# 결과 저장 위치
OUTPUT_JSON = (
    PROJECT_ROOT
    / "Dataset_compare"
    / "quick_test_result.json"
)

# quick test 전용 cache
CACHE_PATH = (
    PROJECT_ROOT
    / "Dataset_compare"
    / ".qa_compare_quick_test_cache.jsonl"
)


# ============================================================
# END USER SETTINGS
# ============================================================


def load_one_document(path: Path) -> dict:
    records = load_input_records(path)

    if not records:
        raise RuntimeError(
            f"No usable records found: {path}"
        )

    if len(records) > 1:
        print(
            f"[WARN] {path} returned {len(records)} records. "
            "Only the first record will be used."
        )

    return records[0]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if not 2 <= len(FILE_PATHS) <= 4:
        raise ValueError(
            "FILE_PATHS must contain 2 to 4 files."
        )

    if not cfg.CLOVA_STUDIO_API_KEY:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY is empty in Config.py"
        )

    print("=" * 70)
    print("Quick comparison dataset test")
    print(f"documents      : {len(FILE_PATHS)}")
    print(f"compare type   : {COMPARE_TYPE}")
    print(f"QA count       : {QUESTIONS_PER_GROUP}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 직접 지정한 파일 로드
    # --------------------------------------------------------
    records = []

    for index, raw_path in enumerate(
        FILE_PATHS,
        start=1,
    ):
        path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        record = load_one_document(path)
        records.append(record)

        print(
            f"[문서 {index}] "
            f"company={record_company(record)} | "
            f"type={record_report_type(record)} | "
            f"id={record_identity(record)}"
        )

    # --------------------------------------------------------
    # 2. 수동 ComparisonGroup 생성
    # --------------------------------------------------------
    group = ComparisonGroup(
        group_id="manual_quick_test",
        compare_type=COMPARE_TYPE,
        records=records,
        metadata={
            "manual": True,
            "file_paths": FILE_PATHS,
        },
    )

    # --------------------------------------------------------
    # 3. 2~4개 문서를 하나의 비교 Context로 변환
    # --------------------------------------------------------
    bundle = comparison_group_to_bundle(
        group,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

    print("\n" + "=" * 70)
    print("Generated comparison context")
    print("=" * 70)
    print(bundle.context)
    print("=" * 70)

    # --------------------------------------------------------
    # 4. HCX client
    # --------------------------------------------------------
    client = ClovaClient(
        cfg.CLOVA_STUDIO_API_KEY,
        cfg.CLOVA_BASE_URL,
        cfg.CLOVA_MODEL,
        cfg.API_TIMEOUT_SECONDS,
        cfg.API_RETRIES,
    )

    # --------------------------------------------------------
    # 5. 기존 2-stage HCX build_rows 재사용
    # --------------------------------------------------------
    from a_dataset_compare import build_rows

    rows, errors = build_rows(
        [bundle],
        client,
        single_count=0,
        multi_count=QUESTIONS_PER_GROUP,
        max_tokens=cfg.MAX_GENERATION_TOKENS,
        temperature=cfg.GENERATION_TEMPERATURE,
        include_context=True,
        system_prompt=cfg.TUNING_SYSTEM_PROMPT,
        cache_path=CACHE_PATH,
        limit=0,
    )

    # --------------------------------------------------------
    # 6. 콘솔 출력
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Generated QA rows: {len(rows)}")
    print("=" * 70)

    for index, row in enumerate(rows, start=1):
        print(f"\n[QA {index}]")
        print(row["Text"])
        print("\n[Completion]")
        print(row["Completion"])
        print("-" * 70)

    if errors:
        print("\n[Errors / Warnings]")
        for error in errors:
            print("-", error)

    # --------------------------------------------------------
    # 7. JSON 저장
    # --------------------------------------------------------
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_group_json(
        OUTPUT_JSON,
        split_name="quick_test",
        group=group,
        rows=rows,
        errors=errors,
    )

    print(
        f"\nSaved quick-test result: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()