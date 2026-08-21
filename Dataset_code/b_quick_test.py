from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Config as cfg

# ------------------------------------------------------------
# 공통 기능: z_base_function.py
# ------------------------------------------------------------
from z_base_function import (
    ClovaClient,
    build_rows,
    load_input_records,
    normalize_qas_container,
    response_schema,
    training_completion,
    training_text,
    validate_comparison_qas,
)

# ------------------------------------------------------------
# 비교 데이터셋 전용 기능: b_dataset_multi.py
# ------------------------------------------------------------
from b_dataset_multi import (
    COMPARE_GENERATOR_SYSTEM_PROMPT,
    COMPARE_REVIEWER_SYSTEM_PROMPT,
    ComparisonGroup,
    comparison_generation_prompt,
    comparison_group_to_bundle,
    comparison_review_prompt,
    record_company,
    record_identity,
    record_report_type,
    write_group_json,
)


# ============================================================
# USER SETTINGS
# ============================================================
# 여기를 직접 수정해서 테스트한다.
#
# 반드시 서로 비교할 2~4개 XML / JSON / JSONL 파일을 넣는다.
# ============================================================

FILE_PATHS = [
    r"C:\mirae\miraeasset-AI-festival\corpus\raw\exchange\HD현대일렉트릭\20230131800162\20230131800162.xml",
    r"C:\mirae\miraeasset-AI-festival\corpus\raw\exchange\HD현대일렉트릭\20230911800103\20230911800103.xml",
    # r"C:\mirae\miraeasset-AI-festival\corpus\raw\exchange\LG유플러스\20250429800933\20250429800933.xml",
    # r"C:\mirae\miraeasset-AI-festival\corpus\raw\exchange\LG이노텍\20240220800842\20240220800842.xml"
]

# 수동 quick test에서는 grouping 로직에 사용하지 않고
# Context metadata / 결과 JSON 표시용 라벨로만 사용한다.
COMPARE_TYPE = "manual_compare"

# 최종 생성할 비교 QA 개수
QUESTIONS_PER_GROUP = 4

# 비교 Context 최대 길이
MAX_CONTEXT_CHARS = getattr(
    cfg,
    "MAX_CONTEXT_CHARS",
    9000,
)

# HCX 생성 설정
MAX_GENERATION_TOKENS = getattr(
    cfg,
    "MAX_GENERATION_TOKENS",
    4096,
)

GENERATION_TEMPERATURE = getattr(
    cfg,
    "GENERATION_TEMPERATURE",
    0.65,
)

CLOVA_BASE_URL = getattr(
    cfg,
    "CLOVA_BASE_URL",
    "https://clovastudio.stream.ntruss.com/v1/openai",
)

CLOVA_MODEL = getattr(
    cfg,
    "CLOVA_MODEL",
    "HCX-005",
)

API_TIMEOUT_SECONDS = getattr(
    cfg,
    "API_TIMEOUT_SECONDS",
    120,
)

API_RETRIES = getattr(
    cfg,
    "API_RETRIES",
    3,
)

TUNING_SYSTEM_PROMPT = getattr(
    cfg,
    "TUNING_SYSTEM_PROMPT",
    "",
)

# 결과 저장 위치
OUTPUT_JSON = (
    PROJECT_ROOT
    / "Dataset"
    / "quick_multi_test_result.json"
)

# Quick test 전용 cache
CACHE_PATH = (
    PROJECT_ROOT
    / "Dataset"
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
            f"[WARN] {path} returned "
            f"{len(records)} records. "
            "Only the first record will be used."
        )

    return records[0]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # --------------------------------------------------------
    # 0. 입력 확인
    # --------------------------------------------------------
    if not 2 <= len(FILE_PATHS) <= 4:
        raise ValueError(
            "FILE_PATHS must contain 2 to 4 files."
        )

    if not getattr(
        cfg,
        "CLOVA_STUDIO_API_KEY",
        "",
    ):
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY is empty in Config.py"
        )

    print("=" * 70)
    print("Quick comparison dataset test")
    print(f"documents      : {len(FILE_PATHS)}")
    print(f"compare type   : {COMPARE_TYPE}")
    print(f"QA count       : {QUESTIONS_PER_GROUP}")
    print(f"context chars  : {MAX_CONTEXT_CHARS}")
    print(f"cache          : {CACHE_PATH}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 직접 지정한 파일 로드
    # --------------------------------------------------------
    records: list[dict] = []

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
        CLOVA_BASE_URL,
        CLOVA_MODEL,
        API_TIMEOUT_SECONDS,
        API_RETRIES,
    )

    # --------------------------------------------------------
    # 5. 비교 QA 생성
    #
    # 공통 실행 엔진은 z_base_function.build_rows를 사용하고,
    # 비교 전용 prompt는 b_dataset_multi에서 주입한다.
    # --------------------------------------------------------
    rows, errors = build_rows(
        [bundle],
        client,
        single_count=0,
        multi_count=QUESTIONS_PER_GROUP,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        include_context=True,
        system_prompt=TUNING_SYSTEM_PROMPT,
        cache_path=CACHE_PATH,
        limit=0,

        # 비교 전용 prompt
        generation_prompt_fn=comparison_generation_prompt,
        review_prompt_fn=comparison_review_prompt,

        # 공통 QA 처리/검증
        normalize_qas_fn=normalize_qas_container,
        validate_reviewed_fn=validate_comparison_qas,
        training_text_fn=training_text,
        training_completion_fn=training_completion,

        # 비교 전용 HCX system prompt
        generator_system_prompt=COMPARE_GENERATOR_SYSTEM_PROMPT,
        reviewer_system_prompt=COMPARE_REVIEWER_SYSTEM_PROMPT,

        # 공통 response schema
        response_schema=response_schema(),

        # single cache와 완전히 분리된 compare quick-test namespace
        generator_cache_version="GENERATOR_COMPARE_QUICK_V1",
        reviewer_cache_version="REVIEWER_COMPARE_QUICK_V1",
    )

    # --------------------------------------------------------
    # 6. 콘솔 출력
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Generated QA rows: {len(rows)}")
    print("=" * 70)

    for index, row in enumerate(
        rows,
        start=1,
    ):
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
        f"\nSaved quick-test result: "
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()