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

from z_base_function import (
    COMPARE_TYPES,
    ClovaClient,
    enrich_records_from_manifest,
    generate_comparison_dataset,
    is_index_record,
    load_manifest_index,
    split_records,
    load_input_records
)


# ============================================================
# USER SETTINGS
# ============================================================

# 1 = 같은 회사 + 같은 공시 유형 + 다른 시점
# 2 = 같은 사건의 원공시/정정공시
# 3 = 같은 회사 + 같은 metric + 다른 분기/연도
# 4 = 다른 회사 + 같은 공시 유형 + 비슷한 기간
COMPARE_TYPE = 1

# 한 질문 Context에서 비교할 문서 개수: 2 / 3 / 4
DOCUMENT_COUNT = 2

# 비교 group 하나당 만들 QA 개수
QUESTIONS_PER_GROUP = 4

# 4번 비교기준에서 "비슷한 기간"으로 허용할 최대 날짜 차이
SIMILAR_PERIOD_DAYS = 90

# 테스트 시 split당 group 개수 제한.
# 0이면 전부 생성.
MAX_GROUPS_PER_SPLIT = 0

# 원본 corpus
INPUT_PATH = cfg.DATASET_INPUT_PATH

# 비교용 데이터셋 출력 폴더
OUTPUT_DIR = PROJECT_ROOT / "Dataset_compare"

# metadata manifest.
# normalized_report_type / disclosure_chain_id 등을 가져오기 위해 사용.
MANIFEST_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
    / "manifest_v3_reviewed.jsonl"
)

# 별도 cache 사용 권장
CACHE_PATH = (
    OUTPUT_DIR
    / ".qa_compare_generation_cache.jsonl"
)


# ============================================================
# END USER SETTINGS
# ============================================================


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if COMPARE_TYPE not in COMPARE_TYPES:
        raise ValueError(
            "COMPARE_TYPE must be 1, 2, 3, or 4"
        )

    if DOCUMENT_COUNT not in {2, 3, 4}:
        raise ValueError(
            "DOCUMENT_COUNT must be 2, 3, or 4"
        )

    compare_type = COMPARE_TYPES[
        COMPARE_TYPE
    ]

    print("=" * 70)
    print("Comparison dataset generation")
    print(f"COMPARE_TYPE   : {COMPARE_TYPE} -> {compare_type}")
    print(f"DOCUMENT_COUNT : {DOCUMENT_COUNT}")
    print(f"QA / GROUP     : {QUESTIONS_PER_GROUP}")
    print(f"INPUT          : {INPUT_PATH}")
    print(f"MANIFEST       : {MANIFEST_PATH}")
    print(f"OUTPUT         : {OUTPUT_DIR}")
    print("=" * 70)

    # 1. raw documents
    records = load_input_records(
        INPUT_PATH
    )

    if not getattr(
        cfg,
        "INCLUDE_INDEX_RECORDS",
        False,
    ):
        records = [
            record
            for record in records
            if not is_index_record(record)
        ]

    print(
        f"Loaded raw records: {len(records)}"
    )

    # 2. enrich raw documents with manifest metadata
    manifest_index = load_manifest_index(
        MANIFEST_PATH
    )

    print(
        f"Manifest index: {len(manifest_index)}"
    )

    records = enrich_records_from_manifest(
        records,
        manifest_index,
    )

    # 3. split documents BEFORE comparison grouping
    #    so a single comparison Context cannot mix train/test docs.
    split_map = split_records(
        records,
        split_mode=cfg.DATASET_SPLIT_MODE,
        split_seed=cfg.SPLIT_SEED,
        hash_ratios=tuple(
            cfg.HASH_SPLIT_RATIOS
        ),
        train_end_year=cfg.TRAIN_END_YEAR,
        validation_years=set(
            cfg.VALIDATION_YEARS
        ),
        test_start_year=cfg.TEST_START_YEAR,
    )

    for split_name, items in split_map.items():
        print(
            f"{split_name}: "
            f"{len(items)} documents"
        )

    if not cfg.CLOVA_STUDIO_API_KEY:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY is empty in Config.py"
        )

    # 4. HCX client
    client = ClovaClient(
        cfg.CLOVA_STUDIO_API_KEY,
        cfg.CLOVA_BASE_URL,
        cfg.CLOVA_MODEL,
        cfg.API_TIMEOUT_SECONDS,
        cfg.API_RETRIES,
    )

    # 5. comparison groups -> HCX QA -> JSON + CSV
    generate_comparison_dataset(
        split_map=split_map,
        output_dir=OUTPUT_DIR,
        compare_type=compare_type,
        document_count=DOCUMENT_COUNT,
        client=client,
        questions_per_group=QUESTIONS_PER_GROUP,
        max_context_chars=cfg.MAX_CONTEXT_CHARS,
        max_tokens=cfg.MAX_GENERATION_TOKENS,
        temperature=cfg.GENERATION_TEMPERATURE,
        system_prompt=cfg.TUNING_SYSTEM_PROMPT,
        cache_path=CACHE_PATH,
        similar_period_days=SIMILAR_PERIOD_DAYS,
        max_groups_per_split=MAX_GROUPS_PER_SPLIT,
    )


if __name__ == "__main__":
    main()