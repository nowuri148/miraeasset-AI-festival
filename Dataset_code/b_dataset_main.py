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
    ClovaClient,
    load_input_records,
)

from b_dataset_multi import (
    COMPARE_TYPES,
    generate_comparison_dataset,
    enrich_records_from_manifest,
    is_index_record,
    load_manifest_index,
    split_records,
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

# 비교 group 하나당 목표 QA 개수
QUESTIONS_PER_GROUP = 4

# 4번 비교기준에서 "비슷한 기간"으로 허용할 최대 날짜 차이
SIMILAR_PERIOD_DAYS = 90

# 테스트 시 split당 group 개수 제한
# 0이면 전부 생성
MAX_GROUPS_PER_SPLIT = 10

# 원본 corpus
INPUT_PATH = cfg.DATASET_INPUT_PATH

# 비교용 데이터셋 출력 폴더
OUTPUT_DIR = PROJECT_ROOT / "Dataset_compare"

# metadata manifest
MANIFEST_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
    / "manifest_v3_reviewed.jsonl"
)

# 비교 전용 cache
CACHE_PATH = (
    OUTPUT_DIR
    / ".qa_compare_generation_cache.jsonl"
)


# ============================================================
# END USER SETTINGS
# ============================================================


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(
            encoding="utf-8"
        )

    if COMPARE_TYPE not in COMPARE_TYPES:
        raise ValueError(
            "COMPARE_TYPE must be 1, 2, 3, or 4"
        )

    if DOCUMENT_COUNT not in {2, 3, 4}:
        raise ValueError(
            "DOCUMENT_COUNT must be 2, 3, or 4"
        )

    if QUESTIONS_PER_GROUP <= 0:
        raise ValueError(
            "QUESTIONS_PER_GROUP must be > 0"
        )

    if MAX_GROUPS_PER_SPLIT < 0:
        raise ValueError(
            "MAX_GROUPS_PER_SPLIT must be >= 0"
        )

    compare_type = COMPARE_TYPES[
        COMPARE_TYPE
    ]

    print("=" * 70)
    print("Comparison dataset generation")
    print(
        f"COMPARE_TYPE       : "
        f"{COMPARE_TYPE} -> {compare_type}"
    )
    print(
        f"DOCUMENT_COUNT     : "
        f"{DOCUMENT_COUNT}"
    )
    print(
        f"QA / GROUP         : "
        f"{QUESTIONS_PER_GROUP}"
    )
    print(
        f"GROUP LIMIT/SPLIT  : "
        f"{MAX_GROUPS_PER_SPLIT}"
    )
    print(
        f"SIMILAR PERIOD     : "
        f"{SIMILAR_PERIOD_DAYS} days"
    )
    print(
        f"INPUT              : "
        f"{INPUT_PATH}"
    )
    print(
        f"MANIFEST           : "
        f"{MANIFEST_PATH}"
    )
    print(
        f"OUTPUT             : "
        f"{OUTPUT_DIR}"
    )
    print(
        f"CACHE              : "
        f"{CACHE_PATH}"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            if not is_index_record(
                record
            )
        ]

    print(
        f"Loaded raw records: {len(records)}"
    )

    if not records:
        raise RuntimeError(
            "No input records were loaded."
        )

    # 2. enrich with manifest metadata
    manifest_index = load_manifest_index(
        MANIFEST_PATH
    )

    print(
        f"Manifest index: {len(manifest_index)}"
    )

    if not manifest_index:
        raise RuntimeError(
            "Manifest index is empty."
        )

    records = enrich_records_from_manifest(
        records,
        manifest_index,
    )

    print(
        f"Enriched records: {len(records)}"
    )

    # 3. split BEFORE comparison grouping
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

    print()
    print("-" * 70)
    print("Document split")
    print("-" * 70)

    for split_name, items in split_map.items():
        print(
            f"{split_name:<12}: "
            f"{len(items)} documents"
        )

    print("-" * 70)

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

    # 5. generate comparison dataset
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