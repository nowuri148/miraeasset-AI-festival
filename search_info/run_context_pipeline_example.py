from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("SEARCH_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search_info.search_pipeline import (
    JsonlHybridChunkStore,
    ManifestGraphCatalog,
    execute_search_plan,
)


QUESTION = (
    "삼성전자의 2023년 사업보고서와 2025년 사업보고서를 비교했을 때 "
    "핵심 사업은 어떻게 변화했는지 설명해줘"
)

SEARCH_PLAN = {
    "retrieval_requests": [
        {
            "query": "삼성전자 2023년 핵심 사업 부문, 매출 비중과 성장 전략",
            "normalized_report_types": ["annual_report"],
            "period": {"start": "2023", "end": "2023"},
            "date_basis": "base_year",
            "use_correction_graph": False,
            "company_scope": {
                "scope_type": "direct",
                "scope_values": ["삼성전자"],
            },
            "exact_keywords": [
                "DX 부문", "DS 부문", "SDC", "Harman", "매출 비중", "주요 사업"
            ],
        },
        {
            "query": "삼성전자 2025년 핵심 사업 부문, 매출 비중과 성장 전략",
            "normalized_report_types": ["annual_report"],
            "period": {"start": "2025", "end": "2025"},
            "date_basis": "base_year",
            "use_correction_graph": False,
            "company_scope": {
                "scope_type": "direct",
                "scope_values": ["삼성전자"],
            },
            "exact_keywords": [
                "DX 부문", "DS 부문", "SDC", "Harman", "매출 비중", "주요 사업"
            ],
        },
    ]
}

MANIFEST_PATH = PROJECT_ROOT / "corpus" / "manifest.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "corpus" / "derived" / "periodic_vector_chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "corpus" / "derived" / "samsung_2023_2025_business_context.json"


def main() -> None:
    catalog = ManifestGraphCatalog(MANIFEST_PATH)
    store = JsonlHybridChunkStore(CHUNKS_PATH)
    context = execute_search_plan(
        SEARCH_PLAN,
        catalog,
        store,
        question=QUESTION,
        dense_k=30,
        lexical_k=30,
        context_char_budget=24_000,
        debug=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(context.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"documents: {len(context.documents)}")
    print(f"chunks: {len(context.chunks)}")
    print(f"total_chars: {context.total_chars}")
    print(f"covered_period_buckets: {context.covered_period_buckets}")
    print(f"covered_requests: {context.covered_request_ids}")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
