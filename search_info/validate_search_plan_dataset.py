from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_info.search_pipeline import validate_and_normalize_search_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCX 검색계획 학습 JSONL의 Completion을 실행기 계약으로 검증"
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    row_count = 0
    request_count = 0
    with args.dataset.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                plan = json.loads(row["Completion"])
                prepared = validate_and_normalize_search_plan(plan)
            except Exception as exc:
                raise ValueError(
                    f"dataset validation failed at line {line_number}: {exc}"
                ) from exc
            row_count += 1
            request_count += len(prepared.retrieval_requests)

    print(f"validated rows: {row_count:,}")
    print(f"validated retrieval requests: {request_count:,}")


if __name__ == "__main__":
    main()
