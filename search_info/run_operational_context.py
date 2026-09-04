from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search_info.z_search_task_hj import (  # noqa: E402
    OperationalSearchExecutor,
    SearchRuntimeConfig,
)


def _load_input(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read input JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError("input JSON must be an object")
    question = str(value.get("question") or "").strip()
    search_plan = value.get("search_plan")
    keyword_result = value.get("keyword_extractor_output")
    if not question:
        raise ValueError("input.question is required")
    if not isinstance(search_plan, dict):
        raise TypeError("input.search_plan must be an object")
    if not isinstance(keyword_result, dict):
        raise TypeError("input.keyword_extractor_output must be an object")
    return question, search_plan, keyword_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="검색계획을 회사별 운영 Chroma DB에서 실행합니다."
    )
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--strict-event-date",
        action="store_true",
        help="event_date 누락 문서의 rcept_dt 대체를 금지합니다.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="경로만 검사하고 모델과 Chroma를 로드하지 않습니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SearchRuntimeConfig.from_environment(
        args.project_root,
        strict_event_date=args.strict_event_date,
    )
    config.validate()
    if args.check_only:
        print("[SEARCH][CHECK] operational paths are ready")
        print(f"[SEARCH][CHECK] manifest={config.manifest_path}")
        print(f"[SEARCH][CHECK] universe={config.universe_path}")
        print(f"[SEARCH][CHECK] correction_edges={config.correction_edges_path}")
        print(f"[SEARCH][CHECK] chroma_db={config.chroma_db_path}")
        print(f"[SEARCH][CHECK] company_map={config.company_map_path}")
        return 0
    if args.input is None:
        raise ValueError("input JSON path is required unless --check-only is used")

    question, search_plan, keyword_result = _load_input(args.input)
    executor = OperationalSearchExecutor(config, debug=args.debug)
    result = executor.run(
        question=question,
        search_plan=search_plan,
        keyword_result=keyword_result,
    )

    output = args.output or config.project_root / "logs" / "search_executor_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SEARCH][RESULT] success={result.get('success')} stage={result.get('stage')}")
    print(f"[SEARCH][RESULT] saved={output}")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
