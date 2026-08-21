from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from z_base_function import (
    ClovaClient,
    ContextBundle,
    build_rows,
    extract_field_pairs,
    load_input_records,
    normalize_qas_container,
    normalize_text,
    record_to_context,
    response_schema,
    training_completion,
    training_text,
    validate_comparison_qas,
    write_rows,
)


COMPARE_TYPES = {
    1: "same_company_same_report_type",
    2: "correction_chain",
    3: "same_company_same_metric",
    4: "cross_company_same_report_type",
}

SUPPORTED_DOCUMENT_COUNTS = {2, 3, 4}


COMPARE_GENERATOR_SYSTEM_PROMPT = """당신은 기업 공시 기반 다중 문서 비교 QA 학습데이터 설계자다.
주어진 여러 문서의 Context에서만 질문과 정답을 만들고 외부 지식이나 추측을 사용하지 않는다.
질문은 실제 투자자가 여러 공시를 비교·분석할 때 할 법한 한국어로 작성한다.
evidence.quote는 Context에 존재하는 문자열을 글자 그대로 복사한다.
JSON Schema에 맞는 JSON만 출력한다."""

COMPARE_REVIEWER_SYSTEM_PROMPT = """당신은 기업 공시 기반 다중 문서 비교 QA 학습데이터 검수자다.
여러 문서를 함께 사용해야 하는 질문인지, 비교 기준과 답변이 Context에서 근거를 갖는지 검수한다.
후보를 안전하게 수정할 수 있으면 수정하고, 외부 지식이나 추측은 사용하지 않는다.
최종 출력은 지정된 JSON 구조만 반환한다."""

COMPARE_TASK_GUIDE = """
- 다중조회: 여러 문서의 같은 항목을 찾아 나열
- 비교연산: 차이, 합계, 평균, 최대·최소, 증감액·증감률
- 복합추론: 시점별 변화, 정정 전후 변화, 공통점과 차이점
- 조건결합: 여러 문서와 여러 필드를 함께 확인
- 근거부족: 제공된 여러 문서만으로 결론 낼 수 없는 질문 식별
가능하면 한 질문이 최소 두 문서를 실제로 사용하도록 한다.
"""


def comparison_generation_prompt(bundle: ContextBundle, count: int) -> str:
    return f"""아래 여러 공시 Context를 이용해 비교·다중문서 QA 후보 {count}개를 생성하라.

[생성 원칙]
1. Context에 있는 문서만 사용한다.
2. 단순히 문서 하나의 필드만 묻는 질문보다 두 문서 이상을 사용하는 질문을 우선한다.
3. 비교 질문은 비교 대상과 기준을 질문에서 명확히 한다.
4. answerable=true이면 답변에 필요한 모든 사실을 evidence로 제공한다.
5. evidence.document는 반드시 Context의 "문서 1", "문서 2" 표기를 사용한다.
6. evidence.field는 Context의 실제 필드명을 사용한다.
7. evidence.quote는 Context에 존재하는 연속된 원문 문자열을 그대로 복사한다.
8. 숫자를 계산하면 계산에 사용한 원본 값들을 모두 evidence로 제공한다.
9. Context에 없는 단위나 사실을 임의로 추가하지 않는다.
10. 서로 같은 의미의 질문을 반복하지 않는다.
11. 정정 관계라면 정정 전후에 실제로 달라진 항목을 중심으로 질문한다.
12. 비교할 수 없는 항목은 억지로 비교하지 않는다.
13. evidence는 반드시 다음 구조로 작성한다.

{
  "document": "문서 1",
  "field": "실제 Context 필드명",
  "quote": "Context에서 그대로 복사한 연속 문자열"
}

14. 비교에 여러 문서를 사용한 경우 문서별 evidence를 각각 별도 항목으로 작성한다.

15. evidence.quote를 요약하거나 새 문장으로 재작성하지 않는다.

16. 단순한 단일 문서 조회 질문은 생성하지 않는다.

[작업 유형]
{COMPARE_TASK_GUIDE}

[Context]
{bundle.context}
"""


def comparison_review_prompt(
    bundle: ContextBundle,
    candidates: list[dict[str, Any]],
    target_count: int,
) -> str:
    candidate_json = json.dumps(
        candidates,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
아래 다중 문서 Context와 1차 QA 후보를 검수하여
최종 비교 QA를 생성하라.

반드시 아래 규칙을 모두 지켜라.

============================================================
[최종 개수]
============================================================

최종 qas 배열에는 최대 {target_count}개만 포함한다.

1차 후보가 {len(candidates)}개이더라도
최종적으로 품질이 가장 좋은 {target_count}개 이하만 반환한다.

잘못된 후보는 Context만으로 수정 가능하면 수정한다.
수정 불가능하면 제거한다.
필요하면 Context만 사용하여 새로운 비교 QA를 만든다.


============================================================
[비교 QA 규칙]
============================================================

1. 가능한 한 2개 이상의 문서를 실제로 사용한다.

2. 한 문서만 보면 답할 수 있는 단순 정보추출 질문은
   비교 QA로 선택하지 않는다.

3. 좋은 질문 예:
   - 두 계약 중 매출액 대비 값이 더 큰 계약은 무엇인가?
   - 두 계약의 계약금액은 각각 얼마이며 어느 계약이 더 큰가?
   - 두 계약의 판매·공급지역은 각각 어디인가?
   - 두 계약에 적용된 환율은 각각 얼마인가?
   - 두 계약의 종료일은 어떻게 다른가?

4. 나쁜 질문 예:
   - 문서 1의 계약일자는?
   - 미국 법인은 어떤 회사인가?
   - 문서 2의 계약상대는?


============================================================
[answerable 규칙]
============================================================

answerable=true:
- answer는 Context만으로 직접 확인하거나 계산할 수 있어야 한다.
- evidence는 반드시 1개 이상 있어야 한다.
- 비교에 두 문서를 사용했다면 두 문서의 evidence를 모두 작성한다.

answerable=false:
- Context만으로 답할 수 없는 경우에만 사용한다.
- factual answer를 작성하지 않는다.
- answer는 반드시 "제공된 문서에서 확인할 수 없음"으로 작성한다.


============================================================
[evidence 규칙 - 매우 중요]
============================================================

evidence의 각 항목은 반드시 다음 세 필드를 가진다.

{{
  "document": "문서 1",
  "field": "매출액 대비",
  "quote": "매출액 대비: 5.37"
}}

document:
- 반드시 "문서 1", "문서 2", "문서 3", "문서 4" 중 하나다.
- quote가 실제 존재하는 문서 번호를 사용한다.

field:
- 반드시 Context에 실제 존재하는 필드명을 그대로 사용한다.
- 예: "계약금액", "매출액 대비", "판매·공급지역", "기타 중요사항"

quote:
- 반드시 Context에 실제 존재하는 연속 문자열을 그대로 복사한다.
- 요약하거나 문장을 새로 만들지 않는다.
- 숫자, 공백, 쉼표, 단위도 가능한 한 원문 그대로 복사한다.

잘못된 evidence:

{{
  "document": "문서 1",
  "field": "매출액 대비",
  "quote": "문서 1의 매출액 대비 비율은 5.37이다"
}}

위 문장은 Context에 실제 존재하지 않으므로 절대 사용하지 않는다.

올바른 evidence:

{{
  "document": "문서 1",
  "field": "매출액 대비",
  "quote": "매출액 대비: 5.37"
}}


============================================================
[다중 문서 evidence 예시]
============================================================

질문:
"두 계약 중 매출액 대비 값이 더 큰 계약은 무엇인가?"

반드시 다음처럼 두 문서의 값을 각각 근거로 작성한다.

"evidence": [
  {{
    "document": "문서 1",
    "field": "매출액 대비",
    "quote": "매출액 대비: 5.37"
  }},
  {{
    "document": "문서 2",
    "field": "매출액 대비",
    "quote": "매출액 대비: 3.22"
  }}
]


============================================================
[단위 규칙]
============================================================

Context에 없는 단위를 answer에 추가하지 않는다.

예를 들어 Context가:

매출액 대비: 5.37

이면 answer에도:

5.37

이라고 작성한다.

5.37%라고 쓰지 않는다.


============================================================
[계산 규칙]
============================================================

비교, 차이, 합계, 평균 등의 계산을 수행하면
계산에 사용한 모든 원본 값을 evidence에 포함한다.

예:

문서 1:
계약금액: 97,000,000,000

문서 2:
계약금액: 67,800,000,000

이면 두 값 모두 evidence에 포함해야 한다.


============================================================
[Context]
============================================================

{bundle.context}


============================================================
[1차 QA 후보]
============================================================

{candidate_json}


============================================================
[출력 형식]
============================================================

설명 문장이나 markdown을 절대 출력하지 않는다.

반드시 다음 JSON 객체 하나만 반환한다.

{{
  "qas": [
    {{
      "task_type": "비교연산",
      "question": "질문",
      "answer": "답변",
      "answerable": true,
      "evidence": [
        {{
          "document": "문서 1",
          "field": "실제 필드명",
          "quote": "Context에서 그대로 복사한 실제 문자열"
        }},
        {{
          "document": "문서 2",
          "field": "실제 필드명",
          "quote": "Context에서 그대로 복사한 실제 문자열"
        }}
      ]
    }}
  ]
}}

qas는 최대 {target_count}개만 반환한다.
""".strip()


@dataclass
class ComparisonGroup:
    group_id: str
    compare_type: str
    records: list[dict[str, Any]]
    metadata: dict[str, Any]


def receipt_no(record: dict[str, Any]) -> str:
    value = normalize_text(
        record.get("rcept_no")
        or record.get("receipt_no")
    )
    if value:
        return value

    source = normalize_text(record.get("_source_path"))
    match = re.search(r"(?<!\d)(20\d{12})(?!\d)", source)
    if match:
        return match.group(1)

    return ""


def record_identity(record: dict[str, Any]) -> str:
    return (
        receipt_no(record)
        or normalize_text(record.get("_source_path"))
        or normalize_text(record.get("title") or record.get("question"))
    )


def record_company(record: dict[str, Any]) -> str:
    value = normalize_text(
        record.get("corp_name")
        or record.get("company")
    )
    if value:
        return value

    title = normalize_text(record.get("title") or record.get("question"))
    if "/" in title:
        return title.split("/", 1)[0].strip() or "unknown"

    source = Path(str(record.get("_source_path") or ""))
    if len(source.parts) >= 3:
        return source.parts[-3]

    return "unknown"


def record_report_type(record: dict[str, Any]) -> str:
    for key in (
        "normalized_report_type",
        "doc_subtype",
        "report_nm",
    ):
        value = normalize_text(record.get(key))
        if value:
            return value

    title = normalize_text(record.get("title") or record.get("question"))
    parts = [part.strip() for part in title.split("/") if part.strip()]
    if len(parts) >= 2:
        return parts[1]

    return "unknown"


def record_chain_id(record: dict[str, Any]) -> str:
    return normalize_text(
        record.get("disclosure_chain_id")
        or record.get("correction_chain_id")
        or record.get("chain_id")
    )


def record_date(record: dict[str, Any]) -> datetime | None:
    values = (
        record.get("event_date"),
        record.get("rcept_dt"),
        receipt_no(record),
        record.get("title"),
        record.get("_source_path"),
    )

    for value in values:
        text = normalize_text(value)
        digits = re.sub(r"\D", "", text)

        for length, fmt in ((8, "%Y%m%d"),):
            if len(digits) >= length:
                candidate = digits[:length]
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    pass

    return None


def record_year(record: dict[str, Any]) -> int | None:
    dt = record_date(record)
    return dt.year if dt else None


def is_index_record(record: dict[str, Any]) -> bool:
    source_name = Path(
        str(record.get("_source_path") or "")
    ).name.lower()

    metadata_keys = {
        "corp_code",
        "corp_name",
        "report_nm",
        "rcept_no",
        "rcept_dt",
    }

    return (
        source_name.startswith("list_")
        or metadata_keys.issubset(record.keys())
    )


def load_manifest_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Index manifest JSONL by receipt number."""
    if path is None or not path.exists():
        return {}

    index: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = normalize_text(
                item.get("rcept_no")
                or item.get("receipt_no")
            )
            if key:
                index[key] = item

    return index


def enrich_records_from_manifest(
    records: list[dict[str, Any]],
    manifest_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add metadata such as normalized_report_type,
    disclosure_chain_id, base_year/base_month, sector, etc.
    Raw document values always win over manifest values.
    """
    enriched: list[dict[str, Any]] = []

    for record in records:
        merged = dict(manifest_index.get(receipt_no(record), {}))
        merged.update(record)
        enriched.append(merged)

    return enriched


def split_records(
    records: list[dict[str, Any]],
    *,
    split_mode: str,
    split_seed: str,
    hash_ratios: tuple[float, float, float],
    train_end_year: int,
    validation_years: set[int],
    test_start_year: int,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all documents in a comparison group inside one split."""
    result = {
        "train": [],
        "validation": [],
        "test": [],
    }

    def hash_split(identity: str) -> str:
        train_ratio, validation_ratio, test_ratio = hash_ratios
        total = train_ratio + validation_ratio + test_ratio
        value = int(
            hashlib.sha256(
                (split_seed + identity).encode("utf-8")
            ).hexdigest()[:12],
            16,
        ) / float(0xFFFFFFFFFFFF)

        if value < train_ratio / total:
            return "train"
        if value < (train_ratio + validation_ratio) / total:
            return "validation"
        return "test"

    for record in records:
        identity = record_identity(record)

        if split_mode == "hash":
            split = hash_split(identity)
        elif split_mode == "time":
            year = record_year(record)
            if year is None:
                split = hash_split(identity)
            elif year <= train_end_year:
                split = "train"
            elif year in validation_years:
                split = "validation"
            elif year >= test_start_year:
                split = "test"
            else:
                split = hash_split(identity)
        else:
            raise ValueError("split_mode must be 'time' or 'hash'")

        result[split].append(record)

    return result


def metric_labels(record: dict[str, Any]) -> set[str]:
    """
    Approximate 'metric' using actual document field labels containing
    numeric values. This is intentionally generic and can be refined later.
    """
    excluded = {
        "문서명",
        "회사명",
        "보고서명",
        "공시일",
        "계약일자",
        "계약 시작일",
        "계약 종료일",
        "유보기한",
    }

    result: set[str] = set()

    for label, value in extract_field_pairs(record):
        label = normalize_text(label)
        value = normalize_text(value)

        if not label or label in excluded:
            continue

        if re.search(r"\d", value):
            result.add(label)

    return result


def _window_groups(
    items: list[dict[str, Any]],
    document_count: int,
) -> Iterable[list[dict[str, Any]]]:
    if len(items) < document_count:
        return

    ordered = sorted(
        items,
        key=lambda r: (
            record_date(r) or datetime.min,
            record_identity(r),
        ),
    )

    for start in range(0, len(ordered) - document_count + 1):
        yield ordered[start : start + document_count]


def _group_id(
    compare_type: str,
    records: list[dict[str, Any]],
) -> str:
    joined = "|".join(record_identity(r) for r in records)
    digest = hashlib.sha1(
        f"{compare_type}|{joined}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{compare_type}_{digest}"


def build_comparison_groups(
    records: list[dict[str, Any]],
    *,
    compare_type: str,
    document_count: int,
    similar_period_days: int = 90,
    max_groups: int = 0,
) -> list[ComparisonGroup]:
    if compare_type not in COMPARE_TYPES.values():
        raise ValueError(
            f"Unsupported compare_type: {compare_type}"
        )

    if document_count not in SUPPORTED_DOCUMENT_COUNTS:
        raise ValueError(
            "document_count must be one of 2, 3, 4"
        )

    groups: list[ComparisonGroup] = []
    seen: set[tuple[str, ...]] = set()

    def add_group(
        selected: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        if len(selected) != document_count:
            return

        ids = tuple(record_identity(r) for r in selected)
        if len(set(ids)) != document_count:
            return

        dedupe_key = tuple(sorted(ids))
        if dedupe_key in seen:
            return

        seen.add(dedupe_key)
        groups.append(
            ComparisonGroup(
                group_id=_group_id(compare_type, selected),
                compare_type=compare_type,
                records=selected,
                metadata=metadata,
            )
        )

    # ---------------------------------------------------------
    # 1. same company + same report type + different time
    # ---------------------------------------------------------
    if compare_type == "same_company_same_report_type":
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            buckets[
                (
                    record_company(record),
                    record_report_type(record),
                )
            ].append(record)

        for (company, report_type), items in buckets.items():
            for selected in _window_groups(items, document_count):
                dates = [record_date(r) for r in selected]
                if len({d for d in dates if d is not None}) < 2:
                    continue

                add_group(
                    selected,
                    {
                        "company": company,
                        "report_type": report_type,
                    },
                )

    # ---------------------------------------------------------
    # 2. same disclosure/correction chain
    # ---------------------------------------------------------
    elif compare_type == "correction_chain":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            chain_id = record_chain_id(record)
            if chain_id:
                buckets[chain_id].append(record)

        for chain_id, items in buckets.items():
            if len(items) < document_count:
                continue

            for selected in _window_groups(items, document_count):
                add_group(
                    selected,
                    {
                        "chain_id": chain_id,
                        "company": record_company(selected[0]),
                        "report_type": record_report_type(selected[0]),
                    },
                )

    # ---------------------------------------------------------
    # 3. same company + same metric + different period
    # ---------------------------------------------------------
    elif compare_type == "same_company_same_metric":
        metric_buckets: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)

        for record in records:
            company = record_company(record)
            for metric in metric_labels(record):
                metric_buckets[(company, metric)].append(record)

        for (company, metric), items in metric_buckets.items():
            for selected in _window_groups(items, document_count):
                add_group(
                    selected,
                    {
                        "company": company,
                        "metric": metric,
                    },
                )

    # ---------------------------------------------------------
    # 4. different companies + same report type + similar period
    # ---------------------------------------------------------
    elif compare_type == "cross_company_same_report_type":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            buckets[record_report_type(record)].append(record)

        for report_type, items in buckets.items():
            for selected in _window_groups(items, document_count):
                companies = [record_company(r) for r in selected]
                if len(set(companies)) != document_count:
                    continue

                dates = [record_date(r) for r in selected]
                if any(d is None for d in dates):
                    continue

                date_values = [d for d in dates if d is not None]
                gap = (max(date_values) - min(date_values)).days

                if gap > similar_period_days:
                    continue

                add_group(
                    selected,
                    {
                        "companies": companies,
                        "report_type": report_type,
                        "period_gap_days": gap,
                    },
                )

    groups.sort(
        key=lambda g: (
            g.compare_type,
            g.group_id,
        )
    )

    if max_groups > 0:
        groups = groups[:max_groups]

    return groups


def comparison_group_to_bundle(
    group: ComparisonGroup,
    *,
    max_context_chars: int,
) -> ContextBundle:
    """
    Render 2~4 documents into one multi-document ContextBundle.
    """
    per_doc_limit = max(
        1000,
        max_context_chars // len(group.records),
    )

    blocks: list[str] = []
    source_ids: list[str] = []

    for index, record in enumerate(group.records, start=1):
        block, source_id = record_to_context(
            record,
            index,
            per_doc_limit,
        )
        blocks.append(block)
        source_ids.append(source_id)

    header = (
        f"[비교유형]\n{group.compare_type}\n\n"
    )

    context = (
        header
        + "\n\n".join(blocks)
    )[:max_context_chars]

    return ContextBundle(
        context=context,
        source_ids=source_ids,
        mode="multi",
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(
        r"[^0-9A-Za-z가-힣._-]+",
        "_",
        value,
    ).strip("._") or "group"

    return cleaned[:120]


def write_group_json(
    path: Path,
    *,
    split_name: str,
    group: ComparisonGroup,
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": split_name,
        "group_id": group.group_id,
        "compare_type": group.compare_type,
        "document_count": len(group.records),
        "metadata": group.metadata,
        "documents": [
            {
                "receipt_no": receipt_no(record),
                "company": record_company(record),
                "report_type": record_report_type(record),
                "event_date": (
                    record_date(record).strftime("%Y-%m-%d")
                    if record_date(record)
                    else None
                ),
                "source": normalize_text(record.get("_source_path")),
            }
            for record in group.records
        ],
        "generated_count": len(rows),
        "rejected_count": len(errors),
        "rows": rows,
        "errors": errors,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def reindex_rows(rows: list[dict[str, Any]]) -> None:
    for cid, row in enumerate(rows):
        row["C_ID"] = cid
        row["T_ID"] = 0


def generate_comparison_dataset(
    *,
    split_map: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    compare_type: str,
    document_count: int,
    client: ClovaClient,
    questions_per_group: int,
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    cache_path: Path,
    similar_period_days: int = 90,
    max_groups_per_split: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = output_dir / "generated"
    rejected_dir = output_dir / "rejected"

    manifest: dict[str, Any] = {
        "compare_type": compare_type,
        "document_count": document_count,
        "questions_per_group": questions_per_group,
        "similar_period_days": similar_period_days,
        "splits": {},
    }

    for split_name, records in split_map.items():
        groups = build_comparison_groups(
            records,
            compare_type=compare_type,
            document_count=document_count,
            similar_period_days=similar_period_days,
            max_groups=max_groups_per_split,
        )

        print(
            f"[{split_name}] comparison groups: {len(groups)} "
            f"(type={compare_type}, docs={document_count})"
        )

        final_rows: list[dict[str, Any]] = []
        split_errors: list[str] = []

        for index, group in enumerate(groups, start=1):
            print(
                f"  [{index}/{len(groups)}] "
                f"{group.group_id}"
            )

            bundle = comparison_group_to_bundle(
                group,
                max_context_chars=max_context_chars,
            )

            try:
                rows, errors = build_rows(
                    [bundle],
                    client,
                    single_count=0,
                    multi_count=questions_per_group,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    include_context=True,
                    system_prompt=system_prompt,
                    cache_path=cache_path,
                    limit=0,
                    generation_prompt_fn=comparison_generation_prompt,
                    review_prompt_fn=comparison_review_prompt,
                    normalize_qas_fn=normalize_qas_container,
                    validate_reviewed_fn=validate_comparison_qas,
                    training_text_fn=training_text,
                    training_completion_fn=training_completion,
                    generator_system_prompt=COMPARE_GENERATOR_SYSTEM_PROMPT,
                    reviewer_system_prompt=COMPARE_REVIEWER_SYSTEM_PROMPT,
                    response_schema=response_schema(),
                    generator_cache_version="GENERATOR_COMPARE_V1",
                    reviewer_cache_version="REVIEWER_COMPARE_V1",
                )
            except Exception as exc:
                rows = []
                errors = [str(exc)]

            # Local C_ID inside each JSON
            reindex_rows(rows)

            group_path = (
                generated_dir
                / split_name
                / f"{safe_filename(group.group_id)}.json"
            )

            write_group_json(
                group_path,
                split_name=split_name,
                group=group,
                rows=rows,
                errors=errors,
            )

            if errors:
                split_errors.extend(
                    f"{group.group_id}: {error}"
                    for error in errors
                )

            final_rows.extend(rows)

        # Final global C_ID for tuning CSV
        reindex_rows(final_rows)

        csv_path = output_dir / f"{split_name}.csv"
        write_rows(
            final_rows,
            csv_path,
            "csv",
            system_prompt,
        )

        if split_errors:
            error_path = (
                rejected_dir
                / f"{split_name}.errors.log"
            )
            error_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            error_path.write_text(
                "\n".join(split_errors),
                encoding="utf-8",
            )

        manifest["splits"][split_name] = {
            "records": len(records),
            "groups": len(groups),
            "rows": len(final_rows),
        }

        print(
            f"Wrote {csv_path}: {len(final_rows)} rows"
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )