from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import CLOVA_STUDIO_API_KEY  # noqa: E402
from z_base_function import extract_field_pairs, load_input_records, normalize_text


COLUMNS = ["C_ID", "T_ID", "Text", "Completion"]
SYSTEM_COLUMNS = ["System_Prompt", *COLUMNS]
DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
DEFAULT_MODEL = "HCX-005"
GENERATOR_SYSTEM_PROMPT = """당신은 기업 공시 기반 금융 QA 학습데이터 설계자다.
주어진 Context에서만 질문과 정답을 만들고 외부 지식이나 추측을 사용하지 않는다.
질문은 실제 투자자가 할 법한 한국어로 작성하고, 서로 다른 사고 과정을 요구해야 한다.
근거 quote는 Context에 존재하는 문자열을 글자 그대로 복사한다.
JSON Schema에 맞는 JSON만 출력한다."""
TARGET_SYSTEM_PROMPT = (
    "제공된 문서를 근거로 질문에 정확히 답하세요. 추측하지 말고, 답변 뒤에 사용한 근거를 표시하세요."
)

REVIEWER_SYSTEM_PROMPT = """당신은 기업 공시 기반 금융 QA 학습데이터 검수자다.
주어진 Context와 QA 후보를 함께 검토한다.
후보를 단순히 삭제하는 것보다 Context만으로 안전하게 수정할 수 있으면 질문, 답변, task_type, answerable, evidence를 수정한다.
외부 지식이나 추측을 사용하지 않는다.
최종 출력은 지정된 JSON 구조만 반환한다."""


def review_prompt(
    bundle: ContextBundle,
    candidates: list[dict[str, Any]],
    target_count: int,
) -> str:
    candidate_json = json.dumps(
        candidates,
        ensure_ascii=False,
        indent=2,
    )

    return f"""아래 Context와 1차 생성된 QA 후보들을 검수하라.

[목표]
- 최종적으로 품질이 가장 좋은 QA {target_count}개를 반환한다.
- 잘못된 후보는 가능한 경우 Context에 맞게 수정해서 살린다.
- 수정해도 신뢰할 수 없는 후보만 제거한다.
- 후보가 부족하면 Context만 사용해서 새 QA를 보충해 최종 {target_count}개를 맞춘다.

[검수 기준]
1. 질문의 모든 전제가 Context에서 성립해야 한다.
2. 단일 문서인데 여러 계약/여러 공시를 비교하는 것처럼 존재하지 않는 비교 집합을 가정하지 않는다.
3. answerable=true이면 답변이 Context만으로 직접 또는 계산을 통해 도출 가능해야 한다.
4. answerable=false이면 정말 Context에 필요한 정보가 없어야 하며, 무엇이 부족한지 구체적으로 설명한다.
5. 답변에 사용한 사실, 금액, 비율, 날짜, 수량, 환율은 evidence로 충분히 뒷받침되어야 한다.
6. evidence.quote는 반드시 Context에 실제로 존재하는 연속된 원문 문자열을 그대로 복사한다.
7. evidence.document에는 "문서 1", "문서 2"처럼 Context의 문서 번호를 사용한다.
8. evidence.field에는 Context의 실제 필드명을 글자 그대로 사용한다.
   예: Context가 "회사와의 관계: 자회사"라면 field는 반드시 "회사와의 관계"로 적는다.
   "관계"처럼 임의로 축약하지 않는다.
9. evidence.quote는 가능한 한 해당 필드의 전체 원문 라인을 그대로 복사한다.
   예: "계약 종료일: 2024-10-31"
10. 질문과 답변이 자연스러운 한국어인지 확인하고 어색하면 수정한다.
10. 중복 질문이나 단순 동의어 반복을 제거한다.
11. 정보추출에만 편중되지 않도록 가능한 범위에서 조건결합, 계산, 요약, 근거부족 등을 섞는다.
12. 계산 질문은 계산에 필요한 원본 값들이 Context에 모두 있을 때만 유지한다.
13. answer에는 불필요한 외부 해석을 추가하지 않는다.
14. "가장 큰", "최대", "최소", "순위" 같은 표현은 Context에 실제 비교 대상이 2개 이상 있을 때만 사용한다.
15. task_type은 아래 값 중 하나만 사용한다:
    정보추출 / 조건결합 / 계산 / 요약 / 다중조회 / 비교연산 / 복합추론 / 근거부족
16. Context에 명시되지 않은 단위를 answer에 임의로 추가하지 않는다.
    예를 들어 Context가 "최근매출액: 1,806,000,000,000"이라고만 제시하면
    answer에 임의로 "원"을 붙이지 않는다.
17. Context가 "매출액 대비: 5.37"이라고만 제시하면,
    evidence나 Context에 %가 명시되지 않은 이상 answer에 "%"를 임의로 붙이지 않는다.
18. 원, 달러, %, 억원, 주, 개, 배, 일, 월 등 단위는 Context 또는 evidence에
    직접 표현되어 있을 때만 answer에 사용한다.
19. 단위가 Context에 없지만 의미상 필요하면 숫자 뒤에 단위를 추측해 붙이지 말고
    "문서에는 5.37로 기재되어 있습니다"처럼 원문 표현을 보존한다.
20. answer의 모든 수치와 단위는 evidence.quote와 의미적으로 일치해야 한다.
21. 최종 QA가 4개라면 최소 3개 이상의 서로 다른 task_type을 포함한다.
22. 가능한 경우 정보추출 / 조건결합 / 계산 / 요약 / 근거부족을 골고루 구성한다.
23. 동일한 task_type의 질문이 2개를 초과하지 않도록 한다.
24. 단순 필드 조회형 질문만 여러 개 나열하지 않는다.
25. 계산 가능한 값이 Context에 충분히 있으면 최소 1개의 계산 질문을 우선 고려한다.
26. 계산 질문은 사용한 원본 값들을 evidence로 모두 제공하고 answer에 계산 근거를 짧게 적는다.
27. 조건결합 질문은 서로 다른 두 개 이상의 필드가 실제 Context에 존재할 때 만든다.
28. 요약 질문은 막연한 전체 요약보다 "계약 규모와 기간", "상대방과 지역"처럼 관점을 명확히 한다.

[Context]
{bundle.context}

[1차 QA 후보]
{candidate_json}

반드시 다음 JSON 객체 형식으로만 반환하라.
{{
  "qas": [
    {{
      "task_type": "정보추출",
      "question": "...",
      "answer": "...",
      "answerable": true,
      "evidence": [
        {{
          "document": "문서 1",
          "field": "계약금액",
          "quote": "계약금액: 97,000,000,000"
        }}
      ]
    }}
  ]
}}
"""


TASK_GUIDE_SINGLE = """
- 정보추출: 사람/회사/금액/날짜/지역/계약명/목적/사유
- 조건 결합: 두 개 이상의 필드를 함께 확인하는 질문
- 계산: 기간 일수, 비율 검산, 금액 단위 변환 등 Context로 계산 가능한 질문
- 요약: 핵심 내용 또는 투자자가 확인할 조건을 선별
- 근거 부족: Context에 없는 정보를 요구하고 확인 불가라고 답하는 질문
- 표현 다양성: 존댓말, 구어체, 짧은 검색어형, 정식 질의형을 고르게 사용
"""

TASK_GUIDE_MULTI = """
- 다중 조회: 여러 문서에서 같은 항목을 찾아 나열
- 비교·연산: 차이, 합계, 평균, 최대·최소, 순위, 증감액·증감률
- 복합 문서 추론: 시점별 변화, 체결-정정-해지 관계, 공통점과 차이점
- 조건 검색: 기간·회사·지역·임계값 등 여러 조건을 만족하는 문서 선별
- 근거 부족: 문서들만으로 결론 낼 수 없는 질문을 식별
계산 질문의 answer에는 사용한 값과 계산 결과를 함께 적는다.
"""


@dataclass
class ContextBundle:
    context: str
    source_ids: list[str]
    mode: str


def safe_text(value: Any, limit: int = 2000) -> str:
    return normalize_text(value)[:limit]


def record_title(record: dict[str, Any]) -> str:
    return safe_text(record.get("title") or record.get("question") or record.get("_source_path") or "문서")


def record_to_context(record: dict[str, Any], index: int, max_chars: int) -> tuple[str, str]:
    title = record_title(record)
    # _source_path는 내부 추적용 source_id로만 사용한다.
    # 실제 tuning Text에는 로컬 파일 경로를 노출하지 않는다.
    source = safe_text(record.get("_source_path"), 1000)
    fields = extract_field_pairs(record)

    lines = [f"[문서 {index}]", f"문서명: {title}"]
    for label, value in fields:
        value = safe_text(value, 3000)
        if value:
            lines.append(f"{label}: {value}")

    if len(lines) <= 2:
        raw_answer = safe_text(record.get("answer") or record.get("text") or record, 5000)
        if raw_answer:
            lines.append(f"본문: {raw_answer}")

    block = "\n".join(lines)
    return block[:max_chars], source or title


def make_bundles(
    records: list[dict[str, Any]],
    max_context_chars: int,
    multi_doc_size: int,
    include_multi: bool,
) -> list[ContextBundle]:
    singles: list[ContextBundle] = []
    rendered: list[tuple[str, str, str]] = []
    for record in records:
        block, source_id = record_to_context(record, 1, max_context_chars)
        company = record_title(record).split("/")[0].strip()
        rendered.append((block, source_id, company))
        singles.append(ContextBundle(block, [source_id], "single"))

    if not include_multi or multi_doc_size < 2:
        return singles

    multi: list[ContextBundle] = []
    by_company: dict[str, list[tuple[str, str, str]]] = {}
    for item in rendered:
        by_company.setdefault(item[2], []).append(item)
    for items in by_company.values():
        for start in range(0, len(items) - 1, multi_doc_size):
            chunk = items[start : start + multi_doc_size]
            if len(chunk) < 2:
                continue
            blocks = []
            sources = []
            for idx, (block, source_id, _) in enumerate(chunk, start=1):
                blocks.append(re.sub(r"^\[문서 1\]", f"[문서 {idx}]", block))
                sources.append(source_id)
            context = "\n\n".join(blocks)[:max_context_chars]
            multi.append(ContextBundle(context, sources, "multi"))
    return singles + multi


def response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grounded_financial_qa",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "qas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_type": {
                                    "type": "string",
                                    "enum": [
                                        "정보추출", "조건결합", "계산", "요약",
                                        "다중조회", "비교연산", "복합추론", "근거부족",
                                    ],
                                },
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "answerable": {"type": "boolean"},
                                "evidence": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "document": {"type": "string"},
                                            "field": {"type": "string"},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["document", "field", "quote"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["task_type", "question", "answer", "answerable", "evidence"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["qas"],
                "additionalProperties": False,
            },
        },
    }


def generation_prompt(bundle: ContextBundle, count: int) -> str:
    guide = TASK_GUIDE_MULTI if bundle.mode == "multi" else TASK_GUIDE_SINGLE
    return f"""아래 Context로 파인튜닝용 질문-답변 {count}개를 생성하라.

[생성 원칙]
1. 질문끼리 핵심 의도와 필요한 연산이 중복되지 않게 한다.
2. 답할 수 있는 질문은 Context의 정확한 사실만 사용한다.
3. answerable=true이면 evidence를 1개 이상 제공한다.
4. evidence.quote는 반드시 Context의 연속된 원문 문자열이어야 한다.
5. answerable=false인 질문은 전체의 10~20%로 하고, answer에는 어떤 정보가 부족한지 쓴다.
6. 단순 동의어 치환으로 개수를 채우지 않는다.
7. 질문과 답변만 읽어도 수치의 단위와 비교 기준이 명확해야 한다.
8. evidence.document에는 반드시 Context의 표기 그대로 "문서 1", "문서 2"처럼 적는다.
9. evidence.field에는 Context의 실제 필드명만 적는다.
10. evidence.quote에는 해당 근거가 있는 Context의 연속된 원문 문자열을 그대로 복사한다.
11. 한 문서에서 단순 필드 조회 질문만 반복하지 말고 조건 결합·계산·요약·근거 부족을 섞는다.
12. 단일 문서 Context에서는 비교 대상이 실제로 2개 이상 존재하지 않는 한
    "가장 큰", "가장 높은", "최대", "최소", "순위", "중 가장" 같은
    비교·최상급 표현을 질문에 사용하지 않는다.
13. 단일 문서 하나만 주어졌는데 "여러 계약 중", "공시들 중", "가장 큰 계약"처럼
    Context에 존재하지 않는 비교 집합을 가정하지 않는다.
14. 질문의 전제 자체가 Context에서 확인되지 않으면 answerable=true 질문으로 만들지 않는다.
    존재하지 않는 전제를 일부러 묻는 경우에는 answerable=false로 만들고,
    answer에는 어떤 정보가 없어 판단할 수 없는지 구체적으로 적는다.
15. 근거 부족 질문은 자연스러운 투자자 질문으로 작성한다.
    "몇 년 전이었는지", "소요된 시간이 몇 년 전"처럼 의미가 어색한 문장을 만들지 않는다.
16. 근거 부족 질문의 answer에는 단순히 "알 수 없습니다"라고만 쓰지 말고,
    예: "제공된 문서에는 계약 준비 기간이 없어 확인할 수 없습니다."처럼
    부족한 정보가 무엇인지 명시한다.
17. answerable=true 질문은 answer에 필요한 사실이 evidence.quote에 직접 포함되도록 한다.
18. 한 Context에서 생성되는 질문은 가능한 한 서로 다른 task_type을 사용한다.
    단순 정보추출이 전체의 절반을 넘지 않도록 한다.
19. 계산 질문은 Context에 계산에 필요한 값이 모두 있을 때만 만들고,
    answer에 사용한 값과 계산식을 간단히 포함한다.
20. 요약 질문은 문서 전체를 막연히 요약시키기보다 투자자가 확인할 핵심 조건,
    계약 규모, 기간, 상대방, 목적, 변경사항 등 명확한 관점을 지정한다.
21. answer에 금액, 비율, 날짜, 수량, 환율 등 숫자를 사용했다면,
    그 숫자가 포함된 evidence.quote를 반드시 모두 제공한다.
22. answer에 여러 숫자를 사용하면 각 숫자를 뒷받침하는 evidence를 빠짐없이 포함한다.
    evidence에 없는 숫자를 answer에 새로 만들어 넣지 않는다.
23. 계산 결과처럼 Context에 그대로 존재하지 않는 새 숫자를 answer에 제시하는 경우,
    계산에 사용한 원본 숫자들을 evidence로 모두 제공하고 answer에 계산식을 간단히 적는다.

24. 서로 다른 task_type의 후보를 다양하게 생성한다.
25. 6개 후보를 만들 경우 정보추출만 반복하지 말고,
    가능하면 정보추출 / 조건결합 / 계산 / 요약 / 근거부족을 골고루 포함한다.
26. 같은 필드만 표현을 바꿔 반복 질문하지 않는다.
27. 계산 가능한 수치가 Context에 있으면 최소 1개의 계산 후보를 우선 생성한다.
28. 두 개 이상의 관련 필드가 있으면 최소 1개의 조건결합 후보를 우선 생성한다.

[필수 작업 유형]
{guide}

[Context]
{bundle.context}
"""


class ClovaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        retries: int,
    ) -> None:
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        use_response_format: bool = True,
    ) -> dict[str, Any] | list[Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "top_p": 0.8,
            "max_tokens": max_tokens,
        }

        if use_response_format:
            payload["response_format"] = response_schema()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        schema_enabled = use_response_format

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if (
                    response.status_code == 400
                    and schema_enabled
                    and "response_format" in payload
                ):
                    print(
                        "[WARN] response_format 요청이 거부되어 "
                        "JSON Schema 없이 다시 시도합니다."
                    )
                    payload.pop("response_format", None)
                    schema_enabled = False

                    response = requests.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )

                response.raise_for_status()

                result = response.json()
                content = (
                    result["choices"][0]
                    ["message"]["content"]
                )

                return parse_json_content(
                    content
                )

            except (
                requests.RequestException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

            if attempt >= self.retries:
                break

            wait_seconds = min(
                2 ** attempt,
                10,
            )
            print(
                f"[WARN] API 호출 실패. "
                f"{wait_seconds}초 후 재시도합니다. "
                f"({attempt + 1}/{self.retries})"
            )
            time.sleep(wait_seconds)

        raise RuntimeError(
            f"CLOVA Studio generation failed: "
            f"{last_error}"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any] | list[Any]:
        return self._call_json(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_response_format=True,
        )

    def review(
        self,
        bundle: ContextBundle,
        candidates: list[dict[str, Any]],
        target_count: int,
        max_tokens: int,
    ) -> dict[str, Any] | list[Any]:
        prompt = review_prompt(
            bundle=bundle,
            candidates=candidates,
            target_count=target_count,
        )

        return self._call_json(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            use_response_format=True,
        )


def parse_json_content(
    content: str | dict[str, Any] | list[Any],
) -> dict[str, Any] | list[Any]:
    if isinstance(content, (dict, list)):
        return content

    text = content.strip()

    # HCX가 ```json ... ``` 형태로 반환하는 경우 방어
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "HyperCLOVA X 응답을 JSON으로 파싱하지 못했습니다.\n"
            f"원본 응답:\n{content}"
        ) from exc


def normalize_question(question: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", question.casefold())


def extract_numeric_tokens(text: str) -> set[str]:
    """
    텍스트에서 비교 가능한 숫자 토큰을 추출한다.

    예:
    97,000,000,000 -> 97000000000
    78,856,650.00  -> 78856650
    5.37           -> 5.37

    날짜/환율/비율 등의 숫자도 그대로 비교한다.
    """
    if not text:
        return set()

    raw_numbers = re.findall(
        r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?",
        text,
    )

    normalized: set[str] = set()

    for value in raw_numbers:
        cleaned = value.replace(",", "").strip()

        if not cleaned:
            continue

        # 소수점 뒤 0만 있는 경우 정수처럼 정규화
        try:
            if "." in cleaned:
                number = float(cleaned)

                if number.is_integer():
                    normalized.add(str(int(number)))
                else:
                    normalized.add(
                        cleaned.rstrip("0").rstrip(".")
                    )
            else:
                normalized.add(str(int(cleaned)))
        except ValueError:
            continue

    return normalized


def validate_numeric_grounding(
    answer: str,
    evidence: list[dict[str, Any]],
) -> tuple[bool, set[str]]:
    """
    answer에 등장하는 숫자가 evidence.quote에 모두 존재하는지 확인한다.

    반환:
    (통과 여부, evidence에 없는 숫자 집합)
    """
    answer_numbers = extract_numeric_tokens(answer)

    if not answer_numbers:
        return True, set()

    evidence_text = " ".join(
        str(item.get("quote", ""))
        for item in evidence
    )

    evidence_numbers = extract_numeric_tokens(
        evidence_text
    )

    unsupported = (
        answer_numbers
        - evidence_numbers
    )

    return not unsupported, unsupported


SINGLE_CONTEXT_UNSUPPORTED_PATTERNS = [
    r"가장\s*큰",
    r"가장\s*높은",
    r"가장\s*낮은",
    r"가장\s*많은",
    r"가장\s*적은",
    r"\b최대\b",
    r"\b최소\b",
    r"\b순위\b",
    r"중\s*가장",
    r"여러\s*계약\s*중",
    r"공시들\s*중",
    r"계약들\s*중",
]


def has_unsupported_single_context_comparison(
    question: str,
    bundle: ContextBundle,
) -> bool:
    """
    단일 문서 하나만 있는데 여러 문서/여러 계약의 비교 집합을
    가정하는 질문을 걸러낸다.

    multi bundle에서는 비교/최상급 표현을 허용한다.
    """
    if bundle.mode != "single":
        return False

    normalized = question.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in SINGLE_CONTEXT_UNSUPPORTED_PATTERNS
    )


def split_context_documents(context: str) -> list[tuple[str, str]]:
    """Context를 ("문서 N", 문서본문) 목록으로 분리한다."""
    matches = list(re.finditer(r"(?m)^\[문서 (\d+)\]\s*$", context))
    documents: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
        documents.append((f"문서 {match.group(1)}", context[start:end].strip()))
    return documents


def locate_evidence_document(context: str, quote: str) -> str | None:
    """quote가 실제로 들어 있는 문서 번호를 Context에서 직접 찾는다."""
    for document, body in split_context_documents(context):
        if quote in body:
            return document
    return None


def normalize_field_name_for_match(value: str) -> str:
    """필드명 비교용 정규화."""
    text = (value or "").strip().casefold()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text


def find_context_line_by_field(
    context: str,
    field: str,
) -> tuple[str, str] | None:
    """
    HCX가 evidence.quote를 원문 그대로 복사하지 못했더라도,
    evidence.field가 실제 Context 필드와 매칭되면
    Context의 원본 '필드명: 값' 라인을 복구한다.

    반환:
      ("문서 1", "계약 종료일: 2024-10-31")
    """
    field_key = normalize_field_name_for_match(field)

    if not field_key:
        return None

    # 너무 일반적인 필드는 자동 복구하지 않는다.
    generic_fields = {
        "본문",
        "내용",
        "기타",
        "정보",
        "근거",
    }
    if field.strip() in generic_fields:
        return None

    for document, body in split_context_documents(context):
        for line in body.splitlines():
            stripped = line.strip()

            if ":" not in stripped:
                continue

            line_field, _ = stripped.split(":", 1)

            if (
                normalize_field_name_for_match(line_field)
                == field_key
            ):
                return document, stripped

    return None


def repair_evidence_item(
    context: str,
    field: str,
    quote: str,
) -> dict[str, str] | None:
    """
    evidence를 deterministic하게 검증/복구한다.

    1. quote가 Context에 그대로 존재하면 그대로 사용
    2. 없으면 field로 실제 Context 라인을 찾아 원문 quote로 교체
    3. 둘 다 실패하면 None
    """
    quote = safe_text(quote, 1500).strip()
    field = safe_text(field, 200).strip()

    if quote:
        document = locate_evidence_document(
            context,
            quote,
        )

        if document is not None:
            return {
                "document": document,
                "field": normalize_evidence_field(
                    field,
                    quote,
                ),
                "quote": quote,
            }

    repaired = find_context_line_by_field(
        context,
        field,
    )

    if repaired is None:
        return None

    document, repaired_quote = repaired

    return {
        "document": document,
        "field": normalize_evidence_field(
            field,
            repaired_quote,
        ),
        "quote": repaired_quote,
    }


def normalize_evidence_field(field: str, quote: str) -> str:
    """field가 비어 있거나 부정확하면 '필드명: 값' 형태의 quote에서 필드명을 복구한다."""
    field = safe_text(field, 200).strip()
    if field:
        return field
    if ":" in quote:
        return quote.split(":", 1)[0].strip()
    return "본문"


ALLOWED_GENERATED_TASK_TYPES = {
    "정보추출",
    "조건결합",
    "계산",
    "요약",
    "다중조회",
    "비교연산",
    "복합추론",
    "근거부족",
}


TASK_TYPE_ALIASES = {
    # 사실 조회 계열
    "사실 추출": "정보추출",
    "사실추출": "정보추출",
    "정보 추출": "정보추출",
    "정보조회": "정보추출",
    "사실 조회": "정보추출",

    # 조건 결합
    "조건 결합": "조건결합",

    # 비교/연산
    "비교 연산": "비교연산",
    "비교": "비교연산",

    # 복합 추론
    "복합 추론": "복합추론",
    "복합 문서 추론": "복합추론",

    # 근거 부족
    "근거 부족": "근거부족",
    "답변불가": "근거부족",
    "답변 불가": "근거부족",
}


def normalize_task_type(value: str) -> str:
    text = safe_text(value, 30).strip()

    if text in ALLOWED_GENERATED_TASK_TYPES:
        return text

    return TASK_TYPE_ALIASES.get(text, text)



def normalize_qas_container(
    data: dict[str, Any] | list[Any],
) -> list[Any]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        raw_qas = (
            data.get("qas")
            or data.get("questions")
            or []
        )
        return raw_qas if isinstance(raw_qas, list) else []

    return []


def task_type_diversity_summary(
    qas: list[dict[str, Any]],
) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}

    for qa in qas:
        task_type = str(
            qa.get("task_type") or ""
        ).strip()

        if not task_type:
            continue

        counts[task_type] = counts.get(task_type, 0) + 1

    return len(counts), counts


def validate_reviewed_qas_minimal(
    data: dict[str, Any] | list[Any],
    bundle: ContextBundle,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    2차 HCX 검수 후에는 의미 판단을 다시 Python에서 과도하게 하지 않는다.

    남기는 검증:
    - 기본 필드 존재
    - task_type 허용값 정규화
    - 중복 질문 제거
    - answerable=true이면 evidence 최소 1개
    - evidence.quote가 Context에 실제 존재
    - evidence.document는 quote 위치 기준으로 코드가 재결정
    """
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    raw_qas = normalize_qas_container(
        data
    )

    for idx, qa in enumerate(raw_qas):
        if not isinstance(qa, dict):
            errors.append(
                f"qa[{idx}] is not an object"
            )
            continue

        question = safe_text(
            qa.get("question"),
            2000,
        )
        answer = safe_text(
            qa.get("answer"),
            4000,
        )

        raw_task_type = str(
            qa.get("task_type")
            or qa.get("type")
            or "정보추출"
        )
        task_type = normalize_task_type(
            raw_task_type
        )

        if task_type not in ALLOWED_GENERATED_TASK_TYPES:
            errors.append(
                f"qa[{idx}] unsupported task_type: "
                f"{raw_task_type}"
            )
            continue

        if not question or not answer:
            errors.append(
                f"qa[{idx}] empty question/answer"
            )
            continue

        key = normalize_question(
            question
        )
        if not key or key in seen:
            errors.append(
                f"qa[{idx}] duplicate question"
            )
            continue

        answerable = bool(
            qa.get("answerable")
        )

        raw_evidence = (
            qa.get("evidence")
            if isinstance(
                qa.get("evidence"),
                list,
            )
            else []
        )

        verified = []

        for item in raw_evidence:
            if isinstance(item, str):
                quote = safe_text(
                    item,
                    1500,
                )
                field = ""
            elif isinstance(item, dict):
                quote = safe_text(
                    item.get("quote"),
                    1500,
                )
                field = safe_text(
                    item.get("field"),
                    200,
                )
            else:
                continue

            repaired_item = repair_evidence_item(
                bundle.context,
                field,
                quote,
            )

            if repaired_item is None:
                continue

            verified.append(
                repaired_item
            )

        if answerable and not verified:
            errors.append(
                f"qa[{idx}] has no verifiable evidence"
            )
            continue

        seen.add(key)

        valid.append({
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "evidence": verified,
        })

        if len(valid) >= target_count:
            break

    return valid, errors


def validate_qas(
    data: dict[str, Any] | list[Any],
    bundle: ContextBundle,
) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    # HCX가 response_format 없이 생성할 때
    # {"qas": [...]} 대신 최상위 배열 [...]을 반환하는 경우도 허용한다.
    if isinstance(data, list):
        raw_qas = data
    elif isinstance(data, dict):
        raw_qas = data.get("qas") or data.get("questions") or []
    else:
        return [], [
            f"unexpected response type: {type(data).__name__}"
        ]
    if not isinstance(raw_qas, list):
        return [], ["response has neither a qas nor questions array"]
    for idx, qa in enumerate(raw_qas):
        if not isinstance(qa, dict):
            errors.append(f"qa[{idx}] is not an object")
            continue
        question = safe_text(qa.get("question") or qa.get("text"), 2000)
        answer = safe_text(qa.get("answer"), 4000)
        raw_task_type = (
            qa.get("task_type")
            or qa.get("type")
            or "정보추출"
        )

        task_type = normalize_task_type(
            str(raw_task_type)
        )

        if task_type not in ALLOWED_GENERATED_TASK_TYPES:
            errors.append(
                f"qa[{idx}] unsupported task_type: "
                f"{raw_task_type} -> {task_type}"
            )
            continue

        answerable = bool(qa.get("answerable"))
        evidence = qa.get("evidence") if isinstance(qa.get("evidence"), list) else []
        key = normalize_question(question)
        if not question or not answer or len(question) < 4 or key in seen:
            errors.append(f"qa[{idx}] empty or duplicate")
            continue

        # 단일 문서에서 지원되지 않는 최상급/순위/비교 집합 가정 방지
        if (
            answerable
            and has_unsupported_single_context_comparison(
                question,
                bundle,
            )
        ):
            errors.append(
                f"qa[{idx}] unsupported comparison/superlative "
                "for a single-document context"
            )
            continue

        verified = []
        for item in evidence:
            if isinstance(item, str):
                quote = safe_text(item, 1500)
                field = ""
            elif isinstance(item, dict):
                quote = safe_text(item.get("quote"), 1500)
                field = safe_text(item.get("field"), 200)
            else:
                continue

            if not quote:
                continue

            # 모델이 적은 document 문자열은 신뢰하지 않는다.
            # quote가 원문 그대로 존재하지 않으면 field 기준으로
            # Context의 실제 원문 라인을 복구한다.
            repaired_item = repair_evidence_item(
                bundle.context,
                field,
                quote,
            )

            if repaired_item is None:
                continue

            verified.append(
                repaired_item
            )
        if answerable and not verified:
            errors.append(f"qa[{idx}] has no verifiable evidence")
            continue

        # 금융 QA에서는 답변 숫자가 반드시 evidence에 직접 근거해야 한다.
        # answer에 숫자가 있는데 evidence.quote에 없는 숫자가 있으면 reject.
        if answerable:
            numeric_ok, unsupported_numbers = (
                validate_numeric_grounding(
                    answer,
                    verified,
                )
            )

            if not numeric_ok:
                errors.append(
                    f"qa[{idx}] answer contains unsupported numeric values: "
                    f"{sorted(unsupported_numbers)}"
                )
                continue

        if not answerable:
            if not any(
                token in answer
                for token in (
                    "확인",
                    "부족",
                    "알 수",
                    "제공되지",
                    "포함",
                    "없어",
                    "없으",
                )
            ):
                errors.append(
                    f"qa[{idx}] unanswerable response is unclear"
                )
                continue

            # 너무 짧은 "알 수 없습니다" 식 응답은 학습 품질이 낮으므로 제외
            compact_answer = re.sub(r"\s+", "", answer)
            if compact_answer in {
                "알수없습니다.",
                "확인할수없습니다.",
                "제공된문서에서알수없습니다.",
            }:
                errors.append(
                    f"qa[{idx}] unanswerable response lacks "
                    "a concrete missing-information reason"
                )
                continue
        seen.add(key)
        valid.append({
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "evidence": verified,
        })
    return valid, errors


def training_text(question: str, context: str, include_context: bool) -> str:
    if not include_context:
        return question
    return f"{context}\n\n[질문]\n{question}"


def format_evidence_text(
    field: str,
    quote: str,
) -> str:
    """
    evidence.field와 quote의 중복을 방지한다.

    예:
      field="관계"
      quote="회사와의 관계: 자회사"
        -> "회사와의 관계: 자회사"

      field="회사와의 관계"
      quote="회사와의 관계: 자회사"
        -> "회사와의 관계: 자회사"

      field="계약금액"
      quote="97,000,000,000"
        -> "계약금액: 97,000,000,000"
    """
    field = (field or "").strip()
    quote = (quote or "").strip()

    if not quote:
        return ""

    if not field or field == "본문":
        return quote

    # quote 자체가 이미 "어떤 필드명: 값" 형태면 quote를 그대로 사용한다.
    # 모델이 field를 "관계"처럼 짧게 줘도
    # "관계: 회사와의 관계: 자회사" 같은 중복을 만들지 않는다.
    if re.match(r"^[^:\n]{1,100}\s*:", quote):
        return quote

    # quote가 field로 이미 시작하는 경우도 그대로 사용
    if re.match(
        rf"^{re.escape(field)}\s*[:：]?",
        quote,
    ):
        return quote

    return f"{field}: {quote}"


def training_completion(qa: dict[str, Any]) -> str:
    lines = [f"답변: {qa['answer']}", "근거:"]

    if qa["evidence"]:
        for item in qa["evidence"]:
            document = item["document"]
            field = item["field"].strip()
            quote = item["quote"].strip()

            evidence_text = format_evidence_text(
                field,
                quote,
            )

            lines.append(
                f"- {document} | {evidence_text}"
            )
    else:
        lines.append("- 제공된 문서에서 답변에 필요한 정보를 확인할 수 없음")

    return "\n".join(lines)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
                cache[item["key"]] = item["result"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def append_cache(path: Path, key: str, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")


def write_rows(rows: list[dict[str, Any]], output: Path, output_format: str, system_prompt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = SYSTEM_COLUMNS if system_prompt else COLUMNS
    if output_format == "csv":
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    else:
        with output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps({column: row.get(column, "") for column in columns}, ensure_ascii=False) + "\n")


def build_rows(
    bundles: Iterable[ContextBundle],
    client: ClovaClient,
    single_count: int,
    multi_count: int,
    max_tokens: int,
    temperature: float,
    include_context: bool,
    system_prompt: str,
    cache_path: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    all_errors: list[str] = []
    cache = load_cache(
        cache_path
    )
    global_seen: set[str] = set()

    for bundle_index, bundle in enumerate(
        bundles
    ):
        if limit and bundle_index >= limit:
            break

        target_count = (
            multi_count
            if bundle.mode == "multi"
            else single_count
        )

        # 1차에서는 최종 목표보다 후보를 조금 더 많이 생성
        candidate_count = max(
            target_count + 2,
            target_count,
        )

        generator_prompt = generation_prompt(
            bundle,
            candidate_count,
        )

        generator_key = hashlib.sha256(
            (
                "GENERATOR_V4_EVIDENCE_REPAIR|"
                + client.model
                + generator_prompt
            ).encode("utf-8")
        ).hexdigest()

        if generator_key in cache:
            generated = cache[
                generator_key
            ]
        else:
            print(
                f"[bundle {bundle_index}] "
                f"1차 HCX 후보 생성: "
                f"{candidate_count}개 요청"
            )
            generated = client.generate(
                generator_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            append_cache(
                cache_path,
                generator_key,
                generated,
            )

        # 1차 결과는 reviewer에 넘기기 위해
        # 최소한 리스트 형태로만 정규화
        raw_candidates = (
            normalize_qas_container(
                generated
            )
        )

        if not raw_candidates:
            all_errors.append(
                f"bundle[{bundle_index}] "
                "generator returned no QA candidates"
            )
            continue

        # 2차 HCX가 전체 후보를 한 번에 검수/수정/보충
        review_input_hash = json.dumps(
            raw_candidates,
            ensure_ascii=False,
            sort_keys=True,
        )

        reviewer_key = hashlib.sha256(
            (
                "REVIEWER_V4_EVIDENCE_REPAIR|"
                + client.model
                + bundle.context
                + review_input_hash
                + str(target_count)
            ).encode("utf-8")
        ).hexdigest()

        if reviewer_key in cache:
            reviewed = cache[
                reviewer_key
            ]
        else:
            print(
                f"[bundle {bundle_index}] "
                f"2차 HCX 검수: "
                f"최종 {target_count}개 요청"
            )
            reviewed = client.review(
                bundle=bundle,
                candidates=raw_candidates,
                target_count=target_count,
                max_tokens=max_tokens,
            )
            append_cache(
                cache_path,
                reviewer_key,
                reviewed,
            )

        # Python은 deterministic 최소 검증만 수행
        qas, errors = (
            validate_reviewed_qas_minimal(
                reviewed,
                bundle,
                target_count,
            )
        )

        all_errors.extend(
            f"bundle[{bundle_index}] "
            f"{error}"
            for error in errors
        )

        diversity_count, diversity_counts = (
            task_type_diversity_summary(qas)
        )

        if (
            target_count >= 4
            and len(qas) >= 4
            and diversity_count < 3
        ):
            all_errors.append(
                f"bundle[{bundle_index}] "
                f"task diversity warning: only "
                f"{diversity_count} task types "
                f"{diversity_counts}"
            )

        if len(qas) < target_count:
            all_errors.append(
                f"bundle[{bundle_index}] "
                f"reviewer returned only "
                f"{len(qas)}/{target_count} "
                f"valid QAs after minimal validation"
            )

        for qa in qas:
            dedupe_key = normalize_question(
                qa["question"]
            )

            if dedupe_key in global_seen:
                continue

            global_seen.add(
                dedupe_key
            )

            row: dict[str, Any] = {
                "C_ID": len(rows),
                "T_ID": 0,
                "Text": training_text(
                    qa["question"],
                    bundle.context,
                    include_context,
                ),
                "Completion": training_completion(
                    qa
                ),
            }

            if system_prompt:
                row[
                    "System_Prompt"
                ] = system_prompt

            if (
                len(row["Text"])
                + len(row["Completion"])
                > 8000
            ):
                all_errors.append(
                    f"bundle[{bundle_index}] "
                    "row exceeds 8000 characters"
                )
                continue

            rows.append(row)

    return rows, all_errors


def main() -> None:
    """원래 evidence-repair 코드 그대로 단일 문서만 테스트한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    TEST_INPUT_PATH = Path(
        r"C:\Users\User\.vscode\mirea_asset\corpus\raw\exchange\HD현대일렉트릭\20230131800162\20230131800162.xml"
    )

    QUESTIONS_PER_DOCUMENT = 4
    MAX_CONTEXT_CHARS = 5500
    MAX_GENERATION_TOKENS = 4096
    GENERATION_TEMPERATURE = 0.65
    API_TIMEOUT_SECONDS = 120
    API_RETRIES = 3

    # 원래 evidence-repair 코드의 기본 cache 경로 그대로 사용.
    CACHE_PATH = PROJECT_ROOT / "Dataset" / ".qa_generation_cache.jsonl"

    OUTPUT_JSON = (
        PROJECT_ROOT
        / "Dataset"
        / "single_quick_test_result.json"
    )

    if not TEST_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"TEST_INPUT_PATH not found: {TEST_INPUT_PATH}"
        )

    if not CLOVA_STUDIO_API_KEY:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY is empty in Config.py"
        )

    print("=" * 70)
    print("Single-document exact restored test")
    print(f"Input : {TEST_INPUT_PATH}")
    print(f"QA    : {QUESTIONS_PER_DOCUMENT}")
    print(f"Cache : {CACHE_PATH}")
    print("=" * 70)

    records = load_input_records(TEST_INPUT_PATH)
    if not records:
        raise RuntimeError(
            f"No usable record found: {TEST_INPUT_PATH}"
        )

    bundles = make_bundles(
        [records[0]],
        max_context_chars=MAX_CONTEXT_CHARS,
        multi_doc_size=3,
        include_multi=False,
    )
    if not bundles:
        raise RuntimeError(
            "Failed to create single-document bundle."
        )

    bundle = bundles[0]

    print("\n" + "=" * 70)
    print("Generated Context")
    print("=" * 70)
    print(bundle.context)
    print("=" * 70)

    client = ClovaClient(
        CLOVA_STUDIO_API_KEY,
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        API_TIMEOUT_SECONDS,
        API_RETRIES,
    )

    rows, errors = build_rows(
        [bundle],
        client,
        single_count=QUESTIONS_PER_DOCUMENT,
        multi_count=QUESTIONS_PER_DOCUMENT,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        include_context=True,
        system_prompt="",
        cache_path=CACHE_PATH,
        limit=0,
    )

    print("\n" + "=" * 70)
    print(f"Generated rows: {len(rows)}")
    print("=" * 70)

    for index, row in enumerate(rows, start=1):
        print(f"\n[QA {index}]")
        print(row.get("Text", ""))
        print("\n[Completion]")
        print(row.get("Completion", ""))
        print("-" * 70)

    if errors:
        print("\n[Errors / Warnings]")
        for error in errors:
            print("-", error)

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "input": str(TEST_INPUT_PATH),
                "generated_count": len(rows),
                "error_count": len(errors),
                "rows": rows,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()