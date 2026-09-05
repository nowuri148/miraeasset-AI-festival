"""
calculate.py

다중 조회 및 비교·연산(Task 2) 실행기.

전체 흐름
---------
question
  + keyword_extractor의 정규화 결과(hint)
  + retrieved chunks
        ↓
[1] HyperCLOVA X / LLM
    - basic_function.py의 허용 연산 중 무엇을 쓸지 선택
    - chunk에서 계산에 필요한 값만 추출
    - 사용한 source_id 선택
        ↓
[2] basic_function.py
    - deterministic Python 계산
        ↓
[3] HyperCLOVA X / LLM
    - 계산 결과를 공시 근거와 함께 최소 답변으로 생성
        ↓
[4] validation으로 넘길 수 있는 구조화 dict return

중요
----
- calculate.py는 특정 HyperCLOVA X SDK에 강하게 결합하지 않는다.
- llm_callable(prompt: str) -> str 형태의 함수를 외부에서 주입한다.
- 기존 프로젝트의 HyperCLOVA X client 함수만 adapter로 감싸서 전달하면 된다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from basic_function import (
    get_operation_specs,
    safe_execute_operation,
    supported_operations,
)


LLMCallable = Callable[[str], str]


# =============================================================================
# Exceptions
# =============================================================================

class CalculateError(RuntimeError):
    pass


class LLMOutputError(CalculateError):
    pass


class CalculationPlanError(CalculateError):
    pass


# =============================================================================
# Data models
# =============================================================================

@dataclass
class SourceItem:
    source_id: str
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    company: Optional[str] = None
    report_name: Optional[str] = None
    report_type: Optional[str] = None
    rcept_dt: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None


@dataclass
class CalculationPlan:
    answerable: bool
    operation: Optional[str]
    arguments: Dict[str, Any]
    unit: Optional[str]
    source_ids: List[str]
    extracted_facts: List[Dict[str, Any]]
    reason: str


# =============================================================================
# Chunk normalization
# =============================================================================

def normalize_chunks(chunks: Sequence[Any]) -> List[Dict[str, Any]]:
    """
    검색기가 어떤 형태의 chunk를 주더라도 LLM 입력 형식을 최대한 통일한다.

    지원:
    - str
    - dict
    - 기타 객체(str 변환)

    dict에서 자주 쓰는 key를 최대한 보존한다.
    """
    normalized: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        default_source_id = f"S{i + 1}"

        if isinstance(chunk, str):
            normalized.append(
                {
                    "source_id": default_source_id,
                    "chunk_id": None,
                    "doc_id": None,
                    "company": None,
                    "report_name": None,
                    "report_type": None,
                    "rcept_dt": None,
                    "title": None,
                    "text": chunk,
                }
            )
            continue

        if isinstance(chunk, Mapping):
            source_id = str(
                chunk.get("source_id")
                or chunk.get("chunk_id")
                or chunk.get("id")
                or default_source_id
            )

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk_text")
                or chunk.get("page_content")
                or ""
            )

            normalized.append(
                {
                    "source_id": source_id,
                    "chunk_id": _none_or_str(chunk.get("chunk_id")),
                    "doc_id": _none_or_str(
                        chunk.get("doc_id")
                        or chunk.get("document_id")
                        or chunk.get("rcept_no")
                    ),
                    "company": _none_or_str(
                        chunk.get("company")
                        or chunk.get("corp_name")
                    ),
                    "report_name": _none_or_str(
                        chunk.get("report_name")
                        or chunk.get("report_nm")
                    ),
                    "report_type": _none_or_str(
                        chunk.get("report_type")
                        or chunk.get("normalized_report_type")
                    ),
                    "rcept_dt": _none_or_str(
                        chunk.get("rcept_dt")
                        or chunk.get("date")
                        or chunk.get("filing_date")
                    ),
                    "title": _none_or_str(chunk.get("title")),
                    "text": str(text),
                }
            )
            continue

        normalized.append(
            {
                "source_id": default_source_id,
                "chunk_id": None,
                "doc_id": None,
                "company": None,
                "report_name": None,
                "report_type": None,
                "rcept_dt": None,
                "title": None,
                "text": str(chunk),
            }
        )

    return normalized


def _none_or_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


# =============================================================================
# JSON parsing
# =============================================================================

def parse_json_object(text: str) -> Dict[str, Any]:
    """
    LLM이 ```json ... ``` 또는 앞뒤 설명을 붙여도 첫 JSON object를 복구한다.
    가능한 한 strict JSON 출력을 프롬프트에서 요구하되 방어적으로 처리한다.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError("LLM returned an empty response.")

    text = text.strip()

    # fenced code 제거
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # 그대로 먼저 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 첫 '{'부터 brace balancing
    start = text.find("{")
    if start < 0:
        raise LLMOutputError(f"No JSON object found in LLM output: {text[:300]}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise LLMOutputError(
                        f"Invalid JSON object from LLM: {candidate[:500]}"
                    ) from exc

                if not isinstance(obj, dict):
                    raise LLMOutputError("LLM JSON output must be an object.")
                return obj

    raise LLMOutputError("Unclosed JSON object in LLM output.")


# =============================================================================
# Prompt construction
# =============================================================================

def build_operation_prompt(
    question: str,
    hint: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> str:
    specs = json.dumps(get_operation_specs(), ensure_ascii=False, indent=2)
    hint_json = json.dumps(hint, ensure_ascii=False, indent=2)
    chunk_json = json.dumps(list(chunks), ensure_ascii=False, indent=2)

    return f"""
당신은 공시 데이터 기반 '다중 조회 및 비교·연산'의 계산 실행 계획 생성기입니다.

목표:
사용자 질문, keyword_extractor의 정규화 힌트, 검색된 공시 chunk를 함께 읽고
basic_function.py에서 실행할 정확한 operation과 arguments를 JSON 하나로 생성하십시오.

중요 규칙:
1. 계산은 직접 하지 마십시오. 어떤 Python 함수를 호출할지와 그 인자만 생성하십시오.
2. operation은 아래 허용 목록 중 정확히 하나만 선택하십시오.
3. 질문의 자연어 표현을 단순 keyword-to-function 방식으로 기계적으로 매핑하지 마십시오.
   질문 전체 의미, hint, 실제 chunk의 값과 단위를 함께 판단하십시오.
4. arguments에 넣는 모든 사실/숫자는 반드시 제공된 chunk에서 확인할 수 있어야 합니다.
5. 같은 계산에 들어가는 값은 단위를 통일하십시오.
   원/천원/백만원/억원/조원 변환이 필요하면 가능한 경우 값을 하나의 동일 단위로 정규화한 뒤 넣으십시오.
6. 증감률은 기준시점(old)과 비교시점(new)의 방향을 정확히 지키십시오.
7. '몇 %p 변했는가'와 '몇 % 증감했는가'를 구분하십시오.
8. 여러 기업/유형/기간을 합산해야 하면 중복 공시/정정 공시 때문에 같은 사실이 중복 집계되지 않았는지 chunk를 확인하십시오.
9. 필요한 값이 부족하거나 질문에 답할 수 없으면 answerable=false로 하십시오.
10. source_ids에는 실제 계산 인자를 추출한 chunk의 source_id만 넣으십시오.
11. 질문에 답하는 데 계산이 사실상 필요 없으면 answerable=false로 하지 말고,
    operation을 선택할 수 없는 이유를 reason에 적으십시오. 단, 이 모듈은 Task 2 계산 전용이므로
    가능하면 허용 연산으로 표현하십시오.
12. 출력은 JSON 하나만 반환하고 설명, 마크다운, 코드펜스를 추가하지 마십시오.

허용 operation:
{specs}

사용자 질문:
{question}

keyword_extractor 힌트:
{hint_json}

검색된 chunk:
{chunk_json}

출력 스키마:
{{
  "answerable": true,
  "operation": "허용 operation 중 하나 또는 null",
  "arguments": {{}},
  "unit": "최종 계산값의 단위 또는 null",
  "source_ids": ["S1", "S2"],
  "extracted_facts": [
    {{
      "label": "계산에 사용한 값의 의미",
      "value": 0,
      "unit": "억원",
      "source_id": "S1"
    }}
  ],
  "reason": "왜 이 operation과 인자를 선택했는지 짧게"
}}
""".strip()


def build_answer_prompt(
    question: str,
    hint: Mapping[str, Any],
    plan: Mapping[str, Any],
    calculation: Mapping[str, Any],
    selected_sources: Sequence[Mapping[str, Any]],
) -> str:
    return f"""
당신은 공시 데이터 기반 질의응답 Agent의 최종 답변 생성기입니다.

아래 사용자 질문에 대해 Python 계산 결과와 선택된 공시 근거만 사용해 최소한의 답변을 생성하십시오.

규칙:
1. 계산 결과를 임의로 다시 계산하거나 수정하지 마십시오.
2. 선택된 공시 근거에 없는 사실을 추가하지 마십시오.
3. 질문의 핵심 결론과 필요한 수치를 간결하게 포함하십시오.
4. 단위가 있으면 반드시 명시하십시오.
5. 근거가 부족하면 그 한계를 명시하십시오.
6. 출처 목록은 별도로 Python이 붙일 것이므로 answer 문자열에는 장황한 출처 나열을 하지 않아도 됩니다.
7. 출력은 JSON 하나만 반환하십시오.

사용자 질문:
{question}

keyword_extractor 힌트:
{json.dumps(hint, ensure_ascii=False)}

계산 계획:
{json.dumps(plan, ensure_ascii=False)}

Python 계산 결과:
{json.dumps(calculation, ensure_ascii=False)}

선택된 공시 근거:
{json.dumps(list(selected_sources), ensure_ascii=False)}

출력:
{{
  "answer": "최종 최소 답변"
}}
""".strip()


# =============================================================================
# Plan validation
# =============================================================================

def validate_plan(plan_obj: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> CalculationPlan:
    answerable = bool(plan_obj.get("answerable", False))
    operation = plan_obj.get("operation")
    arguments = plan_obj.get("arguments") or {}
    unit = plan_obj.get("unit")
    source_ids = plan_obj.get("source_ids") or []
    extracted_facts = plan_obj.get("extracted_facts") or []
    reason = str(plan_obj.get("reason") or "")

    if not isinstance(arguments, dict):
        raise CalculationPlanError("plan.arguments must be an object.")

    if not isinstance(source_ids, list):
        raise CalculationPlanError("plan.source_ids must be a list.")

    if not isinstance(extracted_facts, list):
        raise CalculationPlanError("plan.extracted_facts must be a list.")

    known_source_ids = {str(c["source_id"]) for c in chunks}
    unknown = [str(s) for s in source_ids if str(s) not in known_source_ids]

    if unknown:
        raise CalculationPlanError(
            f"Unknown source_ids in plan: {unknown}. "
            f"Known: {sorted(known_source_ids)}"
        )

    if answerable:
        if not isinstance(operation, str) or operation not in supported_operations():
            raise CalculationPlanError(
                f"Invalid operation: {operation!r}. "
                f"Supported: {supported_operations()}"
            )
    else:
        operation = None

    return CalculationPlan(
        answerable=answerable,
        operation=operation,
        arguments=arguments,
        unit=None if unit is None else str(unit),
        source_ids=[str(s) for s in source_ids],
        extracted_facts=extracted_facts,
        reason=reason,
    )


def select_sources(
    chunks: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str],
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    wanted = set(source_ids)
    result: List[Dict[str, Any]] = []

    for chunk in chunks:
        if str(chunk["source_id"]) not in wanted:
            continue

        source = dict(chunk)
        if not include_text:
            source.pop("text", None)
        result.append(source)

    return result


# =============================================================================
# Main calculation pipeline
# =============================================================================

def calculate(
    question: str,
    hint: Mapping[str, Any],
    chunks: Sequence[Any],
    llm_callable: LLMCallable,
    *,
    generate_answer: bool = True,
) -> Dict[str, Any]:
    """
    Task 2 계산 파이프라인의 메인 함수.

    Parameters
    ----------
    question:
        사용자 원 질문.
    hint:
        keyword_extractor가 만든 정규화 결과.
        예: task_type, companies, periods, metrics, actions 등.
    chunks:
        검색 결과 chunk 목록.
    llm_callable:
        prompt(str) -> response_text(str)
        형태의 HyperCLOVA X 호출 함수.
    generate_answer:
        True면 계산 후 최종 답변 LLM도 호출한다.
        False면 Python 계산 결과를 기반으로 기계적인 최소 답변을 만든다.

    Returns
    -------
    dict
        validation 단계에 그대로 전달할 수 있도록 최소한 아래 필드를 보장한다.

        {
          "answer": "...",
          "sources": [...],
          "retrieved_context": "...",
          "calculation": {...},
          "validation_payload": {
              "question": "...",
              "answer": "...",
              "sources": [...],
              "retrieved_context": "..."
          }
        }
    """
    if not question or not question.strip():
        raise CalculateError("question must not be empty.")
    if not isinstance(hint, Mapping):
        raise CalculateError("hint must be a mapping.")
    if not callable(llm_callable):
        raise CalculateError("llm_callable must be callable.")

    normalized_chunks = normalize_chunks(chunks)

    if not normalized_chunks:
        return _unanswerable_result(
            question=question,
            reason="검색된 공시 chunk가 없습니다.",
            chunks=[],
        )

    # ---------------------------------------------------------------------
    # 1) LLM: operation + arguments + sources 결정
    # ---------------------------------------------------------------------
    operation_prompt = build_operation_prompt(
        question=question,
        hint=hint,
        chunks=normalized_chunks,
    )

    raw_plan = llm_callable(operation_prompt)
    plan_obj = parse_json_object(raw_plan)
    plan = validate_plan(plan_obj, normalized_chunks)

    if not plan.answerable or not plan.operation:
        return _unanswerable_result(
            question=question,
            reason=plan.reason or "제공된 공시 근거만으로 계산에 필요한 값을 확정할 수 없습니다.",
            chunks=select_sources(
                normalized_chunks,
                plan.source_ids,
                include_text=True,
            ),
            plan=asdict(plan),
        )

    # ---------------------------------------------------------------------
    # 2) Python: deterministic calculation
    # ---------------------------------------------------------------------
    calculation = safe_execute_operation(
        plan.operation,
        **plan.arguments,
    )

    selected_sources_full = select_sources(
        normalized_chunks,
        plan.source_ids,
        include_text=True,
    )

    selected_sources_meta = select_sources(
        normalized_chunks,
        plan.source_ids,
        include_text=False,
    )

    if not calculation["success"]:
        return _calculation_error_result(
            question=question,
            plan=asdict(plan),
            calculation=calculation,
            sources_full=selected_sources_full,
            sources_meta=selected_sources_meta,
        )

    # ---------------------------------------------------------------------
    # 3) LLM: 계산 결과 + 근거로 최소 답변 생성
    # ---------------------------------------------------------------------
    if generate_answer:
        answer_prompt = build_answer_prompt(
            question=question,
            hint=hint,
            plan=asdict(plan),
            calculation=calculation,
            selected_sources=selected_sources_full,
        )

        raw_answer = llm_callable(answer_prompt)
        answer_obj = parse_json_object(raw_answer)
        answer = str(answer_obj.get("answer") or "").strip()

        if not answer:
            answer = _fallback_answer(plan, calculation)
    else:
        answer = _fallback_answer(plan, calculation)

    retrieved_context = _build_retrieved_context(selected_sources_full)

    # ---------------------------------------------------------------------
    # 4) validation으로 넘길 최종 return
    # ---------------------------------------------------------------------
    result = {
        "answerable": True,

        # validation에 최소 필요
        "answer": answer,
        "sources": selected_sources_meta,

        # validation이 근거 본문까지 검증할 수 있도록 제공
        "retrieved_context": retrieved_context,

        # 디버깅/추론 검증용
        "calculation": {
            "plan": asdict(plan),
            "execution": calculation,
        },

        # validation 함수가 별도 스키마를 원할 때 바로 넘기기 쉬운 payload
        "validation_payload": {
            "question": question,
            "answer": answer,
            "sources": selected_sources_meta,
            "retrieved_context": retrieved_context,
            "calculation": {
                "operation": plan.operation,
                "arguments": plan.arguments,
                "result": calculation["result"],
                "unit": plan.unit,
                "source_ids": plan.source_ids,
                "extracted_facts": plan.extracted_facts,
            },
        },
    }

    return result


# =============================================================================
# Failure returns
# =============================================================================

def _unanswerable_result(
    question: str,
    reason: str,
    chunks: Sequence[Mapping[str, Any]],
    plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    sources_meta = []
    for chunk in chunks:
        source = dict(chunk)
        source.pop("text", None)
        sources_meta.append(source)

    retrieved_context = _build_retrieved_context(chunks)

    answer = f"제공된 공시에서 확인할 수 없습니다. {reason}".strip()

    return {
        "answerable": False,
        "answer": answer,
        "sources": sources_meta,
        "retrieved_context": retrieved_context,
        "calculation": {
            "plan": dict(plan or {}),
            "execution": None,
        },
        "validation_payload": {
            "question": question,
            "answer": answer,
            "sources": sources_meta,
            "retrieved_context": retrieved_context,
            "calculation": None,
        },
    }


def _calculation_error_result(
    question: str,
    plan: Mapping[str, Any],
    calculation: Mapping[str, Any],
    sources_full: Sequence[Mapping[str, Any]],
    sources_meta: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    answer = (
        "검색된 공시 근거는 확인했지만 계산을 완료할 수 없습니다. "
        f"계산 오류: {calculation.get('error')}"
    )
    retrieved_context = _build_retrieved_context(sources_full)

    return {
        "answerable": False,
        "answer": answer,
        "sources": list(sources_meta),
        "retrieved_context": retrieved_context,
        "calculation": {
            "plan": dict(plan),
            "execution": dict(calculation),
        },
        "validation_payload": {
            "question": question,
            "answer": answer,
            "sources": list(sources_meta),
            "retrieved_context": retrieved_context,
            "calculation": {
                "operation": plan.get("operation"),
                "arguments": plan.get("arguments"),
                "result": None,
                "error": calculation.get("error"),
                "source_ids": plan.get("source_ids", []),
            },
        },
    }


# =============================================================================
# Output helpers
# =============================================================================

def _fallback_answer(
    plan: CalculationPlan,
    calculation: Mapping[str, Any],
) -> str:
    unit = f" {plan.unit}" if plan.unit else ""
    return f"계산 결과는 {calculation.get('result')}{unit}입니다."


def _build_retrieved_context(chunks: Sequence[Mapping[str, Any]]) -> str:
    blocks: List[str] = []

    for chunk in chunks:
        metadata_parts = [
            f"source_id={chunk.get('source_id')}",
            f"doc_id={chunk.get('doc_id')}",
            f"company={chunk.get('company')}",
            f"report_name={chunk.get('report_name')}",
            f"rcept_dt={chunk.get('rcept_dt')}",
        ]
        metadata = ", ".join(
            part for part in metadata_parts
            if not part.endswith("=None")
        )
        text = str(chunk.get("text") or "")
        blocks.append(f"[{metadata}]\n{text}")

    return "\n\n".join(blocks)


# =============================================================================
# Example adapter / usage
# =============================================================================

if __name__ == "__main__":
    def mock_llm(prompt: str) -> str:
        """
        실제 사용 시 이 부분을 기존 HyperCLOVA X 호출 함수로 교체한다.
        이 mock은 파일 실행 예시일 뿐이다.
        """
        if "계산 실행 계획 생성기" in prompt:
            return json.dumps(
                {
                    "answerable": True,
                    "operation": "argmax",
                    "arguments": {
                        "values": {
                            "A사": 12500,
                            "B사": 9800,
                        }
                    },
                    "unit": "억원",
                    "source_ids": ["A_2025", "B_2025"],
                    "extracted_facts": [
                        {
                            "label": "A사 2025년 설비투자",
                            "value": 12500,
                            "unit": "억원",
                            "source_id": "A_2025",
                        },
                        {
                            "label": "B사 2025년 설비투자",
                            "value": 9800,
                            "unit": "억원",
                            "source_id": "B_2025",
                        },
                    ],
                    "reason": "두 기업 중 설비투자 규모가 더 큰 기업을 찾는 질문",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "answer": "2025년 설비투자 규모는 A사가 1조 2,500억원으로 B사의 9,800억원보다 큽니다."
            },
            ensure_ascii=False,
        )

    example_chunks = [
        {
            "source_id": "A_2025",
            "doc_id": "doc_A",
            "company": "A사",
            "report_name": "2025년 사업보고서",
            "rcept_dt": "2026-03-20",
            "text": "2025년 설비투자 규모는 1조 2,500억원입니다.",
        },
        {
            "source_id": "B_2025",
            "doc_id": "doc_B",
            "company": "B사",
            "report_name": "2025년 사업보고서",
            "rcept_dt": "2026-03-18",
            "text": "2025년 설비투자 규모는 9,800억원입니다.",
        },
    ]

    output = calculate(
        question="A사와 B사 중 2025년 설비투자 규모가 더 큰 기업은?",
        hint={
            "task_type": "다중조회_비교연산",
            "companies": ["A사", "B사"],
            "periods": ["2025"],
            "metrics": ["설비투자"],
            "actions": ["비교"],
        },
        chunks=example_chunks,
        llm_callable=mock_llm,
    )

    print(json.dumps(output, ensure_ascii=False, indent=2))
