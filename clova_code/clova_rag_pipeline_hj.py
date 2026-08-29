#!/usr/bin/env python3
r"""HCX-005 Function Calling + disclosure graph RAG baseline.

Pipeline
--------
1. Ask HCX-005 to call ``search_disclosures`` with structured arguments.
2. Validate those arguments and run ``GraphRetriever.retrieve_from_plan``.
3. Build an Evidence Bundle from graph metadata and source XML/HTML.
4. Ask HCX-005 to answer using only that Evidence Bundle.
5. Append the complete success or failure trace to one UTF-8 JSONL log file.

PowerShell example (run from the repository root):

    $env:CLOVASTUDIO_API_KEY = "YOUR_API_KEY"
    python .\clova_code\clova_rag_pipeline_hj.py

One-shot example:

    python .\clova_code\clova_rag_pipeline_hj.py `
      --question "삼성전자의 2024년 11월 자기주식 취득결정은 어떻게 정정됐어?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.4.1-company-enum-removed"
DEFAULT_MODEL = "HCX-005"
DEFAULT_ENDPOINT_TEMPLATE = (
    "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"
)
DEFAULT_MANIFEST = Path("corpus/derived/manifest_v3.jsonl")
DEFAULT_EDGES = Path("corpus/derived/graph_edges_v1.jsonl")
DEFAULT_CORPUS_ROOT = Path("corpus")
DEFAULT_EVIDENCE_OUTPUT = Path("corpus/derived/evidence_v1.json")
DEFAULT_RESULT_OUTPUT = Path("corpus/derived/rag_result_v1.json")
DEFAULT_LOG_FILE = Path("logs/clova_rag_debug.jsonl")
TOOL_NAME = "search_disclosures"
NOT_APPLICABLE = "not_applicable"


# Every normalized_report_type currently present in manifest_v3.jsonl.
# The Korean explanations are supplied to HCX so that semantic mapping remains
# an LLM responsibility instead of being reimplemented as Python keyword rules.
REPORT_TYPE_DESCRIPTIONS: dict[str, str] = {
    "annual_report": "사업보고서. 한 회계연도의 사업 및 재무 내용을 다루는 정기공시",
    "semiannual_report": "반기보고서. 회계연도 상반기 내용을 다루는 정기공시",
    "quarterly_report": "분기보고서. 분기 또는 3분기 누적 내용을 다루는 정기공시",
    "bonus_issue_decision": "무상증자 결정",
    "business_acquisition_decision": "영업양수 결정. 다른 회사 등의 영업 전부 또는 중요 부분을 양수하는 결정",
    "business_suspension": "영업정지",
    "capital_reduction_decision": "감자 결정. 자본금 감소 결정",
    "company_split_decision": "회사분할 결정",
    "convertible_bond_issuance_decision": "전환사채권 발행 결정",
    "exchangeable_bond_issuance_decision": "교환사채권 발행 결정",
    "facility_investment": "신규 시설투자 또는 시설투자 결정",
    "large_shareholding_general": "주식 등의 대량보유상황보고서 일반보고. 경영권 영향 목적 등을 포함하는 5% 보고",
    "large_shareholding_summary": "주식 등의 대량보유상황보고서 약식보고. 단순투자 등 약식 5% 보고",
    "lawsuit_filing": "소송 등의 제기",
    "material_management_matter": "투자판단 관련 주요경영사항",
    "merger_decision": "회사합병 결정",
    "other_company_shares_acquisition_decision": "타법인 주식 및 출자증권 양수 또는 취득 결정",
    "other_company_shares_disposal_decision": "타법인 주식 및 출자증권 양도 또는 처분 결정",
    "overseas_securities_delisting_decision": "해외 증권시장 주권 등의 상장폐지 결정 또는 상장폐지",
    "overseas_securities_listing_decision": "해외 증권시장 주권 등의 상장 결정 또는 상장",
    "own_convertible_bond_sale_decision": "회사가 보유한 자기전환사채 매도 결정",
    "paid_in_capital_increase_decision": "유상증자 결정",
    "regulatory_capital_debt_security_issuance_decision": "자본으로 인정되는 채무증권 발행 결정",
    "split_merger_decision": "회사분할합병 결정",
    "stock_exchange_or_transfer_decision": "주식교환 또는 주식이전 결정",
    "supply_contract": "단일판매 또는 공급계약 체결",
    "supply_contract_termination": "단일판매 또는 공급계약 해지",
    "tangible_asset_acquisition_decision": "유형자산 양수 또는 취득 결정",
    "tangible_asset_disposal_decision": "유형자산 양도 또는 처분 결정",
    "third_party_convertible_bond_call_option_exercise": "제3자의 전환사채 매수선택권 행사",
    "treasury_stock_acquisition_decision": "자기주식 취득 결정. 자사주 매입 결정과 같은 의미",
    "treasury_stock_disposal_decision": "자기주식 처분 결정. 자사주 매각 또는 교부 결정과 같은 의미",
    "treasury_stock_trust_agreement_conclusion_decision": "자기주식 취득 신탁계약 체결 결정",
    "treasury_stock_trust_agreement_termination_decision": "자기주식 취득 신탁계약 해지 결정",
    "write_down_contingent_capital_security_issuance_decision": "상각형 조건부자본증권 발행 결정",
}


def _import_retriever() -> tuple[Any, Any, Any]:
    """Import correctly both in the repo and beside the retriever during tests."""
    script_dir = Path(__file__).resolve().parent
    candidate_roots = [script_dir, script_dir.parent]
    for candidate in candidate_roots:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from data_checking_code.graph_retriever_string_sentinel_hj import (  # type: ignore
            GraphRetriever,
            PipelineDebugLogger,
            RetrievalError,
        )
    except ModuleNotFoundError:
        try:
            from data_checking_code.graph_retriever_model_hj import (  # type: ignore
                GraphRetriever,
                PipelineDebugLogger,
                RetrievalError,
            )
        except ModuleNotFoundError:
            try:
                from data_checking_code.graph_retriever_schema_contract_hj import (  # type: ignore
                    GraphRetriever,
                    PipelineDebugLogger,
                    RetrievalError,
                )
            except ModuleNotFoundError:
                try:
                    from data_checking_code.graph_retriever_model_withlog_hj import (  # type: ignore
                        GraphRetriever,
                        PipelineDebugLogger,
                        RetrievalError,
                    )
                except ModuleNotFoundError:
                    try:
                        from graph_retriever_string_sentinel_hj import (  # type: ignore
                            GraphRetriever,
                            PipelineDebugLogger,
                            RetrievalError,
                        )
                    except ModuleNotFoundError:
                        try:
                            from graph_retriever_model_hj import (  # type: ignore
                                GraphRetriever,
                                PipelineDebugLogger,
                                RetrievalError,
                            )
                        except ModuleNotFoundError:
                            try:
                                from graph_retriever_schema_contract_hj import (  # type: ignore
                                    GraphRetriever,
                                    PipelineDebugLogger,
                                    RetrievalError,
                                )
                            except ModuleNotFoundError:
                                from graph_retriever_model_withlog_hj import (  # type: ignore
                                    GraphRetriever,
                                    PipelineDebugLogger,
                                    RetrievalError,
                                )
    return GraphRetriever, PipelineDebugLogger, RetrievalError


GraphRetriever, PipelineDebugLogger, RetrievalError = _import_retriever()


class ClovaApiError(RuntimeError):
    """Raised when CLOVA Studio returns an invalid or unsuccessful response."""


class ClovaStudioClient:
    """Minimal standard-library client for CLOVA Studio Chat Completions v3."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE,
        timeout: float = 120.0,
    ) -> None:
        if not api_key.strip():
            raise ClovaApiError("CLOVA Studio API key is empty")
        self.api_key = api_key.strip()
        self.model = model
        self.endpoint = endpoint_template.format(model=model)
        self.timeout = timeout

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ClovaApiError(
                f"CLOVA Studio HTTP {exc.code}: {error_body[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ClovaApiError(f"CLOVA Studio connection failed: {exc.reason}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ClovaApiError("CLOVA Studio returned invalid JSON") from exc

        status = parsed.get("status") or {}
        if status.get("code") not in (None, "20000"):
            raise ClovaApiError(
                f"CLOVA Studio error {status.get('code')}: {status.get('message')}"
            )
        result = parsed.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
            raise ClovaApiError("CLOVA Studio response has no result.message")
        return result


def build_search_tool(retriever: Any) -> dict[str, Any]:
    """Build a CLOVA-compatible Function Calling schema."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "공시 corpus에서 회사, 보고서 유형, 기준연도 또는 접수기간으로 "
                "문서를 찾고 FILED/CORRECTS 관계와 원문 근거를 반환한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "correction_history",
                            "temporal_comparison",
                            "disclosure_lookup",
                        ],
                        "description": "질문의 검색 의도",
                    },
                    "company": {
                        "type": "string",
                        "description": (
                            "사용자 질문에 명시된 정확한 회사명 또는 corp_code. "
                            "회사명을 임의로 변경하거나 추측하지 않는다."
                        ),
                    },
                    "normalized_report_type": {
                        "type": "string",
                        "enum": sorted(
                            set(retriever.report_types) | {NOT_APPLICABLE}
                        ),
                        "description": (
                            "manifest_v3.jsonl에 존재하는 정규화 보고서 유형. "
                            "질문에서 특정할 수 없을 때도 키를 생략하지 말고 "
                            f"{NOT_APPLICABLE}."
                        ),
                    },
                    "base_years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "사업보고서·반기보고서·분기보고서가 다루는 기준연도. "
                            "접수연도와 혼동하지 않는다. 해당하지 않으면 빈 배열."
                        ),
                    },
                    "start_date": {
                        "type": "string",
                        "description": (
                            "비정기공시 접수기간 시작일 YYYYMMDD. "
                            "날짜가 없거나 base_years를 쓰면 not_applicable. "
                            "키는 항상 출력."
                        ),
                    },
                    "end_date": {
                        "type": "string",
                        "description": (
                            "비정기공시 접수기간 종료일 YYYYMMDD. "
                            "날짜가 없거나 base_years를 쓰면 not_applicable. "
                            "키는 항상 출력."
                        ),
                    },
                    "include_corrections": {
                        "type": "boolean",
                        "description": (
                            "원본과 최종 정정 상태를 확인해야 하면 true. "
                            "일반적으로 안전하게 true를 사용한다."
                        ),
                    },
                    "focus_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "문서 제목이 아니라 원문 XML 표에서 찾을 구체적인 "
                            "행·열 제목 또는 항목명. 최대 8개."
                        ),
                    },
                },
                "required": [
                    "intent",
                    "company",
                    "normalized_report_type",
                    "base_years",
                    "start_date",
                    "end_date",
                    "include_corrections",
                    "focus_terms",
                ],
            },
        },
    }


def build_report_type_catalog(retriever: Any) -> str:
    """Return the complete corpus report-type catalog for the HCX prompt."""
    corpus_types = set(retriever.report_types)
    missing_descriptions = sorted(corpus_types - set(REPORT_TYPE_DESCRIPTIONS))
    if missing_descriptions:
        raise ClovaApiError(
            "normalized_report_type 설명이 없는 corpus 값: "
            + ", ".join(missing_descriptions)
        )

    return "\n".join(
        f"- {report_type}: {REPORT_TYPE_DESCRIPTIONS[report_type]}"
        for report_type in sorted(corpus_types)
    )


def build_planner_system_prompt(retriever: Any) -> str:
    catalog = build_report_type_catalog(retriever)
    return f"""당신은 한국 기업공시 질문을 검색 함수 인수로 변환하는 계획 생성기입니다.
사용자 질문에 직접 답하지 말고 반드시 search_disclosures 함수를 호출하십시오.

출력 계약:
1. 함수 스키마에 정의된 8개 필드를 전부 출력합니다. 어떤 필드도 생략하지 않습니다.
2. 적용할 수 없는 normalized_report_type, start_date, end_date는 null 대신
   반드시 문자열 "not_applicable"을 출력합니다.
3. 적용할 기준연도가 없으면 base_years=[], 원문 검색 항목이 없으면 focus_terms=[]입니다.
4. 스키마 enum에 없는 회사명·intent·normalized_report_type은 만들지 않습니다.

normalized_report_type 전체 목록과 의미:
{catalog}

의미 해석 규칙:
1. 질문의 표현을 위 목록에서 의미가 가장 정확히 일치하는 하나의
   normalized_report_type으로 변환합니다. 특정 유형을 판단할 근거가 없을 때만
   "not_applicable"입니다.
2. '자사주 매입', '자기주식 매입', '자기주식 취득결정'은
   treasury_stock_acquisition_decision입니다.
3. '자사주 처분', '자기주식 매각', '자기주식 처분결정'은
   treasury_stock_disposal_decision입니다.
4. '사업보고서', '반기보고서', '분기보고서'는 각각 annual_report,
   semiannual_report, quarterly_report입니다.
5. 정정 내역·변경·최종 상태를 묻는 질문은 intent=correction_history,
   서로 다른 시점이나 연도를 비교하면 intent=temporal_comparison,
   그 밖의 특정 공시 조회는 intent=disclosure_lookup입니다.
6. 정정, 변화, 비교 또는 최종 상태를 확인해야 하면 include_corrections=true입니다.

연도·월·기간 변환 규칙:
1. annual_report, semiannual_report, quarterly_report의 연도는 그 보고서가 다루는
   기준연도이므로 base_years에 정수로 넣습니다. 이 경우 start_date와 end_date는
   모두 "not_applicable"입니다.
2. 그 밖의 비정기공시에서 질문의 연·월·일은 접수일 검색 범위로 변환하고
   base_years=[]로 둡니다.
3. 연도만 있으면 해당 연도의 첫날과 마지막 날입니다.
   예: 2024년 -> start_date="20240101", end_date="20241231".
4. 연도와 월이 있으면 해당 월의 실제 첫날과 마지막 날입니다. 윤년을 반영합니다.
   예: 2024년 2월 -> "20240201"~"20240229".
   예: 2024년 11월 -> "20241101"~"20241130".
5. 하루가 지정되면 시작일과 종료일을 같은 날짜로 둡니다.
   예: 2024년 11월 18일 -> 둘 다 "20241118".
6. 기간이 지정되면 양 끝 날짜를 포함합니다.
   예: 2024년 11월부터 2025년 1월까지 -> "20241101"~"20250131".
7. 질문에 날짜가 없으면 start_date="not_applicable",
   end_date="not_applicable"입니다.
8. 질문에 없는 연도나 날짜를 추측하지 않습니다.

focus_terms 생성 규칙:
1. 문서 제목이나 normalized_report_type을 그대로 반복하지 않습니다.
2. 원문 XML 표에서 실제로 찾을 법한 행 제목·열 제목·공시 항목명을 1~8개 생성합니다.
3. 정정 질문이면 '정정사항', '정정사유', '정정 전', '정정 후'와 함께
   질문 주제의 세부 항목명을 선택합니다.
4. 자기주식 취득 결정의 예: '정정사항', '정정사유', '취득예정주식수',
   '취득예정금액', '취득예상기간', '취득목적', '취득방법'.
5. 공급계약의 예: '체결계약명', '계약금액', '계약상대', '계약기간',
   '계약(수주)일자', '판매ㆍ공급지역'.
6. 정기보고서 핵심 사업 비교의 예: '사업의 내용', '주요 제품 및 서비스',
   '매출 및 수주상황', '연구개발활동', '신규사업'.

금지 사항:
- doc_id, rcept_no, disclosure_chain_id 또는 실제 공시 내용을 추측하지 않습니다.
- 질문의 의미를 넓히기 위해 임의의 회사·보고서 유형·날짜를 추가하지 않습니다.

완전한 출력 예시 1:
질문: 삼성전자의 2024년 11월 자기주식 취득결정은 어떻게 정정됐어?
함수 인수:
{{
  "intent": "correction_history",
  "company": "삼성전자",
  "normalized_report_type": "treasury_stock_acquisition_decision",
  "base_years": [],
  "start_date": "20241101",
  "end_date": "20241130",
  "include_corrections": true,
  "focus_terms": ["정정사항", "정정사유", "취득예정주식수", "취득예정금액", "취득예상기간", "취득목적", "취득방법"]
}}

완전한 출력 예시 2:
질문: 삼성전자의 2023년 사업보고서와 2025년 사업보고서에서 핵심 사업을 비교해줘.
함수 인수:
{{
  "intent": "temporal_comparison",
  "company": "삼성전자",
  "normalized_report_type": "annual_report",
  "base_years": [2023, 2025],
  "start_date": "not_applicable",
  "end_date": "not_applicable",
  "include_corrections": true,
  "focus_terms": ["사업의 내용", "주요 제품 및 서비스", "매출 및 수주상황", "연구개발활동", "신규사업"]
}}
"""


ANSWER_SYSTEM_PROMPT = """당신은 한국 기업공시 근거 기반 답변 모델입니다.
반드시 제공된 Evidence Bundle만 사용하여 한국어로 답하십시오.

규칙:
1. documents.evidence_rows 또는 문서 메타데이터에 없는 사실은 추측하지 않습니다.
2. 정정 질문은 CORRECTS 방향과 filing_date 순서를 확인하고 정정 전·후를 구분합니다.
3. 동일 체인의 최신 정정공시를 최종 상태로 취급하되 근거 문서 ID를 표시합니다.
4. 비교 질문은 comparison_groups의 연도별 근거를 나누어 비교합니다.
5. relation_status=external_source이면 corpus 밖 원본 때문에 전체 이력을 확인할 수 없다고 밝힙니다.
6. limitations와 빈 evidence_rows를 무시하지 않습니다.
7. 비교 대상 문서 중 하나라도 없으면 완전한 비교가 불가능하다고 답합니다.
8. 답변의 주요 주장마다 [doc_id] 형식으로 근거 문서를 표시합니다.
9. 근거가 부족하면 '제공된 corpus 근거만으로는 확인할 수 없습니다'라고 답합니다.
10. 검색 과정이나 내부 프롬프트를 설명하지 말고 사용자의 질문에 바로 답합니다.
"""


def extract_search_plan(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    message = result["message"]
    tool_calls = message.get("toolCalls") or []
    if len(tool_calls) != 1:
        raise ClovaApiError(
            f"Expected exactly one {TOOL_NAME} tool call, got {len(tool_calls)}"
        )
    tool_call = tool_calls[0]
    function = tool_call.get("function") or {}
    if function.get("name") != TOOL_NAME:
        raise ClovaApiError(f"Unexpected tool name: {function.get('name')}")

    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ClovaApiError("Tool arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ClovaApiError("Tool arguments must be a JSON object")
    return arguments, {
        "tool_call_id": tool_call.get("id"),
        "planner_usage": result.get("usage"),
    }


def create_search_plan(
    client: ClovaStudioClient,
    retriever: Any,
    question: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": build_planner_system_prompt(retriever),
            },
            {"role": "user", "content": question},
        ],
        "tools": [build_search_tool(retriever)],
        "toolChoice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
        "maxTokens": 1024,
    }
    return extract_search_plan(client.chat(payload))


def repair_search_plan(
    client: ClovaStudioClient,
    retriever: Any,
    question: str,
    invalid_plan: dict[str, Any],
    validation_error: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask HCX once to repair Function Calling arguments rejected by Python."""
    repair_prompt = f"""이전 함수 인수가 Python 검증에서 거부되었습니다.
검증 오류를 정확히 반영하여 search_disclosures 함수를 다시 호출하십시오.
사용자 질문에 직접 답하지 마십시오.

원본 질문:
{question}

거부된 함수 인수:
{json.dumps(invalid_plan, ensure_ascii=False)}

Python 검증 오류:
{type(validation_error).__name__}: {validation_error}

특히 비정기공시는 base_years를 빈 배열로 두고, 질문의 연월은
start_date/end_date 접수기간으로 표현해야 합니다.
적용되지 않는 normalized_report_type, start_date, end_date에는 JSON null이 아니라
문자열 "not_applicable"을 사용해야 합니다.
"""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": build_planner_system_prompt(retriever),
            },
            {"role": "user", "content": repair_prompt},
        ],
        "tools": [build_search_tool(retriever)],
        "toolChoice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
        "maxTokens": 1024,
    }
    return extract_search_plan(client.chat(payload))


def compact_evidence_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Keep the auditable facts needed by the answer model, without parser noise."""
    documents: list[dict[str, Any]] = []
    document_fields = (
        "doc_id",
        "corp_name",
        "normalized_report_type",
        "report_nm",
        "is_correction",
        "filing_date",
        "event_date",
        "base_year",
        "base_month",
        "disclosure_chain_id",
        "relation_status",
        "corrects_doc_id",
        "corrected_by_doc_ids",
        "source_status",
    )
    for document in bundle.get("documents", []):
        compact_document = {key: document.get(key) for key in document_fields}
        compact_document["evidence_rows"] = [
            {
                "text": row.get("text"),
                "source_file": row.get("source_file"),
            }
            for row in document.get("evidence_rows", [])
        ]
        documents.append(compact_document)

    return {
        "schema_version": bundle.get("schema_version"),
        "search_plan": bundle.get("search_plan"),
        "resolved_company": bundle.get("resolved_company"),
        "retrieval_summary": bundle.get("retrieval_summary"),
        "comparison_groups": bundle.get("comparison_groups"),
        "chains": bundle.get("chains"),
        "graph_edges": bundle.get("graph_edges"),
        "documents": documents,
        "limitations": bundle.get("limitations"),
    }


def create_grounded_answer(
    client: ClovaStudioClient,
    question: str,
    bundle: dict[str, Any],
    *,
    max_tokens: int,
) -> tuple[str, dict[str, Any] | None]:
    evidence_json = json.dumps(
        compact_evidence_bundle(bundle),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"사용자 질문:\n{question}\n\n"
                    f"Evidence Bundle(JSON):\n{evidence_json}"
                ),
            },
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": max_tokens,
        "temperature": 0.0,
        "repetitionPenalty": 1.1,
        "stop": [],
    }
    result = client.chat(payload)
    content = result["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ClovaApiError("HCX-005 returned an empty grounded answer")
    return content.strip(), result.get("usage")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_pipeline(
    *,
    client: ClovaStudioClient,
    retriever: Any,
    logger: Any,
    question: str,
    max_chains: int,
    include_xml: bool,
    max_evidence_rows: int,
    answer_max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = logger.new_record(
        question=question,
        model=client.model,
        pipeline_version=SCRIPT_VERSION,
    )
    stage = "hcx_function_call"
    try:
        search_plan, planner_metadata = create_search_plan(client, retriever, question)
        trace["hcx_function_arguments"] = search_plan
        trace["hcx_repaired_function_arguments"] = None
        trace["hcx_function_argument_attempts"] = [
            {"attempt": 1, "kind": "initial", "arguments": search_plan}
        ]
        trace["python_validation_attempts"] = []
        trace["usage"]["planner"] = planner_metadata.get("planner_usage")

        stage = "python_validation"
        try:
            canonical_plan = retriever.validate_plan(search_plan)
            trace["python_validation_attempts"].append(
                {
                    "attempt": 1,
                    "status": "passed",
                    "canonical_plan": canonical_plan,
                    "error": None,
                }
            )
        except RetrievalError as first_validation_error:
            trace["python_validation_attempts"].append(
                {
                    "attempt": 1,
                    "status": "failed",
                    "canonical_plan": None,
                    "error": (
                        f"{type(first_validation_error).__name__}: "
                        f"{first_validation_error}"
                    ),
                }
            )
            stage = "hcx_function_repair"
            repaired_plan, repair_metadata = repair_search_plan(
                client,
                retriever,
                question,
                search_plan,
                first_validation_error,
            )
            trace["hcx_repaired_function_arguments"] = repaired_plan
            trace["hcx_function_argument_attempts"].append(
                {"attempt": 2, "kind": "repair", "arguments": repaired_plan}
            )
            trace["usage"]["planner"] = {
                "initial": planner_metadata.get("planner_usage"),
                "repair": repair_metadata.get("planner_usage"),
            }
            stage = "python_validation"
            try:
                canonical_plan = retriever.validate_plan(repaired_plan)
                trace["python_validation_attempts"].append(
                    {
                        "attempt": 2,
                        "status": "passed",
                        "canonical_plan": canonical_plan,
                        "error": None,
                    }
                )
            except RetrievalError as second_validation_error:
                trace["python_validation_attempts"].append(
                    {
                        "attempt": 2,
                        "status": "failed",
                        "canonical_plan": None,
                        "error": (
                            f"{type(second_validation_error).__name__}: "
                            f"{second_validation_error}"
                        ),
                    }
                )
                raise

        trace["python_validation"] = {
            "status": "passed",
            "canonical_plan": canonical_plan,
            "error": None,
        }

        stage = "evidence_retrieval"
        evidence_bundle = retriever.retrieve_from_plan(
            question=question,
            plan=canonical_plan,
            max_chains=max_chains,
            include_xml=include_xml,
            max_evidence_rows=max_evidence_rows,
        )
        trace["evidence_bundle"] = evidence_bundle

        stage = "grounded_answer"
        answer, answer_usage = create_grounded_answer(
            client,
            question,
            evidence_bundle,
            max_tokens=answer_max_tokens,
        )
        trace["final_answer"] = answer
        trace["usage"]["answer"] = answer_usage

        result = {
            "schema_version": "clova_graph_rag_result_v1",
            "pipeline_version": SCRIPT_VERSION,
            "model": client.model,
            "question": question,
            "search_plan": evidence_bundle.get("search_plan"),
            "retrieval_summary": evidence_bundle.get("retrieval_summary"),
            "limitations": evidence_bundle.get("limitations"),
            "answer": answer,
            "usage": trace["usage"],
            "trace_id": trace["trace_id"],
        }
        trace["pipeline_status"] = "succeeded"
        logger.append(trace)
        return evidence_bundle, result
    except Exception as exc:
        if stage == "python_validation":
            trace["python_validation"] = {
                "status": "failed",
                "canonical_plan": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        trace["pipeline_status"] = "failed"
        trace["failed_stage"] = stage
        trace["error"] = f"{type(exc).__name__}: {exc}"
        logger.append(trace)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HCX-005 Function Calling + disclosure graph RAG baseline"
    )
    parser.add_argument("--question", help="One-shot question; omit for interactive mode")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint-template", default=DEFAULT_ENDPOINT_TEMPLATE)
    parser.add_argument("--api-key-env", default="CLOVASTUDIO_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-chains", type=int, default=10)
    parser.add_argument("--max-evidence-rows", type=int, default=12)
    parser.add_argument("--answer-max-tokens", type=int, default=2048)
    parser.add_argument("--no-xml", action="store_true")
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE_OUTPUT)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT_OUTPUT)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Append one JSON record per question to this JSONL file",
    )
    parser.add_argument(
        "--no-save-evidence",
        action="store_true",
        help="Do not persist the full Evidence Bundle JSON",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(
            f"Error: environment variable {args.api_key_env} is not set.\n"
            f'PowerShell: $env:{args.api_key_env} = "YOUR_API_KEY"',
            file=sys.stderr,
        )
        return 1

    try:
        retriever = GraphRetriever(
            manifest_path=args.manifest,
            edges_path=args.edges,
            corpus_root=args.corpus_root,
        )
        client = ClovaStudioClient(
            api_key=api_key,
            model=args.model,
            endpoint_template=args.endpoint_template,
            timeout=args.timeout,
        )
        logger = PipelineDebugLogger(args.log_file)

        def execute(question: str) -> None:
            evidence_bundle, result = run_pipeline(
                client=client,
                retriever=retriever,
                logger=logger,
                question=question,
                max_chains=args.max_chains,
                include_xml=not args.no_xml,
                max_evidence_rows=args.max_evidence_rows,
                answer_max_tokens=args.answer_max_tokens,
            )
            if not args.no_save_evidence:
                write_json(args.evidence_output, evidence_bundle)
            write_json(args.result_output, result)

            print("\n[HCX 검색 계획]")
            print(json.dumps(result["search_plan"], ensure_ascii=False, indent=2))
            print("\n[검색 요약]")
            print(json.dumps(result["retrieval_summary"], ensure_ascii=False, indent=2))
            print("\n[HCX 근거 기반 답변]")
            print(result["answer"])
            if result["limitations"]:
                print(f"\n주의사항 {len(result['limitations'])}건은 결과 JSON에 저장됐습니다.")
            if not args.no_save_evidence:
                print(f"\nEvidence Bundle: {args.evidence_output}")
            print(f"RAG result: {args.result_output}")
            print(f"Debug log: {args.log_file}")

        if args.question:
            execute(args.question.strip())
            return 0

        print(f"CLOVA graph RAG pipeline {SCRIPT_VERSION} / model={args.model}")
        print("질문을 입력하세요. 종료하려면 q 또는 quit를 입력하세요.")
        while True:
            try:
                question = input("\n질문> ").strip()
            except EOFError:
                print()
                return 0
            if question.casefold() in {"q", "quit", "exit", "종료"}:
                return 0
            if not question:
                continue
            try:
                execute(question)
            except (ClovaApiError, RetrievalError, OSError, ValueError) as exc:
                print(
                    f"실행 오류: {exc}\nDebug log: {args.log_file}",
                    file=sys.stderr,
                )
    except (ClovaApiError, RetrievalError, OSError, ValueError) as exc:
        print(f"Error: {exc}\nDebug log: {args.log_file}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())