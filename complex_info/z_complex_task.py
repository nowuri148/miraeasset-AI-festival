from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import requests
from dotenv import load_dotenv


# =============================================================================
# PATH / ENV
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)


# =============================================================================
# EXISTING PROJECT COMPONENTS
# =============================================================================

from complex_info.complex_answer_generator import (  # noqa: E402
    INSUFFICIENT_EVIDENCE_ANSWER,
    HCX007ComplexAnswerGenerator,
    format_answer_with_sources,
)
from search_info.search_pipeline import (  # noqa: E402
    validate_and_normalize_search_plan,
)
from search_info.z_search_task_hj import (  # noqa: E402
    get_operational_executor,
)


# =============================================================================
# CONSTANTS
# =============================================================================

TASK_TYPE = "복합문서추론"

DEFAULT_TASK_API_BASE_URL = (
    "https://clovastudio.stream.ntruss.com"
)

# 1차 검색계획 모델(A)이 실패했을 때만 키워드 생성형 모델(B)로 재시도한다.
# 계획 생성 오류나 초기화 오류는 검색 품질 문제가 아니므로 재시도하지 않는다.
SEARCH_PLAN_MODEL_A = "A"
SEARCH_PLAN_MODEL_B = "B"
RETRYABLE_STATUSES = frozenset(
    {
        "no_search_context",
        "insufficient_evidence",
    }
)

SEARCH_PLAN_SYSTEM_PROMPT = """당신은 한국 기업 공시 검색을 위한 검색계획 생성기입니다.
입력은 question과 keyword_extractor_output을 담은 JSON object입니다. 질문에 답하지 말고, 아래 계약을 만족하는 검색계획 JSON object 하나만 출력하십시오. 설명, 인사, 코드펜스, 주석, 접두사와 접미사는 절대 출력하지 마십시오.

최상위 형식:
{"retrieval_requests":[{"query":"검색문","normalized_report_types":["annual_report"],"period":{"start":"2023","end":"2025"},"date_basis":"base_year","use_correction_graph":false,"company_scope":{"scope_type":"direct","scope_values":["삼성전자"]},"exact_keywords":["매출액"]}]}

공통 원칙:
1. 최상위에는 retrieval_requests만 두고, 각 요청에는 query, normalized_report_types, period, date_basis, use_correction_graph, company_scope, exact_keywords 일곱 필드를 빠짐없이 두십시오. 예시에도 없는 필드는 만들지 마십시오.
2. 원 질문의 기업·범위·기간·문서유형·지표·사건·비교 대상을 그대로 보존하십시오. 질문에 없는 기업, 기간, 지표, 사건을 추측해서 추가하지 마십시오.
3. 하나의 retrieval request는 하나의 독립된 검색 의도를 표현해야 합니다. 서로 다른 지표·주제에 각각 근거가 필요한 복합 질문은 요청을 분리하십시오.
4. 서로 떨어진 특정 연도들을 비교하는 질문은 연도별 요청으로 분리하십시오. 연속된 기간 전체의 추세를 묻는 질문은 하나의 기간 요청으로 둘 수 있습니다.
5. 같은 기업·기간·문서유형이라도 설비투자, 재고자산, 사업부문 실적처럼 독립적으로 찾아야 하는 주제는 별도 요청으로 분리하십시오.
6. query는 해당 요청의 기업, 기간, 주제와 '관련 공시 근거'라는 목적이 드러나는 자연스러운 한국어 검색문으로 작성하십시오.

normalized_report_types 규칙:
7. 각 요청에는 아래 허용값 중 1개 이상 3개 이하만 넣으십시오.
annual_report, semiannual_report, quarterly_report, facility_investment, supply_contract, other_company_shares_acquisition_decision, material_management_matter, treasury_stock_acquisition_decision, lawsuit_filing, merger_decision, treasury_stock_disposal_decision, paid_in_capital_increase_decision, large_shareholding_general, supply_contract_termination, other_company_shares_disposal_decision, convertible_bond_issuance_decision, tangible_asset_disposal_decision, large_shareholding_summary, tangible_asset_acquisition_decision, company_split_decision, write_down_contingent_capital_security_issuance_decision, treasury_stock_trust_agreement_conclusion_decision, treasury_stock_trust_agreement_termination_decision, stock_exchange_or_transfer_decision, split_merger_decision, overseas_securities_delisting_decision, regulatory_capital_debt_security_issuance_decision, capital_reduction_decision, bonus_issue_decision, exchangeable_bond_issuance_decision, own_convertible_bond_sale_decision, overseas_securities_listing_decision, business_acquisition_decision, third_party_convertible_bond_call_option_exercise, business_suspension.
8. 재무수치, 재고자산, 매출액, 영업이익, 사업부문 실적, 사업 내용처럼 정기보고서에서 확인하는 항목은 annual_report, semiannual_report, quarterly_report 중 질문에 맞는 유형을 사용하십시오.
9. 특정 결정·계약·정정 사건은 대응하는 비정기 공시 유형을 사용하십시오. 다만 질문이 사업연도별 변화나 정기보고서상의 수치를 요구하면 정기보고서 유형을 선택하십시오.

기간과 날짜 기준 규칙:
10. date_basis는 base_year, event_date, rcept_dt 중 하나만 사용하십시오.
11. 사업연도, 분기, 반기 실적이나 정기보고서 기준 비교는 base_year를 사용하십시오. base_year의 period 값은 YYYY, YYYY-Qn, YYYY-Hn 형식만 사용하십시오.
12. 계약일, 결정일, 발생일처럼 사건의 실제 시점을 묻는 경우 event_date를 사용하고, 공시 제출·접수 시점을 묻는 경우 rcept_dt를 사용하십시오. 이 두 기준의 period 값은 YYYY-MM-DD 형식만 사용하십시오.
13. period.start는 period.end보다 늦을 수 없습니다. 질문의 시작과 종료 경계를 임의로 넓히지 마십시오.

회사 범위 규칙:
14. company_scope.scope_type은 direct, group, sector, industry, all 중 하나입니다.
15. 특정 기업은 direct를 사용하고 scope_values에 keyword_extractor_output에서 확정된 기업명을 넣으십시오. 기업집단은 group, 섹터는 sector, 산업은 industry를 사용하십시오. all일 때만 scope_values를 빈 배열로 두십시오.
16. group의 scope_values에는 기업집단 이름을 넣고, 개별 구성회사 목록을 복사하지 마십시오. 구성회사 확장은 Python 실행기가 처리합니다.
17. sector 허용값은 2차전지, AI소프트웨어·플랫폼, 건설, 게임, 금융·보험, 로봇, 바이오·제약, 반도체·전자부품, 방산·항공우주, 비철금속, 소비재·유통, 신재생에너지, 엔터테인먼트, 운송·물류, 원전, 자동차·모빌리티, 전력기기, 조선, 철강, 통신입니다.
18. industry 허용값은 IT, 건강관리, 경기관련소비재, 금융, 산업재, 소재, 커뮤니케이션서비스, 필수소비재입니다.

exact_keywords와 정정 그래프 규칙:
19. exact_keywords에는 keyword_extractor_output의 metrics 또는 topic_keywords에 실제로 존재하는 표현만 선택하십시오. 회사명, 기업집단명, 문서유형, 기간, 일반적인 '비교·분석·설명·확인·조회·검색·요약·추출·계산'을 넣지 마십시오.
20. exact_keywords는 해당 retrieval request에 직접 필요한 표현만 배정하고 다른 요청의 지표를 섞지 마십시오. 적절한 표현이 없으면 빈 배열을 사용하십시오.
21. exact_keywords는 원문 포함 여부를 강제하는 필터가 아니라 보조 검색어입니다. 동의어를 새로 만들거나 과도하게 늘리지 마십시오.
22. 정정 전후, 변경 이력, 최초 공시와 최종 공시의 연결을 추적해야 할 때만 use_correction_graph를 true로 두십시오. 그 외에는 false입니다.
23. question과 keyword_extractor_output이 충돌하면 기업과 추출 키워드는 keyword_extractor_output의 확정값을 따르되, 질문의 비교와 기간은 보존하십시오.
24. 여러 공시를 함께 보라는 질문이어도 독립 주제를 합치지 마십시오. 요청은 하나의 주제를 담당하고 답변 모델이 근거를 종합하게 하십시오.
25. 비정기 공시와 사업연도별 재무 수치가 함께 필요하면 요청을 분리하십시오. 사건 시점만 묻는 질문에는 불필요한 정기보고서 요청을 추가하지 마십시오.
26. 회사 간 비교는 회사별 direct 요청으로 나눌 수 있습니다. sector, industry, group 범위가 명시되면 개별 회사로 축소하지 마십시오.

출력 전 자체 점검:
27. JSON 문법이 유효한지, 큰따옴표와 true/false를 사용했는지 확인하십시오.
28. 모든 요청에 일곱 필드가 정확히 한 번씩 있는지 확인하십시오.
29. report type, date_basis, scope_type 값이 허용 목록 안에 있는지 확인하십시오.
30. 원 질문의 각 독립 요청과 비교 연도가 검색계획에 반영되었는지 확인하십시오.
31. 자체 점검 결과를 출력하지 말고 최종 JSON object 하나만 출력하십시오."""


# =============================================================================
# ERRORS
# =============================================================================

class SearchPlanError(RuntimeError):
    """검색계획 생성 단계의 기본 오류."""


class SearchPlanAPIError(SearchPlanError):
    """튜닝 작업 API 호출 오류."""


class SearchPlanOutputError(SearchPlanError):
    """튜닝 모델 출력 파싱 또는 검증 오류."""


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _env_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)

    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def _debug(
    enabled: bool,
    stage: str,
    message: str,
) -> None:
    if enabled:
        print(
            f"[COMPLEX][{stage}] {message}",
            flush=True,
        )


def _failure_result(
    *,
    status: str,
    answer: str,
    error: Exception | None = None,
    search_plan: dict[str, Any] | None = None,
    retrieved_context: str = "",
    search_plan_model: str = SEARCH_PLAN_MODEL_A,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "answerable": False,
        "status": status,
        "stage": status,
        "task_type": TASK_TYPE,
        "answer": answer,
        "answer_with_sources": answer,
        "sources": [],
        "used_source_ids": [],
        "retrieved_context": retrieved_context,
        "search_plan_model": search_plan_model,
        "retry_attempted": False,
    }

    if search_plan is not None:
        result["search_plan"] = search_plan

    if error is not None:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)

    return result


# =============================================================================
# SEARCH PLAN CLIENT
# =============================================================================

class HCXSearchPlanClient:
    """파인튜닝된 HCX 작업에서 검색계획 JSON을 생성한다."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        task_id: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        debug: bool = False,
    ) -> None:
        self.api_key = str(
            api_key
            or os.getenv("CLOVA_STUDIO_API_KEY", "")
        ).strip()

        self.task_id = str(
            task_id
            or os.getenv("CLOVA_STUDIO_TASK_ID", "")
        ).strip()

        self.base_url = str(
            base_url
            or os.getenv(
                "CLOVA_SEARCH_PLAN_BASE_URL",
                DEFAULT_TASK_API_BASE_URL,
            )
        ).strip().rstrip("/")

        self.timeout = int(
            timeout
            if timeout is not None
            else _env_int(
                "CLOVA_SEARCH_PLAN_TIMEOUT",
                120,
                minimum=1,
            )
        )

        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else _env_int(
                "CLOVA_SEARCH_PLAN_MAX_TOKENS",
                4096,
                minimum=1,
                maximum=4096,
            )
        )

        configured_system_prompt = os.getenv(
            "CLOVA_SEARCH_PLAN_SYSTEM_PROMPT",
            "",
        ).strip()
        self.system_prompt = str(
            system_prompt
            if system_prompt is not None
            else configured_system_prompt or SEARCH_PLAN_SYSTEM_PROMPT
        ).strip()

        self.debug = bool(debug)

        if not self.api_key:
            raise RuntimeError(
                "CLOVA_STUDIO_API_KEY가 설정되어 있지 않습니다."
            )

        if not self.task_id:
            raise RuntimeError(
                "CLOVA_STUDIO_TASK_ID가 설정되어 있지 않습니다."
            )

        if not self.system_prompt:
            raise ValueError("검색계획 시스템 프롬프트가 비어 있습니다.")

    def generate(
        self,
        *,
        question: str,
        keyword_extractor_output: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """질문과 키워드 추출 결과를 전달하고 검증된 검색계획을 반환한다."""

        model_input = {
            "question": question,
            "keyword_extractor_output": keyword_extractor_output,
        }

        input_text = json.dumps(
            model_input,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        url = (
            f"{self.base_url}"
            f"/v3/tasks/{self.task_id}/chat-completions"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": input_text,
                }
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": self.max_tokens,
            "temperature": 0.0,
            "repetitionPenalty": 1.0,
            "stop": [],
            "seed": 1,
        }

        _debug(
            self.debug,
            "PLAN_REQUEST",
            (
                f"task_id={self.task_id} "
                f"system_prompt_chars={len(self.system_prompt):,} "
                f"input_chars={len(input_text):,}"
            ),
        )

        started = time.perf_counter()

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SearchPlanAPIError(
                f"검색계획 API 연결 오류: {exc}"
            ) from exc

        elapsed = time.perf_counter() - started

        if not response.ok:
            response_text = str(response.text or "")[:2000]
            raise SearchPlanAPIError(
                "검색계획 API 오류: "
                f"HTTP {response.status_code} {response_text}"
            )

        try:
            response_data = response.json()
            raw_output = str(
                response_data["result"]["message"]["content"]
            ).strip()
            finish_reason = str(
                response_data["result"].get("finishReason")
                or response_data["result"]["message"].get("finishReason")
                or "unknown"
            ).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise SearchPlanAPIError(
                "검색계획 API 응답에서 result.message.content를 "
                "찾지 못했습니다."
            ) from exc

        _debug(
            self.debug,
            "PLAN_RESPONSE",
            (
                f"elapsed={elapsed:.3f}s "
                f"output_chars={len(raw_output):,}"
            ),
        )

        _debug(
            self.debug,
            "PLAN_FINISH",
            finish_reason,
        )
        _debug(
            self.debug,
            "PLAN_RAW",
            repr(raw_output),
        )

        search_plan = self._parse_and_validate(raw_output)

        _debug(
            self.debug,
            "PLAN_VALID",
            (
                "retrieval_requests="
                f"{len(search_plan['retrieval_requests'])}"
            ),
        )

        return search_plan, raw_output

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        value_text = str(text or "").lstrip("\ufeff").strip()

        if not value_text:
            raise SearchPlanOutputError(
                "검색계획 LLM 출력이 비어 있습니다."
            )

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            value_text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

        candidates = [value_text, cleaned]

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])

        # 최상위 검색계획 객체의 마지막 } 하나만 누락됐을 때 복구
        if (
            cleaned.startswith('{"retrieval_requests"')
            and cleaned.count("{") == cleaned.count("}") + 1
            and cleaned.count("[") == cleaned.count("]")
        ):
            candidates.append(cleaned + "}")

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return parsed

        raise SearchPlanOutputError(
            "검색계획 LLM 출력을 JSON object로 파싱하지 못했습니다."
        )

    @classmethod
    def _parse_and_validate(
        cls,
        raw_output: str,
    ) -> dict[str, Any]:
        search_plan = cls._parse_json_object(raw_output)

        wrapped = search_plan.get("output")

        if isinstance(wrapped, str):
            search_plan = cls._parse_json_object(wrapped)
        elif isinstance(wrapped, dict):
            search_plan = wrapped

        requests_value = search_plan.get("retrieval_requests")

        if not isinstance(requests_value, list) or not requests_value:
            raise SearchPlanOutputError(
                "retrieval_requests가 비어 있거나 배열이 아닙니다."
            )

        try:
            validate_and_normalize_search_plan(search_plan)
        except Exception as exc:
            raise SearchPlanOutputError(
                f"검색계획 형식 검증 실패: {exc}"
            ) from exc

        return search_plan


# =============================================================================
# PLAN KEYWORD MERGE (MODEL B)
# =============================================================================

def _plan_exact_keywords(
    search_plan: dict[str, Any],
) -> list[str]:
    """검색계획의 모든 요청에서 exact_keywords를 순서대로 모은다."""

    collected: list[str] = []

    for request in search_plan.get("retrieval_requests") or []:
        if not isinstance(request, dict):
            continue

        for keyword in request.get("exact_keywords") or []:
            if not isinstance(keyword, str):
                continue

            value = keyword.strip()

            if value:
                collected.append(value)

    return list(dict.fromkeys(collected))


def _extracted_with_plan_keywords(
    extracted: dict[str, Any],
    search_plan: dict[str, Any],
) -> dict[str, Any]:
    """검색계획이 생성한 키워드를 topic_keywords 허용 풀에 합친다.

    search_pipeline.grounded_keywords_by_request는 exact_keywords가
    keyword_extractor_output의 metrics 또는 topic_keywords에 실제로
    존재하는 표현인지 검증한다. 키워드 생성형 모델(B)이 만든 표현은
    원 질문에 없으므로 그대로 두면 전량 폐기된다. 허용 풀만 넓혀
    회사명·문서유형 차단과 일반어 필터는 그대로 살려 둔다.

    원본 extracted는 result["extracted"]로 외부에 노출되므로 수정하지
    않고 얕은 복사본을 만들어 반환한다.
    """

    plan_keywords = _plan_exact_keywords(search_plan)

    if not plan_keywords:
        return extracted

    merged = dict(extracted)
    existing = merged.get("topic_keywords")

    if isinstance(existing, Mapping):
        # 중첩 구조를 쓰는 추출기 결과도 _iter_string_values가 재귀로
        # 훑으므로 별도 키에 담아 두면 그대로 허용 풀에 포함된다.
        merged["topic_keywords"] = {
            **existing,
            "_search_plan_keywords": plan_keywords,
        }
        return merged

    if isinstance(existing, list):
        existing_values = [
            value.strip()
            for value in existing
            if isinstance(value, str) and value.strip()
        ]
        non_string_values = [
            value
            for value in existing
            if not isinstance(value, str)
        ]
        merged["topic_keywords"] = [
            *non_string_values,
            *dict.fromkeys([*existing_values, *plan_keywords]),
        ]
        return merged

    if isinstance(existing, str) and existing.strip():
        merged["topic_keywords"] = list(
            dict.fromkeys([existing.strip(), *plan_keywords])
        )
        return merged

    merged["topic_keywords"] = list(plan_keywords)
    return merged


# =============================================================================
# CONTEXT CONVERSION
# =============================================================================

def _context_chunks_to_results(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """검색 실행기의 packed chunk를 기존 Answer/API 형식으로 변환한다."""

    results: list[dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()

        if not text:
            continue

        metadata_value = chunk.get("metadata")
        retrieval_value = chunk.get("retrieval")

        metadata = (
            dict(metadata_value)
            if isinstance(metadata_value, dict)
            else {}
        )
        retrieval = (
            dict(retrieval_value)
            if isinstance(retrieval_value, dict)
            else {}
        )

        results.append(
            {
                "chunk_id": (
                    chunk.get("chunk_id")
                    or metadata.get("chunk_id")
                ),
                "corp_name": metadata.get("corp_name"),
                "document": text,
                "text": text,
                "metadata": metadata,
                "rerank_rank": len(results) + 1,
                "rerank_score": retrieval.get("rerank_score", 0.0),
                "retrieval_origin": "complex_search_pipeline",
                "request_ids": retrieval.get("request_ids") or [],
                "retrieval": retrieval,
            }
        )

    return results


def _build_context_text(
    results: list[dict[str, Any]],
    *,
    max_results: int,
    max_chars_per_result: int,
) -> str:
    blocks: list[str] = []

    for rank, item in enumerate(
        results[:max_results],
        start=1,
    ):
        metadata = item.get("metadata") or {}
        document = str(
            item.get("document")
            or item.get("text")
            or ""
        ).strip()

        if len(document) > max_chars_per_result:
            document = (
                document[:max_chars_per_result]
                + "\n...[TRUNCATED]"
            )

        header = [f"[SOURCE {rank}]"]

        corp_name = item.get("corp_name") or metadata.get("corp_name")
        report_nm = metadata.get("report_nm")
        rcept_no = metadata.get("rcept_no")
        chunk_id = item.get("chunk_id") or metadata.get("chunk_id")

        if corp_name:
            header.append(f"회사: {corp_name}")
        if report_nm:
            header.append(f"공시: {report_nm}")
        if rcept_no:
            header.append(f"접수번호: {rcept_no}")
        if chunk_id:
            header.append(f"chunk_id: {chunk_id}")

        blocks.append(
            "\n".join(header)
            + "\n\n"
            + document
        )

    return "\n\n".join(blocks)


def _build_validation_context(
    used_sources_with_text: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    *,
    max_chars_per_source: int,
    max_total_chars: int,
) -> str:
    """검증기에 넘길 근거 텍스트를 답변이 실제로 인용한 SOURCE로 구성한다.

    evidence_builder는 used_source_ids로 reranked_results를 인덱싱하지만,
    source_id는 답변 생성기의 prepared_sources 순번이므로 두 리스트가
    항상 같은 정렬을 보장하지는 않는다. 여기서 chunk_id로 직접 맞춰
    validation_context를 만들어 두면 evidence_builder가 이 값을 그대로
    최우선으로 사용한다.
    """

    sources = [
        source
        for source in used_sources_with_text
        if isinstance(source, dict)
    ]

    if not sources:
        # 인용 정보가 없으면 상위 근거로 대체한다. 검증기가 근거 없음으로
        # 즉시 RETRY 판정하는 것보다는 실제 컨텍스트를 보여주는 편이 낫다.
        text_by_chunk = {
            str(item.get("chunk_id") or ""): item
            for item in reranked_results
            if isinstance(item, dict)
        }
        sources = [
            {
                "source_id": index,
                "chunk_id": item.get("chunk_id"),
                "corp_name": item.get("corp_name"),
                "report_nm": (item.get("metadata") or {}).get("report_nm"),
                "rcept_no": (item.get("metadata") or {}).get("rcept_no"),
                "text": str(item.get("document") or item.get("text") or ""),
            }
            for index, item in enumerate(
                list(text_by_chunk.values())[:5],
                start=1,
            )
        ]

    blocks: list[str] = []
    consumed = 0

    for source in sources:
        text = str(source.get("text") or "").strip()

        if not text:
            continue

        remaining = max_total_chars - consumed

        if remaining <= 0:
            break

        limit = min(max_chars_per_source, remaining)

        if len(text) > limit:
            text = text[:limit] + "\n...[TRUNCATED]"

        consumed += len(text)

        lines = [f"[SOURCE {len(blocks) + 1}]"]

        corp_name = str(source.get("corp_name") or "").strip()
        report_nm = str(source.get("report_nm") or "").strip()
        rcept_no = str(source.get("rcept_no") or "").strip()

        if corp_name:
            lines.append(f"회사: {corp_name}")
        if report_nm:
            lines.append(f"공시: {report_nm}")
        if rcept_no:
            lines.append(f"접수번호: {rcept_no}")

        lines.extend(["", text])
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _request_queries_by_id(
    search_plan: dict[str, Any],
) -> dict[str, str]:
    """검색계획의 요청 순번을 실행기 request_id 규칙에 맞춰 매핑한다."""
    return {
        f"request_{index:03d}": str(request.get("query") or "").strip()
        for index, request in enumerate(
            search_plan.get("retrieval_requests") or [],
            start=1,
        )
        if isinstance(request, dict)
    }


def _append_period_fallback_notice(
    answer: str,
    *,
    period_fallbacks: list[dict[str, Any]],
    used_sources: list[dict[str, Any]],
    search_plan: dict[str, Any],
) -> str:
    """실제로 사용된 fallback 청크의 기간 안내를 코드로 덧붙인다."""
    used_chunk_ids = {
        str(source.get("chunk_id") or "").strip()
        for source in used_sources
        if isinstance(source, dict)
    }
    used_chunk_ids.discard("")
    if not used_chunk_ids:
        return answer

    request_queries = _request_queries_by_id(search_plan)
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for fallback in period_fallbacks:
        if not isinstance(fallback, dict):
            continue
        chunk_id = str(fallback.get("chunk_id") or "").strip()
        if chunk_id not in used_chunk_ids:
            continue
        request_id = str(fallback.get("request_id") or "").strip()
        period_bucket = str(fallback.get("period_bucket") or "").strip()
        requested_period = str(
            fallback.get("requested_period") or "지정 기간"
        ).strip()
        used_period = str(
            fallback.get("used_period") or "다른 기간의 정기공시"
        ).strip()
        key = (request_id, period_bucket, used_period)
        if key in seen:
            continue
        seen.add(key)
        query = request_queries.get(request_id) or request_id or "해당 요청"
        lines.append(
            f"- {period_bucket}년 ‘{query}’: {requested_period} 기준의 "
            f"1차 근거를 확보하지 못해 {used_period} 근거를 사용했습니다."
        )

    if not lines:
        return answer
    return (
        answer.rstrip()
        + "\n\n※ 근거 범위 안내\n"
        + "\n".join(lines)
    )


def _append_missing_cell_notice(
    answer: str,
    *,
    missing_period_cells: list[dict[str, Any]],
    search_plan: dict[str, Any],
) -> str:
    """근거를 전혀 확보하지 못한 요청×연도 셀을 코드로 안내한다.

    기간 fallback과 달리 대조할 청크 자체가 없으므로 used_sources로
    걸러내지 않는다. 모델이 누락 범위를 답변 본문에 밝히지 않아도
    사용자가 결손을 알 수 있게 한다.
    """
    if not missing_period_cells:
        return answer

    request_queries = _request_queries_by_id(search_plan)

    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for cell in missing_period_cells:
        if not isinstance(cell, dict):
            continue

        request_id = str(cell.get("request_id") or "").strip()
        period_bucket = str(cell.get("period_bucket") or "").strip()

        if not period_bucket:
            continue

        key = (request_id, period_bucket)

        if key in seen:
            continue

        seen.add(key)
        query = request_queries.get(request_id) or request_id or "해당 요청"
        lines.append(
            f"- {period_bucket}년 ‘{query}’: 관련 공시 근거를 확보하지 "
            "못해 답변에 반영하지 못했습니다."
        )

    if not lines:
        return answer

    return (
        answer.rstrip()
        + "\n\n※ 미확보 근거 안내\n"
        + "\n".join(lines)
    )


# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================

_search_plan_client: HCXSearchPlanClient | None = None
_search_plan_client_b: HCXSearchPlanClient | None = None
_search_plan_client_b_resolved: bool = False
_operational_executor: Any = None
_answer_generator: HCX007ComplexAnswerGenerator | None = None
_grounding_validator: Any = None
_grounding_validator_resolved: bool = False


def initialize_complex_components(
    *,
    debug: bool,
) -> tuple[
    HCXSearchPlanClient,
    Any,
    HCX007ComplexAnswerGenerator,
]:
    """무거운 검색 모델과 API 클라이언트를 프로세스당 한 번 생성한다."""

    global _search_plan_client
    global _operational_executor
    global _answer_generator

    if _search_plan_client is None:
        _search_plan_client = HCXSearchPlanClient(
            debug=debug,
        )

    if _operational_executor is None:
        _operational_executor = get_operational_executor(
            project_root=PROJECT_ROOT,
            debug=debug,
        )

    if _answer_generator is None:
        api_key = os.getenv("CLOVA_STUDIO_API_KEY", "").strip()

        _answer_generator = HCX007ComplexAnswerGenerator(
            api_key=api_key,
            base_url=os.getenv(
                "CLOVA_COMPLEX_ANSWER_BASE_URL",
                DEFAULT_TASK_API_BASE_URL,
            ),
            model_name=os.getenv(
                "CLOVA_COMPLEX_ANSWER_MODEL",
                "HCX-007",
            ),
            timeout=_env_int(
                "COMPLEX_ANSWER_TIMEOUT",
                180,
                minimum=1,
            ),
            max_sources=_env_int(
                "COMPLEX_ANSWER_MAX_SOURCES",
                24,
                minimum=1,
            ),
            max_source_chars=_env_int(
                "COMPLEX_ANSWER_MAX_SOURCE_CHARS",
                6000,
                minimum=500,
            ),
            max_total_source_chars=_env_int(
                "COMPLEX_ANSWER_MAX_TOTAL_SOURCE_CHARS",
                70000,
                minimum=1000,
            ),
            thinking_effort=os.getenv(
                "CLOVA_COMPLEX_THINKING_EFFORT",
                "low",
            ),
            max_completion_tokens=_env_int(
                "CLOVA_COMPLEX_MAX_COMPLETION_TOKENS",
                5120,
                minimum=1,
                maximum=32768,
            ),
            debug=debug,
        )

    return (
        _search_plan_client,
        _operational_executor,
        _answer_generator,
    )


def get_search_plan_client_b(
    *,
    debug: bool,
) -> HCXSearchPlanClient | None:
    """키워드 생성형 검색계획 모델(B) 클라이언트를 지연 생성한다.

    CLOVA_STUDIO_TASK_ID_B가 없거나 생성에 실패하면 None을 반환하고,
    호출 측은 재시도 없이 1차 결과를 그대로 사용한다.
    """

    global _search_plan_client_b
    global _search_plan_client_b_resolved

    if _search_plan_client_b_resolved:
        return _search_plan_client_b

    _search_plan_client_b_resolved = True

    task_id = os.getenv("CLOVA_STUDIO_TASK_ID_B", "").strip()

    if not task_id:
        _debug(
            debug,
            "PLAN_B_DISABLED",
            "CLOVA_STUDIO_TASK_ID_B가 설정되어 있지 않습니다.",
        )
        return None

    configured_prompt_path = os.getenv(
        "CLOVA_SEARCH_PLAN_B_SYSTEM_PROMPT_PATH",
        "",
    ).strip()
    configured_prompt = ""
    if configured_prompt_path:
        prompt_path = Path(configured_prompt_path).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = PROJECT_ROOT / prompt_path
        try:
            configured_prompt = prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as exc:
            _debug(
                debug,
                "PLAN_B_PROMPT_READ_ERROR",
                f"{prompt_path}: {exc}",
            )

    try:
        _search_plan_client_b = HCXSearchPlanClient(
            task_id=task_id,
            system_prompt=configured_prompt or None,
            debug=debug,
        )
    except Exception as exc:
        _debug(
            debug,
            "PLAN_B_INIT_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        _search_plan_client_b = None

    return _search_plan_client_b


def get_grounding_validator(
    *,
    debug: bool,
) -> Any:
    """근거 검증기를 지연 생성한다.

    COMPLEX_GROUNDING_VALIDATION이 꺼져 있거나 validation 패키지를
    불러오지 못하면 None을 반환하고, 호출 측은 검증 없이 진행한다.
    """

    global _grounding_validator
    global _grounding_validator_resolved

    if _grounding_validator_resolved:
        return _grounding_validator

    _grounding_validator_resolved = True

    if not _env_bool("COMPLEX_GROUNDING_VALIDATION", True):
        _debug(
            debug,
            "VALIDATOR_DISABLED",
            "COMPLEX_GROUNDING_VALIDATION=false",
        )
        return None

    try:
        from validation.answer_grounding_validator import (
            HCXGroundingValidator,
        )

        _grounding_validator = HCXGroundingValidator(
            debug=_env_bool("COMPLEX_GROUNDING_DEBUG", False),
        )
    except Exception as exc:
        _debug(
            debug,
            "VALIDATOR_INIT_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        _grounding_validator = None

    return _grounding_validator


# =============================================================================
# GROUNDING VALIDATION
# =============================================================================

def _validate_attempt(
    result: dict[str, Any],
    *,
    question: str,
    debug_enabled: bool,
) -> dict[str, Any]:
    """한 번의 시도 결과에 근거 검증을 적용하고 판정을 기록한다.

    검증 대상은 코드가 덧붙인 안내 문구를 제외한 순수 모델 답변이다.
    안내 문구에는 근거에 없는 연도·표현이 들어 있어 그대로 검증하면
    수치 검증 단계에서 잘못된 RETRY가 발생한다.
    """

    if not result.get("answerable"):
        return result

    validator = get_grounding_validator(debug=debug_enabled)

    if validator is None:
        result["grounding_validation"] = {
            "performed": False,
            "passed": True,
            "verdict": "SKIPPED",
            "reason": "검증기가 비활성화되어 있습니다.",
        }
        return result

    answer_for_validation = str(
        result.get("validation_answer")
        or result.get("answer")
        or ""
    ).strip()
    evidence = str(result.get("validation_context") or "").strip()

    started = time.perf_counter()

    try:
        validation = validator.validate(
            question=question,
            answer=answer_for_validation,
            evidence=evidence,
            task_type=TASK_TYPE,
        )
    except Exception as exc:
        # 검증기 장애로 정상 답변을 잃지 않도록 통과로 처리하고 기록만 남긴다.
        _debug(
            debug_enabled,
            "VALIDATION_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        result["grounding_validation"] = {
            "performed": False,
            "passed": True,
            "verdict": "ERROR",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        return result

    elapsed = time.perf_counter() - started
    passed = bool(validation.get("is_grounded", False))

    result["grounding_validation"] = {
        "performed": True,
        "passed": passed,
        "verdict": validation.get("verdict") or ("PASS" if passed else "RETRY"),
        "reason": validation.get("reason"),
        "unsupported_claims": validation.get("unsupported_claims") or [],
        "retry_feedback": validation.get("retry_feedback"),
        "elapsed_seconds": round(elapsed, 3),
    }

    _debug(
        debug_enabled,
        "VALIDATION",
        (
            f"model={result.get('search_plan_model')} "
            f"passed={passed} "
            f"verdict={result['grounding_validation']['verdict']} "
            f"evidence_chars={len(evidence):,} "
            f"elapsed={elapsed:.3f}s"
        ),
    )

    return result


def _attempt_rank(result: dict[str, Any]) -> int:
    """두 시도 중 무엇을 최종 결과로 쓸지 비교하기 위한 순위."""

    validation = result.get("grounding_validation") or {}

    if not result.get("answerable"):
        return 0

    if validation.get("performed") and not validation.get("passed"):
        return 1

    return 2


def _apply_strict_validation(
    result: dict[str, Any],
    *,
    debug_enabled: bool,
) -> dict[str, Any]:
    """검증을 통과하지 못한 답변을 근거 부족 응답으로 되돌린다."""

    validation = result.get("grounding_validation") or {}

    if not result.get("answerable"):
        return result

    if not validation.get("performed") or validation.get("passed"):
        return result

    if not _env_bool("COMPLEX_GROUNDING_STRICT", True):
        _debug(
            debug_enabled,
            "VALIDATION_NOT_STRICT",
            "검증 실패 답변을 그대로 반환합니다.",
        )
        return result

    _debug(
        debug_enabled,
        "VALIDATION_REJECTED",
        f"model={result.get('search_plan_model')} 답변을 근거 부족으로 대체합니다.",
    )

    result["answerable"] = False
    result["status"] = "grounding_validation_failed"
    result["stage"] = "answer_not_available"
    result["rejected_answer"] = result.get("answer")
    result["answer"] = INSUFFICIENT_EVIDENCE_ANSWER
    result["answer_with_sources"] = INSUFFICIENT_EVIDENCE_ANSWER
    result["sources"] = []
    result["used_source_ids"] = []

    return result


# =============================================================================
# SINGLE ATTEMPT
# =============================================================================

def _run_single_attempt(
    *,
    question: str,
    extracted: dict[str, Any],
    plan_client: HCXSearchPlanClient,
    executor: Any,
    answer_generator: HCX007ComplexAnswerGenerator,
    search_plan_model: str,
    merge_plan_keywords: bool,
    debug_enabled: bool,
) -> dict[str, Any]:
    """검색계획 생성 → 검색 → 답변까지 한 번 실행한다.

    질문 검증과 컴포넌트 초기화는 호출 측이 이미 끝냈다고 가정한다.
    예외를 밖으로 던지지 않고 항상 결과 dict를 반환한다.
    """

    # -------------------------------------------------------------------------
    # 1. SEARCH PLAN
    # -------------------------------------------------------------------------

    plan_started = time.perf_counter()

    try:
        search_plan, raw_search_plan = plan_client.generate(
            question=question,
            keyword_extractor_output=extracted,
        )
    except Exception as exc:
        _debug(
            debug_enabled,
            "PLAN_ERROR",
            f"model={search_plan_model} {type(exc).__name__}: {exc}",
        )
        return _failure_result(
            status="search_plan_error",
            answer="공시 검색 계획을 생성하지 못했습니다.",
            error=exc,
            search_plan_model=search_plan_model,
        )

    plan_seconds = time.perf_counter() - plan_started

    # 키워드 생성형 모델은 원 질문에 없는 표현을 만들므로, 실행기의
    # exact_keywords 검증을 통과하도록 허용 풀만 넓혀서 넘긴다.
    keyword_result_for_search = (
        _extracted_with_plan_keywords(extracted, search_plan)
        if merge_plan_keywords
        else extracted
    )

    if merge_plan_keywords:
        _debug(
            debug_enabled,
            "PLAN_KEYWORDS_MERGED",
            (
                f"model={search_plan_model} "
                f"plan_keywords={_plan_exact_keywords(search_plan)}"
            ),
        )

    # -------------------------------------------------------------------------
    # 2. OPERATIONAL SEARCH
    # -------------------------------------------------------------------------

    search_started = time.perf_counter()

    try:
        search_result = executor.run(
            question=question,
            search_plan=search_plan,
            keyword_result=keyword_result_for_search,
        )
    except Exception as exc:
        _debug(
            debug_enabled,
            "SEARCH_ERROR",
            f"model={search_plan_model} {type(exc).__name__}: {exc}",
        )
        return _failure_result(
            status="search_execution_error",
            answer="공시 근거를 검색하지 못했습니다.",
            error=exc,
            search_plan=search_plan,
            search_plan_model=search_plan_model,
        )

    search_seconds = time.perf_counter() - search_started

    if not search_result.get("success"):
        status = str(
            search_result.get("stage")
            or "search_execution_error"
        )
        error = RuntimeError(
            str(
                search_result.get("message")
                or search_result.get("error")
                or status
            )
        )
        return _failure_result(
            status=status,
            answer="질문에 필요한 공시 근거를 검색하지 못했습니다.",
            error=error,
            search_plan=search_plan,
            search_plan_model=search_plan_model,
        )

    context = search_result.get("context") or {}
    chunks_value = context.get("chunks") or []
    chunks = [
        item
        for item in chunks_value
        if isinstance(item, dict)
    ]

    reranked_results = _context_chunks_to_results(chunks)

    _debug(
        debug_enabled,
        "SEARCH_READY",
        (
            f"model={search_plan_model} "
            f"documents={len(context.get('documents') or [])} "
            f"chunks={len(reranked_results)} "
            f"chars={context.get('total_chars', 0)} "
            f"elapsed={search_seconds:.3f}s"
        ),
    )

    if not reranked_results:
        return _failure_result(
            status="no_search_context",
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            search_plan=search_plan,
            search_plan_model=search_plan_model,
        )

    retrieved_context = _build_context_text(
        reranked_results,
        max_results=len(reranked_results),
        max_chars_per_result=_env_int(
            "COMPLEX_RETRIEVED_CONTEXT_CHARS",
            6000,
            minimum=500,
        ),
    )

    context_summary = {
        "document_count": len(context.get("documents") or []),
        "chunk_count": len(reranked_results),
        "total_chars": context.get("total_chars", 0),
        "covered_request_ids": context.get("covered_request_ids") or [],
        "covered_period_buckets": (
            context.get("covered_period_buckets") or []
        ),
        "coverage_unit": "request_year",
        "covered_request_year_cells": (
            context.get("covered_request_year_cells") or []
        ),
        "request_period_coverage": (
            context.get("request_period_coverage") or []
        ),
        "missing_period_cells": (
            context.get("missing_period_cells") or []
        ),
        "request_year_coverage_complete": bool(
            context.get("request_year_coverage_complete", False)
        ),
    }

    # -------------------------------------------------------------------------
    # 3. FINAL ANSWER
    # -------------------------------------------------------------------------

    answer_started = time.perf_counter()

    try:
        answer_result = answer_generator.generate(
            question=question,
            reranked_results=reranked_results,
            search_plan=search_plan,
            context_summary=context_summary,
        )
    except Exception as exc:
        _debug(
            debug_enabled,
            "ANSWER_ERROR",
            f"model={search_plan_model} {type(exc).__name__}: {exc}",
        )
        return _failure_result(
            status="answer_generation_error",
            answer="검색된 공시 근거로 답변을 생성하지 못했습니다.",
            error=exc,
            search_plan=search_plan,
            retrieved_context=retrieved_context,
            search_plan_model=search_plan_model,
        )

    answer_seconds = time.perf_counter() - answer_started
    answer = str(answer_result.get("answer") or "").strip()
    used_source_ids = answer_result.get("used_source_ids") or []
    sources = answer_result.get("sources") or []
    answerable = bool(
        answer_result.get(
            "answerable",
            answer != INSUFFICIENT_EVIDENCE_ANSWER and bool(used_source_ids),
        )
    )

    # 검증 대상은 안내 문구를 붙이기 전의 순수 모델 답변이다.
    validation_answer = answer

    if answerable:
        answer = _append_period_fallback_notice(
            answer,
            period_fallbacks=context.get("period_fallbacks") or [],
            used_sources=sources,
            search_plan=search_plan,
        )
        answer = _append_missing_cell_notice(
            answer,
            missing_period_cells=context_summary["missing_period_cells"],
            search_plan=search_plan,
        )
        answer_result["answer"] = answer

    validation_context = _build_validation_context(
        answer_result.get("used_sources_with_text") or [],
        reranked_results,
        max_chars_per_source=_env_int(
            "COMPLEX_VALIDATION_MAX_SOURCE_CHARS",
            4000,
            minimum=500,
        ),
        max_total_chars=_env_int(
            "COMPLEX_VALIDATION_MAX_TOTAL_CHARS",
            40000,
            minimum=1000,
        ),
    )

    _debug(
        debug_enabled,
        "ANSWER_DECIDED",
        (
            f"model={search_plan_model} "
            f"answerable={answerable} "
            f"used_sources={used_source_ids} "
            f"missing_cells={len(context_summary['missing_period_cells'])} "
            f"elapsed={answer_seconds:.3f}s"
        ),
    )

    result: dict[str, Any] = {
        "success": True,
        "answerable": answerable,
        "status": "completed" if answerable else "insufficient_evidence",
        "stage": "answer_ready" if answerable else "answer_not_available",
        "task_type": TASK_TYPE,
        "answer": answer or INSUFFICIENT_EVIDENCE_ANSWER,
        "answer_with_sources": format_answer_with_sources(answer_result),
        "sources": sources,
        "used_source_ids": used_source_ids,
        "search_plan": search_plan,
        "search_plan_model": search_plan_model,
        "retry_attempted": False,
        "documents": context.get("documents") or [],
        "reranked_results": reranked_results,
        "retrieved_context": retrieved_context,
        # evidence_builder가 최우선으로 사용하는 검증 근거
        "validation_context": validation_context,
        # 안내 문구를 제외한 검증 대상 답변
        "validation_answer": validation_answer,
        "search_diagnostics": context.get("request_diagnostics") or [],
        "context_summary": context_summary,
        "answer_generation": {
            "model": answer_result.get("model"),
            "thinking_effort": answer_result.get("thinking_effort"),
            "finish_reason": answer_result.get("finish_reason"),
            "usage": answer_result.get("usage") or {},
        },
        "timings": {
            "search_plan_seconds": round(plan_seconds, 3),
            "search_seconds": round(search_seconds, 3),
            "answer_seconds": round(answer_seconds, 3),
        },
    }

    if debug_enabled:
        result["debug"] = {
            "raw_search_plan": raw_search_plan,
            "raw_answer_response": answer_result.get("raw_response"),
        }

    # -------------------------------------------------------------------------
    # 4. GROUNDING VALIDATION
    # -------------------------------------------------------------------------

    return _validate_attempt(
        result,
        question=question,
        debug_enabled=debug_enabled,
    )


def _should_retry_with_b(result: dict[str, Any]) -> bool:
    """1차 결과가 검색 품질 문제 또는 검증 실패일 때만 재시도한다."""

    if result.get("answerable"):
        validation = result.get("grounding_validation") or {}
        # 근거 검증을 통과하지 못한 답변은 근거 부족과 동일하게 취급한다.
        return bool(
            validation.get("performed")
            and not validation.get("passed")
        )

    return str(result.get("status") or "") in RETRYABLE_STATUSES


def _attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    """다른 시도의 결과를 진단용으로 요약한다."""

    context_summary = result.get("context_summary") or {}
    validation = result.get("grounding_validation") or {}

    return {
        "search_plan_model": result.get("search_plan_model"),
        "answerable": bool(result.get("answerable")),
        "status": result.get("status"),
        "used_source_ids": result.get("used_source_ids") or [],
        "chunk_count": context_summary.get("chunk_count", 0),
        "missing_period_cells": len(
            context_summary.get("missing_period_cells") or []
        ),
        "grounding_verdict": validation.get("verdict"),
        "grounding_reason": validation.get("reason"),
        "error_type": result.get("error_type"),
    }


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_complex_task(
    *,
    question: str,
    extracted: dict[str, Any],
    debug: bool | None = None,
) -> dict[str, Any]:
    """
    복합문서추론 전체 흐름을 실행한다.

    질문/키워드 결과
    → 파인튜닝 HCX 검색계획(A)
    → 운영 검색 실행기와 Chroma
    → HCX 최종 답변 및 답변 가능 여부 판단
    → 근거 검증(grounding validation)
    → 근거 부족이거나 검증 실패면 키워드 생성형 검색계획(B)으로 1회 재시도

    재시도는 검색계획 단계부터만 다시 돌린다. 키워드 추출 결과와 회사
    범위 확정은 그대로 재사용하고, B가 만든 계획은 통째로 사용한다.
    B 시도도 실패하면 A 결과를 반환한다.
    """

    started_total = time.perf_counter()
    debug_enabled = (
        _env_bool("COMPLEX_DEBUG", False)
        if debug is None
        else bool(debug)
    )

    question = str(question or "").strip()

    if not question:
        return _failure_result(
            status="empty_question",
            answer="질문을 입력해주세요.",
        )

    if not isinstance(extracted, dict):
        return _failure_result(
            status="invalid_keyword_result",
            answer="질문 분석 결과의 형식이 올바르지 않습니다.",
        )

    if extracted.get("is_complete") is not True:
        result = _failure_result(
            status="clarification_required",
            answer=str(
                extracted.get("clarification_question")
                or "추가 정보가 필요합니다."
            ),
        )
        result["missing_fields"] = extracted.get("missing_fields") or []
        result["clarification_question"] = extracted.get(
            "clarification_question"
        )
        return result

    _debug(debug_enabled, "START", f"question={question}")

    try:
        plan_client, executor, answer_generator = (
            initialize_complex_components(
                debug=debug_enabled,
            )
        )
    except Exception as exc:
        _debug(
            debug_enabled,
            "INIT_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        return _failure_result(
            status="complex_initialization_error",
            answer="복합문서추론 실행기를 초기화하지 못했습니다.",
            error=exc,
        )

    # -------------------------------------------------------------------------
    # 1차 시도: 기본 검색계획 모델(A)
    # -------------------------------------------------------------------------

    result_a = _run_single_attempt(
        question=question,
        extracted=extracted,
        plan_client=plan_client,
        executor=executor,
        answer_generator=answer_generator,
        search_plan_model=SEARCH_PLAN_MODEL_A,
        merge_plan_keywords=False,
        debug_enabled=debug_enabled,
    )

    final_result = result_a

    # -------------------------------------------------------------------------
    # 2차 시도: 키워드 생성형 검색계획 모델(B)
    # -------------------------------------------------------------------------

    if _should_retry_with_b(result_a):
        plan_client_b = get_search_plan_client_b(debug=debug_enabled)

        if plan_client_b is None:
            _debug(
                debug_enabled,
                "RETRY_SKIPPED",
                f"status={result_a.get('status')} plan_b=unavailable",
            )
        else:
            _debug(
                debug_enabled,
                "RETRY_START",
                (
                    f"status={result_a.get('status')} "
                    f"grounding={(result_a.get('grounding_validation') or {}).get('verdict')} "
                    "plan_model=B"
                ),
            )

            result_b: dict[str, Any] | None = None

            try:
                result_b = _run_single_attempt(
                    question=question,
                    extracted=extracted,
                    plan_client=plan_client_b,
                    executor=executor,
                    answer_generator=answer_generator,
                    search_plan_model=SEARCH_PLAN_MODEL_B,
                    merge_plan_keywords=True,
                    debug_enabled=debug_enabled,
                )
            except Exception as exc:
                # _run_single_attempt는 예외를 삼키도록 되어 있지만,
                # 예상 못 한 예외로 1차 결과까지 잃지 않도록 감싼다.
                _debug(
                    debug_enabled,
                    "RETRY_ERROR",
                    f"{type(exc).__name__}: {exc}",
                )
                result_a["retry_attempted"] = True
                result_a["retry_error"] = f"{type(exc).__name__}: {exc}"

            if result_b is not None:
                rank_a = _attempt_rank(result_a)
                rank_b = _attempt_rank(result_b)

                if rank_b > rank_a:
                    final_result = result_b
                    final_result["retry_attempted"] = True
                    final_result["first_attempt"] = _attempt_summary(
                        result_a
                    )
                else:
                    final_result = result_a
                    final_result["retry_attempted"] = True
                    final_result["retry_attempt"] = _attempt_summary(
                        result_b
                    )

                _debug(
                    debug_enabled,
                    "RETRY_DECIDED",
                    (
                        "selected_model="
                        f"{final_result.get('search_plan_model')} "
                        f"rank_a={rank_a} rank_b={rank_b}"
                    ),
                )

    # -------------------------------------------------------------------------
    # 검증 실패 답변 처리
    # -------------------------------------------------------------------------

    final_result = _apply_strict_validation(
        final_result,
        debug_enabled=debug_enabled,
    )

    total_seconds = time.perf_counter() - started_total
    timings = final_result.setdefault("timings", {})
    timings["total_seconds"] = round(total_seconds, 3)

    _debug(
        debug_enabled,
        "DONE",
        (
            f"model={final_result.get('search_plan_model')} "
            f"answerable={bool(final_result.get('answerable'))} "
            f"status={final_result.get('status')} "
            f"retry_attempted={bool(final_result.get('retry_attempted'))} "
            f"total_elapsed={total_seconds:.3f}s"
        ),
    )

    return final_result