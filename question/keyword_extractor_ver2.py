from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import CLOVA_STUDIO_API_KEY  # noqa: E402


# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------

DATA_ROOT = Path(r"C:\Users\User\.vscode\mirea_asset\corpus")
UNIVERSE_PATH = DATA_ROOT / "universe.csv"

DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
DEFAULT_MODEL = "HCX-005"

ALLOWED_TASK_TYPES = {
    "검색_정보추출",
    "다중조회_비교연산",
    "복합문서추론",
}

ALLOWED_DOC_TYPES = {
    "사업보고서",
    "반기보고서",
    "분기보고서",
    "주요사항보고서",
    "거래소공시",
    "지분공시",
}

ALLOWED_SCOPE_TYPES = {
    "direct",
    "group",
    "sector",
    "industry",
    "all",
    "unknown",
}

# universe.csv에 그룹 컬럼이 있다면 아래 후보 중 실제 존재하는 첫 컬럼을 사용한다.
GROUP_COLUMN_CANDIDATES = [
    "group",
    "group_name",
    "business_group",
    "conglomerate",
    "chaebol",
    "enterprise_group",
]

# sector / industry 컬럼 후보.
SECTOR_COLUMN_CANDIDATES = [
    "sector",
    "sector_name",
]

INDUSTRY_COLUMN_CANDIDATES = [
    "industry",
    "industry_name",
]

COMPANY_COLUMN_CANDIDATES = [
    "corp_name",
    "company_name",
    "listed_name",
]


MODEL_SYSTEM_PROMPT = (
    "당신은 기업 공시 및 금융 질의를 분석하는 검색 라우터입니다. "
    "질문에 명시된 정보만 구조화해서 추출하세요. "
    "질문에 없는 회사, 문서 유형, 기간은 추측하지 마세요. "
    "특히 개별 회사와 그룹/업종/분야 범위를 구분하세요."
)


MODEL_PROMPT_TEMPLATE = """
다음 금융/공시 질의를 분석하여 반드시 JSON 객체로만 응답하세요.

[출력 필드]

1) task_type
다음 중 하나:
- 검색_정보추출
- 다중조회_비교연산
- 복합문서추론

판단 기준:
- 검색_정보추출:
  한 회사/문서/지표에 대한 단순 사실 조회
- 다중조회_비교연산:
  둘 이상의 회사/범위/값을 비교하거나 합계, 차이, 순위, 증감률 등을 계산
- 복합문서추론:
  정정 이력, 시점별 변화, 원인-결과, 여러 문서 관계를 종합

2) scope_type
검색 대상 회사의 범위를 다음 중 하나로 분류:
- direct:
  특정 회사명이 직접 언급됨
- group:
  "현대 그룹", "두산 그룹", "삼성 계열"처럼 기업집단 단위
- sector:
  "방산", "2차전지", "반도체"처럼 분야/섹터 단위
- industry:
  특정 산업/업종 단위
- all:
  전체 기업을 대상으로 함
- unknown:
  범위를 판단할 수 없음

3) scope_values
scope_type에 해당하는 범위 이름 목록.

예:
"현대 그룹과 두산 그룹" ->
  scope_type = "group"
  scope_values = ["현대", "두산"]

"방산 기업 중" ->
  scope_type = "sector"
  scope_values = ["방산"]

"삼성전자와 SK하이닉스" ->
  scope_type = "direct"
  scope_values = ["삼성전자", "SK하이닉스"]

4) companies
질문에 직접 명시된 개별 회사명만 넣는다.

중요:
- group/sector/industry 질문에서는 소속 회사를 추측해서 companies에 나열하지 않는다.
- 예를 들어 "현대 그룹"이면 companies=[] 이고 scope_values=["현대"].
- 특정 회사 질문이면 canonical 회사명 목록을 넣는다.
- 질문에 직접 명시되지 않은 회사는 절대 추가하지 않는다.

5) doc_types
반드시 아래 값 중 질문에서 명시적으로 확인되는 것만 선택:
- 사업보고서
- 반기보고서
- 분기보고서
- 주요사항보고서
- 거래소공시
- 지분공시

질문에 문서 유형이 없으면 [].
목록에 없는 문서 유형은 만들지 않는다.

6) time
질문의 시점/기간을 구조화한다.

반드시 아래 JSON 구조를 사용한다.

단일 시점:
"time": {
  "type": "point",
  "start": {
    "year": 2025,
    "quarter": 1,
    "half": null
  },
  "end": null
}

기간 범위:
"time": {
  "type": "range",
  "start": {
    "year": 2023,
    "quarter": 3,
    "half": null
  },
  "end": {
    "year": 2025,
    "quarter": 4,
    "half": null
  }
}

연도만 있는 단일 시점:
"time": {
  "type": "point",
  "start": {
    "year": 2025,
    "quarter": null,
    "half": null
  },
  "end": null
}

규칙:
- type은 "point", "range", "unknown" 중 하나
- year는 integer 또는 null
- quarter는 1~4 integer 또는 null
- half는 "H1", "H2" 또는 null
- 질문에 기간이 없으면:
  "time": {
    "type": "unknown",
    "start": null,
    "end": null
  }
- 범위 질문은 반드시 start와 end를 구분한다.
- 예: "2023년 3분기에서 2025년 4분기까지"
  -> type="range", start=2023 Q3, end=2025 Q4
- 예: "2025년 1분기"
  -> type="point", start=2025 Q1, end=null
- 질문에 없는 기간은 절대 추측하지 않는다.

7) metrics
질문의 핵심 금융/공시 지표 목록.
예:
["매출액"], ["수출액"], ["계약금액"], ["취득예정주식수"]
없으면 []

8) actions
질문이 요구하는 핵심 행동 목록.

예:
- "비교해줘", "차이는?" -> ["비교"]
- "어떻게 변했어?", "추이는?" -> ["추이 분석"]
- "어떻게 정정됐어?", "정정 이력은?" -> ["정정 이력 확인"]
- "가장 큰 곳은?", "순위는?" -> ["순위"]
- 단순 값 조회 -> []

중요:
- "변화", "추이" 질문을 "정정 이력 확인"으로 분류하지 말 것.
- "정정", "정정 이력"이 명시된 경우에만 "정정 이력 확인"을 사용한다.

9) topic_keywords
검색에 유용한 핵심 키워드 목록.

[출력 예시 1]
질문: 현대자동차의 2026년 수출액은?
{
  "task_type": "검색_정보추출",
  "scope_type": "direct",
  "scope_values": ["현대자동차"],
  "companies": ["현대자동차"],
  "doc_types": [],
  "time": {
    "type": "point",
    "start": {
      "year": 2026,
      "quarter": null,
      "half": null
    },
    "end": null
  },
  "metrics": ["수출액"],
  "actions": [],
  "topic_keywords": ["현대자동차", "2026년", "수출액"]
}

[출력 예시 2]
질문: 두산 그룹과 현대 그룹의 수출액을 비교해줘.
{
  "task_type": "다중조회_비교연산",
  "scope_type": "group",
  "scope_values": ["두산", "현대"],
  "companies": [],
  "doc_types": [],
  "time": {
    "type": "unknown",
    "start": null,
    "end": null
  },
  "metrics": ["수출액"],
  "actions": ["비교"],
  "topic_keywords": ["두산 그룹", "현대 그룹", "수출액"]
}

[출력 예시 3]
질문: 방산 기업 중 2025년 매출액이 가장 큰 회사는?
{
  "task_type": "다중조회_비교연산",
  "scope_type": "sector",
  "scope_values": ["방산"],
  "companies": [],
  "doc_types": [],
  "time": {
    "type": "point",
    "start": {
      "year": 2025,
      "quarter": null,
      "half": null
    },
    "end": null
  },
  "metrics": ["매출액"],
  "actions": ["순위"],
  "topic_keywords": ["방산", "2025년", "매출액"]
}

반드시 유효한 JSON 객체만 출력하세요.
추가 설명이나 Markdown 코드 블록은 출력하지 마세요.

질문:
{question}
""".strip()


# ----------------------------------------------------------------------
# 결과 구조
# ----------------------------------------------------------------------

@dataclass
class ExtractedKeywords:
    question: str

    task_type: str = ""

    # HCX가 해석한 검색 범위
    scope_type: str = "unknown"
    scope_values: list = field(default_factory=list)

    # 질문에 직접 명시된 회사
    companies: list = field(default_factory=list)

    # universe.csv에서 실제 검색 대상으로 확장된 회사
    target_companies: list = field(default_factory=list)

    doc_types: list = field(default_factory=list)

    # 구조화된 시간 정보
    time_type: str = "unknown"       # point / range / unknown
    start_year: int | None = None
    start_quarter: int | None = None
    start_half: str | None = None
    end_year: int | None = None
    end_quarter: int | None = None
    end_half: str | None = None

    metrics: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    topic_keywords: list = field(default_factory=list)

    # Resolver 진단용
    scope_column: str | None = None
    unresolved_scope_values: list = field(default_factory=list)

    # 필수 슬롯 검증용
    is_complete: bool = False
    missing_fields: list = field(default_factory=list)
    clarification_question: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "task_type": self.task_type,
            "scope_type": self.scope_type,
            "scope_values": self.scope_values,
            "companies": self.companies,
            "target_companies": self.target_companies,
            "doc_types": self.doc_types,
            "time_type": self.time_type,
            "start_year": self.start_year,
            "start_quarter": self.start_quarter,
            "start_half": self.start_half,
            "end_year": self.end_year,
            "end_quarter": self.end_quarter,
            "end_half": self.end_half,
            "metrics": self.metrics,
            "actions": self.actions,
            "topic_keywords": self.topic_keywords,
            "scope_column": self.scope_column,
            "unresolved_scope_values": self.unresolved_scope_values,
            "is_complete": self.is_complete,
            "missing_fields": self.missing_fields,
            "clarification_question": self.clarification_question,
        }


# ----------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------

def ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_lookup_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().casefold()
    text = re.sub(r"\s+", "", text)

    # 그룹 표현에서 검색을 방해하는 일반 접미사 제거
    for suffix in (
        "그룹전체",
        "전체그룹",
        "그룹",
        "계열사",
        "계열",
        "기업",
        "회사",
        "업체",
        "분야",
        "섹터",
        "산업",
        "업종",
    ):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]

    return text


def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lowered = {
        str(column).casefold(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        actual = lowered.get(candidate.casefold())
        if actual:
            return actual

    return None



def normalize_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_half(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().upper()

    mapping = {
        "H1": "H1",
        "H2": "H2",
        "상반기": "H1",
        "하반기": "H2",
        "1": "H1",
        "2": "H2",
    }

    return mapping.get(text)


def parse_time_body(
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    HCX의 structured time 응답을 정규화한다.
    """
    time_body = body.get("time") or {}

    if not isinstance(time_body, dict):
        time_body = {}

    time_type = str(
        time_body.get("type") or "unknown"
    ).strip().lower()

    if time_type not in {
        "point",
        "range",
        "unknown",
    }:
        time_type = "unknown"

    start = time_body.get("start")
    end = time_body.get("end")

    if not isinstance(start, dict):
        start = {}

    if not isinstance(end, dict):
        end = {}

    start_year = normalize_int_or_none(
        start.get("year")
    )
    start_quarter = normalize_int_or_none(
        start.get("quarter")
    )
    start_half = normalize_half(
        start.get("half")
    )

    end_year = normalize_int_or_none(
        end.get("year")
    )
    end_quarter = normalize_int_or_none(
        end.get("quarter")
    )
    end_half = normalize_half(
        end.get("half")
    )

    if (
        start_quarter is not None
        and start_quarter not in {1, 2, 3, 4}
    ):
        start_quarter = None

    if (
        end_quarter is not None
        and end_quarter not in {1, 2, 3, 4}
    ):
        end_quarter = None

    # start 자체가 없으면 시간 정보 없음
    if start_year is None:
        time_type = "unknown"

    # range인데 end가 없으면 불완전한 range로 유지
    # -> validator에서 누락 처리
    return {
        "time_type": time_type,
        "start_year": start_year,
        "start_quarter": start_quarter,
        "start_half": start_half,
        "end_year": end_year,
        "end_quarter": end_quarter,
        "end_half": end_half,
    }


def parse_json_content(
    content: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(content, dict):
        return content

    text = content.strip()

    # HCX가 지시를 어기고 ```json ... ```을 붙이는 경우 방어
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "HyperCLOVA X 응답을 JSON으로 파싱하지 못했습니다.\n"
            f"원본 응답:\n{content}"
        ) from exc

# ----------------------------------------------------------------------
# HyperCLOVA X
# ----------------------------------------------------------------------

class HyperClovaXKeywordExtractor:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL,
        timeout: int = 30,
        debug: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError(
                "CLOVA Studio API key가 필요합니다."
            )

        self.api_key = api_key
        self.api_endpoint = (
            base_url.rstrip("/")
            + "/chat/completions"
        )
        self.model_name = model_name
        self.timeout = timeout
        self.debug = debug

    def extract_keywords(
        self,
        question: str,
    ) -> ExtractedKeywords:
        question = question.strip()

        if not question:
            raise ValueError("질문이 비어 있습니다.")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": MODEL_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": MODEL_PROMPT_TEMPLATE.replace(
                        "{question}",
                        question
                    ),
                },
            ],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_tokens": 1200,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"HyperCLOVA X API 요청 실패: {exc}"
            ) from exc

        if self.debug:
            print(
                f"[DEBUG] status_code = "
                f"{response.status_code}"
            )
            print(
                f"[DEBUG] response text = "
                f"{response.text}"
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                "HyperCLOVA X API 오류\n"
                f"HTTP {response.status_code}\n"
                f"{response.text}"
            ) from exc

        try:
            data = response.json()
            content = (
                data["choices"][0]
                ["message"]["content"]
            )
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "HyperCLOVA X API 응답 구조가 "
                "예상과 다릅니다.\n"
                f"{response.text}"
            ) from exc

        body = parse_json_content(content)

        if self.debug:
            print(
                f"[DEBUG] parsed body = {body}"
            )

        return self._parse_api_response(
            body,
            question,
        )

    @staticmethod
    def _parse_api_response(
        body: dict[str, Any],
        question: str,
    ) -> ExtractedKeywords:

        task_type = str(
            body.get("task_type")
            or "검색_정보추출"
        ).strip()

        if task_type not in ALLOWED_TASK_TYPES:
            raise ValueError(
                "허용되지 않은 task_type: "
                f"{task_type}"
            )

        scope_type = str(
            body.get("scope_type")
            or "unknown"
        ).strip()

        if scope_type not in ALLOWED_SCOPE_TYPES:
            raise ValueError(
                "허용되지 않은 scope_type: "
                f"{scope_type}"
            )

        companies = [
            str(value).strip()
            for value in ensure_list(
                body.get("companies")
            )
            if str(value).strip()
        ]

        scope_values = [
            str(value).strip()
            for value in ensure_list(
                body.get("scope_values")
            )
            if str(value).strip()
        ]

        doc_types = [
            str(value).strip()
            for value in ensure_list(
                body.get("doc_types")
            )
            if str(value).strip()
            in ALLOWED_DOC_TYPES
        ]

        time_data = parse_time_body(
            body
        )

        return ExtractedKeywords(
            question=question,
            task_type=task_type,
            scope_type=scope_type,
            scope_values=scope_values,
            companies=companies,
            doc_types=doc_types,
            time_type=time_data["time_type"],
            start_year=time_data["start_year"],
            start_quarter=time_data["start_quarter"],
            start_half=time_data["start_half"],
            end_year=time_data["end_year"],
            end_quarter=time_data["end_quarter"],
            end_half=time_data["end_half"],
            metrics=ensure_list(
                body.get("metrics")
            ),
            actions=ensure_list(
                body.get("actions")
            ),
            topic_keywords=ensure_list(
                body.get("topic_keywords")
            ),
        )



# ----------------------------------------------------------------------
# Required Slot Validator
# ----------------------------------------------------------------------

def has_valid_time(
    extracted: ExtractedKeywords,
) -> bool:
    """
    WHEN 슬롯 검사.

    point:
      start_year가 있으면 유효

    range:
      start_year와 end_year가 모두 있어야 유효
    """
    if extracted.time_type == "point":
        return extracted.start_year is not None

    if extracted.time_type == "range":
        return (
            extracted.start_year is not None
            and extracted.end_year is not None
        )

    return False


def has_meaningful_topic(
    extracted: ExtractedKeywords,
) -> bool:
    """
    WHAT 슬롯 검사.

    metrics가 있으면 가장 명확한 검색 주제.
    metrics가 없어도 topic_keywords 중 회사명/시간 외 의미 있는
    이벤트/공시 주제가 있으면 WHAT으로 인정.
    """
    if extracted.metrics:
        return True

    excluded = {
        normalize_lookup_text(value)
        for value in (
            list(extracted.scope_values)
            + list(extracted.companies)
        )
    }

    for keyword in extracted.topic_keywords:
        normalized = normalize_lookup_text(keyword)

        if not normalized:
            continue

        if normalized in excluded:
            continue

        # 연도/분기처럼 시간 표현만 있는 경우 제외
        if re.search(r"20\d{2}", str(keyword)):
            temporal = str(keyword)
            temporal = re.sub(r"20\d{2}\s*년?", "", temporal)
            temporal = re.sub(r"[1-4]\s*분기", "", temporal)
            temporal = temporal.replace("상반기", "")
            temporal = temporal.replace("하반기", "")
            if not temporal.strip():
                continue

        return True

    return False


def validate_required_slots(
    extracted: ExtractedKeywords,
) -> ExtractedKeywords:
    """
    필수 슬롯:
    WHO  -> scope/company
    WHEN -> point 또는 range
    WHAT -> metric/topic
    """
    missing = []

    has_scope = bool(
        extracted.scope_values
        or extracted.companies
        or extracted.scope_type == "all"
    )

    if not has_scope:
        missing.append("scope")

    if not has_valid_time(extracted):
        missing.append("time")

    if not has_meaningful_topic(extracted):
        missing.append("topic")

    extracted.missing_fields = missing
    extracted.is_complete = not missing

    return extracted


def build_clarification_question(
    extracted: ExtractedKeywords,
) -> str | None:
    missing = extracted.missing_fields

    if not missing:
        return None

    if missing == ["scope"]:
        return (
            "어떤 기업, 그룹 또는 분야를 대상으로 확인할까요? "
            "예: 삼성전자, 현대 그룹, 방산 기업"
        )

    if missing == ["time"]:
        return (
            "어느 시점 또는 기간을 기준으로 확인할까요? "
            "예: 2025년 1분기 또는 "
            "2023년 3분기부터 2025년 4분기까지"
        )

    if missing == ["topic"]:
        return (
            "어떤 내용을 확인하고 싶은가요? "
            "예: 매출액, 수출액, 계약금액, 정정 내용"
        )

    parts = []

    if "scope" in missing:
        parts.append(
            "어떤 기업·그룹·분야를 대상으로 할지"
        )

    if "time" in missing:
        parts.append(
            "어느 시점 또는 기간인지"
        )

    if "topic" in missing:
        parts.append(
            "무엇을 확인하고 싶은지"
        )

    return (
        "질문을 처리하려면 "
        + ", ".join(parts)
        + "를 알려주세요."
    )


# ----------------------------------------------------------------------
# Scope Resolver
# ----------------------------------------------------------------------

class CompanyScopeResolver:
    """
    HCX가 추출한 scope를 universe.csv의 실제 기업 목록으로 변환한다.

    핵심 원칙:
    - LLM은 "방산", "현대 그룹" 같은 의미 해석만 담당
    - 실제 검색 대상 회사는 universe.csv를 source of truth로 사용
    """

    def __init__(
        self,
        universe_path: Path,
    ) -> None:
        if not universe_path.exists():
            raise FileNotFoundError(
                f"universe.csv를 찾을 수 없습니다: "
                f"{universe_path}"
            )

        self.universe = pd.read_csv(
            universe_path,
            dtype=str,
            encoding="utf-8-sig",
        ).fillna("")

        self.company_column = (
            find_first_existing_column(
                self.universe,
                COMPANY_COLUMN_CANDIDATES,
            )
        )

        if self.company_column is None:
            raise ValueError(
                "universe.csv에서 회사명 컬럼을 "
                "찾지 못했습니다. "
                f"현재 컬럼: "
                f"{list(self.universe.columns)}"
            )

        self.group_column = (
            find_first_existing_column(
                self.universe,
                GROUP_COLUMN_CANDIDATES,
            )
        )

        self.sector_column = (
            find_first_existing_column(
                self.universe,
                SECTOR_COLUMN_CANDIDATES,
            )
        )

        self.industry_column = (
            find_first_existing_column(
                self.universe,
                INDUSTRY_COLUMN_CANDIDATES,
            )
        )

    def _resolve_group_by_company_name(
        self,
        extracted: ExtractedKeywords,
    ) -> ExtractedKeywords:

        company_series = (
            self.universe[self.company_column]
            .astype(str)
        )

        normalized_company_series = (
            company_series
            .map(normalize_lookup_text)
        )

        matched_companies = []
        unresolved = []

        for scope_value in extracted.scope_values:
            needle = normalize_lookup_text(
                scope_value
            )

            if not needle:
                continue

            mask = (
                normalized_company_series
                .str.contains(
                    re.escape(needle),
                    regex=True,
                    na=False,
                )
            )

            if not mask.any():
                unresolved.append(
                    scope_value
                )
                continue

            matched_companies.extend(
                company_series[
                    mask
                ].tolist()
            )

        extracted.target_companies = sorted(
            {
                company.strip()
                for company in matched_companies
                if company.strip()
            }
        )

        extracted.scope_column = (
            self.company_column
        )

        extracted.unresolved_scope_values = (
            unresolved
        )

        return extracted

    def resolve(
        self,
        extracted: ExtractedKeywords,
    ) -> ExtractedKeywords:

        scope_type = extracted.scope_type

        if scope_type == "direct":
            extracted.target_companies = (
                self._resolve_direct_companies(
                    extracted.companies
                    or extracted.scope_values
                )
            )
            extracted.scope_column = (
                self.company_column
            )
            return extracted

        if scope_type == "all":
            extracted.target_companies = (
                self._all_companies()
            )
            extracted.scope_column = (
                self.company_column
            )
            return extracted

        if scope_type == "group":
            return self._resolve_group_by_company_name(extracted)

        if scope_type == "sector":
            return self._resolve_by_column(
                extracted,
                column=self.sector_column,
                scope_label="sector",
            )

        if scope_type == "industry":
            return self._resolve_by_column(
                extracted,
                column=self.industry_column,
                scope_label="industry",
            )

        # unknown이면 직접 회사명이 있을 때만 그것으로 검색
        if extracted.companies:
            extracted.target_companies = (
                self._resolve_direct_companies(
                    extracted.companies
                )
            )
            extracted.scope_column = (
                self.company_column
            )

        return extracted

    def _all_companies(self) -> list[str]:
        values = (
            self.universe[self.company_column]
            .astype(str)
            .str.strip()
        )

        return sorted(
            {
                value
                for value in values
                if value
            }
        )

    def _resolve_direct_companies(
        self,
        requested_companies: list,
    ) -> list[str]:

        resolved = []

        company_series = (
            self.universe[self.company_column]
            .astype(str)
        )

        normalized_company_series = (
            company_series
            .map(normalize_lookup_text)
        )

        for requested in requested_companies:
            needle = normalize_lookup_text(
                requested
            )

            if not needle:
                continue

            exact_mask = (
                normalized_company_series
                == needle
            )

            if exact_mask.any():
                resolved.extend(
                    company_series[
                        exact_mask
                    ].tolist()
                )
                continue

            # exact가 없을 때만 부분 일치
            contains_mask = (
                normalized_company_series
                .str.contains(
                    re.escape(needle),
                    regex=True,
                    na=False,
                )
            )

            resolved.extend(
                company_series[
                    contains_mask
                ].tolist()
            )

        return sorted(
            {
                company.strip()
                for company in resolved
                if company.strip()
            }
        )

    def _resolve_by_column(
        self,
        extracted: ExtractedKeywords,
        column: str | None,
        scope_label: str,
    ) -> ExtractedKeywords:

        if column is None:
            extracted.unresolved_scope_values = (
                list(extracted.scope_values)
            )

            print(
                f"[WARN] universe.csv에 "
                f"{scope_label}용 컬럼이 없습니다."
            )

            return extracted

        extracted.scope_column = column

        matched_companies = []
        unresolved = []

        column_series = (
            self.universe[column]
            .astype(str)
        )

        normalized_series = (
            column_series
            .map(normalize_lookup_text)
        )

        for scope_value in extracted.scope_values:
            needle = normalize_lookup_text(
                scope_value
            )

            if not needle:
                continue

            # exact 우선
            mask = normalized_series == needle

            # exact가 없으면 contains
            if not mask.any():
                mask = (
                    normalized_series
                    .str.contains(
                        re.escape(needle),
                        regex=True,
                        na=False,
                    )
                )

            if not mask.any():
                unresolved.append(
                    scope_value
                )
                continue

            matched_companies.extend(
                self.universe.loc[
                    mask,
                    self.company_column,
                ].tolist()
            )

        extracted.target_companies = sorted(
            {
                str(company).strip()
                for company
                in matched_companies
                if str(company).strip()
            }
        )

        extracted.unresolved_scope_values = (
            unresolved
        )

        return extracted

    def describe_columns(self) -> None:
        print("\n=== universe.csv Resolver 컬럼 ===")
        print(
            f"회사명: {self.company_column}"
        )
        print(
            f"그룹: {self.company_column} 포함 검색"
        )
        print(
            f"섹터: {self.sector_column}"
        )
        print(
            f"산업: {self.industry_column}"
        )


# ----------------------------------------------------------------------
# 통합 함수
# ----------------------------------------------------------------------

def extract_and_resolve(
    question: str,
    client: HyperClovaXKeywordExtractor,
    resolver: CompanyScopeResolver,
) -> ExtractedKeywords:

    # 1. HCX 구조화
    extracted = client.extract_keywords(
        question
    )

    # 2. WHO / WHEN / WHAT 검증
    extracted = validate_required_slots(
        extracted
    )

    # 3. 부족하면 retrieval 전에 중단
    if not extracted.is_complete:
        extracted.clarification_question = (
            build_clarification_question(
                extracted
            )
        )
        return extracted

    # 4. 모두 갖춰졌을 때만 기업 범위 확장
    return resolver.resolve(
        extracted
    )


def print_result(
    result: ExtractedKeywords,
) -> None:

    print(
        "\n=== HyperCLOVA X + Scope Resolver 결과 ==="
    )

    print(
        f"질문: {result.question}"
    )
    print(
        f"  task_type: {result.task_type}"
    )
    print(
        f"  scope_type: {result.scope_type}"
    )
    print(
        f"  scope_values: "
        f"{result.scope_values}"
    )
    print(
        f"  companies: {result.companies}"
    )
    print(
        f"  target_companies: "
        f"{result.target_companies}"
    )
    print(
        f"  scope_column: "
        f"{result.scope_column}"
    )
    print(
        f"  unresolved_scope_values: "
        f"{result.unresolved_scope_values}"
    )
    print(
        f"  doc_types: {result.doc_types}"
    )
    print(
        f"  time_type: {result.time_type}"
    )
    print(
        f"  start_year: {result.start_year}"
    )
    print(
        f"  start_quarter: {result.start_quarter}"
    )
    print(
        f"  start_half: {result.start_half}"
    )
    print(
        f"  end_year: {result.end_year}"
    )
    print(
        f"  end_quarter: {result.end_quarter}"
    )
    print(
        f"  end_half: {result.end_half}"
    )
    print(
        f"  is_complete: {result.is_complete}"
    )
    print(
        f"  missing_fields: {result.missing_fields}"
    )
    print(
        f"  clarification_question: "
        f"{result.clarification_question}"
    )
    print(
        f"  metrics: {result.metrics}"
    )
    print(
        f"  actions: {result.actions}"
    )
    print(
        f"  topic_keywords: "
        f"{result.topic_keywords}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:

    key = os.getenv(
        "CLOVA_STUDIO_API_KEY",
        CLOVA_STUDIO_API_KEY,
    )

    if not key:
        raise RuntimeError(
            "CLOVA_STUDIO_API_KEY가 "
            "설정되어 있지 않습니다."
        )

    base_url = os.getenv(
        "HYPERCLOVA_X_ENDPOINT",
        DEFAULT_BASE_URL,
    )

    client = HyperClovaXKeywordExtractor(
        api_key=key,
        base_url=base_url,
        model_name=DEFAULT_MODEL,
        debug=False,
    )

    resolver = CompanyScopeResolver(
        UNIVERSE_PATH
    )

    resolver.describe_columns()

    print(
        "\n== HyperCLOVA X Scope Extractor =="
    )

    original_question = input(
        "질의: "
    ).strip()

    conversation_question = original_question

    while True:
        try:
            result = extract_and_resolve(
                conversation_question,
                client,
                resolver,
            )
        except Exception as exc:
            print(
                f"\n추출/범위 해석 실패: {exc}"
            )
            raise SystemExit(1)

        if result.is_complete:
            print_result(result)
            break

        print("\n=== 추가 정보 필요 ===")
        print(
            f"누락 슬롯: {result.missing_fields}"
        )
        print(
            f"확인 질문: "
            f"{result.clarification_question}"
        )

        additional = input(
            "답변: "
        ).strip()

        if not additional:
            print(
                "필요한 정보를 입력해주세요."
            )
            continue

        if additional.lower() in {
            "취소",
            "quit",
            "q",
        }:
            print("종료합니다.")
            break

        conversation_question = (
            conversation_question
            + "\n[사용자 추가 정보] "
            + additional
        )


if __name__ == "__main__":
    main()