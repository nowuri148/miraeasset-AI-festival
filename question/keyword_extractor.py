"""
질의(question) 키워드 추출기
- 회사명 (약칭/오타/영문 표기 포함, 유사도 매칭)
- 문서 유형 (사업보고서/분기보고서/주요사항보고서/거래소공시/지분공시)
- 기간 (연도, 분기, 반기)
- 질문 유형 (검색·정보추출 / 다중조회·비교연산 / 복합문서추론) — 이전에 설계한 라우터의 1차 구현

사용법:
    python keyword_extractor.py
    (하단 __main__ 에서 샘플 질의로 데모 실행. 실제 서비스에서는 extract_keywords()를 import해서 사용)
"""

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

try:
    import requests  # type: ignore[import]
except ImportError:
    requests = None  # requests is optional; only required for external API 사용

DATA_ROOT = Path(r"C:\Users\User\.vscode\mirea_asset\corpus")  # universe.csv 있는 경로로 수정

# ---------------------------------------------------------------
# 0. 추가 키워드 / 태스크 라벨 / 정제용 상수
# ---------------------------------------------------------------

METRIC_KEYWORDS = [
    "매출", "매출액", "매출총이익", "영업이익", "당기순이익", "순이익", "이익", "영업현금흐름",
    "자산", "부채", "자본", "부채비율", "지배주주지분", "ROE", "ROA", "영업이익률", "순이익률",
    "EPS", "PER", "PBR", "시가총액", "기대감", "투자", "설비투자", "연구개발비", "R&D", "배당", "현금", "자금조달",
    "매출원가", "판매관리비", "부채비율", "영업이익률", "증감률", "성장률", "비중"
]

ACTION_KEYWORDS = [
    "정리", "요약", "설명", "비교", "순위", "추이", "변화", "이후", "전망", "유형별", "원인", "분석", "구분", "정정",
    "요청", "요청해", "찾아", "알려줘", "알려주세요", "비교해", "확인", "나열", "정리해줘"
]

COMMON_STOP_WORDS = {
    "은", "는", "이", "가", "의", "와", "과", "을", "를", "에서", "에게", "보다", "이나", "나", "에", "만", "도",
    "보자", "다음", "그리고", "그", "이벤트", "관련", "사항", "요", "좀", "더", "같은",
}

TASK_KEYWORDS = {
    "다중조회_비교연산": [
        "비교", "더 큰", "더 높은", "증감률", "증가율", "감소율", "비중", "몇 배", "차이", "순위", "높은", "많은",
        "A와 B", "A와", "B와", "중 어떤", "둘 중", "비교했을 때", "중에서",
    ],
    "복합문서추론": [
        "추이", "변화", "이후", "정정", "변경 이력", "어떻게 달라", "어떻게 변화", "원인", "결과", "정리해줘",
        "설명해줘", "사유", "배경", "의미", "파악", "추론",
    ],
    "검색_정보추출": [
        "알려줘", "찾아", "요약", "정리", "확인", "문의", "질문", "제공", "정보", "조회", "검색",
    ],
}

TASK_LABELS = {
    "검색_정보추출": "검색 및 정보 추출",
    "다중조회_비교연산": "다중 조회 및 비교 연산",
    "복합문서추론": "복합 문서 추론",
}

MODEL_PROMPT_TEMPLATE = (
    "질의를 분석하여 아래 JSON 스키마에 맞게 응답하세요.\n"
    "1) task_type: 검색_정보추출 / 다중조회_비교연산 / 복합문서추론 중 하나\n"
    "2) companies: 질문에 언급된 회사 이름의 canonical 명칭 목록\n"
    "3) doc_types: 관련 문서 유형 목록\n"
    "4) periods: 언급된 기간 목록\n"
    "5) metrics: 질문 핵심 지표 목록\n"
    "6) actions: 질문 핵심 동사/행동 목록\n"
    "7) topic_keywords: 핵심 키워드 목록\n"
    "질문: {question}"
)

# ---------------------------------------------------------------
# 1. 회사명 별칭 테이블 구축
# ---------------------------------------------------------------

# README에서 확인된 "통용명이 DART 공식 법인명과 다른" 대표 케이스
# (실제 universe.csv를 로드하면 이보다 훨씬 많은 기업이 자동으로 채워짐 — 이건 보강용 수동 별칭)
MANUAL_ALIASES = {
    "현대차": "현대자동차",
    "현대자동차": "현대자동차",
    "기아차": "기아",
    "KT": "케이티",
    "케이티": "케이티",
    "엔씨": "NC",
    "엔씨소프트": "NC",
    "NC소프트": "NC",
    "LIG넥스원": "LIG디펜스앤에어로스페이스",
    "LIG디펜스": "LIG디펜스앤에어로스페이스",
    "삼전": "삼성전자",
    "SK하닉": "SK하이닉스",
    "하이닉스": "SK하이닉스",
    "네이버": "NAVER",
    "카카오": "카카오",
    "포스코": "POSCO홀딩스",
}

# 회사명 뒤에 흔히 붙는 조사 — 매칭 전에 제거해야 정확도가 올라감
PARTICLES = ["은", "는", "이", "가", "의", "와", "과", "을", "를",
             "에서", "에게", "보다", "이나", "나", "에", "만", "도"]


def strip_particle(token: str) -> str:
    for p in sorted(PARTICLES, key=len, reverse=True):
        if token.endswith(p) and len(token) > len(p):
            return token[: -len(p)]
    return token


def build_company_lookup(universe: pd.DataFrame) -> dict:
    """
    universe.csv의 corp_name / listed_name / corp_eng_name / stock_code를
    모두 '별칭 -> 공식 법인명(corp_name)' 형태로 펼친 룩업 테이블 생성
    """
    lookup = {}
    for _, row in universe.iterrows():
        canonical = row["corp_name"]
        for alias in [row["corp_name"], row["listed_name"], row["corp_eng_name"], row["stock_code"]]:
            if isinstance(alias, str) and alias.strip():
                lookup[alias.strip()] = canonical

    # 수동 별칭 병합 (README에서 확인된 특수 케이스 + 흔한 축약형)
    for alias, canonical_hint in MANUAL_ALIASES.items():
        # canonical_hint가 실제 universe의 corp_name과 매칭되는지 확인 후 등록
        matches = universe[universe["corp_name"].str.contains(canonical_hint, na=False, regex=False)]
        if len(matches):
            lookup[alias] = matches.iloc[0]["corp_name"]
        else:
            lookup[alias] = canonical_hint  # universe에 없으면 일단 힌트 그대로 저장

    return lookup


def find_company_mentions(question: str, lookup: dict, fuzzy_threshold: int = 80) -> list:
    """
    질의 텍스트에서 회사명(정확 매칭 + 유사 매칭)을 찾아
    [{'matched_text':.., 'canonical_name':.., 'score':..}, ...] 형태로 반환
    """
    results = []
    seen_canonical = set()

    # 1) 정확 매칭 (긴 이름부터 먼저 검사해야 "삼성" vs "삼성전자" 같은 부분 겹침 방지)
    for alias in sorted(lookup.keys(), key=len, reverse=True):
        if alias and alias in question:
            canonical = lookup[alias]
            if canonical not in seen_canonical:
                results.append({"matched_text": alias, "canonical_name": canonical, "score": 100})
                seen_canonical.add(canonical)

    if results:
        return results  # 정확히 찾았으면 굳이 유사매칭까지 안 감 (오탐 방지)

    # 2) 유사 매칭 (오타/변형 대비) — 질문을 어절 단위로 쪼개서 각각 조사 제거 후 비교
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
    candidates = list(lookup.keys())

    for token in tokens:
        cleaned = strip_particle(token)
        if len(cleaned) < 2:
            continue
        match = process.extractOne(cleaned, candidates, scorer=fuzz.WRatio, score_cutoff=fuzzy_threshold)
        if match:
            matched_alias, score, _ = match
            canonical = lookup[matched_alias]
            if canonical not in seen_canonical:
                results.append({"matched_text": token, "canonical_name": canonical, "score": round(score, 1)})
                seen_canonical.add(canonical)

    return results


# ---------------------------------------------------------------
# 2. 문서 유형 키워드 추출
# ---------------------------------------------------------------

DOC_TYPE_KEYWORDS = {
    ("periodic", "annual"): ["사업보고서", "연간보고서", "연차보고서"],
    ("periodic", "half"): ["반기보고서"],
    ("periodic", "quarter"): ["분기보고서", "1분기", "2분기", "3분기"],
    ("major", None): ["주요사항보고서", "무상증자", "유상증자", "자금조달", "CB", "BW", "EB", "전환사채", "신주인수권부사채"],
    ("exchange", None): ["단일판매", "공급계약", "신규시설투자", "투자판단", "계약체결", "계약해지"],
    ("holding", None): ["대량보유", "지분", "5%보고", "지분변동", "최대주주"],
}


def find_doc_type_mentions(question: str) -> list:
    hits = []
    for (doc_group, doc_subtype), keywords in DOC_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in question:
                hits.append({"doc_group": doc_group, "doc_subtype": doc_subtype, "keyword": kw})
    return hits


# ---------------------------------------------------------------
# 3. 기간(연도/분기/반기) 추출
# ---------------------------------------------------------------

def find_period_mentions(question: str) -> list:
    periods = []

    # "2025년" "2025.12" "2025-12" 형태의 연도
    for m in re.finditer(r"(20\d{2})\s*년?", question):
        periods.append({"type": "year", "value": int(m.group(1))})

    # "1분기" "2025년 1분기" 등
    for m in re.finditer(r"([1-4])\s*분기", question):
        periods.append({"type": "quarter", "value": int(m.group(1))})

    # "상반기" "하반기"
    if "상반기" in question:
        periods.append({"type": "half", "value": "H1"})
    if "하반기" in question:
        periods.append({"type": "half", "value": "H2"})

    return periods


# ---------------------------------------------------------------
# 4. 질문 유형(Task) 분류 및 핵심 키워드 추출
# ---------------------------------------------------------------

def get_task_label(task_type: str) -> str:
    return TASK_LABELS.get(task_type, "검색 및 정보 추출")


def extract_question_keywords(question: str) -> Dict[str, list]:
    normalized = question.strip()
    found_metrics = [kw for kw in METRIC_KEYWORDS if kw in normalized]
    found_actions = [kw for kw in ACTION_KEYWORDS if kw in normalized]

    tokens = [strip_particle(tok) for tok in re.findall(r"[가-힣A-Za-z0-9]+", normalized)]
    keywords = [tok for tok in tokens if tok not in COMMON_STOP_WORDS]
    keywords = [tok for tok in keywords if tok not in found_metrics and tok not in found_actions]
    keywords = [tok for tok in keywords if tok not in {kw for kws in DOC_TYPE_KEYWORDS.values() for kw in kws}]

    topic_counts = Counter(keywords)
    topic_keywords = [tok for tok, _ in topic_counts.most_common(12) if len(tok) > 1]

    return {
        "metrics": sorted(set(found_metrics), key=found_metrics.index),
        "actions": sorted(set(found_actions), key=found_actions.index),
        "topic_keywords": topic_keywords,
    }


def classify_task_type(question: str, company_mentions: list) -> str:
    normalized = question.strip()
    for task, keywords in TASK_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            return task

    if len(company_mentions) >= 2:
        return "다중조회_비교연산"

    return "검색_정보추출"


class HyperClovaXKeywordExtractor:
    def __init__(self, api_endpoint: Optional[str] = None, api_key: Optional[str] = None, model_name: str = "hyperclovax"):
        self.api_endpoint = api_endpoint or os.getenv("HYPERCLOVA_X_ENDPOINT") or "https://clovastudio.stream.ntruss.com/"
        self.api_key = api_key or os.getenv("HYPERCLOVA_X_API_KEY")
        self.model_name = model_name

    def extract_keywords(self, question: str) -> "ExtractedKeywords":
        if requests is None:
            raise ImportError("requests 라이브러리가 필요합니다. pip install requests")
        if not self.api_endpoint:
            raise ValueError("HyperClovaX API endpoint가 설정되어야 합니다. HYPERCLOVA_X_ENDPOINT 환경변수를 확인하세요.")

        payload = {
            "model": self.model_name,
            "input": MODEL_PROMPT_TEMPLATE.format(question=question),
            "metadata": {
                "task": "keyword_extraction",
                "schema": ["task_type", "companies", "doc_types", "periods", "metrics", "actions", "topic_keywords"],
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        return self._parse_api_response(data, question)

    def _parse_api_response(self, response: Dict[str, Any], question: str) -> "ExtractedKeywords":
        body = response.get("output") or response
        task_type = body.get("task_type", "검색_정보추출")
        companies = body.get("companies", []) or []
        doc_types = body.get("doc_types", []) or []
        periods = body.get("periods", []) or []
        metrics = body.get("metrics", []) or []
        actions = body.get("actions", []) or []
        topic_keywords = body.get("topic_keywords", []) or []

        return ExtractedKeywords(
            question=question,
            companies=companies,
            doc_types=doc_types,
            periods=periods,
            task_type=task_type,
            metrics=metrics,
            actions=actions,
            topic_keywords=topic_keywords,
        )


# ---------------------------------------------------------------
# 5. 통합 추출 함수
# ---------------------------------------------------------------

@dataclass
class ExtractedKeywords:
    question: str
    companies: list = field(default_factory=list)
    doc_types: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    task_type: str = ""
    metrics: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    topic_keywords: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "companies": self.companies,
            "doc_types": self.doc_types,
            "periods": self.periods,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "actions": self.actions,
            "topic_keywords": self.topic_keywords,
            "task_label": get_task_label(self.task_type),
        }


def extract_keywords(question: str, company_lookup: dict, model_client: Optional[HyperClovaXKeywordExtractor] = None) -> ExtractedKeywords:
    if model_client is not None:
        try:
            return model_client.extract_keywords(question)
        except Exception:
            pass

    companies = find_company_mentions(question, company_lookup)
    doc_types = find_doc_type_mentions(question)
    periods = find_period_mentions(question)
    extras = extract_question_keywords(question)
    task_type = classify_task_type(question, companies)

    return ExtractedKeywords(
        question=question,
        companies=companies,
        doc_types=doc_types,
        periods=periods,
        task_type=task_type,
        metrics=extras["metrics"],
        actions=extras["actions"],
        topic_keywords=extras["topic_keywords"],
    )


# ---------------------------------------------------------------
# 데모
# ---------------------------------------------------------------

if __name__ == "__main__":
    universe = pd.read_csv(
        DATA_ROOT / "universe.csv",
        dtype={"corp_code": str, "stock_code": str},
        encoding="utf-8-sig",
    )
    
    key = "nv-53d015cd5a4f4240abb926ad7c755937abxC"
    
    company_lookup = build_company_lookup(universe)
    print(f"회사명 룩업 테이블 크기: {len(company_lookup)}개 별칭")

    sample_questions = [
        "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
        "2차전지 기업 A와 B 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?",
        "현대차가 2025년에 실시한 자금조달 내역을 유형별로 정리해줘",
        "삼전 2026년 1분기 분기보고서를 기준으로 주요 투자 계획을 정리해줘",
        "케이티가 2023년 사업보고서와 2025년 사업보고서를 비교했을 때 핵심 사업은 어떻게 변화했는지 설명해줘",
    ]

    # for q in sample_questions:
    #     result = extract_keywords(q, company_lookup)
    #     print(f"\n질의: {q}")
    #     print(f"  회사: {result.companies}")
    #     print(f"  문서유형: {result.doc_types}")
    #     print(f"  기간: {result.periods}")
    #     print(f"  Task유형: {result.task_type} ({get_task_label(result.task_type)})")
    #     print(f"  지표: {result.metrics}")
    #     print(f"  행동: {result.actions}")
    #     print(f"  토픽 키워드: {result.topic_keywords}")

    endpoint = os.getenv("HYPERCLOVA_X_ENDPOINT", "https://clovastudio.stream.ntruss.com/")
    if endpoint:
        print("\nHyperClovaX API 연동 예시를 실행합니다...")
        api_client = HyperClovaXKeywordExtractor(api_endpoint=endpoint, api_key=key)
        for q in sample_questions[:]:
            try:
                model_result = extract_keywords(q, company_lookup, model_client=api_client)
                print(f"\n[모델] 질의: {q}")
                print(f"  task_type: {model_result.task_type}")
                print(f"  companies: {model_result.companies}")
                print(f"  doc_types: {model_result.doc_types}")
                print(f"  periods: {model_result.periods}")
                print(f"  metrics: {model_result.metrics}")
                print(f"  actions: {model_result.actions}")
                print(f"  topic_keywords: {model_result.topic_keywords}")
            except Exception as exc:
                print(f"  HyperClovaX API 호출 실패: {exc}")
    else:
        print("\nHYPERCLOVA_X_ENDPOINT 환경변수가 설정되지 않아 HyperClovaX API 예시는 실행되지 않습니다.")