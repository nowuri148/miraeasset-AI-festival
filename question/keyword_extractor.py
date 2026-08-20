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
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import CLOVA_STUDIO_API_KEY  # noqa: E402

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
    "정리", "요약", "설명", "비교", "더 큰", "더 높은", "가장 큰", "가장 높은", "많은", "높은", "낮은",
    "순위", "추이", "변화", "이후", "전망", "유형별", "원인", "분석", "구분", "정정",
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
    "JYP": "JYP Ent",
    "jyp": "JYP Ent",
    "제이와이피": "JYP Ent",
    "LIG": "LIG디펜스앤에어로스페이스",
    "LIG넥스원": "LIG디펜스앤에어로스페이스",
    "LIG디펜스": "LIG디펜스앤에어로스페이스",
    "신재생": "신재생에너지",
    "신재생에너지": "신재생에너지",
    "SM": "SM",
    "에스엠": "SM",
    "삼전": "삼성전자",
    "SK하닉": "SK하이닉스",
    "하이닉스": "SK하이닉스",
    "네이버": "NAVER",
    "카카오": "카카오",
    "포스코": "POSCO홀딩스",
    "현대 그룹": "현대 그룹",
    "현대 그룹 전체": "현대 그룹",
    "두산 그룹": "두산 그룹",
    "두산 그룹 전체": "두산 그룹",
}

# 회사명 뒤에 흔히 붙는 조사 — 매칭 전에 제거해야 정확도가 올라감
PARTICLES = ["은", "는", "이", "가", "의", "와", "과", "랑", "이랑", "하고", "를",
             "에서", "에게", "보다", "이나", "나", "에", "만", "도", "며"]


def strip_particle(token: str) -> str:
    for p in sorted(PARTICLES, key=len, reverse=True):
        if token.endswith(p) and len(token) > len(p):
            return token[: -len(p)]
    return token


def normalize_choice_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip()
    normalized = re.sub(r"[，,]", " ", normalized)
    normalized = re.sub(r"(?:을|를|이|가)?\s*(말해|알려줘|알려주세요|말해줘|비교해|비교해줘|부탁해요|주세요)$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def split_user_choices(raw_choice: str) -> list[str]:
    if not raw_choice:
        return []
    parts = re.split(
        r"[，,]|(?:\s*그리고\s*)|(?:\s*및\s*)|(?:\s*이랑\s*)|(?:\s*랑\s*)|(?:\s*와\s*)|(?:\s*과\s*)|(?:\s*하고\s*)|(?:\s*&\s*)",
        raw_choice,
    )
    return [normalize_choice_text(part) for part in parts if normalize_choice_text(part)]


def normalize_group_choice(choice: str) -> str:
    if not choice:
        return ""
    normalized = choice
    normalized = re.sub(r"\b전체\s*(현대|두산|삼성|LG|SK)\s*그룹\b", r"\1 그룹", normalized)
    normalized = re.sub(r"\b(현대|두산|삼성|LG|SK)\s*전체\s*그룹\b", r"\1 그룹", normalized)
    normalized = re.sub(r"\b(현대|두산|삼성|LG|SK)\s*그룹\s*전체\b", r"\1 그룹", normalized)
    normalized = re.sub(r"\b(기업|회사|업체|분야|업종|섹터|산업)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_group_mention(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip()
    normalized = re.sub(r"(?:기업|회사|업체|분야|업종|섹터|산업)$", "", normalized)
    return normalize_alias_key(normalized)


def build_choice_map(
    user_choice: str,
    ambiguous_mentions: list,
    ambiguous_companies: list,
    ambiguous_candidate_groups: Optional[Dict[str, list]] = None,
) -> dict:
    choice_map: dict[str, str] = {}
    alias_lookup = {normalize_alias_key(alias): canonical for alias, canonical in MANUAL_ALIASES.items()}
    alias_lookup.update({normalize_alias_key(candidate): candidate for candidate in ambiguous_companies})
    alias_lookup.update({normalize_alias_key(mention): mention for mention in ambiguous_mentions})

    normalized_candidates = {normalize_alias_key(candidate): candidate for candidate in ambiguous_companies}
    candidate_to_mention: dict[str, str] = {}
    if ambiguous_candidate_groups:
        for mention, candidates in ambiguous_candidate_groups.items():
            for candidate in candidates:
                candidate_to_mention[normalize_alias_key(candidate)] = mention

    mention_by_number = {str(idx): mention for idx, mention in enumerate(ambiguous_mentions, start=1)}

    for choice in split_user_choices(user_choice):
        normalized_choice = normalize_alias_key(normalize_group_choice(choice))

        if normalized_choice in mention_by_number:
            mention = mention_by_number[normalized_choice]
            choice_map[mention] = mention
            continue

        resolved_choice = alias_lookup.get(normalized_choice) or normalized_candidates.get(normalized_choice)
        if resolved_choice:
            if resolved_choice in normalized_candidates.values():
                candidate = resolved_choice
                candidate_key = normalize_alias_key(candidate)
                mapped_mention = candidate_to_mention.get(candidate_key)
                if mapped_mention:
                    choice_map[mapped_mention] = candidate
                    continue
                for mention in ambiguous_mentions:
                    if normalize_alias_key(mention) in candidate_key:
                        choice_map[mention] = candidate
                        break
                continue

            for mention in ambiguous_mentions:
                if normalize_alias_key(mention) == normalize_alias_key(resolved_choice):
                    choice_map[mention] = mention
                    break
            continue

        for mention in ambiguous_mentions:
            if normalize_alias_key(mention) in normalized_choice:
                choice_map[mention] = choice
                break

        if ambiguous_candidate_groups:
            for mention, candidates in ambiguous_candidate_groups.items():
                if normalize_alias_key(mention) == normalized_choice:
                    choice_map[mention] = mention
                    break

    return choice_map


def replace_mention_with_selection(question: str, mention: str, selection: str) -> str:
    token_pattern = "|".join(re.escape(p) for p in sorted(PARTICLES, key=len, reverse=True))
    pattern = fr"{re.escape(mention)}(?:{token_pattern})?"
    return re.sub(pattern, selection, question, count=1)


def normalize_alias_key(text: str) -> str:
    folded = text.casefold()
    folded = re.sub(r"\s+", "", folded)
    folded = re.sub(r"[^\w가-힣]", "", folded)
    return folded


def get_generic_prefixes(name: str, max_prefix_length: int = 5) -> set:
    prefixes = set()
    for token in re.findall(r"[가-힣A-Za-z0-9]+", name):
        folded = normalize_alias_key(token)
        if len(folded) < 2:
            continue
        for i in range(2, min(len(folded), max_prefix_length) + 1):
            prefixes.add(folded[:i])
    return prefixes


@dataclass
class CompanyLookup:
    alias_to_canonical: Dict[str, str]
    ambiguous_aliases: Dict[str, List[str]]
    generic_prefixes: Dict[str, List[str]]
    group_aliases: Dict[str, List[str]]
    alias_count: int
    unique_company_count: int


def expand_group_aliases(value: str) -> List[str]:
    if not value or not isinstance(value, str):
        return []
    raw_parts = re.split(r"[/&|,·\s]+", value.strip())
    aliases = set()
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        aliases.add(cleaned)
        if cleaned.endswith("분야"):
            aliases.add(cleaned[:-2])
        if cleaned.endswith("업종"):
            aliases.add(cleaned[:-2])
        if "엔터" in cleaned or "테인먼트" in cleaned:
            aliases.add("엔터")
            aliases.add("엔터테인먼트")
        if "방산" in cleaned:
            aliases.add("방산")
            aliases.add("방산분야")
    return sorted({normalize_alias_key(alias) for alias in aliases if alias})


def expand_company_aliases(value: str) -> List[str]:
    if not value or not isinstance(value, str):
        return []
    compiled = set()
    raw = value.strip()
    if not raw:
        return []
    compiled.add(raw)
    compiled.add(raw.replace("&", "and"))
    compact = re.sub(r"[^A-Za-z0-9]", "", raw)
    if compact:
        compiled.add(compact)
    for part in re.split(r"[\s&/.-]+", raw):
        cleaned = part.strip()
        if cleaned and len(cleaned) >= 2:
            compiled.add(cleaned)
    return [alias for alias in sorted(compiled, key=len, reverse=True) if alias]


def build_company_lookup(universe: pd.DataFrame) -> CompanyLookup:
    """
    universe.csv의 corp_name / listed_name / corp_eng_name / stock_code를
    모두 '별칭 -> 공식 법인명(corp_name)' 형태로 펼친 룩업 테이블 생성
    """
    alias_to_canonical = {}
    alias_to_canonicals = defaultdict(set)
    canonical_names = set()
    group_aliases: Dict[str, List[str]] = defaultdict(list)

    for _, row in universe.iterrows():
        canonical = row["corp_name"]
        canonical_names.add(canonical)
        for alias in [row["corp_name"], row["listed_name"], row["corp_eng_name"], row["stock_code"]]:
            if not isinstance(alias, str) or not alias.strip():
                continue
            for expanded in expand_company_aliases(alias):
                key = normalize_alias_key(expanded)
                alias_to_canonicals[key].add(canonical)
                if key not in alias_to_canonical:
                    alias_to_canonical[key] = canonical

        for field_name in ["sector", "industry"]:
            field_value = row.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                for alias in expand_group_aliases(field_value):
                    group_aliases[alias].append(canonical)

    # 수동 별칭 병합 (README에서 확인된 특수 케이스 + 흔한 축약형)
    for alias, canonical_hint in MANUAL_ALIASES.items():
        key = normalize_alias_key(alias)
        matches = universe[universe["corp_name"].str.contains(canonical_hint, na=False, regex=False)]
        if len(matches):
            canonical = matches.iloc[0]["corp_name"]
        else:
            canonical = canonical_hint
        alias_to_canonicals[key].add(canonical)
        if key not in alias_to_canonical:
            alias_to_canonical[key] = canonical

    prefix_candidates = defaultdict(set)
    for name in canonical_names:
        for prefix in get_generic_prefixes(name):
            prefix_candidates[prefix].add(name)

    ambiguous_aliases = {
        alias: sorted(canonicals)
        for alias, canonicals in alias_to_canonicals.items()
        if len(canonicals) > 1
    }
    for prefix, names in prefix_candidates.items():
        if len(names) > 1 and prefix not in alias_to_canonical:
            ambiguous_aliases[prefix] = sorted(names)

    generic_prefixes = {
        prefix: sorted(names)
        for prefix, names in prefix_candidates.items()
        if len(names) > 1
    }

    normalized_group_aliases = {
        alias: sorted(set(companies))
        for alias, companies in group_aliases.items()
        if len(set(companies)) > 1
    }

    return CompanyLookup(
        alias_to_canonical=alias_to_canonical,
        ambiguous_aliases=ambiguous_aliases,
        generic_prefixes=generic_prefixes,
        group_aliases=normalized_group_aliases,
        alias_count=len(alias_to_canonical),
        unique_company_count=len(canonical_names),
    )


def find_company_mentions(question: str, company_lookup: CompanyLookup, fuzzy_threshold: int = 80) -> list:
    """
    질의 텍스트에서 회사명(정확 매칭 + 유사 매칭)을 찾아
    [{'matched_text':.., 'canonical_name':.., 'score':..}, ...] 형태로 반환
    """
    results = []
    ambiguous_results = []
    seen_canonical = set()
    question_folded = normalize_alias_key(question)

    matched_group_aliases = set()
    for token in re.findall(r"[가-힣A-Za-z0-9]+", question):
        cleaned = strip_particle(token)
        if len(cleaned) < 2:
            continue
        group_key = normalize_group_mention(cleaned)
        if group_key in company_lookup.group_aliases:
            matched_group_aliases.add(group_key)

    for alias in sorted(matched_group_aliases, key=len, reverse=True):
        ambiguous_results.append({
            "matched_text": alias,
            "ambiguous": True,
            "candidate_names": company_lookup.group_aliases[alias],
            "score": 100,
        })

    if ambiguous_results:
        return ambiguous_results

    # 1) 정확 매칭 (긴 이름부터 먼저 검사해야 "삼성" vs "삼성전자" 같은 부분 겹침 방지)
    for alias in sorted(company_lookup.alias_to_canonical.keys(), key=len, reverse=True):
        if alias and alias in question_folded:
            if alias in company_lookup.ambiguous_aliases:
                ambiguous_results.append({
                    "matched_text": alias,
                    "ambiguous": True,
                    "candidate_names": company_lookup.ambiguous_aliases[alias],
                    "score": 100,
                })
                continue
            canonical = company_lookup.alias_to_canonical[alias]
            if canonical not in seen_canonical:
                results.append({"matched_text": alias, "canonical_name": canonical, "score": 100})
                seen_canonical.add(canonical)

    if ambiguous_results:
        return ambiguous_results
    if results:
        return results  # 정확히 찾았으면 굳이 유사매칭까지 안 감 (오탐 방지)

    # 2) 유사 매칭 (오타/변형 대비) — 질문을 어절 단위로 쪼개서 각각 조사 제거 후 비교
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
    candidates = list(company_lookup.alias_to_canonical.keys())

    for token in tokens:
        cleaned = strip_particle(token)
        if len(cleaned) < 2:
            continue
        cleaned_key = normalize_alias_key(cleaned)
        if cleaned_key in company_lookup.ambiguous_aliases:
            ambiguous_results.append({
                "matched_text": cleaned,
                "ambiguous": True,
                "candidate_names": company_lookup.ambiguous_aliases[cleaned_key],
                "score": 100,
            })
            continue
        if cleaned_key in company_lookup.generic_prefixes:
            ambiguous_results.append({
                "matched_text": cleaned,
                "ambiguous": True,
                "candidate_names": company_lookup.generic_prefixes[cleaned_key],
                "score": 100,
            })
            continue
        match = process.extractOne(cleaned_key, candidates, scorer=fuzz.WRatio, score_cutoff=fuzzy_threshold)
        if match:
            matched_alias, score, _ = match
            canonical = company_lookup.alias_to_canonical[matched_alias]
            if canonical not in seen_canonical:
                results.append({"matched_text": token, "canonical_name": canonical, "score": round(score, 1)})
                seen_canonical.add(canonical)

    if ambiguous_results:
        return ambiguous_results

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

    def ask_disambiguation(self, question: str, candidate_names: List[str]) -> str:
        if requests is None:
            raise ImportError("requests 라이브러리가 필요합니다. pip install requests")
        if not self.api_endpoint:
            raise ValueError("HyperClovaX API endpoint가 설정되어야 합니다. HYPERCLOVA_X_ENDPOINT 환경변수를 확인하세요.")

        candidate_text = ", ".join(candidate_names)
        prompt = (
            "질문에서 언급된 회사명이 모호합니다.\n"
            f"질문: {question}\n"
            f"회사 후보: {candidate_text}\n"
            "위 후보 중에서 질문 작성자가 어떤 회사를 의미하는지 명확히 물어보는 한국어 문장을 생성하세요."
        )

        payload = {
            "model": self.model_name,
            "input": prompt,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        body = data.get("output") or data
        if isinstance(body, dict):
            return body.get("text") or body.get("output") or str(body)
        return str(body)


# ---------------------------------------------------------------
# 5. 통합 추출 함수
# ---------------------------------------------------------------

@dataclass
class ExtractedKeywords:
    question: str
    companies: list = field(default_factory=list)
    ambiguous_companies: list = field(default_factory=list)
    ambiguous_mentions: list = field(default_factory=list)
    ambiguous_candidate_groups: dict = field(default_factory=dict)
    clarification_prompt: Optional[str] = None
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
            "ambiguous_companies": self.ambiguous_companies,
            "clarification_prompt": self.clarification_prompt,
            "ambiguous_mentions": self.ambiguous_mentions,
            "doc_types": self.doc_types,
            "periods": self.periods,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "actions": self.actions,
            "topic_keywords": self.topic_keywords,
            "task_label": get_task_label(self.task_type),
        }


def summarize_group_label(label: str) -> str:
    if not label:
        return "기타"
    label_key = normalize_alias_key(label)
    summary_map = {
        normalize_alias_key("방산"): "방산",
        normalize_alias_key("방산분야"): "방산",
        normalize_alias_key("항공우주"): "방산",
        normalize_alias_key("소비재"): "소비재",
        normalize_alias_key("유통"): "소비재",
        normalize_alias_key("소비재유통"): "소비재",
        normalize_alias_key("엔터"): "엔터",
        normalize_alias_key("엔터테인먼트"): "엔터",
        normalize_alias_key("조선"): "조선",
        normalize_alias_key("원전"): "원전",
    }
    if label_key in summary_map:
        return summary_map[label_key]
    if label_key.endswith(normalize_alias_key("그룹")):
        return label.replace("그룹", "그룹")
    return label


def build_ambiguity_prompt(
    question: str,
    candidate_groups: Optional[Dict[str, List[str]]] = None,
    ambiguous_mentions: Optional[list] = None,
    candidates: Optional[List[str]] = None,
) -> str:
    if ambiguous_mentions:
        if len(ambiguous_mentions) == 1:
            group_hint = f"전체 {ambiguous_mentions[0]} 그룹"
        else:
            group_hint = " 또는 ".join(f"전체 {mention} 그룹" for mention in ambiguous_mentions)
    else:
        group_hint = "전체 그룹"

    grouped_candidates: Dict[str, List[str]] = defaultdict(list)
    if candidate_groups:
        for mention, group_candidates in candidate_groups.items():
            group_name = summarize_group_label(mention)
            grouped_candidates[group_name].extend(group_candidates)
    elif candidates:
        for candidate in candidates:
            matched_group = "기타"
            candidate_key = normalize_alias_key(candidate)
            for mention in ambiguous_mentions or []:
                mention_key = normalize_alias_key(mention)
                if mention_key in candidate_key or candidate_key in mention_key:
                    matched_group = f"{mention} 그룹"
                    break
            grouped_candidates[matched_group].append(candidate)

    lines = [
        f"질문이 모호합니다. '{question}'에서 어떤 회사를 의미하나요?",
        "============================================================================",
        f"다음 후보 중 하나를 선택하거나 '{group_hint}'처럼 전체 범위를 말해 주세요.",
    ]

    for idx, (group_name, group_items) in enumerate(grouped_candidates.items(), start=1):
        lines.append(f"{idx}. {group_name}")
        for item in sorted(set(group_items)):
            lines.append(f"   - {item}")

    if not grouped_candidates:
        lines.append(f"- 후보: {', '.join(candidates)}")

    return "\n".join(lines)


def build_period_clarification_prompt(question: str) -> str:
    return (
        "질문에 필요한 기간 정보가 부족합니다. "
        "최소 연도 단위(예: 2024년, 2025년, 2025년 1분기) 중 어느 기간을 의미하나요? "
        f"질문 원문: '{question}'"
    )


def extract_keywords(question: str, company_lookup: CompanyLookup, model_client: Optional[HyperClovaXKeywordExtractor] = None) -> ExtractedKeywords:
    companies = find_company_mentions(question, company_lookup)
    ambiguous_companies = []
    ambiguous_mentions = []
    clarification_prompt = None
    if companies and isinstance(companies[0], dict) and companies[0].get("ambiguous"):
        ambiguous_candidate_groups = {
            company.get("matched_text"): company.get("candidate_names", [])
            for company in companies
            if company.get("matched_text")
        }
        ambiguous_companies = sorted({
            candidate
            for candidates in ambiguous_candidate_groups.values()
            for candidate in candidates
        })
        ambiguous_mentions = list(ambiguous_candidate_groups.keys())
        companies = []
        if model_client is not None and ambiguous_companies:
            try:
                clarification_prompt = model_client.ask_disambiguation(question, ambiguous_companies)
            except Exception:
                clarification_prompt = None
        if not clarification_prompt and ambiguous_companies:
            clarification_prompt = build_ambiguity_prompt(
                question,
                ambiguous_candidate_groups,
                ambiguous_mentions,
            )

        extras = extract_question_keywords(question)
        return ExtractedKeywords(
            question=question,
            companies=[],
            ambiguous_companies=ambiguous_companies,
            ambiguous_mentions=ambiguous_mentions,
            ambiguous_candidate_groups=ambiguous_candidate_groups,
            clarification_prompt=clarification_prompt,
            doc_types=find_doc_type_mentions(question),
            periods=find_period_mentions(question),
            task_type=classify_task_type(question, companies if companies else ambiguous_mentions),
            metrics=extras["metrics"],
            actions=extras["actions"],
            topic_keywords=extras["topic_keywords"],
        )

    if model_client is not None:
        try:
            extracted = model_client.extract_keywords(question)
            if extracted.companies:
                return extracted
        except Exception:
            pass

    doc_types = find_doc_type_mentions(question)
    periods = find_period_mentions(question)
    extras = extract_question_keywords(question)
    task_type = classify_task_type(question, companies)

    return ExtractedKeywords(
        question=question,
        companies=companies,
        ambiguous_companies=ambiguous_companies,
        clarification_prompt=clarification_prompt,
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
    
    key = os.getenv("CLOVA_STUDIO_API_KEY", CLOVA_STUDIO_API_KEY)
    
    company_lookup = build_company_lookup(universe)
    print(f"회사명 룩업 테이블 크기: {company_lookup.alias_count}개 별칭")
    print(f"로드한 회사 수: {company_lookup.unique_company_count}개")

    # sample_questions = [
    #     "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
    #     "2차전지 기업 A와 B 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?",
    #     "현대차가 2025년에 실시한 자금조달 내역을 유형별로 정리해줘",
    #     "삼전 2026년 1분기 분기보고서를 기준으로 주요 투자 계획을 정리해줘",
    #     "케이티가 2023년 사업보고서와 2025년 사업보고서를 비교했을 때 핵심 사업은 어떻게 변화했는지 설명해줘",
    # ]
    
    print("== input을 입력하세요 ==")
    original_question = input("질의: ")
    question = original_question

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
    if endpoint and key:
        print("\nHyperClovaX API 연동 예시를 실행합니다...")
        api_client = HyperClovaXKeywordExtractor(api_endpoint=endpoint, api_key=key)
        final_result = None
        while True:
            try:
                model_result = extract_keywords(question, company_lookup, model_client=api_client)
                print(f"\n[모델] 질의: {question}")
                print(f"  task_type: {model_result.task_type}")
                print(f"  companies: {model_result.companies}")
                print(f"  ambiguous_companies: {model_result.ambiguous_companies}")
                print(f"  clarification_prompt: {model_result.clarification_prompt}")
                print(f"  doc_types: {model_result.doc_types}")
                print(f"  periods: {model_result.periods}")
                print(f"  metrics: {model_result.metrics}")
                print(f"  actions: {model_result.actions}")
                print(f"  topic_keywords: {model_result.topic_keywords}")

                if model_result.ambiguous_companies:
                    prompt = model_result.clarification_prompt or build_ambiguity_prompt(
                        question,
                        model_result.ambiguous_candidate_groups or {
                            mention: model_result.ambiguous_companies
                            for mention in model_result.ambiguous_mentions
                        },
                        model_result.ambiguous_mentions,
                    )
                    print(f"\n확인 질문: {prompt}\n")
                    user_choice = input("회사명 또는 그룹 번호/이름을 입력하세요: ").strip()
                    if not user_choice:
                        print("회사명을 다시 선택해 주세요.")
                        continue
                    if user_choice.lower() == "취소":
                        break
                    choice_map = build_choice_map(
                        user_choice,
                        model_result.ambiguous_mentions,
                        model_result.ambiguous_companies,
                        model_result.ambiguous_candidate_groups,
                    )
                    if not choice_map:
                        print("입력한 회사명을 인식하지 못했습니다. 후보 중 하나를 정확히 입력해 주세요.\n")
                        continue
                    new_question = original_question
                    for mention, selection in choice_map.items():
                        new_question = replace_mention_with_selection(new_question, mention, selection)
                    question = new_question
                    print(f"\n선택 반영 후 질문: {question}\n")

                    if not find_period_mentions(question):
                        print(f"\n기간 확인: {build_period_clarification_prompt(question)}\n")
                        period_input = input("기간을 입력하세요: ").strip()
                        if period_input:
                            question = f"{question} {period_input}"
                            final_result = extract_keywords(question, company_lookup, model_client=api_client)
                        else:
                            print("기간을 다시 입력해 주세요.\n")
                            continue
                    else:
                        final_result = model_result
                    break

                if not find_period_mentions(question):
                    print(f"\n기간 확인: {build_period_clarification_prompt(question)}\n")
                    period_input = input("기간을 입력하세요: ").strip()
                    if period_input:
                        question = f"{question} {period_input}"
                        final_result = extract_keywords(question, company_lookup, model_client=api_client)
                    else:
                        print("기간을 다시 입력해 주세요.\n")
                        continue
                else:
                    final_result = model_result
                break
            except Exception as exc:
                print(f"  HyperClovaX API 호출 실패: {exc}")
                break

        if final_result is not None:
            print("\n=== 최종 추출 결과 ===")
            print(f"질문: {question}")
            print(f"  task_type: {final_result.task_type}")
            print(f"  companies: {final_result.companies}")
            print(f"  ambiguous_companies: {final_result.ambiguous_companies}")
            print(f"  doc_types: {final_result.doc_types}")
            print(f"  periods: {final_result.periods}")
            print(f"  metrics: {final_result.metrics}")
            print(f"  actions: {final_result.actions}")
            print(f"  topic_keywords: {final_result.topic_keywords}")
    else:
        print(
            "\nHYPERCLOVA_X_ENDPOINT 또는 CLOVA_STUDIO_API_KEY 환경변수가 "
            "설정되지 않아 HyperClovaX API 예시는 실행되지 않습니다."
        )
