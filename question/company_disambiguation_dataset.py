import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "corpus" / "universe.csv"
OUTPUT_PATH = ROOT / "corpus" / "company_disambiguation_train.jsonl"


def normalize_alias(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    text = text.replace(".", "").replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


def build_alias_map(df: pd.DataFrame) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for _, row in df.iterrows():
        canonical = str(row["corp_name"]).strip()
        for raw in [row["corp_name"], row["listed_name"], row["corp_eng_name"], row["stock_code"]]:
            if isinstance(raw, str):
                alias = normalize_alias(raw)
                if alias and alias not in alias_map:
                    alias_map[alias] = canonical
    # 예외 별칭 보강
    extra_aliases = {
        "jyp": "JYP Ent",
        "제이와이피": "JYP Ent",
        "현대차": "현대자동차",
        "기아차": "기아",
        "엔씨": "NC",
        "엔씨소프트": "NC",
        "케이티": "케이티",
        "포스코": "POSCO홀딩스",
        "현대": "현대자동차",
        "두산": "두산로보틱스",
        "삼성": "삼성전자",
        "lg": "LG에너지솔루션",
        "sk": "SK하이닉스",
    }
    for alias, canonical in extra_aliases.items():
        alias_map.setdefault(normalize_alias(alias), canonical)
    return alias_map


def build_ambiguous_family_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    family_map: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        corp_name = str(row["corp_name"]).strip()
        if not corp_name:
            continue
        prefix = corp_name[:2]
        if re.search(r"[가-힣]", prefix):
            family_map[prefix].add(corp_name)

    ambiguous = {
        prefix: sorted(names)
        for prefix, names in family_map.items()
        if len(names) > 1 and prefix in {"현대", "삼성", "두산", "LG", "SK", "NC", "KB", "CJ", "한화", "포스코", "OCI", "세아", "롯데"}
    }

    # 명시적 그룹핑: 계열사 이름이 같은 뿌리에서 많이 파생되는 케이스를 포함
    explicit_groups = {
        "현대": ["현대자동차", "현대모비스", "현대오토에버", "현대제철"],
        "삼성": ["삼성전자", "삼성전기", "삼성SDI", "삼성생명"],
        "두산": ["두산로보틱스", "두산에너빌리티", "두산퓨얼셀"],
        "LG": ["LG에너지솔루션", "LG이노텍", "LG생활건강", "LG유플러스"],
        "SK": ["SK하이닉스", "SK텔레콤", "SK이노베이션", "SKC"],
        "NC": ["NC", "엔씨소프트"],
    }
    for prefix, names in explicit_groups.items():
        if names:
            ambiguous[prefix] = sorted(set(ambiguous.get(prefix, []) + names))
    return {k: v for k, v in ambiguous.items() if len(v) > 1}


def make_example(question: str, candidate_companies: list[str], is_ambiguous: bool, canonical_company: str | None = None) -> dict:
    if is_ambiguous:
        clarification = (
            f"질문이 모호합니다. '{question}'에서 어떤 회사를 의미하나요? "
            f"다음 후보 중 하나를 선택해 주세요: {', '.join(candidate_companies)}"
        )
        output = {
            "is_ambiguous": True,
            "candidate_companies": candidate_companies,
            "canonical_company": None,
            "clarification_question": clarification,
        }
    else:
        output = {
            "is_ambiguous": False,
            "candidate_companies": [canonical_company] if canonical_company else [],
            "canonical_company": canonical_company,
            "clarification_question": "",
        }

    return {
        "instruction": "질문에서 회사명이 모호하면 후보를 생성하고 재질문을 하십시오. 명확하면 canonical 회사명을 반환하십시오.",
        "input": {
            "question": question,
            "task": "company_ambiguity_resolution",
            "candidate_companies": candidate_companies,
        },
        "output": output,
    }


def build_dataset(df: pd.DataFrame) -> list[dict]:
    alias_map = build_alias_map(df)
    ambiguous_groups = build_ambiguous_family_groups(df)
    rows: list[dict] = []

    # 모호한 사례 생성
    for family_name, candidate_companies in ambiguous_groups.items():
        for question in [
            f"{family_name} 2025년 매출은 얼마인가?",
            f"{family_name} 2025년 사업보고서를 정리해줘",
            f"{family_name} 2024년 영업이익과 순이익을 비교해줘",
        ]:
            rows.append(make_example(question, candidate_companies, is_ambiguous=True))

    # 명확한 사례 생성
    for _, row in df.iterrows():
        corp_name = str(row["corp_name"]).strip()
        listed_name = str(row["listed_name"]).strip()
        alias_names = [listed_name, corp_name, row["corp_eng_name"]]
        for alias in [n for n in alias_names if n and n.strip()]:
            question = f"{alias} 2025년 매출과 영업이익을 정리해줘"
            rows.append(make_example(question, [alias], is_ambiguous=False, canonical_company=corp_name))

        # alias 표기 변형 예시
        alias_variants = {
            "JYP Ent": "JYP Ent 2025년 연결기준 성과를 설명해줘",
            "현대차": "현대차 2025년 사업보고서 핵심 내용을 요약해줘",
            "기아차": "기아차 2024년 투자계획을 정리해줘",
            "엔씨소프트": "엔씨소프트 2025년 영업이익을 알려줘",
        }
        for alias, question in alias_variants.items():
            if alias in alias_map:
                canonical = alias_map[alias]
                rows.append(make_example(question, [alias], is_ambiguous=False, canonical_company=canonical))

    # 중복 제거
    seen = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    return unique_rows


def main() -> None:
    df = pd.read_csv(UNIVERSE_PATH, dtype={"corp_code": str, "stock_code": str}, encoding="utf-8-sig")
    dataset = build_dataset(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"generated dataset rows: {len(dataset)}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
