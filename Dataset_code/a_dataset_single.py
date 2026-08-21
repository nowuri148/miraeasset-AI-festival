from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import CLOVA_STUDIO_API_KEY  # noqa: E402
from z_base_function import (
    ClovaClient,
    ContextBundle,
    build_rows,
    load_input_records,
    make_bundles,
    normalize_qas_container,
    response_schema,
    training_completion,
    training_text,
    validate_reviewed_qas_minimal,
)

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

TASK_GUIDE_SINGLE = """
- 정보추출: 사람/회사/금액/날짜/지역/계약명/목적/사유
- 조건 결합: 두 개 이상의 필드를 함께 확인하는 질문
- 계산: 기간 일수, 비율 검산, 금액 단위 변환 등 Context로 계산 가능한 질문
- 요약: 핵심 내용 또는 투자자가 확인할 조건을 선별
- 근거 부족: Context에 없는 정보를 요구하고 확인 불가라고 답하는 질문
- 표현 다양성: 존댓말, 구어체, 짧은 검색어형, 정식 질의형을 고르게 사용
"""

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

def generation_prompt(bundle: ContextBundle, count: int) -> str:
    guide = TASK_GUIDE_SINGLE
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

def main() -> None:
    """원래 evidence-repair 코드 그대로 단일 문서만 테스트한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    TEST_INPUT_PATH = Path(
        r"C:\mirae\miraeasset-AI-festival\corpus\raw\exchange\HD현대일렉트릭\20230131800162\20230131800162.xml"
    )

    QUESTIONS_PER_DOCUMENT = 4
    MAX_CONTEXT_CHARS = 5500
    MAX_GENERATION_TOKENS = 4096
    GENERATION_TEMPERATURE = 0.65
    API_TIMEOUT_SECONDS = 120
    API_RETRIES = 3

    # 원래 evidence-repair 코드의 기본 cache 경로 그대로 사용.
    CACHE_PATH = Path(
        r"C:\mirae\miraeasset-AI-festival\Dataset\.qa_generation_cache.jsonl"
    )

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

        # single 문서용 로직을 공통 build_rows에 주입
        generation_prompt_fn=generation_prompt,
        review_prompt_fn=review_prompt,
        normalize_qas_fn=normalize_qas_container,
        validate_reviewed_fn=validate_reviewed_qas_minimal,
        training_text_fn=training_text,
        training_completion_fn=training_completion,

        # HCX system prompt / response schema
        generator_system_prompt=GENERATOR_SYSTEM_PROMPT,
        reviewer_system_prompt=REVIEWER_SYSTEM_PROMPT,
        response_schema=response_schema(),

        # 기존 single 파이프라인 cache namespace 유지
        generator_cache_version="GENERATOR_V4_EVIDENCE_REPAIR",
        reviewer_cache_version="REVIEWER_V4_EVIDENCE_REPAIR",
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