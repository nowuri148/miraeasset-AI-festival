# Company ambiguity training dataset

이 문서는 `universe.csv`를 기반으로, 회사명 모호성 판별과 후보 생성, 재질문 생성용 학습 데이터를 만드는 흐름을 설명합니다.

## 생성 파일

- `corpus/company_disambiguation_train.jsonl`
- `question/company_disambiguation_dataset.py`

## 입력 형식

```json
{
  "instruction": "질문에서 회사명이 모호하면 후보를 생성하고 재질문을 하십시오. 명확하면 canonical 회사명을 반환하십시오.",
  "input": {
    "question": "두산 2025년 매출은 얼마인가?",
    "task": "company_ambiguity_resolution",
    "candidate_companies": ["두산로보틱스", "두산에너빌리티", "두산퓨얼셀", "두산 그룹"]
  },
  "output": {
    "is_ambiguous": true,
    "candidate_companies": ["두산로보틱스", "두산에너빌리티", "두산퓨얼셀", "두산 그룹"],
    "canonical_company": null,
    "clarification_question": "질문이 모호합니다. '두산 2025년 매출은 얼마인가?'에서 어떤 회사를 의미하나요? 다음 후보 중 하나를 선택해 주세요: 두산로보틱스, 두산에너빌리티, 두산퓨얼셀, 두산 그룹"
  }
}
```

## 학습 목적

1. 모호성 판별: 질문에 회사가 언급되었는지, 그리고 회사명이 불명확한지 여부
2. 후보 생성: 올바른 회사 후보 목록을 뽑아내는 능력
3. 재질문 생성: 사용자가 다시 선택할 수 있도록 자연스러운 질문 문장 생성

## 재생성

```bash
python question/company_disambiguation_dataset.py
```

이 스크립트는 `corpus/universe.csv`를 읽어서 모호한 계열사 그룹과 정확한 회사 예시를 자동으로 생성합니다.
