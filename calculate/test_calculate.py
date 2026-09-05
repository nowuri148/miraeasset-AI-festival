import json

from calculate import calculate


def mock_llm(prompt: str) -> str:
    if "계산 실행 계획 생성기" in prompt:
        return json.dumps(
            {
                "answerable": True,
                "operation": "percent_change",
                "arguments": {
                    "old": 1000,
                    "new": 1250
                },
                "unit": "%",
                "source_ids": [
                    "S2024",
                    "S2025"
                ],
                "extracted_facts": [
                    {
                        "label": "2024년 설비투자",
                        "value": 1000,
                        "unit": "억원",
                        "source_id": "S2024"
                    },
                    {
                        "label": "2025년 설비투자",
                        "value": 1250,
                        "unit": "억원",
                        "source_id": "S2025"
                    }
                ],
                "reason": "2024년 대비 2025년 설비투자 증감률 계산"
            },
            ensure_ascii=False
        )

    return json.dumps(
        {
            "answer": "2025년 설비투자는 2024년 대비 25% 증가했습니다."
        },
        ensure_ascii=False
    )


chunks = [
    {
        "source_id": "S2024",
        "doc_id": "doc_2024",
        "company": "A사",
        "report_name": "2024년 사업보고서",
        "rcept_dt": "2025-03-20",
        "text": "2024년 설비투자 규모는 1,000억원입니다."
    },
    {
        "source_id": "S2025",
        "doc_id": "doc_2025",
        "company": "A사",
        "report_name": "2025년 사업보고서",
        "rcept_dt": "2026-03-20",
        "text": "2025년 설비투자 규모는 1,250억원입니다."
    }
]


result = calculate(
    question="A사의 2024년 대비 2025년 설비투자 증가율은?",
    hint={
        "task_type": "다중조회_비교연산",
        "companies": ["A사"],
        "periods": ["2024", "2025"],
        "metrics": ["설비투자"],
        "actions": ["증감"]
    },
    chunks=chunks,
    llm_callable=mock_llm
)

print(json.dumps(result, ensure_ascii=False, indent=2))