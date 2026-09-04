from validation.answer_grounding_validator import HCXGroundingValidator

validator = HCXGroundingValidator(debug=True)

result = validator.validate(
    question="SK텔레콤 2023년 1분기 무선통신사업 매출액은?",
    answer = (
    "SK텔레콤의 2023년 1분기 "
    "무선통신사업 매출액은 "
    "978,257백만원입니다."
),
    evidence = """
    SK텔레콤 분기보고서 (2023.03)

    구분 | 제40기 1분기 매출액 | 비중
    무선통신사업 | 3,249,918 | 74%
    유선통신사업 | 978,257 | 23%
    """,
    task_type="검색_정보추출",
)

print(result)