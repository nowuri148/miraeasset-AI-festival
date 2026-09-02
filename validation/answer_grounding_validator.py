from __future__ import annotations

import json
import re
from typing import Any

import requests

from Config import (
    CLOVA_STUDIO_API_KEY,
    CLOVA_BASE_URL,
    CLOVA_MODEL,
)

from validation.claim_extractor import (
    HCXClaimExtractor,
)

from validation.claim_verifier import (
    HCXClaimVerifier,
)

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_TIMEOUT = 60


# ============================================================
# VALIDATOR
# ============================================================

class HCXGroundingValidator:
    """
    최종 답변이 실제 검색 근거에 의해
    충분히 뒷받침되는지 검증한다.

    검증 순서
    ---------
    1. Deterministic numeric validation
    2. HCX semantic grounding validation

    주의
    ----
    - 답변을 새로 생성하지 않는다.
    - 검색 근거에 없는 사실을 추론하지 않는다.
    - PASS / RETRY 판단만 수행한다.
    """

    def __init__(
        self,
        *,
        api_key: str = CLOVA_STUDIO_API_KEY,
        base_url: str = CLOVA_BASE_URL,
        model_name: str = CLOVA_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> None:

        if not api_key:
            raise RuntimeError(
                "CLOVA_STUDIO_API_KEY가 설정되어 있지 않습니다."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.debug = debug

        self.claim_extractor = (
            HCXClaimExtractor(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                timeout=timeout,
                debug=debug,
            )
        )

        self.claim_verifier = (
            HCXClaimVerifier(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                timeout=timeout,
                debug=debug,
            )
)

    # ========================================================
    # PUBLIC
    # ========================================================

    def validate(
        self,
        *,
        question: str,
        answer: str,
        evidence: str,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """
        질문 + 답변 + 실제 근거를 받아
        Grounding 여부를 판정한다.

        반환 예:

        {
            "is_grounded": True,
            "verdict": "PASS",
            "reason": "...",
            "unsupported_claims": [],
            "retry_feedback": ""
        }
        """

        question = str(
            question
            or ""
        ).strip()

        answer = str(
            answer
            or ""
        ).strip()

        evidence = str(
            evidence
            or ""
        ).strip()

        # ----------------------------------------------------
        # ANSWER EMPTY
        # ----------------------------------------------------

        if not answer:

            return {
                "is_grounded": False,
                "verdict": "RETRY",
                "reason": (
                    "답변이 비어 있습니다."
                ),
                "unsupported_claims": [],
                "retry_feedback": (
                    "최종 답변이 생성되지 않았습니다."
                ),
            }

        # ----------------------------------------------------
        # EVIDENCE EMPTY
        # ----------------------------------------------------

        if not evidence:

            return {
                "is_grounded": False,
                "verdict": "RETRY",
                "reason": (
                    "답변을 검증할 검색 근거가 없습니다."
                ),
                "unsupported_claims": [],
                "retry_feedback": (
                    "답변의 근거 문서를 다시 검색해야 합니다."
                ),
            }

        # ====================================================
        # STEP 1.
        # DETERMINISTIC NUMERIC VALIDATION
        # ====================================================

        numeric_result = (
            self._validate_answer_numbers(
                answer=answer,
                evidence=evidence,
                task_type=task_type,
            )
        )

        if numeric_result is not None:

            if self.debug:

                print(
                    "[GroundingValidator] "
                    "Numeric validation failed:"
                )

                print(
                    numeric_result
                )

            return numeric_result

        # ====================================================
        # STEP 2.
        # CLAIM EXTRACTION
        # ====================================================

        claims = (
            self.claim_extractor.extract(
                question=question,
                answer=answer,
            )
        )

        if self.debug:

            print(
                "[GroundingValidator] Claims:"
            )

            for idx, claim in enumerate(
                claims,
                start=1,
            ):
                print(
                    f"  {idx}. {claim}"
                )


        # ====================================================
        # CLAIM이 하나도 없으면 보수적으로 RETRY
        # ====================================================

        if not claims:

            return {
                "is_grounded": False,
                "verdict": "RETRY",
                "reason": (
                    "최종 답변에서 검증 가능한 "
                    "사실 주장을 추출하지 못했습니다."
                ),
                "checked_claims": [],
                "unsupported_claims": [],
                "retry_feedback": (
                    "답변을 다시 생성하고 "
                    "근거가 명확한 형태로 작성하십시오."
                ),
            }


        # ====================================================
        # STEP 3.
        # CLAIM-BY-CLAIM VERIFICATION
        # ====================================================

        checked_claims: list[
            dict[str, Any]
        ] = []

        unsupported_claims: list[str] = []

        for claim in claims:

            result = (
                self.claim_verifier.verify(
                    question=question,
                    claim=claim,
                    evidence=evidence,
                    task_type=task_type,
                )
            )

            checked_claims.append(
                result
            )

            if not result.get(
                "supported",
                False,
            ):

                unsupported_claims.append(
                    claim
                )


        # ====================================================
        # STEP 4.
        # FINAL DECISION
        # ====================================================

        if unsupported_claims:

            return {
                "is_grounded": False,
                "verdict": "RETRY",
                "reason": (
                    "최종 답변의 일부 주장이 "
                    "공시 근거로 직접 입증되지 않습니다."
                ),
                "checked_claims": (
                    checked_claims
                ),
                "unsupported_claims": (
                    unsupported_claims
                ),
                "retry_feedback": (
                    "근거에서 직접 확인되지 않는 "
                    "주장을 제거하거나, 해당 주장을 "
                    "입증할 수 있는 공시 근거를 다시 검색하십시오."
                ),
            }


        return {
            "is_grounded": True,
            "verdict": "PASS",
            "reason": (
                "최종 답변의 모든 검증 가능한 "
                "주장이 공시 근거로 확인되었습니다."
            ),
            "checked_claims": (
                checked_claims
            ),
            "unsupported_claims": [],
            "retry_feedback": "",
        }


    # ========================================================
    # NUMERIC VALIDATION
    # ========================================================

    def _validate_answer_numbers(
        self,
        *,
        answer: str,
        evidence: str,
        task_type: str | None,
    ) -> dict[str, Any] | None:
        """
        답변에 포함된 핵심 수치가
        실제 Evidence에 존재하는지 deterministic하게 검증한다.

        Answer:
        - 금액
        - 비율
        - 주식 수
        등 핵심 수치만 추출한다.

        Evidence:
        - 표 안의 bare number를 포함해
          모든 숫자를 추출한다.

        다중조회_비교연산은 계산 결과가 Evidence에
        직접 존재하지 않을 수 있으므로 현재 exact check를 생략한다.
        """

        # ----------------------------------------------------
        # 비교/연산 Task는 별도 Calculation Validator 필요
        # ----------------------------------------------------

        if task_type == "다중조회_비교연산":
            return None

        # ----------------------------------------------------
        # Answer에서 검증 대상 숫자 추출
        # ----------------------------------------------------

        answer_numbers = (
            self._extract_answer_numeric_tokens(
                answer
            )
        )

        # 검증 대상 숫자가 없으면
        # HCX semantic validation으로 진행
        if not answer_numbers:
            return None

        # ----------------------------------------------------
        # Evidence에서는 모든 숫자 추출
        # ----------------------------------------------------

        evidence_numbers = (
            self._extract_evidence_numeric_tokens(
                evidence
            )
        )

        normalized_evidence = {
            self._normalize_numeric_token(
                number
            )
            for number in evidence_numbers
        }

        unsupported: list[str] = []

        for number in answer_numbers:

            normalized = (
                self._normalize_numeric_token(
                    number
                )
            )

            if (
                normalized
                not in normalized_evidence
            ):

                unsupported.append(
                    number
                )

        # ----------------------------------------------------
        # 모든 숫자가 Evidence에 존재
        # ----------------------------------------------------

        if not unsupported:
            return None

        # ----------------------------------------------------
        # 숫자 불일치
        # ----------------------------------------------------

        return {
            "is_grounded": False,
            "verdict": "RETRY",
            "reason": (
                "답변에 포함된 수치 중 "
                "검색 근거에서 직접 확인되지 않는 "
                "수치가 있습니다: "
                + ", ".join(
                    unsupported
                )
            ),
            "unsupported_claims": [
                (
                    "근거에서 확인되지 않는 수치: "
                    f"{number}"
                )
                for number in unsupported
            ],
            "retry_feedback": (
                "답변에 사용한 수치를 원문 공시 근거에서 "
                "다시 확인하고 정확한 값으로 답변하십시오."
            ),
        }


    # ========================================================
    # ANSWER NUMBER EXTRACTION
    # ========================================================

    def _extract_answer_numeric_tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        답변에서 deterministic 검증이 필요한
        핵심 숫자만 추출한다.

        포함:
        - 금액
        - 주식 수
        - %
        - 퍼센트

        제외:
        - 2023년
        - 1분기
        - 3월
        - 날짜
        등의 기간 숫자

        예:
        3,249,918백만원 -> 3,249,918
        74%            -> 74%
        120억원         -> 120

        2023년          -> 제외
        1분기           -> 제외
        """

        text = str(
            text
            or ""
        )

        tokens: list[str] = []

        # ----------------------------------------------------
        # 1. 비율
        # ----------------------------------------------------

        percent_pattern = (
            r"(?<!\d)"
            r"("
            r"(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")"
            r"(?:\.\d+)?"
            r")"
            r"\s*"
            r"(?:%|퍼센트)"
        )

        for match in re.finditer(
            percent_pattern,
            text,
        ):

            value = (
                match.group(1)
                + "%"
            )

            tokens.append(
                value
            )

        # ----------------------------------------------------
        # 2. 금액 / 주식수
        #
        # 긴 단위부터 앞에 둔다.
        # ----------------------------------------------------

        financial_pattern = (
            r"(?<!\d)"
            r"("
            r"(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")"
            r"(?:\.\d+)?"
            r")"
            r"\s*"
            r"(?:"
            r"백만원"
            r"|천만원"
            r"|억원"
            r"|조원"
            r"|천원"
            r"|만원"
            r"|백만주"
            r"|천주"
            r"|원"
            r"|주"
            r")"
        )

        for match in re.finditer(
            financial_pattern,
            text,
        ):

            tokens.append(
                match.group(1)
            )

        # ----------------------------------------------------
        # 중복 제거
        # ----------------------------------------------------

        return (
            self._deduplicate_numeric_tokens(
                tokens
            )
        )


    # ========================================================
    # EVIDENCE NUMBER EXTRACTION
    # ========================================================

    def _extract_evidence_numeric_tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        Evidence에서는 단위 유무와 관계없이
        모든 숫자를 추출한다.

        예:

        무선통신사업 | 3,249,918 | 74%

        →
        [
            "3,249,918",
            "74%"
        ]

        Evidence에서 2023, 1 등의 기간 숫자도
        추출될 수 있지만 문제되지 않는다.

        Answer 쪽에서는 검증 대상 핵심 숫자만
        추출하기 때문이다.
        """

        text = str(
            text
            or ""
        )

        pattern = (
            r"(?<!\d)"
            r"(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")"
            r"(?:\.\d+)?"
            r"%?"
        )

        tokens = (
            re.findall(
                pattern,
                text,
            )
        )

        return (
            self._deduplicate_numeric_tokens(
                tokens
            )
        )


    # ========================================================
    # NUMBER NORMALIZATION
    # ========================================================

    def _normalize_numeric_token(
        self,
        value: str,
    ) -> str:
        """
        숫자 비교용 정규화.

        3,249,918 -> 3249918
        74%       -> 74%
        """

        return (
            str(value)
            .replace(",", "")
            .replace(" ", "")
            .strip()
        )


    # ========================================================
    # NUMBER DEDUPLICATION
    # ========================================================

    def _deduplicate_numeric_tokens(
        self,
        tokens: list[str],
    ) -> list[str]:
        """
        동일 숫자가 여러 번 등장하는 경우
        중복 제거.
        """

        unique_tokens: list[str] = []
        seen: set[str] = set()

        for token in tokens:

            normalized = (
                self._normalize_numeric_token(
                    token
                )
            )

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            unique_tokens.append(
                token
            )

        return unique_tokens


    # ========================================================
    # PROMPT
    # ========================================================

    def _build_messages(
        self,
        *,
        question: str,
        answer: str,
        evidence: str,
        task_type: str | None,
    ) -> list[dict[str, str]]:

        system_prompt = """
당신은 금융 공시 AI Agent의 최종 근거 검증기입니다.

당신의 역할은 답변을 새로 작성하는 것이 아니라,
주어진 최종 답변이 제공된 공시 근거(EVIDENCE)에 의해
충분히 뒷받침되는지를 엄격하게 판단하는 것입니다.

반드시 다음 기준으로 검증하십시오.

[검증 원칙]

1. EVIDENCE에 명시적으로 존재하는 정보만 근거로 인정합니다.

2. 답변의 핵심 수치, 회사, 기간, 공시 유형, 사실관계가
   EVIDENCE와 일치해야 합니다.

3. 숫자 또는 금액이 비슷하다는 이유로
   일치한다고 판단하지 마십시오.

예:
근거: 3,249,918
답변: 3,249,981
→ RETRY

수치의 근사치, 반올림, 오차 범위를
임의로 허용하지 마십시오.

4. 수치 질의에서는 다음을 모두 검증합니다.
   - 값
   - 단위
   - 대상 회사
   - 대상 기간
   - 연결/별도 등 범위
   - 해당 지표의 의미

5. 비교/연산 질의에서는 다음을 검증합니다.
   - 비교 대상 각각의 원본 값
   - 계산 결과
   - 비교 결론
   - 계산 방향

6. 변경 이력/복합 문서 추론에서는 다음을 검증합니다.
   - 관련 공시가 모두 근거에 포함되어 있는지
   - 사건의 선후관계
   - 정정/해지/후속 관계
   - 최종 결론

7. FINAL ANSWER를 먼저 독립적인 사실 주장(claim) 단위로 분해하십시오.

예:
"매출액은 100억원이며 전년 대비 크게 성장했습니다."

이 답변에는 최소 두 개의 독립적인 주장이 있습니다.
- 주장 1: 매출액은 100억원이다.
- 주장 2: 전년 대비 크게 성장했다.

8. 분해한 모든 실질적 주장 각각에 대해
EVIDENCE에서 직접적인 근거를 찾아야 합니다.

9. 하나의 주장만 근거가 있어도
나머지 주장에 근거가 없다면 반드시 RETRY입니다.

10. 질문에 대한 핵심 답변이 맞더라도,
FINAL ANSWER에 추가된 부가 설명이 EVIDENCE로
입증되지 않으면 PASS해서는 안 됩니다.

예:

EVIDENCE:
무선통신사업 매출액 | 3,249,918

FINAL ANSWER:
무선통신사업 매출액은 3,249,918백만원이며,
전년 동기 대비 크게 성장했습니다.

판정:
- 매출액 3,249,918백만원 → 근거 있음
- 전년 동기 대비 크게 성장 → 근거 없음
→ RETRY

11. 출처 정보 자체가 잘못된 경우에도 RETRY입니다.

12. 단순 표현 차이, 조사, 문장 순서 차이는
    오류로 보지 않습니다.

13. 답변이 근거보다 더 강한 결론을 주장하면 RETRY입니다.

예:
근거:
"투자를 검토하고 있다"

답변:
"투자를 확정했다"

→ RETRY

14. 답변의 핵심 주장 전체가
    근거에 의해 충분히 입증되면 PASS입니다.

15. 숫자가 EVIDENCE 어딘가에 존재한다는 이유만으로
    해당 답변이 맞다고 판단하지 마십시오.

숫자가 질문에서 요구한 항목,
기업, 기간, 지표와 정확히 연결되는지도 검증하십시오.

예:
EVIDENCE:
무선통신사업 | 3,249,918
유선통신사업 | 978,257

QUESTION:
무선통신사업 매출액은?

ANSWER:
978,257

→ 숫자 978,257은 EVIDENCE에 존재하지만
유선통신사업 값이므로 RETRY입니다.

16. 검증 결과 외에는 새로운 답변을 만들지 마십시오.


[출력 형식]

반드시 아래 JSON 하나만 출력하십시오.

{
  "is_grounded": true 또는 false,
  "verdict": "PASS" 또는 "RETRY",
  "reason": "판단 이유",
  "checked_claims": [
    {
      "claim": "검증한 주장",
      "supported": true 또는 false,
      "evidence": "해당 주장에 대응하는 근거 또는 없음"
    }
  ],
  "unsupported_claims": [
    "근거가 없는 주장"
  ],
  "retry_feedback": "재실행 시 확인해야 할 사항"
}

PASS인 경우:
- unsupported_claims = []
- retry_feedback = ""

RETRY인 경우:
- 왜 실패했는지 구체적으로 작성
- 어떤 정보 또는 근거를 다시 검색해야 하는지
  retry_feedback에 작성
"""

        user_prompt = f"""
[TASK TYPE]
{task_type or "unknown"}

[QUESTION]
{question}

[FINAL ANSWER]
{answer}

[EVIDENCE]
{evidence}

위 답변이 EVIDENCE로 충분히 입증되는지 검증하십시오.
"""

        return [
            {
                "role": "system",
                "content": (
                    system_prompt.strip()
                ),
            },
            {
                "role": "user",
                "content": (
                    user_prompt.strip()
                ),
            },
        ]


    # ========================================================
    # HCX
    # ========================================================

    def _call_hcx(
        self,
        messages: list[
            dict[str, str]
        ],
    ) -> str:

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        payload = {
            "model": (
                self.model_name
            ),
            "messages": (
                messages
            ),
            "temperature": 0.0,
            "max_tokens": 700,
        }

        response = (
            requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        )

        response.raise_for_status()

        data = (
            response.json()
        )

        content = (
            data["choices"][0]
            ["message"]
            ["content"]
        )

        if self.debug:

            print(
                "[GroundingValidator] "
                "HCX RAW:"
            )

            print(
                content
            )

        return str(
            content
        ).strip()


    # ========================================================
    # JSON PARSER
    # ========================================================

    def _parse_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        text = (
            text.strip()
        )

        # ----------------------------------------------------
        # ```json ... ``` 제거
        # ----------------------------------------------------

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=(
                re.IGNORECASE
            ),
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        # ----------------------------------------------------
        # 정상 JSON
        # ----------------------------------------------------

        try:

            return (
                json.loads(
                    text
                )
            )

        except (
            json.JSONDecodeError
        ):

            pass

        # ----------------------------------------------------
        # JSON 객체 부분 복구
        # ----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            text,
            flags=(
                re.DOTALL
            ),
        )

        if not match:

            raise ValueError(
                "Validator가 JSON을 "
                "반환하지 않았습니다.\n"
                f"RAW:\n{text}"
            )

        try:

            return (
                json.loads(
                    match.group(0)
                )
            )

        except (
            json.JSONDecodeError
        ) as exc:

            raise ValueError(
                "Validator JSON 파싱에 "
                "실패했습니다.\n"
                f"RAW:\n{text}"
            ) from exc


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_result(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # is_grounded
        # ----------------------------------------------------

        raw_grounded = (
            data.get(
                "is_grounded",
                False,
            )
        )

        if isinstance(
            raw_grounded,
            bool,
        ):

            is_grounded = (
                raw_grounded
            )

        elif isinstance(
            raw_grounded,
            str,
        ):

            is_grounded = (
                raw_grounded
                .strip()
                .lower()
                == "true"
            )

        else:

            is_grounded = bool(
                raw_grounded
            )

        # ----------------------------------------------------
        # verdict
        # ----------------------------------------------------

        verdict = str(
            data.get(
                "verdict"
            )
            or (
                "PASS"
                if is_grounded
                else "RETRY"
            )
        ).strip().upper()

        # ----------------------------------------------------
        # 일관성 강제
        # ----------------------------------------------------

        if verdict == "PASS":

            is_grounded = True

        elif verdict == "RETRY":

            is_grounded = False

        else:

            verdict = (
                "PASS"
                if is_grounded
                else "RETRY"
            )

        # ----------------------------------------------------
        # unsupported claims
        # ----------------------------------------------------

        unsupported_claims = (
            data.get(
                "unsupported_claims"
            )
            or []
        )

        if not isinstance(
            unsupported_claims,
            list,
        ):

            unsupported_claims = [
                str(
                    unsupported_claims
                )
            ]

        unsupported_claims = [
            str(item).strip()
            for item
            in unsupported_claims
            if str(item).strip()
        ]

        # ----------------------------------------------------
        # PASS인데 unsupported가 존재하면
        # 보수적으로 RETRY
        # ----------------------------------------------------

        if (
            verdict == "PASS"
            and unsupported_claims
        ):

            verdict = "RETRY"
            is_grounded = False

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "is_grounded": (
                is_grounded
            ),
            "verdict": (
                verdict
            ),
            "reason": str(
                data.get(
                    "reason"
                )
                or ""
            ).strip(),
            "unsupported_claims": (
                unsupported_claims
            ),
            "retry_feedback": str(
                data.get(
                    "retry_feedback"
                )
                or ""
            ).strip(),
        }