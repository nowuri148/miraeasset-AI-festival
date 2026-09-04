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


DEFAULT_TIMEOUT = 60


class HCXClaimExtractor:
    """
    최종 답변을 독립적으로 검증 가능한 claim으로 분해한다.
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


    def extract(
        self,
        *,
        question: str,
        answer: str,
    ) -> list[str]:

        messages = self._build_messages(
            question=question,
            answer=answer,
        )

        raw_text = self._call_hcx(
            messages
        )

        parsed = self._parse_json(
            raw_text
        )

        claims = (
            parsed.get("claims")
            or []
        )

        if not isinstance(claims, list):
            raise ValueError(
                "Claim Extractor의 claims가 list가 아닙니다."
            )

        claims = [
            str(claim).strip()
            for claim in claims
            if str(claim).strip()
        ]

        return claims


    def _build_messages(
        self,
        *,
        question: str,
        answer: str,
    ) -> list[dict[str, str]]:

        system_prompt = """
당신은 금융 공시 AI Agent의 Claim Extractor입니다.

당신의 역할은 FINAL ANSWER에 포함된 모든 검증 가능한 사실 주장을
빠짐없이 독립적인 claim으로 분해하는 것입니다.

중요 규칙:

1. 답변의 핵심 주장뿐 아니라 부가적인 사실 주장도 모두 추출하십시오.

2. 다음과 같은 표현도 독립적인 claim입니다.
- 증가했다
- 감소했다
- 크게 성장했다
- 가장 크다
- 더 높다
- 확정했다
- 검토 중이다
- 해지되었다
- 변경되었다

3. 숫자가 포함된 주장과 정성적 판단을 분리하십시오.

예:

FINAL ANSWER:
"매출액은 100억원이며 전년 대비 크게 성장했습니다."

claims:
[
  "매출액은 100억원이다.",
  "전년 대비 크게 성장했다."
]

4. 한 문장 안에 여러 사실이 있으면 반드시 여러 claim으로 나누십시오.

5. 표현만 바꾸지 말고 실제 검증 가능한 사실 단위로 분해하십시오.

6. 새로운 사실을 추가하거나 추론하지 마십시오.

7. 질문에 직접 답하는 핵심 claim뿐 아니라,
답변 안에 포함된 모든 외부 검증 가능한 사실을 추출하십시오.

8. 반드시 JSON만 출력하십시오.

출력 형식:

{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}
"""

        user_prompt = f"""
[QUESTION]
{question}

[FINAL ANSWER]
{answer}

FINAL ANSWER의 모든 검증 가능한 사실 주장을 빠짐없이 분해하십시오.
"""

        return [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]


    def _call_hcx(
        self,
        messages: list[dict[str, str]],
    ) -> str:

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 500,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        content = (
            data["choices"][0]
            ["message"]
            ["content"]
        )

        if self.debug:

            print(
                "[ClaimExtractor] HCX RAW:"
            )

            print(
                content
            )

        return str(content).strip()


    def _parse_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        try:
            return json.loads(text)

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            if not match:
                raise ValueError(
                    "Claim Extractor가 JSON을 반환하지 않았습니다."
                )

            return json.loads(
                match.group(0)
            )