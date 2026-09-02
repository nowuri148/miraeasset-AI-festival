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


class HCXClaimVerifier:
    """
    claim 하나가 evidence만으로 직접 입증되는지 검증한다.
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


    def verify(
        self,
        *,
        question: str,
        claim: str,
        evidence: str,
        task_type: str | None = None,
    ) -> dict[str, Any]:

        messages = self._build_messages(
            question=question,
            claim=claim,
            evidence=evidence,
            task_type=task_type,
        )

        raw_text = self._call_hcx(
            messages
        )

        parsed = self._parse_json(
            raw_text
        )

        return self._normalize_result(
            parsed,
            claim=claim,
        )


    def _build_messages(
        self,
        *,
        question: str,
        claim: str,
        evidence: str,
        task_type: str | None,
    ) -> list[dict[str, str]]:

        system_prompt = """
당신은 금융 공시 AI Agent의 Claim Verifier입니다.

당신의 역할은 단 하나의 CLAIM이 제공된 EVIDENCE만으로
직접 입증되는지 엄격하게 판단하는 것입니다.

반드시 다음 원칙을 따르십시오.

1. EVIDENCE에 존재하는 내용만 사용하십시오.

2. 외부 지식, 상식, 일반적인 기업 관행을 절대 사용하지 마십시오.

3. 다음과 같은 표현으로 근거 없는 claim을 정당화해서는 안 됩니다.
- 일반적으로
- 통상적으로
- 기업에서는 보통
- 상식적으로
- 추론할 수 있다
- 가능성이 있다

4. EVIDENCE에 CLAIM을 직접 지지하는 내용이 없다면
무조건 UNSUPPORTED입니다.

5. CLAIM이 숫자를 포함하면:
- 값
- 단위
- 대상
- 기간
- 지표
가 모두 일치해야 합니다.

6. 숫자가 EVIDENCE 어딘가에 존재한다는 이유만으로
SUPPORTED로 판단하지 마십시오.
그 숫자가 CLAIM이 말하는 항목과 연결되어 있어야 합니다.

예:

EVIDENCE:
무선통신사업 | 3,249,918
유선통신사업 | 978,257

CLAIM:
무선통신사업 매출액은 978,257이다.

→ UNSUPPORTED

7. CLAIM이 증가, 감소, 성장, 축소 등을 말한다면
그 방향성을 확인할 수 있는 비교 근거가 EVIDENCE에 있어야 합니다.

예:

EVIDENCE:
2023년 매출액 100억원

CLAIM:
전년 대비 크게 성장했다.

→ 전년도 비교값이 없으므로 UNSUPPORTED

8. CLAIM이 정정, 해지, 후속공시, 변경 이력을 말한다면
그 관계를 확인할 수 있는 관련 공시가 EVIDENCE에 있어야 합니다.

9. 근거가 불완전하거나 애매하면 UNSUPPORTED로 판단하십시오.

10. 새 답변을 생성하지 마십시오.

반드시 JSON 하나만 출력하십시오.

{
  "claim": "검증한 claim",
  "supported": true 또는 false,
  "reason": "판단 이유",
  "evidence": "claim을 직접 지지하는 evidence 요약. 없으면 빈 문자열"
}
"""

        user_prompt = f"""
[TASK TYPE]
{task_type or "unknown"}

[QUESTION]
{question}

[CLAIM]
{claim}

[EVIDENCE]
{evidence}

이 CLAIM이 EVIDENCE만으로 직접 입증되는지 검증하십시오.
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
            "max_tokens": 400,
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
                "[ClaimVerifier] HCX RAW:"
            )

            print(
                content
            )

        return str(content).strip()


    def _parse_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        text = str(
            text
            or ""
        ).strip()

        # ----------------------------------------------------
        # markdown code fence 제거
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 1. 정상 JSON 시도
        # ----------------------------------------------------

        try:
            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------
        # 2. JSON 객체 부분만 추출
        # ----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if match:

            candidate = (
                match.group(0)
            )

            try:
                return json.loads(
                    candidate
                )

            except json.JSONDecodeError:
                pass

        # ----------------------------------------------------
        # 3. 최소 필드 fallback 추출
        #
        # JSON이 깨졌더라도 supported 여부 정도는
        # 문자열에서 복구 시도
        # ----------------------------------------------------

        lower_text = (
            text.lower()
        )

        supported = None

        # false를 먼저 본다.
        # "unsupported" 안에 supported가 포함되므로
        # 단순 substring 검색은 피한다.
        if re.search(
            r'"supported"\s*:\s*false',
            lower_text,
        ):
            supported = False

        elif re.search(
            r'"supported"\s*:\s*true',
            lower_text,
        ):
            supported = True

        # ----------------------------------------------------
        # reason 복구
        # ----------------------------------------------------

        reason = ""

        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL,
        )

        if reason_match:
            reason = (
                reason_match.group(1)
                .strip()
            )

        # ----------------------------------------------------
        # evidence 복구
        # ----------------------------------------------------

        evidence = ""

        evidence_match = re.search(
            r'"evidence"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL,
        )

        if evidence_match:
            evidence = (
                evidence_match.group(1)
                .strip()
            )

        # ----------------------------------------------------
        # supported 값을 복구했다면 사용
        # ----------------------------------------------------

        if supported is not None:

            return {
                "supported": supported,
                "reason": (
                    reason
                    or (
                        "HCX 응답 JSON 형식이 일부 깨졌으나 "
                        "supported 판정값을 복구했습니다."
                    )
                ),
                "evidence": evidence,
            }

        # ----------------------------------------------------
        # 4. 아무것도 복구 못 했으면
        # 보수적으로 unsupported
        # ----------------------------------------------------

        if self.debug:

            print(
                "[ClaimVerifier] "
                "JSON parse failed. "
                "Fallback to unsupported."
            )

            print(
                "[ClaimVerifier] RAW:"
            )

            print(
                text
            )

        return {
            "supported": False,
            "reason": (
                "Claim Verifier의 응답을 "
                "정상적으로 파싱할 수 없어 "
                "보수적으로 근거 불충분으로 처리했습니다."
            ),
            "evidence": "",
        }


    def _normalize_result(
        self,
        data: dict[str, Any],
        *,
        claim: str,
    ) -> dict[str, Any]:

        raw_supported = (
            data.get(
                "supported",
                False,
            )
        )

        if isinstance(
            raw_supported,
            bool,
        ):
            supported = raw_supported

        elif isinstance(
            raw_supported,
            str,
        ):
            supported = (
                raw_supported
                .strip()
                .lower()
                == "true"
            )

        else:
            supported = bool(
                raw_supported
            )

        return {
            "claim": claim,
            "supported": supported,
            "reason": str(
                data.get(
                    "reason"
                )
                or ""
            ).strip(),
            "evidence": str(
                data.get(
                    "evidence"
                )
                or ""
            ).strip(),
        }