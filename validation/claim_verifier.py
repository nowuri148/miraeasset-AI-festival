from __future__ import annotations

import json
import re
from typing import Any, Iterable

import requests

from Config import (
    CLOVA_STUDIO_API_KEY,
    CLOVA_BASE_URL,
    CLOVA_MODEL,
)


DEFAULT_TIMEOUT = 60


class HCXClaimVerifier:
    """
    Claim 하나가 제공된 Evidence만으로
    직접적이고 완전하게 입증되는지 검증한다.

    검증 구조
    --------
    1. Python deterministic topic coverage check
    2. HCX semantic claim verification

    핵심 원칙
    --------
    - 외부 지식 사용 금지
    - 숫자 존재만으로 PASS 금지
    - 질문 표현과 공시 표현의 대응 관계 검증
    - multi-hop 관계가 필요한 경우
      필요한 연결 근거가 Evidence 안에 있어야 함
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
        self.timeout = int(timeout)
        self.debug = bool(debug)

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def verify(
        self,
        *,
        question: str,
        claim: str,
        evidence: str,
        task_type: str | None = None,
        topic_keywords: Iterable[str] | None = None,
        metrics: Iterable[str] | None = None,
        target_companies: Iterable[str] | None = None,
    ) -> dict[str, Any]:

        question = str(
            question or ""
        ).strip()

        claim = str(
            claim or ""
        ).strip()

        evidence = str(
            evidence or ""
        ).strip()

        # ---------------------------------------------------------------------
        # 1. Deterministic topic coverage
        # ---------------------------------------------------------------------

        deterministic_result = (
            self._validate_topic_coverage(
                evidence=evidence,
                topic_keywords=topic_keywords,
                metrics=metrics,
                target_companies=target_companies,
            )
        )

        if deterministic_result is not None:

            return {
                "claim": claim,
                **deterministic_result,
            }

        # ---------------------------------------------------------------------
        # 2. HCX semantic verification
        # ---------------------------------------------------------------------

        messages = (
            self._build_messages(
                question=question,
                claim=claim,
                evidence=evidence,
                task_type=task_type,
            )
        )

        raw_text = (
            self._call_hcx(
                messages
            )
        )

        parsed = (
            self._parse_json(
                raw_text
            )
        )

        return (
            self._normalize_result(
                parsed,
                claim=claim,
            )
        )

    # =========================================================================
    # DETERMINISTIC TOPIC COVERAGE
    # =========================================================================

    def _validate_topic_coverage(
        self,
        *,
        evidence: str,
        topic_keywords: Iterable[str] | None,
        metrics: Iterable[str] | None,
        target_companies: Iterable[str] | None,
    ) -> dict[str, Any] | None:
        """
        Keyword Extractor가 추출한 topic_keywords 중
        실제 질문의 세부 대상을 나타내는 표현이
        Evidence 전체에서 완전히 사라졌는지 확인한다.

        목적
        ----
        예:
        topic = 반도체

        Evidence:
        DS 부문 | 483,723

        → "반도체"라는 대상 자체가 근거에 없음
        → HCX가 DS=반도체를 외부 지식으로 추론하기 전에 차단

        반대로 Evidence pool 어딘가에
        "반도체"라는 표현과 DS 관련 근거가 있으면
        semantic verifier 단계로 넘긴다.

        이 단계에서는 최종 관계가 맞는지를 판단하지 않는다.
        명백한 evidence coverage 부족만 차단한다.
        """

        evidence_text = (
            self._normalize_text(
                evidence
            )
        )

        if not evidence_text:

            return {
                "supported": False,
                "reason": (
                    "검증할 EVIDENCE가 없습니다."
                ),
                "evidence": "",
            }

        scope_terms = (
            self._extract_scope_terms(
                topic_keywords=topic_keywords,
                metrics=metrics,
                target_companies=target_companies,
            )
        )

        # topic이 없거나
        # 의미 있는 scope term을 추출하지 못했다면
        # semantic verifier로 넘긴다.
        if not scope_terms:
            return None

        covered_terms = [
            term
            for term in scope_terms
            if (
                self._normalize_text(term)
                in evidence_text
            )
        ]

        # 하나 이상의 핵심 topic이 Evidence에 있다면
        # 관계 자체의 적절성은 HCX verifier가 판단
        if covered_terms:
            return None

        return {
            "supported": False,

            "reason": (
                "질문의 핵심 대상 표현이 "
                "검증 Evidence에서 확인되지 않습니다. "
                f"확인 대상: {scope_terms}. "
                "다른 사업부문명이나 표의 행명을 "
                "외부 지식으로 질문 대상과 연결할 수 없으므로 "
                "근거 불충분으로 판정합니다."
            ),

            "evidence": "",
        }

    # =========================================================================
    # SCOPE TERM EXTRACTION
    # =========================================================================

    def _extract_scope_terms(
        self,
        *,
        topic_keywords: Iterable[str] | None,
        metrics: Iterable[str] | None,
        target_companies: Iterable[str] | None,
    ) -> list[str]:

        topics = self._normalize_string_list(
            topic_keywords
        )

        if not topics:
            return []

        metric_terms = {
            self._normalize_text(term)
            for term in self._normalize_string_list(
                metrics
            )
        }

        company_terms = {
            self._normalize_text(term)
            for term in self._normalize_string_list(
                target_companies
            )
        }

        # -----------------------------------------------------
        # 질문의 세부 대상이 아닌 일반 표현
        # -----------------------------------------------------

        generic_terms = {
            "금액",
            "금액은",
            "얼마",
            "얼마야",
            "수치",
            "규모",
            "현황",
            "실적",
            "변화",
            "비교",
            "증가",
            "감소",
            "성장",
            "하락",
            "상승",
            "분석",
            "설비투자",
            "시설투자",
            "투자",
            "투자액",
            "매출",
            "매출액",
            "영업이익",
            "당기순이익",
            "순이익",
            "재고자산",
            "자산",
            "부채",
            "계약금액",
            "계약",
            "비율",
        }

        tokens: list[str] = []

        for topic in topics:

            # 공백/문장부호 단위로 분리
            parts = re.findall(
                r"[가-힣A-Za-z][가-힣A-Za-z0-9_-]*",
                str(topic),
            )

            for part in parts:

                part = (
                    part.strip()
                )

                if len(part) < 2:
                    continue

                normalized = (
                    self._normalize_text(
                        part
                    )
                )

                # 회사명 제거
                if any(
                    (
                        normalized == company
                        or normalized in company
                        or company in normalized
                    )
                    for company in company_terms
                    if company
                ):
                    continue

                # metric 제거
                if any(
                    (
                        normalized == metric
                        or normalized in metric
                        or metric in normalized
                    )
                    for metric in metric_terms
                    if metric
                ):
                    continue

                # 연도/기간 표현 제거
                if re.fullmatch(
                    r"\d{4}년?",
                    normalized,
                ):
                    continue

                if re.fullmatch(
                    r"\d+(?:분기|월|년|반기)",
                    normalized,
                ):
                    continue

                if normalized in generic_terms:
                    continue

                tokens.append(
                    part
                )

        return list(
            dict.fromkeys(
                tokens
            )
        )

    # =========================================================================
    # PROMPT
    # =========================================================================

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

당신의 역할은 하나의 CLAIM이 제공된 EVIDENCE만으로
직접적이고 완전하게 입증되는지 엄격하게 검증하는 것입니다.

새 답변을 생성하지 마십시오.


============================================================
절대 원칙
============================================================

1. EVIDENCE 안의 정보만 사용하십시오.

2. 외부 지식, 사전학습 지식, 상식, 기업에 대해 이미 알고 있는 정보,
업계 관행을 절대 사용하지 마십시오.

3. QUESTION이나 CLAIM에 있다고 해서
그 내용을 사실이라고 가정해서는 안 됩니다.

4. 다음과 같은 방식으로 근거를 보완해서는 안 됩니다.

- 일반적으로 그렇다
- 통상적으로 그렇다
- 잘 알려져 있다
- 해당 회사는 원래 이런 사업을 한다
- 이 약어는 보통 이것을 뜻한다
- 암시하고 있다
- 유추할 수 있다
- 문맥상 그렇다고 볼 수 있다
- 가능성이 높다

5. 필요한 근거 관계가 하나라도 빠져 있으면
UNSUPPORTED입니다.

6. 애매하거나 여러 해석이 가능하면
UNSUPPORTED입니다.


============================================================
숫자 검증
============================================================

7. CLAIM이 숫자를 포함하는 경우 다음 항목을 모두 확인하십시오.

- 회사
- 기간
- 대상
- 사업부문 또는 제품군
- 지표
- 값
- 단위

8. 숫자가 EVIDENCE 어딘가에 존재한다는 이유만으로
SUPPORTED로 판정하지 마십시오.

그 숫자가 CLAIM이 말하는 정확한 대상에 연결되어 있어야 합니다.

예:

EVIDENCE:
무선통신사업 | 3,249,918
유선통신사업 | 978,257

CLAIM:
무선통신사업 매출액은 978,257이다.

→ UNSUPPORTED


============================================================
질문 표현과 공시 표현의 연결
============================================================

9. QUESTION 또는 CLAIM의 핵심 표현과
공시 표의 행명 또는 사업부문 표현이 다르다면
두 표현의 대응 관계를 EVIDENCE 안에서 확인해야 합니다.

10. 대응 관계가 없다면
외부 지식으로 연결해서는 안 됩니다.

예:

QUESTION:
반도체 설비투자액은?

EVIDENCE:
DS 부문 | 투자액 483,723

여기에는 "반도체"와 "DS 부문"의 관계가 없습니다.

→ UNSUPPORTED


11. 다음과 같이 여러 EVIDENCE를 연결하는 것은 허용됩니다.

EVIDENCE A:
DRAM, NAND Flash, 모바일AP 등은 반도체 부품이다.

EVIDENCE B:
DS 부문의 주요 제품은 DRAM, NAND Flash, 모바일AP 등이다.

EVIDENCE C:
DS 부문 투자액은 483,723이다.

이 경우 EVIDENCE 내부에서 다음 관계를 확인할 수 있습니다.

반도체
→ DRAM / NAND / 모바일AP
→ DS 부문
→ 투자액 483,723

따라서 CLAIM이 이를 정확히 표현한다면
SUPPORTED로 판단할 수 있습니다.


12. 중요한 것은 각 연결 단계가
EVIDENCE 자체에서 확인 가능해야 한다는 점입니다.

모델의 기존 지식을 연결 단계에 사용해서는 안 됩니다.


============================================================
표 범위 검증
============================================================

13. 표의 숫자를 검증할 때는
숫자가 속한 행과 열을 함께 확인하십시오.

14. 특정 하위 항목을 묻는데
"합계", "총계", "전체" 값을 사용했다면
UNSUPPORTED입니다.

예:

DS 부문 | 483,723
SDC | 23,856
기타 | 23,560
합계 | 531,139

CLAIM:
DS 부문의 투자액은 531,139이다.

→ UNSUPPORTED

15. 반대로 전체 값을 묻는데
특정 하위 항목 하나의 값만 사용해도 UNSUPPORTED입니다.


============================================================
비교 및 변화
============================================================

16. 증가, 감소, 성장, 축소, 개선 등의 표현은
그 방향을 확인할 비교값이 EVIDENCE에 있어야 합니다.

17. 비교값이 하나뿐이라면
전년 대비 증가 또는 감소를 주장할 수 없습니다.


============================================================
계산
============================================================

18. 비율, 합계, 차이, 증감률 등 계산 결과는
계산에 필요한 입력값들이 EVIDENCE에 모두 있어야 합니다.

19. task_type이 다중조회_비교연산인 경우
최종 계산 결과가 원문에 직접 존재하지 않아도 됩니다.

단, 계산에 사용한 원본 값들은 모두 EVIDENCE에 존재해야 합니다.


============================================================
정정 / 후속 공시
============================================================

20. 정정, 변경, 해지, 후속공시 관계를 주장한다면
그 관계를 보여주는 관련 문서 근거가 EVIDENCE에 있어야 합니다.

21. 문서 제목이나 접수번호가 비슷하다는 이유만으로
관계를 추정하지 마십시오.


============================================================
판정
============================================================

다음 경우는 반드시 UNSUPPORTED입니다.

- 값만 있고 대상이 불명확함
- 회사가 다름
- 기간이 다름
- 지표가 다름
- 질문 표현과 표 행명이 다른데 연결 근거가 없음
- 합계를 하위항목 값으로 사용함
- 비교 표현인데 비교값이 없음
- 외부 지식이 있어야만 관계를 만들 수 있음
- 근거가 애매함
- 근거가 불완전함

근거가 충분하지 않다면
통과시키는 것보다 UNSUPPORTED로 판정하십시오.


============================================================
출력
============================================================

반드시 JSON object 하나만 출력하십시오.

Markdown 코드블록은 사용하지 마십시오.

{
  "claim": "검증한 claim",
  "supported": true 또는 false,
  "reason": "판단 이유",
  "evidence": "직접 사용한 evidence 요약. 없으면 빈 문자열"
}
""".strip()

        user_prompt = f"""
[TASK TYPE]
{task_type or "unknown"}

[QUESTION]
{question}

[CLAIM]
{claim}

[EVIDENCE]
{evidence}

위 CLAIM이 EVIDENCE만으로
직접적이고 완전하게 입증되는지 검증하십시오.

특히 질문의 대상과 표의 사업부문/행명이 다르면
그 두 표현 사이의 모든 연결 단계가
EVIDENCE에 실제로 존재하는지 확인하십시오.

외부 지식으로 연결하지 마십시오.
""".strip()

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    # =========================================================================
    # HCX CALL
    # =========================================================================

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
            "max_tokens": 500,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = (
            response.json()
        )

        try:

            content = (
                data["choices"][0]
                ["message"]
                ["content"]
            )

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:

            raise RuntimeError(
                "HCX Claim Verifier 응답에서 "
                "message.content를 찾지 못했습니다."
            ) from exc

        if self.debug:

            print(
                "[ClaimVerifier] HCX RAW:"
            )

            print(
                content
            )

        return str(
            content
        ).strip()

    # =========================================================================
    # JSON PARSER
    # =========================================================================

    def _parse_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        text = str(
            text or ""
        ).strip()

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
        ).strip()

        # ---------------------------------------------------------------------
        # 1. 정상 JSON
        # ---------------------------------------------------------------------

        try:

            value = (
                json.loads(
                    text
                )
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except json.JSONDecodeError:
            pass

        # ---------------------------------------------------------------------
        # 2. 객체 부분 추출
        # ---------------------------------------------------------------------

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

                value = (
                    json.loads(
                        candidate
                    )
                )

                if isinstance(
                    value,
                    dict,
                ):
                    return value

            except json.JSONDecodeError:
                pass

        # ---------------------------------------------------------------------
        # 3. supported fallback
        # ---------------------------------------------------------------------

        lower_text = (
            text.lower()
        )

        supported = None

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

        reason = ""

        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL,
        )

        if reason_match:

            reason = (
                reason_match
                .group(1)
                .strip()
            )

        evidence = ""

        evidence_match = re.search(
            r'"evidence"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL,
        )

        if evidence_match:

            evidence = (
                evidence_match
                .group(1)
                .strip()
            )

        if supported is not None:

            return {
                "supported": supported,

                "reason": (
                    reason
                    or (
                        "HCX JSON 형식이 일부 깨졌으나 "
                        "supported 값을 복구했습니다."
                    )
                ),

                "evidence": evidence,
            }

        # ---------------------------------------------------------------------
        # Fail closed
        # ---------------------------------------------------------------------

        if self.debug:

            print(
                "[ClaimVerifier] "
                "JSON parse failed."
            )

            print(
                text
            )

        return {
            "supported": False,

            "reason": (
                "Claim Verifier 응답을 정상적으로 "
                "파싱하지 못해 보수적으로 "
                "근거 불충분으로 처리했습니다."
            ),

            "evidence": "",
        }

    # =========================================================================
    # NORMALIZE RESULT
    # =========================================================================

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

            supported = (
                raw_supported
            )

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

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _normalize_string_list(
        values: Iterable[str] | None,
    ) -> list[str]:

        if not values:
            return []

        result = []

        for value in values:

            text = str(
                value or ""
            ).strip()

            if text:
                result.append(
                    text
                )

        return list(
            dict.fromkeys(
                result
            )
        )

    @staticmethod
    def _normalize_text(
        text: Any,
    ) -> str:

        value = str(
            text or ""
        ).lower()

        value = re.sub(
            r"\s+",
            "",
            value,
        )

        return value
