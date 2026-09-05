from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


# =============================================================================
# PATH / ENV
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)


DEFAULT_BASE_URL = os.getenv(
    "CLOVA_BASE_URL",
    "https://clovastudio.stream.ntruss.com/v1/openai",
)

DEFAULT_MODEL = os.getenv(
    "CLOVA_MODEL",
    "HCX-005",
)

DEFAULT_MAX_SOURCES = 5
DEFAULT_MAX_SOURCE_CHARS = 3500

INSUFFICIENT_EVIDENCE_ANSWER = (
    "제공된 공시 근거만으로 답변을 확정할 수 없습니다."
)


# =============================================================================
# ANSWER GENERATOR
# =============================================================================

class HCXAnswerGenerator:
    """
    Reranker 결과를 근거로 HyperCLOVA X 최종 답변을 생성한다.

    핵심 원칙
    --------
    1. HCX는 제공된 SOURCE 안에서만 답변한다.
    2. HCX는 실제 출처 metadata를 직접 생성하지 않는다.
    3. HCX는 답변에 실제로 사용한 SOURCE 번호만 반환한다.
    4. 실제 source 정보는 Python 코드가 reranker metadata에서 복원한다.
    5. source attribution이 없는 구체적 답변은 Python에서 차단한다.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL,
        timeout: int = 60,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_source_chars: int = DEFAULT_MAX_SOURCE_CHARS,
        debug: bool = False,
    ) -> None:

        self.api_key = (
            api_key
            or os.getenv(
                "CLOVA_STUDIO_API_KEY",
                "",
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "CLOVA_STUDIO_API_KEY가 없습니다."
            )

        self.base_url = (
            str(base_url)
            .strip()
            .rstrip("/")
        )

        self.model_name = str(
            model_name
        ).strip()

        self.timeout = int(
            timeout
        )

        self.max_sources = int(
            max_sources
        )

        self.max_source_chars = int(
            max_source_chars
        )

        self.debug = bool(
            debug
        )

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def generate(
        self,
        *,
        question: str,
        reranked_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        질문 + reranker 결과를 받아
        최종 답변과 실제 사용 출처를 생성한다.
        """

        question = str(
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "question is empty"
            )

        if not reranked_results:
            return {
                "answer": (
                    "검색된 근거가 없어 "
                    "답변할 수 없습니다."
                ),
                "used_source_ids": [],
                "sources": [],
                "raw_response": None,
            }

        # ---------------------------------------------------------------------
        # SOURCE 구성
        # ---------------------------------------------------------------------

        source_items = (
            self._prepare_sources(
                reranked_results
            )
        )

        if not source_items:
            return {
                "answer": (
                    "검색된 근거가 없어 "
                    "답변할 수 없습니다."
                ),
                "used_source_ids": [],
                "sources": [],
                "raw_response": None,
            }

        context = (
            self._build_context(
                source_items
            )
        )

        messages = (
            self._build_messages(
                question=question,
                context=context,
            )
        )

        # ---------------------------------------------------------------------
        # HCX 호출
        # ---------------------------------------------------------------------

        start = time.time()

        raw_response = (
            self._call_hcx(
                messages
            )
        )

        elapsed = (
            time.time()
            - start
        )

        if self.debug:
            print(
                "[HCXAnswerGenerator] "
                f"generation time: "
                f"{elapsed:.3f}s"
            )

            print(
                "[HCXAnswerGenerator] "
                "raw response:"
            )

            print(
                raw_response
            )

        # ---------------------------------------------------------------------
        # JSON parsing
        # ---------------------------------------------------------------------

        parsed = (
            self._parse_answer_json(
                raw_response
            )
        )

        answer = str(
            parsed.get(
                "answer",
                "",
            )
            or ""
        ).strip()

        used_source_ids = (
            self._normalize_source_ids(
                parsed.get(
                    "used_source_ids"
                ),
                max_source_id=(
                    len(source_items)
                ),
            )
        )

        # ---------------------------------------------------------------------
        # Fail-closed:
        #
        # HCX가 구체적인 answer를 생성했는데
        # used_source_ids를 하나도 고르지 않았다면
        # 근거가 귀속되지 않은 답변이므로 폐기한다.
        # ---------------------------------------------------------------------

        if (
            answer
            and not used_source_ids
        ):

            if self.debug:
                print(
                    "[HCXAnswerGenerator] "
                    "WARNING: answer exists "
                    "but no source selected. "
                    "Rejecting ungrounded answer."
                )

            answer = (
                INSUFFICIENT_EVIDENCE_ANSWER
            )

        # answer가 아예 비어 있으면
        # 명시적인 근거 부족 답변으로 정규화
        if not answer:
            answer = (
                INSUFFICIENT_EVIDENCE_ANSWER
            )

            used_source_ids = []

        # ---------------------------------------------------------------------
        # source id → 실제 metadata 연결
        # ---------------------------------------------------------------------

        sources = (
            self._resolve_sources(
                source_items=source_items,
                used_source_ids=(
                    used_source_ids
                ),
            )
        )

        return {
            "answer": answer,

            "used_source_ids": (
                used_source_ids
            ),

            "sources": (
                sources
            ),

            "generation_time": (
                elapsed
            ),

            "raw_response": (
                raw_response
            ),
        }

    # =========================================================================
    # SOURCE PREPARATION
    # =========================================================================

    def _prepare_sources(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        prepared = []

        for index, item in enumerate(
            results[
                :self.max_sources
            ],
            start=1,
        ):

            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            document = str(
                item.get(
                    "document"
                )
                or item.get(
                    "text"
                )
                or ""
            ).strip()

            if not document:
                continue

            if (
                len(document)
                > self.max_source_chars
            ):
                document = (
                    document[
                        :self.max_source_chars
                    ]
                    + "\n...[truncated]"
                )

            prepared.append(
                {
                    "source_id": index,

                    "chunk_id": (
                        item.get(
                            "chunk_id"
                        )
                        or metadata.get(
                            "chunk_id"
                        )
                    ),

                    "corp_name": (
                        item.get(
                            "corp_name"
                        )
                        or metadata.get(
                            "corp_name"
                        )
                    ),

                    "report_nm": (
                        metadata.get(
                            "report_nm"
                        )
                    ),

                    "rcept_no": (
                        metadata.get(
                            "rcept_no"
                        )
                        or self._extract_rcept_no(
                            item.get(
                                "chunk_id"
                            )
                            or metadata.get(
                                "chunk_id"
                            )
                        )
                    ),

                    "doc_id": (
                        metadata.get(
                            "doc_id"
                        )
                    ),

                    "doc_group": (
                        metadata.get(
                            "doc_group"
                        )
                    ),

                    "base_year": (
                        metadata.get(
                            "base_year"
                        )
                    ),

                    "base_month": (
                        metadata.get(
                            "base_month"
                        )
                    ),

                    "rerank_rank": (
                        item.get(
                            "rerank_rank"
                        )
                    ),

                    "rerank_score": (
                        item.get(
                            "rerank_score"
                        )
                    ),

                    "retrieval_origin": (
                        item.get(
                            "retrieval_origin"
                        )
                    ),

                    "document": (
                        document
                    ),
                }
            )

        # source_id를 1부터 다시 연속적으로 부여
        # 중간에 document가 비어 skip된 경우를 방어
        for index, source in enumerate(
            prepared,
            start=1,
        ):
            source["source_id"] = index

        return prepared

    # =========================================================================
    # CONTEXT
    # =========================================================================

    @staticmethod
    def _build_context(
        source_items: list[
            dict[str, Any]
        ],
    ) -> str:

        blocks = []

        for source in source_items:

            header_lines = [
                (
                    f"[SOURCE "
                    f"{source['source_id']}]"
                ),
                (
                    f"회사: "
                    f"{source.get('corp_name')}"
                ),
                (
                    f"보고서: "
                    f"{source.get('report_nm')}"
                ),
                (
                    f"접수번호: "
                    f"{source.get('rcept_no')}"
                ),
                (
                    f"chunk_id: "
                    f"{source.get('chunk_id')}"
                ),
            ]

            document = str(
                source.get(
                    "document"
                )
                or ""
            )

            block = (
                "\n".join(
                    header_lines
                )
                + "\n\n"
                + document
            )

            blocks.append(
                block
            )

        return (
            "\n\n"
            + "\n\n".join(
                blocks
            )
        )

    # =========================================================================
    # PROMPT
    # =========================================================================

    @staticmethod
    def _build_messages(
        *,
        question: str,
        context: str,
    ) -> list[dict[str, str]]:

        system_prompt = """
당신은 기업 공시 문서를 근거로 질문에 답하는 금융 정보 분석 AI입니다.

반드시 다음 규칙을 지키십시오.

[기본 원칙]

1. 제공된 SOURCE의 내용만 사용하십시오.

2. SOURCE에 없는 사실을 외부 지식이나 상식으로 보완하거나 추측하지 마십시오.

3. 질문에 직접 필요한 내용만 간결하고 명확하게 답하십시오.

4. 수치, 단위, 기간, 회사명, 사업부문, 제품군, 지역, 지표를 정확하게 유지하십시오.

5. 답변에 필요한 사실이 여러 SOURCE에 나뉘어 있다면 여러 SOURCE를 연결하여 사용할 수 있습니다.

6. 하나의 SOURCE만으로 답을 확정할 수 없다면 필요한 다른 SOURCE의 근거까지 함께 확인하십시오.


[출처 선택 규칙]

7. 최종 answer에 포함된 모든 사실은 반드시 하나 이상의 SOURCE에 근거해야 합니다.

8. 구체적인 answer를 생성했다면 실제로 답변 작성에 사용한 SOURCE 번호를
used_source_ids에 반드시 1개 이상 포함하십시오.

9. 질문에 답하기 위해 여러 SOURCE의 정보를 연결했다면
그 연결에 실제로 사용한 모든 SOURCE 번호를 used_source_ids에 포함하십시오.

10. used_source_ids에는 실제로 제공된 SOURCE 번호만 넣으십시오.

11. SOURCE 번호 외의 접수번호, 보고서명, 회사명 등의 출처 metadata를
임의로 생성하지 마십시오.

12. answer에 구체적인 회사명, 기간, 사업부문, 제품군, 수치 또는 사실이 포함되어 있는데
used_source_ids가 빈 배열인 출력은 허용되지 않습니다.

13. 근거가 충분하지 않아 사용할 SOURCE를 하나도 선택할 수 없다면
구체적인 수치나 사실을 답하지 마십시오.
이 경우 반드시 다음과 같이 반환하십시오.

{
  "answer": "제공된 공시 근거만으로 답변을 확정할 수 없습니다.",
  "used_source_ids": []
}


[세부 범위 및 표 해석 규칙]

14. 질문에 특정 사업부문, 제품군, 지역, 자산군, 항목, 계약유형 등
세부 대상이 명시되어 있으면 전체 합계나 다른 하위 항목의 값을
그 대상의 값으로 답하지 마십시오.

15. 표에 여러 행 또는 열이 있는 경우 숫자 값만 보지 말고,
반드시 질문의 대상과 해당 숫자가 속한 행 이름 및 열 제목이
의미적으로 대응하는지 확인하십시오.

16. 질문이 특정 하위 범주를 묻는 경우:
- 해당 하위 범주에 직접 대응하는 행 또는 문장을 우선 사용하십시오.
- 전체 "합계", "총계" 값을 해당 하위 범주의 값으로 사용하지 마십시오.

17. 반대로 질문이 전체 값을 묻는 경우
특정 사업부문이나 일부 항목의 값만을 전체 값으로 답하지 마십시오.

18. 질문의 표현과 공시의 표현이 서로 다를 수 있습니다.
이 경우 제공된 SOURCE들 안에서 그 둘의 대응관계를 확인할 수 있을 때만
의미적으로 연결하십시오.

19. 질문의 표현과 공시의 표현 사이의 대응관계가
SOURCE만으로 충분히 확인되지 않으면 외부 지식을 사용하여 연결하지 마십시오.

20. 여러 SOURCE를 통해 다음과 같은 연결이 확인된다면 사용할 수 있습니다.

질문의 대상
→ SOURCE A에서 공시상의 사업부문/항목과 대응
→ SOURCE B에서 해당 사업부문/항목의 수치 확인

이 경우 SOURCE A와 SOURCE B를 모두 used_source_ids에 포함하십시오.

21. 여러 후보 숫자가 존재하면 다음 조건을 순서대로 확인하십시오.
- 회사가 일치하는가
- 기간이 일치하는가
- 질문의 세부 대상이 일치하는가
- 지표가 일치하는가
- 단위가 일치하는가

위 조건을 가장 정확하게 만족하는 값만 답하십시오.

22. 질문에 포함된 제한 표현을 임의로 제거하거나 더 넓은 범위로 해석하지 마십시오.

23. 질문이 특정 하위 범주를 묻는데 해당 범주에 직접 대응하는 값을
SOURCE에서 확정할 수 없는 경우,
전체 합계나 유사한 다른 값으로 대신 답하지 마십시오.


[출력 규칙]

24. 반드시 JSON object 하나만 출력하십시오.

25. Markdown 코드블록은 사용하지 마십시오.

26. 설명, 주석, 머리말, 꼬리말을 JSON 바깥에 추가하지 마십시오.


출력 형식:

{
  "answer": "질문에 대한 최종 답변",
  "used_source_ids": [1, 2]
}


올바른 예시:

{
  "answer": "A사의 2023년 특정 사업부문 설비투자액은 100억 원입니다.",
  "used_source_ids": [2, 4]
}


잘못된 예시:

{
  "answer": "A사의 2023년 특정 사업부문 설비투자액은 100억 원입니다.",
  "used_source_ids": []
}
""".strip()

        user_prompt = f"""
[질문]
{question}

[검색된 근거]
{context}

위 SOURCE들만 사용하여 질문에 답하십시오.

답변에 사용한 근거가 있다면 반드시 그 SOURCE 번호를
used_source_ids에 포함하십시오.

여러 SOURCE를 연결하여 답했다면 사용한 모든 SOURCE 번호를 포함하십시오.
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
                f"Bearer "
                f"{self.api_key}"
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
            "top_p": 0.8,
            "max_tokens": 700,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                "HyperCLOVA X "
                "Answer API 오류\n"
                f"HTTP "
                f"{response.status_code}\n"
                f"{response.text}"
            )

        data = (
            response.json()
        )

        try:
            return str(
                data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            ).strip()

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:

            raise RuntimeError(
                "HyperCLOVA X 응답에서 "
                "message.content를 "
                "찾을 수 없습니다.\n"
                f"{data}"
            ) from exc

    # =========================================================================
    # JSON PARSER
    # =========================================================================

    def _parse_answer_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        text = str(
            text or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "HCX answer response is empty"
            )

        # ---------------------------------------------------------------------
        # 1. 그대로 JSON parse
        # ---------------------------------------------------------------------

        try:
            value = json.loads(
                text
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except json.JSONDecodeError:
            pass

        # ---------------------------------------------------------------------
        # 2. ```json ... ``` 방어
        # ---------------------------------------------------------------------

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

        try:
            value = json.loads(
                cleaned
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except json.JSONDecodeError:
            pass

        # ---------------------------------------------------------------------
        # 3. 문자열 내부 JSON object 추출
        # ---------------------------------------------------------------------

        start = (
            cleaned.find("{")
        )

        end = (
            cleaned.rfind("}")
        )

        if (
            start != -1
            and end != -1
            and end > start
        ):

            candidate = (
                cleaned[
                    start:end + 1
                ]
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

        raise RuntimeError(
            "HCX 답변을 JSON으로 "
            "파싱하지 못했습니다.\n"
            f"RAW:\n{text}"
        )

    # =========================================================================
    # SOURCE ID NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_source_ids(
        value: Any,
        *,
        max_source_id: int,
    ) -> list[int]:

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            value = [value]

        result = []

        for item in value:

            try:
                source_id = int(
                    item
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                1
                <= source_id
                <= max_source_id
            ):
                result.append(
                    source_id
                )

        # 중복 제거, 순서 유지
        return list(
            dict.fromkeys(
                result
            )
        )

    # =========================================================================
    # SOURCE RESOLUTION
    # =========================================================================

    @staticmethod
    def _resolve_sources(
        *,
        source_items: list[
            dict[str, Any]
        ],
        used_source_ids: Iterable[int],
    ) -> list[dict[str, Any]]:

        by_id = {
            int(
                source[
                    "source_id"
                ]
            ): source
            for source in source_items
        }

        resolved = []

        for source_id in (
            used_source_ids
        ):

            source = (
                by_id.get(
                    int(source_id)
                )
            )

            if source is None:
                continue

            resolved.append(
                {
                    "source_id": (
                        int(source_id)
                    ),

                    "corp_name": (
                        source.get(
                            "corp_name"
                        )
                    ),

                    "report_nm": (
                        source.get(
                            "report_nm"
                        )
                    ),

                    "rcept_no": (
                        source.get(
                            "rcept_no"
                        )
                    ),

                    "doc_id": (
                        source.get(
                            "doc_id"
                        )
                    ),

                    "chunk_id": (
                        source.get(
                            "chunk_id"
                        )
                    ),

                    "doc_group": (
                        source.get(
                            "doc_group"
                        )
                    ),

                    "base_year": (
                        source.get(
                            "base_year"
                        )
                    ),

                    "base_month": (
                        source.get(
                            "base_month"
                        )
                    ),
                }
            )

        return resolved

    # =========================================================================
    # RECEIPT NUMBER FALLBACK
    # =========================================================================

    @staticmethod
    def _extract_rcept_no(
        chunk_id: Any,
    ) -> str | None:
        """
        metadata에 rcept_no가 없을 경우
        chunk_id 안의 14자리 접수번호를 보조적으로 추출.

        예:
        periodic_20230512000710_file_01_table_0058_part_001
                 ^^^^^^^^^^^^^^
        """

        if not chunk_id:
            return None

        match = re.search(
            r"(?<!\d)(\d{14})(?!\d)",
            str(chunk_id),
        )

        if not match:
            return None

        return (
            match.group(1)
        )


# =============================================================================
# ANSWER + SOURCE DISPLAY
# =============================================================================

def format_answer_with_sources(
    result: dict[str, Any],
) -> str:
    """
    최종 사용자 표시용 문자열.

    동일 공시의 여러 chunk가 선택된 경우
    rcept_no 기준으로 출처를 중복 제거한다.
    """

    answer = str(
        result.get(
            "answer"
        )
        or ""
    ).strip()

    sources = (
        result.get(
            "sources"
        )
        or []
    )

    if not sources:
        return answer

    # -------------------------------------------------------------------------
    # 동일 공시 중복 제거
    # -------------------------------------------------------------------------

    unique_sources = []
    seen = set()

    for source in sources:

        dedup_key = (
            source.get(
                "rcept_no"
            )
            or source.get(
                "doc_id"
            )
            or source.get(
                "chunk_id"
            )
        )

        if dedup_key in seen:
            continue

        seen.add(
            dedup_key
        )

        unique_sources.append(
            source
        )

    lines = [
        answer,
        "",
        "출처:",
    ]

    for display_index, source in enumerate(
        unique_sources,
        start=1,
    ):

        parts = []

        corp_name = (
            source.get(
                "corp_name"
            )
        )

        report_nm = (
            source.get(
                "report_nm"
            )
        )

        rcept_no = (
            source.get(
                "rcept_no"
            )
        )

        if corp_name:
            parts.append(
                str(
                    corp_name
                )
            )

        if report_nm:
            parts.append(
                str(
                    report_nm
                )
            )

        if rcept_no:
            parts.append(
                f"접수번호 "
                f"{rcept_no}"
            )

        if parts:
            lines.append(
                f"[{display_index}] "
                + ", ".join(
                    parts
                )
            )

    return "\n".join(
        lines
    )


# =============================================================================
# SIMPLE TEST
# =============================================================================

def main() -> None:
    """
    HCX 호출 없이 source 구성 로직만 확인하는 간단한 테스트.
    """

    fake_results = [
        {
            "chunk_id": (
                "periodic_20230512000710_"
                "file_01_table_0058_part_001"
            ),

            "corp_name": (
                "SK텔레콤"
            ),

            "rerank_rank": 1,

            "rerank_score": (
                0.81
            ),

            "document": (
                "회사: SK텔레콤\n"
                "보고서: 분기보고서 (2023.03)\n"
                "무선통신사업 | "
                "3,249,918 | 74%"
            ),

            "metadata": {
                "report_nm": (
                    "분기보고서 (2023.03)"
                ),

                "base_year": (
                    2023
                ),

                "base_month": (
                    3
                ),

                "doc_group": (
                    "periodic"
                ),
            },
        }
    ]

    generator = (
        HCXAnswerGenerator(
            debug=True
        )
    )

    sources = (
        generator._prepare_sources(
            fake_results
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SOURCE PREPARATION TEST"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            sources,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "=" * 80
    )

    if (
        sources
        and sources[0].get(
            "rcept_no"
        )
        == "20230512000710"
    ):
        print(
            "PASS: source metadata "
            "prepared correctly"
        )

    else:
        print(
            "FAIL: source metadata "
            "preparation error"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
