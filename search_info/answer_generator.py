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
    3. HCX는 사용한 SOURCE 번호만 반환한다.
    4. 실제 source 정보는 Python 코드가 reranker metadata에서 복원한다.
    5. 이를 통해 출처 hallucination을 방지한다.

    출력 예
    -------
    {
        "answer": "...",
        "used_source_ids": [1, 2],
        "sources": [
            {
                "source_id": 1,
                "corp_name": "SK텔레콤",
                "report_nm": "분기보고서 (2023.03)",
                "rcept_no": "20230512000710",
                "doc_id": "...",
                "chunk_id": "..."
            }
        ]
    }
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
        질문 + reranker 결과를 받아 최종 답변과 출처를 생성한다.
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

        # HCX가 source id를 비워버렸지만
        # 답변은 생성한 경우:
        # hallucination 방지를 위해 source 없는 답변을 그대로 신뢰하지 않음
        if (
            answer
            and not used_source_ids
        ):
            if self.debug:
                print(
                    "[HCXAnswerGenerator] "
                    "WARNING: answer exists "
                    "but no source selected."
                )

        return {
            "answer": answer,
            "used_source_ids": (
                used_source_ids
            ),
            "sources": sources,
            "generation_time": elapsed,
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
                or ""
            ).strip()

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

                    "document": (
                        document
                    ),
                }
            )

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

1. 제공된 SOURCE의 내용만 사용하십시오.
2. SOURCE에 없는 사실을 추측하거나 만들어내지 마십시오.
3. 질문에 직접 필요한 내용만 간결하고 명확하게 답하십시오.
4. 수치, 단위, 기간, 회사명을 정확하게 유지하십시오.
5. 여러 SOURCE가 같은 사실을 뒷받침하면 필요한 SOURCE만 선택하십시오.
6. 답변 작성에 실제로 사용한 SOURCE 번호만 used_source_ids에 넣으십시오.
7. SOURCE 번호 외의 출처 정보(접수번호, 보고서명 등)를 임의로 생성하지 마십시오.
8. 근거가 충분하지 않으면 이를 명확히 밝히고 used_source_ids는 빈 배열로 반환하십시오.
9. 반드시 JSON object 하나만 출력하십시오.
10. Markdown 코드블록은 사용하지 마십시오.

출력 형식:

{
  "answer": "질문에 대한 최종 답변",
  "used_source_ids": [1, 2]
}
""".strip()

        user_prompt = f"""
[질문]
{question}

[검색된 근거]
{context}

위 근거만 사용하여 질문에 답하십시오.
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
        # 1. 바로 JSON
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
        # 3. 문자열 내부 첫 JSON object 추출
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
                        source_id
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
        result.get("answer")
        or ""
    ).strip()

    sources = (
        result.get("sources")
        or []
    )

    if not sources:
        return answer

    # 동일 공시 중복 제거
    unique_sources = []
    seen = set()

    for source in sources:

        dedup_key = (
            source.get("rcept_no")
            or source.get("doc_id")
            or source.get("chunk_id")
        )

        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        unique_sources.append(source)

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

        corp_name = source.get(
            "corp_name"
        )

        report_nm = source.get(
            "report_nm"
        )

        rcept_no = source.get(
            "rcept_no"
        )

        if corp_name:
            parts.append(
                str(corp_name)
            )

        if report_nm:
            parts.append(
                str(report_nm)
            )

        if rcept_no:
            parts.append(
                f"접수번호 {rcept_no}"
            )

        lines.append(
            f"[{display_index}] "
            + ", ".join(parts)
        )

    return "\n".join(lines)

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
            "rerank_score": 0.81,

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
                "base_year": 2023,
                "base_month": 3,
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
    print("=" * 80)
    print("SOURCE PREPARATION TEST")
    print("=" * 80)

    print(
        json.dumps(
            sources,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 80)

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

    print("=" * 80)


if __name__ == "__main__":
    main()

