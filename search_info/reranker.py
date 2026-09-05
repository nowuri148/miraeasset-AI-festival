from __future__ import annotations

import re
from typing import Any, Iterable


DEFAULT_VECTOR_WEIGHT = 0.60
DEFAULT_KEYWORD_WEIGHT = 0.25
DEFAULT_METRIC_WEIGHT = 0.15


class HybridReranker:
    """
    VectorRetriever 결과를 가볍게 재정렬하는 Hybrid Reranker.

    입력
    ----
    - VectorRetriever 결과
    - 원 질문
    - Keyword Extractor가 추출한 metrics
    - topic_keywords

    점수
    ----
    final_score =
        vector_score   * vector_weight
        + keyword_score * keyword_weight
        + metric_score  * metric_weight
        + evidence_bonus * 0.05

    목적
    ----
    BGE-M3 vector retrieval의 의미 기반 검색 결과를 유지하면서,
    질문에서 명시적으로 요구된 지표/토픽이 실제 chunk 안에
    존재하는 근거 문서를 상위로 올린다.

    이 버전은 별도 reranker 모델을 사용하지 않으므로
    CPU 환경에서도 매우 빠르게 동작한다.
    """

    def __init__(
        self,
        *,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        metric_weight: float = DEFAULT_METRIC_WEIGHT,
    ) -> None:

        total = (
            vector_weight
            + keyword_weight
            + metric_weight
        )

        if total <= 0:
            raise ValueError(
                "reranker weight sum must be > 0"
            )

        self.vector_weight = (
            vector_weight / total
        )

        self.keyword_weight = (
            keyword_weight / total
        )

        self.metric_weight = (
            metric_weight / total
        )

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def rerank(
        self,
        *,
        question: str,
        results: list[dict[str, Any]],
        metrics: Iterable[str] | None = None,
        topic_keywords: Iterable[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        VectorRetriever 결과를 재정렬한다.
        """

        if not results:
            return []

        if top_k <= 0:
            return []

        question = str(
            question or ""
        ).strip()

        metrics = (
            self._normalize_terms(
                metrics
            )
        )

        topic_keywords = (
            self._normalize_terms(
                topic_keywords
            )
        )

        reranked: list[
            dict[str, Any]
        ] = []

        for original_rank, item in enumerate(
            results,
            start=1,
        ):

            document = str(
                item.get(
                    "document"
                )
                or ""
            )

            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            searchable_text = (
                self._build_searchable_text(
                    document=document,
                    metadata=metadata,
                )
            )

            # =================================================================
            # 1. Vector score
            # =================================================================

            distance = float(
                item.get(
                    "distance",
                    1.0,
                )
            )

            vector_score = (
                self._distance_to_similarity(
                    distance
                )
            )

            # =================================================================
            # 2. Topic keyword score
            # =================================================================

            keyword_score = (
                self._term_match_score(
                    searchable_text,
                    topic_keywords,
                )
            )

            # =================================================================
            # 3. Metric score
            # =================================================================

            metric_score = (
                self._term_match_score(
                    searchable_text,
                    metrics,
                )
            )

            # =================================================================
            # 4. Question token overlap 보너스
            # =================================================================

            question_overlap = (
                self._question_overlap_score(
                    question,
                    searchable_text,
                )
            )

            keyword_score = min(
                1.0,
                (
                    keyword_score * 0.8
                    + question_overlap * 0.2
                ),
            )

            # =================================================================
            # 5. 숫자/표 근거 보너스
            # =================================================================

            evidence_bonus = (
                self._evidence_bonus(
                    document
                )
            )

            # =================================================================
            # 6. 최종 점수
            # =================================================================

            final_score = (
                vector_score
                * self.vector_weight
                +
                keyword_score
                * self.keyword_weight
                +
                metric_score
                * self.metric_weight
            )

            final_score += (
                evidence_bonus
                * 0.05
            )

            new_item = dict(
                item
            )

            new_item[
                "original_rank"
            ] = original_rank

            new_item[
                "rerank_vector_score"
            ] = round(
                vector_score,
                6,
            )

            new_item[
                "rerank_keyword_score"
            ] = round(
                keyword_score,
                6,
            )

            new_item[
                "rerank_metric_score"
            ] = round(
                metric_score,
                6,
            )

            new_item[
                "rerank_evidence_bonus"
            ] = round(
                evidence_bonus,
                6,
            )

            new_item[
                "rerank_score"
            ] = round(
                final_score,
                6,
            )

            reranked.append(
                new_item
            )

        # ---------------------------------------------------------------------
        # 높은 rerank_score 우선
        # 동점이면 원래 vector distance 우선
        # ---------------------------------------------------------------------

        reranked.sort(
            key=lambda item: (
                -float(
                    item[
                        "rerank_score"
                    ]
                ),
                float(
                    item.get(
                        "distance",
                        999.0,
                    )
                ),
            )
        )

        reranked = (
            reranked[:top_k]
        )

        for rank, item in enumerate(
            reranked,
            start=1,
        ):
            item[
                "rerank_rank"
            ] = rank

        return reranked

    # =========================================================================
    # VECTOR SCORE
    # =========================================================================

    @staticmethod
    def _distance_to_similarity(
        distance: float,
    ) -> float:
        """
        Chroma cosine distance를 similarity score로 변환한다.

        BGE-M3 embedding은 normalize_embeddings=True로 사용하고 있으므로
        distance가 작을수록 의미적으로 가까운 결과다.

        기존처럼 후보 집합 내부 min-max normalization을 사용하면
        0.24와 0.25 같은 아주 작은 차이도 1.0과 0.0으로 확대될 수 있다.

        따라서 단순히:

            similarity = 1 - distance

        를 사용한다.
        """

        similarity = (
            1.0
            - float(distance)
        )

        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

    # =========================================================================
    # TERM MATCH
    # =========================================================================

    def _term_match_score(
        self,
        text: str,
        terms: list[str],
    ) -> float:

        if not terms:
            return 0.0

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        matched = 0.0

        for term in terms:

            normalized_term = (
                self._normalize_text(
                    term
                )
            )

            if not normalized_term:
                continue

            # -----------------------------------------------------------------
            # phrase 전체 포함
            # -----------------------------------------------------------------

            if (
                normalized_term
                in normalized_text
            ):
                matched += 1.0
                continue

            # -----------------------------------------------------------------
            # phrase 전체는 없지만
            # 구성 token 일부가 존재하는 경우
            # -----------------------------------------------------------------

            tokens = (
                self._tokenize(
                    term
                )
            )

            if not tokens:
                continue

            token_hits = sum(
                1
                for token in tokens
                if (
                    self._normalize_text(
                        token
                    )
                    in normalized_text
                )
            )

            partial_score = (
                0.7
                * token_hits
                / len(tokens)
            )

            matched += (
                partial_score
            )

        return min(
            1.0,
            matched
            / len(terms),
        )

    # =========================================================================
    # QUESTION OVERLAP
    # =========================================================================

    def _question_overlap_score(
        self,
        question: str,
        text: str,
    ) -> float:

        question_tokens = (
            self._tokenize(
                question
            )
        )

        if not question_tokens:
            return 0.0

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        hits = 0

        for token in question_tokens:

            normalized_token = (
                self._normalize_text(
                    token
                )
            )

            if (
                normalized_token
                and normalized_token
                in normalized_text
            ):
                hits += 1

        return (
            hits
            / len(
                question_tokens
            )
        )

    # =========================================================================
    # EVIDENCE BONUS
    # =========================================================================

    @staticmethod
    def _evidence_bonus(
        document: str,
    ) -> float:
        """
        숫자/지표 질문에서 실제 근거 역할을 할 가능성이 높은
        표/숫자/단위가 포함된 chunk를 소폭 우대한다.

        이 보너스는 최대 final score +0.05이므로
        vector/keyword/metric 점수를 압도하지 않는다.
        """

        score = 0.0

        # ---------------------------------------------------------------------
        # 표 형태
        # ---------------------------------------------------------------------

        if "|" in document:
            score += 0.5

        # ---------------------------------------------------------------------
        # 숫자 존재
        # ---------------------------------------------------------------------

        if re.search(
            r"\d[\d,]*",
            document,
        ):
            score += 0.3

        # ---------------------------------------------------------------------
        # 금융/공시에서 자주 사용되는 단위 존재
        # ---------------------------------------------------------------------

        unit_terms = (
            "백만원",
            "억원",
            "천원",
            "원",
            "%",
            "명",
            "주",
        )

        if any(
            unit in document
            for unit in unit_terms
        ):
            score += 0.2

        return min(
            1.0,
            score,
        )

    # =========================================================================
    # SEARCHABLE TEXT
    # =========================================================================

    @staticmethod
    def _build_searchable_text(
        *,
        document: str,
        metadata: dict[str, Any],
    ) -> str:

        metadata_values = []

        for key in (
            "corp_name",
            "report_nm",
            "doc_group",
            "doc_subtype",
            "section_title",
            "table_title",
        ):

            value = (
                metadata.get(
                    key
                )
            )

            if value is not None:
                metadata_values.append(
                    str(value)
                )

        return (
            document
            + "\n"
            + " ".join(
                metadata_values
            )
        )

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        value = str(
            value or ""
        ).lower()

        # 공백과 구두점 일부 제거
        value = re.sub(
            r"[\s\-_()/\[\]{}]+",
            "",
            value,
        )

        return value

    @staticmethod
    def _tokenize(
        value: str,
    ) -> list[str]:

        tokens = re.findall(
            r"[가-힣A-Za-z0-9%]+",
            str(
                value or ""
            ),
        )

        # 1글자 토큰은 노이즈 가능성이 높아 제거
        return [
            token
            for token in tokens
            if len(token) >= 2
        ]

    @staticmethod
    def _normalize_terms(
        values: Iterable[str]
        | None,
    ) -> list[str]:

        if values is None:
            return []

        result = []

        for value in values:

            text = str(
                value
            ).strip()

            if text:
                result.append(
                    text
                )

        # 중복 제거, 순서 유지
        return list(
            dict.fromkeys(
                result
            )
        )


# =============================================================================
# SIMPLE TEST
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # chunk_1:
    # vector distance는 살짝 불리하지만
    # 질문에 필요한 "무선통신사업 매출액" 근거를 직접 포함
    #
    # chunk_2:
    # vector distance는 0.01 더 좋지만
    # 질문과 다른 "유선통신사업" 설명
    #
    # reranker라면 chunk_1을 1위로 올려야 정상
    # -------------------------------------------------------------------------

    sample_results = [
        {
            "chunk_id": (
                "chunk_1"
            ),
            "distance": 0.25,
            "document": (
                "회사: SK텔레콤\n"
                "보고서: 분기보고서 "
                "(2023.03)\n"
                "기준기간: 2023년 3월\n"
                "무선통신사업 | 매출액 | "
                "3,249,918 백만원 | "
                "비중 74%"
            ),
            "metadata": {
                "corp_name": (
                    "SK텔레콤"
                ),
                "report_nm": (
                    "분기보고서 (2023.03)"
                ),
                "doc_group": (
                    "periodic"
                ),
            },
        },
        {
            "chunk_id": (
                "chunk_2"
            ),
            "distance": 0.24,
            "document": (
                "회사: SK텔레콤\n"
                "보고서: 분기보고서 "
                "(2023.03)\n"
                "유선통신사업은 IPTV와 "
                "초고속인터넷 서비스를 "
                "제공하고 있습니다."
            ),
            "metadata": {
                "corp_name": (
                    "SK텔레콤"
                ),
                "report_nm": (
                    "분기보고서 (2023.03)"
                ),
                "doc_group": (
                    "periodic"
                ),
            },
        },
    ]

    reranker = (
        HybridReranker()
    )

    results = (
        reranker.rerank(
            question=(
                "SK텔레콤 2023년 "
                "1분기 무선통신사업 "
                "매출액은?"
            ),
            results=(
                sample_results
            ),
            metrics=[
                "무선통신사업 매출액"
            ],
            topic_keywords=[
                "SK텔레콤",
                "2023년 1분기",
                "무선통신사업",
                "매출액",
            ],
            top_k=2,
        )
    )

    print()
    print("=" * 80)
    print("RERANKER TEST")
    print("=" * 80)

    for item in results:

        print()
        print(
            f"[RANK "
            f"{item['rerank_rank']}]"
        )

        print(
            "chunk_id      :",
            item[
                "chunk_id"
            ],
        )

        print(
            "original_rank :",
            item[
                "original_rank"
            ],
        )

        print(
            "distance      :",
            item[
                "distance"
            ],
        )

        print(
            "vector_score  :",
            item[
                "rerank_vector_score"
            ],
        )

        print(
            "keyword_score :",
            item[
                "rerank_keyword_score"
            ],
        )

        print(
            "metric_score  :",
            item[
                "rerank_metric_score"
            ],
        )

        print(
            "evidence_bonus:",
            item[
                "rerank_evidence_bonus"
            ],
        )

        print(
            "final_score   :",
            item[
                "rerank_score"
            ],
        )

    print()
    print("=" * 80)

    if (
        results
        and results[0][
            "chunk_id"
        ]
        == "chunk_1"
    ):
        print(
            "PASS: relevant evidence "
            "chunk ranked first"
        )

    else:
        print(
            "FAIL: reranker ranking "
            "is not correct"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
