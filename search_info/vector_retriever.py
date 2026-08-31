from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# PATH / CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DB_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "chroma_company"
)

DEFAULT_COMPANY_MAP_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "company_collection_map.json"
)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"

DEFAULT_TOP_K = 20

# metadata 조건을 만족하는 결과가 부족할 때
# HNSW 후보를 단계적으로 늘림
DEFAULT_CANDIDATE_SIZES = (
    100,
    300,
    1000,
    3000,
    10000,
)


# =============================================================================
# VECTOR RETRIEVER
# =============================================================================

class VectorRetriever:
    """
    회사별 Chroma collection 기반 Vector Retriever.

    검색 방식
    ----------
    1. BGE-M3로 질문 embedding 생성
    2. 회사별 Chroma HNSW 검색
       - Chroma metadata where filter 사용하지 않음
    3. Python에서 metadata post-filter
    4. 결과가 부족하면 후보 수를 단계적으로 확대

    이유
    ----
    현재 Chroma 1.5.9 환경에서 metadata where filter를 포함한
    vector query가 매우 느리게 동작하는 문제가 확인됨.

    따라서 빠른 HNSW 검색을 먼저 수행한 뒤,
    metadata filtering은 Python에서 수행한다.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        company_map_path: str | Path = DEFAULT_COMPANY_MAP_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        max_seq_length: int = 512,
        candidate_sizes: Iterable[int] = DEFAULT_CANDIDATE_SIZES,
    ) -> None:

        self.db_path = Path(db_path)
        self.company_map_path = Path(company_map_path)

        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length

        self.candidate_sizes = tuple(
            sorted(
                set(
                    int(x)
                    for x in candidate_sizes
                    if int(x) > 0
                )
            )
        )

        if not self.candidate_sizes:
            raise ValueError(
                "candidate_sizes가 비어 있습니다."
            )

        # ---------------------------------------------------------------------
        # 파일 검증
        # ---------------------------------------------------------------------

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Chroma DB를 찾을 수 없습니다: "
                f"{self.db_path}"
            )

        if not self.company_map_path.exists():
            raise FileNotFoundError(
                f"회사 collection map을 찾을 수 없습니다: "
                f"{self.company_map_path}"
            )

        # ---------------------------------------------------------------------
        # 회사 -> collection map
        # ---------------------------------------------------------------------

        with self.company_map_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.company_map: dict[str, str] = json.load(f)

        # ---------------------------------------------------------------------
        # Chroma
        # ---------------------------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        # collection object cache
        self._collection_cache: dict[str, Any] = {}

        # 회사별 collection count cache
        self._collection_count_cache: dict[str, int] = {}

        # ---------------------------------------------------------------------
        # BGE-M3
        # ---------------------------------------------------------------------

        print(
            f"[VectorRetriever] "
            f"Loading embedding model: "
            f"{self.model_name} ({self.device})"
        )

        model_start = time.time()

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        self.model.max_seq_length = (
            self.max_seq_length
        )

        print(
            "[VectorRetriever] "
            f"Embedding model loaded "
            f"({time.time() - model_start:.3f}s)"
        )

    # =========================================================================
    # COMPANY
    # =========================================================================

    def list_companies(
        self,
    ) -> list[str]:

        return sorted(
            self.company_map.keys()
        )

    def has_company(
        self,
        corp_name: str,
    ) -> bool:

        return (
            corp_name.strip()
            in self.company_map
        )

    def get_collection_name(
        self,
        corp_name: str,
    ) -> str:

        corp_name = corp_name.strip()

        if corp_name not in self.company_map:
            raise KeyError(
                f"Vector DB에 없는 회사입니다: "
                f"{corp_name}"
            )

        return self.company_map[
            corp_name
        ]

    def get_collection(
        self,
        corp_name: str,
    ):

        corp_name = corp_name.strip()

        if corp_name in self._collection_cache:
            return self._collection_cache[
                corp_name
            ]

        collection_name = (
            self.get_collection_name(
                corp_name
            )
        )

        collection = (
            self.client.get_collection(
                name=collection_name
            )
        )

        self._collection_cache[
            corp_name
        ] = collection

        return collection

    def get_collection_count(
        self,
        corp_name: str,
    ) -> int:

        corp_name = corp_name.strip()

        if corp_name in self._collection_count_cache:
            return self._collection_count_cache[
                corp_name
            ]

        collection = self.get_collection(
            corp_name
        )

        count = int(
            collection.count()
        )

        self._collection_count_cache[
            corp_name
        ] = count

        return count

    # =========================================================================
    # QUERY EMBEDDING
    # =========================================================================

    def encode_query(
        self,
        question: str,
    ) -> list[float]:

        question = question.strip()

        if not question:
            raise ValueError(
                "질문이 비어 있습니다."
            )

        embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise RuntimeError(
                "Query embedding shape 오류: "
                f"{embedding.shape}"
            )

        if embedding.shape[0] != 1024:
            raise RuntimeError(
                "BGE-M3 embedding dimension 오류: "
                f"{embedding.shape[0]}"
            )

        return embedding.tolist()

    # =========================================================================
    # METADATA FILTER
    # =========================================================================

    @staticmethod
    def _has_metadata_filter(
        *,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
    ) -> bool:

        return any(
            value is not None
            for value in (
                base_year,
                base_month,
                doc_group,
                doc_subtype,
                search_priority,
                is_correction,
                rcept_no,
                doc_id,
            )
        )

    @staticmethod
    def _metadata_matches(
        metadata: dict[str, Any] | None,
        *,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
    ) -> bool:

        metadata = metadata or {}

        if base_year is not None:
            try:
                actual_year = int(
                    metadata.get(
                        "base_year"
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                return False

            if actual_year != int(
                base_year
            ):
                return False

        if base_month is not None:
            try:
                actual_month = int(
                    metadata.get(
                        "base_month"
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                return False

            if actual_month != int(
                base_month
            ):
                return False

        if (
            doc_group is not None
            and str(
                metadata.get(
                    "doc_group",
                    "",
                )
            )
            != str(doc_group)
        ):
            return False

        if (
            doc_subtype is not None
            and str(
                metadata.get(
                    "doc_subtype",
                    "",
                )
            )
            != str(doc_subtype)
        ):
            return False

        if (
            search_priority is not None
            and str(
                metadata.get(
                    "search_priority",
                    "",
                )
            )
            != str(search_priority)
        ):
            return False

        if (
            is_correction is not None
            and metadata.get(
                "is_correction"
            )
            != bool(is_correction)
        ):
            return False

        if (
            rcept_no is not None
            and str(
                metadata.get(
                    "rcept_no",
                    "",
                )
            )
            != str(rcept_no)
        ):
            return False

        if (
            doc_id is not None
            and str(
                metadata.get(
                    "doc_id",
                    "",
                )
            )
            != str(doc_id)
        ):
            return False

        return True

    # =========================================================================
    # CHROMA RESULT PARSER
    # =========================================================================

    def _parse_chroma_result(
        self,
        *,
        corp_name: str,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:

        ids = (
            result.get("ids", [[]])[0]
            if result.get("ids")
            else []
        )

        documents = (
            result.get(
                "documents",
                [[]],
            )[0]
            if result.get(
                "documents"
            )
            else []
        )

        metadatas = (
            result.get(
                "metadatas",
                [[]],
            )[0]
            if result.get(
                "metadatas"
            )
            else []
        )

        distances = (
            result.get(
                "distances",
                [[]],
            )[0]
            if result.get(
                "distances"
            )
            else []
        )

        parsed: list[
            dict[str, Any]
        ] = []

        collection_name = (
            self.get_collection_name(
                corp_name
            )
        )

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            distance = float(
                distance
            )

            parsed.append(
                {
                    "corp_name": (
                        corp_name
                    ),
                    "collection_name": (
                        collection_name
                    ),
                    "candidate_rank": (
                        rank
                    ),
                    "chunk_id": (
                        chunk_id
                    ),
                    "distance": (
                        distance
                    ),
                    # cosine distance가
                    # 작을수록 유사
                    "score": (
                        1.0 - distance
                    ),
                    "document": (
                        document or ""
                    ),
                    "metadata": (
                        metadata or {}
                    ),
                }
            )

        return parsed

    # =========================================================================
    # RAW HNSW SEARCH
    # =========================================================================

    def _raw_vector_search(
        self,
        *,
        corp_name: str,
        query_embedding: list[float],
        candidate_k: int,
    ) -> list[dict[str, Any]]:

        collection = self.get_collection(
            corp_name
        )

        collection_count = (
            self.get_collection_count(
                corp_name
            )
        )

        actual_k = min(
            int(candidate_k),
            collection_count,
        )

        if actual_k <= 0:
            return []

        # IMPORTANT:
        # Chroma where filter를 사용하지 않는다.
        result = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=actual_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return (
            self._parse_chroma_result(
                corp_name=corp_name,
                result=result,
            )
        )

    # =========================================================================
    # FILTER RESULT
    # =========================================================================

    def _filter_results(
        self,
        results: list[
            dict[str, Any]
        ],
        *,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:

        filtered = []

        for item in results:

            if self._metadata_matches(
                item["metadata"],
                base_year=base_year,
                base_month=base_month,
                doc_group=doc_group,
                doc_subtype=doc_subtype,
                search_priority=search_priority,
                is_correction=is_correction,
                rcept_no=rcept_no,
                doc_id=doc_id,
            ):
                filtered.append(
                    item
                )

        return filtered

    # =========================================================================
    # SEARCH WITH PRE-COMPUTED EMBEDDING
    # =========================================================================

    def _search_company_with_embedding(
        self,
        *,
        query_embedding: list[float],
        corp_name: str,
        top_k: int,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
        debug: bool = False,
    ) -> list[dict[str, Any]]:

        corp_name = corp_name.strip()

        if not self.has_company(
            corp_name
        ):
            raise KeyError(
                f"Vector DB에 없는 회사: "
                f"{corp_name}"
            )

        if top_k <= 0:
            return []

        collection_count = (
            self.get_collection_count(
                corp_name
            )
        )

        has_filter = (
            self._has_metadata_filter(
                base_year=base_year,
                base_month=base_month,
                doc_group=doc_group,
                doc_subtype=doc_subtype,
                search_priority=search_priority,
                is_correction=is_correction,
                rcept_no=rcept_no,
                doc_id=doc_id,
            )
        )

        # ---------------------------------------------------------------------
        # metadata filter가 없다면
        # 필요한 top_k만 바로 HNSW 검색
        # ---------------------------------------------------------------------

        if not has_filter:

            results = (
                self._raw_vector_search(
                    corp_name=corp_name,
                    query_embedding=(
                        query_embedding
                    ),
                    candidate_k=top_k,
                )
            )

            return results[:top_k]

        # ---------------------------------------------------------------------
        # metadata filter 존재:
        #
        # HNSW 후보를 단계적으로 확대하고
        # Python에서 post-filter
        # ---------------------------------------------------------------------

        candidate_sizes = list(
            self.candidate_sizes
        )

        # top_k 자체가 첫 candidate보다 크면
        # candidate size에 포함
        candidate_sizes.append(
            max(
                top_k,
                top_k * 5,
            )
        )

        # collection 전체보다 큰 값 제거
        candidate_sizes = sorted(
            {
                min(
                    int(size),
                    collection_count,
                )
                for size in candidate_sizes
                if int(size) > 0
            }
        )

        last_filtered: list[
            dict[str, Any]
        ] = []

        last_candidate_k = 0

        for candidate_k in candidate_sizes:

            if candidate_k <= (
                last_candidate_k
            ):
                continue

            start = time.time()

            candidates = (
                self._raw_vector_search(
                    corp_name=corp_name,
                    query_embedding=(
                        query_embedding
                    ),
                    candidate_k=(
                        candidate_k
                    ),
                )
            )

            filtered = (
                self._filter_results(
                    candidates,
                    base_year=base_year,
                    base_month=base_month,
                    doc_group=doc_group,
                    doc_subtype=doc_subtype,
                    search_priority=search_priority,
                    is_correction=is_correction,
                    rcept_no=rcept_no,
                    doc_id=doc_id,
                )
            )

            if debug:
                print(
                    f"[VectorRetriever] "
                    f"{corp_name} "
                    f"candidate_k="
                    f"{candidate_k:,} "
                    f"filtered="
                    f"{len(filtered):,} "
                    f"time="
                    f"{time.time() - start:.3f}s"
                )

            last_filtered = filtered
            last_candidate_k = (
                candidate_k
            )

            if len(filtered) >= top_k:
                return filtered[
                    :top_k
                ]

            # collection 전체까지 검색했으면
            # 더 이상 확대 불가능
            if (
                candidate_k
                >= collection_count
            ):
                break

        # ---------------------------------------------------------------------
        # candidate_sizes 최대치까지 갔는데
        # top_k가 안 채워졌을 경우
        #
        # 필요하면 collection 전체를 한 번 검색한다.
        # ---------------------------------------------------------------------

        if (
            last_candidate_k
            < collection_count
            and len(
                last_filtered
            ) < top_k
        ):

            start = time.time()

            candidates = (
                self._raw_vector_search(
                    corp_name=corp_name,
                    query_embedding=(
                        query_embedding
                    ),
                    candidate_k=(
                        collection_count
                    ),
                )
            )

            last_filtered = (
                self._filter_results(
                    candidates,
                    base_year=base_year,
                    base_month=base_month,
                    doc_group=doc_group,
                    doc_subtype=doc_subtype,
                    search_priority=search_priority,
                    is_correction=is_correction,
                    rcept_no=rcept_no,
                    doc_id=doc_id,
                )
            )

            if debug:
                print(
                    f"[VectorRetriever] "
                    f"{corp_name} "
                    f"FULL-SCAN-HNSW "
                    f"candidate_k="
                    f"{collection_count:,} "
                    f"filtered="
                    f"{len(last_filtered):,} "
                    f"time="
                    f"{time.time() - start:.3f}s"
                )

        return last_filtered[
            :top_k
        ]

    # =========================================================================
    # SINGLE COMPANY SEARCH
    # =========================================================================

    def search_company(
        self,
        *,
        question: str,
        corp_name: str,
        top_k: int = DEFAULT_TOP_K,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
        debug: bool = False,
    ) -> list[dict[str, Any]]:

        query_start = time.time()

        query_embedding = (
            self.encode_query(
                question
            )
        )

        if debug:
            print(
                "[VectorRetriever] "
                f"query embedding: "
                f"{time.time() - query_start:.3f}s"
            )

        results = (
            self._search_company_with_embedding(
                query_embedding=(
                    query_embedding
                ),
                corp_name=corp_name,
                top_k=top_k,
                base_year=base_year,
                base_month=base_month,
                doc_group=doc_group,
                doc_subtype=doc_subtype,
                search_priority=search_priority,
                is_correction=is_correction,
                rcept_no=rcept_no,
                doc_id=doc_id,
                debug=debug,
            )
        )

        for rank, item in enumerate(
            results,
            start=1,
        ):
            item["local_rank"] = rank
            item["global_rank"] = rank

        return results

    # =========================================================================
    # MULTI-COMPANY SEARCH
    # =========================================================================

    def search_companies(
        self,
        *,
        question: str,
        companies: Iterable[str],
        top_k_per_company: int = 10,
        final_top_k: int | None = None,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
        debug: bool = False,
    ) -> list[dict[str, Any]]:

        companies = [
            company.strip()
            for company in companies
            if company
            and company.strip()
        ]

        # 중복 제거 / 순서 유지
        companies = list(
            dict.fromkeys(
                companies
            )
        )

        if not companies:
            return []

        unknown = [
            company
            for company in companies
            if not self.has_company(
                company
            )
        ]

        if unknown:
            raise KeyError(
                "Vector DB에 없는 회사: "
                + ", ".join(
                    unknown
                )
            )

        # 질문 embedding은 한 번만 수행
        query_start = time.time()

        query_embedding = (
            self.encode_query(
                question
            )
        )

        if debug:
            print(
                "[VectorRetriever] "
                f"query embedding: "
                f"{time.time() - query_start:.3f}s"
            )

        merged: list[
            dict[str, Any]
        ] = []

        for corp_name in companies:

            company_results = (
                self._search_company_with_embedding(
                    query_embedding=(
                        query_embedding
                    ),
                    corp_name=corp_name,
                    top_k=(
                        top_k_per_company
                    ),
                    base_year=base_year,
                    base_month=base_month,
                    doc_group=doc_group,
                    doc_subtype=doc_subtype,
                    search_priority=search_priority,
                    is_correction=is_correction,
                    rcept_no=rcept_no,
                    doc_id=doc_id,
                    debug=debug,
                )
            )

            for local_rank, item in enumerate(
                company_results,
                start=1,
            ):
                item["local_rank"] = (
                    local_rank
                )

            merged.extend(
                company_results
            )

        # cosine distance가 작을수록 유사
        merged.sort(
            key=lambda item: (
                item["distance"]
            )
        )

        if final_top_k is not None:
            merged = merged[
                :final_top_k
            ]

        for global_rank, item in enumerate(
            merged,
            start=1,
        ):
            item["global_rank"] = (
                global_rank
            )

        return merged


# =============================================================================
# TEST
# =============================================================================

def main() -> None:

    total_start = time.time()

    retriever = VectorRetriever(
        device="cpu"
    )

    question = (
        "SK텔레콤 2023년 1분기 "
        "무선통신사업 매출액은?"
    )

    print()
    print("=" * 100)
    print("VECTOR RETRIEVER TEST")
    print("=" * 100)

    print(
        "question:",
        question,
    )

    search_start = time.time()

    results = (
        retriever.search_company(
            question=question,
            corp_name="SK텔레콤",
            top_k=10,
            base_year=2023,
            base_month=3,
            doc_group="periodic",
            debug=True,
        )
    )

    search_elapsed = (
        time.time()
        - search_start
    )

    print()
    print("=" * 100)
    print("SEARCH RESULT")
    print("=" * 100)

    for item in results:

        metadata = (
            item["metadata"]
        )

        print()
        print(
            f"[TOP "
            f"{item['global_rank']}]"
        )

        print(
            "chunk_id :",
            item["chunk_id"],
        )

        print(
            "distance :",
            item["distance"],
        )

        print(
            "company  :",
            item["corp_name"],
        )

        print(
            "report   :",
            metadata.get(
                "report_nm"
            ),
        )

        print(
            "year     :",
            metadata.get(
                "base_year"
            ),
        )

        print(
            "month    :",
            metadata.get(
                "base_month"
            ),
        )

        print(
            "group    :",
            metadata.get(
                "doc_group"
            ),
        )

        print(
            "text     :",
            item["document"][
                :700
            ],
        )

    print()
    print("=" * 100)
    print("TIME")
    print("=" * 100)

    print(
        "search time:",
        f"{search_elapsed:.3f}s",
    )

    print(
        "total time :",
        f"{time.time() - total_start:.3f}s",
    )

    print()

    if results:
        print(
            "PASS: vector retrieval "
            "completed"
        )
    else:
        print(
            "WARNING: no result"
        )


if __name__ == "__main__":
    main()