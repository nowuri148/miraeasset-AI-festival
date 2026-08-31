from __future__ import annotations

import json
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


# =============================================================================
# RETRIEVER
# =============================================================================

class VectorRetriever:
    """
    회사별 Chroma collection 기반 BGE-M3 Vector Retriever.

    주요 기능
    ---------
    1. BGE-M3 모델 1회 로드
    2. 회사명 -> Chroma collection 매핑
    3. metadata filter 적용
    4. 단일 회사 검색
    5. 복수 회사 검색
    6. 검색 결과 공통 포맷 반환
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        company_map_path: str | Path = DEFAULT_COMPANY_MAP_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        max_seq_length: int = 512,
    ) -> None:

        self.db_path = Path(db_path)
        self.company_map_path = Path(company_map_path)
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length

        # ---------------------------------------------------------------------
        # 파일/경로 검증
        # ---------------------------------------------------------------------

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Chroma DB를 찾을 수 없습니다: {self.db_path}"
            )

        if not self.company_map_path.exists():
            raise FileNotFoundError(
                f"회사-collection 매핑 파일을 찾을 수 없습니다: "
                f"{self.company_map_path}"
            )

        # ---------------------------------------------------------------------
        # 회사 매핑 로드
        # ---------------------------------------------------------------------

        with self.company_map_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.company_map: dict[str, str] = json.load(f)

        # ---------------------------------------------------------------------
        # Chroma 연결
        # ---------------------------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        # ---------------------------------------------------------------------
        # BGE-M3 로드
        # 실제 서비스에서는 이 객체를 한 번만 생성해야 함.
        # ---------------------------------------------------------------------

        print(
            f"[VectorRetriever] Loading embedding model: "
            f"{self.model_name} ({self.device})"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        self.model.max_seq_length = self.max_seq_length

        print("[VectorRetriever] Embedding model loaded.")

    # =========================================================================
    # COMPANY
    # =========================================================================

    def has_company(
        self,
        corp_name: str,
    ) -> bool:
        return corp_name in self.company_map

    def get_collection_name(
        self,
        corp_name: str,
    ) -> str:

        corp_name = corp_name.strip()

        if corp_name not in self.company_map:
            raise KeyError(
                f"Vector DB에 없는 회사입니다: {corp_name}"
            )

        return self.company_map[corp_name]

    def get_collection(
        self,
        corp_name: str,
    ):
        collection_name = self.get_collection_name(
            corp_name
        )

        return self.client.get_collection(
            name=collection_name
        )

    def list_companies(self) -> list[str]:
        return sorted(self.company_map.keys())

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    def encode_query(
        self,
        question: str,
    ) -> list[float]:

        question = question.strip()

        if not question:
            raise ValueError("질문이 비어 있습니다.")

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
                f"예상하지 못한 embedding shape: "
                f"{embedding.shape}"
            )

        if embedding.shape[0] != 1024:
            raise RuntimeError(
                f"BGE-M3 embedding dimension 오류: "
                f"{embedding.shape[0]}"
            )

        return embedding.tolist()

    # =========================================================================
    # FILTER
    # =========================================================================

    @staticmethod
    def _build_where(
        *,
        base_year: int | None = None,
        base_month: int | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        search_priority: str | None = None,
        is_correction: bool | None = None,
        rcept_no: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any] | None:

        conditions: list[dict[str, Any]] = []

        if base_year is not None:
            conditions.append(
                {
                    "base_year": {
                        "$eq": int(base_year)
                    }
                }
            )

        if base_month is not None:
            conditions.append(
                {
                    "base_month": {
                        "$eq": int(base_month)
                    }
                }
            )

        if doc_group:
            conditions.append(
                {
                    "doc_group": {
                        "$eq": str(doc_group)
                    }
                }
            )

        if doc_subtype:
            conditions.append(
                {
                    "doc_subtype": {
                        "$eq": str(doc_subtype)
                    }
                }
            )

        if search_priority:
            conditions.append(
                {
                    "search_priority": {
                        "$eq": str(search_priority)
                    }
                }
            )

        if is_correction is not None:
            conditions.append(
                {
                    "is_correction": {
                        "$eq": bool(is_correction)
                    }
                }
            )

        if rcept_no:
            conditions.append(
                {
                    "rcept_no": {
                        "$eq": str(rcept_no)
                    }
                }
            )

        if doc_id:
            conditions.append(
                {
                    "doc_id": {
                        "$eq": str(doc_id)
                    }
                }
            )

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }

    # =========================================================================
    # RAW QUERY
    # =========================================================================

    def _query_collection(
        self,
        *,
        corp_name: str,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:

        collection = self.get_collection(
            corp_name
        )

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where is not None:
            query_kwargs["where"] = where

        result = collection.query(
            **query_kwargs
        )

        ids = (
            result.get("ids", [[]])[0]
            if result.get("ids")
            else []
        )

        documents = (
            result.get("documents", [[]])[0]
            if result.get("documents")
            else []
        )

        metadatas = (
            result.get("metadatas", [[]])[0]
            if result.get("metadatas")
            else []
        )

        distances = (
            result.get("distances", [[]])[0]
            if result.get("distances")
            else []
        )

        output: list[dict[str, Any]] = []

        for local_rank, (
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

            metadata = metadata or {}

            output.append(
                {
                    "corp_name": corp_name,
                    "collection_name": (
                        self.company_map[
                            corp_name
                        ]
                    ),
                    "local_rank": local_rank,
                    "chunk_id": chunk_id,
                    "distance": float(distance),
                    "score": (
                        1.0 - float(distance)
                    ),
                    "document": document or "",
                    "metadata": metadata,
                }
            )

        return output

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
    ) -> list[dict[str, Any]]:

        query_embedding = self.encode_query(
            question
        )

        where = self._build_where(
            base_year=base_year,
            base_month=base_month,
            doc_group=doc_group,
            doc_subtype=doc_subtype,
            search_priority=search_priority,
            is_correction=is_correction,
            rcept_no=rcept_no,
            doc_id=doc_id,
        )

        results = self._query_collection(
            corp_name=corp_name,
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        for global_rank, item in enumerate(
            results,
            start=1,
        ):
            item["global_rank"] = global_rank

        return results

    # =========================================================================
    # MULTI COMPANY SEARCH
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
    ) -> list[dict[str, Any]]:

        companies = [
            company.strip()
            for company in companies
            if company and company.strip()
        ]

        if not companies:
            return []

        # 중복 제거 + 입력 순서 보존
        companies = list(
            dict.fromkeys(companies)
        )

        unknown = [
            corp_name
            for corp_name in companies
            if not self.has_company(corp_name)
        ]

        if unknown:
            raise KeyError(
                "Vector DB에 없는 회사: "
                + ", ".join(unknown)
            )

        # -----------------------------------------------------
        # 중요한 부분:
        # 질문 embedding을 회사마다 다시 만들지 않고
        # 딱 한 번만 생성한다.
        # -----------------------------------------------------

        query_embedding = self.encode_query(
            question
        )

        where = self._build_where(
            base_year=base_year,
            base_month=base_month,
            doc_group=doc_group,
            doc_subtype=doc_subtype,
            search_priority=search_priority,
            is_correction=is_correction,
            rcept_no=rcept_no,
            doc_id=doc_id,
        )

        merged_results: list[
            dict[str, Any]
        ] = []

        for corp_name in companies:

            company_results = (
                self._query_collection(
                    corp_name=corp_name,
                    query_embedding=query_embedding,
                    top_k=top_k_per_company,
                    where=where,
                )
            )

            merged_results.extend(
                company_results
            )

        # cosine distance가 작을수록 유사
        merged_results.sort(
            key=lambda x: x["distance"]
        )

        if final_top_k is not None:
            merged_results = (
                merged_results[
                    :final_top_k
                ]
            )

        for global_rank, item in enumerate(
            merged_results,
            start=1,
        ):
            item["global_rank"] = global_rank

        return merged_results


# =============================================================================
# SIMPLE TEST
# =============================================================================

def main() -> None:

    retriever = VectorRetriever(
        device="cpu"
    )

    question = (
        "SK텔레콤 2023년 1분기 "
        "무선통신사업 매출액은?"
    )

    results = retriever.search_company(
        question=question,
        corp_name="SK텔레콤",
        top_k=10,
        base_year=2023,
        base_month=3,
        doc_group="periodic",
    )

    print()
    print("=" * 100)
    print("VECTOR RETRIEVER TEST")
    print("=" * 100)

    print("question:", question)
    print("results :", len(results))

    for item in results:

        metadata = item["metadata"]

        print()
        print(
            f"[TOP {item['global_rank']}]"
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
            item["document"][:700],
        )


if __name__ == "__main__":
    main()