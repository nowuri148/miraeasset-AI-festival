from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "embedding_export"
)

# 기존 DB와 완전히 다른 테스트 경로
TEST_DB_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "chroma_company_test"
)

COLLECTION_NAME = "kb_financial_test"

# chunk가 가장 많은 회사 중 하나
TARGET_COMPANY = "KB금융"


def main():
    print("=" * 100)
    print("Company-level Chroma Test")
    print("=" * 100)

    print(f"target company : {TARGET_COMPANY}")
    print(f"export dir     : {EXPORT_DIR}")
    print(f"test db        : {TEST_DB_PATH}")
    print()

    if not EXPORT_DIR.exists():
        raise FileNotFoundError(
            f"embedding_export 없음: {EXPORT_DIR}"
        )

    shard_paths = sorted(
        EXPORT_DIR.glob("shard_*.pkl")
    )

    if not shard_paths:
        raise RuntimeError("shard 파일이 없습니다.")

    print(f"shards: {len(shard_paths):,}")

    # 주의:
    # chroma_company_test만 삭제함.
    # chroma / chroma_linux는 절대 건드리지 않음.
    if TEST_DB_PATH.exists():
        print()
        print(f"[TEST DB ONLY] 기존 테스트 DB 삭제:")
        print(TEST_DB_PATH)

        shutil.rmtree(TEST_DB_PATH)

    TEST_DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(TEST_DB_PATH)
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "target_company": TARGET_COMPANY,
        },
    )

    inserted = 0
    start_time = time.time()

    # 검증용 첫 embedding 저장
    first_embedding = None
    first_id = None

    for shard_index, shard_path in enumerate(
        shard_paths,
        start=1,
    ):
        with shard_path.open("rb") as f:
            shard = pickle.load(f)

        ids = shard["ids"]
        documents = shard["documents"]
        metadatas = shard["metadatas"]
        embeddings = np.asarray(
            shard["embeddings"],
            dtype=np.float32,
        )

        selected_ids = []
        selected_documents = []
        selected_metadatas = []
        selected_embeddings = []

        for i, metadata in enumerate(metadatas):
            metadata = metadata or {}

            corp_name = str(
                metadata.get("corp_name", "")
            ).strip()

            if corp_name != TARGET_COMPANY:
                continue

            selected_ids.append(ids[i])
            selected_documents.append(documents[i])
            selected_metadatas.append(metadata)
            selected_embeddings.append(embeddings[i])

            if first_embedding is None:
                first_embedding = embeddings[i].copy()
                first_id = ids[i]

        if not selected_ids:
            continue

        selected_embeddings = np.asarray(
            selected_embeddings,
            dtype=np.float32,
        )

        collection.upsert(
            ids=selected_ids,
            documents=selected_documents,
            metadatas=selected_metadatas,
            embeddings=selected_embeddings.tolist(),
        )

        inserted += len(selected_ids)

        elapsed = time.time() - start_time
        speed = inserted / elapsed if elapsed else 0

        print(
            f"[SHARD {shard_index:04d}/{len(shard_paths):04d}] "
            f"inserted={inserted:,} "
            f"speed={speed:.1f} chunks/s"
        )

    print()
    print("=" * 100)
    print("BUILD COMPLETE")
    print("=" * 100)

    print(f"company  : {TARGET_COMPANY}")
    print(f"inserted : {inserted:,}")
    print(f"test db  : {TEST_DB_PATH}")

    if inserted == 0:
        raise RuntimeError(
            f"{TARGET_COMPANY} chunk를 찾지 못했습니다."
        )

    # 나중에 별도 프로세스에서 query 검증할 수 있도록
    # 첫 embedding 저장
    query_path = TEST_DB_PATH.parent / "company_test_query.pkl"

    with query_path.open("wb") as f:
        pickle.dump(
            {
                "id": first_id,
                "embedding": first_embedding,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"query sample saved: {query_path}")
    print()
    print("중요: 실제 HNSW 재로딩 검증은")
    print("이 프로세스 종료 후 별도 명령으로 수행합니다.")


if __name__ == "__main__":
    main()
