from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "embedding_export"
)

OUTPUT_DB_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "chroma_linux"
)

COLLECTION_NAME = "disclosure_chunks"

EXPECTED_TOTAL = 1_343_770


def main():
    print("=" * 100)
    print("Linux Chroma Rebuild")
    print("=" * 100)

    if not EXPORT_DIR.exists():
        raise FileNotFoundError(
            f"embedding export 없음: {EXPORT_DIR}"
        )

    shard_paths = sorted(
        EXPORT_DIR.glob("shard_*.pkl")
    )

    print(f"shards: {len(shard_paths):,}")

    if not shard_paths:
        raise RuntimeError("shard 파일이 없습니다.")

    # 새 DB를 처음부터 생성
    if OUTPUT_DB_PATH.exists():
        print(f"기존 chroma_linux 삭제: {OUTPUT_DB_PATH}")
        shutil.rmtree(OUTPUT_DB_PATH)

    OUTPUT_DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(OUTPUT_DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "BAAI/bge-m3",
        },
    )

    inserted = 0
    start_time = time.time()

    for shard_index, shard_path in enumerate(
        shard_paths,
        start=1,
    ):
        with shard_path.open("rb") as f:
            shard = pickle.load(f)

        ids = shard["ids"]
        documents = shard["documents"]
        metadatas = shard["metadatas"]
        embeddings = shard["embeddings"]

        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
            == len(embeddings)
        ):
            raise RuntimeError(
                f"shard 길이 불일치: {shard_path.name}"
            )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        inserted += len(ids)

        elapsed = time.time() - start_time
        speed = inserted / elapsed if elapsed else 0

        print(
            f"[SHARD {shard_index:04d}/{len(shard_paths):04d}] "
            f"inserted={inserted:,} "
            f"speed={speed:.1f} chunks/s"
        )

    print()
    print("=" * 100)
    print("REBUILD COMPLETE")
    print("=" * 100)

    count = collection.count()

    print(f"inserted         : {inserted:,}")
    print(f"collection count : {count:,}")
    print(f"expected         : {EXPECTED_TOTAL:,}")
    print(f"DB path          : {OUTPUT_DB_PATH}")

    if count != EXPECTED_TOTAL:
        raise RuntimeError(
            f"collection count 불일치: "
            f"{count:,} != {EXPECTED_TOTAL:,}"
        )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
