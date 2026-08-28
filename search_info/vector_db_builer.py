from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "corpus" / "derived" / "vector_chunks.jsonl"
DB_PATH = PROJECT_ROOT / "corpus" / "vector_db" / "chroma"

COLLECTION_NAME = "disclosure_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 처음에는 5,000개로 검색 품질 검증 후,
# 전체 구축 시 None으로 변경하세요.
MAX_CHUNKS: int | None = 5000

READ_BATCH_SIZE = 500
EMBED_BATCH_SIZE = 32
RESET_COLLECTION = True
NORMALIZE_EMBEDDINGS = True


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sanitize_metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple, set)):
        return " | ".join(
            clean_text(v)
            for v in value
            if clean_text(v)
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def sanitize_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:

    result = {}

    for key, value in metadata.items():
        if key in {"embedding", "vector"}:
            continue

        result[str(key)] = sanitize_metadata_value(value)

    return result


def iter_jsonl(path: Path, max_chunks: int | None = None):
    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON parse error at line {line_number}: {exc}"
                ) from exc

            yield record
            count += 1

            if max_chunks is not None and count >= max_chunks:
                break


def batched(iterator, batch_size: int):
    batch = []

    for item in iterator:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def prepare_record(
    record: dict[str, Any],
) -> tuple[str, str, dict[str, str | int | float | bool]]:

    chunk_id = clean_text(record.get("chunk_id"))
    text = clean_text(record.get("text"))

    metadata = record.get("metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    if not chunk_id:
        raise ValueError("chunk_id is empty")

    if not text:
        raise ValueError(f"text is empty: {chunk_id}")

    safe_metadata = sanitize_metadata(
        {
            **metadata,
            "chunk_id": chunk_id,
        }
    )

    return chunk_id, text, safe_metadata


def create_client():
    DB_PATH.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(DB_PATH)
    )


def create_collection(client):
    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"기존 collection 삭제: {COLLECTION_NAME}")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
    )


def build_vector_db() -> None:
    print()
    print("=" * 100)
    print("Vector DB Builder")
    print("=" * 100)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다:\n{INPUT_PATH}"
        )

    print(f"입력: {INPUT_PATH}")
    print(f"DB:   {DB_PATH}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"MAX_CHUNKS: {MAX_CHUNKS}")

    print()
    print("Embedding model 로딩...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    client = create_client()
    collection = create_collection(client)

    total_inserted = 0
    skipped = 0

    group_counter = Counter()
    company_counter = Counter()

    start_time = time.time()

    records = iter_jsonl(
        INPUT_PATH,
        max_chunks=MAX_CHUNKS,
    )

    for batch_index, records_batch in enumerate(
        batched(records, READ_BATCH_SIZE),
        start=1,
    ):
        ids = []
        texts = []
        metadatas = []

        for record in records_batch:
            try:
                chunk_id, text, metadata = prepare_record(record)
            except ValueError as exc:
                skipped += 1
                print(f"[SKIP] {exc}")
                continue

            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)

            group_counter[
                clean_text(metadata.get("doc_group")) or "unknown"
            ] += 1

            company_counter[
                clean_text(metadata.get("corp_name")) or "unknown"
            ] += 1

        if not ids:
            continue

        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        total_inserted += len(ids)

        elapsed = time.time() - start_time
        speed = total_inserted / elapsed if elapsed > 0 else 0

        print(
            f"[BATCH {batch_index:04d}] "
            f"inserted={total_inserted:,} "
            f"skipped={skipped:,} "
            f"speed={speed:.1f} chunks/s"
        )

    elapsed = time.time() - start_time

    print()
    print("=" * 100)
    print("Vector DB Build Statistics")
    print("=" * 100)

    print(f"입력/저장 chunk: {total_inserted:,}")
    print(f"skip: {skipped:,}")
    print(f"collection count: {collection.count():,}")
    print(f"소요 시간: {elapsed / 60:.1f}분")

    print()
    print("[DOC GROUP]")

    for key, value in group_counter.most_common():
        print(f"{key:30s}: {value:,}")

    print()
    print("[TOP COMPANIES]")

    for key, value in company_counter.most_common(20):
        print(f"{key:30s}: {value:,}")

    print()
    print("=" * 100)
    print("Vector DB 생성 완료")
    print("=" * 100)
    print(f"저장 위치: {DB_PATH}")
    print(f"collection: {COLLECTION_NAME}")


def main() -> None:
    build_vector_db()


if __name__ == "__main__":
    main()
