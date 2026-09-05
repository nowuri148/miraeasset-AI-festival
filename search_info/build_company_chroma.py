from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "embedding_export"
)

# 기존 chroma / chroma_linux와 완전히 별개
OUTPUT_DB_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "chroma_company"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "company_build_checkpoint.json"
)

COMPANY_MAP_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "vector_db"
    / "company_collection_map.json"
)

EXPECTED_TOTAL = 1_343_770


# ============================================================
# COLLECTION NAME
# ============================================================

def collection_name_for_company(corp_name: str) -> str:
    """
    회사명을 직접 collection 이름으로 쓰지 않고
    안정적인 hash 기반 이름을 사용.

    예:
        KB금융
        -> corp_a1b2c3d4e5f6
    """

    digest = hashlib.sha1(
        corp_name.encode("utf-8")
    ).hexdigest()[:16]

    return f"corp_{digest}"


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> int:
    """
    마지막으로 완전히 처리한 shard index를 반환.
    없으면 -1.
    """

    if not CHECKPOINT_PATH.exists():
        return -1

    with CHECKPOINT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return int(
        data.get("last_completed_shard", -1)
    )


def save_checkpoint(
    shard_index: int,
    total_inserted: int,
) -> None:

    data = {
        "last_completed_shard": shard_index,
        "total_inserted": total_inserted,
    }

    with CHECKPOINT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("Company-sharded Chroma Builder")
    print("=" * 100)

    if not EXPORT_DIR.exists():
        raise FileNotFoundError(
            f"embedding_export 없음: {EXPORT_DIR}"
        )

    shard_paths = sorted(
        EXPORT_DIR.glob("shard_*.pkl")
    )

    if not shard_paths:
        raise RuntimeError(
            "embedding shard 파일이 없습니다."
        )

    print(f"export dir : {EXPORT_DIR}")
    print(f"output DB  : {OUTPUT_DB_PATH}")
    print(f"shards     : {len(shard_paths):,}")
    print()

    # --------------------------------------------------------
    # 중요:
    # chroma_company만 사용.
    #
    # 기존:
    #   chroma
    #   chroma_linux
    #
    # 전혀 접근/삭제하지 않음.
    # --------------------------------------------------------

    OUTPUT_DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(OUTPUT_DB_PATH)
    )

    # corp_name -> collection object
    collection_cache = {}

    # corp_name -> collection_name
    company_map = {}

    # 회사별 삽입 개수
    company_counts = defaultdict(int)

    last_completed = load_checkpoint()

    if last_completed >= 0:
        print(
            f"[RESUME] completed shard: "
            f"{last_completed:04d}"
        )

    # checkpoint가 존재해도 전체 inserted 수는
    # 실제 collection 기반으로 마지막에 검증할 것이므로
    # 진행 출력용 변수로만 사용.
    total_processed_this_run = 0

    start_time = time.time()

    # ========================================================
    # SHARD LOOP
    # ========================================================

    for shard_index, shard_path in enumerate(
        shard_paths
    ):

        if shard_index <= last_completed:
            continue

        with shard_path.open("rb") as f:
            shard = pickle.load(f)

        ids = shard["ids"]
        documents = shard["documents"]
        metadatas = shard["metadatas"]

        embeddings = np.asarray(
            shard["embeddings"],
            dtype=np.float32,
        )

        n = len(ids)

        if not (
            n
            == len(documents)
            == len(metadatas)
            == len(embeddings)
        ):
            raise RuntimeError(
                f"shard 길이 불일치: "
                f"{shard_path.name}"
            )

        # ----------------------------------------------------
        # 이번 shard를 회사별로 분리
        # ----------------------------------------------------

        grouped = defaultdict(
            lambda: {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }
        )

        for i in range(n):

            metadata = metadatas[i] or {}

            corp_name = str(
                metadata.get(
                    "corp_name",
                    ""
                )
            ).strip()

            if not corp_name:
                raise RuntimeError(
                    f"corp_name 누락\n"
                    f"shard={shard_path.name}\n"
                    f"id={ids[i]}"
                )

            group = grouped[corp_name]

            group["ids"].append(
                ids[i]
            )

            group["documents"].append(
                documents[i]
            )

            group["metadatas"].append(
                metadata
            )

            group["embeddings"].append(
                embeddings[i]
            )

        # ----------------------------------------------------
        # 회사별 collection에 삽입
        # ----------------------------------------------------

        for corp_name, group in grouped.items():

            collection_name = (
                collection_name_for_company(
                    corp_name
                )
            )

            company_map[
                corp_name
            ] = collection_name

            if corp_name not in collection_cache:

                collection = (
                    client.get_or_create_collection(
                        name=collection_name,
                        metadata={
                            "hnsw:space": "cosine",
                            "embedding_model": (
                                "BAAI/bge-m3"
                            ),
                            "corp_name": corp_name,
                        },
                    )
                )

                collection_cache[
                    corp_name
                ] = collection

            else:

                collection = (
                    collection_cache[
                        corp_name
                    ]
                )

            company_embeddings = np.asarray(
                group["embeddings"],
                dtype=np.float32,
            )

            collection.upsert(
                ids=group["ids"],
                documents=group[
                    "documents"
                ],
                metadatas=group[
                    "metadatas"
                ],
                embeddings=(
                    company_embeddings.tolist()
                ),
            )

            count = len(
                group["ids"]
            )

            company_counts[
                corp_name
            ] += count

            total_processed_this_run += count

        # ----------------------------------------------------
        # shard 전체가 정상적으로 끝난 뒤 checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            shard_index,
            total_processed_this_run,
        )

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            total_processed_this_run
            / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[SHARD "
            f"{shard_index + 1:04d}/"
            f"{len(shard_paths):04d}] "
            f"this_run="
            f"{total_processed_this_run:,} "
            f"companies="
            f"{len(collection_cache):,} "
            f"speed="
            f"{speed:.1f} chunks/s"
        )

    # ========================================================
    # COMPANY MAP 저장
    # ========================================================

    with COMPANY_MAP_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            company_map,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL VALIDATION")
    print("=" * 100)

    listed = client.list_collections()

    names = [
        item
        if isinstance(item, str)
        else item.name
        for item in listed
    ]

    print(
        f"collections : {len(names):,}"
    )

    grand_total = 0
    validation = []

    for idx, collection_name in enumerate(
        names,
        start=1,
    ):

        col = client.get_collection(
            name=collection_name
        )

        count = col.count()

        metadata = (
            col.metadata
            or {}
        )

        corp_name = metadata.get(
            "corp_name",
            "UNKNOWN",
        )

        grand_total += count

        validation.append(
            (
                count,
                corp_name,
                collection_name,
            )
        )

        print(
            f"[{idx:04d}/{len(names):04d}] "
            f"{corp_name} "
            f"-> {count:,}"
        )

    print()
    print("=" * 100)
    print("BUILD COMPLETE")
    print("=" * 100)

    print(
        f"company collections : "
        f"{len(names):,}"
    )

    print(
        f"total chunks        : "
        f"{grand_total:,}"
    )

    print(
        f"expected            : "
        f"{EXPECTED_TOTAL:,}"
    )

    print(
        f"DB path             : "
        f"{OUTPUT_DB_PATH}"
    )

    print(
        f"company map         : "
        f"{COMPANY_MAP_PATH}"
    )

    if grand_total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"전체 chunk 수 불일치: "
            f"{grand_total:,} "
            f"!= "
            f"{EXPECTED_TOTAL:,}"
        )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
