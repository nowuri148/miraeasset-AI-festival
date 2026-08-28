from pathlib import Path
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "derived"
    / "periodic_vector_chunks.jsonl"
)

chunks = []

with INPUT_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        chunks.append(
            json.loads(line)
        )


print("=" * 100)
print("PERIODIC CHUNK VALIDATION")
print("=" * 100)

print("총 chunk:", len(chunks))


# ============================================================
# 1. CHUNK ID 중복
# ============================================================

chunk_ids = [
    chunk.get("chunk_id")
    for chunk in chunks
]

id_counter = Counter(chunk_ids)

duplicate_ids = [
    chunk_id
    for chunk_id, count in id_counter.items()
    if count > 1
]

print()
print("[1] CHUNK ID")

print(
    "unique:",
    len(id_counter)
)

print(
    "duplicate:",
    len(duplicate_ids)
)

if duplicate_ids:

    print("중복 예시:")

    for x in duplicate_ids[:10]:
        print(" -", x)


# ============================================================
# 2. CHUNK TYPE
# ============================================================

type_counter = Counter()

strategy_counter = Counter()

for chunk in chunks:

    metadata = (
        chunk.get("metadata")
        or {}
    )

    type_counter[
        metadata.get(
            "chunk_type",
            "unknown",
        )
    ] += 1

    strategy_counter[
        metadata.get(
            "chunk_strategy",
            "unknown",
        )
    ] += 1


print()
print("[2] CHUNK TYPE")

for key, value in sorted(
    type_counter.items()
):
    print(
        f"{key:25s}: {value}"
    )


print()
print("[3] CHUNK STRATEGY")

for key, value in sorted(
    strategy_counter.items()
):
    print(
        f"{key:25s}: {value}"
    )


# ============================================================
# 3. TEXT LENGTH
# ============================================================

table_lengths = []
narrative_lengths = []

oversized_table = []

for chunk in chunks:

    text = str(
        chunk.get("text")
        or ""
    )

    metadata = (
        chunk.get("metadata")
        or {}
    )

    chunk_type = metadata.get(
        "chunk_type"
    )

    strategy = metadata.get(
        "chunk_strategy"
    )

    if chunk_type == "narrative":

        narrative_lengths.append(
            len(text)
        )

    else:

        table_lengths.append(
            len(text)
        )

        if len(text) > 1500:

            oversized_table.append(
                chunk
            )


print()
print("[4] TEXT LENGTH")

if narrative_lengths:

    print(
        "narrative min:",
        min(narrative_lengths)
    )

    print(
        "narrative max:",
        max(narrative_lengths)
    )

    print(
        "narrative avg:",
        round(
            sum(narrative_lengths)
            / len(narrative_lengths),
            1,
        )
    )


if table_lengths:

    print()

    print(
        "table min:",
        min(table_lengths)
    )

    print(
        "table max:",
        max(table_lengths)
    )

    print(
        "table avg:",
        round(
            sum(table_lengths)
            / len(table_lengths),
            1,
        )
    )

    print(
        "table > 1500:",
        len(oversized_table)
    )


# ============================================================
# 4. OVERSIZED TABLE이 정상 예외인지
# ============================================================

bad_oversized = []

for chunk in oversized_table:

    metadata = (
        chunk.get("metadata")
        or {}
    )

    if not metadata.get(
        "oversized_single_row",
        False,
    ):

        bad_oversized.append(
            chunk
        )


print()
print("[5] OVERSIZED TABLE")

print(
    ">1500 total:",
    len(oversized_table)
)

print(
    ">1500 but not oversized_single_row:",
    len(bad_oversized)
)

if bad_oversized:

    print("비정상 예시:")

    for chunk in bad_oversized[:10]:

        m = chunk["metadata"]

        print(
            chunk["chunk_id"],
            len(chunk["text"]),
            m.get("table_title"),
        )


# ============================================================
# 5. TABLE PART 연속성
# ============================================================

table_groups = defaultdict(list)

for chunk in chunks:

    metadata = (
        chunk.get("metadata")
        or {}
    )

    if metadata.get(
        "chunk_type"
    ) == "narrative":
        continue

    raw_file_index = metadata.get(
        "raw_file_index"
    )

    table_index = metadata.get(
        "table_index"
    )

    doc_id = metadata.get(
        "doc_id"
    )

    key = (
        doc_id,
        raw_file_index,
        table_index,
    )

    table_groups[key].append(
        metadata
    )


bad_parts = []

for key, items in table_groups.items():

    part_count = max(
        int(
            item.get(
                "table_part_count",
                1,
            )
            or 1
        )
        for item in items
    )

    parts = sorted(
        int(
            item.get(
                "table_part_index",
                1,
            )
            or 1
        )
        for item in items
    )

    expected = list(
        range(
            1,
            part_count + 1,
        )
    )

    if parts != expected:

        bad_parts.append(
            (
                key,
                parts,
                expected,
            )
        )


print()
print("[6] TABLE PART")

print(
    "table groups:",
    len(table_groups)
)

print(
    "broken part sequence:",
    len(bad_parts)
)

if bad_parts:

    for item in bad_parts[:10]:
        print(item)


# ============================================================
# 6. NARRATIVE META
# ============================================================

bad_narrative = []

for chunk in chunks:

    metadata = (
        chunk.get("metadata")
        or {}
    )

    if metadata.get(
        "chunk_type"
    ) != "narrative":
        continue

    if not metadata.get(
        "section_path"
    ):

        bad_narrative.append(
            chunk["chunk_id"]
        )


print()
print("[7] NARRATIVE META")

print(
    "section_path missing:",
    len(bad_narrative)
)


# ============================================================
# 7. DOCUMENT COVERAGE
# ============================================================

doc_ids = set()

for chunk in chunks:

    metadata = (
        chunk.get("metadata")
        or {}
    )

    doc_id = metadata.get(
        "doc_id"
    )

    if doc_id:
        doc_ids.add(
            doc_id
        )


print()
print("[8] DOCUMENT COVERAGE")

print(
    "unique doc_id:",
    len(doc_ids)
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)

critical_errors = (
    len(duplicate_ids)
    + len(bad_oversized)
    + len(bad_parts)
    + len(bad_narrative)
)

if critical_errors == 0:

    print(
        "VALIDATION RESULT: PASS"
    )

else:

    print(
        "VALIDATION RESULT: FAIL"
    )

    print(
        "critical errors:",
        critical_errors
    )

print("=" * 100)