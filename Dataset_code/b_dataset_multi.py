from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from z_base_function import extract_field_pairs, load_input_records, normalize_text

# Prefer the user's finalized single-document generator.
# Fallback keeps this file usable before the rename is applied.
try:
    from a_dataset_single import (
        ClovaClient,
        ContextBundle,
        build_rows,
        record_to_context,
        write_rows,
    )
except ImportError:
    from z_base_function import (
        ClovaClient,
        ContextBundle,
        build_rows,
        record_to_context,
        write_rows,
    )


COMPARE_TYPES = {
    1: "same_company_same_report_type",
    2: "correction_chain",
    3: "same_company_same_metric",
    4: "cross_company_same_report_type",
}

SUPPORTED_DOCUMENT_COUNTS = {2, 3, 4}


@dataclass
class ComparisonGroup:
    group_id: str
    compare_type: str
    records: list[dict[str, Any]]
    metadata: dict[str, Any]


def receipt_no(record: dict[str, Any]) -> str:
    value = normalize_text(
        record.get("rcept_no")
        or record.get("receipt_no")
    )
    if value:
        return value

    source = normalize_text(record.get("_source_path"))
    match = re.search(r"(?<!\d)(20\d{12})(?!\d)", source)
    if match:
        return match.group(1)

    return ""


def record_identity(record: dict[str, Any]) -> str:
    return (
        receipt_no(record)
        or normalize_text(record.get("_source_path"))
        or normalize_text(record.get("title") or record.get("question"))
    )


def record_company(record: dict[str, Any]) -> str:
    value = normalize_text(
        record.get("corp_name")
        or record.get("company")
    )
    if value:
        return value

    title = normalize_text(record.get("title") or record.get("question"))
    if "/" in title:
        return title.split("/", 1)[0].strip() or "unknown"

    source = Path(str(record.get("_source_path") or ""))
    if len(source.parts) >= 3:
        return source.parts[-3]

    return "unknown"


def record_report_type(record: dict[str, Any]) -> str:
    for key in (
        "normalized_report_type",
        "doc_subtype",
        "report_nm",
    ):
        value = normalize_text(record.get(key))
        if value:
            return value

    title = normalize_text(record.get("title") or record.get("question"))
    parts = [part.strip() for part in title.split("/") if part.strip()]
    if len(parts) >= 2:
        return parts[1]

    return "unknown"


def record_chain_id(record: dict[str, Any]) -> str:
    return normalize_text(
        record.get("disclosure_chain_id")
        or record.get("correction_chain_id")
        or record.get("chain_id")
    )


def record_date(record: dict[str, Any]) -> datetime | None:
    values = (
        record.get("event_date"),
        record.get("rcept_dt"),
        receipt_no(record),
        record.get("title"),
        record.get("_source_path"),
    )

    for value in values:
        text = normalize_text(value)
        digits = re.sub(r"\D", "", text)

        for length, fmt in ((8, "%Y%m%d"),):
            if len(digits) >= length:
                candidate = digits[:length]
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    pass

    return None


def record_year(record: dict[str, Any]) -> int | None:
    dt = record_date(record)
    return dt.year if dt else None


def is_index_record(record: dict[str, Any]) -> bool:
    source_name = Path(
        str(record.get("_source_path") or "")
    ).name.lower()

    metadata_keys = {
        "corp_code",
        "corp_name",
        "report_nm",
        "rcept_no",
        "rcept_dt",
    }

    return (
        source_name.startswith("list_")
        or metadata_keys.issubset(record.keys())
    )


def load_manifest_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Index manifest JSONL by receipt number."""
    if path is None or not path.exists():
        return {}

    index: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = normalize_text(
                item.get("rcept_no")
                or item.get("receipt_no")
            )
            if key:
                index[key] = item

    return index


def enrich_records_from_manifest(
    records: list[dict[str, Any]],
    manifest_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add metadata such as normalized_report_type,
    disclosure_chain_id, base_year/base_month, sector, etc.
    Raw document values always win over manifest values.
    """
    enriched: list[dict[str, Any]] = []

    for record in records:
        merged = dict(manifest_index.get(receipt_no(record), {}))
        merged.update(record)
        enriched.append(merged)

    return enriched


def split_records(
    records: list[dict[str, Any]],
    *,
    split_mode: str,
    split_seed: str,
    hash_ratios: tuple[float, float, float],
    train_end_year: int,
    validation_years: set[int],
    test_start_year: int,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all documents in a comparison group inside one split."""
    result = {
        "train": [],
        "validation": [],
        "test": [],
    }

    def hash_split(identity: str) -> str:
        train_ratio, validation_ratio, test_ratio = hash_ratios
        total = train_ratio + validation_ratio + test_ratio
        value = int(
            hashlib.sha256(
                (split_seed + identity).encode("utf-8")
            ).hexdigest()[:12],
            16,
        ) / float(0xFFFFFFFFFFFF)

        if value < train_ratio / total:
            return "train"
        if value < (train_ratio + validation_ratio) / total:
            return "validation"
        return "test"

    for record in records:
        identity = record_identity(record)

        if split_mode == "hash":
            split = hash_split(identity)
        elif split_mode == "time":
            year = record_year(record)
            if year is None:
                split = hash_split(identity)
            elif year <= train_end_year:
                split = "train"
            elif year in validation_years:
                split = "validation"
            elif year >= test_start_year:
                split = "test"
            else:
                split = hash_split(identity)
        else:
            raise ValueError("split_mode must be 'time' or 'hash'")

        result[split].append(record)

    return result


def metric_labels(record: dict[str, Any]) -> set[str]:
    """
    Approximate 'metric' using actual document field labels containing
    numeric values. This is intentionally generic and can be refined later.
    """
    excluded = {
        "문서명",
        "회사명",
        "보고서명",
        "공시일",
        "계약일자",
        "계약 시작일",
        "계약 종료일",
        "유보기한",
    }

    result: set[str] = set()

    for label, value in extract_field_pairs(record):
        label = normalize_text(label)
        value = normalize_text(value)

        if not label or label in excluded:
            continue

        if re.search(r"\d", value):
            result.add(label)

    return result


def _window_groups(
    items: list[dict[str, Any]],
    document_count: int,
) -> Iterable[list[dict[str, Any]]]:
    if len(items) < document_count:
        return

    ordered = sorted(
        items,
        key=lambda r: (
            record_date(r) or datetime.min,
            record_identity(r),
        ),
    )

    for start in range(0, len(ordered) - document_count + 1):
        yield ordered[start : start + document_count]


def _group_id(
    compare_type: str,
    records: list[dict[str, Any]],
) -> str:
    joined = "|".join(record_identity(r) for r in records)
    digest = hashlib.sha1(
        f"{compare_type}|{joined}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{compare_type}_{digest}"


def build_comparison_groups(
    records: list[dict[str, Any]],
    *,
    compare_type: str,
    document_count: int,
    similar_period_days: int = 90,
    max_groups: int = 0,
) -> list[ComparisonGroup]:
    if compare_type not in COMPARE_TYPES.values():
        raise ValueError(
            f"Unsupported compare_type: {compare_type}"
        )

    if document_count not in SUPPORTED_DOCUMENT_COUNTS:
        raise ValueError(
            "document_count must be one of 2, 3, 4"
        )

    groups: list[ComparisonGroup] = []
    seen: set[tuple[str, ...]] = set()

    def add_group(
        selected: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        if len(selected) != document_count:
            return

        ids = tuple(record_identity(r) for r in selected)
        if len(set(ids)) != document_count:
            return

        dedupe_key = tuple(sorted(ids))
        if dedupe_key in seen:
            return

        seen.add(dedupe_key)
        groups.append(
            ComparisonGroup(
                group_id=_group_id(compare_type, selected),
                compare_type=compare_type,
                records=selected,
                metadata=metadata,
            )
        )

    # ---------------------------------------------------------
    # 1. same company + same report type + different time
    # ---------------------------------------------------------
    if compare_type == "same_company_same_report_type":
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            buckets[
                (
                    record_company(record),
                    record_report_type(record),
                )
            ].append(record)

        for (company, report_type), items in buckets.items():
            for selected in _window_groups(items, document_count):
                dates = [record_date(r) for r in selected]
                if len({d for d in dates if d is not None}) < 2:
                    continue

                add_group(
                    selected,
                    {
                        "company": company,
                        "report_type": report_type,
                    },
                )

    # ---------------------------------------------------------
    # 2. same disclosure/correction chain
    # ---------------------------------------------------------
    elif compare_type == "correction_chain":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            chain_id = record_chain_id(record)
            if chain_id:
                buckets[chain_id].append(record)

        for chain_id, items in buckets.items():
            if len(items) < document_count:
                continue

            for selected in _window_groups(items, document_count):
                add_group(
                    selected,
                    {
                        "chain_id": chain_id,
                        "company": record_company(selected[0]),
                        "report_type": record_report_type(selected[0]),
                    },
                )

    # ---------------------------------------------------------
    # 3. same company + same metric + different period
    # ---------------------------------------------------------
    elif compare_type == "same_company_same_metric":
        metric_buckets: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)

        for record in records:
            company = record_company(record)
            for metric in metric_labels(record):
                metric_buckets[(company, metric)].append(record)

        for (company, metric), items in metric_buckets.items():
            for selected in _window_groups(items, document_count):
                add_group(
                    selected,
                    {
                        "company": company,
                        "metric": metric,
                    },
                )

    # ---------------------------------------------------------
    # 4. different companies + same report type + similar period
    # ---------------------------------------------------------
    elif compare_type == "cross_company_same_report_type":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            buckets[record_report_type(record)].append(record)

        for report_type, items in buckets.items():
            for selected in _window_groups(items, document_count):
                companies = [record_company(r) for r in selected]
                if len(set(companies)) != document_count:
                    continue

                dates = [record_date(r) for r in selected]
                if any(d is None for d in dates):
                    continue

                date_values = [d for d in dates if d is not None]
                gap = (max(date_values) - min(date_values)).days

                if gap > similar_period_days:
                    continue

                add_group(
                    selected,
                    {
                        "companies": companies,
                        "report_type": report_type,
                        "period_gap_days": gap,
                    },
                )

    groups.sort(
        key=lambda g: (
            g.compare_type,
            g.group_id,
        )
    )

    if max_groups > 0:
        groups = groups[:max_groups]

    return groups


def comparison_group_to_bundle(
    group: ComparisonGroup,
    *,
    max_context_chars: int,
) -> ContextBundle:
    """
    Render 2~4 documents into one multi-document ContextBundle.
    """
    per_doc_limit = max(
        1000,
        max_context_chars // len(group.records),
    )

    blocks: list[str] = []
    source_ids: list[str] = []

    for index, record in enumerate(group.records, start=1):
        block, source_id = record_to_context(
            record,
            index,
            per_doc_limit,
        )
        blocks.append(block)
        source_ids.append(source_id)

    header = (
        f"[비교유형]\n{group.compare_type}\n\n"
    )

    context = (
        header
        + "\n\n".join(blocks)
    )[:max_context_chars]

    return ContextBundle(
        context=context,
        source_ids=source_ids,
        mode="multi",
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(
        r"[^0-9A-Za-z가-힣._-]+",
        "_",
        value,
    ).strip("._") or "group"

    return cleaned[:120]


def write_group_json(
    path: Path,
    *,
    split_name: str,
    group: ComparisonGroup,
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": split_name,
        "group_id": group.group_id,
        "compare_type": group.compare_type,
        "document_count": len(group.records),
        "metadata": group.metadata,
        "documents": [
            {
                "receipt_no": receipt_no(record),
                "company": record_company(record),
                "report_type": record_report_type(record),
                "event_date": (
                    record_date(record).strftime("%Y-%m-%d")
                    if record_date(record)
                    else None
                ),
                "source": normalize_text(record.get("_source_path")),
            }
            for record in group.records
        ],
        "generated_count": len(rows),
        "rejected_count": len(errors),
        "rows": rows,
        "errors": errors,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def reindex_rows(rows: list[dict[str, Any]]) -> None:
    for cid, row in enumerate(rows):
        row["C_ID"] = cid
        row["T_ID"] = 0


def generate_comparison_dataset(
    *,
    split_map: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    compare_type: str,
    document_count: int,
    client: ClovaClient,
    questions_per_group: int,
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    cache_path: Path,
    similar_period_days: int = 90,
    max_groups_per_split: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = output_dir / "generated"
    rejected_dir = output_dir / "rejected"

    manifest: dict[str, Any] = {
        "compare_type": compare_type,
        "document_count": document_count,
        "questions_per_group": questions_per_group,
        "similar_period_days": similar_period_days,
        "splits": {},
    }

    for split_name, records in split_map.items():
        groups = build_comparison_groups(
            records,
            compare_type=compare_type,
            document_count=document_count,
            similar_period_days=similar_period_days,
            max_groups=max_groups_per_split,
        )

        print(
            f"[{split_name}] comparison groups: {len(groups)} "
            f"(type={compare_type}, docs={document_count})"
        )

        final_rows: list[dict[str, Any]] = []
        split_errors: list[str] = []

        for index, group in enumerate(groups, start=1):
            print(
                f"  [{index}/{len(groups)}] "
                f"{group.group_id}"
            )

            bundle = comparison_group_to_bundle(
                group,
                max_context_chars=max_context_chars,
            )

            try:
                rows, errors = build_rows(
                    [bundle],
                    client,
                    single_count=0,
                    multi_count=questions_per_group,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    include_context=True,
                    system_prompt=system_prompt,
                    cache_path=cache_path,
                    limit=0,
                )
            except Exception as exc:
                rows = []
                errors = [str(exc)]

            # Local C_ID inside each JSON
            reindex_rows(rows)

            group_path = (
                generated_dir
                / split_name
                / f"{safe_filename(group.group_id)}.json"
            )

            write_group_json(
                group_path,
                split_name=split_name,
                group=group,
                rows=rows,
                errors=errors,
            )

            if errors:
                split_errors.extend(
                    f"{group.group_id}: {error}"
                    for error in errors
                )

            final_rows.extend(rows)

        # Final global C_ID for tuning CSV
        reindex_rows(final_rows)

        csv_path = output_dir / f"{split_name}.csv"
        write_rows(
            final_rows,
            csv_path,
            "csv",
            system_prompt,
        )

        if split_errors:
            error_path = (
                rejected_dir
                / f"{split_name}.errors.log"
            )
            error_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            error_path.write_text(
                "\n".join(split_errors),
                encoding="utf-8",
            )

        manifest["splits"][split_name] = {
            "records": len(records),
            "groups": len(groups),
            "rows": len(final_rows),
        }

        print(
            f"Wrote {csv_path}: {len(final_rows)} rows"
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )