#!/usr/bin/env python3
"""Convert JSON/XML datasets to HyperCLOVA X instruction tuning format.

Supported input examples:
- [{"question": "...", "answer": "..."}]
- [{"input": "...", "output": "..."}]
- [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
- <root><item><question>..</question><answer>..</answer></item></root>

Output format:
- CSV: C_ID,T_ID,Text,Completion
- CSV with optional System_Prompt if present
- JSONL: each row is a JSON object with the same fields

Usage:
  python a_dataset_try1.py --input data.json --output out.csv --format csv
  python a_dataset_try1.py --input data.xml --output out.jsonl --format jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


REQUIRED_COLUMNS = ["C_ID", "T_ID", "Text", "Completion"]
SYSTEM_COLUMNS = ["System_Prompt", "C_ID", "T_ID", "Text", "Completion"]
DEFAULT_SYSTEM_PROMPT = (
    "기업 공시 질문에 정확히 답하세요. 제공된 근거에서 확인되는 사실만 사용하고, "
    "답변 뒤에 반드시 문서와 사용한 원문 항목을 근거로 제시하세요. "
    "근거가 부족하면 추측하지 말고 확인할 수 없다고 답하세요."
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [normalize_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        # Common dicts may contain nested values; use the first meaningful text.
        for key in ("text", "content", "value", "answer", "response", "output", "completion"):
            if key in value:
                return normalize_text(value[key])
        return " ".join(normalize_text(v) for v in value.values() if normalize_text(v))
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def get_first_value(data: dict[str, Any], keys: Iterable[str]) -> str:
    lower_map = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value not in (None, ""):
            return normalize_text(value)
    return ""


def extract_system_prompt(record: dict[str, Any]) -> str:
    for key in ("System_Prompt", "system_prompt", "systemPrompt"):
        value = record.get(key)
        if value not in (None, ""):
            return normalize_text(value)
    return ""


def normalize_label_name(label: str) -> str:
    text = str(label).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\s*(?:\d+[.)]|[-※])\s*", "", text)
    return text


def canonical_field_name(label: str) -> str:
    """Map disclosure-table labels to stable names used by Q/A templates."""
    compact = re.sub(r"[\sㆍ·()（）/%원]", "", normalize_label_name(label))
    aliases = (
        ("판매공급계약구분", "계약 구분"),
        ("체결계약명", "체결계약명"),
        ("계약금액", "계약금액"),
        ("최근매출액", "최근매출액"),
        ("매출액대비", "매출액 대비"),
        ("대규모법인여부", "대규모법인 여부"),
        ("계약상대", "계약상대"),
        ("회사와의관계", "회사와의 관계"),
        ("판매공급지역", "판매·공급지역"),
        ("시작일", "계약 시작일"),
        ("종료일", "계약 종료일"),
        ("주요계약조건", "주요 계약조건"),
        ("계약수주일자", "계약일자"),
        ("기타투자판단과관련한중요사항", "기타 중요사항"),
        ("관련공시", "관련공시"),
    )
    for needle, name in aliases:
        if needle in compact:
            return name
    return normalize_label_name(label)


def extract_field_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    if not isinstance(record, dict):
        return []

    pairs: list[tuple[str, str]] = []
    ignored = {"title", "question", "answer", "system_prompt", "summary", "text", "completion"}
    for key, value in record.items():
        if str(key).lower() in ignored or str(key).startswith("_"):
            continue
        if isinstance(value, (str, int, float)):
            text = normalize_text(value)
            if text:
                pairs.append((canonical_field_name(key), text))
    return pairs


def looks_like_html_summary_record(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False

    title = record.get("title") or record.get("question")
    question = record.get("question")
    if not title or not question:
        return False

    if str(question).strip() != str(title).strip():
        return False

    field_count = sum(
        1
        for key in record
        if str(key).lower() not in {"title", "question", "answer", "summary", "text", "completion"}
    )
    return field_count >= 3


def build_natural_qa_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Expand one extracted record into many natural instruction-tuning Q/A pairs."""
    title = normalize_text(record.get("title") or record.get("question"))
    fields = extract_field_pairs(record)
    if not fields:
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_pair(
        question: str,
        answer: str,
        evidence: list[tuple[str, str]] | None = None,
    ) -> None:
        q = re.sub(r"\s+", " ", question).strip()
        a = re.sub(r"\s+", " ", answer).strip()
        if not q or not a:
            return
        evidence_items = evidence
        if evidence_items is None:
            evidence_items = [(label, value) for label, value in fields if value and value in a][:6]
        document = title or normalize_text(record.get("_source_path")) or "입력 문서"
        source_path = normalize_text(record.get("_source_path"))
        evidence_lines = [f"- 문서: {document}"]
        if source_path:
            evidence_lines.append(f"- 출처: {source_path}")
        for label, value in evidence_items:
            evidence_lines.append(f"- {label}: {value}")
        a = f"답변: {a}\n근거:\n" + "\n".join(evidence_lines)
        key = (q.lower(), a.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append((q, a))

    field_map = {label: value for label, value in fields}
    title_parts = [part.strip() for part in title.split("/") if part.strip()]
    company = title_parts[0] if title_parts else "해당 회사"
    document_type = title_parts[1] if len(title_parts) > 1 else "공시"

    def choose_variant(options: tuple[str, ...], salt: str) -> str:
        """Choose reproducibly so the corpus varies without random output changes."""
        score = sum((idx + 1) * ord(char) for idx, char in enumerate(title + salt))
        return options[score % len(options)]

    def format_value(label: str, value: str) -> str:
        if label in {"계약금액", "최근매출액"} and re.fullmatch(r"[\d,]+", value):
            return value + "원"
        if label == "매출액 대비" and re.fullmatch(r"[\d.]+", value):
            return value + "%"
        return value

    start = field_map.get("계약 시작일", "")
    end = field_map.get("계약 종료일", "")
    period = f"{start} ~ {end}" if start and end else start or end

    summary_labels = (
        "계약 구분", "체결계약명", "계약금액", "계약상대",
        "판매·공급지역", "계약일자",
    )
    summary_parts = [
        f"{label}: {format_value(label, field_map[label])}"
        for label in summary_labels
        if field_map.get(label) and field_map[label] != "-"
    ]
    if period:
        summary_parts.append(f"계약기간: {period}")
    generic_parts = [
        f"{label}: {format_value(label, value)}"
        for label, value in fields
        if value and value != "-" and label not in {"기타 중요사항", "관련공시"}
    ]
    summary = "; ".join(summary_parts or generic_parts[:8])

    if summary:
        summary_question = choose_variant((
            f"{company}의 {document_type} 내용을 요약해줘.",
            f"{company}가 공시한 {document_type}의 핵심 내용은 무엇인가요?",
            f"{company}의 이번 공시에서 중요한 내용을 간단히 정리해줘.",
            f"{document_type} 공시의 주요 정보를 항목별로 알려줘.",
            f"투자자가 확인해야 할 {company}의 이번 공시 내용을 정리해줘.",
        ), "summary")
        add_pair(summary_question, summary)
    if field_map.get("계약금액"):
        amount_question = choose_variant((
            "계약금액은 얼마인가요?",
            f"{company}가 체결한 계약의 규모를 알려줘.",
            "이번 공급계약의 금액은 얼마야?",
            "공시된 계약금액을 원화 기준으로 알려주세요.",
        ), "계약금액")
        add_pair(amount_question, format_value("계약금액", field_map["계약금액"]))
    if field_map.get("계약상대"):
        counterparty_question = choose_variant((
            "계약상대는 누구인가요?",
            f"{company}는 어느 회사와 계약했나요?",
            "이번 계약의 상대방을 알려줘.",
            "공시에 나온 계약상대가 어디인지 알려주세요.",
        ), "계약상대")
        add_pair(counterparty_question, field_map["계약상대"])
    if period:
        period_question = choose_variant((
            "계약기간은 언제인가요?",
            "이번 계약은 언제부터 언제까지인가요?",
            "계약 시작일과 종료일을 알려줘.",
            f"{company}가 공시한 계약의 수행 기간은 어떻게 되나요?",
        ), "계약기간")
        add_pair(period_question, period)
    if summary:
        detail = field_map.get("기타 중요사항", "")
        answer = summary + (f". {detail}" if detail and detail != "-" else "")
        detail_question = choose_variant((
            "핵심 정보를 정리해줘.",
            "이 공시의 주요 조건과 참고사항을 함께 설명해줘.",
            f"{company}의 이번 공시를 자세히 설명해주세요.",
        ), "detail")
        add_pair(detail_question, answer[:1200])

    # Every usable field can become a grounded extraction question. This also
    # supports disclosure types not explicitly known by this script.
    handled = {"계약금액", "계약상대", "계약 시작일", "계약 종료일"}
    for label, value in fields:
        if label in handled or not value or value == "-":
            continue
        if label == "기타 중요사항":
            question_options = (
                "공시에서 추가로 설명한 중요사항은 무엇인가요?",
                f"{company}가 밝힌 기타 투자 참고사항을 설명해줘.",
                "이 공시를 해석할 때 함께 봐야 할 내용은 무엇인가요?",
            )
        elif "금액" in label or "매출액" in label:
            question_options = (
                f"{label}은 얼마인가요?",
                f"공시에 기재된 {label}을 알려줘.",
                f"{company}의 이번 공시에서 {label} 규모는 얼마야?",
            )
        elif "대비" in label or "비율" in label or "%" in label:
            question_options = (
                f"{label}는 어느 정도인가요?",
                f"공시 기준 {label} 비율을 알려줘.",
                f"{label} 수치는 몇 퍼센트인가요?",
            )
        elif any(token in label for token in ("일자", "기간", "시작일", "종료일")):
            question_options = (
                f"{label}은 언제인가요?",
                f"공시에 나온 {label}을 알려줘.",
                f"{label}이 어떻게 되나요?",
            )
        elif "지역" in label or "장소" in label:
            question_options = (
                f"{label}은 어디인가요?",
                f"이번 공시와 관련된 {label}을 알려줘.",
                f"{company}의 {label}은 어디야?",
            )
        elif "여부" in label:
            question_options = (
                f"{label}를 확인해줘.",
                f"공시상 {label}는 어떻게 되나요?",
                f"{label}에 해당하는지 알려줘.",
            )
        else:
            question_options = (
                f"{label}은 무엇인가요?",
                f"공시에 기재된 {label}을 알려줘.",
                f"{company}의 이번 {document_type}에서 {label}은 어떻게 되나요?",
            )
        add_pair(choose_variant(question_options, label), format_value(label, value))

    return pairs


def iter_messages_from_record(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Return list of (text, completion) pairs from a single record."""
    # Prefer generated Q/A pairs for HTML table summaries because they contain a title + many field labels.
    if looks_like_html_summary_record(record):
        generated = build_natural_qa_pairs(record)
        if generated:
            return generated

    # 1) direct object fields
    direct_pairs = [
        ("text", "completion"),
        ("input", "output"),
        ("question", "answer"),
        ("prompt", "completion"),
        ("user", "assistant"),
        ("instruction", "output"),
        ("query", "response"),
        ("text", "response"),
    ]
    for text_key, completion_key in direct_pairs:
        text_value = get_first_value(record, [text_key])
        completion_value = get_first_value(record, [completion_key])
        if text_value and completion_value:
            return [(text_value, completion_value)]

    # 2) messages / conversation format
    for key in ("messages", "dialogue", "conversation", "turns", "qas"):
        items = record.get(key)
        if not isinstance(items, list):
            continue

        turns: list[tuple[str, str]] = []
        pending_user: str = ""
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("speaker") or "").lower()
            content = normalize_text(item.get("content") or item.get("text") or item.get("message") or item.get("value"))
            if not content:
                continue
            if role in {"user", "human", "question", "q", "input"}:
                pending_user = content
            elif role in {"assistant", "bot", "ai", "answer", "a", "output"}:
                if pending_user:
                    turns.append((pending_user, content))
                    pending_user = ""
        if turns:
            return turns

    # 3) nested data structures such as {"data": [{"question": ..., "answer": ...}]}
    for key in ("data", "items", "records", "examples", "samples"):
        items = record.get(key)
        if isinstance(items, list):
            turns: list[tuple[str, str]] = []
            for item in items:
                if isinstance(item, dict):
                    nested = iter_messages_from_record(item)
                    turns.extend(nested)
            if turns:
                return turns

    # 4) HTML/XML table extraction fallback: build many natural Q/A pairs from fields.
    generated = build_natural_qa_pairs(record)
    if generated:
        return generated

    return []


def xml_element_to_dict(element: ET.Element) -> dict[str, Any]:
    """Convert an XML node to a dict of child tag names to text values."""
    result: dict[str, Any] = {}
    for child in list(element):
        tag = child.tag.split("}")[-1].lower()
        value = (child.text or "").strip()
        if child:
            nested = xml_element_to_dict(child)
            if nested:
                result[tag] = nested
                continue
        result[tag] = value
    if not result:
        text = (element.text or "").strip()
        if text:
            result["text"] = text
    return result


def iter_xml_items(root: ET.Element) -> Iterable[ET.Element]:
    tag = root.tag.split("}")[-1].lower()
    item_like = {"item", "record", "row", "qa", "example", "sample", "dialog", "turn", "message"}
    if tag in item_like or any(k in tag for k in ("item", "record", "row", "qa", "example", "sample")):
        yield root
    for child in list(root):
        yield from iter_xml_items(child)


def load_records(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        records = load_json_like_records(input_path)
    elif suffix == ".xml":
        records = load_xml_records(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .json, .jsonl, or .xml.")
    for record in records:
        record.setdefault("_source_path", str(input_path))
    return records


def load_input_records(input_path: Path) -> list[dict[str, Any]]:
    """Load one supported file or every supported file below a directory."""
    if input_path.is_file():
        return load_records(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    paths = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".xml"}
    )
    for path in paths:
        try:
            records.extend(load_records(path))
        except (OSError, ValueError, ET.ParseError) as exc:
            print(f"Skipped {path}: {exc}", file=sys.stderr)
    return records


def load_json_like_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}: {exc}") from exc
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append(item)
    elif isinstance(data, dict):
        # Common wrapper forms: {"data": [...]} or {"examples": [...]}
        for key in ("data", "items", "records", "examples", "samples"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    if isinstance(item, dict):
                        records.append(item)
                return records
        records.append(data)
    return records


def html_to_records(raw_text: str) -> list[dict[str, Any]]:
    """Handle malformed HTML/XForms documents that are not valid XML but still contain structured data."""
    text = raw_text or ""
    text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!\?[^>]*\?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    if BeautifulSoup is not None:
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""

        rows: list[tuple[str, str]] = []
        pending_heading = ""
        for tr in soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) >= 2:
                # For merged disclosure tables, the last cell is the value
                # and the preceding cell is the actual field label.
                label = cells[-2]
                value = cells[-1]
                rows.append((label, value))
                pending_heading = ""
            elif len(cells) == 1:
                # Some long fields use one row for the heading and the next
                # row for its value (for example, "기타 중요사항").
                if pending_heading:
                    rows.append((pending_heading, cells[0]))
                    pending_heading = ""
                else:
                    pending_heading = cells[0]

        if rows:
            record: dict[str, Any] = {}
            for label, value in rows:
                if label and value:
                    key = canonical_field_name(label)
                    if key in record:
                        key = f"{key} {len(record)}"
                    record[key] = value

            if title:
                record["title"] = title
                record["question"] = title
                record["answer"] = "\n".join(f"{label}: {value}" for label, value in rows if value)
            return [record]

        if title:
            return [{"question": title, "answer": soup.get_text(" ", strip=True)}]

        return [{"question": "html_doc", "answer": soup.get_text(" ", strip=True)}]

    # Fallback: regex-based extraction for HTML-like documents without BeautifulSoup.
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    title = unescape(re.sub(r"<.*?>", "", title_match.group(1))) if title_match else ""

    rows: list[tuple[str, str]] = []
    pending_heading = ""
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.IGNORECASE | re.DOTALL)
        clean_cells = [
            unescape(re.sub(r"\s+", " ", re.sub(r"<.*?>", " ", cell))).strip()
            for cell in cells
        ]
        if len(clean_cells) >= 2:
            label = clean_cells[-2]
            value = clean_cells[-1]
            if label and value:
                rows.append((label, value))
            pending_heading = ""
        elif len(clean_cells) == 1 and clean_cells[0]:
            if pending_heading:
                rows.append((pending_heading, clean_cells[0]))
                pending_heading = ""
            else:
                pending_heading = clean_cells[0]

    if rows:
        record: dict[str, Any] = {}
        for label, value in rows:
            key = canonical_field_name(label)
            if key in record:
                key = f"{key} {len(record)}"
            record[key] = value
        if title:
            record["title"] = title
            record["question"] = title
            record["answer"] = "\n".join(f"{label}: {value}" for label, value in rows if value)
        return [record]

    if title:
        return [{"question": title, "answer": re.sub(r"<.*?>", " ", text)}]
    return [{"question": "html_doc", "answer": re.sub(r"<.*?>", " ", text)}]


def load_xml_records(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        lower = raw.lower()
        if any(token in lower for token in ("<html", "<body", "<table", "<div class=\"xforms\"", "<style>", "<br")):
            return html_to_records(raw)

        tree = ET.parse(path)
        root = tree.getroot()
        records: list[dict[str, Any]] = []

        if root.tag and "item" in root.tag.lower():
            records.append(xml_element_to_dict(root))
            return records

        for elem in iter_xml_items(root):
            item = xml_element_to_dict(elem)
            if item:
                records.append(item)
        return records
    except ET.ParseError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return html_to_records(raw)


def record_metadata(record: dict[str, Any]) -> tuple[str, str, str]:
    title = normalize_text(record.get("title") or record.get("question"))
    parts = [part.strip() for part in title.split("/") if part.strip()]
    company = parts[0] if parts else ""
    document_type = parts[1] if len(parts) > 1 else "공시"
    date_match = re.search(r"(20\d{2})[.\-/](\d{2})[.\-/](\d{2})", title)
    date = "-".join(date_match.groups()) if date_match else ""
    return company, document_type, date


def numeric_value(value: str) -> float | None:
    cleaned = re.sub(r"[^\d.+-]", "", value.replace(",", ""))
    if not cleaned or cleaned in {"+", "-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def format_number(value: float, label: str) -> str:
    number = f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")
    if "금액" in label or "매출액" in label:
        return number + "원"
    if "대비" in label or "비율" in label:
        return number + "%"
    return number


def grounded_completion(
    answer: str,
    evidence: list[tuple[dict[str, Any], list[tuple[str, str]]]],
) -> str:
    lines = [f"답변: {answer}", "근거:"]
    for record, items in evidence:
        title = normalize_text(record.get("title") or record.get("question")) or "입력 문서"
        lines.append(f"- 문서: {title}")
        source_path = normalize_text(record.get("_source_path"))
        if source_path:
            lines.append(f"- 출처: {source_path}")
        for label, value in items:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def ensure_grounded_completion(record: dict[str, Any], completion: str) -> str:
    """Apply the answer/evidence contract to imported Q/A records as well."""
    if "답변:" in completion and "근거:" in completion:
        return completion
    fields = extract_field_pairs(record)
    matched = [(label, value) for label, value in fields if value and value in completion][:6]
    if not matched:
        original_answer = normalize_text(record.get("answer") or record.get("completion"))
        if original_answer:
            matched = [("원문 답변", original_answer)]
    return grounded_completion(completion, [(record, matched)])


def build_cross_document_pairs(
    records: list[dict[str, Any]],
    max_pairs: int = 1000,
) -> list[tuple[str, str]]:
    """Create grounded comparison/calculation and temporal-reasoning examples."""
    pairs: list[tuple[str, str]] = []
    structured: list[tuple[dict[str, Any], str, str, str, dict[str, str]]] = []
    for record in records:
        company, document_type, date = record_metadata(record)
        fields = {
            label: value for label, value in extract_field_pairs(record)
            if value and value != "-" and len(value) <= 500
        }
        if company and date and fields:
            structured.append((record, company, document_type, date, fields))

    # Complex-document reasoning: compare two dates for the same company/type.
    temporal_groups: dict[tuple[str, str], list[tuple[dict[str, Any], str, dict[str, str]]]] = {}
    for record, company, document_type, date, fields in structured:
        temporal_groups.setdefault((company, document_type), []).append((record, date, fields))
    for (company, document_type), items in temporal_groups.items():
        items.sort(key=lambda item: item[1])
        for (old_record, old_date, old_fields), (new_record, new_date, new_fields) in zip(items, items[1:]):
            changed = [
                label for label in old_fields.keys() & new_fields.keys()
                if old_fields[label] != new_fields[label]
                and label not in {"기타 중요사항", "관련공시"}
            ][:6]
            if not changed:
                continue
            changes = "; ".join(
                f"{label}: {old_fields[label]} → {new_fields[label]}" for label in changed
            )
            question = (
                f"{company}의 {old_date}와 {new_date} {document_type} 공시를 비교하면 "
                "주요 내용이 어떻게 달라졌나요?"
            )
            evidence = [
                (old_record, [(label, old_fields[label]) for label in changed]),
                (new_record, [(label, new_fields[label]) for label in changed]),
            ]
            pairs.append((question, grounded_completion(changes, evidence)))
            if len(pairs) >= max_pairs:
                return pairs

            numeric_labels = [
                label for label in changed
                if numeric_value(old_fields[label]) is not None and numeric_value(new_fields[label]) is not None
            ]
            if numeric_labels:
                label = numeric_labels[0]
                old_number = numeric_value(old_fields[label])
                new_number = numeric_value(new_fields[label])
                assert old_number is not None and new_number is not None
                difference = new_number - old_number
                direction = "증가" if difference > 0 else "감소" if difference < 0 else "변동 없음"
                answer = (
                    f"{label}은 {old_date} {format_number(old_number, label)}에서 "
                    f"{new_date} {format_number(new_number, label)}로 "
                    f"{format_number(abs(difference), label)} {direction}했습니다."
                )
                question = (
                    f"{company}의 두 {document_type} 공시에서 {label}은 얼마나 변했나요? "
                    "계산 과정의 기준값도 함께 알려줘."
                )
                pairs.append((question, grounded_completion(answer, evidence)))
                if len(pairs) >= max_pairs:
                    return pairs

    # Multi-lookup: compare the same numeric field across companies in the same year/type.
    comparison_groups: dict[tuple[str, str], list[tuple[dict[str, Any], str, str, dict[str, str]]]] = {}
    for record, company, document_type, date, fields in structured:
        comparison_groups.setdefault((date[:4], document_type), []).append((record, company, date, fields))
    for (year, document_type), items in comparison_groups.items():
        items.sort(key=lambda item: (item[1], item[2]))
        for left, right in zip(items, items[1:]):
            left_record, left_company, left_date, left_fields = left
            right_record, right_company, right_date, right_fields = right
            if left_company == right_company:
                continue
            common_numeric = [
                label for label in left_fields.keys() & right_fields.keys()
                if numeric_value(left_fields[label]) is not None
                and numeric_value(right_fields[label]) is not None
                and any(token in label for token in ("금액", "매출액", "비율", "대비", "투자"))
            ]
            if not common_numeric:
                continue
            label = common_numeric[0]
            left_number = numeric_value(left_fields[label])
            right_number = numeric_value(right_fields[label])
            assert left_number is not None and right_number is not None
            winner = left_company if left_number > right_number else right_company if right_number > left_number else "두 회사"
            difference = abs(left_number - right_number)
            answer = (
                f"{left_company}는 {format_number(left_number, label)}, "
                f"{right_company}는 {format_number(right_number, label)}입니다. "
                f"{winner}의 값이 더 크며 차이는 {format_number(difference, label)}입니다."
            )
            question = (
                f"{year}년 {left_company}와 {right_company}의 {document_type} 공시를 기준으로 "
                f"{label}을 비교하면 어느 회사가 더 크고 차이는 얼마인가요?"
            )
            evidence = [
                (left_record, [(label, left_fields[label]), ("공시일", left_date)]),
                (right_record, [(label, right_fields[label]), ("공시일", right_date)]),
            ]
            pairs.append((question, grounded_completion(answer, evidence)))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_rows(
    records: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_cross_document: bool = True,
    max_cross_pairs: int = 1000,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    conversation_id = 1

    for record in records:
        turns = iter_messages_from_record(record)
        if not turns:
            continue

        row_system_prompt = extract_system_prompt(record) or system_prompt
        for turn_idx, (text, completion) in enumerate(turns, start=1):
            row: dict[str, str | int] = {
                "C_ID": conversation_id,
                "T_ID": turn_idx,
                "Text": text,
                "Completion": ensure_grounded_completion(record, completion),
            }
            if row_system_prompt:
                row["System_Prompt"] = row_system_prompt
            rows.append(row)
        conversation_id += 1

    if include_cross_document:
        for text, completion in build_cross_document_pairs(records, max_pairs=max_cross_pairs):
            row = {
                "C_ID": conversation_id,
                "T_ID": 1,
                "Text": text,
                "Completion": completion,
            }
            if system_prompt:
                row["System_Prompt"] = system_prompt
            rows.append(row)
            conversation_id += 1

    return rows


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_system = any("System_Prompt" in row for row in rows)
    fieldnames = SYSTEM_COLUMNS if has_system else REQUIRED_COLUMNS

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(row_out)


def write_jsonl(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_system = any("System_Prompt" in row for row in rows)
    fieldnames = SYSTEM_COLUMNS if has_system else REQUIRED_COLUMNS

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {key: row.get(key, "") for key in fieldnames}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON/XML datasets to HyperCLOVA X tuning format.")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to one JSON/JSONL/XML file or a directory containing them.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV/JSONL file.")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format.")
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt stored in the first column. Pass an empty string to omit it.",
    )
    parser.add_argument(
        "--no-cross-document", action="store_true",
        help="Do not generate multi-document comparison/calculation examples.",
    )
    parser.add_argument(
        "--max-cross-pairs", type=int, default=1000,
        help="Maximum number of multi-document examples to generate.",
    )
    args = parser.parse_args()

    try:
        records = load_input_records(args.input)
        rows = build_rows(
            records,
            system_prompt=args.system_prompt,
            include_cross_document=not args.no_cross_document,
            max_cross_pairs=max(0, args.max_cross_pairs),
        )
        if not rows:
            print(f"No valid Q&A rows were found in {args.input}. Check the input structure.")
            return

        if args.format == "csv":
            write_csv(rows, args.output)
        else:
            write_jsonl(rows, args.output)

        print(f"Converted {len(records)} input records -> {len(rows)} rows")
        print(f"Output file: {args.output}")
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
