from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


COLUMNS = ["C_ID", "T_ID", "Text", "Completion"]
SYSTEM_COLUMNS = ["System_Prompt", *COLUMNS]
REQUIRED_COLUMNS = ["C_ID", "T_ID", "Text", "Completion"]

DEFAULT_SYSTEM_PROMPT = (
    "기업 공시 질문에 정확히 답하세요. 제공된 근거에서 확인되는 사실만 사용하고, "
    "답변 뒤에 반드시 문서와 사용한 원문 항목을 근거로 제시하세요. "
    "근거가 부족하면 추측하지 말고 확인할 수 없다고 답하세요."
)

# =====================================================================
# Original a_dataset_try1 parsing / normalization code
# =====================================================================

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

# =====================================================================
# Shared context helpers
# =====================================================================

@dataclass
class ContextBundle:
    context: str
    source_ids: list[str]
    mode: str  # "single" or "multi"

def safe_text(
    value: Any,
    limit: int = 2000,
) -> str:
    return normalize_text(value)[:limit]

def record_title(
    record: dict[str, Any],
) -> str:
    return safe_text(
        record.get("title")
        or record.get("question")
        or record.get("_source_path")
        or "문서"
    )

def record_to_context(
    record: dict[str, Any],
    index: int,
    max_chars: int,
) -> tuple[str, str]:
    title = record_title(record)

    # Local file path is only an internal source ID.
    source = safe_text(
        record.get("_source_path"),
        1000,
    )

    fields = extract_field_pairs(record)

    lines = [
        f"[문서 {index}]",
        f"문서명: {title}",
    ]

    for label, value in fields:
        value = safe_text(
            value,
            3000,
        )

        if value:
            lines.append(
                f"{label}: {value}"
            )

    if len(lines) <= 2:
        raw_answer = safe_text(
            record.get("answer")
            or record.get("text")
            or record,
            5000,
        )
        if raw_answer:
            lines.append(
                f"본문: {raw_answer}"
            )

    block = "\n".join(lines)

    return (
        block[:max_chars],
        source or title,
    )

def make_bundles(
    records: list[dict[str, Any]],
    max_context_chars: int,
    multi_doc_size: int,
    include_multi: bool,
) -> list[ContextBundle]:
    """
    Generic bundle builder.

    Single-document generation:
        include_multi=False

    Simple same-company multi-document generation:
        include_multi=True

    More advanced comparison grouping should remain in a dedicated
    comparison module, not here.
    """
    singles: list[ContextBundle] = []
    rendered: list[
        tuple[str, str, str]
    ] = []

    for record in records:
        block, source_id = record_to_context(
            record,
            1,
            max_context_chars,
        )

        company = (
            record_title(record)
            .split("/")[0]
            .strip()
        )

        rendered.append(
            (
                block,
                source_id,
                company,
            )
        )

        singles.append(
            ContextBundle(
                context=block,
                source_ids=[source_id],
                mode="single",
            )
        )

    if (
        not include_multi
        or multi_doc_size < 2
    ):
        return singles

    by_company: dict[
        str,
        list[tuple[str, str, str]],
    ] = {}

    for item in rendered:
        by_company.setdefault(
            item[2],
            [],
        ).append(item)

    multi: list[ContextBundle] = []

    for items in by_company.values():
        for start in range(
            0,
            len(items) - 1,
            multi_doc_size,
        ):
            chunk = items[
                start:start + multi_doc_size
            ]

            if len(chunk) < 2:
                continue

            blocks: list[str] = []
            sources: list[str] = []

            for idx, (
                block,
                source_id,
                _,
            ) in enumerate(
                chunk,
                start=1,
            ):
                blocks.append(
                    re.sub(
                        r"^\[문서 1\]",
                        f"[문서 {idx}]",
                        block,
                    )
                )
                sources.append(source_id)

            context = "\n\n".join(
                blocks
            )[:max_context_chars]

            multi.append(
                ContextBundle(
                    context=context,
                    source_ids=sources,
                    mode="multi",
                )
            )

    return singles + multi

# =====================================================================
# Shared cache / output helpers
# =====================================================================

def load_cache(
    path: Path,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}

    if not path.exists():
        return cache

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            try:
                item = json.loads(line)
                cache[item["key"]] = item[
                    "result"
                ]
            except (
                json.JSONDecodeError,
                KeyError,
            ):
                continue

    return cache

def append_cache(
    path: Path,
    key: str,
    result: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                {
                    "key": key,
                    "result": result,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

def write_rows(
    rows: list[dict[str, Any]],
    output: Path,
    output_format: str,
    system_prompt: str,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = (
        SYSTEM_COLUMNS
        if system_prompt
        else COLUMNS
    )

    if output_format == "csv":
        with output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
            )
            writer.writeheader()

            writer.writerows(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in columns
                }
                for row in rows
            )

        return

    if output_format != "jsonl":
        raise ValueError(
            "output_format must be "
            "'csv' or 'jsonl'"
        )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    {
                        column: row.get(
                            column,
                            "",
                        )
                        for column in columns
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

# =====================================================================
# Shared HyperCLOVA X engine (requests-based working version)
# =====================================================================

DEFAULT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_financial_qa",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "qas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_type": {"type": "string"},
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "answerable": {"type": "boolean"},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "document": {"type": "string"},
                                        "field": {"type": "string"},
                                        "quote": {"type": "string"},
                                    },
                                    "required": [
                                        "document",
                                        "field",
                                        "quote",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "task_type",
                            "question",
                            "answer",
                            "answerable",
                            "evidence",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["qas"],
            "additionalProperties": False,
        },
    },
}

def parse_json_content(
    content: str | dict[str, Any] | list[Any],
) -> dict[str, Any] | list[Any]:
    """Parse HCX JSON, including ```json fenced responses."""
    if isinstance(content, (dict, list)):
        return content

    text = content.strip()
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "HyperCLOVA X 응답을 JSON으로 파싱하지 못했습니다.\n"
            f"원본 응답:\n{content}"
        ) from exc

def normalize_question(question: str) -> str:
    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        question.casefold(),
    )


# =====================================================================
# Shared QA / evidence / training utilities
# =====================================================================

ALLOWED_GENERATED_TASK_TYPES = {
    "정보추출",
    "조건결합",
    "계산",
    "요약",
    "다중조회",
    "비교연산",
    "복합추론",
    "근거부족",
}

TASK_TYPE_ALIASES = {
    # 사실 조회 계열
    "사실 추출": "정보추출",
    "사실추출": "정보추출",
    "정보 추출": "정보추출",
    "정보조회": "정보추출",
    "사실 조회": "정보추출",

    # 조건 결합
    "조건 결합": "조건결합",

    # 비교/연산
    "비교 연산": "비교연산",
    "비교": "비교연산",

    # 복합 추론
    "복합 추론": "복합추론",
    "복합 문서 추론": "복합추론",

    # 근거 부족
    "근거 부족": "근거부족",
    "답변불가": "근거부족",
    "답변 불가": "근거부족",
}

def response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grounded_financial_qa",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "qas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_type": {
                                    "type": "string",
                                    "enum": [
                                        "정보추출", "조건결합", "계산", "요약",
                                        "다중조회", "비교연산", "복합추론", "근거부족",
                                    ],
                                },
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "answerable": {"type": "boolean"},
                                "evidence": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "document": {"type": "string"},
                                            "field": {"type": "string"},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["document", "field", "quote"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["task_type", "question", "answer", "answerable", "evidence"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["qas"],
                "additionalProperties": False,
            },
        },
    }

def split_context_documents(context: str) -> list[tuple[str, str]]:
    """Context를 ("문서 N", 문서본문) 목록으로 분리한다."""
    matches = list(re.finditer(r"(?m)^\[문서 (\d+)\]\s*$", context))
    documents: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
        documents.append((f"문서 {match.group(1)}", context[start:end].strip()))
    return documents

def locate_evidence_document(context: str, quote: str) -> str | None:
    """quote가 실제로 들어 있는 문서 번호를 Context에서 직접 찾는다."""
    for document, body in split_context_documents(context):
        if quote in body:
            return document
    return None

def normalize_field_name_for_match(value: str) -> str:
    """필드명 비교용 정규화."""
    text = (value or "").strip().casefold()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text

def find_context_line_by_field(
    context: str,
    field: str,
    preferred_document: str = "",
) -> tuple[str, str] | None:
    """
    field와 일치하는 실제 Context 라인을 찾는다.

    preferred_document가 있으면 해당 문서에서 먼저 찾고,
    찾지 못하면 전체 문서를 다시 탐색한다.
    """

    field_key = normalize_field_name_for_match(field)

    if not field_key:
        return None

    generic_fields = {
        "본문",
        "내용",
        "기타",
        "정보",
        "근거",
    }

    if field.strip() in generic_fields:
        return None

    documents = split_context_documents(context)

    # --------------------------------------------------------
    # 1. reviewer가 지정한 문서에서 먼저 탐색
    # --------------------------------------------------------
    preferred_document = normalize_text(
        preferred_document
    )

    if preferred_document:
        for document, body in documents:
            if (
                normalize_text(document)
                != preferred_document
            ):
                continue

            for line in body.splitlines():
                stripped = line.strip()

                if ":" not in stripped:
                    continue

                line_field, _ = stripped.split(
                    ":",
                    1,
                )

                if (
                    normalize_field_name_for_match(
                        line_field
                    )
                    == field_key
                ):
                    return document, stripped

    # --------------------------------------------------------
    # 2. 지정 문서에서 못 찾으면 전체 문서 탐색
    # --------------------------------------------------------
    for document, body in documents:
        for line in body.splitlines():
            stripped = line.strip()

            if ":" not in stripped:
                continue

            line_field, _ = stripped.split(
                ":",
                1,
            )

            if (
                normalize_field_name_for_match(
                    line_field
                )
                == field_key
            ):
                return document, stripped

    return None

def normalize_evidence_field(field: str, quote: str) -> str:
    """field가 비어 있거나 부정확하면 '필드명: 값' 형태의 quote에서 필드명을 복구한다."""
    field = safe_text(field, 200).strip()
    if field:
        return field
    if ":" in quote:
        return quote.split(":", 1)[0].strip()
    return "본문"

def repair_evidence_item(
    context: str,
    document: str,
    field: str,
    quote: str,
) -> dict[str, str] | None:
    """
    evidence를 deterministic하게 검증/복구한다.

    탐색 순서:
    1. reviewer가 지정한 document 안에서 quote 확인
    2. 전체 Context에서 quote 확인
    3. 지정 document에서 field 기반 원문 라인 복구
    4. 전체 Context에서 field 기반 원문 라인 복구

    따라서 single-document 기존 동작도 유지된다.
    """

    document = safe_text(
        document,
        50,
    ).strip()

    field = safe_text(
        field,
        200,
    ).strip()

    quote = safe_text(
        quote,
        1500,
    ).strip()

    documents = split_context_documents(context)

    # --------------------------------------------------------
    # 1. 지정된 문서 안에서 quote 직접 확인
    # --------------------------------------------------------
    if document and quote:
        normalized_document = normalize_text(
            document
        )

        for actual_document, body in documents:
            if (
                normalize_text(actual_document)
                != normalized_document
            ):
                continue

            if quote in body:
                return {
                    "document": actual_document,
                    "field": normalize_evidence_field(
                        field,
                        quote,
                    ),
                    "quote": quote,
                }

    # --------------------------------------------------------
    # 2. 지정 문서에서 못 찾았어도 전체 Context에서 quote 탐색
    #
    # single 호환성을 위해 반드시 fallback 유지
    # --------------------------------------------------------
    if quote:
        located_document = locate_evidence_document(
            context,
            quote,
        )

        if located_document is not None:
            return {
                "document": located_document,
                "field": normalize_evidence_field(
                    field,
                    quote,
                ),
                "quote": quote,
            }

    # --------------------------------------------------------
    # 3~4. field 기반 복구
    #
    # preferred_document에서 먼저 찾고
    # 실패하면 함수 내부에서 전체 문서를 탐색한다.
    # --------------------------------------------------------
    repaired = find_context_line_by_field(
        context,
        field,
        preferred_document=document,
    )

    if repaired is None:
        return None

    repaired_document, repaired_quote = repaired

    return {
        "document": repaired_document,
        "field": normalize_evidence_field(
            field,
            repaired_quote,
        ),
        "quote": repaired_quote,
    }

def normalize_task_type(value: str) -> str:
    text = safe_text(value, 30).strip()

    if text in ALLOWED_GENERATED_TASK_TYPES:
        return text

    return TASK_TYPE_ALIASES.get(text, text)

def normalize_qas_container(
    data: dict[str, Any] | list[Any],
) -> list[Any]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        raw_qas = (
            data.get("qas")
            or data.get("questions")
            or []
        )
        return raw_qas if isinstance(raw_qas, list) else []

    return []

def task_type_diversity_summary(
    qas: list[dict[str, Any]],
) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}

    for qa in qas:
        task_type = str(
            qa.get("task_type") or ""
        ).strip()

        if not task_type:
            continue

        counts[task_type] = counts.get(task_type, 0) + 1

    return len(counts), counts


# ---------------------------------------------------------------------------
# Answer grounding checks
# ---------------------------------------------------------------------------

SUPPORTED_NUMERIC_UNITS = (
    "억원",
    "만원",
    "천원",
    "원",
    "달러",
    "USD",
    "%",
    "대",
    "주",
    "개",
    "배",
    "명",
    "건",
)


def normalize_numeric_token(value: str) -> str:
    """숫자 비교용 정규화: 97,000.00 -> 97000"""
    cleaned = (value or "").replace(",", "").strip()

    try:
        if "." in cleaned:
            number = float(cleaned)
            if number.is_integer():
                return str(int(number))
            return cleaned.rstrip("0").rstrip(".")
        return str(int(cleaned))
    except ValueError:
        return cleaned


def extract_numeric_tokens(text: str) -> set[str]:
    """텍스트의 숫자 토큰을 비교 가능한 형태로 추출한다."""
    if not text:
        return set()

    values = re.findall(
        r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?",
        text,
    )

    return {
        normalize_numeric_token(value)
        for value in values
        if normalize_numeric_token(value)
    }


def extract_number_unit_pairs(text: str) -> set[tuple[str, str]]:
    """
    숫자에 직접 붙은 단위를 추출한다.

    예:
      97,000원 -> ("97000", "원")
      5.37% -> ("5.37", "%")
      3,500대 -> ("3500", "대")
      USD 78,856,650 -> ("78856650", "USD")
    """
    if not text:
        return set()

    pairs: set[tuple[str, str]] = set()

    unit_pattern = "|".join(
        sorted(
            (re.escape(unit) for unit in SUPPORTED_NUMERIC_UNITS),
            key=len,
            reverse=True,
        )
    )

    # 숫자 뒤 단위: 5.37%, 3,500대, 1,062억원
    suffix_pattern = re.compile(
        rf"(\d[\d,]*(?:\.\d+)?)\s*({unit_pattern})",
        flags=re.IGNORECASE,
    )

    for match in suffix_pattern.finditer(text):
        number = normalize_numeric_token(match.group(1))
        unit = match.group(2)
        if unit.upper() == "USD":
            unit = "USD"
        pairs.add((number, unit))

    # 단위가 숫자 앞에 오는 대표적인 경우: USD 78,856,650
    prefix_pattern = re.compile(
        r"\b(USD)\s*(\d[\d,]*(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )

    for match in prefix_pattern.finditer(text):
        number = normalize_numeric_token(match.group(2))
        pairs.add((number, "USD"))

    return pairs


def validate_answer_units(
    answer: str,
    evidence: list[dict[str, Any]],
) -> tuple[bool, set[tuple[str, str]]]:
    """
    answer에 숫자+단위가 있으면 동일 숫자+단위가 evidence.quote에도
    직접 존재하는지 확인한다.

    숫자만 답하는 경우에는 이 검사에서 제한하지 않는다.
    """
    answer_pairs = extract_number_unit_pairs(answer)

    if not answer_pairs:
        return True, set()

    evidence_text = "\n".join(
        str(item.get("quote", ""))
        for item in evidence
    )

    evidence_pairs = extract_number_unit_pairs(evidence_text)

    unsupported = answer_pairs - evidence_pairs

    return not unsupported, unsupported


CALCULATION_QUESTION_TOKENS = (
    "계산",
    "환산",
    "합계",
    "평균",
    "차이",
    "증감",
    "몇 배",
    "비율을 구",
)

CALCULATION_ANSWER_MARKERS = (
    "=",
    "+",
    "×",
    "x",
    "÷",
    "/",
    "계산",
    "합계",
    "평균",
    "차이",
    "증가",
    "감소",
)


def looks_like_fake_calculation(
    question: str,
    answer: str,
    task_type: str,
    evidence: list[dict[str, Any]],
) -> bool:
    """
    '계산/환산'을 요구했는데 실제 계산 없이 evidence의 숫자 하나를
    그대로 답한 경우를 보수적으로 탐지한다.

    실제 계산 결과가 evidence에 없는 정상 계산 QA는 이 조건에 걸리지 않는다.
    """
    asks_calculation = (
        task_type == "계산"
        or any(token in question for token in CALCULATION_QUESTION_TOKENS)
    )

    if not asks_calculation:
        return False

    answer_numbers = extract_numeric_tokens(answer)
    if len(answer_numbers) != 1:
        return False

    evidence_text = "\n".join(
        str(item.get("quote", ""))
        for item in evidence
    )
    evidence_numbers = extract_numeric_tokens(evidence_text)

    # 결과 숫자가 근거에 그대로 없다면 실제 연산으로 생성된 값일 수 있으므로 허용.
    if not answer_numbers.issubset(evidence_numbers):
        return False

    # 답변 자체에 계산 과정/관계 표현이 있으면 허용.
    if any(marker in answer for marker in CALCULATION_ANSWER_MARKERS):
        return False

    return True


def validate_reviewed_qas_minimal(
    data: dict[str, Any] | list[Any],
    bundle: ContextBundle,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    2차 HCX 검수 후에는 의미 판단을 다시 Python에서 과도하게 하지 않는다.

    남기는 검증:
    - 기본 필드 존재
    - task_type 허용값 정규화
    - 중복 질문 제거
    - answerable=true이면 evidence 최소 1개
    - evidence.quote가 Context에 실제 존재
    - evidence.document는 quote 위치 기준으로 코드가 재결정
    - answer의 숫자+단위는 evidence에 동일하게 존재해야 함
    - 계산/환산 질문이 단순 원문 숫자 복사로 끝나면 제외
    """
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    raw_qas = normalize_qas_container(
        data
    )

    for idx, qa in enumerate(raw_qas):
        if not isinstance(qa, dict):
            errors.append(
                f"qa[{idx}] is not an object"
            )
            continue

        question = safe_text(
            qa.get("question"),
            2000,
        )
        answer = safe_text(
            qa.get("answer"),
            4000,
        )

        raw_task_type = str(
            qa.get("task_type")
            or qa.get("type")
            or "정보추출"
        )
        task_type = normalize_task_type(
            raw_task_type
        )

        if task_type not in ALLOWED_GENERATED_TASK_TYPES:
            errors.append(
                f"qa[{idx}] unsupported task_type: "
                f"{raw_task_type}"
            )
            continue

        if not question or not answer:
            errors.append(
                f"qa[{idx}] empty question/answer"
            )
            continue

        key = normalize_question(
            question
        )
        if not key or key in seen:
            errors.append(
                f"qa[{idx}] duplicate question"
            )
            continue

        answerable = bool(
            qa.get("answerable")
        )

        raw_evidence = (
            qa.get("evidence")
            if isinstance(
                qa.get("evidence"),
                list,
            )
            else []
        )

        verified = []

        for item in raw_evidence:
            # evidence마다 값 초기화
            document = ""
            field = ""
            quote = ""

            if isinstance(item, str):
                quote = safe_text(
                    item,
                    1500,
                )

            elif isinstance(item, dict):
                document = safe_text(
                    item.get("document"),
                    50,
                )

                quote = safe_text(
                    item.get("quote"),
                    1500,
                )

                field = safe_text(
                    item.get("field"),
                    200,
                )

            else:
                continue

            repaired_item = repair_evidence_item(
                bundle.context,
                document,
                field,
                quote,
            )

            if repaired_item is None:
                continue

            verified.append(
                repaired_item
            )

        if answerable and not verified:
            errors.append(
                f"qa[{idx}] has no verifiable evidence"
            )
            continue

        if answerable:
            units_ok, unsupported_units = validate_answer_units(
                answer,
                verified,
            )

            if not units_ok:
                formatted_units = sorted(
                    f"{number}{unit}"
                    for number, unit in unsupported_units
                )
                errors.append(
                    f"qa[{idx}] answer contains unsupported units: "
                    f"{formatted_units}"
                )
                continue

            if looks_like_fake_calculation(
                question,
                answer,
                task_type,
                verified,
            ):
                errors.append(
                    f"qa[{idx}] calculation question appears to "
                    "copy an evidence value without an actual calculation"
                )
                continue

        seen.add(key)

        valid.append({
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "evidence": verified,
        })

        if len(valid) >= target_count:
            break

    return valid, errors

def training_text(question: str, context: str, include_context: bool) -> str:
    if not include_context:
        return question
    return f"{context}\n\n[질문]\n{question}"

def format_evidence_text(
    field: str,
    quote: str,
) -> str:
    """
    evidence.field와 quote의 중복을 방지한다.

    예:
      field="관계"
      quote="회사와의 관계: 자회사"
        -> "회사와의 관계: 자회사"

      field="회사와의 관계"
      quote="회사와의 관계: 자회사"
        -> "회사와의 관계: 자회사"

      field="계약금액"
      quote="97,000,000,000"
        -> "계약금액: 97,000,000,000"
    """
    field = (field or "").strip()
    quote = (quote or "").strip()

    if not quote:
        return ""

    if not field or field == "본문":
        return quote

    # quote 자체가 이미 "어떤 필드명: 값" 형태면 quote를 그대로 사용한다.
    # 모델이 field를 "관계"처럼 짧게 줘도
    # "관계: 회사와의 관계: 자회사" 같은 중복을 만들지 않는다.
    if re.match(r"^[^:\n]{1,100}\s*:", quote):
        return quote

    # quote가 field로 이미 시작하는 경우도 그대로 사용
    if re.match(
        rf"^{re.escape(field)}\s*[:：]?",
        quote,
    ):
        return quote

    return f"{field}: {quote}"

def training_completion(qa: dict[str, Any]) -> str:
    lines = [f"답변: {qa['answer']}", "근거:"]

    if qa["evidence"]:
        for item in qa["evidence"]:
            document = item["document"]
            field = item["field"].strip()
            quote = item["quote"].strip()

            evidence_text = format_evidence_text(
                field,
                quote,
            )

            lines.append(
                f"- {document} | {evidence_text}"
            )
    else:
        lines.append("- 제공된 문서에서 답변에 필요한 정보를 확인할 수 없음")

    return "\n".join(lines)

class ClovaClient:
    """Shared HyperCLOVA X OpenAI-compatible client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        retries: int,
    ) -> None:
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        use_response_format: bool = True,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "top_p": 0.8,
            "max_tokens": max_tokens,
        }

        if use_response_format:
            payload["response_format"] = (
                response_schema
                or DEFAULT_RESPONSE_SCHEMA
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        schema_enabled = use_response_format

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Some HCX deployments do not accept response_format.
                if (
                    response.status_code == 400
                    and schema_enabled
                    and "response_format" in payload
                ):
                    print(
                        "[WARN] response_format 요청이 거부되어 "
                        "JSON Schema 없이 다시 시도합니다."
                    )
                    payload.pop("response_format", None)
                    schema_enabled = False

                    response = requests.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )

                response.raise_for_status()

                result = response.json()
                content = (
                    result["choices"][0]
                    ["message"]["content"]
                )

                return parse_json_content(content)

            except (
                requests.RequestException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

            if attempt >= self.retries:
                break

            wait_seconds = min(
                2 ** attempt,
                10,
            )
            print(
                f"[WARN] API 호출 실패. "
                f"{wait_seconds}초 후 재시도합니다. "
                f"({attempt + 1}/{self.retries})"
            )
            time.sleep(wait_seconds)

        raise RuntimeError(
            "CLOVA Studio generation failed: "
            f"{last_error}"
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        return self.call_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_response_format=True,
            response_schema=response_schema,
        )

    def review(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        return self.call_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            use_response_format=True,
            response_schema=response_schema,
        )

def build_rows(
    bundles: Iterable[ContextBundle],
    client: ClovaClient,
    *,
    single_count: int,
    multi_count: int,
    max_tokens: int,
    temperature: float,
    include_context: bool,
    system_prompt: str,
    cache_path: Path,
    limit: int,
    generation_prompt_fn: Callable[[ContextBundle, int], str],
    review_prompt_fn: Callable[
        [ContextBundle, list[dict[str, Any]], int],
        str,
    ],
    normalize_qas_fn: Callable[
        [dict[str, Any] | list[Any]],
        list[Any],
    ],
    validate_reviewed_fn: Callable[
        [dict[str, Any] | list[Any], ContextBundle, int],
        tuple[list[dict[str, Any]], list[str]],
    ],
    training_text_fn: Callable[[str, str, bool], str],
    training_completion_fn: Callable[[dict[str, Any]], str],
    generator_system_prompt: str,
    reviewer_system_prompt: str,
    response_schema: dict[str, Any] | None = None,
    generator_cache_version: str = "GENERATOR_COMMON_V1",
    reviewer_cache_version: str = "REVIEWER_COMMON_V1",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Generic two-stage HCX dataset generation engine.

    Task-specific prompts and validation are injected by callers such as
    a_dataset_single.py or a_dataset_compare.py.
    """
    rows: list[dict[str, Any]] = []
    all_errors: list[str] = []
    cache = load_cache(cache_path)
    global_seen: set[str] = set()

    for bundle_index, bundle in enumerate(bundles):
        if limit and bundle_index >= limit:
            break

        target_count = (
            multi_count
            if bundle.mode == "multi"
            else single_count
        )

        if target_count <= 0:
            continue

        candidate_count = target_count + 2

        generator_prompt = generation_prompt_fn(
            bundle,
            candidate_count,
        )

        generator_key = hashlib.sha256(
            (
                generator_cache_version
                + "|"
                + client.model
                + "|"
                + generator_prompt
            ).encode("utf-8")
        ).hexdigest()

        if generator_key in cache:
            generated = cache[generator_key]
        else:
            print(
                f"[bundle {bundle_index}] "
                f"1차 HCX 후보 생성: "
                f"{candidate_count}개 요청"
            )
            generated = client.generate(
                generator_prompt,
                system_prompt=generator_system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_schema=response_schema,
            )
            append_cache(
                cache_path,
                generator_key,
                generated,
            )

        raw_candidates = normalize_qas_fn(
            generated
        )

        if not raw_candidates:
            all_errors.append(
                f"bundle[{bundle_index}] "
                "generator returned no QA candidates"
            )
            continue

        reviewer_prompt = review_prompt_fn(
            bundle,
            raw_candidates,
            target_count,
        )

        review_input_hash = json.dumps(
            raw_candidates,
            ensure_ascii=False,
            sort_keys=True,
        )

        reviewer_key = hashlib.sha256(
            (
                reviewer_cache_version
                + "|"
                + client.model
                + "|"
                + bundle.context
                + "|"
                + review_input_hash
                + "|"
                + str(target_count)
                + "|"
                + reviewer_prompt
            ).encode("utf-8")
        ).hexdigest()

        if reviewer_key in cache:
            reviewed = cache[reviewer_key]
        else:
            print(
                f"[bundle {bundle_index}] "
                f"2차 HCX 검수: "
                f"최종 {target_count}개 요청"
            )
            reviewed = client.review(
                reviewer_prompt,
                system_prompt=reviewer_system_prompt,
                max_tokens=max_tokens,
                response_schema=response_schema,
            )
            append_cache(
                cache_path,
                reviewer_key,
                reviewed,
            )

        qas, errors = validate_reviewed_fn(
            reviewed,
            bundle,
            target_count,
        )

        all_errors.extend(
            f"bundle[{bundle_index}] {error}"
            for error in errors
        )

        if len(qas) < target_count:
            all_errors.append(
                f"bundle[{bundle_index}] "
                f"reviewer returned only "
                f"{len(qas)}/{target_count} "
                "valid QAs after validation"
            )

        for qa in qas:
            dedupe_key = normalize_question(
                qa["question"]
            )

            if dedupe_key in global_seen:
                continue

            global_seen.add(dedupe_key)

            row: dict[str, Any] = {
                "C_ID": len(rows),
                "T_ID": 0,
                "Text": training_text_fn(
                    qa["question"],
                    bundle.context,
                    include_context,
                ),
                "Completion": training_completion_fn(
                    qa
                ),
            }

            if system_prompt:
                row["System_Prompt"] = (
                    system_prompt
                )

            if (
                len(row["Text"])
                + len(row["Completion"])
                > 8000
            ):
                all_errors.append(
                    f"bundle[{bundle_index}] "
                    "row exceeds 8000 characters"
                )
                continue

            rows.append(row)

    return rows, all_errors


def validate_comparison_qas(
    reviewed: dict[str, Any] | list[Any],
    bundle: ContextBundle,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    비교 QA 전용 validator.

    1. 공통 validate_reviewed_qas_minimal() 먼저 적용
    2. factual answer인데 evidence 없음 -> reject
    3. 비교/다중문서 질문인데 한 문서만 근거로 사용 -> reject
    4. 사실 답변을 하면서 answerable=False -> reject
    5. multi dataset인데 사실상 single-document 질문 -> reject
    """

    # --------------------------------------------------------
    # 1. 공통 validator 먼저 적용
    # --------------------------------------------------------
    base_qas, base_errors = validate_reviewed_qas_minimal(
        reviewed,
        bundle,
        target_count,
    )

    errors = list(base_errors)
    valid_qas: list[dict[str, Any]] = []

    # Context 안에 실제 몇 개 문서가 있는지 확인
    document_ids = set(
        re.findall(
            r"\[문서\s*(\d+)\]",
            bundle.context,
        )
    )

    multi_document_context = len(document_ids) >= 2

    for idx, qa in enumerate(base_qas):
        question = normalize_text(
            qa.get("question")
        )
        answer = normalize_text(
            qa.get("answer")
        )

        answerable = bool(
            qa.get("answerable", True)
        )

        evidence = qa.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        # ----------------------------------------------------
        # 2. answerable=False인데 실제 factual answer가 있으면 reject
        # ----------------------------------------------------
        no_answer_markers = {
            "",
            "확인할 수 없음",
            "알 수 없음",
            "근거 부족",
            "제공된 문서에서 확인할 수 없음",
            "제공된 문서에서 답변에 필요한 정보를 확인할 수 없음",
        }

        normalized_answer = answer.strip()

        if (
            not answerable
            and normalized_answer
            and normalized_answer not in no_answer_markers
        ):
            errors.append(
                f"compare qa[{idx}] answerable=False "
                "but contains a factual answer"
            )
            continue

        # ----------------------------------------------------
        # 3. factual answer인데 evidence가 없으면 reject
        # ----------------------------------------------------
        if answerable and not evidence:
            errors.append(
                f"compare qa[{idx}] factual answer has no evidence"
            )
            continue

        # ----------------------------------------------------
        # 4. evidence가 '근거 없음' sentinel이면 reject
        # ----------------------------------------------------
        evidence_quotes = [
            normalize_text(
                item.get("quote")
                if isinstance(item, dict)
                else ""
            )
            for item in evidence
        ]

        invalid_evidence_markers = (
            "제공된 문서에서 답변에 필요한 정보를 확인할 수 없음",
            "제공된 문서에서 확인할 수 없음",
            "근거 없음",
            "알 수 없음",
        )

        if answerable and any(
            any(
                marker in quote
                for marker in invalid_evidence_markers
            )
            for quote in evidence_quotes
        ):
            errors.append(
                f"compare qa[{idx}] uses invalid no-evidence marker"
            )
            continue

        # ----------------------------------------------------
        # 5. evidence가 몇 개 문서를 사용하는지 확인
        # ----------------------------------------------------
        evidence_documents: set[str] = set()

        for item in evidence:
            if not isinstance(item, dict):
                continue

            document = normalize_text(
                item.get("document")
            )

            match = re.search(
                r"문서\s*(\d+)",
                document,
            )

            if match:
                evidence_documents.add(
                    match.group(1)
                )

        # ----------------------------------------------------
        # 6. 질문이 실제 비교/다중문서 질문인지 판별
        # ----------------------------------------------------
        comparison_tokens = (
            "비교",
            "각",
            "두 ",
            "두 문서",
            "두 계약",
            "각각",
            "더 큰",
            "더 작은",
            "차이",
            "어느",
            "어떤 계약",
            "변화",
            "증가",
            "감소",
            "높은",
            "낮은",
            "동일",
            "다른",
            "차이는",
        )

        looks_multi_question = any(
            token in question
            for token in comparison_tokens
        )

        # ----------------------------------------------------
        # 7. 비교 질문인데 한 문서만 evidence로 사용하면 reject
        # ----------------------------------------------------
        if (
            multi_document_context
            and looks_multi_question
            and len(evidence_documents) < 2
        ):
            errors.append(
                f"compare qa[{idx}] comparison question "
                "does not use evidence from at least 2 documents"
            )
            continue

        # ----------------------------------------------------
        # 8. multi dataset인데 사실상 single-document 정보추출이면 reject
        # ----------------------------------------------------
        explicit_single_tokens = (
            "문서 1",
            "문서1",
            "첫 번째 문서",
            "문서 2",
            "문서2",
            "두 번째 문서",
        )

        explicitly_single = any(
            token in question
            for token in explicit_single_tokens
        )

        if (
            multi_document_context
            and not looks_multi_question
            and len(evidence_documents) <= 1
            and not explicitly_single
        ):
            errors.append(
                f"compare qa[{idx}] appears to be "
                "a single-document question"
            )
            continue

                # ----------------------------------------------------
        # 질문에서 명시적으로 요구한 Context field가
        # evidence에 모두 포함되어 있는지 검사
        # ----------------------------------------------------
        context_field_labels = set()

        for line in bundle.context.splitlines():
            stripped = line.strip()

            if ":" not in stripped:
                continue

            label, _ = stripped.split(
                ":",
                1,
            )

            label = normalize_text(label)

            if label:
                context_field_labels.add(label)

        # 질문에 실제로 언급된 field만 추출
        requested_fields: set[str] = set()

        for label in context_field_labels:
            # 띄어쓰기 차이를 조금 허용
            normalized_label = re.sub(
                r"\s+",
                "",
                label,
            )
            normalized_question = re.sub(
                r"\s+",
                "",
                question,
            )

            if normalized_label in normalized_question:
                requested_fields.add(label)

        # 실제 evidence에서 사용한 field
        evidence_fields: set[str] = set()

        for item in evidence:
            if not isinstance(item, dict):
                continue

            field = normalize_text(
                item.get("field")
            )

            if field:
                evidence_fields.add(field)

        # field 명칭의 띄어쓰기 차이를 제거해서 비교
        normalized_requested = {
            re.sub(r"\s+", "", field)
            for field in requested_fields
        }

        normalized_evidence = {
            re.sub(r"\s+", "", field)
            for field in evidence_fields
        }

        missing_fields = (
            normalized_requested
            - normalized_evidence
        )

        if missing_fields:
            errors.append(
                f"compare qa[{idx}] missing evidence "
                f"for requested fields: "
                f"{sorted(missing_fields)}"
            )
            continue

        # ----------------------------------------------------
        # 계약상대와 국가/국적/소재지를 임의로 연결한 질문 차단
        # ----------------------------------------------------
        compact_question = re.sub(
            r"\s+",
            "",
            question,
        )

        party_country_patterns = (
            ("계약상대", "국가"),
            ("계약상대", "국적"),
            ("계약상대", "소재지"),
            ("계약상대", "어느나라"),
        )

        invalid_party_country_relation = any(
            left in compact_question
            and right in compact_question
            for left, right in party_country_patterns
        )

        if invalid_party_country_relation:
            errors.append(
                f"compare qa[{idx}] infers country/nationality/"
                "location of contract counterparty without "
                "an explicit source field"
            )
            continue

        valid_qas.append(qa)

    return valid_qas, errors