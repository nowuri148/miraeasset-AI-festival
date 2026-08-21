#!/usr/bin/env python3
"""Generate grounded HyperCLOVA X instruction data from JSON/XML documents.

Unlike a_dataset_try1.py, this script asks an LLM to create diverse questions.
Every training Text contains the source context, and every Completion contains
an answer plus quotes that were verified to exist in that context.

Examples:
  python a_dataset_try2.py --input corpus/raw/exchange --output Dataset/train.csv
  python a_dataset_try2.py --input document.xml --output Dataset/train.jsonl --format jsonl

Environment:
  CLOVA_STUDIO_API_KEY  CLOVA Studio test/service API key
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import CLOVA_STUDIO_API_KEY  # noqa: E402
from a_dataset_try1 import extract_field_pairs, load_input_records, normalize_text


COLUMNS = ["C_ID", "T_ID", "Text", "Completion"]
SYSTEM_COLUMNS = ["System_Prompt", *COLUMNS]
DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"
DEFAULT_MODEL = "HCX-005"
GENERATOR_SYSTEM_PROMPT = """당신은 기업 공시 기반 금융 QA 학습데이터 설계자다.
주어진 Context에서만 질문과 정답을 만들고 외부 지식이나 추측을 사용하지 않는다.
질문은 실제 투자자가 할 법한 한국어로 작성하고, 서로 다른 사고 과정을 요구해야 한다.
근거 quote는 Context에 존재하는 문자열을 글자 그대로 복사한다.
JSON Schema에 맞는 JSON만 출력한다."""
TARGET_SYSTEM_PROMPT = (
    "제공된 문서를 근거로 질문에 정확히 답하세요. 추측하지 말고, 답변 뒤에 사용한 근거를 표시하세요."
)

TASK_GUIDE_SINGLE = """
- 사실 추출: 사람/회사/금액/날짜/지역/계약명/목적/사유
- 조건 결합: 두 개 이상의 필드를 함께 확인하는 질문
- 계산: 기간 일수, 비율 검산, 금액 단위 변환 등 Context로 계산 가능한 질문
- 요약: 핵심 내용 또는 투자자가 확인할 조건을 선별
- 근거 부족: Context에 없는 정보를 요구하고 확인 불가라고 답하는 질문
- 표현 다양성: 존댓말, 구어체, 짧은 검색어형, 정식 질의형을 고르게 사용
"""

TASK_GUIDE_MULTI = """
- 다중 조회: 여러 문서에서 같은 항목을 찾아 나열
- 비교·연산: 차이, 합계, 평균, 최대·최소, 순위, 증감액·증감률
- 복합 문서 추론: 시점별 변화, 체결-정정-해지 관계, 공통점과 차이점
- 조건 검색: 기간·회사·지역·임계값 등 여러 조건을 만족하는 문서 선별
- 근거 부족: 문서들만으로 결론 낼 수 없는 질문을 식별
계산 질문의 answer에는 사용한 값과 계산 결과를 함께 적는다.
"""


@dataclass
class ContextBundle:
    context: str
    source_ids: list[str]
    mode: str


def safe_text(value: Any, limit: int = 2000) -> str:
    return normalize_text(value)[:limit]


def record_title(record: dict[str, Any]) -> str:
    return safe_text(record.get("title") or record.get("question") or record.get("_source_path") or "문서")


def record_to_context(record: dict[str, Any], index: int, max_chars: int) -> tuple[str, str]:
    title = record_title(record)
    # _source_path는 내부 추적용 source_id로만 사용한다.
    # 실제 tuning Text에는 로컬 파일 경로를 노출하지 않는다.
    source = safe_text(record.get("_source_path"), 1000)
    fields = extract_field_pairs(record)

    lines = [f"[문서 {index}]", f"문서명: {title}"]
    for label, value in fields:
        value = safe_text(value, 3000)
        if value:
            lines.append(f"{label}: {value}")

    if len(lines) <= 2:
        raw_answer = safe_text(record.get("answer") or record.get("text") or record, 5000)
        if raw_answer:
            lines.append(f"본문: {raw_answer}")

    block = "\n".join(lines)
    return block[:max_chars], source or title


def make_bundles(
    records: list[dict[str, Any]],
    max_context_chars: int,
    multi_doc_size: int,
    include_multi: bool,
) -> list[ContextBundle]:
    singles: list[ContextBundle] = []
    rendered: list[tuple[str, str, str]] = []
    for record in records:
        block, source_id = record_to_context(record, 1, max_context_chars)
        company = record_title(record).split("/")[0].strip()
        rendered.append((block, source_id, company))
        singles.append(ContextBundle(block, [source_id], "single"))

    if not include_multi or multi_doc_size < 2:
        return singles

    multi: list[ContextBundle] = []
    by_company: dict[str, list[tuple[str, str, str]]] = {}
    for item in rendered:
        by_company.setdefault(item[2], []).append(item)
    for items in by_company.values():
        for start in range(0, len(items) - 1, multi_doc_size):
            chunk = items[start : start + multi_doc_size]
            if len(chunk) < 2:
                continue
            blocks = []
            sources = []
            for idx, (block, source_id, _) in enumerate(chunk, start=1):
                blocks.append(re.sub(r"^\[문서 1\]", f"[문서 {idx}]", block))
                sources.append(source_id)
            context = "\n\n".join(blocks)[:max_context_chars]
            multi.append(ContextBundle(context, sources, "multi"))
    return singles + multi


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


def generation_prompt(bundle: ContextBundle, count: int) -> str:
    guide = TASK_GUIDE_MULTI if bundle.mode == "multi" else TASK_GUIDE_SINGLE
    return f"""아래 Context로 파인튜닝용 질문-답변 {count}개를 생성하라.

[생성 원칙]
1. 질문끼리 핵심 의도와 필요한 연산이 중복되지 않게 한다.
2. 답할 수 있는 질문은 Context의 정확한 사실만 사용한다.
3. answerable=true이면 evidence를 1개 이상 제공한다.
4. evidence.quote는 반드시 Context의 연속된 원문 문자열이어야 한다.
5. answerable=false인 질문은 전체의 10~20%로 하고, answer에는 어떤 정보가 부족한지 쓴다.
6. 단순 동의어 치환으로 개수를 채우지 않는다.
7. 질문과 답변만 읽어도 수치의 단위와 비교 기준이 명확해야 한다.
8. evidence.document에는 반드시 Context의 표기 그대로 "문서 1", "문서 2"처럼 적는다.
9. evidence.field에는 Context의 실제 필드명만 적는다.
10. evidence.quote에는 해당 근거가 있는 Context의 연속된 원문 문자열을 그대로 복사한다.
11. 한 문서에서 단순 필드 조회 질문만 반복하지 말고 조건 결합·계산·요약·근거 부족을 섞는다.

[필수 작업 유형]
{guide}

[Context]
{bundle.context}
"""


class ClovaClient:
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

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": 0.8,
            "max_tokens": max_tokens,
            "response_format": response_schema(),
        }
        last_error: Exception | None = None
        schema_enabled = True
        for attempt in range(self.retries + 2):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return parse_json_content(content)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
                # Some CLOVA models/accounts reject response_format even on
                # the OpenAI-compatible endpoint. The prompt still requires
                # JSON, so retry once without JSON Schema.
                if exc.code == 400 and schema_enabled:
                    payload.pop("response_format", None)
                    schema_enabled = False
                    continue
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 10))
            except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"CLOVA Studio generation failed: {last_error}")


def parse_json_content(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_question(question: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", question.casefold())


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


def normalize_evidence_field(field: str, quote: str) -> str:
    """field가 비어 있거나 부정확하면 '필드명: 값' 형태의 quote에서 필드명을 복구한다."""
    field = safe_text(field, 200).strip()
    if field:
        return field
    if ":" in quote:
        return quote.split(":", 1)[0].strip()
    return "본문"


def validate_qas(data: dict[str, Any], bundle: ContextBundle) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    raw_qas = data.get("qas") or data.get("questions") or []
    if not isinstance(raw_qas, list):
        return [], ["response has neither a qas nor questions array"]
    for idx, qa in enumerate(raw_qas):
        if not isinstance(qa, dict):
            errors.append(f"qa[{idx}] is not an object")
            continue
        question = safe_text(qa.get("question") or qa.get("text"), 2000)
        answer = safe_text(qa.get("answer"), 4000)
        task_type = safe_text(qa.get("task_type") or qa.get("type") or "정보추출", 30)
        answerable = bool(qa.get("answerable"))
        evidence = qa.get("evidence") if isinstance(qa.get("evidence"), list) else []
        key = normalize_question(question)
        if not question or not answer or len(question) < 4 or key in seen:
            errors.append(f"qa[{idx}] empty or duplicate")
            continue
        verified = []
        for item in evidence:
            if isinstance(item, str):
                quote = safe_text(item, 1500)
                field = ""
            elif isinstance(item, dict):
                quote = safe_text(item.get("quote"), 1500)
                field = safe_text(item.get("field"), 200)
            else:
                continue

            if not quote:
                continue

            # 모델이 적은 document 문자열을 그대로 믿지 않고,
            # 실제 quote가 존재하는 Context 문서를 코드에서 판별한다.
            document = locate_evidence_document(bundle.context, quote)
            if document is None:
                continue

            verified.append({
                "document": document,
                "field": normalize_evidence_field(field, quote),
                "quote": quote,
            })
        if answerable and not verified:
            errors.append(f"qa[{idx}] has no verifiable evidence")
            continue
        if not answerable and not any(token in answer for token in ("확인", "부족", "알 수", "제공되지", "포함")):
            errors.append(f"qa[{idx}] unanswerable response is unclear")
            continue
        seen.add(key)
        valid.append({
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "evidence": verified,
        })
    return valid, errors


def training_text(question: str, context: str, include_context: bool) -> str:
    if not include_context:
        return question
    return f"{context}\n\n[질문]\n{question}"


def training_completion(qa: dict[str, Any]) -> str:
    lines = [f"답변: {qa['answer']}", "근거:"]

    if qa["evidence"]:
        for item in qa["evidence"]:
            document = item["document"]
            field = item["field"].strip()
            quote = item["quote"].strip()

            # quote 자체가 이미 "필드명: 값"이면 field를 한 번 더 붙이지 않는다.
            if field and re.match(rf"^{re.escape(field)}\s*:", quote):
                evidence_text = quote
            elif field and field != "본문":
                evidence_text = f"{field}: {quote}"
            else:
                evidence_text = quote

            lines.append(f"- {document} | {evidence_text}")
    else:
        lines.append("- 제공된 문서에서 답변에 필요한 정보를 확인할 수 없음")

    return "\n".join(lines)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
                cache[item["key"]] = item["result"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def append_cache(path: Path, key: str, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")


def write_rows(rows: list[dict[str, Any]], output: Path, output_format: str, system_prompt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = SYSTEM_COLUMNS if system_prompt else COLUMNS
    if output_format == "csv":
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    else:
        with output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps({column: row.get(column, "") for column in columns}, ensure_ascii=False) + "\n")


def build_rows(
    bundles: Iterable[ContextBundle],
    client: ClovaClient,
    single_count: int,
    multi_count: int,
    max_tokens: int,
    temperature: float,
    include_context: bool,
    system_prompt: str,
    cache_path: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    all_errors: list[str] = []
    cache = load_cache(cache_path)
    global_seen: set[str] = set()
    for bundle_index, bundle in enumerate(bundles):
        if limit and bundle_index >= limit:
            break
        count = multi_count if bundle.mode == "multi" else single_count
        prompt = generation_prompt(bundle, count)
        key = hashlib.sha256((client.model + prompt).encode("utf-8")).hexdigest()
        if key in cache:
            generated = cache[key]
        else:
            generated = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            append_cache(cache_path, key, generated)
        qas, errors = validate_qas(generated, bundle)
        all_errors.extend(f"bundle[{bundle_index}] {error}" for error in errors)
        for qa in qas:
            dedupe_key = normalize_question(qa["question"])
            if dedupe_key in global_seen:
                continue
            global_seen.add(dedupe_key)
            row: dict[str, Any] = {
                "C_ID": len(rows),
                "T_ID": 0,
                "Text": training_text(qa["question"], bundle.context, include_context),
                "Completion": training_completion(qa),
            }
            if system_prompt:
                row["System_Prompt"] = system_prompt
            if len(row["Text"]) + len(row["Completion"]) > 8000:
                all_errors.append(f"bundle[{bundle_index}] row exceeds 8000 characters")
                continue
            rows.append(row)
    return rows, all_errors


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate LLM-authored grounded HyperCLOVA X tuning data.")
    parser.add_argument("--input", type=Path, required=True, help="JSON/XML file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output .csv or .jsonl")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    parser.add_argument(
        "--api-key",
        default=os.getenv("CLOVA_STUDIO_API_KEY", CLOVA_STUDIO_API_KEY),
        help="Defaults to CLOVA_STUDIO_API_KEY from the environment or Config.py",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions-per-document", type=int, default=4)
    parser.add_argument("--questions-per-multi-context", type=int, default=4)
    parser.add_argument("--multi-doc-size", type=int, default=3)
    parser.add_argument("--no-multi-document", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=5500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Limit context bundles; 0 means all")
    parser.add_argument("--cache", type=Path, default=Path("Dataset/.qa_generation_cache.jsonl"))
    parser.add_argument("--without-context", action="store_true", help="Omit source context from Text (not recommended)")
    parser.add_argument("--system-prompt", default="", help="Optional first CSV column")
    parser.add_argument("--dry-run", action="store_true", help="Print the first generation prompt without calling the API")
    args = parser.parse_args()

    if args.output.suffix.lower() not in {".csv", ".jsonl"}:
        parser.error("--output must end in .csv or .jsonl")
    output_format = "jsonl" if args.output.suffix.lower() == ".jsonl" else args.format

    records = load_input_records(args.input)
    bundles = make_bundles(
        records,
        max_context_chars=max(1000, args.max_context_chars),
        multi_doc_size=max(2, args.multi_doc_size),
        include_multi=not args.no_multi_document,
    )
    if not bundles:
        raise RuntimeError("No usable JSON/XML records were found.")
    if args.dry_run:
        first = bundles[0]
        count = args.questions_per_multi_context if first.mode == "multi" else args.questions_per_document
        print(generation_prompt(first, max(1, count)))
        return
    if not args.api_key:
        parser.error("Set CLOVA_STUDIO_API_KEY or pass --api-key.")
    client = ClovaClient(args.api_key, args.base_url, args.model, args.timeout, args.retries)
    rows, errors = build_rows(
        bundles,
        client,
        single_count=max(1, args.questions_per_document),
        multi_count=max(1, args.questions_per_multi_context),
        max_tokens=args.max_tokens,
        temperature=min(1.0, max(0.0, args.temperature)),
        include_context=not args.without_context,
        system_prompt=args.system_prompt,
        cache_path=args.cache,
        limit=max(0, args.limit),
    )
    error_path = args.output.with_suffix(args.output.suffix + ".errors.log")
    if errors:
        error_path.write_text("\n".join(errors), encoding="utf-8")
    if not rows:
        detail = f" Validation details: {error_path}" if errors else ""
        raise RuntimeError(f"No valid rows were generated.{detail}")
    write_rows(rows, args.output, output_format, args.system_prompt)
    print(f"Loaded records: {len(records)}")
    print(f"Context bundles: {len(bundles)}")
    print(f"Valid training rows: {len(rows)}")
    print(f"Rejected items: {len(errors)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()