from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag


# ============================================================
# SETTINGS
# ============================================================

MIN_CHARS = 100
TARGET_CHARS = 600
MAX_CHARS = 900

# 다음 chunk에 직전 마지막 문장 1개 정도만 중복
OVERLAP_SENTENCES = 1

SECTION_TAG_RE = re.compile(r"^section-\d+$", re.I)

# 예:
# (1) 식품 사업
# (2) 바이오 사업
# 1) 생산실적
# 가. 시장위험
# 나. 위험관리 정책
SUBTITLE_RE = re.compile(
    r"^\s*(?:"
    r"\(\d+\)"          # (1)
    r"|\d+\)"           # 1)
    r"|[가-힣]\."       # 가.
    r"|[가-힣]\)"       # 가)
    r")\s*"
)


# ============================================================
# BASIC TEXT UTILS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def tag_text(tag: Tag | None) -> str:
    if tag is None:
        return ""

    return clean_text(tag.get_text(" ", strip=True))


def is_section(tag: Tag) -> bool:
    return bool(
        tag.name
        and SECTION_TAG_RE.match(tag.name)
    )


def direct_title(section: Tag) -> str:
    """
    현재 SECTION의 직계 TITLE만 가져온다.
    하위 SECTION의 TITLE이 섞이지 않도록 한다.
    """

    for child in section.children:

        if (
            isinstance(child, Tag)
            and child.name
            and child.name.lower() == "title"
        ):
            return tag_text(child)

    return ""


# ============================================================
# DOCUMENT META
# ============================================================

def extract_document_meta(
    xml_path: Path,
) -> dict[str, Any]:

    raw = xml_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    company = tag_text(
        soup.find("company-name")
    )

    document_name = tag_text(
        soup.find("document-name")
    )

    period_from = ""
    period_to = ""

    from_tag = soup.find(
        attrs={"aunit": "PERIODFROM"}
    )

    to_tag = soup.find(
        attrs={"aunit": "PERIODTO"}
    )

    if from_tag is not None:

        period_from = clean_text(
            from_tag.get("aunitvalue")
            or tag_text(from_tag)
        )

    if to_tag is not None:

        period_to = clean_text(
            to_tag.get("aunitvalue")
            or tag_text(to_tag)
        )

    rcept_no = re.sub(
        r"\D",
        "",
        xml_path.stem,
    )

    return {
        "corp_name": company,
        "document_name": document_name,
        "rcept_no": rcept_no,
        "period_from": period_from,
        "period_to": period_to,
    }


# ============================================================
# SENTENCE SPLIT
# ============================================================

def split_sentences(
    text: str,
) -> list[str]:

    text = clean_text(text)

    if not text:
        return []

    # --------------------------------------------------------
    # 1차: 일반 문장부호 기준
    # --------------------------------------------------------

    parts = re.split(
        r"(?<=[.!?。])\s+",
        text,
    )

    result: list[str] = []

    for part in parts:

        part = clean_text(part)

        if not part:
            continue

        # ----------------------------------------------------
        # 너무 긴 문장인 경우 한국어 종결어미 기준 추가 분할
        # ----------------------------------------------------

        if len(part) > MAX_CHARS:

            subparts = re.split(
                r"(?<=다\.)\s*"
                r"|(?<=니다\.)\s*"
                r"|(?<=습니다\.)\s*",
                part,
            )

            subparts = [
                clean_text(x)
                for x in subparts
                if clean_text(x)
            ]

            if len(subparts) > 1:

                result.extend(
                    subparts
                )

                continue

        result.append(
            part
        )

    # --------------------------------------------------------
    # 2차:
    # 그래도 단일 문장이 MAX_CHARS보다 길면
    # 공백/쉼표 위치를 이용해 강제 분할
    # --------------------------------------------------------

    final_result: list[str] = []

    for sentence in result:

        if len(sentence) <= MAX_CHARS:

            final_result.append(
                sentence
            )

            continue

        start = 0

        while start < len(sentence):

            end = min(
                start + MAX_CHARS,
                len(sentence),
            )

            piece = sentence[
                start:end
            ]

            # 마지막 조각이 아니면 가능한 자연스러운 위치 탐색
            if end < len(sentence):

                candidates = [
                    piece.rfind(". "),
                    piece.rfind(", "),
                    piece.rfind("; "),
                    piece.rfind(" "),
                ]

                cut = max(
                    candidates
                )

                # 너무 앞에서 자르는 것은 방지
                if cut > int(
                    MAX_CHARS * 0.6
                ):

                    end = (
                        start
                        + cut
                        + 1
                    )

                    piece = sentence[
                        start:end
                    ]

            piece = clean_text(
                piece
            )

            if piece:

                final_result.append(
                    piece
                )

            start = end

    return final_result

# ============================================================
# SUBTITLE DETECTION
# ============================================================

def is_subtitle_paragraph(
    text: str,
) -> bool:

    text = clean_text(text)

    if not text:
        return False

    # 너무 긴 문장은 소제목으로 보지 않음
    if len(text) > 120:
        return False

    return bool(
        SUBTITLE_RE.match(text)
    )


# ============================================================
# SECTION PARAGRAPH EXTRACTION
# ============================================================

def collect_section_groups(
    xml_path: Path,
) -> list[dict[str, Any]]:
    """
    SECTION 단위로 narrative를 수집한다.

    TABLE은 제외한다.

    SECTION 안에서도
    (1) 식품 사업
    (2) 바이오 사업
    같은 소제목성 P가 나타나면
    별도 subgroup으로 나눈다.
    """

    raw = xml_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    body = soup.find("body") or soup

    groups: list[dict[str, Any]] = []

    group_index = 0

    def walk_section(
        section: Tag,
        parent_path: list[str],
    ) -> None:

        nonlocal group_index

        title = direct_title(section)

        current_path = list(parent_path)

        if title:
            current_path.append(title)

        current_subtitle = ""
        current_paragraphs: list[str] = []

        def flush() -> None:

            nonlocal group_index
            nonlocal current_paragraphs

            if not current_paragraphs:
                return

            total_text = clean_text(
                " ".join(current_paragraphs)
            )

            if len(total_text) < MIN_CHARS:
                current_paragraphs = []
                return

            group_index += 1

            groups.append(
                {
                    "group_index": group_index,
                    "section_path": list(
                        current_path
                    ),
                    "subtitle": current_subtitle,
                    "paragraphs": list(
                        current_paragraphs
                    ),
                }
            )

            current_paragraphs = []

        for child in section.children:

            if not isinstance(child, Tag):
                continue

            if not child.name:
                continue

            name = child.name.lower()

            # SECTION TITLE
            if name == "title":
                continue

            # 하위 SECTION 등장
            if is_section(child):

                flush()

                walk_section(
                    child,
                    current_path,
                )

                continue

            # 기존 table chunk가 담당
            if name in {
                "table",
                "table-group",
            }:

                flush()
                continue

            # 일반 문단
            if name == "p":

                segments = split_paragraph_by_subtitle_spans(
                    child
                )

                for segment in segments:

                    subtitle = clean_text(
                        segment.get("subtitle")
                    )

                    text = clean_text(
                        segment.get("text")
                    )

                    # ================================================
                    # P 내부에서 새로운 소제목 발견
                    # ================================================

                    if subtitle:

                        # 이전 소제목의 내용 종료
                        flush()

                        current_subtitle = subtitle

                        # 소제목 자체도 embedding text에 포함
                        current_paragraphs.append(
                            subtitle
                        )

                        if text:
                            current_paragraphs.append(
                                text
                            )

                        continue

                    # ================================================
                    # 별도의 SPAN 소제목이 없는 일반 P
                    # ================================================

                    if not text:
                        continue

                    # P 전체가 짧은 소제목인 기존 경우
                    if is_subtitle_paragraph(text):

                        flush()

                        current_subtitle = text

                        current_paragraphs.append(
                            text
                        )

                    else:

                        current_paragraphs.append(
                            text
                        )

                continue

            # LIBRARY 같은 wrapper 처리
            nested_sections = [
                t
                for t in child.find_all(
                    recursive=False
                )
                if (
                    isinstance(t, Tag)
                    and is_section(t)
                )
            ]

            if nested_sections:

                flush()

                for nested in nested_sections:

                    walk_section(
                        nested,
                        current_path,
                    )

        flush()

    # 가장 상위 SECTION만 시작점으로 사용
    top_sections: list[Tag] = []

    for section in body.find_all(
        SECTION_TAG_RE
    ):

        parent_section_exists = False

        for parent in section.parents:

            if parent is body:
                break

            if (
                isinstance(parent, Tag)
                and is_section(parent)
            ):
                parent_section_exists = True
                break

        if not parent_section_exists:
            top_sections.append(section)

    for section in top_sections:

        walk_section(
            section,
            [],
        )

    return groups

def is_bold_subtitle_span(tag: Tag) -> bool:
    """
    DART XML에서 굵은 글씨로 표시된 SPAN 중
    소제목으로 사용할 만한 것을 판별한다.

    예:
    <SPAN USERMARK=" B">(1) 식품 사업</SPAN>
    <SPAN USERMARK="F-14 B">가. 시장위험</SPAN>
    """

    if not isinstance(tag, Tag):
        return False

    if not tag.name or tag.name.lower() != "span":
        return False

    usermark = clean_text(
        tag.get("usermark")
    ).upper()

    # USERMARK에 B가 포함되어 있어야 함
    if "B" not in usermark:
        return False

    text = tag_text(tag)

    if not text:
        return False

    # 너무 긴 경우 일반 강조문일 가능성이 높음
    if len(text) > 150:
        return False

    # 기존 소제목 패턴
    if SUBTITLE_RE.match(text):
        return True

    # [시장위험 관리 방식] 같은 대괄호 제목
    if (
        text.startswith("[")
        and "]" in text
        and len(text) <= 100
    ):
        return True

    return False


def split_paragraph_by_subtitle_spans(
    p_tag: Tag,
) -> list[dict[str, str]]:
    """
    하나의 <P> 내부를

    1. bold SPAN 소제목 기준
    2. 일반 텍스트 내부 소제목 기준

    두 번 분리한다.
    """

    raw_segments: list[dict[str, str]] = []

    current_subtitle = ""
    current_text_parts: list[str] = []

    def flush() -> None:
        nonlocal current_text_parts

        body = clean_text(
            " ".join(current_text_parts)
        )

        if current_subtitle or body:

            raw_segments.append(
                {
                    "subtitle": current_subtitle,
                    "text": body,
                }
            )

        current_text_parts = []

    for child in p_tag.children:

        # ----------------------------------------------------
        # 일반 문자열
        # ----------------------------------------------------
        if not isinstance(child, Tag):

            text = clean_text(child)

            if text:
                current_text_parts.append(
                    text
                )

            continue

        # ----------------------------------------------------
        # bold SPAN 소제목
        # ----------------------------------------------------
        if is_bold_subtitle_span(child):

            flush()

            current_subtitle = tag_text(
                child
            )

            continue

        # ----------------------------------------------------
        # 일반 SPAN / 기타 태그
        # ----------------------------------------------------
        text = tag_text(child)

        if text:
            current_text_parts.append(
                text
            )

    flush()

    # --------------------------------------------------------
    # 2차:
    # 일반 텍스트 내부의 (3), (4), 가., 나. 등을 다시 분리
    # --------------------------------------------------------

    final_segments: list[dict[str, str]] = []

    for segment in raw_segments:

        parent_subtitle = clean_text(
            segment.get("subtitle")
        )

        text = clean_text(
            segment.get("text")
        )

        if not text:

            if parent_subtitle:
                final_segments.append(
                    {
                        "subtitle": parent_subtitle,
                        "text": "",
                    }
                )

            continue

        inline_parts = split_inline_subtitles(
            text
        )

        # inline 소제목이 없는 경우
        if (
            len(inline_parts) == 1
            and not inline_parts[0]["subtitle"]
        ):

            final_segments.append(
                {
                    "subtitle": parent_subtitle,
                    "text": text,
                }
            )

            continue

        # inline 소제목이 있는 경우
        for part in inline_parts:

            inline_subtitle = clean_text(
                part.get("subtitle")
            )

            body = clean_text(
                part.get("text")
            )

            # 첫 부분이 일반 텍스트면
            # 기존 parent subtitle을 유지
            if not inline_subtitle:

                final_segments.append(
                    {
                        "subtitle": parent_subtitle,
                        "text": body,
                    }
                )

            else:

                final_segments.append(
                    {
                        "subtitle": inline_subtitle,
                        "text": body,
                    }
                )

    # 아무것도 없으면 fallback
    if not final_segments:

        full_text = tag_text(
            p_tag
        )

        if full_text:

            final_segments.append(
                {
                    "subtitle": "",
                    "text": full_text,
                }
            )

    return final_segments

INLINE_SUBTITLE_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))"
    r"(?P<prefix>"
    r"\(\d+\)"     # (1)
    r"|\d+\)"      # 1)
    r")"
    r"(?=\s)"
)


def split_inline_subtitles(text: str) -> list[dict[str, str]]:
    """
    일반 텍스트 안에서 명확한 숫자형 소제목 경계만 분리한다.

    예:
    "...내용... (3) 판매방법 및 조건 ... (4) 판매전략 ..."

    주의:
    inline plain text만 보고
    '(4) 판매전략'에서 제목이 어디까지인지 정확히 알 수 없으므로
    제목을 억지로 추론하지 않는다.
    """

    text = clean_text(text)

    if not text:
        return []

    matches = list(
        INLINE_SUBTITLE_RE.finditer(text)
    )

    if not matches:
        return [
            {
                "subtitle": "",
                "text": text,
            }
        ]

    results: list[dict[str, str]] = []

    # 첫 marker 이전 내용
    if matches[0].start() > 0:

        head = clean_text(
            text[:matches[0].start()]
        )

        if head:
            results.append(
                {
                    "subtitle": "",
                    "text": head,
                }
            )

    for i, match in enumerate(matches):

        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        segment = clean_text(
            text[start:end]
        )

        if not segment:
            continue

        prefix = clean_text(
            match.group("prefix")
        )

        # marker 이후 전체를 body에 그대로 보존
        body = clean_text(
            segment[len(prefix):]
        )

        results.append(
            {
                "subtitle": prefix,
                "text": body,
            }
        )

    return results

# ============================================================
# CHUNK PACKING
# ============================================================

def pack_sentences(
    paragraphs: list[str],
) -> list[list[str]]:
    """
    핵심 chunking 로직.

    1. 모든 paragraph를 sentence로 분해
    2. TARGET_CHARS 약 600자
    3. MAX_CHARS 900자를 절대 넘지 않도록 함
    4. 이전 chunk 마지막 문장 1개만 overlap
    """

    sentences: list[str] = []

    for paragraph in paragraphs:

        sentences.extend(
            split_sentences(paragraph)
        )

    if not sentences:
        return []

    chunks: list[list[str]] = []

    current: list[str] = []
    current_len = 0

    i = 0

    while i < len(sentences):

        sentence = clean_text(
            sentences[i]
        )

        if not sentence:

            i += 1
            continue

        additional = len(sentence)

        if current:
            additional += 1

        # 넣으면 MAX 초과
        if (
            current
            and current_len + additional
            > MAX_CHARS
        ):

            chunks.append(
                current[:]
            )

            # 마지막 1문장만 overlap
            if OVERLAP_SENTENCES > 0:

                overlap = current[
                    -OVERLAP_SENTENCES:
                ]

                current = overlap[:]

                current_len = sum(
                    len(x)
                    for x in current
                )

                if len(current) > 1:
                    current_len += (
                        len(current) - 1
                    )

            else:

                current = []
                current_len = 0

            # overlap된 문장 + 새 문장조차
            # MAX 초과라면 overlap 제거
            if (
                current
                and current_len
                + len(sentence)
                + 1
                > MAX_CHARS
            ):

                current = []
                current_len = 0

            continue

        current.append(sentence)

        current_len += additional

        i += 1

        # 600자를 넘었으면
        # 다음 문장을 더 넣을 필요 없이 여기서 종료
        if (
            current_len >= TARGET_CHARS
            and current
        ):

            chunks.append(
                current[:]
            )

            if OVERLAP_SENTENCES > 0:

                overlap = current[
                    -OVERLAP_SENTENCES:
                ]

                current = overlap[:]

                current_len = sum(
                    len(x)
                    for x in current
                )

            else:

                current = []
                current_len = 0

    if current:

        # 마지막 chunk가 overlap 한 문장만 남은 경우
        # 지나치게 짧으면 이전 chunk에 굳이 추가하지 않음
        current_text = clean_text(
            " ".join(current)
        )

        if len(current_text) >= MIN_CHARS:

            if (
                not chunks
                or current != chunks[-1]
            ):
                chunks.append(
                    current[:]
                )

    return chunks


# ============================================================
# BUILD CHUNKS
# ============================================================

def make_narrative_chunks(
    xml_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:

    xml_path = Path(xml_path)

    meta = extract_document_meta(
        xml_path
    )

    if extra_metadata:

        meta = {
            **meta,
            **extra_metadata,
        }

    groups = collect_section_groups(
        xml_path
    )

    chunks: list[dict[str, Any]] = []

    narrative_index = 0

    for group in groups:

        packed_chunks = pack_sentences(
            group["paragraphs"]
        )

        for local_index, sentences in enumerate(
            packed_chunks,
            start=1,
        ):

            body = clean_text(
                " ".join(sentences)
            )

            if len(body) < MIN_CHARS:
                continue

            # 안전 검증
            if len(body) > MAX_CHARS:

                print(
                    "[WARN] body exceeds MAX_CHARS:",
                    len(body),
                    group["section_path"],
                )

            narrative_index += 1

            section_path = (
                " > ".join(
                    group["section_path"]
                )
                if group["section_path"]
                else "본문"
            )

            subtitle = clean_text(
                group.get("subtitle")
            )

            context_lines = []

            corp_name = clean_text(
                meta.get("corp_name")
            )

            if corp_name:
                context_lines.append(
                    f"회사: {corp_name}"
                )

            document_name = clean_text(
                meta.get("document_name")
            )

            if document_name:
                context_lines.append(
                    f"보고서: {document_name}"
                )

            if (
                meta.get("period_from")
                or meta.get("period_to")
            ):

                context_lines.append(
                    "사업연도: "
                    f"{clean_text(meta.get('period_from'))}"
                    " ~ "
                    f"{clean_text(meta.get('period_to'))}"
                )

            context_lines.append(
                f"섹션: {section_path}"
            )

            if subtitle:

                context_lines.append(
                    f"소제목: {subtitle}"
                )

            text = (
                "\n".join(context_lines)
                + "\n\n"
                + body
            )

            stable_source = (
                f"{meta.get('rcept_no', '')}|"
                f"{section_path}|"
                f"{subtitle}|"
                f"{group['group_index']}|"
                f"{local_index}|"
                f"{body[:200]}"
            )

            digest = hashlib.sha1(
                stable_source.encode(
                    "utf-8"
                )
            ).hexdigest()[:12]

            doc_id = clean_text(
                meta.get("doc_id")
            )

            if not doc_id:

                if clean_text(
                    meta.get("rcept_no")
                ):

                    doc_id = (
                        "periodic_"
                        + clean_text(
                            meta.get("rcept_no")
                        )
                    )

                else:

                    doc_id = (
                        f"periodic_{xml_path.stem}"
                    )

            chunk_id = (
                f"{doc_id}"
                f"_narrative_"
                f"{narrative_index:04d}_"
                f"{digest}"
            )

            metadata = {
                **meta,

                "doc_id": doc_id,

                "doc_group": (
                    clean_text(
                        meta.get("doc_group")
                    )
                    or "periodic"
                ),

                "raw_file": str(
                    xml_path
                ),

                "chunk_type": "narrative",

                "chunk_strategy": (
                    "section_sentence"
                ),

                "search_priority": "medium",

                "narrative_index": (
                    narrative_index
                ),

                "section_path": (
                    section_path
                ),

                "subtitle": subtitle,

                "section_depth": len(
                    group["section_path"]
                ),

                "sentence_count": len(
                    sentences
                ),

                "body_length": len(
                    body
                ),

                "text_length": len(
                    text
                ),
            }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

    return chunks


# ============================================================
# SAVE
# ============================================================

def save_jsonl(
    chunks: list[dict[str, Any]],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# CHECK STATS
# ============================================================

def print_stats(
    chunks: list[dict[str, Any]],
) -> None:

    if not chunks:

        print("chunk 없음")
        return

    body_lengths = [
        c["metadata"]["body_length"]
        for c in chunks
    ]

    over_900 = [
        x
        for x in body_lengths
        if x > MAX_CHARS
    ]

    print()
    print("=" * 80)
    print("Narrative Chunk Statistics")
    print("=" * 80)

    print(
        f"총 narrative chunk: "
        f"{len(chunks)}"
    )

    print(
        f"평균 body 길이: "
        f"{sum(body_lengths) / len(body_lengths):.1f}"
    )

    print(
        f"최소 body 길이: "
        f"{min(body_lengths)}"
    )

    print(
        f"최대 body 길이: "
        f"{max(body_lengths)}"
    )

    print(
        f"900자 초과 chunk: "
        f"{len(over_900)}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Periodic XML Narrative Chunk Builder"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    chunks = make_narrative_chunks(
        args.input
    )

    print()
    print("=" * 80)
    print(
        "Periodic Narrative Chunk Builder"
    )
    print("=" * 80)

    print(
        "input:",
        args.input,
    )

    print_stats(chunks)

    print()
    print("[SAMPLE]")

    for chunk in chunks[:5]:

        print()
        print("-" * 80)

        print(
            "chunk_id:",
            chunk["chunk_id"],
        )

        print(
            "section:",
            chunk["metadata"][
                "section_path"
            ],
        )

        print(
            "subtitle:",
            chunk["metadata"][
                "subtitle"
            ],
        )

        print(
            "body_length:",
            chunk["metadata"][
                "body_length"
            ],
        )

        print()

        print(
            chunk["text"][:1200]
        )

    if args.output:

        save_jsonl(
            chunks,
            args.output,
        )

        print()
        print(
            "saved:",
            args.output,
        )


if __name__ == "__main__":
    main()