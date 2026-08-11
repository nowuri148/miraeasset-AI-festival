"""
DART 원문 XML의 '&' 이스케이프 오류를 고쳐서 새 파일로 저장하고, MS Edge로 열어주는 스크립트.

사용법 (터미널/명령 프롬프트에서):
    python fix_and_open_xml.py "C:\\경로\\원본파일.xml"

인자를 안 주면 아래 SAMPLE_XML 경로를 사용함.
"""

import re
import subprocess
import sys
from pathlib import Path

# 인자로 경로를 안 줬을 때 기본으로 쓸 경로 (필요시 수정)
SAMPLE_XML = Path(r"C:\Users\User\.vscode\mirea_asset\corpus\raw\periodic\NAVER\20230512001100_quarter_2023_03\20230512001100.xml")

# 정상 처리된 엔터티(&amp; &lt; &gt; &quot; &apos; &#123; &#x1F;)는 건드리지 않고
# 나머지 날것의 '&'만 안전하게 &amp;로 치환
BARE_AMP_PATTERN = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)")


FAKE_TAG_PATTERN = re.compile(r"<(?![!?/A-Za-z])([^<>]*)>")


def build_viewable_html(fixed_content: str, title: str) -> str:
    """이스케이프까지 고친 내용을 실제 표로 렌더링되는 HTML 문서로 감싸기"""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: "맑은 고딕", sans-serif; padding: 20px; }}
  table {{ border-collapse: collapse; margin-bottom: 24px; }}
  td, th {{ border: 1px solid #999; padding: 4px 8px; font-size: 13px; }}
  th {{ background: #f0f0f0; }}
</style>
</head>
<body>
{fixed_content}
</body>
</html>"""


def fix_xml_entities(src_path: Path) -> Path:
    """원본 XML을 읽어 '&'와 진짜 태그가 아닌 '<...>'(예: <관계기업>) 오류를 고친 뒤,
    표가 실제로 렌더링되는 '_view.html' 파일로 저장하고 그 경로를 반환"""
    content = src_path.read_text(encoding="utf-8", errors="ignore")

    fixed_content = BARE_AMP_PATTERN.sub("&amp;", content)
    amp_count = len(BARE_AMP_PATTERN.findall(content))

    # XML 선언(<?xml version="1.0"...?>)은 HTML에선 불필요 + 화면에 그대로 텍스트로 노출되니 제거
    fixed_content = re.sub(r"<\?xml[^>]*\?>\s*", "", fixed_content)

    # 진짜 태그(<TD>, </TR>, <!--..--> 등)가 아닌 '<...>'는 사람이 쓴 텍스트로 보고 이스케이프
    # 예: <관계기업> → &lt;관계기업&gt;   (한글/숫자 등 ASCII 알파벳이 아닌 글자로 시작하면 가짜 태그로 판단)
    fake_tag_count = len(FAKE_TAG_PATTERN.findall(fixed_content))
    fixed_content = FAKE_TAG_PATTERN.sub(lambda m: f"&lt;{m.group(1)}&gt;", fixed_content)

    print(f"고친 '&' 개수: {amp_count}건")
    print(f"고친 가짜 태그(예: <관계기업>) 개수: {fake_tag_count}건")

    # DOCUMENT/BODY 같은 상위 래퍼 태그는 실제 화면 렌더링에 의미 없으니 그대로 둬도 무방
    # (브라우저가 모르는 태그는 그냥 무시하고 안의 TABLE/TD 등만 그려줌)
    html_content = build_viewable_html(fixed_content, title=src_path.stem)

    view_path = src_path.with_name(src_path.stem + "_view.html")
    view_path.write_text(html_content, encoding="utf-8")
    print(f"웹페이지로 볼 수 있는 파일 저장 위치: {view_path}")
    return view_path


def open_in_edge(path: Path):
    """MS Edge로 파일 열기 (Windows 전용)"""
    try:
        subprocess.run(["cmd", "/c", "start", "msedge", str(path.resolve())], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"⚠ Edge를 자동으로 여는 데 실패했어: {e}")
        print(f"  아래 경로를 직접 Edge 주소창에 드래그하거나 붙여넣어줘:")
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_XML

    if not target.exists():
        print(f"⚠ 파일을 찾을 수 없음: {target}")
        print("  실제 경로를 인자로 넘기거나, 스크립트 안의 SAMPLE_XML을 수정해줘.")
        print('  예: python fix_and_open_xml.py "C:\\Users\\User\\...\\재무제표.xml"')
        sys.exit(1)

    fixed_path = fix_xml_entities(target)
    open_in_edge(fixed_path)