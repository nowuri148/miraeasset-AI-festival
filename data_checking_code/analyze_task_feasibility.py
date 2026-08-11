"""
DART 공시 코퍼스 - Task 수행 가능성 분석 스크립트
(검색/정보추출, 다중조회/비교연산, 복합문서추론 3개 Task를 실제로 풀 수 있는지 사전 점검)

사용법: 데이터셋 루트 디렉터리에서 실행하거나 DATA_ROOT 경로를 수정
python analyze_task_feasibility.py
"""

import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import os

DATA_ROOT = Path(r"C:\Users\User\.vscode\mirea_asset\corpus")  # universe.csv, manifest.jsonl, raw/ 가 있는 경로로 수정
random.seed(42)


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def resolve_path(base: Path, rel_path: str) -> Path | None:
    """
    NFC/NFD 유니코드 정규화 차이를 무시하고 실제 존재하는 경로를 찾아 반환.
    못 찾으면 None.
    """
    for form in ("NFC", "NFD"):
        candidate = base / unicodedata.normalize(form, rel_path)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------
# 0. 기본 로드
# ---------------------------------------------------------
universe = pd.read_csv(
    os.path.join(DATA_ROOT, "universe.csv"),
    dtype={"corp_code": str, "stock_code": str},
    encoding="utf-8-sig",
)

records = []
with open(os.path.join(DATA_ROOT, "manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
manifest = pd.DataFrame(records)
manifest["rcept_dt"] = pd.to_datetime(manifest["rcept_dt"], format="%Y%m%d", errors="coerce")


# ---------------------------------------------------------
# 1. 문서 유형별 실제 구조 샘플링 (XML 태그 확인)
# ---------------------------------------------------------
section("1. 문서 유형별 XML 구조 샘플링")

for doc_group in manifest["doc_group"].unique():
    sample = manifest[manifest["doc_group"] == doc_group].sample(1, random_state=1).iloc[0]
    folder = resolve_path(DATA_ROOT, sample["file_path"])
    xml_files = list(folder.glob("*.xml")) if folder else []

    print(f"\n[{doc_group}] {sample['corp_name']} - {sample['report_nm']}")
    print(f"  경로: {sample['file_path']}")
    print(f"  XML 파일 수: {len(xml_files)}")

    if xml_files:
        content = xml_files[0].read_text(encoding="utf-8", errors="ignore")
        tags = re.findall(r"<([A-Za-z가-힣_]+)[ >]", content[:5000])
        top_tags = pd.Series(tags).value_counts().head(8)
        print(f"  상위 태그(첫 5000자 기준): {dict(top_tags)}")
    else:
        print("  ⚠ 파일을 찾을 수 없음 (경로/다운로드 확인 필요)")


# ---------------------------------------------------------
# 2. 문서 길이 분포 (chunking 전략 설계 근거)
# ---------------------------------------------------------
section("2. 문서 유형별 길이 분포 (전체 XML 파일 크기 합산 기준, 샘플 30건)")

length_stats = defaultdict(list)
sample_size = 30

for doc_group in manifest["doc_group"].unique():
    subset = manifest[manifest["doc_group"] == doc_group]
    sampled = subset.sample(min(sample_size, len(subset)), random_state=1)
    for _, row in sampled.iterrows():
        folder = resolve_path(DATA_ROOT, row["file_path"])
        if folder:
            total_chars = sum(
                len(p.read_text(encoding="utf-8", errors="ignore")) for p in folder.glob("*.xml")
            )
            length_stats[doc_group].append(total_chars)

for doc_group, lengths in length_stats.items():
    if lengths:
        s = pd.Series(lengths)
        print(
            f"{doc_group:10s}: 평균 {s.mean():,.0f}자 / 중앙값 {s.median():,.0f}자 "
            f"/ 최대 {s.max():,.0f}자 / 최소 {s.min():,.0f}자 (n={len(s)})"
        )


# ---------------------------------------------------------
# 3. 정정공시 체인 구조 파악 (변경 이력 분석 Task 대비)
# ---------------------------------------------------------
section("3. 정정공시 체인 구조")

corrections = manifest[manifest["is_correction"]]
print(f"정정공시 총 건수: {len(corrections)}")

chain_counts = (
    corrections.groupby(["corp_name", "doc_subtype"]).size().sort_values(ascending=False)
)
print("\n기업×문서유형별 정정 건수 상위 10 (체인이 긴 케이스):")
print(chain_counts.head(10))

print(f"\n정정이 2건 이상 발생한 (기업, 유형) 조합 수: {(chain_counts >= 2).sum()}")


# ---------------------------------------------------------
# 4. 기업 × 연도 커버리지 매트릭스 (다중조회·비교 Task 대비)
# ---------------------------------------------------------
section("4. 사업보고서(annual) 기업 × 연도 커버리지")

annual = manifest[(manifest["doc_group"] == "periodic") & (manifest["doc_subtype"] == "annual")]
coverage = annual.pivot_table(
    index="corp_name", columns="base_year", values="rcept_no", aggfunc="count", fill_value=0
)
print(f"사업보고서 보유 기업 수: {coverage.shape[0]}")
print(f"연도 컬럼: {list(coverage.columns)}")

year_cols = [c for c in coverage.columns if c in [2023, 2024, 2025]]
if year_cols:
    full_coverage = coverage[(coverage[year_cols] > 0).all(axis=1)]
    print(f"\n2023~2025년 사업보고서를 모두 보유한 기업 수: {len(full_coverage)}")
    print(f"→ '연도별 변화 비교' 질문에 안전하게 답할 수 있는 기업 풀")

print("\n섹터별 기업 수 (동종업계 비교 질문 대비, 상위 10):")
print(universe["sector"].value_counts().head(10))


# ---------------------------------------------------------
# 5. 재무 수치 계산 가능성 확인 (증감률·비중 계산 Task 대비)
# ---------------------------------------------------------
section("5. 재무 수치 태그 일관성 확인 (사업보고서 샘플)")

sample_annuals = annual.sample(min(5, len(annual)), random_state=1)
account_tag_pattern = re.compile(r"<(매출액|영업이익|당기순이익|자산총계|부채총계)[^>]*>")

for _, row in sample_annuals.iterrows():
    folder = resolve_path(DATA_ROOT, row["file_path"])
    if not folder:
        continue
    found_accounts = set()
    for xml_file in folder.glob("*.xml"):
        content = xml_file.read_text(encoding="utf-8", errors="ignore")
        found_accounts.update(account_tag_pattern.findall(content))
    print(f"{row['corp_name']} ({row['base_year']}): 발견된 주요 계정과목 = {found_accounts or '없음(태그명 재확인 필요)'}")

print("\n⚠ 위 결과가 기업마다 다르게 나오면, 계정과목명을 정규화하는 매핑 테이블이 필요함")
print("   (예: '매출액' vs '매출' vs '수익(매출액)' 등 표기 차이 대응)")


# ---------------------------------------------------------
# 6. 정보 커버리지 공백 파악 (정보한계 대응 Task 대비)
# ---------------------------------------------------------
section("6. 커버리지 공백 확인")

universe["listing_date"] = pd.to_datetime(universe["listing_date"], errors="coerce")
late_listed = universe[universe["listing_date"] > "2023-01-01"]
print(f"2023년 이후 상장한 기업 수: {len(late_listed)}")
if len(late_listed):
    print(late_listed[["corp_name", "listing_date", "n_periodic"]].to_string(index=False))
    print("→ 이 기업들은 상장 이전 연도 데이터가 없으므로, 해당 기간 질문에는 '확인 불가' 응답 필요")

zero_major = universe[universe["n_major"] == 0]
zero_exchange = universe[universe["n_exchange"] == 0]
print(f"\n주요사항보고서 0건 기업: {len(zero_major)}개")
print(f"거래소공시 0건 기업: {len(zero_exchange)}개")
print("→ 이런 기업 대상 '자금조달 내역', '계약 체결 이력' 질문은 정상적으로 '해당 없음'이 정답일 수 있음")


# ---------------------------------------------------------
# 7. 종합 요약
# ---------------------------------------------------------
section("7. 종합 요약 — 설계에 반영할 점")

print("- 문서 유형별 XML 태그 구조가 다르면 → 유형별 파싱 로직 분리 필요 (1번 결과 참고)")
print("- 문서 길이 편차가 크면 → 고정 chunk 크기 대신 유형별 chunking 전략 필요 (2번 결과 참고)")
print("- 정정 체인이 긴 기업/유형이 있으면 → 최신본 우선 로직에서 우선순위 명확히 정의 (3번 결과 참고)")
print("- 연속 연도 커버리지가 있는 기업 풀을 확보해야 → '비교' 질문 테스트셋 구성 가능 (4번 결과 참고)")
print("- 계정과목 표기가 기업마다 다르면 → 정규화 매핑 테이블 구축 필요 (5번 결과 참고)")
print("- 커버리지 공백은 → 정보한계 대응 로직의 화이트리스트/블랙리스트로 활용 (6번 결과 참고)")