"""
DART 공시 코퍼스 탐색 스크립트
- README에 명시된 수치(건수, 조인 키, 정정공시 등)가 실제 데이터와 일치하는지 검증
- 사용법: 데이터셋 루트 디렉터리에서 실행하거나 DATA_ROOT 경로를 수정

python explore_dart_corpus.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import unicodedata

DATA_ROOT = r"C:\Users\User\.vscode\mirea_asset\corpus"  # universe.csv, manifest.jsonl, raw/ 가 있는 경로로 수정


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------
# 1. 유니버스 테이블 로드 (선행 0 유실 방지)
# ---------------------------------------------------------
section("1. 유니버스 테이블 (universe.csv)")

universe = pd.read_csv(
    os.path.join(DATA_ROOT, "universe.csv"),
    dtype={"corp_code": str, "stock_code": str},
    encoding="utf-8-sig",
)
print(f"기업 수: {len(universe)}개 (README 기준 70개)")
print(f"corp_code 자릿수 확인: {universe['corp_code'].str.len().unique()} (8자리여야 함)")
print(f"stock_code 자릿수 확인: {universe['stock_code'].str.len().unique()} (6자리여야 함)")

print("\n시장 구분:")
print(universe["market"].value_counts())  # KOSPI 61 / KOSDAQ 9

print("\n업종(대분류) 분포:")
print(universe["industry"].value_counts())

print("\n섹터(테마) 상위 10개:")
print(universe["sector"].value_counts().head(10))

# 유형별 수집 문서 건수 컬럼이 실제 파일 건수와 일치하는지는 3번에서 재검증


# ---------------------------------------------------------
# 2. 문서 메타데이터 로드 (manifest.jsonl)
# ---------------------------------------------------------
section("2. 문서 메타데이터 (manifest.jsonl)")

records = []
with open(os.path.join(DATA_ROOT, "manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

manifest = pd.DataFrame(records)
print(f"총 문서 수: {len(manifest)}건 (README 기준 4,204건)")

print("\n문서 유형(doc_group)별 건수:")
print(manifest["doc_group"].value_counts())
# 기대값: periodic 1054 / major 598 / exchange 1469 / holding 1083

print("\n정정공시(is_correction) 비율:")
print(manifest["is_correction"].value_counts())
print(f"정정공시 총 건수: {manifest['is_correction'].sum()}건 (README 기준 1,004건)")

print("\n파일 포맷 분포:")
print(manifest["file_format"].value_counts())
# 기대값: xml 4201 / pdf+html 3

pdf_html_docs = manifest[manifest["file_format"] == "pdf+html"]
if len(pdf_html_docs):
    print("\n대체 수집(pdf+html) 문서 상세:")
    print(pdf_html_docs[["corp_name", "report_nm", "rcept_dt"]].to_string(index=False))


# ---------------------------------------------------------
# 3. 공시 기간 확인
# ---------------------------------------------------------
section("3. 공시 기간 확인")

manifest["rcept_dt"] = pd.to_datetime(manifest["rcept_dt"], format="%Y%m%d", errors="coerce")
print(f"접수일 범위: {manifest['rcept_dt'].min().date()} ~ {manifest['rcept_dt'].max().date()}")
print("(README 기준: 2023-01-01 ~ 2026-03-31)")

out_of_range = manifest[
    (manifest["rcept_dt"] < "2023-01-01") | (manifest["rcept_dt"] > "2026-03-31")
]
print(f"범위를 벗어난 문서: {len(out_of_range)}건")


# ---------------------------------------------------------
# 4. manifest ↔ universe 조인 검증
# ---------------------------------------------------------
section("4. manifest ↔ universe 조인 (corp_name 기준) 검증")

manifest_corps = set(manifest["corp_name"].unique())
universe_corps = set(universe["corp_name"].unique())

only_in_manifest = manifest_corps - universe_corps
only_in_universe = universe_corps - manifest_corps

print(f"manifest에만 있는 법인명: {len(only_in_manifest)}개 -> {only_in_manifest}")
print(f"universe에만 있는 법인명: {len(only_in_universe)}개 -> {only_in_universe}")
print("(둘 다 비어 있어야 정상 조인 — 사명 변경/폴더명 제약 케이스 주의: 현대차, KT, 엔씨소프트, LIG넥스원 등)")


# ---------------------------------------------------------
# 5. 실제 파일 시스템(raw/)과 manifest 일치 여부
# ---------------------------------------------------------
section("5. raw/ 디렉터리 실존 여부 확인 (file_path 기준)")

def exists_normalized(base: Path, rel_path: str) -> bool:
    """NFC/NFD 정규화 차이를 무시하고 경로 존재 여부 확인"""
    for form in ("NFC", "NFD"):
        candidate = base / unicodedata.normalize(form, rel_path)
        if candidate.exists():
            return True
    return False

DATA_ROOT_PATH = Path(DATA_ROOT)  # 문자열이면 Path로 변환

missing_paths = []
for _, row in manifest.iterrows():
    if not exists_normalized(DATA_ROOT_PATH, row["file_path"]):
        missing_paths.append(row["file_path"])

print(f"manifest에는 있으나 실제 폴더가 없는 문서: {len(missing_paths)}건")
if missing_paths[:5]:
    print("예시:", missing_paths[:5])


# ---------------------------------------------------------
# 6. 기업별 상장일 대비 정기공시 건수 정합성 체크 (샘플)
# ---------------------------------------------------------
section("6. 상장일 대비 정기공시 건수 체크 (README 예시: 시프트업 2024-07 상장 → 정기 7건)")

check = universe[universe["corp_name"].str.contains("시프트업", na=False)]
if len(check):
    print(check[["corp_name", "listing_date", "n_periodic"]].to_string(index=False))
else:
    print("유니버스에 '시프트업'을 찾을 수 없음 — 사명 변경 등 확인 필요")


# ---------------------------------------------------------
# 7. 종합 요약
# ---------------------------------------------------------
section("7. 종합 요약")

print(f"기업 수            : {len(universe)} (기대: 70)")
print(f"총 문서 수         : {len(manifest)} (기대: 4204)")
print(f"정기공시(A)        : {(manifest['doc_group'] == 'periodic').sum()} (기대: 1054)")
print(f"주요사항보고서(B)   : {(manifest['doc_group'] == 'major').sum()} (기대: 598)")
print(f"거래소공시(I)       : {(manifest['doc_group'] == 'exchange').sum()} (기대: 1469)")
print(f"지분공시(D)        : {(manifest['doc_group'] == 'holding').sum()} (기대: 1083)")
print(f"정정공시 총 건수    : {manifest['is_correction'].sum()} (기대: 1004)")
print(f"pdf+html 대체수집  : {(manifest['file_format'] == 'pdf+html').sum()} (기대: 3)")
print(f"corp_name 조인 불일치: {len(only_in_manifest) + len(only_in_universe)} (기대: 0)")
print(f"실제 파일 누락      : {len(missing_paths)} (기대: 0)")