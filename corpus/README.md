# DART 공시 코퍼스

국내 주요 산업 대표 상장기업 **70개사**의 DART 전자공시 원문 코퍼스입니다.
공시 기간은 **2023-01-01 ~ 2026-03-31** (정기공시는 FY2023 ~ 2026년 1분기),
총 **4,204개 문서 / XML 4,616개**입니다.

## 파일 구성

```
├── README.md          # 이 문서
├── universe.xlsx      # 기업 마스터 테이블 (Excel — 열람용 권장)
├── universe.csv       # 기업 마스터 테이블 (UTF-8 BOM — 프로그래밍용)
├── manifest.jsonl     # 문서 메타데이터 (4,204건, 1문서 = 1행, JSON Lines)
├── data_filter.md     # 수집 조건 명세 (유니버스·문서종류·기간·선정기준)
└── raw/               # 공시 원문
    ├── periodic/  <법인명>/{접수번호}_{annual|half|quarter}_{연도}_{월}/*.xml   # 정기공시
    ├── major/     <법인명>/{접수번호}/*.xml                                     # 주요사항보고서
    ├── exchange/  <법인명>/{접수번호}/*.xml                                     # 거래소공시
    └── holding/   <법인명>/{접수번호}/*.xml                                     # 지분공시(대량보유)
    (각 기업 폴더의 list_*.json = 해당 유형 전체 공시 목록(DART list API 원본))
```

## 유니버스 (universe.csv / universe.xlsx)

70개사 × 17컬럼. 업종(대분류 8개) > 섹터(테마 20개)의 2단 분류.

| 컬럼 | 내용 |
|---|---|
| `corp_code` | DART 고유번호 8자리 (예: `00126380`) |
| `stock_code` | 종목코드 6자리 (예: `005930`) |
| `corp_name` | DART 공식 법인명 — **`raw/` 하위 폴더명과 동일 (조인 키)** |
| `listed_name` | 거래소 통용 종목명 (현대차, KT, LIG넥스원 등) |
| `corp_eng_name` | 영문 법인명 |
| `market` | 시장구분 (KOSPI 61 / KOSDAQ 9) |
| `industry` | 업종 — 대분류 8개 (IT, 산업재, 소재, 금융, 커뮤니케이션서비스, 건강관리, 필수소비재, 경기관련소비재) |
| `sector_no` / `sector` | 섹터 — 테마 20개 (반도체·전자부품, 2차전지, 조선, 방산·항공우주 등) |
| `listing_date` | 상장일 (YYYY-MM-DD) |
| `fiscal_month` | 결산월 (전 기업 12월) |
| `market_cap` | 시가총액 (억원, 2026-07-24 조회) |
| `n_periodic` / `n_major` / `n_exchange` / `n_holding` | 유형별 수집 문서 건수 |
| `note` | 예외사항 (사명 변경, 대체 수집 등) |

## 문서 메타데이터 (manifest.jsonl)

수집된 모든 문서의 메타데이터. 한 줄이 문서 하나입니다.

| 필드 | 내용 |
|---|---|
| `doc_id` | `{doc_group}_{rcept_no}` |
| `corp_code` / `corp_name` / `listed_name` / `stock_code` | 기업 식별 (universe와 조인) |
| `industry` / `sector` | 업종 / 섹터 |
| `doc_group` | `periodic` \| `major` \| `exchange` \| `holding` |
| `doc_subtype` | 정기: annual/half/quarter · 거래소: 단일판매공급계약체결/해지, 신규시설투자등, 투자판단관련주요경영사항 · 지분: 대량보유상황보고서 |
| `report_nm` / `rcept_no` / `rcept_dt` / `flr_nm` | 보고서명 / 접수번호 / 접수일 / 제출인 |
| `is_correction` | `[기재정정]` 정정공시 여부 (true 1,004건) |
| `base_year` / `base_month` | 정기공시 보고 기준기간 |
| `file_path` | 원문 폴더 상대경로 (예: `raw/periodic/삼성전자/20230307000542_annual_2023_12`) |
| `file_format` | `xml` (4,201건) \| `pdf+html` (대체 수집 3건) |
| `n_files` | 폴더 내 파일 수 |

## 문서 구성 (4,204건)

| 유형 | 건수 | 비고 |
|---|---|---|
| 정기공시 (A) | 1,054 | 사업·반기·분기보고서, 기재정정 159 포함. 사업보고서는 감사보고서 첨부 XML 포함 |
| 주요사항보고서 (B) | 598 | 기재정정 173 포함 |
| 거래소공시 (I) | 1,469 | 공급계약 체결 1,106 / 해지 20 / 신규시설투자 43 / 투자판단관련 300, 기재정정 631 포함 |
| 지분공시 (D) | 1,083 | 주식등의대량보유상황보고서(5% 보고), 기재정정 41 포함 |

정정공시는 원본과 **병행 수집**되어 있습니다 (`is_correction`으로 구분).
`[기재정정]` 외 태그 공시는 수집 대상이 아니나, 원본이 `[첨부추가]`본으로만
제공되는 공시 6건은 해당 본을 원본으로 수집했습니다 (`is_correction=false`).

## 사용 시 주의

1. **코드 컬럼의 선행 0** — `corp_code`(8자리)·`stock_code`(6자리)는 문자열입니다.
   pandas 로딩 시 반드시 `dtype={'corp_code': str, 'stock_code': str}` 지정.
   Excel에서 CSV를 직접 열면 선행 0이 유실되니 열람은 `universe.xlsx` 사용을 권장합니다.
2. **조인 키** — `corp_name`(DART 공식 법인명) = `raw/` 하위 폴더명.
   통용명과 다른 예: 현대차→현대자동차, KT→케이티, 엔씨소프트→NC,
   LIG넥스원→LIG디펜스앤에어로스페이스(2026-04 사명 변경), JYP Ent.→`JYP Ent`(폴더명 제약).
3. **건수 0 ≠ 결측** — 정기공시는 제출의무 발생 이후 분만 존재하고(`listing_date` 참조,
   예: 시프트업 2024-07 상장 → 정기 7건), 주요사항·거래소공시는 해당 이벤트가 없으면 0건입니다.
4. **대체 수집 3건** — DART가 원본 XML을 제공하지 않는(status 014) 문서는
   공식 PDF(`{접수번호}.pdf`) + 공시뷰어 HTML(`{접수번호}_viewer.html`)로 대체:
   한화에어로스페이스 분기보고서(2026.03), KB금융 [기재정정]사업보고서(2025.12),
   한화오션 [기재정정]분기보고서(2024.03). `file_format=pdf+html`로 식별 가능합니다.
5. **원문 XML**은 DART `document.xml` API 원본 그대로이며 인코딩은 UTF-8입니다.

## 파이썬 로딩 예시

```python
import pandas as pd

universe = pd.read_csv("universe.csv", dtype={"corp_code": str, "stock_code": str})
manifest = pd.read_json("manifest.jsonl", lines=True, dtype={"corp_code": str, "stock_code": str})

# 예: 방산·항공우주 섹터의 공급계약 공시(정정 제외)
docs = manifest[(manifest.sector == "방산·항공우주")
                & (manifest.doc_subtype == "단일판매공급계약체결")
                & ~manifest.is_correction]
```
