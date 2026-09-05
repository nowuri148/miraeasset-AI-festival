# 검색계획 Python 실행기 — 회사별 Chroma 운영 연결본

검색계획 LLM이 만든 `retrieval_requests`를 받아 문서 범위를 확정하고,
실제 회사별 Chroma DB에서 검색한 뒤 답변 LLM에 전달할 컨텍스트를 만든다.
답변 생성과 답변 검증은 포함하지 않는다.

## 처리 경계

검색계획 LLM은 검색 의도를 구조화한다. Python 실행기는 그 이후를 담당한다.

1. 검색계획 형식과 허용값 검증
2. 기업 범위를 실제 기업 코드와 회사명으로 확장
3. 표준 문서유형과 기간으로 문서 범위 확정
4. 필요한 경우 정정 그래프 전체 체인 확장
5. 회사별 Chroma HNSW 벡터 검색
6. 확정 문서의 BM25 키워드 검색
7. RRF 통합과 중복 제거
8. 결정적 리랭킹
9. 인접 서술·표 청크 확장
10. 최종 컨텍스트 패킹

## 운영 파일 기본 경로

- 문서 메타데이터: `corpus/derived/manifest_v3_reviewed.jsonl`
- 대체 문서 메타데이터: `corpus/derived/manifest_v3.jsonl`, `corpus/manifest.jsonl`
- 기업 유니버스: `corpus/universe.csv`
- 정정 그래프: `corpus/derived/graph_edges_v1_reviewed.jsonl` 또는 `graph_edges_v1.jsonl`
- 회사별 Chroma DB: `corpus/vector_db/chroma_company`
- 회사-컬렉션 매핑: `corpus/vector_db/company_collection_map.json`
- 임베딩 모델: `BAAI/bge-m3`

`manifest_v3` 계열 사용을 권장한다. 이벤트 공시 검색에는
`normalized_report_type`, `event_date`, `disclosure_chain_id`가 필요하기 때문이다.

운영 검색기는 저장소에 이미 있는 `search_info/vector_retriever.py`를 재사용한다.
회사를 매핑 파일로 컬렉션에 연결하고, BGE-M3 임베딩을 한 번 생성한 뒤 회사별
컬렉션에서 검색한다.

## 기간 처리

- `base_year`: 사업연도·보고기간. `YYYY`, `YYYY-Q1`~`YYYY-Q4`,
  `YYYY-H1`~`YYYY-H2`만 허용한다.
- `rcept_dt`: 공시 제출·접수일. `YYYY-MM-DD`만 허용한다.
- `event_date`: 공시에 기재된 결정일·발생일·변동일·계약일·해지일.
  `YYYY-MM-DD`만 허용한다.

문서유형과 날짜 기준을 코드에서 강제로 묶지 않는다. 질문의 의미에 따라
검색계획 LLM이 날짜 기준을 선택한다. 예를 들어 시설투자 결정 시점을 묻는다면
`event_date`, 공시 접수 시점을 묻는다면 `rcept_dt`, 정기보고서의 사업연도별
설비투자 실적을 묻는다면 `base_year`가 된다.

`event_date` 요청에서 개별 문서의 event_date가 없거나 유효하지 않을 때만
그 문서의 `rcept_dt`를 대체 기준으로 사용한다. event_date가 정상인 문서는
절대 rcept_dt로 다시 판정하지 않는다. `base_year`로는 자동 전환하지 않는다.
대체 적용 여부와 문서 ID는 `request_diagnostics`에 남는다.

엄격 모드에서는 `--strict-event-date`를 사용해 대체를 끌 수 있다.

## 회사별 Chroma 검색

벡터 검색은 Chroma `where` 조건을 사용하지 않는다. 각 회사 컬렉션에서 HNSW
검색을 먼저 수행한 뒤 확정된 `doc_id`만 Python에서 남긴다. 결과가 부족하면
후보 개수를 단계적으로 확대하고 마지막에는 해당 회사 컬렉션 전체까지 확인한다.

BM25와 인접 청크 확장에 필요한 청크는 `doc_id` 조건으로 읽는다. Chroma의
필터 조회가 실패하면 해당 회사 컬렉션을 읽은 뒤 Python에서 문서 범위를
필터링한다.

## 회사 범위

- `direct`: 정규화 회사명 또는 universe의 정확한 별칭
- `group`: 키워드 추출기가 확정한 `target_companies`를 실행 단계에서 사용
- `sector`: 허용된 20개 섹터를 universe의 실제 회사로 확장
- `industry`: 허용된 8개 산업을 universe의 실제 회사로 확장
- `all`: manifest에 문서가 존재하는 전체 회사

## 서버 초기화

서버 시작 시 `SearchRuntimeConfig.from_environment()`로 설정을 만들고
`OperationalSearchExecutor`를 한 번만 생성한다. BGE-M3와 Chroma 클라이언트를
요청마다 다시 로드하면 안 된다. 각 요청에서는 생성해 둔 실행기의 `run()`만
호출한다.

경로가 기본값과 다르면 다음 환경변수로 덮어쓸 수 있다.

- `SEARCH_MANIFEST_PATH`
- `SEARCH_UNIVERSE_PATH`
- `SEARCH_CORRECTION_EDGES_PATH`
- `SEARCH_CHROMA_DB_PATH`
- `SEARCH_COMPANY_MAP_PATH`
- `SEARCH_EMBEDDING_MODEL`
- `SEARCH_DEVICE`: `cpu` 또는 `cuda`

## 터미널 검증

저장소 루트에서 경로만 먼저 확인한다.

```text
python -m search_info.run_operational_context --check-only
```

원 질문, 검색계획, 키워드 추출 결과가 들어 있는 입력 JSON을 실제 DB에서
실행한다.

```text
python -m search_info.run_operational_context examples/samsung_2023_2025_search_input.json --debug
```

기본 결과는 `logs/search_executor_result.json`에 저장된다. 별도 파일은
`--output`으로 지정한다.

입력 JSON의 최상위 필드는 다음 세 개다.

- `question`
- `search_plan`
- `keyword_extractor_output`

## 디버그 출력

`--debug` 또는 `OperationalSearchExecutor(..., debug=True)`를 사용하면 다음
단계가 짧게 출력된다.

- `CONFIG`: 운영 경로와 모델
- `CATALOG`: manifest·회사·문서유형·정정 그래프
- `VALIDATE`: 검색계획 검증
- `SCOPE`: 회사·문서유형 범위
- `PERIOD`: 날짜 기준, 문서 수, event_date 대체 수
- `GRAPH`: 정정 체인 확장 수
- `DENSE`: 회사별 HNSW 후보 확장과 필터 결과
- `LEXICAL`: BM25 인덱스와 결과
- `RETRIEVE`, `FUSE`, `RERANK`, `EXPAND`, `PACK`: 각 처리 결과 수
- `ERROR`: 실패한 단계의 예외 유형과 메시지

## 테스트

```text
python -m unittest tests.test_search_pipeline -v
python search_info/run_context_pipeline_example.py
python -m search_info.validate_search_plan_dataset <학습데이터.jsonl>
```

테스트에는 실제 삼성전자 2023·2025년 사업보고서 컨텍스트, 입력 계약,
회사 범위, 정정 체인, event_date 누락 대체, 회사별 Chroma의 unfiltered HNSW
후보 확장이 포함된다.
