# Mirae Asset AI Festival - Disclosure QA Agent

기업 공시 데이터를 기반으로 사용자 질의에 답변하는 HyperCLOVA X 기반 AI Agent입니다.

사용자 질의를 분석해 `검색_정보추출`, `다중조회_비교연산`, `복합문서추론`으로 분류하고, 공시 검색·재정렬·필요 시 Python 연산·HyperCLOVA X 기반 답변 생성·근거 검증 과정을 거쳐 최종 답변을 생성합니다.

---

## 1. 실행 환경

- OS: Ubuntu 24.04 LTS
- Python: Python 3.x
- Python 가상환경: `.venv`
- API Server: FastAPI + Uvicorn
- Vector DB: ChromaDB
- Embedding Model: `BAAI/bge-m3`
- LLM: HyperCLOVA X

현재 프로젝트는 Docker 기반 실행 환경을 사용하지 않습니다.

---

## 2. 환경 구성

저장소를 clone한 뒤 프로젝트 디렉터리로 이동합니다.

```bash
git clone https://github.com/nowuri148/miraeasset-AI-festival.git
cd miraeasset-AI-festival
```

Python 가상환경을 생성하고 활성화합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

필요한 Python 패키지를 설치합니다.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

HyperCLOVA X 호출에 필요한 인증 정보는 프로젝트에서 사용하는 환경 변수 또는 설정 방식에 맞게 구성합니다.

API Key, Secret 등의 민감 정보는 Git 저장소에 포함하지 않습니다.

---

## 3. 데이터 준비

실행을 위해 공시 Corpus와 ChromaDB가 필요합니다.

```text
corpus/
├─ universe.csv
├─ manifest.jsonl
├─ raw/
├─ derived/
│  ├─ manifest_v3.jsonl
│  └─ graph_edges_v1.jsonl
└─ vector_db/
   ├─ chroma_company/
   └─ company_collection_map.json
```

회사별 벡터 검색은 다음 ChromaDB를 사용합니다.

```text
corpus/vector_db/chroma_company
```

회사명과 Chroma collection 간 매핑은 다음 파일을 사용합니다.

```text
corpus/vector_db/company_collection_map.json
```

`corpus/derived/` 및 `corpus/vector_db/`는 Git 저장소에 포함하지 않으므로 새로운 환경에서 실행할 경우 별도로 동일한 데이터를 준비해야 합니다.

---

## 4. 시스템 처리 구조

```text
사용자 질문
  ↓
Keyword Extractor
  ↓
Task Routing
  ├─ 검색_정보추출
  ├─ 다중조회_비교연산
  └─ 복합문서추론
  ↓
검색 계획 생성
  ↓
공시 검색 / Reranking
  ↓
요청·기업·연도 단위 근거 구성
  ↓
필요 시 Python 연산
  ↓
HyperCLOVA X 답변 생성
  ↓
Grounding Validation
  ↓
최종 답변 + 공시 출처
```

### 질의 분석

질문에서 다음 정보를 구조화합니다.

- 기업 및 기업 범위
- 기간
- 문서 유형
- 작업 유형
- 지표
- 행동 및 주제 키워드
- 모호 후보 및 재질문 정보

질문이 충분히 구체적이지 않은 경우 임의로 범위를 확정하지 않고 추가 정보를 요청할 수 있도록 구성했습니다.

### 검색 계획 A/B

검색 계획 LLM은 질문과 키워드 추출 결과를 바탕으로 하나 이상의 검색 요청을 생성합니다.

- **모델 A**: 질문에 명시된 기업·기간·지표를 중심으로 보수적으로 검색 계획 생성
- **모델 B**: 모델 A의 답변이 근거 검증을 통과하지 못한 경우 동의어·관련 표현·보완 문서 유형 등을 확장하여 한 차례 재검색

모델 B에서도 기업·기간·지표 범위는 유지하여 검색 범위가 임의로 확장되지 않도록 합니다.

### 공시 검색

검색은 기업 범위, 문서 유형, 기간을 기준으로 후보를 제한한 뒤 BGE-M3 기반 Dense 검색과 키워드 검색을 수행하고 후보를 재순위화합니다.

여러 기업·연도·지표가 포함된 질문에서는 전체 상위 청크만 선택하지 않고 **요청×기업×연도 단위로 근거를 우선 배정**하여 특정 조건의 근거가 누락되지 않도록 구성했습니다.

### 비교 및 수치 연산

차이, 합계, 평균, 증감률 등 계산이 필요한 질문은 LLM이 직접 계산하지 않고 Python 연산기로 분리합니다.

```text
근거 검색
→ 연산 계획 생성
→ source_id 검증
→ Python 연산
→ 결과와 출처 반환
```

검색 결과에 실제 존재하는 source ID와 기업·기간·지표·단위가 일치하는 값을 사용하며, 근거가 부족하면 계산을 중단합니다.

### 복합문서 추론

여러 연도 또는 여러 공시를 함께 해석해야 하는 질문은 다음 정보를 최종 답변 모델에 전달합니다.

- 사용자 원문 질문
- 요청·기업·연도 정보
- 선택된 공시 청크와 메타데이터
- 사용 가능한 source ID
- 근거가 누락된 요청×연도 정보

최종 답변은 제공된 공시 청크 안의 근거만 사용하도록 제한하며, 사용한 source ID를 실제 공시명과 접수번호로 변환하여 함께 제공합니다.

예시:

```text
근거 공시:
- 삼성전자 · 사업보고서 (2023.12) (접수번호 20240312000736)
- 삼성전자 · 사업보고서 (2024.12) (접수번호 20250311001085)
```

### 근거 검증 및 재시도

최종 답변을 반환하기 전에 답변의 주장과 검색된 공시 근거가 실제로 일치하는지 검증합니다.

```text
PASS  → 답변 및 출처 반환
RETRY → 검색 계획 B로 재검색·재답변
```

이를 통해 검색 결과가 존재하는지뿐 아니라, 최종 답변의 수치와 설명이 실제 근거에 의해 지지되는지 확인합니다.

---

## 5. 실행 방법

### CLI 실행

```bash
source .venv/bin/activate
python main.py
```

실행 후 자연어 질의를 입력합니다.

```text
질문> 삼성전자의 2023~2025년 설비투자, 재고자산, DS부문 실적 변화를 설명해줘
```

종료:

```text
q
quit
exit
```

### API 서버 실행

```bash
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000
```

Health Check:

```bash
curl http://127.0.0.1:8000/health
```

---

## 6. 평가용 API

### End-point URL

```text
https://110.165.17.56/answer
```

### Health Check

```text
https://110.165.17.56/health
```

### 요청 방식

- Method: `GET`
- Path: `/answer`

### Query Parameters

| Parameter | Type | Description |
|---|---|---|
| `question_id` | string | 평가 질의 ID |
| `question` | string | 사용자 질의 원문 |

### 요청 예시

#### Linux / macOS

```bash
curl -G "https://110.165.17.56/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=삼성전자의 2023년 매출액은 얼마야?"
```

#### Windows PowerShell

PowerShell에서는 `curl`이 `Invoke-WebRequest` 별칭으로 동작할 수 있으므로 `curl.exe`를 사용합니다.

```powershell
curl.exe -G "https://110.165.17.56/answer" `
  --data-urlencode "question_id=Q-001" `
  --data-urlencode "question=삼성전자의 2023년 매출액은 얼마야?"
```

한 줄로 실행할 경우:

```powershell
curl.exe -G "https://110.165.17.56/answer" --data-urlencode "question_id=Q-001" --data-urlencode "question=삼성전자의 2023년 매출액은 얼마야?"
```

#### Health Check

Linux / macOS:

```bash
curl "https://110.165.17.56/health"
```

Windows PowerShell:

```powershell
curl.exe "https://110.165.17.56/health"
```

### 요청 스키마

실제 요청은 GET Query Parameter 방식으로 전달합니다.

```json
{
  "question_id": "string",
  "question": "string"
}
```

### 응답 스키마

```json
{
  "question_id": "string",
  "question": "string",
  "retrieved_context": "string",
  "think_trace": "string",
  "answer": "string"
}
```

### 응답 필드

| Field | Type | Description |
|---|---|---|
| `question_id` | string | 평가 질의 ID |
| `question` | string | 입력된 질의 원문 |
| `retrieved_context` | string | 답변 생성에 참고한 검색 문서 및 근거 |
| `think_trace` | string | 질의 처리 및 도구 사용 과정 |
| `answer` | string | 최종 생성 답변 |

---

## 7. 주요 프로젝트 구조

```text
miraeasset-AI-festival/
├─ main.py
├─ api.py
├─ requirements.txt
├─ question/
│  └─ keyword_extractor_ver2.py
├─ search_info/
│  ├─ search_pipeline.py
│  └─ z_search_task_hj.py
├─ calculate/
│  └─ calculate.py
├─ complex_info/
│  └─ z_complex_task.py
└─ corpus/
```

---

## 8. 재현 환경

새로운 환경에서는 다음 순서로 실행 환경을 구성합니다.

```bash
git clone https://github.com/nowuri148/miraeasset-AI-festival.git
cd miraeasset-AI-festival

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# HyperCLOVA X 인증 정보 설정
# Corpus 및 ChromaDB 데이터 배치

python main.py
```

API 방식으로 실행할 경우:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

재현 가능한 Python 의존성은 `requirements.txt`를 기준으로 관리합니다.
