---
source_sha256: !!str "d2531d2fe99d6fce9387c5c51758a02170864469670fbe2bd75bd5203e85d42f"
---

# 이미지 벤치마킹

이 가이드에서는 비전 언어 모델(VLM)의 시각 질의응답(VQA, `chat/completions`)과 이미지 설명 생성(`chat/completions`) 성능을 벤치마킹하는 방법을 설명합니다.

## 설정

먼저 OpenAI Chat API와 호환되는 모델이 추론 서버에서 실행 중인지 확인하세요. GuideLLM은 chat/completions 엔드포인트를 통해 이미지를 입력받을 수 있는 모든 OpenAI 호환 서버를 지원합니다. 아래 예제에서는 vLLM으로 Qwen3-VL 모델을 실행합니다.

```bash
# Qwen3-VL VQA/Captioning
vllm serve Qwen/Qwen3-VL-2B-Instruct
```

다음으로, 같은 인스턴스나 서버에 접근할 수 있는 다른 장비(권장)에 시각 데이터 처리에 필요한 의존성과 함께 GuideLLM을 설치하세요:

```bash
pip install guidellm[vision,recommended]
```

마지막으로, 지원되는 형식의 이미지 파일이 포함된 벤치마크용 데이터셋을 준비하세요. GuideLLM은 Hugging Face 데이터셋, 로컬 파일, URL 등에서 이미지 데이터를 불러올 수 있습니다. 아래 예제에서는 `lmms-lab/MMBench_EN` 데이터셋을 사용합니다.

## 처리 옵션

`--profile`, 프로파일의 요청률 파라미터, `--constraint kind=max_requests,count=<n>` 등 일반 벤치마크 인자는 이미지 작업에도 동일하게 적용됩니다. 이미지 데이터를 처리하고 요청을 구성하는 방식은 다음 추가 옵션으로 설정할 수 있습니다.

### 데이터 로딩

GuideLLM은 여러 방식으로 이미지 데이터를 불러올 수 있습니다. 먼저 GuideLLM이 데이터 소스를 역직렬화하여 Hugging Face 데이터셋으로 변환할 수 있어야 합니다. 로컬 파일, Hugging Face 데이터셋, JSON 파일 등을 사용할 수 있습니다.

데이터 소스에서 사용할 이미지 열은 GuideLLM의 이미지 디코더와 인코더가 지원하는 형식이어야 합니다. 지원되는 형식은 다음과 같습니다:

- Hugging Face의 Image 특성(권장)
- 로컬 파일 경로(예: .jpg, .png)
- 이미지 파일의 URL
- Base64로 인코딩한 이미지 데이터
- NumPy 배열 또는 PIL 이미지

### 데이터 열 매핑

데이터셋을 지정할 때는 사용할 이미지 열을 GuideLLM의 `image_column`에 매핑하여 어떤 데이터를 이미지로 처리할지 알려주세요. 지정하지 않으면 GuideLLM이 image, picture 등 일반적으로 사용하는 이름을 기준으로 이미지 열을 자동으로 찾습니다.

열 매핑은 `--data-column-mapper` 인자에 JSON 문자열로 전달합니다. `image_column`의 값으로 데이터셋에 있는 열 이름을 지정하세요. 예를 들어 이미지 열의 이름이 `photo`이면 다음과 같이 설정합니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"image_column": "photo"}}'
```

여러 데이터셋을 결합하는 경우(예: 프롬프트와 이미지를 각각 다른 데이터셋에서 가져오는 경우) 열 이름 앞에 0부터 시작하는 데이터셋 인덱스나 데이터셋 별칭을 붙이고 점으로 구분하세요. 예를 들어 이미지 열이 두 번째 데이터셋(인덱스 1)에 있으면 다음과 같이 지정합니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.image_column": "photo"}}'
```

### 요청 형식

지원되는 각 엔드포인트에서 요청 포매터는 이미지 데이터를 인코딩하고 요청 페이로드를 구성합니다. 기본 설정을 그대로 사용하거나 필요에 따라 변경할 수 있습니다. 다음 이미지 요청 옵션을 `--request-formatter-kwargs` 인자에 JSON 문자열로 전달하세요.

#### "encode_kwargs"

이미지 인코더에 전달하는 인자 딕셔너리입니다. 요청에 이미지를 포함하기 전에 이미지 데이터를 어떻게 전처리할지 지정합니다.

**중첩 구조 사용 시 참고 사항:**

- **Chat Completions**(`chat_completions`)의 경우 이미지 인자는 `encode_kwargs` 안의 `image` 키 아래에 지정해야 합니다.

지원되는 인자는 다음과 같습니다:

- "max_size": 이미지에서 가장 긴 변의 최대 길이(픽셀)입니다. 가로세로 비율을 유지하면서 큰 이미지를 축소할 때 유용합니다.
- "max_width": 이미지의 최대 너비(픽셀)입니다. 높이도 함께 지정하지 않으면 가로세로 비율을 유지합니다.
- "max_height": 이미지의 최대 높이(픽셀)입니다. 너비도 함께 지정하지 않으면 가로세로 비율을 유지합니다.
- "width": 이미지 너비를 지정한 픽셀 수로 조정합니다.
- "height": 이미지 높이를 지정한 픽셀 수로 조정합니다.
- "encode_type": 요청에 이미지를 포함하는 방식입니다.
  - "base64"(기본값): URL이면 이미지를 다운로드하고, 파일이면 이미지를 읽습니다. 크기 조정 등의 처리를 마친 뒤 Base64 문자열로 인코딩합니다.
  - "url": 요청에 이미지 URL을 직접 전달합니다. 추론 서버에서 해당 URL에 접근할 수 있어야 하며, 크기 조정 옵션은 사용할 수 없습니다.

**예제:**

**Chat Completions**에서 이미지의 가장 긴 변을 최대 512픽셀로 제한하려면 다음과 같이 설정하세요:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"image": {"max_size": 512}}}'
```

#### "extras"

요청에 직접 포함할 추가 인자 딕셔너리입니다. extras에서 추가 인자를 포함할 위치를 지정할 수 있습니다:

- "headers": 요청 헤더에 포함
- "params" / "body": 요청 파라미터 또는 본문에 포함

예를 들어 특정 시스템 프롬프트나 다른 요청 본문 파라미터를 지정하려면 다음과 같이 설정하세요:

```bash
--request-formatter-kwargs '{"extras": {"body": {"temperature": 0.7}}}'
```

#### "stream"

백엔드가 지원하는 경우 불리언 값으로 응답 스트리밍을 활성화하거나 비활성화할 수 있습니다. 기본값은 활성화입니다. 비활성화하려면 백엔드 설정에 `stream=false`를 전달하세요:

```bash
--backend kind=openai_http,target=http://localhost:8000,stream=false
```

## 예상 결과

GuideLLM은 요청의 전체 처리 과정에서 지표를 수집하여 `GenerativeRequestStats`에 저장하고 `GenerativeMetrics`에 집계합니다. 결과는 콘솔에 표시되며, 추가 분석을 위해 로컬 파일에도 저장됩니다.

### 출력 파일

- **`benchmarks.json`**: 스케줄러의 시간 정보, 요청 분포, 텍스트와 이미지에 대한 상세 지표 요약을 포함하는 전체 통계 객체를 계층 구조로 저장합니다.
- **`benchmarks.csv`**: 벤치마크별로 처리량, 지연 시간 및 입출력 지표를 한 행에 요약하여 저장합니다. 개별 요청의 상세 데이터는 포함하지 않습니다.
- **`benchmarks.html`**: 성능을 시각적으로 요약한 보고서입니다.

### 수집되는 지표

이미지 벤치마크에서는 지연 시간, 첫 토큰 생성까지 걸리는 시간(TTFT), 토큰 간 지연 시간(ITL) 등 일반적인 성능 지표와 함께 입력과 출력에 대해 다음 사용량 지표를 추적합니다:

- **이미지 수**: 처리한 이미지의 수입니다.
- **이미지 픽셀 수**: 전체 픽셀 수입니다.
- **이미지 바이트 수**: 이미지 페이로드의 크기(바이트)입니다.
- **이미지 토큰 수**: 확인할 수 있는 경우 이미지를 표현하는 데 사용한 토큰 수입니다.
- **텍스트 지표**: 프롬프트와 응답의 토큰 수, 단어 수 및 문자 수도 추적합니다.

### 통계 분석

GuideLLM은 위의 각 지표에 대해 다음 통계를 계산합니다:

- **측정값**: 평균, 중앙값, P95, P99, 최솟값 및 최댓값입니다.
- **처리율**: 초당 처리량입니다(예: `image_pixels/sec`).
- **동시 처리 수**: 동시에 처리 중인 작업의 수를 나타냅니다.

이 통계는 `StatusDistributionSummary` 구조를 사용하여 성공한 요청, 완료되지 않은 요청, 오류가 발생한 요청을 구분하여 추적합니다.

## 예제

### 1. 시각 질의응답(VQA)

이 벤치마크에서는 비전 언어 모델이 이미지에 관한 질문에 답하는 능력을 평가합니다. 이미지와 텍스트 질문이 포함된 `lmms-lab/MMBench_EN` 데이터셋을 사용합니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://192.168.4.12:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"lmms-lab/MMBench_EN","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"image_column":"image","text_column":"question"}}'
```

**주요 파라미터**

- `--backend`: 서버 URL, 모델 및 chat completions 엔드포인트
- `--data`: 불러올 데이터셋. `kind=huggingface`와 함께 데이터셋 식별자를 지정하고, `load_kwargs`로 로딩 옵션을 설정합니다(여기서는 "test" 분할 선택). 지원되는 전체 옵션은 [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset)을 참고하세요
- `--data-column-mapper`: 데이터셋의 `image` 열을 `image_column`에, `question` 열을 `text_column`에 매핑

이 명령어는 MMBench_EN 데이터셋의 프롬프트 텍스트와 이미지를 사용하여 대상 서버의 chat/completions 엔드포인트를 벤치마킹합니다. 다음과 비슷한 결과가 출력됩니다:

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```

### 2. 이미지 설명 생성

이 벤치마크에서는 모델이 구체적인 질문 없이(또는 일반적인 프롬프트만으로) 이미지를 설명하는 능력을 평가합니다. 여기서는 텍스트 열을 생략하고 이미지만 전송합니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"lmms-lab/MMBench_EN","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"image_column":"image"}}'
```

**주요 파라미터:**

- `--data-column-mapper`: `image_column`만 매핑합니다. 이미지만 전송하여 모델이 이미지의 캡션이나 설명을 생성하도록 하는 요청에 사용합니다.

이 명령어는 MMBench_EN 데이터셋의 이미지를 프롬프트로 사용하여 대상 서버의 chat/completions 엔드포인트를 벤치마킹합니다. 다음과 비슷한 결과가 출력됩니다:

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```
