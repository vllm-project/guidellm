---
weight: -4
source_sha256: !!str "da734b0200c8e7a70ee9d46cf7f0c84440ee052c98d0661dab48a9b60310b6dd"
---

# 결과 분석

[벤치마크 실행](benchmark.md)을 완료하면 GuideLLM은 LLM 배포 환경의 성능을 파악하는 데 도움이 되는 종합적인 결과를 제공합니다. 이 가이드에서는 콘솔 출력과 파일로 저장된 결과를 해석하는 방법을 설명합니다.

## 콘솔 출력 이해하기

벤치마크가 완료되면 GuideLLM은 결과를 자동으로 콘솔에 표시하며, 결과는 크게 세 가지 주요 섹션으로 나뉩니다:

### 1. 벤치마크 메타데이터

이 섹션에서는 벤치마크 실행에 대한 전반적인 요약 정보를 제공하며, 다음 항목을 포함합니다:

- **서버 설정**: 대상 URL, 모델 이름 및 백엔드 세부 정보
- **데이터 설정**: 데이터 소스, 토큰 수 및 데이터셋 속성
- **프로파일 인자**: 요청률 유형, 최대 실행 시간, 요청 수 제한 등
- **추가 정보**: `--output-extras` 인자를 통해 제공된 추가 메타데이터

예제:

```
Benchmarks Metadata
------------------
Args:        {"backend_type": "openai", "target": "http://localhost:8000", "model": "Meta-Llama-3.1-8B-Instruct-quantized", ...}
Worker:      {"type_": "generative", "backend_type": "openai", "backend_args": {"timeout": 120.0, ...}, ...}
Request Loader: {"type_": "generative", "data_args": {"prompt_tokens": 256, "output_tokens": 128, ...}, ...}
Extras:      {}
```

### 2. 벤치마크 정보

이 섹션에서는 각 벤치마크 실행에 대한 주요 정보를 다음 열로 구성된 표로 요약합니다:

- **유형**: 벤치마크 유형(예: synchronous, constant, poisson 등)
- **시작/종료 시각**: 벤치마크가 시작되고 종료된 시각
- **소요 시간**: 벤치마크의 총 소요 시간(초)
- **요청 수**: 성공한 요청, 완료되지 않은 요청 및 오류가 발생한 요청의 수
- **토큰 통계**: 프롬프트와 출력의 평균 토큰 수 및 총 토큰 수

이 섹션에서는 실행된 내용을 파악하고 결과를 빠르게 살펴볼 수 있습니다.

### 3. 벤치마크 통계

성능 분석에서 가장 중요한 섹션으로, 각 지표에 대한 상세 통계를 표시합니다:

- **처리량 지표**:

  - 초당 요청 수(RPS)
  - 동시 요청 수
  - 초당 출력 토큰 수
  - 초당 총 토큰 수

- **지연 시간 지표**:

  - 요청 지연 시간(평균, 중앙값, p99)
  - 첫 토큰 생성까지 걸리는 시간(TTFT)(평균, 중앙값, p99)
  - 토큰 간 지연 시간(ITL)(평균, 중앙값, p99)
  - 출력 토큰당 소요 시간(평균, 중앙값, p99)

p99(99번째 백분위수) 값은 요청의 99%에 대한 최악의 성능을 나타내므로, SLO 분석에서 특히 중요합니다.

## 저장된 결과 분석하기

GuideLLM은 더 심층적인 분석을 위해 기본적으로 다음 파일에 상세 결과를 저장합니다:

- `benchmarks.json`: JSON 형식의 전체 벤치마크 데이터
- `benchmarks.csv`: CSV 형식의 주요 지표 요약

파일은 `GUIDELLM__DEFAULT_RESULTS_DIR`로 설정한 디렉터리에 저장되며, 이 변수가 설정되어 있지 않으면 현재 디렉터리에 저장됩니다. 파일 이름과 저장 위치를 선택하려면 [파일 출력 설정](../guides/outputs.md#configuring-file-outputs)을 참고하세요.

### 파일 형식

사용 가능한 보고서는 [지원되는 파일 형식](../guides/outputs.md#supported-file-formats)을, 보고서를 선택하는 예시는 [출력 설정](../guides/outputs.md#cli-output-configuration)을 참고하세요. `--output` 옵션을 명시적으로 지정하면 기본 JSON 및 CSV 출력 대신 지정한 출력을 사용합니다. 콘솔 표시를 제어하는 방법은 [콘솔 출력](../guides/outputs.md#console-output)에 설명되어 있습니다.

### 프로그램을 통한 분석

사용자 지정 분석을 위해 Python에서 결과를 다시 불러올 수 있습니다:

```python
from guidellm.benchmark import GenerativeBenchmarksReport

# Load results from file
report = GenerativeBenchmarksReport.load_file("benchmarks.json")

# Access individual benchmarks
for benchmark in report.benchmarks:
    # Print basic info
    print(f"Benchmark: {benchmark.id_}")
    print(f"Type: {benchmark.type_}")

    # Access metrics
    print(f"Avg RPS: {benchmark.metrics.requests_per_second.successful.mean}")
    print(
        f"p99 latency: {benchmark.metrics.request_latency.successful.percentiles.p99}"
    )
    print(
        f"TTFT (p99): {benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99}"
    )
```

## 주요 성능 지표

결과를 분석할 때는 다음 주요 지표에 집중하세요:

### 1. 처리량 및 처리 용량

- **최대 RPS**: 서버가 처리할 수 있는 최대 요청률은 얼마인가요?
- **동시 요청 수**: 서버가 동시에 처리할 수 있는 요청은 몇 개인가요?
- **토큰 처리량**: 서버가 초당 생성할 수 있는 토큰은 몇 개인가요?

### 2. 지연 시간 및 응답성

- **첫 토큰 생성까지 걸리는 시간(TTFT)**: 모델이 얼마나 빠르게 출력 생성을 시작하나요?
- **토큰 간 지연 시간(ITL)**: 모델이 후속 토큰을 얼마나 매끄럽게 생성하나요?
- **전체 요청 지연 시간**: 요청을 처음부터 끝까지 완료하는 데 얼마나 걸리나요?

### 3. 신뢰성 및 오류율

- **성공률**: 성공적으로 완료되는 요청의 비율은 얼마인가요?
- **오류 분포**: 어떤 유형의 오류가 발생하며, 각 오류의 발생 비율은 얼마인가요?

## 추가 분석 기법

### 서로 다른 모델 또는 하드웨어 비교

서로 다른 모델이나 하드웨어 설정으로 벤치마크를 실행한 후 비교하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://server1:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=model1/benchmarks.json

guidellm run \
  --backend kind=openai_http,target=http://server2:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=model2/benchmarks.json
```

### 비용 최적화

다음 항목을 분석하여 비용 효율성을 계산하세요:

- 하드웨어 비용 1달러당 초당 생성 토큰 수
- 하드웨어 설정별 최대 처리량
- 최적의 배치 크기와 지연 시간 사이의 상충 관계

### 확장 요구 사항 파악

벤치마크 결과를 활용하여 다음 사항을 계획하세요:

- 예상 부하를 처리하는 데 필요한 서버 수
- 수요에 따라 서버 규모를 자동으로 확장하거나 축소할 시점
- 워크로드에서 가격 대비 성능이 가장 뛰어난 하드웨어
