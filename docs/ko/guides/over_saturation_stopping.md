---
weight: 4
source_sha256: !!str "7a76866522ee0c2d964aef4f2b76328407ecb21f0d4ac44c91e0d346a929e561"
---

# 과포화 시 중단

GuideLLM은 모델이 과포화 상태에 도달하면 벤치마크를 자동으로 중단하는 과포화 감지(OSD) 기능을 제공합니다. 응답률이 요청률을 따라가지 못하는 상황을 감지하여 컴퓨팅 리소스 낭비를 방지하고 벤치마크 결과의 유효성을 유지합니다.

## 과포화란?

LLM 추론 서버가 요청을 처리하는 속도보다 더 빠르게 요청을 받으면 과포화가 발생하고 요청이 큐에 쌓입니다. 큐가 길어질수록 서버가 각 요청의 처리를 시작하기까지 더 오래 걸리며, 성능 지표도 저하됩니다. 성능 벤치마킹 도구가 LLM 추론 서버에 과도한 부하를 가하면 측정 지표가 크게 왜곡되어 참고할 수 없는 결과가 됩니다.

갑자기 손님이 몰려 계산원이 감당하지 못하는 상황을 떠올려 보세요. 대기 줄(부하)이 길어지면 계산원이 손님을 제때 응대하지 못하고, 줄은 계속 길어져 더 이상 손님을 받을 공간이 없어집니다. 과포화를 자동으로 감지하여 벤치마크를 중단하면 고가의 컴퓨팅 자원이 불필요하게 사용되는 것을 막을 수 있습니다.

## 동작 방식

GuideLLM의 과포화 감지(OSD) 알고리즘은 통계적인 기울기 감지를 통해 모델이 과포화 상태에 도달했는지 판단합니다. 이 알고리즘은 시간에 따라 다음 두 가지 핵심 지표를 추적합니다:

1. **동시 요청 수**: 동시에 처리 중인 요청의 수
2. **첫 토큰 생성까지 걸리는 시간(TTFT)**: 각 응답의 첫 토큰이 생성될 때까지의 지연 시간

알고리즘은 각 지표에 대해 다음 작업을 수행합니다:

- 최근 데이터 포인트로 구성된 슬라이딩 윈도우를 유지합니다
- 데이터를 받을 때마다 갱신하는 통계량을 사용하여 선형 회귀 기울기를 계산합니다
- t 분포의 신뢰 구간을 사용하여 오차 범위(MOE)를 계산합니다
- 오차 범위가 작은 양의 기울기를 감지하여 성능 저하를 판단합니다

다음 조건이 충족되면 과포화로 판단합니다:

- 동시 요청 수와 TTFT 모두 통계적으로 유의한 양의 기울기를 보임
- 최소 실행 시간 임곗값에 도달함
- 기울기를 신뢰할 수 있게 추정할 만큼 충분한 데이터 포인트가 있음

과포화가 감지되면 제약 조건에 따라 새 요청을 큐에 추가하는 작업이 자동으로 중단됩니다. 필요에 따라 기존 요청의 처리도 중단하여 추가적인 리소스 낭비를 방지할 수 있습니다.

## 사용법

제약 조건 설정에서 `kind=over_saturation`을 사용하여 과포화 감지를 설정합니다:

```bash
--constraint kind=over_saturation,...
```

### 기본 사용법

기본 설정으로 과포화 감지를 활성화하려면 다음과 같이 실행하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=throughput,max_concurrency=10 \
  --constraint kind=over_saturation
```

### 고급 설정

제약 조건 설정 문자열에 감지 파라미터를 지정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=16 \
  --constraint kind=over_saturation,mode=enforce,min_seconds=60,max_window_seconds=300,moe_threshold=1.5
```

또는 JSON을 사용하세요:

```bash
--constraint '{"kind":"over_saturation","mode":"enforce","min_seconds":60,"max_window_seconds":300,"moe_threshold":1.5}'
```

## 설정 옵션

과포화 감지를 활성화할 때 다음 파라미터를 설정할 수 있습니다:

- **`mode`** (`"enforce"` | `"monitor"`, 기본값: `"enforce"`): 과포화가 감지되었을 때의 동작을 제어합니다. `"enforce"`는 벤치마크를 중단하고, `"monitor"`는 모니터링과 보고만 수행합니다.
- **`min_seconds`** (float, 기본값: `30.0`): 과포화 검사를 시작하기 전까지 대기할 최소 시간(초)입니다. 초기 워밍업 단계에서 발생할 수 있는 오탐을 방지합니다.
- **`max_window_seconds`** (float, 기본값: `120.0`): 데이터를 유지할 최대 시간 구간(초)입니다. 오래된 데이터 포인트를 자동으로 제거하여 메모리 사용량을 제한합니다.
- **`moe_threshold`** (float, 기본값: `2.0`): 기울기 감지에 사용하는 오차 범위 임곗값입니다. 값이 작을수록 성능 저하를 더 민감하게 감지합니다.
- **`minimum_ttft`** (float, 기본값: `2.5`): 위반 횟수를 집계하는 기준이 되는 최소 TTFT 임곗값(초)입니다. 이 임곗값을 초과하는 TTFT 값만 위반으로 집계합니다.
- **`maximum_window_ratio`** (float, 기본값: `0.75`): 전체 요청 수 대비 윈도우 크기의 최대 비율입니다. 추적할 요청 수의 상한을 정하여 메모리 사용량을 제한합니다.
- **`minimum_window_size`** (int, 기본값: `5`): 기울기 추정에 필요한 최소 데이터 포인트 수입니다. 과포화 여부를 판단하기 전에 통계적 신뢰성을 확보합니다.
- **`confidence`** (float, 기본값: `0.95`): t 분포 계산에 사용하는 통계적 신뢰 수준(0-1)입니다. 값이 높을수록 과포화로 판단하기 위해 더 강한 근거가 필요합니다.

## 활용 사례

과포화 감지는 특히 다음과 같은 상황에서 유용합니다:

### 스트레스 테스트 및 용량 계획

점점 증가하는 부하를 시스템이 어떻게 처리하는지 테스트할 때, 과포화 감지는 시스템이 더 이상 부하를 감당하지 못하면 벤치마크를 자동으로 중단합니다. 이를 통해 유효하지 않은 결과를 얻는 데 컴퓨팅 시간을 낭비하지 않도록 합니다.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep,sweep_size=5 \
  --constraint kind=over_saturation
```

### 비용 효율적인 벤치마킹

여러 모델, GPU 및 설정 조합을 대상으로 대규모 벤치마크를 실행할 때, 과포화 감지는 유효하지 않은 실행을 조기에 중단하여 비용을 크게 줄일 수 있습니다.

### 안전한 운영 범위 찾기

과포화 감지를 사용하면 배포 환경에서 지속적으로 유지할 수 있는 최대 처리량을 파악할 수 있습니다. 이를 바탕으로 적절한 요청률 제한과 용량 계획 목표를 설정하세요.

## 결과 해석

과포화 감지를 설정하면(`"enforce"` 또는 `"monitor"` 모드 모두 해당) 벤치마크 출력에 감지 상태를 나타내는 메타데이터가 포함됩니다. 이 정보는 스케줄러 동작 메타데이터에서 확인할 수 있으며, 다음 항목을 포함합니다:

- **`is_over_saturated`** (bool): 평가 시점에 과포화가 감지되었는지 여부
- **`concurrent_slope`** (float): 동시 요청 수에 대해 계산한 기울기
- **`concurrent_slope_moe`** (float): 동시 요청 수 기울기의 오차 범위
- **`concurrent_n`** (int): 동시 요청 수의 기울기를 계산하는 데 사용된 데이터 포인트 수
- **`ttft_slope`** (float): 계산된 TTFT의 기울기
- **`ttft_slope_moe`** (float): TTFT 기울기의 오차 범위
- **`ttft_n`** (int): TTFT의 기울기를 계산하는 데 사용된 데이터 포인트 수
- **`ttft_violations`** (int): 최소 임곗값을 초과한 TTFT 값의 수

이 지표를 통해 과포화가 감지된 이유를 파악하고, 필요에 따라 감지 파라미터를 세부적으로 조정할 수 있습니다.

## 예제: 과포화 감지를 사용하는 전체 벤치마크

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=16 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=300 \
  --constraint kind=over_saturation,mode=enforce,min_seconds=30,max_window_seconds=120 \
  --output kind=json,path=benchmark.json \
  --output kind=html,path=benchmark.html
```

이 예제는 다음 작업을 수행합니다:

- 16개의 동시 요청으로 벤치마크 실행
- 프롬프트 토큰 256개와 출력 토큰 128개로 구성된 합성 데이터 사용
- 사용자 지정 시간 파라미터로 과포화 감지 활성화
- 보조 제한으로 최대 실행 시간을 300초로 설정
- 결과를 JSON 및 HTML 형식으로 출력

## 추가 자료

알고리즘 개발, 평가 지표 및 구현 세부 사항을 포함하여 과포화 감지에 대해 자세히 알아보려면 다음 Red Hat Developer 블로그 글을 참고하세요:

- [과포화 감지로 LLM 벤치마킹 비용 줄이기(영문)](https://developers.redhat.com/articles/2025/11/18/reduce-llm-benchmarking-costs-oversaturation-detection) - 과포화 문제와 이 문제가 LLM 벤치마킹에서 중요한 이유 소개
- [성공 기준 정의: 과포화 감지를 위한 평가 지표와 데이터 증강(영문)](https://developers.redhat.com/articles/2025/11/20/oversaturation-detection-evaluation-metrics) - 사용자 지정 지표, 데이터셋 레이블링 및 부하 증강 기법을 통해 OSD 알고리즘의 성능을 평가하는 방법
- [반복적인 오류 분석을 통한 과포화 감지기 구현(영문)](https://developers.redhat.com/articles/2025/11/24/building-oversaturation-detector-iterative-error-analysis) - OSD 알고리즘의 구현 과정을 자세히 설명
