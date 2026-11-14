---
weight: 3
source_sha256: !!str "8a23e6039413624e05c852697978047945eae971a85bc9a2115d6a24b8dfc7e2"
---

# 수집기

수집기는 GuideLLM 외부의 데이터를 벤치마크 보고서에 추가합니다. 내장된 `prometheus` 수집기는 각 벤치마크가 실행되는 동안 서버의 Prometheus 엔드포인트에서 지표를 수집합니다. 이를 통해 서버의 큐 길이, KV 캐시 사용량 및 지연 시간을 GuideLLM 자체 측정값과 함께 확인할 수 있습니다. 예를 들어, 요청이 큐에 쌓여서 지연 시간이 늘어났는지, KV 캐시가 가득 차서 늘어났는지 파악할 수 있습니다.

## 빠른 시작

`--collector`에 서버의 지표 엔드포인트를 지정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=60 \
  --collector kind=prometheus,url=http://localhost:8000/metrics
```

JSON 또는 YAML 보고서의 각 벤치마크에는 `collectors` 리스트가 추가됩니다. 수집기를 지정한 순서대로 수집기당 하나의 결과가 포함됩니다(`jq '.benchmarks[].collectors' benchmarks.json`). 각 결과에는 해당 결과를 생성한 수집기의 이름을 나타내는 `kind` 필드가 포함됩니다. 콘솔, CSV 및 HTML 출력에는 수집기 결과가 표시되지 않습니다.

## 수집기 실행 방식

모든 수집기는 하나의 별도 프로세스를 공유하므로 요청을 전송하는 프로세스의 실행 시간을 사용하지 않습니다. 각 벤치마크의 요청이 시작되고 끝날 때 해당 프로세스에 알림이 전달됩니다. 이후 각 수집기에 벤치마크 측정 구간의 결과를 요청합니다(워밍업과 쿨다운은 제외됩니다).

엔드포인트에 접근할 수 없는 경우처럼 수집기에 오류가 발생해도 벤치마크는 계속 실행됩니다. 이때 수집기는 결과 대신 `kind`와 `error`를 보고하므로, 리스트의 항목은 지정한 수집기와 계속 일대일로 대응합니다.

## Prometheus 수집기

### 옵션

| 옵션       | 기본값           | 설명                                                        |
| ---------- | ---------------- | ----------------------------------------------------------- |
| `url`      | 필수             | 지표를 수집할 Prometheus 엔드포인트                         |
| `interval` | `1.0`            | 벤치마크 실행 중 지표를 수집하는 간격(초)                   |
| `timeout`  | `5.0`            | 한 번의 수집을 실패로 처리하기 전까지 대기할 시간(초)       |
| `metrics`  | vLLM의 주요 지표 | 유지할 지표군. 엔드포인트에서 제공하지 않는 지표군은 건너뜀 |

백엔드 대상이 게이트웨이나 로드 밸런서를 가리킬 수 있으므로, `url`은 백엔드 대상 주소와 별도로 설정합니다. 여러 서버에서 수집하려면 서버마다 `--collector`를 한 번씩 지정하세요. 로드 밸런서를 대상으로 지정하면 수집할 때마다 서로 다른 복제본에 접근하게 됩니다.

기본 `metrics`에는 vLLM의 큐, KV 캐시, 토큰, 요청 결과 및 지연 시간 지표가 포함됩니다. 이전 vLLM 릴리스에서 제공하는 `vllm:gpu_cache_usage_perc`와 `vllm:time_per_output_token_seconds`도 포함됩니다. 지표를 직접 선택하려면 수집기를 JSON 형식으로 지정하세요:

```bash
--collector '{"kind": "prometheus", "url": "http://localhost:8000/metrics", "interval": 0.5, "metrics": ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"]}'
```

### 지표 수집

벤치마크의 첫 요청 직전에 엔드포인트에서 지표를 수집하고, 실행 중에는 `interval`초마다 수집하며, 마지막 요청이 끝난 뒤 한 번 더 수집합니다. 수집에 실패하면 로그를 기록하고 `scrape_errors`에 집계하며, 벤치마크는 계속 실행됩니다. `scrapes`가 0이면 엔드포인트에 한 번도 접근하지 못했다는 뜻이며, 일반적으로 URL이 잘못된 경우에 해당합니다. 파싱할 수 없는 줄은 건너뜁니다.

### 결과

지표는 측정 구간을 기준으로 요약되며, 레이블 조합마다 별도의 시계열로 구분됩니다(예: `model_name` 또는 데이터 병렬 배포 환경의 `engine`).

- `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`와 같은 **게이지**(Gauge)는 `summary`(평균, 최솟값, 최댓값 및 백분위수)와 `(timestamp, value)` 쌍으로 구성된 원본 `samples`를 보고합니다.
- `vllm:request_success`, `vllm:generation_tokens`와 같은 **카운터**(Counter)는 측정 구간의 증가량인 `increase`와 초당 증가율인 `rate`를 보고합니다. 수집 시각이 측정 구간의 시작과 끝에 정확히 일치하는 경우는 드물기 때문에, 각 경계의 값은 보간하여 계산합니다. 값이 감소하면 Prometheus의 `increase()`처럼 카운터가 초기화된 것으로 처리합니다.
- `vllm:time_to_first_token_seconds`와 같은 **히스토그램**(Histogram)은 `count`, `sum`, `mean` 및 `quantiles`(`p50`, `p90`, `p95`, `p99`)를 보고하며, Prometheus의 `histogram_quantile()`처럼 버킷 내부에서 보간합니다. vLLM의 지연 시간 버킷은 간격이 넓으므로 이 값은 대략적인 추정치로 참고하고, 정확한 값은 GuideLLM 자체 백분위수를 사용하세요.

결과는 다음과 같습니다(일부 생략):

```json
"collectors": [
  {
    "kind": "prometheus",
    "source": "http://localhost:8000/metrics",
    "scrapes": 62,
    "scrape_errors": 0,
    "gauges": {
      "vllm:num_requests_waiting": [
        {
          "labels": {"engine": "0", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "summary": {"mean": 12.4, "max": 31.0, "...": "..."},
          "samples": [[1760000000.5, 0.0], [1760000001.5, 4.0], "..."]
        }
      ]
    },
    "counters": {
      "vllm:request_success": [
        {
          "labels": {"engine": "0", "finished_reason": "length", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "increase": 584.0,
          "rate": 9.73
        }
      ]
    },
    "histograms": {
      "vllm:time_to_first_token_seconds": [
        {
          "labels": {"engine": "0", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "count": 584.0,
          "sum": 146.2,
          "mean": 0.25,
          "quantiles": {"p50": 0.21, "p90": 0.42, "p95": 0.61, "p99": 0.93}
        }
      ]
    }
  }
]
```

### 제한 사항

- 측정 구간이 수집 간격의 두 배보다 짧으면 수집 횟수가 적으므로 GuideLLM이 경고를 기록합니다. 짧게 실행할 때는 `interval`을 줄이세요.
- 서버 지표에는 GuideLLM의 요청뿐 아니라 서버가 처리한 모든 요청이 집계됩니다. 따라서 다른 작업이 서버를 사용하지 않을 때만 클라이언트 측 결과와 일치합니다.
- 엔드포인트는 Prometheus 텍스트 형식을 제공해야 합니다. vLLM은 기본적으로 이 형식을 제공합니다.
