---
source_sha256: !!str "e45f9ddcd4f665ee0310663d99b85bdb39bb44b10de50a168aaa1fe45272cac4"
---

# 경고 {#warnings}

GuideLLM은 각 벤치마크가 끝난 뒤 특정 지표 또는 두 지표의 비율이 임곗값을 초과하면 경고를 보고할 수 있습니다. 경고가 발생해도 실행은 정상적으로 완료됩니다. 경고는 콘솔 표 뒤에 출력되며, JSON 보고서의 각 벤치마크 항목에도 저장됩니다.

규칙은 생성형 지표 설정의 `metrics.warnings`에 정의합니다. 시나리오 파일이나 `--metrics`로 설정하세요. 이 필드를 생략하면 내장 규칙이 유지됩니다. 필드를 지정하면 내장 규칙을 대체하므로, 계속 사용할 규칙을 모두 포함해야 합니다.

## 내장 규칙 {#built-in-rules}

`metrics.warnings`를 생략하면 다음 네 가지 규칙이 적용됩니다:

- `generation_delay_ttft`:
  - `generation_delay`의 평균이 `time_to_first_token_ms.total` 평균의 1%를 초과하면 경고합니다.
  - 데이터 생성 지연이 서버의 TTFT에서 차지하는 비율이 커서 벤치마크 실행 속도를 떨어뜨릴 가능성이 있다는 뜻입니다. [요청이 예정 시각보다 늦게 로딩되는 경우](troubleshooting.md#requests-load-after-they-are-due)를 참고하세요.
- `worker_loop_lag_ttft`:
  - `worker_loop_lag`의 95번째 백분위수가 `time_to_first_token_ms.total` 평균의 10%를 초과하면 경고합니다.
  - GuideLLM 작업자의 이벤트 루프가 늦게 실행되어 첫 토큰 생성까지 걸리는 시간이 실제보다 길게 측정될 수 있다는 뜻입니다. GuideLLM에 더 많은 CPU 자원을 할당하거나 `GUIDELLM__MAX_WORKER_PROCESSES`(기본값 10)를 늘리세요. 출력 길이가 고정된 동시 실행에서 모든 스트림이 한꺼번에 첫 토큰을 받는 경우처럼 짧은 순간에 부하가 몰리면, 95번째 백분위수가 높아지지 않더라도 첫 토큰 생성까지 걸리는 시간이 실제보다 길게 측정될 수 있습니다.
- `root_late`:
  - `root_dispatch_delay`의 95번째 백분위수가 0.75초를 초과하거나 평균이 0.25초를 초과하면 경고합니다.
  - 대화의 첫 번째 턴이 스케줄러에서 지정한 시각보다 늦게 도착했다는 뜻입니다. 대개 데이터 로딩 지연 때문에 발생합니다. [요청이 예정 시각보다 늦게 로딩되는 경우](troubleshooting.md#requests-load-after-they-are-due)를 참고하세요.
- `dataset_incomplete`:
  - Trace 실행 구성에서 `dataset_incomplete`가 true이면 경고합니다.
  - Trace 데이터셋을 완전히 불러오지 못했다는 뜻입니다. 많은 trace 대화에는 벤치마크 시작 시점에 시작되어야 하는 턴이 있으므로, 데이터를 모두 불러오지 못하면 대화의 도착이 늦어지거나 누락될 수 있습니다. [요청이 예정 시각보다 늦게 로딩되는 경우](troubleshooting.md#requests-load-after-they-are-due)를 참고하세요.

기본 규칙이 설정된 위치는 `src/guidellm/schemas/benchmark/warnings.py`의 `default_warning_rules()`를 참고하세요.

## 규칙 필드 {#rule-fields}

`metrics.warnings`의 각 항목에는 다음 필드가 있습니다:

- `code`: 경고에 저장되는 자유 형식 태그입니다. 모든 문자열을 사용할 수 있습니다. 일반적으로 짧은 snake_case 태그를 사용하며, 지표 이름과 같을 필요는 없습니다. 기계가 읽을 수 있는 출력에서 경고를 쉽게 식별하기 위한 필드입니다.
- `metric.name`: 집계된 스케줄러 지표나 생성형 지표를 가리키는 점으로 구분된 경로입니다.
- `metric.statistic`: `mean`, `max`, `p95` 또는 `sum`입니다. 기본값은 `mean`입니다.
- `relative_to`: 선택적으로 지정하는 두 번째 지표입니다. 지정하면 관측값은 `metric / relative_to`가 되고 `threshold`는 비율을 나타냅니다. 생략하면 지표 자체가 관측값이 되며 `threshold`는 해당 지표의 단위를 사용합니다.
- `threshold`: 관측값이 이 값보다 크면 경고합니다.
- `scale`: 비교하기 전에 통계값에 곱할 값입니다. 기본값은 `1`입니다. 밀리초 단위 지표와 초 단위 지표를 비교할 때는 `0.001`을 사용하세요.
- `enabled`: `false`로 설정하면 파일에 규칙을 유지하면서 해당 경고를 보고하지 않을 수 있습니다.
- `note`: 측정 결과 설명 아래에 출력할 추가 정보입니다. 변경해야 할 사항을 안내하거나 링크를 제공하는 데 사용하세요.
- `when`: 선택적으로 지정하는 경로와 값입니다. 해당 경로의 값이 지정한 값과 같을 때만 규칙이 적용됩니다. 값 자체를 비교하므로 실행 구성 유형과 같은 텍스트도 비교할 수 있습니다.

불리언 지표는 참 또는 거짓으로 비교합니다. `True`는 임곗값 `0`보다 크지만 `False`는 그렇지 않습니다. 경고의 단위는 `boolean`입니다.

상태별로 나뉜 지표는 단일 분포가 아닙니다. `time_to_first_token_ms`에는 `successful`, `incomplete`, `errored`, `total` 각각의 요약이 있습니다. 경로의 `.total` 부분은 모든 상태를 포함하는 요약을 선택하고, `statistic: mean`은 해당 요약의 평균을 선택합니다. 두 설정은 서로 독립적이므로, 첫 토큰 생성까지 걸리는 시간의 평균은 `time_to_first_token_ms.total`과 `statistic: mean`으로 지정합니다. `request_latency.total`도 같은 구조입니다. `generation_delay`는 이미 단일 분포이므로 경로에 상태를 나타내는 부분이 없습니다.

집계된 스키마에 없는 경로를 지정하면 로그에 기록되며, 진행 상황 표시가 끝난 뒤 다른 경고와 함께 출력됩니다. 필드는 존재하지만 값이 null이면 건너뜁니다. `root_dispatch_delay`는 각 대화의 첫 번째 요청만 대상으로 합니다. 이 요청들에 전송 지연이 하나도 없으면 null이며, 도착 일정이 없는 프로파일이 이에 해당합니다.

## 자주 사용하는 지표 경로 {#common-metric-paths}

- `generation_delay`: 데이터셋에서 다음 대화를 제공할 때까지 대기한 시간(초)입니다. 스케줄러 지표입니다.
- `request_latency.total`: 모든 상태를 포함한 요청 소요 시간(초)입니다.
- `time_to_first_token_ms.total`: 첫 토큰 생성까지 걸리는 시간(밀리초)입니다.
- `time_to_first_output_token_ms.total`: 첫 번째 내용 토큰이 생성될 때까지 걸리는 시간(밀리초)입니다.
- `time_per_output_token_ms.total`: 출력 토큰당 평균 소요 시간(밀리초)입니다.
- `inter_token_latency_ms.total`: 토큰 간 지연 시간(밀리초)입니다.
- `request_dispatch_delay.total`: 예정된 시작 시각보다 늦어진 시간(초)입니다. 프로파일에 도착 일정이 정의되어 있지 않으면 null입니다.
- `root_dispatch_delay`: 각 대화의 첫 번째 요청이 늦어진 시간(초)입니다. 이후 턴은 제외합니다. 첫 번째 턴에 전송 지연이 하나도 없으면 null입니다.
- `worker_loop_lag`: 작업자가 요청을 실행하는 동안 이벤트 루프의 타이머가 예정 시각보다 늦게 실행된 시간(초)입니다. 스케줄러 지표입니다.
- `dataset_incomplete`: 요청 반복자의 모든 항목을 사용하기 전에 스케줄러가 큐에 요청을 추가하는 작업을 중단하면 `true`입니다. 스케줄러는 로더가 모든 대화를 제공한 뒤에만 반복자가 완료되었다고 기록합니다. 따라서 `max_requests`나 `max_duration`으로 실행을 중단하면 데이터셋이 불완전한 것으로 표시되며, 의도적으로 조기에 종료한 실행에서도 내장 trace 규칙이 발동할 수 있습니다. `data_loader.samples`는 반복자 자체의 샘플 수를 제한하므로, 제공된 샘플을 모두 사용한 실행은 완료된 것으로 간주되어 이 규칙이 발동하지 않습니다.
- `strategy_type`: `trace`나 `concurrent`와 같은 실행 구성 유형의 식별자입니다.

다음과 같이 설정하면 trace 실행에서 데이터셋을 완전히 불러오지 못했을 때만 경고하고, 다른 실행 구성 유형에서는 경고하지 않습니다:

```yaml
- code: dataset_incomplete
  metric:
    name: dataset_incomplete
  threshold: 0
  when:
    name: strategy_type
    equals: trace
  note: The trace dataset was not fully loaded.
```

## 시나리오 파일 {#scenario-file}

```yaml
spec:
  metrics:
    kind: generative
    warnings:
      - code: generation_delay
        metric:
          name: generation_delay
          statistic: mean
        relative_to:
          name: request_latency.total
          statistic: mean
        threshold: 0.25
        note: >-
          Dataset generation is a large share of request time.
          https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/warnings.md
      - code: slow_ttft
        metric:
          name: time_to_first_token_ms.total
          statistic: mean
        threshold: 250
        note: Time to first token is above 250 ms.
```

시나리오의 다른 설정과 함께 불러오세요:

```bash
guidellm run \
  --config scenario.yaml \
  --backend kind=openai_http,target=http://localhost:8000
```

`metrics.warnings`는 [내장 규칙](#built-in-rules)을 대체합니다. 계속 사용할 규칙을 모두 포함하세요.

같은 리스트를 명령줄에서도 전달할 수 있습니다. 서비스 수준 목표를 전달할 때처럼 JSON 객체를 사용하면 편리합니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256  --metrics '{"kind":"generative","warnings":[{"code":"slow_ttft","metric":{"name":"time_to_first_token_ms.total"},"threshold":250,"note":"Time to first token is above 250 ms."}]}'
```

## 환경 변수 {#environment-variable}

`GUIDELLM__SPEC__METRICS__WARNINGS`는 한 번의 실행에 사용할 동일한 규칙 리스트를 설정합니다. 값은 JSON 리스트입니다.

이 설정을 적용하려면 `GUIDELLM__SPEC__METRICS__KIND=generative`도 설정해야 합니다.

```bash
GUIDELLM__SPEC__METRICS__KIND=generative GUIDELLM__SPEC__METRICS__WARNINGS='[{"code":"generation_delay","metric":{"name":"generation_delay","statistic":"mean"},"relative_to":{"name":"request_latency.total","statistic":"mean"},"threshold":0.25,"note":"Dataset generation is a large share of request time."}]' \
  guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256
```

시나리오 파일이나 `--metrics`에서 `metrics.warnings`를 명시적으로 지정하면 환경 변수보다 우선 적용됩니다. `warnings`를 생략하면 내장 규칙이 유지됩니다.

## 경고가 표시되는 위치 {#where-warnings-appear}

콘솔은 실행 구성별로 경고를 묶어서 출력합니다. 각 경고는 경고 아이콘으로 시작하며, 아이콘 오른쪽 열에 측정 결과 설명과 참고 사항이 표시됩니다:

```text
Warnings (constant@2.00)
⚠ mean generation delay was 40% of mean request latency.total (0.040s / 0.100s; threshold 25%).
  Dataset generation is a large share of request time.
```

JSON 보고서는 각 벤치마크의 `warnings`에 같은 기록을 저장합니다. 여기에는 `code`, `message`, `observed`, `threshold`, `unit`, `sample_count`, `note`가 포함됩니다.

HTML 보고서는 각 경고를 헤더 아래에 카드로 표시합니다. 카드에는 메시지가 표시되며, 참고 사항이 설정되어 있으면 함께 표시됩니다. 여러 실행이 포함된 보고서에서는 각 카드에 어떤 실행 구성에서 발생한 경고인지도 표시됩니다.

CSV 보고서는 경고 리스트를 하나의 `Warnings` 열에 저장합니다. 셀에는 동일한 기록으로 구성된 JSON 리스트가 들어가며, 해당 실행에 경고가 없으면 `[]`가 저장됩니다.
