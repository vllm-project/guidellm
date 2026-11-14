---
weight: 7
source_sha256: !!str "67189587e84e6a0c48437fc8a7353c9fbd615bffe79d2e06668a0a04307ea796"
---

# Trace 파일 형식

Trace를 정확하게 재생하려면 파일 형식에 맞게 데이터를 처리해야 합니다. 이 가이드에서는 GuideLLM이 지원하는 모든 trace 파일 형식과 각 형식에 공통으로 적용되는 데이터 인자, 형식별 데이터 인자를 설명합니다.

Replay 프로파일과 파일 기반 데이터셋을 사용하는 방법은 [Trace 재생 벤치마킹](../getting-started/benchmark.md#trace-replay-benchmarking)을 참고하세요.

## 지원되는 형식 {#supported-formats}

`--data` 인자에 `kind=format`으로 형식을 지정합니다:

- `trace_synthetic`: 합성 프롬프트를 생성하여 trace 재생 벤치마크를 실행하는 데 필요한 최소한의 정보를 사용하는 형식
- `mooncake`: 서빙 플랫폼 *Mooncake*에서 사용하는 trace 형식. 자세한 정의는 [https://doi.org/10.48550/arXiv.2407.00079](https://doi.org/10.48550/arXiv.2407.00079) 참고
- `weka`: WEKA의 *Augmented Memory Grid*에서 사용하는 trace 형식. 자세한 사양은 [원본 연구 저장소](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md) 참고
- `otel`(별칭 `opentelemetry`): OpenTelemetry GenAI 스팬 형식. GuideLLM은 성공한 LLM 스팬만 사용하며 각 `trace_id`를 하나의 대화로 재생합니다. `history=runtime`을 설정하지 않으면 기록된 `gen_ai.input.messages`를 사용하여 각 스팬의 전체 입력을 전송합니다(`history=trace`).

## Trace 데이터 로딩

Trace를 재생할 때는 항상 `--profile kind=replay`를 사용합니다. **형식**(`trace_synthetic`, `mooncake`, `weka` 등)을 선택하고, 다른 역직렬화 유형(`json_file`, `huggingface` 등) 중에서 **소스** 유형을 선택하세요. 예를 들면 다음과 같습니다:

**로컬 `json_file`을 사용하는 `trace_synthetic` 예제:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,time_scale=2.0 \
  --constraint kind=max_requests,count=30
```

**`huggingface`에서 WEKA 데이터셋 불러오기:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay,prefetch=all \
  --data kind=weka,source.kind=huggingface,source.source=semianalysisai/cc-traces-weka-no-subagents-051226 \
  --constraint kind=max_requests,count=30
```

**`huggingface`에서 Mooncake 데이터셋 불러오기**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=mooncake,source.kind=huggingface,source.source=valeriol29/mooncake-traces,load_kwargs.name=mooncake \
  --constraint kind=max_requests,count=30
```

**`huggingface`에서 OTEL 데이터셋 불러오기:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
  --constraint kind=max_requests,count=30
```

## 공통 데이터 인자

모든 trace 형식에서 다음 데이터 인자를 선택적으로 사용할 수 있습니다:

| 인자                      | 기본값          | 설명                                                                                                       |
| ------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| `timestamp_column`        | "timestamp"     | Trace 파일에서 타임스탬프를 담은 열 이름                                                                   |
| `prompt_tokens_column`    | "input_length"  | Trace 파일에서 프롬프트 토큰 수를 담은 열 이름                                                             |
| `output_tokens_column`    | "output_length" | Trace 파일에서 출력 토큰 수를 담은 열 이름                                                                 |
| `duration_column`         | "duration"      | 각 요청의 실행 시간(초)을 담은 선택적 열. `schedule_turn=idle_gap`에서 사용                                |
| `time_scale`              | 1.0             | 대기 시간과 패킹 제한을 적용한 뒤 남은 상대 타임스탬프에 적용할 배율                                       |
| `max_wait`                | 미설정          | 한 세션에서 연속된 두 요청 사이의 최대 간격(원본 trace 기준, 초)                                           |
| `max_session_wait`        | 미설정          | 이전 세션의 마지막 요청부터 현재 세션까지 허용할 최대 유휴 시간(원본 trace 기준, 초)                       |
| `min_concurrent_sessions` | 미설정          | 정상 상태에서 최소한 이 개수만큼 세션이 겹치도록 세션을 배치                                               |
| `copies`                  | 1               | 전체 데이터셋을 순차적으로 재생하는 횟수. k+1번째 재생은 k번째 재생의 마지막 요청 예정 시각에 시작         |
| `copy_offset`             | 1.0             | 이전 재생 구간을 기준으로 다음 복사본을 시작할 위치. 0은 시작, 1은 끝이며 1보다 크면 재생 사이에 간격을 둠 |

다음과 같이 `--data` 인자로 전달합니다:

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data "kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,timestamp_column=ts,prompt_tokens_column=input_tokens,output_tokens_column=generated_tokens,time_scale=1.0,max_session_wait=30"
```

`trace_synthetic`는 특정 형식에 의존하지 않습니다. 타임스탬프, 프롬프트 토큰 수, 출력 토큰 수 열만 읽고 데이터셋의 다른 정보는 무시합니다. 주로 테스트에 사용하지만 GuideLLM이 아직 지원하지 않는 trace 형식의 대안으로도 `trace_synthetic`을 사용할 수 있습니다.

Replay 프로파일의 `schedule_turn` 인자로 타임스탬프를 적용하는 방식을 선택합니다. 기본값인 `idle_gap`은 `duration_column`에 기록된 요청 소요 시간 이후의 유휴 간격을 유지합니다. 예를 들어 기록된 요청 소요 시간이 1초이고 다음 요청의 타임스탬프가 5초 뒤라면, 현재 요청이 실제로 완료된 시점에서 4초 후에 다음 요청을 시작합니다. `idle_gap`에서는 요청이 늦어져도 앞선 턴이 끝나면 전송할 수 있어 서버 과부하에 더 유연하게 대응합니다. `duration_column`이 없으면 로더가 경고를 한 번 기록하고 각 요청의 기록상 소요 시간을 0으로 처리하므로 실제 대기 간격이 조금 더 길어집니다. `timestamp`는 각 요청을 trace에 기록된 시각에 시작하며, 이전 턴이 아직 실행 중인 경우에만 기다립니다.

`trace_synthetic`과 `mooncake`는 각 행을 단일 요청으로 구성된 독립적인 대화로 재생합니다. 행을 타임스탬프 순으로 정렬하고 trace의 첫 번째 요청을 기준으로 각 행의 시간 차이를 유지합니다. 프롬프트는 행을 읽을 때 생성하며, 같은 `copies` 재생 구간에서는 여러 행이 Mooncake 해시 ID를 공유합니다. 독립적인 요청 간의 간격을 제한하려면 `max_session_wait`를 사용하세요. `max_wait`는 WEKA 세션처럼 요청이 여러 개인 대화 내부의 간격만 제한합니다.

먼저 `min_concurrent_sessions`, 대기 시간 제한, `time_scale`을 사용하여 병렬성을 높이세요. 패킹한 데이터셋을 한 번 재생하는 것만으로 벤치마크의 `max_duration` 또는 `max_requests` 조건을 충족하기 어려울 때 `copies`를 사용하세요. 기본 설정(`copy_offset=1`)에서 `copies`는 패킹된 전체 데이터셋을 연이어 재생하며, 다음 재생은 이전 재생의 마지막 요청 타임스탬프에서 시작합니다. `copy_offset=0`이면 이전 재생의 첫 번째 타임스탬프에서 시작합니다. 0과 1 사이의 값은 구간 안의 비례하는 시점을 나타내며, 1보다 크면 재생 사이에 간격이 생깁니다. 해시 ID를 사용하는 형식(`mooncake`, `weka`)은 재생마다 서로 다른 salt를 적용한 전역 토큰 블록 테이블을 사용합니다. 따라서 이후 재생에서 앞선 토큰을 재사용하여 접두사 캐시 적중률이 부풀려지는 일을 방지합니다.

## 형식별 데이터 인자

### `mooncake`

Mooncake 형식에는 접두사 기반 캐시의 해시 ID를 담는 열이 추가로 필요합니다. 프롬프트를 생성할 때 같은 이전 ID를 공유하는 서로 다른 해시 ID는 각각 다른 토큰 ID 블록을 나타내야 합니다.

| 인자                 | 기본값     | 설명                                         |
| -------------------- | ---------- | -------------------------------------------- |
| `hash_ids_column`    | "hash_ids" | Trace 파일에서 해시 ID 리스트를 담은 열 이름 |
| `hash_id_block_size` | 512        | 해시 ID 하나가 나타내는 토큰 수              |

### `weka`

**참고:** 미리 캐시에 적재한 `tool_tokens` 및 `system_tokens` 접두사와 해시 ID의 최장 공통 접두사(LCP)를 기준으로 평탄화된 에이전트를 분리하는 기능은 구현되어 있지 않습니다. `type: "subagent"`으로 선언한 그룹과 도구 호출 이벤트(`stop: tool_use`, `input_types: ["tool_result"]`)는 재생할 수 있습니다.

WEKA 형식에서는 대화 UUID를 중첩되지 않은 별도의 열에 저장해야 합니다. 타임스탬프, 입력 토큰 수, 출력 토큰 수 및 해시 ID는 JSON 객체 리스트 형태로 하나의 JSON 열(예: "requests") 안에 저장해야 합니다.

Mooncake와 마찬가지로 WEKA는 접두사 기반 캐시의 해시 ID를 사용합니다. Trace의 [원본 사양](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md)에서는 해시 ID가 1 이상이어야 하며, 마지막 블록을 채우기에 입력 토큰이 부족하면 해당 해시 ID를 제외해야 합니다. GuideLLM은 이 사양을 정확히 따르지 않는 데이터셋(예: [semianalysisai/cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226))도 처리할 수 있도록 0 이상의 정수를 유효한 해시 ID로 허용하며, 블록이 일부만 채워진 해시 ID가 있으면 제외합니다.

GuideLLM은 첫 번째 대화부터 프롬프트를 생성하고, 해당 대화가 끝나면 다음 대화를 사용합니다. 상대 타임스탬프는 데이터셋에서 가장 이른 요청 시각을 기준으로 한 시간 차이이므로 이후 대화는 첫 대화보다 늦게 시작할 수 있습니다.

WEKA trace를 사용할 때는 일반적으로 `prefetch=all`을 권장합니다. Trace의 세션이 서로 겹치지 않고 올바른 순서로 정렬되어 있을 때만 미리 불러오기를 생략할 수 있습니다. 그렇지 않으면 데이터 역직렬화가 늦어져 세션이 trace의 상대 시각보다 늦게 시작될 수 있습니다. 벤치마크 시작 시점이나 그 부근에 요청을 시작하는 세션이 대부분이면 이 문제가 더 심해집니다. 모든 세션이 제때 시작되도록 요청을 모두 미리 불러오세요. `--profile kind=replay,prefetch=all`을 사용하면 벤치마크 시간 측정을 시작하기 전에 요청 그래프를 모두 불러옵니다. [요청이 예정 시각보다 늦게 로딩되는 경우](troubleshooting.md#requests-load-after-they-are-due)를 참고하세요.

해시 ID의 적용 범위는 각 행의 `hash_id_scope` 필드에 따라 결정됩니다:

- `"global"` 또는 생략: Mooncake와 마찬가지로 여러 대화가 하나의 토큰 블록 테이블을 공유합니다. 이후 대화에서 같은 해시 ID를 만나면 이전 토큰 블록을 재사용하므로 접두사 캐시 적중률이 원본 trace와 비슷하게 유지됩니다. `copies`로 반복 재생할 때는 매번 서로 다른 salt를 적용한 전역 테이블을 사용합니다.
- `"local"`: 해시 ID를 해당 대화 안에서만 사용합니다. 대화를 생성한 뒤에는 테이블을 폐기합니다. `copies`로 반복 재생할 때도 각 재생 구간에서 독립적으로 적용됩니다.

프롬프트 생성 속도를 높이기 위해 각 해시 ID의 토큰 블록을 한 번만 텍스트로 디코딩하고 같은 블록이 다시 나타날 때 디코딩한 문자열을 재사용할 수 있습니다. 이 캐시의 수명과 범위는 토큰 테이블과 같습니다. 연속된 턴에서 거의 같은 컨텍스트를 다시 보내는 멀티턴 에이전트 trace에서는 요청마다 전체 토큰 리스트를 디코딩할 필요 없이 고유한 블록만 디코딩합니다. 이 최적화는 블록별로 디코딩한 문자열을 이어 붙인 결과가 전체 토큰 리스트를 한 번에 디코딩한 결과와 바이트 단위로 동일한 토크나이저에서 자동으로 활성화됩니다. Qwen, Llama 3, GPT-2 등의 바이트 단위 BPE 토크나이저가 여기에 해당합니다. SentencePiece와 같은 다른 토크나이저에서는 역직렬화기가 요청마다 전체 리스트를 디코딩하므로 모든 토크나이저에서 정확히 동일한 바이트의 프롬프트를 얻을 수 있습니다. 두 방식 모두 토큰 수와 접두사 캐시 재사용 방식은 그대로 유지됩니다.

`type: "subagent"`으로 선언된 항목은 독립적인 하위 체인이 됩니다. 하위 체인은 직전 상위 API 턴에서 새로운 대화 기록(`history_context="new"`)으로 시작합니다. 다음 상위 턴은 그사이에 시작된 모든 하위 체인이 완료될 때까지 기다린 뒤 각 체인의 마지막 출력을 받습니다(`history_context="last"`). 따라서 같은 상위 턴 사이에 나열된 하위 에이전트는 병렬로 실행되며, 모두 완료된 뒤에 상위 체인이 재개됩니다. 요청 리스트의 순서가 분기와 합류 구조를 정의하므로 모든 중첩 단계에서 원래 순서를 유지하고 타임스탬프로 재정렬하지 않습니다.

하위 요청의 타임스탬프는 시작 시점을 기준으로 한 상대 시각이면 사양에 맞게 처리하고, 공개 Hugging Face 데이터셋에서처럼 이미 절대 시각이면 그대로 사용합니다. 첫 번째 하위 요청의 `t`가 하위 에이전트 항목의 시작 시각 `t`보다 작으면 하위 시각을 `spawn_t + inner_t`로 계산합니다. 이후 대화 타임스탬프는 `absolute_t`에서 데이터셋의 가장 이른 API 요청 시각을 빼서 구합니다.

같은 에이전트의 연속된 턴은 항상 순차적으로 실행됩니다. 이 턴들의 시각이 겹치면(`t[i] + api_time[i] > t[i+1]` 또는 `api_time`이 없을 때 `t[i+1] <= t[i]`) GuideLLM이 디버그 메시지를 기록합니다. 서로 다른 하위 에이전트의 실행 시간이 겹치는 것은 의도된 병렬 실행이므로 경고하지 않습니다.

도구 호출 이벤트는 GuideLLM의 기존 클라이언트 도구 호출 파이프라인으로 처리합니다. `stop: "tool_use"`와 사용자 텍스트 입력이 있는 요청은 `client_tool_call` 턴이 됩니다. 다음 요청에 `input_types: ["tool_result"]`가 있으면 `tool_response_injection` 턴이 됩니다. `input_types`가 없으면 같은 에이전트 체인에서 `stop: "tool_use"` 다음에 오는 요청을 사용합니다. 결과 주입 행에도 `stop: "tool_use"`가 있으면 도구 결과를 전송하면서 `tools`를 유지하므로 모델이 추가 도구 호출을 생성할 수 있습니다. Trace에는 실제 도구 스키마나 결과가 없습니다. [합성 데이터](tool_calling.md#providing-tool-definitions)와 같은 방식으로 `tools`와 필요한 경우 `tool_response_tokens`를 전달하세요. 생략하면 GuideLLM이 기본 합성 도구 정의와 기본 도구 응답을 사용합니다. 채팅 핸들러는 결과 주입 턴에서 해시 ID 기반 프롬프트를 사용자 메시지로 보내지 않습니다.

| 인자                         | 기본값     | 설명                                                                            |
| ---------------------------- | ---------- | ------------------------------------------------------------------------------- |
| `conversation_id_column`     | "id"       | Trace 파일에서 대화 UUID를 담은 열 이름                                         |
| `hash_ids_column`            | "hash_ids" | Trace 파일에서 해시 ID 리스트를 담은 열 이름                                    |
| `hash_id_block_size`         | 64         | 해시 ID 하나가 나타내는 토큰 수                                                 |
| `tools`                      | `None`     | 도구 호출 턴에서 사용할 OpenAI 형식의 도구 정의. 생략하면 내장 기본 도구 사용   |
| `tool_response_tokens`       | `None`     | 모의 도구 결과의 평균 토큰 수. 생략하면 짧은 기본 응답(`{"status": "ok"}`) 사용 |
| `tool_response_tokens_stdev` | `None`     | 도구 응답 토큰 수의 표준편차                                                    |
| `tool_response_tokens_min`   | `None`     | 도구 응답의 최소 토큰 수                                                        |
| `tool_response_tokens_max`   | `None`     | 도구 응답의 최대 토큰 수                                                        |

변경되는 기본값은 다음과 같습니다:

| 인자                   | 변경된 기본값 |
| ---------------------- | ------------- |
| `timestamp_column`     | "t"           |
| `prompt_tokens_column` | "in"          |
| `output_tokens_column` | "out"         |

### `otel` {#otel}

OpenTelemetry GenAI trace는 기록된 시각에 따라 대화로 재생됩니다. 다음 두 가지 파일 구조를 지원합니다:

- **줄당 세션 하나**: JSONL의 각 행이 `{ "trace_id": ..., "spans": [ ... ] }` 형식입니다.
- **줄당 스팬 하나**: JSONL의 각 행이 하나의 스팬이며, `trace_id`가 같은 연속된 행을 하나의 대화로 묶습니다. 행을 스트리밍하면서 인접한 행만 묶으므로 `a, b, a`처럼 ID가 교차하면 두 개가 아니라 세 개의 대화가 됩니다. 공개 재생 데이터셋은 같은 `trace_id`의 행을 연속해서 저장하지만, 여러 trace를 동시에 수집하여 내보낸 파일은 그렇지 않을 수 있습니다.

성공한 LLM 스팬만 재생합니다(`gen_ai.operation.name`이 `chat`, `generate` 또는 `text_completion`인 스팬이나 토큰 사용량 속성이 이미 있는 스팬). `invoke_agent` 스팬과 실패한 스팬(`status.code` 오류)은 제외합니다. `execute_tool` 스팬은 HTTP 요청으로 전송하지 않습니다.

각 LLM 스팬은 한 번의 API 호출 전체를 기록합니다. `gen_ai.input.messages`는 대화 전문을 다시 기록하며, `prompt_tokens` 및 `input_tokens`는 이전 호출과의 차이가 아니라 **해당 호출의 전체 입력 크기**입니다. `history` 설정으로 이 기록을 요청으로 변환하는 방식을 지정합니다:

| `history=trace`(기본값)                                                                                                                                                                                                                               | `history=runtime`                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 스팬에 기록된 메시지를 모두 전송합니다. `history_context=new`이므로 현재 실행에서 생성한 응답을 다음 프롬프트에 삽입하지 않습니다. 재생 정확도가 가장 높지만, 실제 응답이 기록된 어시스턴트 텍스트와 달라지는 지점부터 접두사 캐시를 재사용할 수 없음 | DAG의 `full` 대화 기록에는 *현재 실행*의 출력을 사용하며, 현재 턴에는 **새로 추가된** 메시지만 전송합니다. `input[i] == input[i-1] + output[i-1] + delta` 조건을 충족해야 합니다. 캐시 재사용에 유리하지만 현재 모델의 응답이 기록과 달라지면 재생 정확도가 낮아짐 |

출력 길이에는 항상 스팬의 완성 토큰 수를 사용합니다(`max_tokens` 및 `ignore_eos`).

`gen_ai.input.messages`가 없는 스팬은 OTEL 형식으로 재생할 수 없습니다. 토큰 수만 기록한 데이터는 평탄화하여 `kind=trace_synthetic` 형식으로 사용하세요.

`/v1/chat/completions`(백엔드 기본값)을 사용하여 재생하세요. OTEL은 Chat Completions 메시지 딕셔너리를 `raw_messages_column`에 저장합니다. 해당 핸들러는 이 메시지를 `messages` 배열로 전송합니다. `/v1/completions`와 `/v1/responses`는 이 열을 읽지 않으므로 빈 본문을 전송하는 대신 `prompt` 또는 `input` 누락 오류를 발생시키고 실행을 중단합니다.

기록된 도구 호출 루프는 GuideLLM의 클라이언트 도구 호출 파이프라인에 맞게 미리 분할됩니다. 출력 메시지에 `tool_calls`가 있거나 `gen_ai.response.finish_reasons`이 `tool_calls`, `tool_call`, `tool_use` 또는 `function_call`인 LLM 스팬은 `client_tool_call` 턴이 됩니다. 다음 스팬에서 `input[i] + output[i]` 이후에 추가된 메시지가 모두 `role=tool`인 도구 결과이면 해당 스팬을 `tool_response_injection` 턴으로 처리합니다. 채팅 핸들러는 기록된 결과 문자열을 **현재 실행**의 `tool_call_id`에 다시 연결합니다. 해당 턴의 `tools_column`에는 `gen_ai.tool.definitions` 값을 사용하고, 값이 없으면 기본 합성 도구를 사용합니다. 도구 정의의 유무만으로 턴 유형을 결정하지는 않습니다. 결과 주입 턴의 부모에는 `history=trace`인 경우를 포함하여 항상 `history_context=full`을 사용합니다. 도구 호출이 누락되었을 때의 동작은 기존처럼 `--backend tool_call_missing_behavior=...`으로 설정합니다.

다음 스팬을 도구 결과로 파싱할 수 없으면 기본 응답을 사용한 결과 주입 턴을 생성하고 해당 스팬도 그대로 재생합니다. 뒤에 오는 `execute_tool` 스팬이 도구 호출 LLM 스팬의 하위 스팬이거나, 같은 부모를 둔 다음 LLM 스팬이 나오기 전에 null이 아닌 `parent_span_id`를 공유하는 형제 스팬이면 해당 스팬에서 주입할 텍스트와 전송 시점을 가져옵니다. 그 외에는 도구 호출 타임스탬프에서 합성 기본 응답을 사용합니다.

완료된 요청의 통계를 집계할 때는 요청의 예상 토큰 수에 응답에서 보고한 사용량을 덮어쓰므로, 예상값과 실제값 간의 평균 절대 오차(MAE)를 따로 보고하지 않습니다. 편차를 확인하려면 병합 전에 `request.input_metrics` 및 `output_metrics`(스팬의 토큰 수)을 응답 사용량과 비교하세요.

스케줄링 전에 ISO-8601 형식의 `start_time` 값(시간대 정보 없음, `Z` 또는 오프셋 포함)과 Hugging Face에서 디코딩한 `datetime` 객체를 에포크 초로 변환합니다. 토큰 수는 스팬의 `attributes`에서 읽으며, 현재 GenAI 속성 이름을 먼저 확인한 뒤 더 이상 권장되지 않는 별칭을 확인합니다. OTel의 `parts`(`text`, `tool_call`, `tool_call_response`)는 OpenAI 채팅 형식의 딕셔너리로 변환합니다.

| 인자                       | 기본값                                                             | 설명                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `spans_column`             | "spans"                                                            | 줄당 세션 하나를 저장하는 파일에서 중첩된 스팬 리스트를 담은 열 이름                                                       |
| `trace_id_column`          | "trace_id"                                                         | 줄당 스팬 하나를 저장하는 파일에서 스팬을 대화로 묶는 기준이 되는 열                                                       |
| `span_timestamp_field`     | "start_time"                                                       | 요청 시작 시각을 담은 스팬 필드                                                                                            |
| `input_tokens_attributes`  | `["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"]`      | 프롬프트 토큰 수를 가져올 때 순서대로 확인하는 속성 키                                                                     |
| `output_tokens_attributes` | `["gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"]` | 출력 토큰 수를 가져올 때 순서대로 확인하는 속성 키                                                                         |
| `history`                  | `trace`                                                            | `trace`는 각 스팬의 전체 입력을 다시 전송하고, `runtime`은 DAG의 대화 기록과 함께 새 메시지만 전송                         |
| `tool_choice`              | `required`                                                         | 클라이언트 도구 호출 턴에서 `required` 또는 `auto` 사용. `auto`는 `tool_call_missing_behavior=ignore_continue`와 함께 사용 |

먼저 Hugging Face 데이터셋으로 시작하세요. IBM trace에는 각각 30~50개의 LLM 호출이 있으므로 처음 실행할 때는 `--constraint kind=max_requests`로 요청 수를 제한하면 유용합니다.

**기본값(`history=trace`):** 각 스팬에 기록된 메시지를 모두 전송합니다. 이후 턴은 DAG의 선행 턴이 완료될 때까지 기다리지만 `history_context=new`를 사용하므로 현재 실행의 응답이 다음 프롬프트에 삽입되지 않습니다.

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
    --constraint kind=max_requests,count=30
```

**DAG 대화 기록과 함께 기록된 메시지 사용(`history=runtime`):** 새 메시지만 전송하고 이전 턴에는 현재 실행의 응답(`history_context=full`)을 사용합니다. 각 스팬의 입력은 이전 스팬의 입력과 출력을 이어받아야 합니다.

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces,history=runtime \
    --constraint kind=max_requests,count=30
```

로컬 JSONL에서도 `source.kind=json_file,source.path=...`과 함께 동일한 `history` 설정을 사용합니다.

Hugging Face의 공개 데이터셋은 다음과 같습니다:

- [ibm-research/synthetic-conversations-traces](https://huggingface.co/datasets/ibm-research/synthetic-conversations-traces) — 멀티턴 채팅, 줄당 세션 하나의 JSONL, `prompt_tokens` 및 `completion_tokens`
- [Exgentic/agent-llm-traces-v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2) — 에이전트 세션, 중첩된 `spans`, `input_tokens` 및 `output_tokens`
- [ibm-research/lmcache-agentic-traces_Otel](https://huggingface.co/datasets/ibm-research/lmcache-agentic-traces_Otel) — 에이전트 세션, 줄당 세션 하나의 JSONL, 더 이상 권장되지 않는 토큰 키

관련 데이터셋에는 [DiscoPosse/agent-llm-traces](https://huggingface.co/datasets/DiscoPosse/agent-llm-traces)(Exgentic v1 스키마)와 [lenadan/otel-test-snippet-jsonl](https://huggingface.co/datasets/lenadan/otel-test-snippet-jsonl)(줄당 스팬 하나의 짧은 예제)가 있습니다. [ibm-research/codex_swebenchpro_traces_Otel](https://huggingface.co/datasets/ibm-research/codex_swebenchpro_traces_Otel)에는 토큰 사용량 속성이 없어 토큰 수 기반 재생에 사용할 수 없습니다. `gen_ai.input.messages`가 없는 스팬에는 `trace_synthetic`을 사용하세요.
