---
weight: 6
source_sha256: !!str "1b9b1e2303c7eb8d4284e25cea190e54254bead366414b53bde50d857a227a36"
---

# 도구 호출

GuideLLM은 여러 턴에 걸쳐 도구를 호출하는 워크로드의 벤치마킹을 지원합니다. 클라이언트에서 도구를 호출할 턴은 **미리 지정**합니다. 데이터 파이프라인은 어떤 사용자 턴이 도구 호출을 기대하고 어떤 턴이 일반 텍스트를 기대하는지 사전에 정합니다. 클라이언트에서 도구를 호출하는 사용자 턴마다 `tool_response_injection` 요청이 하나씩 추가되므로, 대화당 전체 요청 수는 `turns + len(tool_call_turns)`입니다. 예를 들어 `turns=3, tool_call_turns=[0, 1]`로 설정하면 도구 호출, 결과 주입, 도구 호출, 결과 주입, 일반 요청 순으로 요청 5개가 생성됩니다.

클라이언트 도구 호출 턴이 완료되면 GuideLLM은 별도의 결과 주입 턴에서 도구 응답을 서버로 전송하고 텍스트 응답을 받은 뒤 다음 사용자 턴을 진행합니다. 도구 응답의 내용은 데이터셋의 도구 응답 열, `tool_response_tokens`으로 설정한 합성 데이터, 짧은 기본 응답(`{"status": "ok"}`) 순서로 선택합니다. 요청 본문에는 해당 턴의 데이터에 `tools_column`이 있을 때만 도구 정의가 포함됩니다. 도구를 호출하지 않는 턴은 도구나 `tool_choice`를 포함하지 않은 일반 채팅 요청으로 전송합니다.

## 지원되는 요청 형식

`/v1/chat/completions`와 `/v1/responses` 모두 도구 호출을 지원합니다. 이 가이드의 CLI 예제는 기본적으로 `/v1/chat/completions`를 사용하지만, 백엔드 설정에서 `request_format=/v1/responses`로 지정해도 모든 기능이 동일하게 동작합니다. 차이는 이후 요청에서 도구 호출 기록을 표현하는 방식에 있습니다:

- **`/v1/chat/completions`** 엔드포인트는 `tool_calls` 배열이 포함된 `role: "assistant"` 메시지 뒤에 도구 결과를 담은 `role: "tool"` 메시지를 배치합니다.
- **`/v1/responses`** 엔드포인트는 `input` 배열에서 `function_call` 항목 뒤에 `function_call_output` 항목을 배치합니다.

도구 호출 메시지의 전체 전송 형식은 [OpenAI 함수 호출 가이드](https://developers.openai.com/api/docs/guides/function-calling)를 참고하세요.

## 클라이언트 측 모의 도구 호출

GuideLLM은 현재 클라이언트 측 모의 도구 호출(`turn_type="client_tool_call"`)을 지원합니다. 추론 서버는 모델을 실행하여 실제 `tool_calls`를 반환할 수 있지만, GuideLLM은 실제 API나 다른 실행 환경에서 해당 함수를 **실행하지 않습니다**. 클라이언트 도구 호출 턴마다 벤치마크는 모의 도구 출력을 담은 별도의 결과 주입 요청(`turn_type="tool_response_injection"`)을 전송하고 서버의 텍스트 응답을 받은 뒤 다음 사용자 턴을 진행합니다. 이 방식으로 외부 도구의 지연 시간이나 실행에 따른 부수 효과를 제외하고 도구 호출을 처리하는 LLM의 처리량을 측정할 수 있습니다.

## 서버 측 도구 호출

도구를 내부에서 실행하는 서버(예: `container_auto`를 사용하는 OpenAI Responses API 또는 MCP 도구 그룹을 사용하는 [OGX](https://github.com/ogx-ai/ogx))에서는 `server_tool_call` 턴을 사용하세요. GuideLLM은 이 턴을 일반 턴처럼 처리하지만 `tool_choice`를 `"none"`으로 덮어쓰지 않고 요청을 전송하므로 서버에 설정된 도구를 사용할 수 있습니다.

별도의 결과 주입 턴은 생성되지 않으며, GuideLLM도 모의 도구 응답을 생성하지 않습니다. 서버가 전체 도구 호출 루프를 내부에서 실행하고 최종 텍스트 답변을 반환합니다. 지연 시간 지표에는 서버 측 도구 실행 시간이 포함됩니다.

### 합성 데이터에서 서버 측 도구 호출 사용하기

합성 데이터에서는 `server_tool_call_turns`로 서버가 관리할 사용자 턴을 지정하세요. 정수 N(처음 N개 턴), 턴 인덱스 리스트 또는 모든 턴을 나타내는 `-1`을 전달할 수 있습니다:

```bash
# All 3 turns are server_tool_call — the server decides per-turn whether to use tools
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}}}' \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,server_tool_call_turns=-1 \
  --constraint kind=max_requests,count=30
```

단축 표기 `-1`은 N을 `turns`와 같은 값으로 지정한 `server_tool_call_turns=N`과 같습니다. 턴 수를 바꿀 때 두 값을 함께 변경할 필요가 없습니다. 클라이언트 측 도구 호출을 지정하는 `tool_call_turns`에도 같은 표기를 사용할 수 있습니다.

### 실제 데이터셋에서 서버 측 도구 호출 사용하기

참고: 현재 일반적인 도구 호출 데이터셋을 매핑할 때는 일부 제약이 있습니다.

`tools` 열과 같이 도구 정의가 포함된 실제 데이터셋에서는 최종 처리기의 `tool_call_mode` 설정에 따라 해당 턴을 클라이언트 측 또는 서버 측 도구 호출로 처리합니다:

- `tool_call_mode="client"`(기본값) — 도구 정의가 있는 턴을 `client_tool_call`과 `tool_response_injection` 쌍으로 변환합니다. GuideLLM은 모의 도구 응답을 생성하여 서버로 전송합니다.
- `tool_call_mode="server"` — 도구 정의가 있는 턴을 `server_tool_call`로 변환하며, 별도의 결과 주입 턴은 생성하지 않습니다. 데이터셋의 도구 정의는 제거되므로 `--backend`를 통해 백엔드에 도구를 설정하거나 서버에서 직접 설정해야 합니다.

**OpenAI Responses API** — 백엔드 설정의 `extras.body.tools`로 요청 본문에 도구를 전달합니다:

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"https://api.openai.com","request_format":"/v1/responses","extras":{"body":{"tools":[{"type":"shell","environment":{"type":"container_auto"}}]}}}' \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai,load_kwargs.split=train \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer kind=generative,tool_call_mode=server \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

**OGX(Llama Stack)** — `/v1/responses`의 오케스트레이션 루프에서 도구를 실행하는 OpenAI 호환 오픈 소스 에이전트 서버입니다. MCP 도구 그룹이나 내장 도구는 OGX 서버에 설정하므로 GuideLLM 명령어에는 `extras.body.tools`가 필요하지 않습니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8321,request_format=/v1/responses \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai,load_kwargs.split=train \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer kind=generative,tool_call_mode=server \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

서버 측 도구는 `--backend`를 통해 요청 본문에 지정하거나(예: OpenAI의 `extras.body.tools`) 서버에서 직접 설정합니다(예: OGX MCP 도구 그룹). 도구를 호출할지는 모델이 턴마다 결정합니다.

## 서버 설정

도구를 호출하려면 서버에서 도구 호출을 지원해야 합니다.

**vLLM** — 자동 도구 선택 기능을 활성화하고 모델에 맞는 파서를 지정하세요:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

일반적으로 사용하는 파서는 `hermes`(Qwen/Hermes), `llama3_json`(Llama 3.x), `mistral`(Mistral)입니다. 이 플래그를 설정하지 않으면 vLLM은 도구 호출 출력에 문법 오류를 반환합니다.

**OGX(Llama Stack)** — [OGX](https://github.com/ogx-ai/ogx)는 전체 도구 호출 루프를 서버 내부에서 처리하는 OpenAI 호환 `/v1/responses` 엔드포인트를 제공합니다. vLLM을 추론 백엔드로 사용할 수 있으며, 서버 측 도구 실행을 위한 MCP 도구 그룹을 지원합니다. 설치 및 도구 그룹 설정 방법은 [OGX 문서](https://ogx-ai.github.io/docs)를 참고하세요.

## 도구 정의 전달하기 {#providing-tool-definitions}

도구 정의는 데이터 파이프라인을 통해 전달하며, 별도의 전역 CLI 플래그를 사용하지 않습니다. 다음 네 가지 방법을 사용할 수 있습니다:

**1. 합성 데이터** — 데이터 설정에 `tool_call_turns`를 지정하고, 필요한 경우 `tools`도 지정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

연속되지 않은 턴에서 도구를 호출하려면 0부터 시작하는 턴 인덱스 리스트를 전달하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --data '{"kind":"synthetic_text","prompt_tokens":200,"output_tokens":100,"turns":4,"tool_call_turns":[0,2]}' \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

도구 호출에 사용할 수 있는 합성 데이터 설정 필드는 다음과 같습니다:

| 필드                         | 유형               | 기본값 | 설명                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tool_call_turns`            | `int \| list[int]` | `0`    | 도구 정의를 포함하고 도구 호출 응답을 기대하는 사용자 턴. 인덱스는 확장된 요청 리스트가 아닌 사용자 턴을 기준으로 0부터 시작합니다. 정수 N은 처음 N개 사용자 턴을, 리스트는 지정한 인덱스(예: `[0, 2]`)를, `-1`은 모든 턴을 나타냅니다. 각 도구 호출 턴에는 결과 주입 요청이 하나씩 추가되므로 `tool_call_turns=[0,1]`와 `turns=3`을 함께 사용하면 총 5개의 요청이 생성됩니다. `0` 또는 `[]`이면 도구 호출을 사용하지 않음     |
| `server_tool_call_turns`     | `int \| list[int]` | `0`    | 서버 측 도구 호출을 사용할 사용자 턴. 이 턴에는 `server_tool_call`을 지정하며, `tool_choice="none"`를 적용하지 않으므로 서버에 설정된 도구를 사용할 수 있습니다. 결과 주입 턴은 생성하지 않으며 `tool_call_turns`와 겹치면 안 됩니다. 정수 N은 처음 N개 사용자 턴을, 리스트는 지정한 인덱스를, `-1`은 모든 턴을 나타냄                                                                                                         |
| `tools`                      | `list`             | `None` | [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools) 형식(`function` 키로 중첩) 또는 [Responses API](https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools) 형식(중첩 없음)의 도구 정의. GuideLLM은 요청 시 `request_format`에 맞게 자동으로 변환합니다. 대상 엔드포인트와 같은 형식을 사용하는 것을 권장합니다. `None`이면 내장된 기본 도구 사용 |
| `tool_response_tokens`       | `int`              | `None` | 합성 도구 호출 응답의 평균 토큰 수. `None`이면 짧은 기본 응답(`{"status": "ok"}`) 사용                                                                                                                                                                                                                                                                                                                                         |
| `tool_response_tokens_stdev` | `int`              | `None` | 도구 응답 토큰 수의 표준편차                                                                                                                                                                                                                                                                                                                                                                                                   |
| `tool_response_tokens_min`   | `int`              | `None` | 도구 응답의 최소 토큰 수                                                                                                                                                                                                                                                                                                                                                                                                       |
| `tool_response_tokens_max`   | `int`              | `None` | 도구 응답의 최대 토큰 수                                                                                                                                                                                                                                                                                                                                                                                                       |

참고: 이 토큰 수는 모의 도구 응답에서 한 필드의 내용에만 적용됩니다. 응답을 JSON 구조로 감싸는 데 약 5토큰이 추가됩니다.

**도구 응답 내용 설정하기** — 도구 결과에는 기본적으로 짧은 응답(`{"status": "ok"}`)을 사용합니다. 기본 응답은 `GUIDELLM__DEFAULT_SYNTHETIC_TOOL_RESPONSE` 환경 변수로 변경할 수 있습니다. 실제 환경에 더 가까운 벤치마크를 실행하려면 `tool_response_tokens`를 설정하여 길이가 다양한 JSON 응답을 생성하세요:

다음 예제에서는 `tool_response_tokens`를 50으로 설정합니다.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2,tool_response_tokens=50 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

`tool_response_tokens_stdev`, `tool_response_tokens_min`, `tool_response_tokens_max` 필드는 `prompt_tokens_*` 및 `output_tokens_*`의 분포 파라미터와 동일하게 동작합니다.

**2. WEKA trace** — 어떤 턴에서 도구를 호출할지는 파일의 `stop` 및 `input_types`에서 가져오지만, 파일에는 도구 스키마가 없습니다. 합성 데이터와 마찬가지로 `--data kind=weka`에 `tools`를 지정하고 필요하면 `tool_response_tokens`도 전달하세요. 도구를 호출할 턴은 직접 설정할 수 없으며 `tool_call_turns`는 적용되지 않습니다.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data '{"kind":"weka","source":{"kind":"json_file","path":"trace.jsonl"},"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_response_tokens":50}' \
  --constraint kind=max_requests,count=30
```

`tools`를 생략하면 합성 데이터와 동일한 내장 기본 도구를 사용합니다.

**3. OTEL trace** — 파일에서 스팬 출력에 포함된 어시스턴트의 `tool_call` 부분이나 `tool_calls`, `tool_call`, `tool_use`, `function_call` 중 하나로 기록된 `gen_ai.response.finish_reasons` 값을 기준으로 도구 호출 턴을 결정합니다. 스키마는 `gen_ai.tool.definitions`에서 가져오며, 해당 속성이 없으면 기본 합성 도구를 사용합니다. 기록된 도구 결과 문자열을 주입할 때는 현재 실행의 `tool_call_id`에 다시 연결합니다. `--data kind=otel`에 `tool_choice=required`(기본값) 또는 `tool_choice=auto`를 전달하세요. 기록된 메시지는 Chat Completions 형식의 딕셔너리이므로 `/v1/chat/completions`에서 재생하세요. 다른 요청 형식에서는 `raw_messages_column`으로 `prompt` 또는 `input`을 구성할 수 없어 실행이 중단됩니다. 백엔드의 `tool_call_missing_behavior` 설정은 그대로 적용됩니다. [Trace 재생](trace_replay.md#otel)을 참고하세요.

**4. 도구 열이 있는 데이터셋** — `madroid/glaive-function-calling-openai`처럼 도구 정의가 포함된 데이터셋은 그대로 사용할 수 있습니다. 열 매퍼는 이름이 `tools`, `functions` 또는 `tool_definitions`인 열을 자동으로 찾습니다:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

`tool_calling_message_extractor` 전처리기는 기본 설정에 포함되지 않으므로 `--data-preprocessor`로 명시적으로 활성화해야 합니다. 이 전처리기는 각 행의 `messages` 배열을 파싱하고 프롬프트, 시스템 메시지 및 도구 결과를 추출하여 해당 열에 저장합니다. 데이터셋에 도구 결과 메시지가 없으면 기본 응답(`{"status": "ok"}`)을 대신 사용합니다.

## 도구 선택과 도구 호출 누락 처리

실행 중 도구 호출 턴을 처리하는 방식은 다음 두 가지 백엔드 설정으로 제어합니다. 두 설정 모두 `--backend`에 지정합니다:

| 설정                         | 값                                             | 기본값       | 설명                                                                                                                                                                    |
| ---------------------------- | ---------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extras.body.tool_choice`    | `required`, `auto`, `none`                     | `required`   | `client_tool_call` 턴에서 API의 `tool_choice` 파라미터로 전송할 값. 요청 본문에 도구가 있으면 핸들러는 도구를 호출하지 않는 턴에서 자동으로 `tool_choice="none"`을 설정 |
| `tool_call_missing_behavior` | `ignore_continue`, `ignore_stop`, `error_stop` | `error_stop` | 도구 호출을 기대한 턴에 모델이 일반 텍스트를 생성했을 때 백엔드의 처리 방식                                                                                             |

**백엔드에서 `tool_choice` 설정하기:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,extras.body.tool_choice=auto \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30 \
  --profile kind=synchronous
```

**백엔드에서 `tool_call_missing_behavior` 설정하기:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,tool_call_missing_behavior=ignore_continue,extras.body.tool_choice=auto \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30
```

**`tool_choice` 값에 따른 동작:**

- `required`(기본값) — 모델이 **반드시** 도구 호출을 생성해야 합니다. 서버가 출력을 유효한 도구 호출 JSON으로 제한하므로 결과를 예측하기 쉽고 오류가 적습니다. 모델의 자율적인 도구 선택에 의존하지 않으려면 이 값을 사용하세요. 다만 서버가 생성 확률이 낮은 토큰을 선택하도록 강제할 수 있어 처리 속도가 느려질 수 있습니다.
- `auto` — 모델이 도구를 호출할지 결정합니다. 모델이 도구 호출을 얼마나 자주 선택하는지 테스트할 때 유용하지만, 예상한 도구 호출이 누락될 가능성이 높아집니다(`tool_call_missing_behavior` 참고).
- `none` — 요청에 도구가 포함되어 있어도 모델은 도구를 호출할 수 없습니다. 데이터 파이프라인은 이 값을 자동으로 설정하지 않습니다(도구를 호출하지 않는 턴에서는 도구 자체를 생략). 백엔드의 extras에 전역 `tools` 정의를 지정하고 백엔드 설정에 이 값을 명시적으로 지정할 때 사용합니다.

`required`와 `auto`는 모델의 동작에도 영향을 줄 수 있습니다. 예를 들어 Qwen 모델은 `auto`일 때만 도구를 호출하기 전의 추론 내용을 보여 줍니다.

**`tool_call_missing_behavior` 값에 따른 동작:**

이 설정은 `tool_choice`가 `auto`이거나, `required`이지만 서버가 도구 호출을 강제하지 않을 때만 적용됩니다:

- `error_stop`(기본값) — 현재 턴을 **오류**로 표시하고 남은 턴을 모두 취소합니다. 문제를 즉시 확인할 수 있어 모델과 서버가 올바르게 설정되었는지 검증할 때 적합합니다.
- `ignore_stop` — 현재 턴을 **취소**(완료되지 않음)로 표시하고 남은 턴을 모두 취소합니다. 모델의 응답은 출력에 보관하지만, 턴 상태에는 기대한 도구 호출이 생성되지 않았다는 점을 반영합니다. 도구 호출이 누락되어 대화를 계속하기 어렵지만 이를 오류로 처리할 필요는 없을 때 사용하세요.
- `ignore_continue` — 현재 턴을 **완료**로 처리하고 다음 턴을 진행합니다. 이후의 도구 호출 턴은 각각 독립적으로 평가합니다. 대화를 중단하지 않고 `auto` 모드에서 실제 도구 호출 횟수를 측정하려면 이 값을 사용하세요.

### 권장 사용 사례

| 도구 선택  | 호출 누락 시 동작 | 현재 턴 상태 | 설명                                                                                                 |
| ---------- | ----------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| `required` | `error_stop`      | 오류         | (기본값) 합성 데이터에서 일관되고 예측 가능한 동작을 원할 때 적합. 서버의 처리 속도가 느려질 수 있음 |
| `auto`     | `ignore_continue` | 완료         | 모델이 도구를 호출하지 않아도 오류를 발생시키지 않고 `auto`의 동작을 테스트할 때 적합                |
| `auto`     | `ignore_stop`     | 취소         | `auto`의 동작을 테스트하되 모델이 도구 호출 대신 일반 응답을 생성하면 대화를 조기에 종료할 때 적합   |

## 도구 호출 턴의 출력 토큰 제한

합성 데이터나 데이터셋 열에서 `output_tokens`를 설정하면 GuideLLM은 일반적으로 `ignore_eos=True`를 설정하고 중단 시퀀스를 제거하여 모델이 정확히 N개의 토큰을 생성하도록 합니다. 도구 호출 턴에서는 vLLM의 제약 디코딩 문법과 호환되지 않는 설정을 **자동으로 제거**합니다:

- **`ignore_eos`** 설정은 문법의 종료 상태와 충돌합니다. 제약 디코딩은 유한 상태 머신으로 토큰 선택을 제어하며, JSON이 완성되면 EOS만 유효한 토큰으로 허용합니다. `ignore_eos`는 EOS를 제외하므로 유효한 토큰이 하나도 남지 않아 서버 오류가 발생하거나 생성이 끝없이 이어질 수 있습니다.
- **`stop=None`** 설정은 도구 호출 파서가 내부적으로 사용할 수 있는 중단 시퀀스를 제거합니다(예: Llama 모델의 `<|eot_id|>`).
- **`max_tokens` / `max_completion_tokens`** 설정은 JSON이 완성되기 전에 출력을 잘라낼 수 있습니다. 그 결과 도구 호출 인자가 유효하지 않게 되고, 이후 턴의 대화 기록도 손상될 수 있습니다.

따라서 도구 호출 턴의 출력 길이는 설정한 `output_tokens` 대신 모델과 도구 스키마에 따라 결정됩니다. 모델은 함수 이름과 인자가 포함된 유효한 JSON을 생성하는 즉시 멈춥니다. 출력은 일반적으로 20~80토큰으로, 설정한 목표보다 짧아집니다.

일반 텍스트 턴은 기존처럼 `output_tokens` 및 `ignore_eos` 설정을 따릅니다. 이 턴에서는 요청에 도구 정의나 `tool_choice`를 포함하지 않습니다.

## 특수한 경우

- **단일 턴 도구 호출**(`turns=1, tool_call_turns=1` 또는 `tool_call_turns=[0]`)을 지원합니다. 대화는 일반 텍스트 응답 대신 도구 호출을 기대하는 하나의 턴으로 구성됩니다.
- **모든 턴에서 도구를 호출하는 대화**(예: `tool_call_turns=3`과 `turns=3`을 함께 사용하거나 `tool_call_turns=[0,1,2]`로 지정)를 지원합니다. 모든 턴이 도구 호출 턴이므로 모델은 최종 일반 텍스트 응답을 생성하지 않습니다. 각 요청에서 `benchmarks.json`의 `output` 필드는 `None`이므로, 모델의 출력은 `tool_calls` 필드에서 확인하세요.
- **연속되지 않은 턴의 도구 호출**(예: `tool_call_turns=[0, 2]`와 `turns=4`를 함께 사용)을 지원합니다. 지정한 턴만 도구 호출을 기대하며 나머지 턴은 일반 텍스트를 생성합니다.
- **도구를 호출하지 않는 턴의 도구 정의**는 요청에 포함되지 않습니다. 데이터 파이프라인은 도구 호출을 기대하는 턴에만 `tools_column`을 추가하므로 나머지 턴은 도구나 `tool_choice`가 없는 일반 채팅 요청으로 전송됩니다.
- 일부 행에만 `tools_column`이 있는 **혼합 데이터셋**도 사용할 수 있습니다. 도구가 없는 행은 일반 텍스트 대화로 처리하며, 도구가 있는 행은 도구 호출 절차를 따릅니다.
- **요청률을 제한하는 프로파일**(예: `--profile kind=constant,rate=1`)은 다른 요청과 동일한 스케줄러를 사용하여 이후 도구 호출 턴의 전송 시점을 조절합니다. 이후 턴을 다시 큐에 넣고 다음 스케줄링 슬롯이 확보될 때까지 기다리므로, 턴 사이의 실제 대기 시간은 프로파일에 따라 결정됩니다.
