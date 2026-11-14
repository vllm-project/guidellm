---
weight: 1
source_sha256: !!str "0ae53546dc19bb85f9b81a776e3e699df36371787c385de9decb7b8fa14e11be"
---

# 멀티턴 대화 벤치마킹

이 가이드에서는 GuideLLM으로 멀티턴 벤치마크를 구성하여 여러 번의 요청과 응답으로 이어지는 사용자 대화를 시뮬레이션하는 방법을 설명합니다.

## 설정

먼저 호환되는 모델이 추론 서버에서 실행 중인지 확인하세요. GuideLLM은 대화를 처리할 수 있는 모든 OpenAI 호환 서버를 지원합니다. 아래 예제에서는 vLLM으로 대화형 모델을 실행합니다.

```bash
# Example: vLLM with a conversational model
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

다음으로, 같은 인스턴스나 서버에 접근할 수 있는 다른 장비에 GuideLLM을 설치하세요:

```bash
pip install guidellm[recommended]
```

## 멀티턴 데이터 구조 이해하기

GuideLLM의 멀티턴 벤치마크는 인덱스가 붙은 열로 대화를 표현합니다. 각 턴은 여러 열로 구성되며, 열 이름 끝의 숫자는 턴 인덱스를 나타냅니다. 턴 인덱스에는 임의의 숫자를 사용할 수 있지만, 쉽게 구분할 수 있도록 0부터 오름차순으로 지정하는 것을 권장합니다.

### 턴 인덱스가 있는 열 형식

예를 들어 3턴 대화의 데이터셋은 다음과 같은 열로 구성할 수 있습니다:

- **prefix**: 시스템 메시지로 사용할 선택적 시스템 프롬프트(첫 번째 턴에 적용)
- **prompt_0**, **output_tokens_count_0**: 첫 번째 턴의 프롬프트와 요청할 출력 토큰 수
- **prompt_1**, **output_tokens_count_1**: 두 번째 턴의 프롬프트와 요청할 출력 토큰 수
- **prompt_2**, **output_tokens_count_2**: 세 번째 턴의 프롬프트와 요청할 출력 토큰 수

파일이나 Hugging Face 데이터셋을 작성할 때는 계속 인덱스가 붙은 열을 사용합니다. 열을 매핑한 뒤에는 인덱스가 있는 행과 그래프 페이로드가 같은 최종 처리 과정을 거칩니다. 먼저 `ConversationGraphData` 형식으로 정규화하고, 클라이언트 도구 호출 턴을 도구 호출 노드와 결과 주입 노드로 확장한 뒤 실행에 사용할 대화 그래프를 구성합니다.

### 합성 데이터의 그래프 형식

합성 멀티턴 데이터는 단일 턴과 하위 에이전트 분기를 포함해 항상 그래프를 생성합니다. `prompt_0`, `prompt_1`처럼 인덱스가 붙은 열 대신 `conversation_turns` JSON 열을 출력하며, 이 그래프 페이로드도 동일한 최종 처리 과정을 거칩니다.

### 멀티턴 대화 실행 방식

멀티턴 벤치마크를 실행할 때 GuideLLM은 다음 작업을 수행합니다:

1. **턴 전송**: 첫 번째 턴(prefix + prompt_0)을 모델에 전송하고 응답을 받습니다.
2. **요청 반환**: 요청과 응답을 하나의 요청으로 묶어 집계기에 저장합니다.
3. **대화 기록 구성**: 요청과 모델의 응답을 결합합니다.
4. **다음 턴 전송**: 대화 기록과 함께 다음 턴(prompt_i)을 전송합니다.
5. **2단계부터 반복**: 지정한 `n`개의 턴을 처리할 때까지 반복합니다.

`/v1/chat/completions`에서는 사용자와 어시스턴트 역할이 번갈아 나오는 메시지 배열로 대화 기록을 전달합니다. `/v1/responses`에서는 두 역할이 번갈아 나오는 메시지를 전달하거나 이전 요청의 ID를 전달합니다. `/v1/completions`에서는 대화 기록을 하나의 프롬프트 문자열로 이어 붙입니다.

자세한 내용은 [요청 형식](#request-formatting)과 [서버 측 대화 기록](#server-side-conversation-history-v1responses-only)을 참고하세요.

### 접두사 열과 시스템 프롬프트

접두사 열이 있으면 다음과 같이 처리합니다:

- `/v1/chat/completions`에서는 접두사를 대화 배열의 시스템 메시지로 사용합니다
- `/v1/responses`에서는 접두사를 `instructions` 필드로 전달합니다
- `/v1/completions`에서는 접두사를 해당 턴의 프롬프트 앞에 붙입니다
- 턴 인덱스별로 접두사를 지정할 수 있지만, 첫 번째 턴에만 접두사를 하나 지정하는 방식을 권장합니다
- 합성 데이터는 첫 번째 턴에만 접두사를 지원합니다

## 처리 옵션

`--profile`, 프로파일의 요청률 파라미터, `--constraint kind=max_requests,count=<n>` 등 일반 벤치마크 인자는 멀티턴 작업에도 동일하게 적용됩니다. 요청 단위로 동작하는 옵션은 각 턴을 별도의 요청으로 처리합니다(예: 턴이 3개인 데이터셋 행 하나를 요청 3개로 집계).

### 합성 데이터 설정

합성 데이터 설정의 `turns` 파라미터를 사용하면 GuideLLM이 멀티턴 합성 데이터를 자동으로 생성합니다.

#### 기본 멀티턴 합성 데이터

멀티턴 합성 데이터를 생성하려면 `--data` 인자에 `turns`를 지정하세요:

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=3
```

이 설정은 3턴 대화를 생성합니다. 각 턴의 토큰 수는 설정한 `prompt_tokens` 및 `output_tokens` 분포에서 독립적으로 샘플링합니다(이 예제에서는 표준편차를 설정하지 않아 각각 256개와 128개로 고정).

#### 첫 번째 메시지 크기

기본적으로 첫 번째 턴을 포함한 모든 턴이 동일한 `prompt_tokens` 및 `output_tokens` 분포를 사용합니다. 검토할 긴 파일이나 에세이로 대화를 시작하고 이후에는 짧은 질문을 보내는 경우처럼 첫 번째 메시지의 길이가 달라야 한다면, 선택적 파라미터인 `first_prompt_tokens` 및 `first_output_tokens`를 지정하세요. 다른 턴과 마찬가지로 `_stdev`, `_min`, `_max`도 설정할 수 있습니다:

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=4,first_prompt_tokens=4096
```

이렇게 하면 이후 턴의 프롬프트는 약 256토큰을 유지하고 첫 번째 사용자 메시지는 평균 4096토큰이 됩니다. 하위 에이전트 분기의 첫 번째 턴은 별도로 재정의하지 않는 한 상위 대화의 `first_*` 설정을 따릅니다([하위 에이전트 분기](#sub-agent-branches) 참고).

#### 접두사를 포함한 합성 데이터

다음 두 가지 방법으로 합성 대화에 시스템 프롬프트(접두사)를 추가할 수 있습니다:

**기본 접두사 설정:**

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=3,prefix_count=5,prefix_tokens=50
```

이 설정은 길이가 50토큰인 서로 다른 접두사 5개를 생성합니다. 각 대화는 이 중 하나를 무작위로 선택하여 시스템 메시지로 사용합니다.

**고급 접두사 설정:**

더 복잡한 시나리오에서는 `prefix_buckets`를 사용하여 여러 접두사 설정에 가중치를 부여할 수 있습니다. 이 경우 설정을 JSON으로 전달해야 합니다:

```bash
--data '{
  "kind": "synthetic_text",
  "prompt_tokens": 256,
  "output_tokens": 128,
  "turns": 3,
  "prefix_buckets": [
    {
      "bucket_weight": 60,
      "prefix_count": 10,
      "prefix_tokens": 100
    },
    {
      "bucket_weight": 40,
      "prefix_count": 1,
      "prefix_tokens": 50
    }
  ]
}'
```

위 설정에서 접두사의 분포는 다음과 같습니다:

- 대화의 60%는 각각 100토큰인 접두사 10개 중 하나를 사용합니다
- 대화의 40%는 50토큰인 접두사를 사용합니다

#### 하위 에이전트 분기 {#sub-agent-branches}

GuideLLM은 오케스트레이터가 대화 도중 여러 하위 에이전트를 병렬로 실행하는 워크로드를 시뮬레이션할 수 있습니다. `branches` 파라미터로 메인 대화의 특정 턴에서 나뉘었다가 이후에 다시 합류할 하위 에이전트 분기를 지정하세요.

각 분기는 다음과 같이 동작합니다:

- `at_turn`에서 새로운 컨텍스트로 **시작**합니다(하위 에이전트는 메인 대화의 기록을 볼 수 없음)
- `turns`에 지정한 턴 수만큼 독립적으로 **실행**됩니다
- `at_turn + merge_after`에서 메인 대화에 **합류**합니다(기본값: `merge_after=1`). 이때 메인 대화가 하위 에이전트의 최종 출력을 받습니다

같은 턴에서 여러 분기를 시작할 수 있으며, 각 분기의 길이는 서로 달라도 됩니다. 메인 대화가 합류 지점을 지나 계속 진행하려면 모든 분기가 완료되어야 합니다.

**기본 예제:**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":2,"turns":3}]}'
```

이 설정은 5턴짜리 메인 대화를 생성합니다. 하위 에이전트 분기 하나가 2번 턴에서 시작하여 3개 턴을 실행한 뒤 3번 턴에서 메인 대화에 합류합니다.

**합류 시점 늦추기:**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":1,"turns":2,"merge_after":2}]}'
```

하위 에이전트가 1번 턴에서 시작하여 2개 턴을 실행하고 3번 턴(`at_turn + merge_after`)에서 합류합니다. 그동안 메인 대화는 병렬로 2번 턴까지 진행합니다.

**여러 분기 사용하기:**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":2,"turns":3},{"at_turn":2,"turns":1,"agent_id":"reviewer"}]}'
```

2번 턴에서 하위 에이전트 두 개가 시작됩니다. 하나는 3개 턴, 다른 하나는 1개 턴을 실행한 뒤 둘 다 3번 턴에서 합류합니다. 메인 대화는 3번 턴에서 0~2번 턴의 전체 대화 기록과 각 하위 에이전트의 최종 출력을 받습니다.

**결과에 표시되는 대화 기록 길이와 턴 인덱스:**

`benchmarks.json`의 각 요청 항목에는 다음 필드가 포함됩니다:

- `info.history_len`: 해당 요청과 함께 서버에 전달한 대화 기록에 포함된 이전 메시지 수입니다. `last` 간선으로 합류한 여러 분기의 메시지가 기록에 포함될 수 있으므로 합류 지점에서는 `history_len`이 한 번에 1보다 많이 증가할 수 있습니다. 새로운 컨텍스트(`new`)로 시작하는 분기의 루트에서는 `0`부터 시작합니다.
- `info.turn_index`: 그래프에서 해당 요청까지의 경로 깊이를 나타냅니다. `new` 간선에서는 `0`으로 초기화되고, `full` 간선을 지날 때마다 1씩 증가합니다. 각 `last` 간선은 그 부모의 상위 노드까지 거슬러 올라가지 않고 깊이에 최대 `1`만 더합니다. 부모가 여러 개인 노드의 `turn_index`는 이렇게 계산한 깊이 중 최댓값, 즉 가장 긴 경로의 깊이입니다.
- `info.preceding_nodes`: 위상 정렬에 따른 실행 순서에서 이 요청보다 앞선 그래프 노드의 수입니다(0부터 시작). `turn_index` 및 `history_len`과 달리 `history_context`에 영향을 받지 않습니다. 재생 시 `history_context=new`를 사용하여(예: OTEL의 `history=trace`) 모든 요청의 `turn_index`가 `0`으로 유지되는 경우 이 값을 사용하세요. 기록된 도구 호출 루프는 그래프 노드(호출과 결과 주입)를 추가하므로 `preceding_nodes`는 원본 LLM 스팬 수가 아니라 DAG 노드 수를 나타냅니다. 병렬로 실행되는 같은 단계의 하위 에이전트라도 위상 정렬에서 동순위를 처리하는 순서에 따라 값이 다를 수 있습니다.

하위 에이전트 분기가 합류한 뒤에는 `history_len`이 `turn_index`보다 큰 경우가 많습니다. 합류한 `last` 출력은 대화 기록에 포함되지만 경로 깊이에는 그 상위 노드를 거슬러 올라가지 않고 최대 `1`만 더하기 때문입니다. `full` 경로가 이미 더 길면 `turn_index`는 증가하지 않습니다.

**분기 설정 필드:**

| 필드                  | 유형          | 기본값     | 설명                                                                          |
| --------------------- | ------------- | ---------- | ----------------------------------------------------------------------------- |
| `at_turn`             | `int`         | 필수       | 메인 대화에서 분기가 시작되는 턴 인덱스                                       |
| `turns`               | `int`         | 필수       | 이 분기에서 실행할 턴 수                                                      |
| `agent_id`            | `str`         | `"worker"` | 분기 노드에 사용할 에이전트 식별자                                            |
| `merge_after`         | `int`         | `1`        | `at_turn` 이후 분기가 합류할 때까지 메인 대화에서 진행할 턴 수                |
| `first_prompt_tokens` | `int \| null` | `null`     | 첫 번째 턴의 프롬프트 크기 재정의(기본값은 상위 대화의 `first_prompt_tokens`) |
| `first_output_tokens` | `int \| null` | `null`     | 첫 번째 턴의 출력 크기 재정의(기본값은 상위 대화의 `first_output_tokens`)     |

분기에 속한 턴은 메인 대화의 `prompt_tokens` 및 `output_tokens` 분포에서 토큰 수를 샘플링합니다. 분기의 `first_*` 필드를 지정하지 않으면 상위 대화의 `first_*` 설정을 따르며, 상위 설정도 없으면 메인 대화의 기본 분포를 사용합니다. 분기의 `first_*` 필드에도 같은 `_stdev`, `_min`, `_max` 파라미터를 사용할 수 있습니다.

> [!NOTE]\
> 합류 지점이 존재하려면 `at_turn + merge_after`가 `turns`보다 작아야 합니다. 메인 대화의 마지막 턴을 넘어가거나 대화가 끝나는 시점에 합류하는 분기는 허용되지 않습니다.

### 요청 형식 {#request-formatting}

멀티턴 대화는 요청 형식에 따라 다르게 구성됩니다:

#### Chat Completions(`/v1/chat/completions`)

Chat Completions의 경우 GuideLLM은 대화 기록을 담은 `messages` 배열을 생성합니다:

```json
{
  "messages": [
    {"role": "system", "content": "prefix content"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_2 content"}]}
  ]
}
```

#### Responses API(`/v1/responses`)

Responses API에서 `server_history`가 비활성화되어 있으면 GuideLLM은 대화 기록을 담은 `input` 배열을 생성하고 접두사를 `instructions`로 설정합니다:

```json
{
  "instructions": "prefix content",
  "input": [
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_2 content"}]}
  ]
}
```

#### Text Completions(`/v1/completions`)

Text Completions의 경우 대화 기록을 하나의 문자열로 이어 붙입니다:

```text
prefix content prompt_0 content response to prompt_0 prompt_1 content response to prompt_1 prompt_2 content
```

### 서버 측 대화 기록(`/v1/responses` 전용) {#server-side-conversation-history-v1responses-only}

기본적으로 GuideLLM은 요청마다 전체 대화 기록을 다시 전송합니다(클라이언트 측 기록). Responses API에서는 `previous_response_id` 필드를 사용하여 **서버 측 대화 기록**을 사용할 수 있습니다. 이 경우 서버가 대화 컨텍스트를 저장하고 관리합니다.

서버 측 대화 기록을 사용하려면 백엔드에서 다음과 같이 설정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/responses,server_history=true \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3
```

이 기능을 활성화하면 GuideLLM은 현재 턴의 입력만 전송하고 ID로 이전 응답을 참조합니다. 서버는 내부적으로 전체 대화 컨텍스트를 재구성합니다.

**요구 사항:**

- 서버는 `previous_response_id`를 지원해야 하며 응답 저장 기능이 활성화되어 있어야 합니다. vLLM은 서버를 시작할 때 `VLLM_ENABLE_RESPONSES_API_STORE=1` 환경 변수를 설정하세요.
- 서버가 응답 저장 기능을 지원하지 않으면 두 번째 턴부터 요청이 실패합니다(일반적으로 404 오류).
- 이 옵션은 `/v1/responses`에서만 사용할 수 있습니다. 다른 요청 형식과 함께 사용하면 시작 시 오류가 발생합니다.

## 도구 호출

멀티턴 벤치마크에서는 여러 턴에 걸친 도구 호출을 지원합니다. 서버 설정, 도구 정의, 도구 선택 설정 및 예외 상황에 대한 자세한 내용은 [도구 호출 가이드](tool_calling.md)를 참고하세요.

## TurnPivot 전처리기

GuideLLM은 각각 별도의 데이터셋을 가리키는 여러 개의 `--data` 옵션을 지원합니다. 여러 데이터셋에서 같은 유형의 열이 발견되면 각각 별도의 배치로 처리합니다. 일반적으로 이 기능은 같은 요청에 서로 다른 데이터셋의 열을 결합할 때 유용합니다. 예를 들어 한 데이터셋의 텍스트 열을 이미지 데이터셋에 추가하거나, 정규 분포를 따르는 여러 합성 데이터셋을 결합하여 봉우리가 여러 개인 분포를 만들 수 있습니다. **TurnPivot** 전처리기를 사용하면 턴별 열과 데이터셋 배치의 축을 서로 바꿀 수 있습니다.

예를 들어 다음 두 데이터셋이 있다고 가정합니다:

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_0 prompt_0 | dataset_0 prompt_1 | 11              | 12              |

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_1 prompt_0 | dataset_1 prompt_1 | 21              | 22              |

**TurnPivot**을 사용하지 않으면 두 번째 턴은 다음과 같습니다:

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_0"}
        ]
    },
    {"role": "assistant", "content": "32 token response to prompt_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_1"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 34,
  ...
}
```

**TurnPivot**을 사용하면 두 번째 턴은 다음과 같습니다:

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_0 prompt_1"}
        ]
    },
    {"role": "assistant", "content": "23 token response to dataset_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_1 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 43,
  ...
}
```

### 사용법

CLI에서 TurnPivot을 사용하려면 데이터 전처리기로 지정하세요:

```bash
--data kind=json_file,path=dataset0.jsonl \
--data kind=json_file,path=dataset1.jsonl \
--data-preprocessor kind=encode_media \
--data-preprocessor kind=turn_pivot
```

> [!WARNING]\
> `--data-preprocessor`를 설정하면 *열 매퍼를 제외한* **모든** 전처리기 설정을 덮어쓰므로, 활용 사례에 필요한 전처리기를 모두 지정하세요.

## 예제

### 1. 합성 데이터를 사용하는 기본 멀티턴 대화

다음은 합성 데이터로 간단한 3턴 대화를 벤치마킹하는 예제입니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=6 \
  --constraint kind=max_requests,count=30 \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3
```

**주요 파라미터:**

- `--backend`: 서버 URL, 모델 및 요청 형식
- `--profile`: 벤치마크 실행 프로파일(`concurrent`는 일정한 수의 동시 요청을 유지)
- `--constraint kind=max_requests,count=30`: 전송할 최대 요청 수(요청 30개는 3턴 대화 약 10개에 해당)
- `--data`: 프롬프트 토큰 200개, 출력 토큰 100개, 턴 3개로 구성된 합성 데이터 설정

이 명령어는 3턴 대화 10개를 벤치마킹합니다. 각 턴은 설정한 토큰 수를 사용합니다(표준편차를 지정하지 않은 경우 프롬프트 200토큰, 출력 100토큰). 모델은 세 턴에 걸쳐 대화 기록을 유지합니다.

### 2. 시스템 프롬프트(접두사)를 사용하는 멀티턴 대화

다음은 멀티턴 대화에 시스템 프롬프트를 포함하는 예제입니다. 에이전트 시스템을 모의 실행하는 등 다양한 상황에서 유용합니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=constant,rate=2.0 \
  --constraint kind=max_requests,count=100 \
  --data kind=synthetic_text,prompt_tokens=150,output_tokens=75,turns=4,prefix_tokens=100
```

**주요 파라미터:**

- `--profile kind=constant,rate=2.0`: 초당 요청 2개의 일정한 요청률로 전송
- `--constraint kind=max_requests,count=100`: 최대 요청 수(요청 100개는 4턴 대화 약 25개에 해당)
- `--data`: `prefix_tokens=100`을 추가하여 100토큰의 시스템 프롬프트 생성

이 벤치마크에서 각 대화는 시스템 메시지(접두사)로 시작하며, 이후 사용자와 어시스턴트가 4개 턴에 걸쳐 대화를 주고받습니다. 실제 사용 환경에서는 시스템 프롬프트로 대화 전체에 적용할 컨텍스트나 지시 사항을 설정하며, 모든 사용자에게 동일한 프롬프트를 사용하는 경우가 많습니다.

> [!NOTE]\
> `turns=4`와 `--constraint kind=max_requests,count=100`을 함께 지정하면 대화가 25개 **이상** 실행됩니다. 다음 턴은 이전 턴이 완료된 뒤에만 스케줄링할 수 있습니다. 턴이 완료되면 해당 대화를 현재 작업자의 큐 맨 앞에 배치합니다.

### 3. 고급 접두사 분포

다음은 여러 접두사 설정에 가중치를 부여하여 다양한 시스템 프롬프트 시나리오를 테스트하는 예제입니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=constant,rate=1.5 \
  --constraint kind=max_duration,seconds=60 \
  --data '{
  "kind": "synthetic_text",
  "prompt_tokens": 180,
  "output_tokens": 90,
  "turns": 3,
  "prefix_buckets": [
    {
      "bucket_weight": 60,
      "prefix_count": 5,
      "prefix_tokens": 100
    },
    {
      "bucket_weight": 40,
      "prefix_count": 1,
      "prefix_tokens": 0
    }
  ]
}'
```

**주요 파라미터:**

- `--profile kind=constant,rate=1.5`: 초당 요청 1.5개의 일정한 요청률로 전송
- `--constraint kind=max_duration,seconds=60`: 각 실행 구성을 최대 60초 동안 실행
- `--data`: `prefix_buckets`로 두 가지 접두사 분포를 정의한 JSON 설정

이렇게 하면 대화의 60%는 각각 100토큰인 접두사 5개 중 하나를 사용하고, 40%는 0토큰의 접두사를 사용합니다(사실상 접두사 없음).

### 4. 파일 기반 데이터셋을 사용하는 멀티턴 대화

다음은 멀티턴 구조로 작성된 기존 데이터셋 파일을 사용하는 예제입니다.

**JSONL 파일 예제**(multiturn_conversations.jsonl):

```json
{"prefix": "You are a helpful assistant.", "prompt_0": "What is Python?", "output_tokens_count_0": 50, "prompt_1": "How do I install it?", "output_tokens_count_1": 40}
{"prefix": "You are a coding expert.", "prompt_0": "Explain functions", "output_tokens_count_0": 60, "prompt_1": "Give me an example", "output_tokens_count_1": 45}
```

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=200 \
  --data kind=json_file,path=multiturn_conversations.jsonl
```

**주요 파라미터:**

- `--profile kind=concurrent,streams=10`: 동시 요청 10개 유지
- `--constraint kind=max_requests,count=200`: 최대 요청 수(요청 200개는 2턴 대화 약 100개에 해당)
- `--data`: 턴 인덱스가 있는 열을 포함한 JSONL 파일의 경로

### 5. 여러 데이터셋에 TurnPivot 사용하기

다음은 TurnPivot 전처리기를 사용하여 턴마다 서로 다른 분포를 따르는 합성 데이터셋을 만드는 예제입니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=150 \
  --data kind=synthetic_text,prefix_tokens=512,prompt_tokens=128,output_tokens=256 \
  --data kind=synthetic_text,prompt_tokens=256,prompt_token_stdev=32,output_tokens=128 \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=128,output_tokens_stdev=16 \
  --data-preprocessor kind=turn_pivot
```

**주요 파라미터:**

- `--constraint kind=max_requests,count=150`: 최대 요청 수(요청 150개는 3턴 대화 약 50개에 해당)
- `--data`: 데이터셋별로 지정하며, 배열로 한 번에 지정할 수도 있음
- `--data-preprocessor kind=turn_pivot`: 데이터셋과 턴별 열의 축을 서로 바꿈

> [!WARNING]\
> 현재 CLI에서 `--data-preprocessor`를 설정하면 *열 매퍼를 제외한* **모든** 전처리기 설정을 덮어쓰므로, 활용 사례에 필요한 전처리기를 모두 지정하세요.

## 대화 기록에 추론 내용 포함하기

추론 모델은 최종 답변을 생성하기 전에 추론 과정에 해당하는 토큰을 출력합니다. 기본적으로 이후 턴의 대화 기록에는 이 토큰이 포함되지 않습니다. 포함하려면 백엔드 설정에서 `multiturn_reasoning`을 활성화하세요:

- `false`(기본값) — 대화 기록에 추론 내용을 포함하지 않음
- `true` — 추론 내용을 `<think>...</think>` 태그로 감쌈(`"<think>{reasoning}</think>"`과 동일)
- `{reasoning}`이 포함된 형식 문자열 — 추론 내용을 감싸는 형식을 직접 지정

### Think 태그(대부분의 모델)

```bash
--backend kind=openai_http,target=http://localhost:8000,multiturn_reasoning=true
```

이 설정을 사용하면 대화 기록에 해당 턴이 다음과 같이 포함되어 전송됩니다:

```json
{"role": "assistant", "content": "<think>step-by-step reasoning...</think>Final answer."}
```

### Granite 형식

```bash
--backend 'kind=openai_http,target=http://localhost:8000,multiturn_reasoning="Here is my thought process:{reasoning}Here is my response:"'
```

### 원문 그대로 사용(구분 기호 없음)

```bash
--backend 'kind=openai_http,target=http://localhost:8000,multiturn_reasoning="{reasoning}"'
```

### 모델 계열별 권장 설정

| 모델 계열                                                          | 권장 값                                                        |
| ------------------------------------------------------------------ | -------------------------------------------------------------- |
| DeepSeek R1, QwQ, Qwen3, Gemma 4, GLM-4.5, Holo2, Cohere Command A | `true`                                                         |
| IBM Granite 3.2                                                    | `"Here is my thought process:{reasoning}Here is my response:"` |
| OpenAI o 시리즈(o1, o3)                                            | `false`(API에서 추론 내용 재전송 불가)                         |
| 사용자 지정 / 구분 기호 없음                                       | `"{reasoning}"`                                                |

### 추론 관련 추가 참고 사항:

추론 내용은 클라이언트가 "reasoning" 청크를 수신할 때만 인식됩니다. 일반적으로 추론 파서를 설정해야 이 청크가 전송됩니다. 그렇지 않으면 GuideLLM은 해당 토큰을 일반 토큰으로 처리합니다. 추론 파서가 추론 내용의 형식을 바꾸므로 추론 토큰을 다시 전송해도 KV 캐시와 정확히 일치하지 않아 일부 캐시 미스가 발생할 수 있습니다. 이 문서를 작성한 시점에는 vLLM 캐시에서 추론 내용을 제거하는 설정을 검토 중입니다.

추론 기록을 올바르게 설정하려면 사용하는 모델과 모델 서버의 설계를 확인하세요.

## 제한 사항 및 고려 사항

### 지원되는 요청 형식

현재 다음 요청 형식에서 멀티턴 벤치마킹을 지원합니다:

- `/v1/chat/completions` - 채팅 템플릿 형식 사용
- `/v1/responses` - OpenAI Responses API 입력 형식 사용
- `/v1/completions` - 대화 기록을 단순히 이어 붙이는 방식 사용

오디오 엔드포인트(`/v1/audio/transcriptions`, `/v1/audio/translations`)는 멀티턴 벤치마킹을 지원하지 않습니다.

### 열 이름 지정 규칙

턴 인덱스가 있는 열은 다음 이름 지정 규칙을 따라야 합니다:

- 열 매핑은 기본 이름에 적용합니다. 예: `--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"text_column": "prompt"}}'`
- 턴 인덱스는 `-0` 또는 `_0` 형식으로 지정할 수 있습니다. 인덱스가 연속되지 않아도 빈 번호가 없도록 턴 번호를 다시 매깁니다.
- 모든 턴의 열은 동일한 기본 이름을 사용해야 합니다. 예: `prompt_0`, `prompt_2`

### 모델 컨텍스트 고려 사항

멀티턴 대화는 대화 기록이 누적되므로 메모리 사용량이 증가합니다:

- 각 턴에는 이전 모든 턴의 전체 대화 기록이 포함됩니다
- 대화가 길어져 턴 수가 늘어날수록 프롬프트가 커집니다
- 대화 기록이 누적되면서 턴마다 토큰 수가 증가합니다
- 턴 수와 토큰 수를 설정할 때 모델의 컨텍스트 윈도우를 고려하세요

예를 들어 `--data kind=synthetic_text,prefix_tokens=50,prompt_tokens=100,output_tokens=200,turns=5`로 설정하면 각 턴의 토큰 수는 다음과 같습니다:

- 1번째 턴: 입력 150토큰, 출력 200토큰
- 2번째 턴: 입력 (150 + 200) + 100 = 450토큰, 출력 200토큰
- 3번째 턴: 입력 (450 + 200) + 100 = 750토큰, 출력 200토큰
- 4번째 턴: 입력 (750 + 200) + 100 = 1050토큰, 출력 200토큰
- 5번째 턴: 입력 (1050 + 200) + 100 = 1350토큰, 출력 200토큰

### 추가 고려 사항

멀티턴 벤치마크를 단일 턴 벤치마크와 비교할 때는 다음 특성을 고려하세요:

- 턴 크기의 분포가 넓으므로 요청 단위 지표의 편차가 클 수 있습니다
- 한 턴에서 오류가 발생하면 나머지 턴이 취소되고 대화 전체가 종료됩니다
