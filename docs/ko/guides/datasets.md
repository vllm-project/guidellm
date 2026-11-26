---
weight: -2
source_sha256: !!str "f80165145781b50f132462684229058aaeb9b76c429a46e788f53ebeea8d819f"
---

# 데이터셋 {#datasets}

GuideLLM은 대규모 언어 모델(LLM)의 벤치마킹 및 평가를 위해 다양한 데이터셋 설정을 지원합니다. 이 문서에서는 다양한 활용 사례에 맞게 데이터셋을 설정하는 방법을 상세한 예제와 함께 설명하고, 각 방식을 선택하는 이유를 안내합니다.

## 데이터 인자 개요 {#data-arguments-overview}

데이터셋 및 처리 옵션은 "kind" 형식으로 등록된 구현체를 선택하고 파라미터를 설정합니다:

```bash
guidellm run --data kind=<TYPE>,key=value,... [other options...]
```

여러 소스를 결합하려면 `--data`를 반복해서 지정하세요. 중첩 구조가 없는 설정에는 쉼표로 구분한 key=value 쌍을 사용하고, 중첩된 값이 있으면 직렬화한 JSON을 사용하세요.

### 데이터셋 소스 및 유형

다음 인자로 데이터셋과 처리 방식을 설정합니다:

- 데이터셋 소스 및 유형. 지원되는 유형:
  - `synthetic_text` — 합성 프롬프트를 실시간으로 생성합니다. 필수 필드: `prompt_tokens`. 선택 사항: `output_tokens`, `turns`, `prefix_tokens`, `prefix_count`, `prefix_buckets` 및 분포 제어 파라미터(`prompt_tokens_stdev`, `output_tokens_stdev` 등).
  - `huggingface`(별칭 `hf`) — HuggingFace Hub 또는 로컬 디렉터리/파일에서 불러옵니다. 필수 필드: `source`(데이터셋 ID 또는 경로). 데이터셋을 불러오는 데 필요한 인자(예를 들어 `split`, `name`)는 `load_kwargs`로 전달합니다.
  - `json_file`, `csv_file`, `text_file`, `parquet_file`, `arrow_file`, `hdf5_file`, `tar_file` — 로컬 파일에서 불러옵니다. 필수 필드: `path`.
  - `db_file` — `load_kwargs.sql`을 사용하여 데이터베이스에서 행을 불러옵니다. 필수 필드: SQLAlchemy 형식의 URI인 `uri`. GuideLLM은 현재 `sqlite:///prompts.db`와 같은 SQLite URI만 지원합니다.
  - `trace_synthetic`, `mooncake`, `weka`, `otel` — `--profile kind=replay`로 trace를 재생합니다. 필수 필드: 다른 데이터셋 유형을 가리키는 중첩된 `source`(예를 들어 `source.kind=json_file,source.path=trace.jsonl` 또는 `source.kind=huggingface,source.source=org/dataset`). 선택 사항: `timestamp_column`, `prompt_tokens_column`, `output_tokens_column`, `time_scale`, `copies` 및 기타 형식별 옵션. [Trace 파일 형식](trace_replay.md)을 참고하세요.

`*_file` 데이터셋 유형 중 로컬 `path`를 받는 유형에서 경로는 정확히 하나의 파일을 지정합니다. 파일이나 디렉터리 이름의 glob 문자는 패턴으로 해석하지 않고 문자 그대로 처리합니다.

또한 데이터 인자인 `load_kwargs`로 데이터셋을 불러오는 데 필요한 추가 인자를 지정할 수 있습니다:

- load_kwargs: 데이터셋을 불러오는 데 필요한 추가 인자입니다. 예를 들어, `--data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"test"}}'`로 데이터셋의 분할을 지정할 수 있습니다. Trace 유형도 동일한 방식으로 `load_kwargs`를 받아 소스 로더에 전달합니다.

SQLite 데이터베이스의 경우 `load_kwargs.sql`로 쿼리를 전달하세요. 예를 들어, `--data '{"kind":"db_file","uri":"sqlite:///prompts.db","load_kwargs":{"sql":"SELECT text FROM samples"}}'`로 지정합니다.

### 데이터 로더

`--data-loader`와 설정 문자열로 데이터 로더를 지정할 수 있습니다. 현재 지원되는 유형은 `pytorch`뿐입니다:

- `--data-loader kind=pytorch,samples=1000,num_workers=4,shuffle=true`

### 토크나이저 {#tokenizer}

`--tokenizer`와 설정 문자열로 토크나이저를 지정할 수 있습니다. 현재 지원되는 유형은 `huggingface_auto`뿐입니다:

- `--tokenizer '{"kind":"huggingface_auto","model":"path/to/processor","load_kwargs":{"use_fast":false}}'`

HuggingFace 모델을 불러오는 데 사용자 지정 코드가 필요하면 `trust_remote_code`를 `load_kwargs`로 전달하세요. [문제 해결: trust_remote_code](troubleshooting.md#tokenizer-trust_remote_code)를 참고하세요.

데이터셋이 표준과 다른 열 이름을 사용하는 경우 `--data-column-mapper`로 해당 열을 GuideLLM이 예상하는 열 이름에 매핑할 수 있습니다. 특히 다음과 같은 경우에 유용합니다:

1. 데이터셋이 다른 열 이름을 사용하는 경우(예를 들어 `question`을 `prompt` 대신, `instruction`을 `text_column` 대신 사용)
2. 여러 데이터셋 중 어느 데이터셋의 열을 사용할지 지정해야 하는 경우
3. 데이터셋의 시스템 프롬프트나 접두사(prefix)가 별도의 열에 있는 경우

### 열 매핑

`--data-column-mapper`는 열 유형을 열 이름에 매핑하는 문자열을 받습니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}'
```

### 전처리기

`--data-preprocessor`로 적용할 GuideLLM 전처리기를 제어할 수 있습니다: 전처리기에는 `encode_media`, `tool_calling_message_extractor`, `turn_pivot`이 포함됩니다.

### 데이터 최종 처리기

`--data-finalizer`로 데이터셋의 행을 요청으로 변환하는 방식을 제어할 수 있습니다(기본값: `generative`).

### 난수 시드

`--seed`와 설정 문자열을 사용하면 스케줄링 및 합성 데이터 생성에서 시퀀스의 무작위 생성 방식을 제어할 수 있습니다: 예를 들어, `--seed kind=static,value=42`로 지정합니다.

### 사용 예제

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}' \
  --tokenizer '{"kind":"huggingface_auto","model":"path/to/processor","load_kwargs":{"use_fast":false}}' \
  --data-loader kind=pytorch,shuffle=true
```

## 데이터셋 유형

GuideLLM은 여러 유형의 데이터셋을 지원하며, 각 유형에는 고유한 장점과 활용 사례가 있습니다. 아래에서는 합성 데이터, Hugging Face 데이터셋, 파일 기반 데이터셋 및 메모리 내 데이터셋을 포함한 주요 데이터셋 유형을 설명합니다.

### 합성 데이터 {#synthetic-data}

합성 데이터셋은 사용자 지정 파라미터로 데이터를 실시간으로 생성할 수 있습니다. 통제된 실험, 스트레스 테스트 및 특정 시나리오 시뮬레이션에 유용합니다. 예를 들어, 모델이 긴 프롬프트를 처리하거나 특정 특성의 출력을 생성하는 방식을 평가할 수 있습니다.

GuideLLM은 아래에서 설명하는 합성 *텍스트* 데이터와 비전 언어 모델 벤치마킹을 위한 합성 *시각* 데이터(이미지 및 짧은 비디오)를 모두 지원합니다. [합성 시각 데이터](multimodal/synthetic_vision.md)에서 `synthetic_image` 및 `synthetic_video`의 `--data` 유형을 확인하세요. 멀티모달 프롬프트를 구성할 때 `synthetic_text`와 결합할 수 있습니다.

#### 명령어 예제

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

또는 JSON 설정 문자열을 사용하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128}'
```

#### 설정 옵션

- `prompt_tokens`: 프롬프트의 평균 토큰 수입니다. 다른 설정을 지정하지 않으면 모든 요청의 토큰 수가 이 값으로 설정됩니다.
- `prompt_tokens_stdev`: 프롬프트 토큰 수의 표준편차입니다. 이 값과 최솟값/최댓값을 모두 지정하지 않으면 편차를 적용하지 않습니다. 이 값은 지정하지 않고 최솟값/최댓값을 지정하면 균등 분포를 사용합니다.
- `prompt_tokens_min`: 프롬프트 토큰 수의 최솟값입니다. 지정하지 않으면 표준편차가 설정된 경우 `max(1, prompt_tokens - 5 * prompt_tokens_stdev)`를, 그렇지 않은 경우 `prompt_tokens`를 기본값으로 사용합니다.
- `prompt_tokens_max`: 프롬프트 토큰 수의 최댓값입니다. 지정하지 않으면 표준편차가 설정된 경우 `prompt_tokens + 5 * prompt_tokens_stdev`를, 그렇지 않은 경우 `prompt_tokens`를 기본값으로 사용합니다.
- `output_tokens`: 출력의 평균 토큰 수입니다. 선택 사항이며, 임베딩처럼 출력 토큰을 생성하지 않는 엔드포인트를 벤치마킹할 때는 생략하세요.
- `output_tokens_stdev`: 출력 토큰 수의 표준편차입니다. 이 값과 최솟값/최댓값을 모두 지정하지 않으면 편차를 적용하지 않습니다. 이 값은 지정하지 않고 최솟값/최댓값을 지정하면 균등 분포를 사용합니다.
- `output_tokens_min`: 출력 토큰 수의 최솟값입니다. 지정하지 않으면 표준편차가 설정된 경우 `max(1, output_tokens - 5 * output_tokens_stdev)`를, 그렇지 않은 경우 `output_tokens`를 기본값으로 사용합니다.
- `output_tokens_max`: 출력 토큰 수의 최댓값입니다. 지정하지 않으면 표준편차가 설정된 경우 `output_tokens + 5 * output_tokens_stdev`를, 그렇지 않은 경우 `output_tokens`를 기본값으로 사용합니다.

표준편차를 지정하면 토큰 수는 설정된 범위로 제한된 정규 분포를 따릅니다. 표준편차 없이 서로 다른 최솟값과 최댓값을 지정하면 균등 정수 분포를 사용합니다. 예를 들어, `prompt_tokens=256,prompt_tokens_stdev=16`의 기본 범위는 176토큰부터 336토큰까지입니다.

합성 텍스트는 필요할 때 생성됩니다. 벤치마크를 중단하려면 `--constraint kind=max_requests,count=1000`과 같은 실행 중 제약 조건을 사용하세요. 고정된 수의 행을 미리 생성하고 데이터셋을 반복해서 순회할 때 재사용하려면 데이터 로더의 `samples`를 설정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --data-loader kind=pytorch,samples=100
```

기본값인 `samples=-1`은 고정된 수의 행을 미리 생성하지 않고 스트리밍합니다. 기존 텍스트 파일의 프롬프트로 벤치마크를 실행하려면 `--data kind=text_file,path=prompts.txt`를 사용하세요. [파일 기반 데이터셋](#file-based-datasets)을 참고하세요.

#### 참고 사항 {#notes}

- 토크나이저가 필요합니다. 기본적으로 백엔드의 모델을 사용합니다. 사용할 수 없으면 `--tokenizer kind=huggingface_auto,model=<path-or-id>`로 토크나이저 파일이 있는 디렉터리 또는 Hugging Face 모델 ID를 지정하세요.

### Hugging Face 데이터셋

GuideLLM은 Hugging Face Hub 또는 `datasets` 라이브러리 형식을 따르는 로컬 디렉터리의 데이터셋을 지원합니다. 이를 통해 다양한 데이터셋을 손쉽게 활용하여 실제 데이터로 벤치마킹 및 평가를 수행할 수 있습니다.

#### 명령어 예제

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=huggingface,source=garage-bAInd/Open-Platypus
```

또는 로컬 데이터셋 디렉터리를 사용하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=huggingface,source=path/to/dataset
```

HuggingFace의 `load_dataset` 옵션은 `load_kwargs`로 전달하세요:

```bash
--data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"train"}}'
```

#### 참고 사항

- Hugging Face 데이터셋은 ID, 로컬 디렉터리 또는 로컬 Python 파일의 경로로 지정할 수 있습니다.
- 지원되는 Hugging Face 데이터셋 형식은 `datasets` 라이브러리의 `load_dataset` 함수로 불러올 수 있어야 하며, `Dataset`, `DatasetDict`, `IterableDataset` 또는 `IterableDatasetDict`로 표현할 수 있는 형식입니다. 지원되는 데이터 유형과 내부적으로 사용하는 `load_dataset`의 추가 인자에 대한 자세한 내용은 [Hugging Face 데이터셋 문서](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub)를 참고하세요.
- 환경 변수에 `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` 또는 `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"`이 설정된 경우에만 토크나이저가 필요합니다. 이 경우 `--tokenizer kind=huggingface_auto,model=...`로 토크나이저를 지정해야 합니다. 설정되지 않은 경우 전달된 모델이나 백엔드에서 가져온 모델을 토크나이저에 사용합니다.

### 파일 기반 데이터셋 {#file-based-datasets}

GuideLLM은 텍스트, CSV, JSON 등 다양한 데이터셋 파일 형식을 지원합니다. 이러한 데이터셋을 벤치마킹 및 평가에 사용하면 활용 사례에 맞는 익숙한 형식으로 구조화된 데이터를 처리할 수 있습니다.

#### 지원되는 형식 및 예제

- **텍스트 파일(`.txt`, `.text`)**: 각 줄을 별도의 프롬프트로 사용합니다.

  ```
  Hello, how are you?
  What is your name?
  ```

- **CSV 파일(`.csv`)**: 각 행은 별도의 데이터셋 항목이며, 첫 번째 행에는 열 이름이 포함됩니다. 열에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 열을 포함할 수 있습니다.

  ```csv
  prompt,output_tokens_count,additional_column,additional_column2
  Hello, how are you?,5,foo,bar
  What is your name?,3,baz,qux
  ```

- **JSON Lines 파일(`.jsonl`)**: 각 줄은 별도의 JSON 객체입니다. 객체에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 필드를 포함할 수 있습니다.

  ```json
  {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"}
  {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
  ```

- **Trace 파일(지원되는 trace 파일 형식의 `.jsonl`, `.json`, `.csv` 또는 `.parquet`)**: 재생을 위한 전용 파일입니다. `--profile kind=replay`와 함께 사용하여 각 행의 타임스탬프와 토큰 길이로 trace 이벤트를 재생합니다. 타임스탬프는 일관된 영점을 기준으로 하는 공통 시간축에서 초 단위 숫자로 표현되어야 합니다. 단, `otel`은 ISO-8601 형식의 `start_time` 값과 HuggingFace에서 디코딩한 `datetime` 객체를 에포크 초로 변환합니다. GuideLLM은 타임스탬프를 정렬하고 첫 번째 이벤트를 기준으로 한 상대 시간으로 변환한 후 스케줄링합니다. [Trace 재생 벤치마킹](../getting-started/benchmark.md#trace-replay-benchmarking)을 참고하세요.

  ```json
  {"timestamp": 1234500.0, "input_length": 256, "output_length": 128}
  {"timestamp": 1234500.5, "input_length": 512, "output_length": 64}
  ```

  이 예시에서 두 번째 요청은 첫 번째 요청으로부터 0.5초 후에 실행되도록 예약됩니다. GuideLLM이 요청을 스케줄링하고 합성 페이로드를 생성하기 전에 trace 행을 타임스탬프 순서로 정렬합니다. 이를 통해 각 예약된 이벤트가 같은 행의 프롬프트 및 출력 토큰 길이와 일치하도록 유지합니다.

  Trace를 불러오려면 지원되는 [trace 파일 형식](trace_replay.md#supported-formats)을 사용하세요:

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=trace_synthetic,source.kind=json_file,source.path=path/to/trace.jsonl,time_scale=1.0
  ```

  모든 trace 형식은 기본적으로 "timestamp", "input_length", "output_length" 열을 찾습니다. Trace에서 다른 열 이름을 사용한다면 데이터 설정에 `timestamp_column`, `prompt_tokens_column`, `output_tokens_column`을 포함하세요:

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,timestamp_column=timestamp,prompt_tokens_column=input_length,output_tokens_column=output_length
  ```

  `otel`도 동일한 중첩된 `source` 필드를 사용합니다. 토큰 수는 최상위 열이 아닌 스팬의 속성에서 가져옵니다. Hugging Face를 사용하면 가장 빠르게 시작할 수 있습니다:

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
    --constraint kind=max_requests,count=30
  ```

  기본값은 기록된 `gen_ai.input.messages`와 `history=trace`이며, 각 스팬의 전체 입력을 사용합니다(`history_context=new`). `history=runtime`을 전달하면 새 메시지만 전송하고 DAG를 통해 실제 생성된 응답을 연결합니다. `gen_ai.input.messages`가 없는 스팬에는 `trace_synthetic`을 사용해야 합니다. 로컬 JSONL은 `source.kind=json_file,source.path=...`와 동일한 `history` 설정을 사용합니다. [Trace 파일 형식](trace_replay.md#otel)을 참고하세요.

  재생 시 `time_scale`을 `--data`에 지정하면 대기 시간 및 패킹 제한이 적용된 뒤 trace 이벤트 사이의 시간 간격에 배율을 적용합니다. 대기 시간 제한(`max_wait`, `max_session_wait`, `min_concurrent_sessions`)은 `time_scale` 적용 전에 원본 trace의 시간(초)을 기준으로 적용됩니다. 병렬성을 높이려면 이러한 패킹 옵션을 우선 사용하세요. 패킹된 데이터셋이 벤치마크를 실행하기에 너무 짧으면 `copies`로 순차적으로 반복 재생하세요. Replay 프로파일은 스케줄러 측의 `time_scale`도 `--profile kind=replay`에서 받습니다. 불러와서 재생할 trace 행의 수를 제한하려면 `--data-loader kind=pytorch,samples=1000`을 사용하세요. `--constraint kind=max_requests,count=<n>`은 실행 중 요청 완료 수를 기준으로 종료하는 제약 조건으로만 사용하며, 파일에서 불러오는 trace 행의 수를 제한하지 않습니다. `--constraint kind=max_duration,seconds=<n>`은 재생 중 대기를 포함하여 진행 중인 대기 작업도 취소합니다.

- **JSON 파일(`.json`)**: 전체 데이터셋이 특정 키 아래에 중첩된 JSON 객체 배열로 표현됩니다. 사용할 키를 올바르게 지정하려면 `--data-column-mapper` 인자에 배열이 있는 위치를 나타내는 `"field": "NAME"`을 전달해야 합니다. 객체에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 필드를 포함할 수 있습니다.

  ```json
  {
    "version": "1.0",
    "data": [
      {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"},
      {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
    ]
  }
  ```

- **Parquet 파일(`.parquet`)** 예제: 효율적인 데이터 처리를 위한 바이너리 열 기반 저장 형식입니다. 지원되는 형식에 대한 자세한 내용은 [참고 사항](#notes)에 연결된 Hugging Face 데이터셋 문서를 참고하세요.

- **Arrow 파일(`.arrow`)** 예제: 메모리 내 데이터를 위한 여러 언어 간 개발 플랫폼입니다. 지원되는 형식에 대한 자세한 내용은 [참고 사항](#notes)에 연결된 Hugging Face 데이터셋 문서를 참고하세요.

- **HDF5 파일(`.hdf5`)** 예제: 대용량 데이터를 저장하기 위한 계층형 데이터 형식입니다. 지원되는 형식에 대한 자세한 내용은 [참고 사항](#notes)에 연결된 Hugging Face 데이터셋 문서를 참고하세요.

#### 명령어 예제

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=json_file,path=path/to/dataset.json \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}'
```

다른 형식을 사용하려면 `json_file`을 `csv_file`, `text_file`, `parquet_file`, `arrow_file` 또는 `hdf5_file`로 바꾸세요.

#### 참고 사항

- 파일 형식이 데이터셋에 필요한 구조와 일치하고 지원되는 형식으로 나열되어 있는지 확인하세요.
- `--data-column-mapper` 인자로 프롬프트 열 이름이나 사용할 분할 등 데이터셋을 파싱하는 데 필요한 추가 파라미터를 지정할 수 있습니다.
- 지원되는 형식과 내부적으로 사용하는 `load_dataset`의 추가 인자에 대한 자세한 내용은 [Hugging Face 데이터셋 문서](https://huggingface.co/docs/datasets/en/loading#local-and-remote-files)를 참고하세요.

### 메모리 내 데이터셋

메모리 내 데이터셋은 데이터를 Python 객체로 직접 전달할 수 있으므로, 디스크에 데이터를 저장하지 않고 빠르게 프로토타입을 만들거나 테스트하기에 적합합니다.

#### 지원되는 형식 및 예제

- **열과 값으로 구성된 딕셔너리**: 각 키는 열 이름이고 값은 데이터 포인트의 리스트입니다. 키에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 열을 포함할 수 있습니다.
  ```python
  {"column1": ["value1", "value2"], "column2": ["value3", "value4"]}
  ```
- **딕셔너리 리스트**: 각 딕셔너리는 키-값 쌍으로 구성된 단일 데이터 포인트를 나타냅니다. 딕셔너리에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 필드를 포함할 수 있습니다.
  ```python
  [{"column1": "value1", "column2": "value3"}, {"column1": "value2", "column2": "value4"}]
  ```
- **항목 리스트**: 각 항목은 단일 데이터 포인트입니다. 항목에는 프롬프트 열로 사용할 `prompt` 또는 일반적으로 사용되는 다른 프롬프트 이름이 포함되어야 합니다. 앞서 설명한 `--data-column-mapper` 인자의 별칭을 기준으로 추가 필드를 포함할 수 있습니다.
  ```python
  ["value1", "value2", "value3"]
  ```

#### 사용 예제

```python
from guidellm.benchmark import benchmark_generative_text

data = [
    {"prompt": "Hello", "output": "Hi"},
    {"prompt": "How are you?", "output": "I'm fine."}
]

benchmark_generative_text(data=data, ...)
```

#### 참고 사항

- 데이터 형식이 일관되고 지원되는 구조 중 하나를 따르는지 확인하세요.
- 딕셔너리에서는 모든 열의 샘플 수가 같아야 합니다.
- 딕셔너리 리스트에서는 모든 항목의 키가 같아야 합니다.
- 항목 리스트에서는 모든 요소의 유형이 같아야 합니다.
- 환경 변수에 `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` 또는 `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"`이 설정된 경우에만 토크나이저가 필요합니다. 이 경우 `--tokenizer kind=huggingface_auto,model=...`로 토크나이저를 지정해야 합니다. 설정되지 않은 경우 전달된 모델이나 서버에서 가져온 모델을 토크나이저에 사용합니다.

## 데이터셋 전처리

GuideLLM은 프롬프트 및 출력 토큰 수가 지정한 크기가 되도록 데이터셋을 처리하는 전처리 명령어를 제공합니다. 벤치마킹을 위해 데이터셋을 표준화하거나, 데이터셋의 프롬프트가 목표 토큰 수 요구 사항과 맞지 않을 때 특히 유용합니다.

전처리 명령어로 다음 작업을 수행할 수 있습니다:

- 프롬프트를 목표 토큰 길이로 조정
- 레지스트리 기반 전략으로 목표 길이보다 짧거나 긴 프롬프트 처리
- 데이터셋의 열을 GuideLLM이 예상하는 열 이름에 매핑
- 전략 설정에 따라 출력 토큰 수 생성
- 처리된 데이터셋을 다양한 형식으로 저장

### 기본 사용법

```bash
guidellm preprocess dataset \
    kind=<DATA_TYPE>,key=value,... \
    <OUTPUT_PATH> \
    --tokenizer kind=huggingface_auto,model=<MODEL> \
    --strategy kind=<STRATEGY>,prompt_tokens=<N>,output_tokens=<N>
```

### 필수 인자

| 인자          | 설명                                                                                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`        | 처리할 데이터셋 지정. [데이터셋 설정](#datasets)에 설명된 모든 데이터셋 형식 지원. 이 데이터셋 설정의 `load_kwargs`로 로더 옵션 전달                      |
| `OUTPUT_PATH` | 파일 확장자를 포함한 처리된 데이터셋의 저장 경로(예를 들어 `processed_dataset.jsonl`, `output.csv`)                                                       |
| `--tokenizer` | **필수.** 레지스트리 형식의 토크나이저 설정(예를 들어 `--tokenizer kind=huggingface_auto,model=gpt2`). 선택적인 생성자 키워드 인자는 `load_kwargs`로 전달 |
| `--strategy`  | **필수.** 목표 토큰 수 및 짧은 프롬프트 처리 방식을 포함한 전처리 전략. [전처리 전략](#preprocess-strategy) 참고                                          |

### 예제

```bash
guidellm preprocess dataset \
    kind=json_file,path=path/to/input_dataset.jsonl \
    "path/to/processed_dataset.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256,prefix_tokens_max=100
```

### 전처리 전략 {#preprocess-strategy}

`--strategy` 옵션은 다른 GuideLLM 옵션과 동일한 레지스트리 형식(`kind=<name>,...` 또는 JSON)을 사용합니다. 목표 토큰 수 설정과 짧은 프롬프트 처리 유형을 함께 지정합니다.

**공통 필드(모든 유형):**

- `prompt_tokens`: 프롬프트의 평균 토큰 수입니다. 다른 설정을 지정하지 않으면 모든 프롬프트의 길이를 이 토큰 수로 조정합니다.
- `prompt_tokens_stdev`: 프롬프트 토큰 수의 표준편차입니다. 이 값과 최솟값/최댓값을 모두 지정하지 않으면 편차를 적용하지 않습니다. 이 값은 지정하지 않고 최솟값/최댓값을 지정하면 균등 분포를 사용합니다.
- `prompt_tokens_min`: 프롬프트 토큰 수의 최솟값입니다. 이 값은 지정하지 않고 `prompt_tokens_stdev`를 설정하면 최솟값은 1입니다.
- `prompt_tokens_max`: 프롬프트 토큰 수의 최댓값입니다. 이 값은 지정하지 않고 `prompt_tokens_stdev`를 설정하면 최댓값은 표준편차의 5배입니다.
- `output_tokens`: 출력의 평균 토큰 수입니다. 다른 설정을 지정하지 않으면 모든 출력의 토큰 수가 이 값으로 설정됩니다.
- `output_tokens_stdev`: 출력 토큰 수의 표준편차입니다. 이 값과 최솟값/최댓값을 모두 지정하지 않으면 편차를 적용하지 않습니다. 이 값은 지정하지 않고 최솟값/최댓값을 지정하면 균등 분포를 사용합니다.
- `output_tokens_min`: 출력 토큰 수의 최솟값입니다. 이 값은 지정하지 않고 `output_tokens_stdev`를 설정하면 최솟값은 1입니다.
- `output_tokens_max`: 출력 토큰 수의 최댓값입니다. 이 값은 지정하지 않고 `output_tokens_stdev`를 설정하면 최댓값은 표준편차의 5배입니다.
- `prefix_tokens_max`: 유지할 접두사 토큰 수의 최댓값입니다. 설정하면 접두사를 이 최대 길이로 잘라냅니다.
- `count_prefix`: `true`이면 프롬프트 토큰 수 한도에 접두사 토큰을 포함합니다(기본값: `false`).

**전략 유형:**

| 유형          | 설명                                                         | 유형별 필드                                                       |
| ------------- | ------------------------------------------------------------ | ----------------------------------------------------------------- |
| `ignore`      | 목표 길이보다 짧은 프롬프트 제외                             | —                                                                 |
| `concatenate` | 목표 길이에 도달할 때까지 여러 짧은 프롬프트를 이어 붙임     | `delimiter`(이어 붙이는 프롬프트 사이에 넣을 문자열; 기본값 `""`) |
| `pad`         | 목표 길이에 도달할 때까지 짧은 프롬프트를 문자/문자열로 채움 | `pad`(필수이며 비어 있지 않은 패딩 문자열)                        |
| `error`       | 프롬프트가 목표 길이보다 짧으면 오류 발생                    | —                                                                 |

**전략 예제:**

```bash
# Ignore short prompts (default style)
--strategy kind=ignore,prompt_tokens=512,output_tokens=256

# Concatenate short prompts
--strategy kind=concatenate,prompt_tokens=512,output_tokens=256,delimiter="\n\n"

# Pad short prompts
--strategy kind=pad,prompt_tokens=512,output_tokens=256,pad=" "

# JSON form with count_prefix
--strategy '{"kind":"ignore","prompt_tokens":512,"output_tokens":256,"count_prefix":true}'
```

`--tokenizer` 인자는 토큰 수를 계산하는 데 사용할 토크나이저를 지정합니다(`guidellm run --tokenizer`와 동일한 레지스트리 형식). 전처리 명령어는 프롬프트를 토큰화하여 목표 토큰 수와 일치하는지 확인해야 하므로 토크나이저가 필수입니다. 벤치마크에서 토크나이저를 사용하는 방법은 [데이터 인자 개요](#data-arguments-overview)를 참고하세요.

### 열 매핑

데이터셋이 표준과 다른 열 이름을 사용하는 경우 `--data-column-mapper`로 해당 열을 GuideLLM이 예상하는 열 이름에 매핑할 수 있습니다. 특히 다음과 같은 경우에 유용합니다:

1. **데이터셋이 다른 열 이름을 사용하는 경우**(예를 들어 `question`을 `prompt` 대신, `instruction`을 `text_column` 대신 사용)
2. **여러 데이터셋을 사용하는 경우**, 어느 데이터셋의 열을 사용할지 지정해야 할 때
3. **데이터셋의 시스템 프롬프트나 접두사가** 별도의 열에 있는 경우

**열 매핑 형식:** `--data-column-mapper` 옵션은 레지스트리 형식을 사용합니다(`guidellm run`과 동일):

```bash
--data-column-mapper kind=generative_column_mapper,column_mappings.text_column=question,column_mappings.prefix_column=system_prompt

# Or as JSON:
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {
  "text_column": "question",
  "prefix_column": "system_prompt",
  "prompt_tokens_count_column": "input_tokens",
  "output_tokens_count_column": "completion_tokens"
}}'
```

**지원되는 열 유형:**

- `text_column`: 기본 프롬프트 텍스트(기본값: `prompt`, `instruction`, `question`, `input`, `context`, `content`, `text`)
- `raw_messages_column`: 현재 턴의 Chat Completions 메시지(`[{role, content}, …]`; 기본값: `messages`, `chat_messages`). `/v1/chat/completions`는 메시지를 그대로 전송합니다. 다른 요청 형식에서는 이 메시지로부터 `prompt`/`input`을 구성할 수 없으면 중단합니다.
- `prefix_column`: 시스템 프롬프트 또는 접두사(기본값: `system_prompt`, `system`, `prefix`)
- `prompt_tokens_count_column`: 프롬프트 토큰 수를 포함한 열(기본값: `prompt_tokens_count`, `input_tokens_count`)
- `output_tokens_count_column`: 출력 토큰 수를 포함한 열(기본값: `output_tokens_count`, `completion_tokens_count`)
- `image_column`: 이미지 데이터 열
- `video_column`: 비디오 데이터 열
- `audio_column`: 오디오 데이터 열

**예제: 사용자 지정 열 이름 매핑**

데이터셋의 CSV 파일에 `user_query` 및 `system_message` 열이 있는 경우:

```csv
user_query,system_message
"What is AI?","You are a helpful assistant."
"How does ML work?","You are a technical expert."
```

다음과 같이 사용합니다:

```bash
guidellm preprocess dataset \
    "kind=csv_file,path=dataset.csv" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=user_query,column_mappings.prefix_column=system_message
```

**예제: 여러 데이터셋**

여러 데이터셋 중 어느 데이터셋의 열을 사용할지 지정해야 한다면 `<dataset_index>.<column_name>` 또는 `<dataset_name>.<column_name>` 형식을 사용할 수 있습니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"text_column": "0.prompt", "prefix_column": "1.system"}}'
```

### 추가 옵션

| 옵션                    | 설명                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--data-column-mapper`  | 열 매핑 전처리기 설정(기본값: `kind=generative_column_mapper`)                                                                    |
| `--data-loader`         | 데이터 로더 설정(기본값: `kind=pytorch`). 저장할 처리된 행의 수는 `samples`로 제한. 전처리에서는 `shuffle`과 `num_workers`를 무시 |
| `--seed`                | 재현 가능한 토큰 샘플링을 위한 난수 시드 설정(기본값: `kind=static,value=42`)                                                     |
| `--push-to-hub`         | 처리된 데이터셋을 Hugging Face Hub에 업로드(플래그)                                                                               |
| `--hub-dataset-id <ID>` | 업로드할 Hugging Face Hub 데이터셋 ID(`--push-to-hub`를 설정한 경우 필수)                                                         |

### 전체 예제

**예제 1: 사용자 지정 열 이름을 사용하는 기본 전처리**

```bash
guidellm preprocess dataset \
    kind=csv_file,path=my_dataset.csv \
    "processed_dataset.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=user_question,column_mappings.prefix_column=system_instruction
```

**예제 2: 분포와 concatenate 전략을 사용하는 전처리**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=concatenate,prompt_tokens=512,prompt_tokens_stdev=50,output_tokens=256,output_tokens_stdev=25,delimiter="\n\n" \
    --seed kind=static,value=123
```

**예제 3: 토크나이저 로딩 키워드 인자, 접두사 제한 및 count_prefix를 사용하는 전처리**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2,load_kwargs.use_fast=false \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256,prefix_tokens_max=100,count_prefix=true
```

**예제 4: --data-loader로 처리할 행 수 제한**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=large_dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-loader kind=pytorch,samples=1000
```

**예제 5: 전처리 후 Hugging Face Hub에 업로드**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=my_dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --push-to-hub \
    --hub-dataset-id "username/processed-dataset"
```

### 참고 사항

- `--strategy` 옵션은 `PreprocessStrategyArgs` 레지스트리 유형을 받습니다. 이 유형에는 토큰 수 필드(`prompt_tokens`, `output_tokens` 및 관련 분포 필드), `prefix_tokens_max`, `count_prefix` 및 짧은 프롬프트 처리 유형인 `kind`가 포함됩니다. [전처리 전략](#preprocess-strategy)을 참고하세요.
- 전처리 명령어는 프롬프트를 토큰화하여 목표 토큰 수와 일치하는지 확인해야 하므로 토크나이저가 필수입니다. 벤치마크에서 토크나이저를 사용하는 방법은 [데이터 인자 개요](#data-arguments-overview)를 참고하세요.
- 데이터셋 로더의 키워드 인자는 위치 인자인 `DATA`의 데이터셋 설정(`load_kwargs=...`)에 지정하며, 별도의 `--data-args` 플래그로 전달하지 않습니다.
- `--data-loader kind=pytorch,samples=<N>`을 사용하면 처리된 행을 `N`개 저장한 뒤 중단합니다. `shuffle`과 `num_workers`는 `guidellm run`과 CLI 옵션을 일관되게 유지하기 위해 받지만, 전처리 중에는 무시됩니다.
- 열 매핑은 데이터셋이 표준과 다른 열 이름을 사용하는 경우에만 필요합니다. 매핑을 지정하지 않으면 GuideLLM은 일반적으로 사용되는 열 이름을 자동으로 시도합니다.
- 출력 파일 형식은 `OUTPUT_PATH`의 파일 확장자(예를 들어 `.json`, `.jsonl`, `.csv`, `.parquet`)로 결정됩니다.
