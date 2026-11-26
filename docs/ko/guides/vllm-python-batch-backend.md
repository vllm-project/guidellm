---
source_sha256: !!str "046e403229dd94372f3270db867f592c074f19f01a1fbb383d523e9402024b4b"
---

# vLLM Python 배치 백엔드 {#vllm-python-batch-backend}

**vLLM Python 배치 백엔드**(`vllm_python_batch`)는 vLLM의 동기식 [LLM](https://docs.vllm.ai/) 엔진을 사용하여 GuideLLM과 동일한 프로세스에서 배치 추론을 실행합니다. 요청을 큐에 추가하고 `LLM.generate()`를 통해 설정한 배치 단위로 전송하므로, 요청별 스케줄링 오버헤드를 제거할 수 있습니다. 개별 요청의 지연 시간보다 전체 처리량이 중요한 처리량 벤치마킹에 적합합니다.

`vllm_python_async` 백엔드와 마찬가지로 HTTP 서버를 거치지 않습니다. `target`은 전달하지 **않으며**, 백엔드 설정에 `model`을 **반드시** 전달해야 합니다.

전체 엔진 옵션과 지원되는 모델을 확인하려면 vLLM의 [엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/) 및 [vLLM 문서](https://docs.vllm.ai/)를 참고하세요.

## 설치 {#installation}

설치 방법은 [vLLM Python 백엔드](vllm-python-backend.md#installation)와 같습니다. 배치 백엔드도 같은 vLLM 패키지를 사용합니다.

## 기본 예제 {#basic-example}

vLLM Python 배치 백엔드로 벤치마크를 실행하려면 다음과 같이 설정하세요:

```bash
guidellm run \
  --backend kind=vllm_python_batch,model=Qwen/Qwen3-0.6B,batch_size=8 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=throughput,max_concurrency=20 \
  --constraint kind=max_requests,count=100
```

## 비동기 백엔드와 배치 백엔드 비교 {#async-vs-batch-backend}

| 항목        | `vllm_python_async`      | `vllm_python_batch`                |
| ----------- | ------------------------ | ---------------------------------- |
| 엔진        | `AsyncLLMEngine`(비동기) | `LLM`(동기식, 배치)                |
| 스트리밍    | 지원함                   | 지원하지 않음                      |
| 배치 처리   | 요청별 비동기 스케줄링   | 크기를 설정할 수 있는 마이크로배치 |
| 적합한 용도 | 지연 시간 분석, 스트리밍 | 처리량 벤치마킹                    |

## 백엔드 옵션 {#backend-options}

- **`batch_size`** (기본값: `32`)\
  한 번의 `LLM.generate()` 호출을 실행하기 전에 모을 최대 요청 수입니다. 배치가 이 크기에 도달하면 즉시 전송합니다. 요청 수가 `batch_size`보다 적은 배치는 `batch_timeout`초가 지나면 전송합니다. 따라서 실제 배치 크기는 `min(batch_size, requests arriving within the timeout window)`입니다. 값을 크게 설정하면 요청당 엔진 오버헤드를 줄일 수 있지만 개별 요청의 지연 시간은 늘어납니다.

- **`batch_timeout`** (기본값: `0.01`)\
  배치가 가득 차지 않았을 때 추가 요청을 기다리는 시간(초)입니다. 가득 찬 배치는 이 대기 시간 없이 바로 전송합니다. 동시 요청 수가 많아 배치마다 더 많은 요청을 모을 수 있으면 값을 늘리고, 지연 시간에 민감한 워크로드에서는 값을 줄이거나 기본값을 유지하세요.

- **`model`** (필수)\
  vLLM에서 불러올 Hugging Face 모델 식별자 또는 파일 시스템 경로입니다.

- **`request_format`**\
  채팅 프롬프트를 구성하는 방식을 제어합니다. `vllm_python_async`와 동일하게 `plain`, `default-template`, 또는 Jinja2 템플릿 경로나 문자열을 지정할 수 있습니다.

- **`vllm_config`**\
  중첩된 딕셔너리로 전달하는 엔진 옵션입니다. vLLM의 `EngineArgs` 파라미터 이름을 사용하며, CLI 형식이 아닌 Python 형식으로 지정합니다. `vllm_config`에 대한 자세한 내용은 [vLLM Python 백엔드 문서](vllm-python-backend.md#request-format-and-backend-options)를 참고하세요.

  JSON을 사용하는 예제:

  ```bash
  --backend '{"kind":"vllm_python_batch","model":"Qwen/Qwen3-0.6B","batch_size":64,"vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

> [!IMPORTANT]
>
> `vllm_python_batch`의 백엔드 설정에는 `model` 필드가 필수입니다. `vllm_config` 안에도 `model`을 설정하면 최상위 `model` 필드가 우선 적용됩니다.

## 엔진 수명 주기 {#engine-lifecycle}

vLLM의 `LLM` 엔진은 `process_startup()` 중에는 로딩되지 않습니다. 엔진 생성 여부는 메인 프로세스(사전 검사)와 스케줄러 작업 프로세스를 구분하는 검사에 따라 결정됩니다:

- **메인 프로세스의 사전 검사** (`resolve_backend`): `validate()`가 메인 프로세스에서 실행되며(`multiprocessing.parent_process()`가 `None`), 모델 가중치를 불러오지 않고 오버헤드가 적은 준비 상태 검사만 수행합니다.
- **작업 프로세스**: `multiprocessing.parent_process()`에 값이 있으면(`fork`와 `spawn` 작업 프로세스 모두 해당하며, `GUIDELLM__MP_CONTEXT_TYPE` 참고) `validate()`가 `_ensure_engine()`을 호출하여 엔진을 **미리 불러옵니다**. 이를 통해 벤치마크 시간 측정에서 초기 시작 시간을 제외합니다.
- **추론 시 보조 처리**: 데이터셋에서 요청을 생성할 때 `_ensure_engine()`을 멱등적인 보조 처리로 호출하므로, `validate()`를 건너뛰었더라도 추론이 정상적으로 실행됩니다.

## 관련 문서 {#see-also}

- [vLLM Python 비동기 백엔드](vllm-python-backend.md) -- 요청별로 비동기 처리하는 백엔드입니다.
- [백엔드](backends.md) -- 지원되는 백엔드 개요입니다.
- [벤치마크 실행](../getting-started/benchmark.md) -- 일반적인 벤치마크 옵션입니다.
- [vLLM 엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/) -- CLI 기준 문서입니다. `vllm_config`에는 Python 이름을 사용하세요.
- [vLLM 문서](https://docs.vllm.ai/)
