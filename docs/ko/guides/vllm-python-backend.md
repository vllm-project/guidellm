---
source_sha256: !!str "47fb621405c7bef81427fcd7828c704b202cd62ca7f21839a17c79523e4063a5"
---

# vLLM Python 백엔드 {#vllm-python-backend}

**vLLM Python 백엔드**(`vllm_python_async`)는 vLLM의 [AsyncLLMEngine](https://docs.vllm.ai/)을 사용하여 GuideLLM과 동일한 프로세스에서 추론을 실행합니다. HTTP 서버를 거치지 않으므로 네트워크 오버헤드가 없습니다. 성능 병목을 분리하여 확인하거나 벤치마크 설정을 간소화할 때 유용합니다. `target`은 전달하지 **않으며**, 백엔드 설정에 `model`을 **반드시** 전달해야 합니다. 지정한 모델은 vLLM에서 다운로드하여 실행합니다.

전체 엔진 옵션과 지원되는 모델을 확인하려면 vLLM의 [엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/) 및 [vLLM 문서](https://docs.vllm.ai/)를 참고하세요.

## 설치 {#installation}

### 권장 방법 {#recommended-methods}

- **공식 GuideLLM + vLLM 이미지**\
  vLLM 기본 이미지를 사용하는 이미지(예: [Containerfile.vllm](https://github.com/vllm-project/guidellm/blob/main/Containerfile.vllm))를 빌드하고 실행하세요. `vllm/vllm-openai`를 기반으로 GuideLLM을 추가 설치하므로, 동작이 검증된 vLLM + GuideLLM 환경과 기본 이미지가 제공하는 하드웨어 지원을 사용할 수 있습니다.

  **참고:** 이 방법은 GuideLLM보다 vLLM의 의존성 요구 사항을 우선합니다. vLLM이 더 복잡한 프로젝트이므로 이 구성을 권장하지만, 이전 버전의 Python이나 의존성을 사용하게 될 수 있습니다. 그 결과 일부 상황에서는 GuideLLM의 성능이나 동작이 최적이 아닐 수 있습니다.

- **기존 vLLM 설치 환경**\
  사용 환경(GPU/CPU, CUDA 등)에 맞게 vLLM을 먼저 설치한 뒤, 같은 환경에 GuideLLM을 설치하세요(예: `pip install guidellm` 또는 선택적 의존성을 포함한 설치). vLLM을 중복 설치하지 않고 기존의 가속 설정을 재사용할 수 있습니다.

  **참고:** 잠금 파일(lockfile)을 사용하여 vLLM Python 백엔드를 설치하면 하드웨어 가속에 적합한 의존성이 설치되지 않을 수 있습니다.

## 기본 예제 {#basic-example}

vLLM Python 백엔드로 벤치마크를 실행하려면 다음과 같이 설정하세요:

```bash
guidellm run \
  --backend kind=vllm_python_async,model=Qwen/Qwen3-0.6B \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=constant,rate=3 \
  --constraint kind=max_duration,seconds=20
```

백엔드 설정의 `vllm_config`로 재정의하지 않으면 장치, 메모리 등의 엔진 동작은 vLLM 기본값을 따릅니다. GPU 없이 실행할 때(예: GPU에 접근할 수 없는 GuideLLM + vLLM 컨테이너)는 `vllm_config`에 `device`를 지정하지 않는 한 백엔드가 자동으로 CPU 장치를 사용합니다. 엔진 설정 옵션은 vLLM의 [엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/)를 참고하세요.

## 요청 형식 및 백엔드 옵션 {#request-format-and-backend-options}

- **`request_format`**\
  채팅 프롬프트를 구성하는 방식을 제어합니다. `plain`(채팅 템플릿 없이 메시지 내용을 일반 텍스트로 연결), `default-template`(토크나이저의 기본 채팅 템플릿 사용), 또는 vLLM에서 지원하는 파일 경로나 한 줄짜리 템플릿 문자열을 지정할 수 있습니다. 이 값은 vLLM의 채팅 템플릿 처리에 그대로 전달됩니다. 자세한 내용은 vLLM의 [채팅 템플릿](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) 문서를 참고하세요.

- **`vllm_config`**\
  백엔드별 엔진 옵션은 백엔드 설정 안의 중첩된 딕셔너리로 전달합니다. 엔진 옵션 이름과 값으로 구성된 딕셔너리를 `vllm_config` 키의 값으로 지정하세요.

  **`vllm_config`에서 엔진 인자 사용하기:** [엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/) 문서는 옵션을 **CLI 형식**으로 설명합니다(예: `--gpu-memory-utilization`, `--max-model-len`). `vllm_config`에는 **Python 파라미터 이름**을 사용해야 합니다. 앞의 `--`를 제거하고 하이픈을 밑줄로 바꾸세요(예: `gpu_memory_utilization`, `max_model_len`). 키는 vLLM의 `EngineArgs`와 `AsyncEngineArgs` 데이터 클래스의 필드 이름과 같습니다. 허용되는 키와 자료형의 정확한 목록은 [vLLM 소스: `vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)에서 `class EngineArgs`를 검색하여 확인하세요.

  예제 — GPU 메모리 사용량과 컨텍스트 길이 제한:

  ```bash
  --backend '{"kind":"vllm_python_async","model":"Qwen/Qwen3-0.6B","vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

  전체 옵션과 자료형은 vLLM의 [엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/)(CLI 형식) 및 [EngineArgs 소스](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)(`vllm_config`에 사용할 Python 필드 이름)를 참고하세요.

> [!IMPORTANT]
>
> `vllm_python_async`의 백엔드 설정에는 `model` 필드가 필수입니다. `vllm_config` 안에도 `model`을 설정하면 최상위 `model` 필드가 우선 적용됩니다.

## 관련 문서 {#see-also}

- [백엔드](backends.md) — 지원되는 백엔드 개요입니다.
- [벤치마크 실행](../getting-started/benchmark.md) — 일반적인 벤치마크 옵션입니다.
- [vLLM 엔진 인자](https://docs.vllm.ai/en/stable/configuration/engine_args/) — CLI 기준 문서입니다. `vllm_config`에는 Python 이름(예: `gpu_memory_utilization`)을 사용하세요.
- [vLLM 소스: `vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) — `vllm_config` 키에 해당하는 `EngineArgs` / `AsyncEngineArgs` 필드 이름과 자료형입니다.
- [vLLM AsyncEngineArgs API](https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.AsyncEngineArgs) — 이 옵션을 받는 클래스의 API 레퍼런스입니다.
- [vLLM 채팅 템플릿](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) — `request_format`의 동작을 설명합니다.
- [vLLM 문서](https://docs.vllm.ai/)
- [vLLM 설치](https://docs.vllm.ai/en/latest/getting_started/installation)
- [vLLM OpenAI 호환 서버](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — Python 백엔드 대신 HTTP 서버를 사용하는 경우 참고하세요.
