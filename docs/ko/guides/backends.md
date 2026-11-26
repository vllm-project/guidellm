---
weight: -3
source_sha256: !!str "c9aec42abdccd6a24fb8022ca60b70d56a2286f3d4b914931642a0962eb07ed6"
---

# 백엔드

GuideLLM은 OpenAI 호환 HTTP 서버와 함께 동작하도록 설계되어 다양한 생성형 AI 백엔드와 원활하게 연동할 수 있습니다. 이러한 호환성을 통해 대규모 언어 모델(LLM) 배포 환경을 효율적으로 평가하고 최적화할 수 있습니다. 현재는 OpenAI 호환 서버에 중점을 두고 있지만, 추가 서버 구현체와 Python 인터페이스 등 다른 백엔드로 지원을 확장하는 기여도 환영합니다.

## CLI 백엔드 설정

백엔드는 `--backend` 옵션으로 설정합니다. 한 명령어에는 백엔드를 하나만 지정할 수 있습니다. `kind=<TYPE>`으로 등록된 백엔드 유형을 선택하고, key=value 쌍으로 파라미터를 설정하세요:

```bash
guidellm run --backend kind=<TYPE>,key=value,...
```

HTTP 서버의 경우 `kind=openai_http`와 함께 대상 URL 및 기타 연결 설정을 전달하세요:

```bash
--backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

중첩 구조가 없는 설정에는 쉼표로 구분한 key=value 쌍을 사용하고, 중첩된 설정에는 직렬화한 JSON 또는 YAML을 사용하세요. 주요 `openai_http` 파라미터에는 `target`, `model`, `request_format`, `api_key`, `stream`, `verify`, `timeout` 및 요청 본문, 헤더, 쿼리 파라미터를 위한 중첩된 `extras`가 포함됩니다:

```bash
--backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}'
```

## 지원되는 백엔드

### OpenAI 호환 HTTP 서버

GuideLLM은 LLM과 상호작용하기 위한 표준화된 API를 제공하는 OpenAI 호환 HTTP 서버를 지원합니다. 여기에는 [vLLM](https://github.com/vllm-project/vllm) 및 [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)와 같이 널리 사용되는 구현체가 포함됩니다. 이러한 서버를 사용하면 최소한의 설정으로 GuideLLM의 평가, 벤치마킹 및 최적화를 수행할 수 있습니다.

### vLLM Python 백엔드

GuideLLM은 **vLLM Python 백엔드**(`vllm_python_async`)를 사용하여 동일한 프로세스에서 추론을 실행할 수 있습니다. 이 백엔드는 HTTP 서버 없이 vLLM의 Python API(AsyncLLMEngine)를 사용하여 GuideLLM과 동일한 프로세스에서 추론을 실행합니다. 설정, 설치 옵션(컨테이너, 기존 vLLM 환경, pip) 및 예제는 [vLLM Python 백엔드](vllm-python-backend.md)를 참고하세요.

### vLLM Python 배치 백엔드

**vLLM Python 배치 백엔드**(`vllm_python_batch`)는 vLLM의 동기식 `LLM` 엔진을 사용하여 배치 단위로 추론을 실행합니다. 요청을 큐에 추가하고 설정한 배치 단위로 전송하여 요청별 스케줄링 오버헤드를 제거합니다. 처리량 벤치마킹에 적합한 방식입니다. 설정 및 예제는 [vLLM Python 배치 백엔드](vllm-python-batch-backend.md)를 참고하세요.

## 호환 서버 실행 예제

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm)은 효율적인 LLM 추론을 위해 설계된 고성능 OpenAI 호환 서버입니다. 다양한 모델을 지원하며 배포를 위한 간단한 인터페이스를 제공합니다.

먼저 vLLM이 설치되어 있는지 확인하고(`pip install vllm`), 다음 명령어를 실행하여 양자화된 Llama 3.1 8B 모델로 vLLM 서버를 시작하세요:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

vLLM 서버 실행에 대한 자세한 내용은 [vLLM 문서](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)를 참고하세요.

### 2. Text Generation Inference (TGI)

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)는 Hugging Face에서 호스팅하는 모델을 포함하여 다양한 모델을 지원하는 또 다른 OpenAI 호환 서버입니다. TGI는 높은 처리량과 낮은 지연 시간의 추론에 최적화되어 있습니다.

Docker를 사용하여 Llama 3.1 8B 모델로 TGI 서버를 시작하려면 다음 명령어를 실행하세요:

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

TGI 서버 실행에 대한 자세한 내용은 [TGI 문서](https://huggingface.co/docs/text-generation-inference/index)를 참고하세요.

### 3. llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp)는 [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server) 도구를 통해 경량 OpenAI 호환 서버를 제공합니다.

gpt-oss-20b 모델로 llama.cpp 서버를 시작하려면 다음 명령어를 사용할 수 있습니다:

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --alias gpt-oss-20b --ctx-size 0 --jinja -ub 2048 -b 2048
```

모델 이름에 `gpt-oss-20b`라는 별칭을 지정하는 이유는 GuideLLM이 이 이름으로 JSON 형식의 모델 메타데이터를 가져오며, GGUF 모델 저장소에는 이러한 메타데이터가 포함되어 있지 않기 때문입니다. 간단한 해결 방법은 safetensors 저장소에서 메타데이터 파일을 다운로드하고, 별칭과 같은 이름의 로컬 디렉터리에 저장하는 것입니다:

```bash
huggingface-cli download openai/gpt-oss-20b --include "*.json" --local-dir gpt-oss-20b/
```

이제 평소처럼 `guidellm`을 실행하면 로컬 디렉터리에서 모델 메타데이터를 가져올 수 있습니다.

## API 키 설정 {#api-key-configuration}

일부 OpenAI 호환 서버는 API 키를 통한 인증을 요구합니다. 일반적으로 다음과 같은 경우에 필요합니다:

- OpenAI API에 직접 연결하는 경우
- 인증이 필요한 호스팅 또는 클라우드 기반 추론 서비스를 사용하는 경우
- 인증이 활성화된 서버에 연결하는 경우

vLLM과 같은 로컬 서버는 인증을 명시적으로 설정하지 않았다면 일반적으로 API 키가 필요하지 않습니다.

### API 키 지정

벤치마크 실행 시 API 키를 제공하려면 백엔드 설정으로 전달하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=https://api.openai.com/v1,api_key=sk-...,model=gpt-3.5-turbo \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

또는 JSON을 사용하세요:

```bash
--backend '{"kind":"openai_http","target":"https://api.openai.com/v1","api_key":"sk-...","model":"gpt-3.5-turbo"}'
```

API 키는 백엔드 서버로 보내는 HTTP 요청의 `Authorization: Bearer {api_key}` 헤더를 설정하는 데 사용됩니다.

> [!IMPORTANT]\
> 보안을 위해 스크립트에 API 키를 하드코딩하지 마세요. `--backend`로 API 키를 전달할 때는 환경 변수나 안전한 자격 증명 관리 도구를 사용하는 것을 고려하세요.

## 샘플링 파라미터 전달

기본적으로 GuideLLM은 백엔드 서버에 보내는 요청에 `temperature`, `top_p`, `top_k`와 같은 샘플링 파라미터를 설정하지 않습니다. 벤치마킹 중에 이러한 파라미터를 제어하려면 백엔드의 `extras` 필드로 전달하세요.

`extras` 필드는 `body` 키를 받으며, 해당 키의 값은 백엔드 서버로 보내는 API 요청 본문에 직접 병합됩니다. 따라서 OpenAI completions 또는 chat completions API나 백엔드의 확장 기능이 지원하는 모든 파라미터를 전달할 수 있습니다.

### 예제: temperature, top_p 및 top_k 설정

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

이렇게 설정하면 서버로 보내는 모든 요청 본문에 `temperature`, `top_p`, `top_k`가 포함됩니다.

## 시퀀스 종료 동작 제어

정확히 지정한 출력 길이에서 처리량과 지연 시간을 측정하기 위해 GuideLLM은 서버에 요청한 수만큼의 토큰을 생성하도록 지시합니다. `openai_http` 백엔드에서 특정 출력 토큰 수를 요청하면 각 요청에 `ignore_eos: true`를 설정하고, `max_tokens`/`max_completion_tokens` 및 `stop: null`도 함께 설정합니다. 이를 통해 모델이 시퀀스 종료 토큰을 출력해도 서버가 멈추지 않고 생성을 계속하도록 합니다.

일부 모델에서는 시퀀스 종료 토큰을 무시하면 안 됩니다. Harmony / `gpt-oss`와 같은 형식은 모델이 자체 턴 종료 토큰에서 생성을 멈추는 것을 전제로 합니다. 해당 토큰 이후에도 생성을 강제로 계속하면 서버가 뒤따르는 토큰을 거부하여 요청이 실패합니다. 이러한 모델이나 유사한 모델에서 `ignore_eos`를 비활성화하려면 `false`를 백엔드의 `extras.body` 필드로 전달하세요:

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"openai/gpt-oss-20b","extras":{"body":{"ignore_eos":false}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

또는 GuideLLM의 간결한 `key=value` 파서를 사용하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=openai/gpt-oss-20b,extras.body.ignore_eos=false \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

`extras.body`의 값은 요청 본문에 병합되고 기본값보다 우선 적용되므로, `ignore_eos: false`는 기본 설정인 `true`를 덮어씁니다. 시퀀스 종료 토큰을 무시해도 문제가 없는 모델에서는 `ignore_eos`를 설정하지 않은 상태(기본값)로 두세요. 그러면 GuideLLM이 출력 길이를 정확히 제어할 수 있습니다.

> [!NOTE] `ignore_eos: true`를 제거하면 모델이 자체적으로 생성을 중단할 수 있으므로, 시퀀스 길이가 원하는 것보다 상당히 짧아질 수 있습니다.

## 구조화된 채팅 콘텐츠 페이로드

일부 채팅 템플릿은 구조화된 각 콘텐츠 객체에 텍스트와 함께 메타데이터를 요구합니다. 이러한 필드는 `extras.content`를 통해 `openai_http` 백엔드 설정으로 전달하세요. GuideLLM은 Chat Completions 및 Responses API 요청용으로 생성하는 모든 텍스트 콘텐츠 객체에 해당 필드를 추가합니다.

```bash
guidellm run \
  --backend '{
    "kind": "openai_http",
    "target": "http://localhost:8000",
    "model": "google/translategemma-12b-it",
    "request_format": "/v1/chat/completions",
    "extras": {
      "content": {
        "source_lang_code": "en",
        "target_lang_code": "es"
      }
    }
  }' \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --constraint kind=max_duration,seconds=60
```

### 동작 방식

`--backend` 설정은 백엔드 생성자의 키워드 인자로 파싱됩니다. 설정 내의 `extras` 필드는 다음 하위 필드를 지원하는 `GenerationRequestArguments` 객체에 매핑됩니다:

- `body`: HTTP 요청 본문에 병합할 키-값 쌍의 딕셔너리입니다. `temperature`, `top_p`, `top_k`, `repetition_penalty` 등의 샘플링 파라미터에 사용하세요.
- `content`: 생성되는 각 텍스트 콘텐츠 객체에 병합할 필드의 딕셔너리입니다.
- `headers`: 요청에 포함할 추가 HTTP 헤더의 딕셔너리입니다.
- `params`: 요청 URL에 추가할 쿼리 파라미터의 딕셔너리입니다.

### 예제: 샘플링 파라미터와 다른 백엔드 옵션 함께 사용하기

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","api_key":"sk-...","extras":{"body":{"temperature":0.8,"top_p":0.9}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

## 백엔드 지원 확장

GuideLLM은 개방형 플랫폼이며, 백엔드 지원을 확장하기 위한 기여를 권장합니다. 새로운 서버 구현체 추가, Python 기반 백엔드 연동 또는 기존 기능 개선 등 다양한 기여를 환영합니다. 기여 방법에 대한 자세한 내용은 [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md) 파일을 참고하세요.
