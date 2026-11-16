---
weight: -8
source_sha256: !!str "4cd5aecda0fc2bd1a70721e8f14b9b95a74a79e17749fa4e9222830b7c2628ae"
---

# 서버 실행

GuideLLM 벤치마크를 실행하려면 먼저 벤치마크 대상이 될 OpenAI 호환 서버가 필요합니다. 이 가이드에서는 서버를 빠르게 설정하는 방법을 설명합니다.

## **권장 옵션: vLLM**

vLLM은 성능과 호환성 측면에서 GuideLLM 벤치마크 실행에 권장되는 백엔드입니다.

### vLLM 설치

```bash
pip install vllm
```

### vLLM 서버 실행

양자화된 Llama 3.1 8B 모델로 vLLM 서버를 실행하려면 다음 명령어를 실행하세요:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

이 명령어를 실행하면 `http://localhost:8000`에서 OpenAI 호환 서버가 실행됩니다.

추가 설정 옵션은 [vLLM 문서](https://docs.vllm.ai/en/latest/)를 참고하세요.

## **다른 서버**

GuideLLM은 TGI, SGLang 등 모든 OpenAI 호환 서버를 지원합니다. 지원되는 모든 백엔드에 대한 자세한 내용은 [백엔드 문서(영문)](../../guides/backends.md)를 참고하세요.

## **서버 동작 확인**

서버가 실행되면 간단한 curl 명령어로 서버가 정상적으로 동작하고 벤치마크를 실행하는 서버에서 접근할 수 있는지 확인할 수 있습니다(모델 서버가 다른 장비에서 실행 중이라면 `localhost`를 해당 서버의 IP 주소로 변경하세요):

```bash
curl http://localhost:8000/v1/models
```

서버에서 사용 가능한 모델 목록이 응답으로 표시되어야 합니다.
