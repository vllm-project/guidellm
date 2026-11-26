---
weight: 15
source_sha256: !!str "4746fb9ee04e8a0307e680518dd67c832e94c67f8705760ea9e9ee6283cbdbda"
---

# 문제 해결

아래에서 해당하는 증상을 찾고, 연결된 항목의 해결 방법을 확인하세요. CLI 문법은 [벤치마크 실행](../getting-started/benchmark.md#cli-option-format)을 참고하세요.

| 증상                                                                   | 항목                                                                         |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 요청이 실패하거나 결과가 올바르지 않음                                 | [디버그 logging](#debug-logging)                                             |
| 요청이 예정 시각보다 늦게 도착하거나 trace를 완전히 불러오지 못함      | [요청이 예정 시각보다 늦게 로딩되는 경우](#requests-load-after-they-are-due) |
| 모델의 토크나이저를 불러올 때 사용자 지정 코드 오류 발생               | [토크나이저: trust_remote_code](#tokenizer-trust_remote_code)                |
| macOS에서 `Worker process ... died unexpectedly (signal 11)` 오류 발생 | [macOS 작업 프로세스의 비정상 종료](#macos-worker-crash-signal-11)           |

## 디버그 logging {#debug-logging}

요청 처리와 작업 프로세스의 시작 과정을 확인하려면 디버그 출력을 활성화하세요:

```bash
GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG guidellm run ... --disable-progress
```

`guidellm env`를 실행하여 설정이 적용되었는지 확인하세요. `--disable-progress`는 선택 사항이지만, 대화형 진행 상황 표시가 콘솔 로그 메시지를 덮어쓸 수 있습니다. 또는 [logging 가이드(영문)](../../developer/developing.md#logging)에서 설명하는 파일 로그를 사용할 수 있습니다.

모든 logging 옵션(파일 출력, 로그 수준)은 개발 가이드의 [Logging(영문)](../../developer/developing.md#logging) 항목을 참고하세요.

## 토크나이저: trust_remote_code {#tokenizer-trust_remote_code}

### 증상

다음과 같은 오류가 발생합니다:

```text
The repository moonshotai/Kimi-K2.6 contains custom code which must be executed
to correctly load the model. You can inspect the repository content at
https://hf.co/moonshotai/Kimi-K2.6.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
```

### 해결 방법

모델을 완전히 신뢰하는 경우, `--tokenizer`의 `load_kwargs`를 통해 `trust_remote_code`를 전달하세요:

```bash
--tokenizer '{"kind":"huggingface_auto","load_kwargs":{"trust_remote_code":true}}'
```

이 설정은 사용자 컴퓨터에서 코드를 실행하도록 허용하므로, 모델을 신뢰하지 않는 경우에는 사용하지 마세요.

다른 토크나이저 옵션은 [데이터셋: 토크나이저](datasets.md#tokenizer)를 참고하세요.

## 요청이 예정 시각보다 늦게 로딩되는 경우 {#requests-load-after-they-are-due}

### 증상

데이터 생성으로 인해 벤치마크 실행이 느려지거나, 대화가 늦게 로딩되거나, trace 데이터셋을 완전히 불러오지 못했다는 경고가 표시됩니다.

### 해결 방법

`--profile`의 `prefetch`를 사용하여 실행 시간 측정을 시작하기 전에 요청 그래프를 불러오세요:

- `0`(기본값)은 실행 중 요청 그래프가 필요할 때 불러옵니다.
- 개수를 지정하면 해당 개수만큼 요청 노드를 먼저 불러옵니다.
- `start`는 실행 구성에 시작 시 동시 실행 수가 정의되어 있으면 그 수만큼 불러옵니다.
- `all`은 크기가 유한한 데이터셋의 모든 요청을 불러옵니다. 무한한 데이터 소스에서는 `--data-loader kind=pytorch,samples=<n>`으로 샘플 수를 제한하지 않으면 실패합니다.

모든 프로파일은 `prefetch` 파라미터를 지원합니다. 다음은 `concurrent` 프로파일에서 사용하는 예제입니다:

```bash
--profile kind=concurrent,prefetch=all
```

WEKA trace에는 `prefetch=all`을 사용해야 합니다. 각 세션에는 시각 0 또는 그 부근에 시작하는 요청이 있으므로, 실행 중에 데이터를 불러오면 필요한 시각까지 데이터가 준비되지 않아 대화가 늦게 시작됩니다.

## macOS 작업 프로세스의 비정상 종료(signal 11) {#macos-worker-crash-signal-11}

### 증상

```text
RuntimeError: Worker process group startup failed: Worker process <pid> died
unexpectedly (signal 11). Check system logs for details
```

다음과 같은 로그가 반복해서 표시될 수도 있습니다:

```text
Worker process <pid> died unexpectedly (signal 11)
```

### 해결 방법

GuideLLM 0.8.0 이상은 모든 플랫폼에서 기본적으로 `spawn` 방식의 멀티프로세싱을 사용하며 더 이상 `fork`를 허용하지 않으므로, 기본 설정에서는 이 문제가 발생하지 않아야 합니다. 0.8.0 이전 버전은 기본적으로 `fork`를 사용하므로 macOS에서 세그멘테이션 오류가 발생할 수 있습니다. 해당 버전에서는 `spawn`을 명시적으로 선택하세요:

```bash
GUIDELLM__MP_CONTEXT_TYPE=spawn guidellm run ...
```
