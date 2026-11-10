---
source_sha256: !!str "2cc877926f72d888605a25f9028265e4c2627b97edd20cbcd51a1c44919f276f"
weight: -10
---

# 설치

GuideLLM은 요구 사항에 따라 여러 방법으로 설치할 수 있습니다. 아래에서 각 설치 방법을 자세히 설명합니다.

## 사전 요구 사항

GuideLLM을 설치하기 전에, 다음 요구 사항을 충족하는지 확인하세요:

- **운영 체제:** Linux 또는 MacOS

- **Python 버전:** 3.10 – 3.13

- **Pip 버전:** 최신 버전의 pip가 설치되어 있는지 확인하세요. 다음 명령어를 사용하여 pip를 업그레이드할 수 있습니다:

  ```bash
  python -m pip install --upgrade pip
  ```

## 설치 방법

### 1. PyPI에서 최신 버전 설치

GuideLLM을 설치하는 가장 간단한 방법은 pip를 사용해 Python Package Index(PyPI)에서 설치하는 것입니다:

```bash
pip install guidellm[recommended]
```

이 명령어를 통해 권장 의존성과 함께 GuideLLM의 최신 안정 버전을 설치할 수 있습니다.

### 2. PyPI에서 특정 버전 설치

GuideLLM의 특정 버전이 필요한 경우 설치 시 버전 번호를 지정할 수 있습니다. 예를 들어 `0.2.0` 버전을 설치하려면 다음 명령어를 실행하세요:

```bash
pip install guidellm==0.2.0
```

### 3. Main 브랜치의 소스 코드에서 설치

Main 브랜치에서 GuideLLM의 최신 개발 버전을 설치하려면 다음 명령어를 실행하세요:

```bash
pip install git+https://github.com/vllm-project/guidellm.git
```

이 명령어는 저장소를 복제하고 main 브랜치에서 GuideLLM을 직접 설치합니다.

### 4. 특정 브랜치에서 설치

특정 브랜치(예: `feature-branch`)에서 GuideLLM을 설치하려면 다음 명령어를 실행하세요:

```bash
pip install git+https://github.com/vllm-project/guidellm.git@feature-branch
```

`feature-branch`를 설치하려는 브랜치 이름으로 변경하세요.

### 5. 로컬에 복제한 저장소에서 설치

GuideLLM 저장소를 로컬에 복제한 경우, 저장소 디렉터리로 이동한 뒤 다음 명령어를 실행하세요:

```bash
pip install .
```

개발 목적으로 사용하는 경우에는 편집 가능 모드로 설치할 수도 있습니다:

```bash
pip install -e .
```

이 모드에서는 소스 코드의 변경 사항이 재설치 없이 즉시 반영됩니다.

## 설치 확인

설치 후 다음 명령어를 실행하여 GuideLLM이 올바르게 설치되었는지 확인할 수 있습니다:

```bash
guidellm --help
```

이 명령어를 실행하면 설치된 GuideLLM의 도움말이 표시됩니다.

vLLM Python 백엔드(프로세스 내 추론)를 사용하려면 [vLLM Python 백엔드(영문)](../../guides/vllm-python-backend.md) 문서에서 권장 설치 방법(컨테이너 또는 기존 vLLM 환경)과 pip 설치 시 참고 사항을 확인하세요.

## 문제 해결

설치 중 문제가 발생하면 Python과 pip 버전이 사전 요구 사항을 충족하는지 확인하세요. 일반적인 실행 오류에 대한 해결 방법(디버그 logging, 토크나이저 로딩, macOS 작업 프로세스의 비정상 종료)은 [문제 해결 가이드(영문)](../../guides/troubleshooting.md)를 참고하세요. 추가 도움이 필요하면 [GitHub Issues](https://github.com/vllm-project/guidellm/issues) 페이지를 참고하세요.
