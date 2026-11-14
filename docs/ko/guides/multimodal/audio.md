---
source_sha256: !!str "9bfcb3c145e3d74e96826259f71b20106588111e49badd14821e0667b0c04810"
---

# 오디오 벤치마킹

이 가이드에서는 음성 인식(ASR, `audio/transcriptions`), 음성 번역(`audio/translations`), 오디오 채팅(`chat/completions`) 작업에서 오디오 모델을 벤치마킹하는 방법을 설명합니다.

## 설정

먼저 사용할 오디오 API와 호환되는 모델이 추론 서버에서 실행 중인지 확인하세요. GuideLLM은 chat/completions, audio/transcriptions 또는 audio/translations 엔드포인트를 통해 오디오를 입력받을 수 있는 모든 OpenAI 호환 서버를 지원합니다. 아래 예제에서는 vLLM으로 음성 인식과 번역에는 Whisper 모델을, 채팅에는 Ultravox를 실행합니다. 각 서버는 다음과 같이 시작할 수 있습니다:

```bash
# Whisper ASR/Translation
vllm serve openai/whisper-small

# Ultravox Audio Chat
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

다음으로, 같은 인스턴스나 서버에 접근할 수 있는 다른 장비(권장)에 오디오 처리에 필요한 의존성과 함께 GuideLLM을 설치하세요:

```bash
pip install guidellm[audio,recommended]
```

마지막으로, 지원되는 형식의 오디오 파일이 포함된 벤치마크용 데이터셋을 준비하세요. GuideLLM은 Hugging Face 데이터셋, 로컬 파일, URL 등에서 오디오 데이터를 불러올 수 있습니다. 아래 예제에서는 `openslr/librispeech_asr` 데이터셋을 사용합니다.

## 처리 옵션

`--profile`, 프로파일의 요청률 파라미터, `--constraint kind=max_requests,count=<n>` 등 일반 벤치마크 인자는 오디오 작업에도 동일하게 적용됩니다. 오디오 데이터를 처리하고 요청을 구성하는 방식은 다음 추가 옵션으로 설정할 수 있습니다.

### 데이터 로딩

GuideLLM은 여러 방식으로 오디오 데이터를 불러올 수 있습니다. 먼저 GuideLLM이 데이터 소스를 역직렬화하여 Hugging Face 데이터셋으로 변환할 수 있어야 합니다. 로컬 파일, Hugging Face 데이터셋, JSON 파일 등을 사용할 수 있습니다.

데이터 소스에서 사용할 오디오 열은 GuideLLM의 오디오 디코더와 인코더가 지원하는 형식이어야 합니다. 지원되는 형식은 다음과 같습니다:

- Hugging Face의 Audio 특성(권장)
- 로컬 파일 경로(예: .wav, .mp3, .flac)
- 오디오 파일의 URL
- Base64로 인코딩한 오디오 데이터
- 원본 오디오 샘플이 담긴 NumPy 또는 PyTorch 배열(모노 오디오의 1차원 배열 또는 첫 번째 차원이 채널인 배열)

원본 부동 소수점 배열에는 샘플링 주파수를 지정해야 하며, 인코딩할 때 배열을 float32로 변환합니다. 예를 들어 오디오 항목을 `{"data": samples, "sample_rate": 16000}` 형식으로 전달할 수 있습니다. 이때 `samples`의 형태는 모노 오디오의 경우 `(num_samples,)`, 다채널 오디오의 경우 `(num_channels, num_samples)`입니다.

### 데이터 열 매핑

데이터셋을 지정할 때는 사용할 오디오 열을 GuideLLM의 audio_column에 매핑하여 어떤 데이터를 오디오로 처리할지 알려주세요. 지정하지 않으면 GuideLLM이 audio, speech, wav 등 일반적으로 사용하는 이름을 기준으로 오디오 열을 자동으로 찾습니다.

열 매핑은 `--data-column-mapper` 인자에 JSON 문자열로 전달합니다. audio_column의 값으로 데이터셋에 있는 열 이름을 지정하세요. 예를 들어 오디오 열의 이름이 speech_data이면 다음과 같이 설정합니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"audio_column": "speech_data"}}'
```

여러 데이터셋을 결합하는 경우(예: 프롬프트와 오디오를 각각 다른 데이터셋에서 가져오는 경우) 열 이름 앞에 0부터 시작하는 데이터셋 인덱스나 데이터셋 별칭을 붙이고 점으로 구분하세요. 예를 들어 오디오 열이 두 번째 데이터셋(인덱스 1)에 있으면 다음과 같이 지정합니다:

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.audio_column": "speech_data"}}'
```

### 요청 형식

지원되는 각 오디오 엔드포인트에서 요청 포매터는 오디오 데이터를 인코딩하고 요청 페이로드를 구성합니다. 기본 설정을 그대로 사용하거나 필요에 따라 변경할 수 있습니다. 다음 오디오 요청 옵션을 `--request-formatter-kwargs` 인자에 JSON 문자열로 전달하세요.

#### "encode_kwargs"

오디오 인코더에 전달하는 인자 딕셔너리입니다. 요청에 오디오를 포함하기 전에 오디오 데이터를 어떻게 전처리할지 지정합니다.

**중첩 구조 사용 시 참고 사항:**

- **Chat Completions**(`chat_completions`)의 경우 오디오 인자는 `encode_kwargs` 안의 `audio` 키 아래에 지정해야 합니다.
- **음성 인식 및 번역**(`audio_transcriptions`, `audio_translations`)의 경우 인자를 `encode_kwargs`의 최상위에 지정합니다.

지원되는 인자는 다음과 같습니다:

- "sample_rate": 입력 오디오 데이터의 샘플링 주파수입니다. 원본 NumPy 또는 PyTorch 배열처럼 샘플링 주파수를 추론할 수 없는 경우에만 필요합니다.
- "encode_sample_rate": API로 전송할 오디오의 목표 샘플링 주파수입니다(기본값: 16000Hz).
- "audio_format": 페이로드의 파일 형식입니다. "wav", "mp3", "flac"을 지원합니다. 지정하지 않으면 원본 오디오 코덱에서 형식을 자동으로 감지합니다. 원본 배열처럼 형식을 감지할 수 없는 경우에는 "wav"를 사용합니다.
- "bitrate": mp3와 같은 손실 압축 형식의 비트 전송률입니다(기본값: "64k").
- "max_duration": 이 값을 초 단위로 지정하면 지정한 길이를 초과하는 오디오의 뒷부분을 잘라냅니다.
- "mono": 오디오를 모노로 변환할지 지정합니다(기본값: True).
- "file_name": 요청 메타데이터에 포함할 파일 이름입니다. 선택 사항이며, 파일 확장자를 사용하는 엔드포인트에서 유용합니다. 기본값은 "audio.wav"입니다.

**예제:**

**음성 인식**에서 오디오를 16kHz WAV로 변환하려면 다음과 같이 중첩 없이 설정하세요:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio_format": "wav", "encode_sample_rate": 16000}}'
```

**오디오 채팅**에서 오디오 길이를 30초로 제한하려면 다음과 같이 중첩된 형식으로 설정하세요:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio": {"max_duration": 30.0}}}'
```

#### "extras"

요청에 직접 포함할 추가 인자 딕셔너리입니다. Whisper 모델의 언어 설정처럼 엔드포인트별 파라미터를 직접 지정할 수 있습니다. extras에서 추가 인자를 포함할 위치를 지정하세요:

- "headers": 요청 헤더에 포함
- "params" / "body": 요청 파라미터 또는 본문에 포함(엔드포인트에 따라 자동 감지)

예를 들어 음성 번역 요청의 목표 언어를 프랑스어로 지정하려면 다음과 같이 설정하세요:

```bash
--request-formatter-kwargs '{"extras": {"body": {"language": "fr"}}}'
```

#### "stream"

백엔드가 지원하는 경우 불리언 값으로 응답 스트리밍을 활성화하거나 비활성화할 수 있습니다. 기본값은 활성화입니다. 비활성화하려면 백엔드 설정에 `stream=false`를 전달하세요:

```bash
--backend kind=openai_http,target=http://localhost:8000,stream=false
```

## 예상 결과

GuideLLM은 요청의 전체 처리 과정에서 지표를 수집하여 `GenerativeRequestStats`에 저장하고 `GenerativeMetrics`에 집계합니다. 결과는 콘솔에 표시되며, 추가 분석을 위해 로컬 파일에도 저장됩니다.

### 출력 파일

- **`benchmarks.json`**: 스케줄러의 시간 정보, 요청 분포, 텍스트와 오디오에 대한 상세 지표 요약을 포함하는 전체 통계 객체를 계층 구조로 저장합니다.
- **`benchmarks.csv`**: 벤치마크별로 처리량, 지연 시간 및 입출력 지표를 한 행에 요약하여 저장합니다. 개별 요청의 상세 데이터는 포함하지 않습니다.
- **`benchmarks.html`**: 성능을 시각적으로 요약한 보고서입니다.

### 수집되는 지표

오디오 벤치마크에서는 지연 시간, 첫 토큰 생성까지 걸리는 시간(TTFT), 토큰 간 지연 시간(ITL) 등 일반적인 성능 지표와 함께 입력과 출력에 대해 다음 사용량 지표를 추적합니다:

- **오디오 토큰 수**: 모델이 지원하는 경우, 처리한 오디오 토큰의 수입니다.
- **오디오 샘플 수**: 원본 오디오 샘플의 수입니다.
- **오디오 길이**: 오디오 콘텐츠의 총 재생 시간(초)입니다.
- **오디오 바이트 수**: 오디오 페이로드의 크기(바이트)입니다.
- **텍스트 지표**: 음성 인식 결과나 채팅 응답의 토큰 수, 단어 수 및 문자 수도 추적합니다.

### 통계 분석

GuideLLM은 위의 각 지표에 대해 다음 통계를 계산합니다:

- **측정값**: 평균, 중앙값, P95, P99, 최솟값 및 최댓값입니다.
- **처리율**: 초당 처리량입니다(예: `audio_seconds/sec`).
- **동시 처리 수**: 동시에 처리 중인 작업의 수를 나타냅니다.

이 통계는 `StatusDistributionSummary` 구조를 사용하여 성공한 요청, 완료되지 않은 요청, 오류가 발생한 요청을 구분하여 추적합니다.

## 예제

### 1. 음성 인식(ASR)

이 벤치마크에서는 Whisper와 같은 음성 인식(ASR) 모델이 음성을 텍스트로 변환하는 작업을 평가합니다. 위의 vLLM 명령어로 Whisper를 실행하거나 음성 인식 엔드포인트를 지원하는 유사한 모델을 사용하세요. 이 예제에서는 LibriSpeech 데이터셋의 오디오만 사용하지만, 오디오 채팅 예제처럼 필요에 따라 프롬프트를 함께 전달할 수도 있습니다.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/audio/transcriptions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}'
```

**주요 파라미터**

- `--backend`: 서버 URL 및 음성 인식을 위한 `request_format=/v1/audio/transcriptions` 설정
- `--profile kind=synchronous`: 요청을 순차적으로 실행
- `--constraint kind=max_requests,count=20`: 벤치마크의 전체 요청 수를 20개로 제한
- `--data`: Hugging Face 데이터셋을 지정하고 `load_kwargs`로 "clean" 설정과 "test" 분할을 선택
- `--data-column-mapper`: 데이터셋의 오디오 열을 GuideLLM의 `audio_column`에 매핑

이 명령어는 LibriSpeech 데이터셋의 오디오를 입력하여 대상 서버의 음성 인식용 audio/transcriptions 엔드포인트를 벤치마킹합니다. 다음과 비슷한 결과가 출력됩니다:

```bash
✔ OpenAIHTTPBackend backend validated with model openai/whisper-small

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7565.1 | 7329.1 | 16000.0 | 16000.0 | 129722.1 | 141848.5 | 6.4  | 16.8  | 75.3 | 72.9 | 52172.0 | 135692.0 | 610195.0 | 592749.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```

### 2. 음성 번역

이 벤치마크에서는 Whisper와 같은 음성 번역 모델이 한 언어의 음성을 다른 언어의 텍스트로 변환하는 작업을 평가합니다. 위의 vLLM 명령어로 Whisper를 실행하거나 음성 번역 엔드포인트를 지원하는 유사한 모델을 사용하세요. 이 예제에서는 LibriSpeech 데이터셋의 오디오만 사용하지만, 오디오 채팅 예제처럼 필요에 따라 프롬프트를 함께 전달할 수도 있습니다.

**명령어:**

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","request_format":"/v1/audio/translations","extras":{"body":{"language":"fr"}}}' \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}'
```

**주요 파라미터:**

- `--backend`: 서버 URL과 번역 엔드포인트를 지정하고 `extras.body`로 목표 언어 설정
- `--profile kind=synchronous`: 순차 실행 모드
- `--constraint kind=max_requests,count=20`: 테스트의 요청 수를 20개로 제한
- `--data`: Hugging Face 데이터셋을 지정하고 `load_kwargs`로 "clean" 설정과 "test" 분할을 선택합니다. 지원되는 전체 옵션은 [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset)을 참고하세요.
- `--data-column-mapper`: 오디오 처리에 사용할 열 지정

이 명령어는 LibriSpeech 데이터셋의 오디오를 사용하여 대상 서버의 audio/translations 엔드포인트에 프랑스어 번역을 요청합니다. 다음과 비슷한 벤치마크 결과가 출력됩니다:

```bash
✔ OpenAIHTTPBackend backend validated with model openai/whisper-small

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7483.6 | 7563.5 | 16000.0 | 16000.0 | 133404.0 | 146385.0 | 6.4  | 16.8  | 74.5 | 75.2 | 52172.0 | 135692.0 | 603620.5 | 611706.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html

```

### 3. 오디오 채팅(Chat Completions)

이 벤치마크에서는 Ultravox처럼 대화 형식으로 오디오 입력을 처리하는 모델을 평가합니다. 위의 vLLM 명령어로 Ultravox를 실행하거나 Chat Completions API에서 오디오를 지원하는 유사한 모델을 사용하세요. 다음 예제에서는 LibriSpeech 데이터셋과 함께 텍스트 프롬프트용 합성 데이터셋을 사용합니다. 활용 사례에 맞게 데이터셋과 열 매핑을 변경하세요.

**명령어:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"1.audio","text_column":"0.prompt"}}'
```

**주요 파라미터**

- `--backend`: 서버 URL과 멀티모달 입력용 chat completions 엔드포인트 지정
- `--profile kind=synchronous`: 요청을 순차적으로 실행
- `--constraint kind=max_requests,count=20`: 요청 수를 20개로 제한
- `--data`: 두 번 지정합니다. 첫 번째는 합성 프롬프트(`kind=synthetic_text`), 두 번째는 `openslr/librispeech_asr`의 실제 오디오에 사용합니다(`kind=huggingface`와 `load_kwargs`로 데이터셋 설정 지정).
- `--data-column-mapper`: 데이터셋 인덱스 1의 오디오(`"1.audio"`, LibriSpeech)와 데이터셋 인덱스 0의 텍스트(`"0.prompt"`, 합성 프롬프트)를 각 요청에 매핑합니다.

이 명령어는 합성 데이터셋의 프롬프트 텍스트와 LibriSpeech 데이터셋의 오디오를 사용하여 대상 서버의 chat/completions 엔드포인트를 벤치마킹합니다. 다음과 비슷한 결과가 출력됩니다:

```bash
✔ OpenAIHTTPBackend backend validated with model fixie-ai/ultravox-v0_5-llama-3_2-1b

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7565.1 | 7329.1 | 16000.0 | 16000.0 | 129722.1 | 141848.5 | 6.4  | 16.8  | 75.3 | 72.9 | 52172.0 | 135692.0 | 610195.0 | 592749.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

ℹ GuideLLM Request Metrics Statistics (Completed Requests)
|=============|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|
| Benchmark   | Request Latency (ms)          ||||| Output Tokens / Sec          ||||| Time to First Token (ms)      ||||| Time per Output Token (ms)    |||||
| Strategy    | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 |
|-------------|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|
| synchronous | 125.4 | 130.2 | 125.4 | 145.1 | 150.2| 160.5| 45.2  | 44.8  | 45.2  | 42.1  | 41.5 | 40.2| 25.1  | 26.5  | 25.1  | 30.2  | 32.5 | 35.1| 22.1  | 22.3  | 22.1  | 23.7  | 24.1 | 24.8|
|=============|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```
