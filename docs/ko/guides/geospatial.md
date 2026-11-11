---
weight: 0
source_sha256: !!str "bee11357a8afa610c922cf8dc8bb77564135591d7b4a2941ac5bf69558591ebf"
---

# 지리 공간 모델 벤치마킹

GuideLLM은 위성 영상과 원격 탐사 영상을 처리하는 지리 공간 모델의 벤치마킹을 지원합니다. 이 가이드에서는 vLLM의 `/pooling` 엔드포인트를 사용하여 IBM-NASA의 Prithvi와 같은 [TerraTorch](https://terrastackai.github.io/terratorch/stable/guide/vllm/intro/) 지리 공간 모델을 벤치마킹하는 방법을 설명합니다.

## 개요

GuideLLM에서 지리 공간 모델을 벤치마킹할 때는 다음과 같은 특성이 있습니다:

- URL 또는 Base64 형식으로 위성 영상 입력
- 후속 작업에 사용할 임베딩 또는 풀링된 표현 반환
- 텍스트 토큰 대신 이미지 관련 지표 추적
- 다중 밴드 영상 지원(예: Sentinel-1, Sentinel-2)

## 지원되는 모델

이 가이드의 예제는 다음 지리 공간 모델로 테스트했습니다:

- **IBM-NASA Prithvi-EO-2.0-300M-TL-Sen1Floods11**: Sentinel-1 데이터로 학습한 홍수 탐지 모델

## 설정

### 기본 사용법

지리 공간 모델을 벤치마킹하려면 백엔드 설정에 `request_format=/pooling`을 지정하세요:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=10 \
  --output kind=json,path=results.json
```

## 데이터셋 형식

vLLM으로 실행하는 지리 공간 모델은 데이터셋 파일에서 다음과 같은 중첩 JSON 구조를 사용합니다:

### JSONL 형식

```json
{
  "prompt": {
    "data": {
      "data": "https://example.com/image.tif",
      "data_format": "url",
      "out_data_format": "b64_json",
      "indices": [1, 2, 3, 8, 11, 12]
    },
    "priority": 0
  }
}
```

### 필드 설명

- **prompt**: 요청 데이터를 담는 최상위 객체
  - **data**: 이미지 정보를 담는 중첩 객체
    - **data**: 이미지의 URL 또는 Base64로 인코딩한 이미지 데이터(예: GeoTIFF 파일)
    - **data_format**: 입력 데이터 형식(`"url"` 또는 `"base64"`)
    - **out_data_format**: 원하는 출력 형식(예: `"b64_json"`)
    - **indices**: 처리할 밴드 또는 채널의 인덱스 배열(위성 센서에 따라 다름)
  - **priority**: 요청 스케줄링 우선순위(선택 사항, 기본값: 0)

### 밴드 인덱스

사용할 밴드는 모델에 따라 다릅니다. [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)는 침수 지도를 작성하기 위해 청색, 녹색, 적색, 협대역 근적외선(Narrow NIR), 단파 적외선(SWIR), 단파 적외선 2(SWIR 2)의 여섯 밴드로 학습했습니다. 추론 시 올바른 데이터를 입력하려면 GeoTIFF에 포함된 밴드 중 vLLM이 사용할 밴드를 지정해야 합니다. GeoTIFF 전처리는 [terratorch_segmentation](https://terrastackai.github.io/terratorch/stable/guide/vllm/vllm_io_plugins/) 플러그인이 담당합니다. 모델에 맞는 밴드 설정은 해당 모델의 문서를 참고하세요.

## 전체 예제: Prithvi로 홍수 탐지

다음은 Prithvi 홍수 탐지 모델을 벤치마킹하는 전체 예제입니다:

### 1. 데이터셋 준비

`flood_detection_dataset.jsonl` 파일을 만들고 `data`가 가리키는 GeoTIFF URL을 네트워크에서 접근할 수 있는 주소로 변경하세요. 예제는 [Hugging Face의 Prithvi 저장소](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main/examples)에서 확인할 수 있습니다.

다음 명령어는 벤치마킹을 위해 동일한 항목 100개로 구성된 예제 데이터셋을 생성합니다:

```bash
printf '{"prompt":{"data": {"data": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif","data_format": "url","out_data_format": "b64_json","indices": [1, 2, 3, 8, 11, 12]},"priority": 0}}\n%.0s' {1..100} > flood_detection_dataset.jsonl
```

### 2. vLLM 서버 시작

```bash
vllm serve \
  ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --enforce-eager \
  --skip-tokenizer-init \
  --enable-mm-embeds \
  --io-processor-plugin \
  terratorch_segmentation
```

vLLM에서 TerraTorch 모델을 실행하는 방법은 [TerraTorch 문서](https://terrastackai.github.io/terratorch/stable/guide/vllm/intro/)를 참고하세요.

### 3. 벤치마크 실행

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=100 \
  --output kind=json,path=results.json
```

## vLLM Bench와 비교

GuideLLM의 지리 공간 모델 벤치마크는 다음과 같이 vLLM의 벤치마크 도구와 호환됩니다:

### vLLM Bench 명령어

```bash
vllm bench serve \
  --base-url http://localhost:8000 \
  --dataset-name=custom \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --skip-tokenizer-init \
  --endpoint /pooling \
  --backend vllm-pooling \
  --percentile-metrics e2el \
  --metric-percentiles 25,75,99 \
  --num-prompts 10 \
  --dataset-path flood_detection_dataset.jsonl
```

### 동일한 작업을 수행하는 GuideLLM 명령어

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=10 \
  --output kind=json,path=results.json
```

## 관련 문서

- [데이터셋 가이드](datasets.md) - 데이터셋 준비 및 형식
- [지표 가이드](metrics.md) - 벤치마크 지표 이해하기
- [멀티모달 가이드](multimodal/index.md) - 멀티모달 데이터 사용하기
