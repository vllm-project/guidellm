---
weight: 40
source_sha256: !!str "485e84d06ac83f55deb318e0fe735d74013ab78ce89bcf2b8bce41e45ab8a8c6"
---

# 합성 시각 데이터

GuideLLM은 이미지와 짧은 비디오를 실시간으로 생성하므로, 별도의 데이터셋을 준비하지 않고도 비전 언어 모델(VLM)의 서빙 설정을 벤치마킹할 수 있습니다. `--data`의 두 유형인 `synthetic_image`와 `synthetic_video`를 `synthetic_text`와 조합하면 하나의 명령어로 텍스트와 시각 데이터를 갖춘 멀티모달 요청을 생성할 수 있습니다.

합성 시각 데이터는 이미지 크기, 프레임 수, 초당 프레임 수 등 페이로드의 구성을 정밀하게 제어하거나, 전처리기 캐시를 사용하면 부하가 걸리지 않는 처리 경로를 스트레스 테스트할 때 유용합니다. 기본 설정에서는 요청마다 바이트 내용이 다른 페이로드를 생성합니다. 따라서 vLLM의 멀티모달 전처리기 캐시를 사용하지 않으면서도 실제 미디어와 비슷한 압축률로 데이터를 전송할 수 있습니다.

## 사전 요구 사항

이미지와 비디오 생성을 활성화하려면 `vision` 선택적 의존성과 함께 GuideLLM을 설치하세요:

```bash
pip install guidellm[vision]
```

## 합성 이미지

`--data "kind=synthetic_image"`를 사용하면 요청마다 이미지 한 장을 생성합니다. 같은 요청에 합성 프롬프트 텍스트를 포함하려면 `synthetic_text` 데이터 소스를 별도로 추가하세요.

### 명령어 예제

720p 이미지 한 장과 함께 텍스트 토큰 200개와 출력 토큰 64개를 지정하는 예제입니다:

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=200" \
  --data "kind=synthetic_image,resolution=720p,output_tokens=64"
```

요청마다 1280×720 JPEG 이미지 두 장을 생성하는 예제입니다:

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=200" \
  --data "kind=synthetic_image,width=1280,height=720,format=jpeg,images_per_request=2,output_tokens=64"
```

### 설정 옵션

- `width`: 생성할 이미지의 너비(픽셀)입니다.
- `height`: 생성할 이미지의 높이(픽셀)입니다.
- `resolution`: `height`를 `480p`, `720p`, `1080p` 등 정해진 이름으로 지정하는 단축 표기입니다. `aspect_ratio`와 함께 사용하여 `width`를 계산합니다.
- `aspect_ratio`: `width`, `height`, `resolution` 중 하나만 지정했을 때 나머지 치수를 계산하는 데 사용하는 가로세로 비율입니다. `16:9` 또는 `4:3`와 같이 지정합니다.
- `format`: 이미지 인코딩 형식입니다. `jpeg`(기본값) 또는 `png`를 사용합니다.
- `jpeg_quality`: `format=jpeg`일 때의 JPEG 품질 계수(1~100)입니다. 기본값은 85입니다.
- `content`: 데이터셋의 각 행에서 생성할 이미지의 내용입니다. `gradient`(기본값)는 행별 시드로 그라데이션을 생성하여 실제 사진과 비슷한 압축률을 얻습니다. `noise`는 균등 분포에서 무작위로 픽셀을 생성하여 전송 크기가 가장 큰 경우를 재현합니다. `solid`와 `checkerboard`는 전처리기 캐시가 성능에 미치는 영향을 비교할 때 유용합니다.
- `images_per_request`: 각 요청에 첨부할 이미지 수입니다. 기본값은 1입니다.
- `output_tokens`: 모델이 생성할 평균 토큰 수입니다. 합성 텍스트와 동일하게 `stdev`, `min`, `max` 접미사를 사용할 수 있습니다.
- `seed`: 실행 간에 동일한 데이터를 생성하기 위한 난수 시드입니다.

## 합성 비디오

`--data "kind=synthetic_video"`를 사용하면 요청마다 짧은 비디오 클립을 생성합니다. 같은 요청에 합성 프롬프트 텍스트를 포함하려면 `synthetic_text` 데이터 소스를 별도로 추가하세요. 출력 형식은 `mp4`(h264, yuv420p)입니다.

### 명령어 예제

프롬프트와 출력 토큰 수를 작게 설정하고 480p 프레임 6개로 구성된 클립을 초당 1프레임으로 생성하는 예제입니다:

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=64" \
  --data "kind=synthetic_video,width=854,height=480,frames=6,fps=1,output_tokens=128"
```

h264의 목표 비트 전송률을 지정하고 720p 프레임 12개로 구성된 클립을 초당 3프레임으로 생성하는 예제입니다:

```bash
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=64" \
  --data "kind=synthetic_video,width=1280,height=720,frames=12,fps=3,video_bitrate=2M,output_tokens=128"
```

### 설정 옵션

- `width`: 생성할 비디오의 너비(픽셀)입니다.
- `height`: 생성할 비디오의 높이(픽셀)입니다. 합성 이미지와 동일하게 `resolution` 및 `aspect_ratio` 단축 표기를 사용할 수 있습니다.
- `frames`: 클립의 프레임 수입니다.
- `fps`: 초당 프레임 수입니다. 이 값과 `frames`에 따라 클립의 재생 시간이 결정됩니다.
- `video_bitrate`: h264의 목표 비트 전송률(예: `1M`, `500k`)입니다. 선택 사항이며, 실행마다 일정한 크기의 데이터를 전송하도록 설정할 때 유용합니다.
- `content`: 데이터셋의 각 행에서 생성할 클립의 내용입니다. `gradient`(기본값)는 시드에 따라 그라데이션을 생성하고 좌표를 변형하여 각 클립이 실제 비디오와 비슷한 압축률을 갖도록 합니다. `noise`는 균등 분포에서 무작위로 픽셀을 생성하여 전송 크기가 가장 큰 경우를 재현합니다.
- `output_tokens`: 모델이 생성할 평균 토큰 수입니다. 동일하게 `stdev`, `min`, `max` 접미사를 사용할 수 있습니다.
- `seed`: 실행 간에 동일한 데이터를 생성하기 위한 난수 시드입니다.

## 참고 사항

- `synthetic_text`를 포함하면 토크나이저가 필요합니다. 기본적으로 직접 지정하거나 서버에서 가져온 모델을 사용합니다. 다른 토크나이저를 사용하려면 `--tokenizer`로 지정하세요.
- 행별 시드로 그라데이션을 생성하면 요청마다 페이로드의 바이트 내용이 달라져 vLLM의 멀티모달 전처리기 캐시를 우회합니다. 의도적으로 캐시 적중을 유도하려면 이미지에 `content=solid`를 사용하거나 `seed`와 `--data-loader "kind=pytorch,samples=..."` 값을 모두 고정하는 등 페이로드 설정을 고정하세요.
- 같은 시드에서 생성되는 mp4 파일의 실제 바이트 내용은 설치된 `ffmpeg` 및 `PIL` 버전에 따라 달라집니다. 출력 토큰 수와 요청 구조는 버전이 달라도 일정하지만, 여러 장비에서 바이트 단위의 출력이나 전송 크기를 비교하면 약간의 차이가 있을 수 있습니다.
