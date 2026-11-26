---
weight: 10
source_sha256: !!str "7a4a3bfe208797eeac8b5faa6e02984358945ee0cf244715320d956f8f31536c"
---

# 멀티모달 벤치마킹 {#multimodal-benchmarking}

GuideLLM을 사용하면 시각, 오디오 및 비디오 작업에서 멀티모달 모델의 성능을 평가할 수 있습니다. 이 섹션에서는 vLLM과 같은 서버가 제공하는 OpenAI 호환 엔드포인트를 사용하여 입력 유형별로 벤치마크를 설정하고 실행하는 방법을 설명합니다.

## 사전 요구 사항 {#prerequisites}

멀티모달 벤치마크를 실행하려면 사용할 입력 유형에 필요한 선택적 의존성과 함께 GuideLLM을 설치하세요:

```bash
# For all multimodal features
pip install guidellm[vision,audio]

# For specific modalities
pip install guidellm[vision]  # Images and Video
pip install guidellm[audio]   # Audio
```

테스트할 입력 유형을 지원하는 모델이 OpenAI 호환 추론 서버에서 실행 중인지 확인하세요. 입력 유형별 벤치마크 실행 방법은 아래 가이드를 참고하세요.

## 입력 유형별 가이드 {#available-guides}

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } 이미지

  ______________________________________________________________________

  Chat Completions API로 이미지를 입력하여 비전 언어 모델(VLM)을 벤치마킹합니다. 이미지에 관한 질문에 답하거나 이미지 설명을 생성하는 작업을 다룹니다.

  [:octicons-arrow-right-24: 이미지 가이드](image.md)

- :material-video:{ .lg .middle } 비디오

  ______________________________________________________________________

  비디오를 입력하여 모델의 비디오 이해 성능을 평가합니다. 비디오 요청 형식과 인코딩 옵션을 설정하는 방법을 설명합니다.

  [:octicons-arrow-right-24: 비디오 가이드](video.md)

- :material-microphone:{ .lg .middle } 오디오

  ______________________________________________________________________

  음성 인식(ASR), 음성 번역 및 오디오를 직접 처리하는 채팅 모델을 벤치마킹합니다.

  [:octicons-arrow-right-24: 오디오 가이드](audio.md)

- :material-image-multiple-outline:{ .lg .middle } 합성 시각 데이터

  ______________________________________________________________________

  별도의 데이터셋을 준비하지 않고도 이미지와 짧은 비디오를 실시간으로 생성하여 비전 언어 모델(VLM)의 서빙 설정을 벤치마킹할 수 있습니다. `--data`에서 사용할 수 있는 `synthetic_image` 및 `synthetic_video` 유형을 설명합니다.

  [:octicons-arrow-right-24: 합성 시각 데이터 가이드](synthetic_vision.md)

</div>
