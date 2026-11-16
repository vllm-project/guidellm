---
weight: 10
source_sha256: !!str "50d59f77afa69e1b01c6763c20811557d3562270ef797342e03923a5f4311a40"
---

# 多模态基准测试 {#multimodal-benchmarking}

GuideLLM 为多模态模型的基准测试提供完善支持，可以评估视觉、音频和视频任务的性能。本节介绍如何使用兼容 OpenAI 的端点（例如 vLLM 提供的端点），为不同模态设置和运行基准测试。

## 前提条件 {#prerequisites}

要运行多模态基准测试，必须安装带有所需可选依赖的 GuideLLM：

```bash
# For all multimodal features
pip install guidellm[vision,audio]

# For specific modalities
pip install guidellm[vision]  # Images and Video
pip install guidellm[audio]   # Audio
```

请确保已有正在运行的推理服务器和模型，它们兼容 OpenAI API，并支持你要测试的模态。各模态的基准测试说明请参阅下方对应指南。

## 可用指南 {#available-guides}

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } 图像

  ______________________________________________________________________

  使用 Chat Completions API 和图像输入对视觉语言模型（VLM）进行基准测试，涵盖视觉问答和图像描述。

  [:octicons-arrow-right-24: 图像指南](image.md)

- :material-video:{ .lg .middle } 视频

  ______________________________________________________________________

  通过处理视频输入评估视频理解模型，涵盖视频请求格式和编码选项的配置。

  [:octicons-arrow-right-24: 视频指南](video.md)

- :material-microphone:{ .lg .middle } 音频

  ______________________________________________________________________

  对音频转录（ASR）、翻译和原生音频聊天模型进行基准测试。

  [:octicons-arrow-right-24: 音频指南](audio.md)

- :material-image-multiple-outline:{ .lg .middle } 合成视觉

  ______________________________________________________________________

  即时生成图像和短视频，无需自备数据集即可对视觉语言模型（VLM）的服务配置进行基准测试。涵盖 `--data` 的 `synthetic_image` 和 `synthetic_video` 类型。

  [:octicons-arrow-right-24: 合成视觉指南](synthetic_vision.md)

</div>
