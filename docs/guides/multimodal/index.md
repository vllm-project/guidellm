---
weight: 10
---

# Multimodal Benchmarking

GuideLLM provides robust support for benchmarking multimodal models, allowing evaluation of performance across vision, audio, and video tasks. This section contains guides for setting up and running benchmarks for different modalities using OpenAI-compatible endpoints, such as those provided by vLLM.

## Prerequisites

To run multimodal benchmarks, you must install GuideLLM with the appropriate extras:

```bash
# For all multimodal features
pip install guidellm[vision,audio]

# For specific modalities
pip install guidellm[vision]  # Images and Video
pip install guidellm[audio]   # Audio
```

Ensure you have a running inference server and model compatible with the OpenAI API that supports the specific modality you intend to test. Refer to the individual guides below for instructions on benchmarking each modality.

## Available Guides

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } Images

  ______________________________________________________________________

  Benchmark Vision-Language Models (VLMs) with image inputs using the Chat Completions API. Covers visual question answering and image captioning.

  [:octicons-arrow-right-24: Image Guide](image.md)

- :material-video:{ .lg .middle } Video

  ______________________________________________________________________

  Evaluate video understanding models by processing video inputs. Includes configuration for frame sampling and video encoding.

  [:octicons-arrow-right-24: Video Guide](video.md)

- :material-microphone:{ .lg .middle } Audio

  ______________________________________________________________________

  Benchmark audio transcription (ASR), translation, and audio-native chat models.

  [:octicons-arrow-right-24: Audio Guide](audio.md)

</div>
