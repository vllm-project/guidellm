---
source_sha256: !!str "50678255554905d3f85e09e3eff0ddf029a87ed6950d8594c81520d2b0fbc04a"
---

# 视频基准测试

本指南介绍如何对视频语言模型（Video-LLM）进行基准测试，涵盖视频问答（Video QA，`chat/completions`）和视频描述（`chat/completions`）等任务。

## 环境准备

首先，确保已有正在运行的推理服务器和模型，并且它们兼容 OpenAI Chat API。GuideLLM 支持任何兼容 OpenAI API、且能通过 `chat/completions` 端点处理视频输入的服务器。以下示例使用 vLLM 部署 Qwen3-VL 模型。

```bash
# Qwen3-VL Video QA/Captioning
vllm serve Qwen/Qwen3-VL-2B-Instruct
```

接下来，在同一台机器或另一台能够访问服务器的机器上（推荐）安装带视觉支持的 GuideLLM：

```bash
pip install guidellm[vision,recommended]
```

最后，准备包含受支持视频文件的数据集以进行基准测试。GuideLLM 可以从 Hugging Face 数据集、本地文件、URL 等来源读取视频数据。以下示例使用 IMAVB 数据集。

### 准备 IMAVB 提示和 URL

IMAVB 的 `video` 列包含仓库相对路径。以下脚本将其转换为 MP4 直链，并将每个问题与六个选项合并。准备时只下载标注数据并写入 `imavb.jsonl`，无需下载或解压视频。

```python
import json

from datasets import load_dataset
from huggingface_hub import hf_hub_url

repo_id = "ngqtrung/IMAVB"
annotations = load_dataset(repo_id, split="standard_vision")
with open("imavb.jsonl", "w", encoding="utf-8") as output:
    for row in annotations:
        options = "\n".join(
            f"{letter.upper()}. {row[f'option_{letter}']}" for letter in "abcdef"
        )
        record = {
            "video": hf_hub_url(repo_id, row["video"], repo_type="dataset"),
            "prompt": f"{row['question']}\n{options}",
            "caption": "Describe the events in this video.",
            "output_tokens_count": 128,
        }
        output.write(json.dumps(record) + "\n")
```

下列示例使用 `encode_type=url`，由推理服务器直接从 Hugging Face 获取 MP4；服务器必须支持远程视频 URL 并能访问这些地址。默认的 `encode_type=base64` 则由 GuideLLM 下载视频并将字节嵌入请求。

## 处理选项

视频任务同样支持所有标准基准测试参数，例如 `--profile`、配置文件的速率参数以及 `--constraint kind=max_requests,count=<n>` 等。此外，还有一些选项可用于控制视频专属的数据处理和请求格式。

### 加载数据

GuideLLM 支持多种视频数据加载方式。首先，数据源整体必须能够由 GuideLLM 反序列化为 Hugging Face 数据集，包括本地文件、Hugging Face 数据集、JSON 文件等。

接下来，可反序列化数据源中的目标视频列必须受 GuideLLM 视频数据解码器/编码器支持。支持的格式包括：

- Hugging Face Video 特征
- 本地文件路径
- 指向视频文件的 URL
- Base64 编码的视频数据
- 原始视频字节

### 数据列映射

指定数据集时，通常需要将具体的视频列映射到 GuideLLM 预期的 `video_column`，以便 GuideLLM 知道要将哪一列作为视频处理。如果未指定，GuideLLM 会根据 `video`、`clip` 等常用名称尝试自动检测视频列。

要指定映射，请使用 `--data-column-mapper` 参数并传入 JSON 字符串，指定现有列名对应的 `video_column`。例如，如果数据集中的视频列名为 `url`，可以这样设置：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"video_column": "url"}}'
```

如果要组合多个数据集（例如分别提供提示词和视频），请在列名前加上数据集索引（从 0 开始），或者数据集别名和点号。例如，视频列位于第二个数据集（索引为 1）时：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.video_column": "url"}}'
```

### 请求格式

`encode_media` 数据预处理器会在后端构造请求前编码视频。使用 `--data-preprocessor` 配置预处理器，并将编码选项放在 `video_kwargs` 中。

下面的单视频示例使用 IMAVB 托管的 MP4 直链，无需执行上述准备步骤：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=1 \
  --data '{"kind":"in_memory_dict_list","data":[{"video":"https://huggingface.co/datasets/ngqtrung/IMAVB/resolve/main/videos/--aqjaJyZLk.mp4","prompt":"Describe the events in this video.","output_tokens_count":128}]}' \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

#### `video_kwargs`

支持的参数包括：

- `"encode_type"`：指定视频在请求中的传递方式。
  - `"base64"`（默认值）：下载视频（如果来源是 URL）或读取视频，然后将其编码为 Base64 字符串。
  - `"url"`：在请求中直接发送视频 URL。推理服务器必须能够访问该视频 URL。

直接发送视频 URL：

```bash
--data-preprocessor '{"kind":"encode_media","video_kwargs":{"encode_type":"url"}}'
```

#### `extras`

在现有的 `--backend` 配置中加入 `extras`，以传递请求选项：

- `body`：请求体中的字段
- `headers`：HTTP 请求头
- `params`：URL 查询参数

设置采样温度：

```bash
--backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"temperature":0.7}}}'
```

#### `"stream"`

如果后端支持，可以使用布尔值开启或关闭流式响应。默认启用流式传输。在后端配置中传入 `stream=false`：

```bash
--backend kind=openai_http,target=http://localhost:8000,stream=false
```

## 预期结果

GuideLLM 会捕获整个请求生命周期的全面指标，并将其存储在 `GenerativeRequestStats` 中，再汇总到 `GenerativeMetrics`。结果会显示在控制台中，也会保存到本地文件供后续分析。

### 输出文件

- **`benchmarks.json`**：完整的分层统计对象，包含调度器耗时、请求分布，以及文本和视频的详细指标摘要。
- **`benchmarks.csv`**：逐请求导出 `GenerativeRequestStats`，适用于分析单个请求的性能、延迟以及具体输入/输出配置。
- **`benchmarks.html`**：汇总性能数据的可视化报告。

### 采集的指标

除延迟、首 token 延迟（TTFT）和 token 间延迟（ITL）等标准性能指标外，视频基准测试还会跟踪输入和输出中的特定用量指标：

- **视频帧数**：检测/处理的帧数。
- **视频时长**：视频内容的时长。
- **视频字节数**：视频负载的大小。
- **视频 token 数**：如果模型/服务器提供此数据，则表示处理的视频 token 数量。
- **文本指标**：提示词和回复的标准 token、词数和字符数统计。

### 统计分析

对于以上每项指标，GuideLLM 都会计算以下统计分布：

- **数值**：平均值、中位数、P95、P99、最小值、最大值。
- **速率**：每秒吞吐量（例如 `video_frames/sec`）。
- **并发数**：衡量同时处于活动状态的处理任务数。

这些指标使用 `StatusDistributionSummary` 结构，分别跟踪成功、未完成和出错的请求。

## 示例

### 1. 视频问答（Video QA）

此基准测试用于衡量模型回答 IMAVB 视频问题时的性能，使用上一步准备的 `imavb.jsonl`。GuideLLM 衡量推理性能，不会根据数据集的答案评估回答。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data kind=json_file,path=imavb.jsonl \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

**关键参数**

- `--backend`：服务器 URL、模型和聊天补全端点
- `--data`：加载准备好的 JSONL，其中包含 MP4 直链、问题和选项，以及 128 token 的输出上限
- 列映射自动完成：`video` 映射到 `video_column`，`prompt` 映射到 `text_column`，`output_tokens_count` 映射到 `output_tokens_count_column`
- `--data-preprocessor`：将 MP4 URL 传给推理服务器

### 2. 视频描述

此基准测试使用相同的 `imavb.jsonl` 文件，选择通用的 `caption` 提示词，而不是问题和选项。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data kind=json_file,path=imavb.jsonl \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"video_column":"video","text_column":"caption","output_tokens_count_column":"output_tokens_count"}}' \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

**关键参数：**

- `--data-column-mapper`：从每行选择视频 URL、通用描述提示词和输出上限
- `--data-preprocessor`：将 MP4 URL 传给推理服务器
