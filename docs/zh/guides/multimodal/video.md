---
source_sha256: !!str "a5896cccd6d8e20aa2b550c2c4f39e2410839985285faa7aa6faea2fa92626ab"
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

最后，准备包含受支持视频文件的数据集以进行基准测试。GuideLLM 可以从 Hugging Face 数据集、本地文件、URL 等来源读取视频数据。以下示例使用 `lmms-lab/Video-MME` 数据集。

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

对于所有受支持的端点，请求格式化器都会对视频数据进行编码并设置请求负载格式。系统提供了合理的默认值，也可以按需自定义。以下视频请求格式选项通过 `--request-formatter-kwargs` 参数配置，并以 JSON 字符串形式传入。

#### `"encode_kwargs"`

传递给视频编码器的参数字典，用于控制将视频数据加入请求前的预处理方式。

**关于嵌套结构的说明：**

- 对于 **Chat Completions**（`chat_completions`），视频参数必须嵌套在 `encode_kwargs` 中的 `video` 键下。

支持的参数包括：

- `"encode_type"`：指定视频在请求中的传递方式。
  - `"base64"`（默认值）：下载视频（如果来源是 URL）或读取视频，然后将其编码为 Base64 字符串。
  - `"url"`：在请求中直接发送视频 URL。推理服务器必须能够访问该视频 URL。

**示例：**

对于 **Chat Completions**，直接发送视频 URL：

```bash
--request-formatter-kwargs '{"encode_kwargs": {"video": {"encode_type": "url"}}}'
```

#### `"extras"`

可直接加入请求的额外参数字典。在 `extras` 中，可以指定额外参数的放置位置：

- `"headers"`：加入请求头
- `"params"` / `"body"`：加入请求参数或请求体

例如，指定系统提示词或其他请求体参数：

```bash
--request-formatter-kwargs '{"extras": {"body": {"temperature": 0.7}}}'
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

此基准测试用于评估视频语言模型回答视频相关问题的能力。我们使用 `lmms-lab/Video-MME` 数据集，其中包含视频和文本问题。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data '{"kind":"huggingface","source":"lmms-lab/Video-MME","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"video_column":"url","text_column":"question"}}'
```

**关键参数**

- `--backend`：服务器 URL、模型和聊天补全端点
- `--data`：要加载的数据集——使用 `kind=huggingface` 和数据集标识符，并通过 `load_kwargs` 配置数据集加载方式（选择 `"test"` 划分）。所有有效选项请参阅 [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset)
- `--data-column-mapper`：将包含视频链接的数据集 `url` 列映射到 `video_column`，并将 `question` 映射到 `text_column`

### 2. 视频描述

此基准测试用于评估模型在没有具体问题（或使用通用提示词）的情况下描述视频的能力。此处不映射文本列，只发送视频。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data '{"kind":"huggingface","source":"lmms-lab/Video-MME","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"video_column":"url"}}'
```

**关键参数：**

- `--data-column-mapper`：只映射 `video_column`，表示发送仅包含视频的请求（或让模型生成视频标题/描述的请求）。
