---
source_sha256: !!str "b37ba8240d04adbd03a0f95750aca4f71f560aac7e17db9fe505f4f78433292b"
---

# 图像基准测试

本指南介绍如何对视觉语言模型（VLM）进行基准测试，涵盖视觉问答（VQA，`chat/completions`）和图像描述（`chat/completions`）等任务。

## 环境准备

首先，确保已有正在运行的推理服务器和模型，并且它们兼容 OpenAI Chat API。GuideLLM 支持任何兼容 OpenAI API、且能通过 `chat/completions` 端点处理图像输入的服务器。以下示例使用 vLLM 部署 Qwen3-VL 模型。

```bash
# Qwen3-VL VQA/Captioning
vllm serve Qwen/Qwen3-VL-2B-Instruct
```

接下来，在同一台机器或另一台能够访问服务器的机器上（推荐）安装带视觉支持的 GuideLLM：

```bash
pip install guidellm[vision,recommended]
```

最后，准备包含受支持图像文件的数据集以进行基准测试。GuideLLM 可以从 Hugging Face 数据集、本地文件、URL 等来源读取图像数据。以下示例使用 `lmms-lab/MMBench_EN` 数据集。

## 处理选项

图像任务同样支持所有标准基准测试参数，例如 `--profile`、配置文件的速率参数以及 `--constraint kind=max_requests,count=<n>` 等。此外，还有一些选项可用于控制图像专属的数据处理和请求格式。

### 加载数据

GuideLLM 支持多种图像数据加载方式。首先，数据源整体必须能够由 GuideLLM 反序列化为 Hugging Face 数据集，包括本地文件、Hugging Face 数据集、JSON 文件等。

接下来，可反序列化数据源中的目标图像列必须受 GuideLLM 图像数据解码器/编码器支持。支持的格式包括：

- Hugging Face Image 特征（首选）
- 本地文件路径（例如 `.jpg`、`.png`）
- 指向图像文件的 URL
- Base64 编码的图像数据
- NumPy 数组或 PIL Image 对象

### 数据列映射

指定数据集时，通常需要将具体的图像列映射到 GuideLLM 预期的 `image_column`，以便 GuideLLM 知道要将哪一列作为图像处理。如果未指定，GuideLLM 会根据 `image`、`picture` 等常用名称尝试自动检测图像列。

要指定映射，请使用 `--data-column-mapper` 参数并传入 JSON 字符串，指定现有列名对应的 `image_column`。例如，如果数据集中的图像列名为 `photo`，可以这样设置：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"image_column": "photo"}}'
```

如果要组合多个数据集（例如分别提供提示词和图像），请在列名前加上数据集索引（从 0 开始），或者数据集别名和点号。例如，图像列位于第二个数据集（索引为 1）时：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.image_column": "photo"}}'
```

### 请求格式

对于所有受支持的端点，请求格式化器都会对图像数据进行编码并设置请求负载格式。系统提供了合理的默认值，也可以按需自定义。以下图像请求格式选项通过 `--request-formatter-kwargs` 参数配置，并以 JSON 字符串形式传入。

#### `"encode_kwargs"`

传递给图像编码器的参数字典，用于控制将图像数据加入请求前的预处理方式。

**关于嵌套结构的说明：**

- 对于 **Chat Completions**（`chat_completions`），图像参数必须嵌套在 `encode_kwargs` 中的 `image` 键下。

支持的参数包括：

- `"max_size"`：图像最长边的最大尺寸（像素）。可用于缩小大图，同时保持宽高比。
- `"max_width"`：图像的最大宽度（像素）。除非同时指定高度，否则保持宽高比。
- `"max_height"`：图像的最大高度（像素）。除非同时指定宽度，否则保持宽高比。
- `"width"`：强制将图像缩放到指定宽度（像素）。
- `"height"`：强制将图像缩放到指定高度（像素）。
- `"encode_type"`：指定图像在请求中的传递方式。
  - `"base64"`（默认值）：下载图像（如果来源是 URL）或读取图像，进行处理（缩放），然后将其编码为 Base64 字符串。
  - `"url"`：在请求中直接发送图像 URL。推理服务器必须能够访问该图像 URL。此类型不能使用缩放选项。

**示例：**

对于 **Chat Completions**，将图像缩放到最长边不超过 512 像素：

```bash
--request-formatter-kwargs '{"encode_kwargs": {"image": {"max_size": 512}}}'
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

- **`benchmarks.json`**：完整的分层统计对象，包含调度器耗时、请求分布，以及文本和图像的详细指标摘要。
- **`benchmarks.csv`**：逐请求导出 `GenerativeRequestStats`，适用于分析单个请求的性能、延迟以及具体输入/输出配置。
- **`benchmarks.html`**：汇总性能数据的可视化报告。

### 采集的指标

除延迟、首 token 延迟（TTFT）和 token 间延迟（ITL）等标准性能指标外，图像基准测试还会跟踪输入和输出中的特定用量指标：

- **图像数量**：处理的图像数。
- **图像像素数**：像素总数。
- **图像字节数**：图像负载的大小（字节）。
- **图像 token 数**：用于表示图像的 token 数量（如果可用）。
- **文本指标**：提示词和回复的标准 token、词数和字符数统计。

### 统计分析

对于以上每项指标，GuideLLM 都会计算以下统计分布：

- **数值**：平均值、中位数、P95、P99、最小值、最大值。
- **速率**：每秒吞吐量（例如 `image_pixels/sec`）。
- **并发数**：衡量同时处于活动状态的处理任务数。

这些指标使用 `StatusDistributionSummary` 结构，分别跟踪成功、未完成和出错的请求。

## 示例

### 1. 视觉问答（VQA）

此基准测试用于评估视觉语言模型回答图像相关问题的能力。我们使用 `lmms-lab/MMBench_EN` 数据集，其中包含图像和文本问题。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://192.168.4.12:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"lmms-lab/MMBench_EN","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"image_column":"image","text_column":"question"}}'
```

**关键参数**

- `--backend`：服务器 URL、模型和聊天补全端点
- `--data`：要加载的数据集——使用 `kind=huggingface` 和数据集标识符，并通过 `load_kwargs` 配置数据集加载方式（选择 `"test"` 划分）。所有有效选项请参阅 [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset)
- `--data-column-mapper`：将数据集的 `image` 列映射到 `image_column`，并将 `question` 映射到 `text_column`

上述命令使用 MMBench_EN 数据集中的提示词文本和图像，在目标服务器上对 `chat/completions` 端点进行基准测试。输出示例如下：

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```

### 2. 图像描述

此基准测试用于评估模型在没有具体问题（或使用通用提示词）的情况下描述图像的能力。此处不映射文本列，只发送图像。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"lmms-lab/MMBench_EN","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"image_column":"image"}}'
```

**关键参数：**

- `--data-column-mapper`：只映射 `image_column`，表示发送仅包含图像的请求（或让模型生成图像标题/描述的请求）。

上述命令使用 MMBench_EN 数据集中的提示图像，在目标服务器上对 `chat/completions` 端点进行基准测试。输出示例如下：

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```
