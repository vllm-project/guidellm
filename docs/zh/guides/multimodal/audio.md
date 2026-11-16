---
source_sha256: !!str "c382ca56157aa6282b1d058da8d03719753ac3341355d86e8cbea4b6d5bde496"
---

# 音频基准测试

本指南介绍如何对音频模型进行基准测试，涵盖自动语音识别（ASR，`audio/transcriptions`）、翻译（`audio/translations`）和音频聊天（`chat/completions`）等任务。

## 环境准备

首先，确保已有正在运行的推理服务器和模型，并且它们兼容所需的音频 API。GuideLLM 支持任何兼容 OpenAI API、且能通过 `chat/completions`、`audio/transcriptions` 或 `audio/translations` 端点处理音频输入的服务器。以下示例使用 vLLM 部署 Whisper 模型来执行转写和翻译任务，并使用 Ultravox 执行聊天任务。启动服务器的示例命令如下：

```bash
# Whisper ASR/Translation
vllm serve openai/whisper-small

# Ultravox Audio Chat
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

接下来，在同一台机器或另一台能够访问服务器的机器上（推荐）安装带音频支持的 GuideLLM：

```bash
pip install guidellm[audio,recommended]
```

最后，准备包含受支持音频文件的数据集以进行基准测试。GuideLLM 可以从 Hugging Face 数据集、本地文件、URL 等来源读取音频数据。以下示例使用 `openslr/librispeech_asr` 数据集。

## 处理选项

音频任务同样支持所有标准基准测试参数，例如 `--profile`、配置文件的速率参数以及 `--constraint kind=max_requests,count=<n>` 等。此外，还有一些选项可用于控制音频专属的数据处理和请求格式。

### 加载数据

GuideLLM 支持多种音频数据加载方式。首先，数据源整体必须能够由 GuideLLM 反序列化为 Hugging Face 数据集，包括本地文件、Hugging Face 数据集、JSON 文件等。

接下来，可反序列化数据源中的目标音频列必须受 GuideLLM 音频数据解码器/编码器支持。支持的格式包括：

- Hugging Face Audio 特征（首选）
- 本地文件路径（例如 `.wav`、`.mp3`、`.flac`）
- 指向音频文件的 URL
- Base64 编码的音频数据
- 包含原始音频样本的 NumPy 或 PyTorch 数组（单声道一维数组或通道优先数组）

原始浮点数组必须提供采样率，并会在编码时转换为 float32。例如，音频项可以写作 `{"data": samples, "sample_rate": 16000}`，其中单声道的 `samples` 形状为 `(num_samples,)`，多声道则为 `(num_channels, num_samples)`。

### 数据列映射

指定数据集时，通常需要将具体的音频列映射到 GuideLLM 预期的 `audio_column`，以便 GuideLLM 知道要将哪一列作为音频处理。如果未指定，GuideLLM 会根据 `audio`、`speech`、`wav` 等常用名称尝试自动检测音频列。

要指定映射，请使用 `--data-column-mapper` 参数并传入 JSON 字符串，指定现有列名对应的 `audio_column`。例如，如果数据集中的音频列名为 `speech_data`，可以这样设置：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"audio_column": "speech_data"}}'
```

如果要组合多个数据集（例如分别提供提示词和音频），请在列名前加上数据集索引（从 0 开始），或者数据集别名和点号。例如，音频列位于第二个数据集（索引为 1）时：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.audio_column": "speech_data"}}'
```

### 请求格式

对于所有受支持的音频端点，请求格式化器都会对音频数据进行编码并设置请求负载格式。系统提供了合理的默认值，也可以按需自定义。以下选项通过 `--request-formatter-kwargs` 参数配置，并以 JSON 字符串形式传入。

#### `"encode_kwargs"`

传递给音频编码器的参数字典，用于控制将音频数据加入请求前的预处理方式。

**关于嵌套结构的说明：**

- 对于 **Chat Completions**（`chat_completions`），音频参数必须嵌套在 `encode_kwargs` 中的 `audio` 键下。
- 对于 **Transcription/Translation**（`audio_transcriptions`、`audio_translations`），参数直接放在 `encode_kwargs` 顶层。

支持的参数包括：

- `"sample_rate"`：输入音频数据的采样率。仅在无法推断时才需要设置（例如原始 NumPy/PyTorch 数组）。
- `"encode_sample_rate"`：发送到 API 的音频目标采样率（默认值：16000 Hz）。
- `"audio_format"`：请求负载的文件格式。支持 `"wav"`、`"mp3"` 和 `"flac"`。如果未指定，会根据源音频编解码器自动检测格式。无法检测时（例如原始数组）默认为 `"wav"`。
- `"bitrate"`：有损格式（如 mp3）的比特率（默认值：`"64k"`）。
- `"max_duration"`：指定后，超过此时长（秒）的音频会被截断。
- `"mono"`：是否将音频转换为单声道（默认值：True）。
- `"file_name"`：可选的文件名，会包含在请求元数据中（对依赖文件扩展名的端点很有用）。默认值为 `"audio.wav"`。

**示例：**

对于 **Audio Transcription**（扁平结构），将音频转换为 16kHz WAV：

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio_format": "wav", "encode_sample_rate": 16000}}'
```

对于 **Audio Chat**（嵌套结构），将音频截断至 30 秒：

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio": {"max_duration": 30.0}}}'
```

#### `"extras"`

可直接加入请求的额外参数字典，可用于设置端点专属参数（例如 Whisper 模型的语言）。在 `extras` 中，可以指定额外参数的放置位置：

- `"headers"`：加入请求头
- `"params"` / `"body"`：加入请求参数或请求体（根据端点自动检测）

例如，为音频翻译请求指定法语作为目标语言：

```bash
--request-formatter-kwargs '{"extras": {"body": {"language": "fr"}}}'
```

#### `"stream"`

如果后端支持，可以使用布尔值开启或关闭流式响应。默认启用流式传输。在后端配置中传入 `stream=false`：

```bash
--backend kind=openai_http,target=http://localhost:8000,stream=false
```

## 预期结果

GuideLLM 会捕获整个请求生命周期的全面指标，并将其存储在 `GenerativeRequestStats` 中，再汇总到 `GenerativeMetrics`。结果会显示在控制台中，也会保存到本地文件供后续分析。

### 输出文件

- **`benchmarks.json`**：完整的分层统计对象，包含调度器耗时、请求分布，以及文本和音频的详细指标摘要。
- **`benchmarks.csv`**：逐请求导出 `GenerativeRequestStats`，适用于分析单个请求的性能、延迟以及具体输入/输出配置。
- **`benchmarks.html`**：汇总性能数据的可视化报告。

### 采集的指标

除延迟、首 token 延迟（TTFT）和 token 间延迟（ITL）等标准性能指标外，音频基准测试还会跟踪输入和输出中的特定用量指标：

- **音频 token 数**：处理的音频 token 数量（如果模型支持）。
- **音频样本数**：原始音频样本的数量。
- **音频时长**：音频内容的总时长（秒）。
- **音频字节数**：音频负载的大小（字节）。
- **文本指标**：转写文本或聊天回复的标准 token、词数和字符数统计。

### 统计分析

对于以上每项指标，GuideLLM 都会计算以下统计分布：

- **数值**：平均值、中位数、P95、P99、最小值、最大值。
- **速率**：每秒吞吐量（例如 `audio_seconds/sec`）。
- **并发数**：衡量同时处于活动状态的处理任务数。

这些指标使用 `StatusDistributionSummary` 结构，分别跟踪成功、未完成和出错的请求。

## 示例

### 1. 音频转写（ASR）

此基准测试用于评估 Whisper 等自动语音识别（ASR）模型将音频转换为文本的能力。请使用上文的 Whisper vLLM 启动命令，或使用支持音频转写端点的类似模型。本示例只使用 LibriSpeech 数据集中的音频数据；也可以像音频聊天示例那样提供提示词。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/audio/transcriptions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}'
```

**关键参数：**

- `--backend`：服务器 URL，以及用于 ASR 的 `request_format=/v1/audio/transcriptions`
- `--profile kind=synchronous`：按顺序运行请求
- `--constraint kind=max_requests,count=20`：将基准测试限制为总计 20 个请求
- `--data`：通过 `load_kwargs` 选择 `"clean"` 配置和 `"test"` 划分的 Hugging Face 数据集
- `--data-column-mapper`：将数据集的音频列映射到 GuideLLM 的 `audio_column`

上述命令使用 LibriSpeech 数据集中的音频，在目标服务器上对 `audio/transcriptions` 端点进行 ASR 基准测试。输出示例如下：

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

### 2. 音频翻译

此基准测试用于评估 Whisper 等音频翻译模型将一种语言的音频转换为另一种语言文本的能力。请使用上文的 Whisper vLLM 启动命令，或使用支持音频翻译端点的类似模型。本示例只使用 LibriSpeech 数据集中的音频；也可以像音频聊天示例那样提供提示词。

**命令：**

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","request_format":"/v1/audio/translations","extras":{"body":{"language":"fr"}}}' \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}'
```

**关键参数：**

- `--backend`：服务器 URL、翻译端点，以及通过 `extras.body` 设置的目标语言
- `--profile kind=synchronous`：按顺序执行
- `--constraint kind=max_requests,count=20`：将测试限制为 20 个请求
- `--data`：使用 `load_kwargs` 指定 `"clean"` 配置和 `"test"` 划分的 Hugging Face 数据集。所有有效选项请参阅 [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset)。
- `--data-column-mapper`：指定用于音频处理的音频列

上述命令使用 LibriSpeech 数据集中的音频，并请求将其翻译为法语，从而在目标服务器上对 `audio/translations` 端点进行基准测试。输出示例如下：

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

### 3. 音频聊天补全

此基准测试用于评估 Ultravox 等能够以对话形式处理音频输入的模型。请使用上文的 Ultravox vLLM 启动命令，或使用支持聊天补全流程中音频格式的类似模型。除 LibriSpeech 数据集外，以下示例还添加了一个用于文本提示词的合成数据集。请根据使用场景替换数据集和列映射。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=20 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --data '{"kind":"huggingface","source":"openslr/librispeech_asr","load_kwargs":{"name":"clean","split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"1.audio","text_column":"0.prompt"}}'
```

**关键参数：**

- `--backend`：服务器 URL 和用于多模态输入的聊天补全端点
- `--profile kind=synchronous`：顺序执行
- `--constraint kind=max_requests,count=20`：将请求数量限制为 20
- `--data`：指定两次——第一次用于合成提示词（`kind=synthetic_text`），第二次用于从 `openslr/librispeech_asr` 读取真实音频（`kind=huggingface`，并通过 `load_kwargs` 指定数据集配置）
- `--data-column-mapper`：将数据集索引 1 中的音频（`"1.audio"`，LibriSpeech）以及索引 0 中的文本（`"0.prompt"`，合成提示词）映射到每个请求中。

上述命令使用合成数据集中的提示词文本以及 LibriSpeech 数据集中的音频，在目标服务器上对 `chat/completions` 端点进行基准测试。输出示例如下：

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
