---
weight: -2
source_sha256: !!str "a9ee73fa2742ed6e4a262e1ea21d4cf885d8efac8455890b54170627ebe91025"
---

# 数据集

GuideLLM 支持多种数据集配置，以便对大型语言模型（LLM）进行基准测试和评估。本文将全面介绍如何为不同用例配置数据集，并提供详细示例，说明如何选择合适的配置方式。

## 数据参数概览 {#data-arguments-overview}

数据集和处理选项采用 `kind` 模式，用于选择已注册的实现并配置参数：

```bash
guidellm run --data kind=<TYPE>,key=value,... [other options...]
```

可以重复指定 `--data` 以组合多个数据源。对于扁平设置，请使用逗号分隔的 key=value 对；如果值中包含嵌套结构，请使用序列化后的 JSON。

### 数据集来源和类型

以下参数用于配置数据集及其处理方式：

- 数据集来源和类型。支持的类型：
  - `synthetic_text` — 按需生成合成提示词。必需字段：`prompt_tokens`。可选字段：`output_tokens`、`turns`、`prefix_tokens`、`prefix_count`、`prefix_buckets` 和分布控制参数（`prompt_tokens_stdev`、`output_tokens_stdev` 等）。
  - `huggingface`（别名 `hf`）— 从 Hugging Face Hub 或本地目录/文件加载数据。必需字段：`source`（数据集 ID 或路径）。通过 `load_kwargs` 传入数据集加载参数（例如 `split`、`name`）。
  - `json_file`、`csv_file`、`text_file`、`parquet_file`、`arrow_file`、`hdf5_file`、`tar_file` — 从本地文件加载数据。必需字段：`path`。
  - `db_file` — 使用 `load_kwargs.sql` 从数据库加载行。必需字段：`uri`，其格式为 SQLAlchemy 风格的 URI。目前 GuideLLM 仅支持 SQLite URI，例如 `sqlite:///prompts.db`。
  - `trace_synthetic`、`mooncake`、`weka`、`otel` — 使用 `--profile kind=replay` 重放 trace。必需字段：嵌套的 `source`，指向另一种数据集类型，例如 `source.kind=json_file,source.path=trace.jsonl` 或 `source.kind=huggingface,source.source=org/dataset`。可选字段：`timestamp_column`、`prompt_tokens_column`、`output_tokens_column`、`time_scale`、`copies` 和其他格式专属选项。请参阅[Trace 文件格式](trace_replay.md)。

对于接受本地 `path` 的 `*_file` 数据集类型，路径必须指向一个确切的文件。文件名或目录名中的 glob 字符会按字面值处理。

此外，还可以通过数据参数 `load_kwargs` 指定数据集加载时使用的其他参数：

- `load_kwargs`：数据集加载的其他参数。例如，可以使用 `--data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"test"}}'` 指定数据集分割。Trace 类型也以相同方式接受 `load_kwargs`，并将其转发给来源加载器。

对于 SQLite 数据库，可以通过 `load_kwargs.sql` 提供查询，例如 `--data '{"kind":"db_file","uri":"sqlite:///prompts.db","load_kwargs":{"sql":"SELECT text FROM samples"}}'`。

### 数据加载器

可以通过 `--data-loader` 和配置字符串指定数据加载器。目前唯一支持的类型是 `pytorch`：

- `--data-loader kind=pytorch,samples=1000,num_workers=4,shuffle=true`

### Tokenizer

可以通过 `--tokenizer` 和配置字符串指定 tokenizer。目前唯一支持的类型是 `huggingface_auto`：

- `--tokenizer '{"kind":"huggingface_auto","model":"path/to/processor","load_kwargs":{"use_fast":false}}'`

如果 Hugging Face 模型需要自定义代码才能加载，请在 `load_kwargs` 中传入 `trust_remote_code`。请参阅[故障排查：trust_remote_code](troubleshooting.md#tokenizer-trust_remote_code)。

如果数据集使用了非标准列名，可以使用 `--data-column-mapper` 将这些列映射到 GuideLLM 预期的列名。在以下情况下尤其有用：

1. 数据集使用了不同的列名（例如使用 `question` 而不是 `prompt`，或使用 `instruction` 而不是 `text_column`）
2. 使用多个数据集，需要指定要采用哪个数据集的列
3. 数据集将 system prompt 或前缀放在单独的列中

### 列映射

`--data-column-mapper` 接受一个字符串，用于将列类型映射到列名：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}'
```

### 预处理器

使用 `--data-preprocessor` 可以控制要应用的 GuideLLM 预处理器，包括 `encode_media`、`tool_calling_message_extractor`、`turn_pivot`。

### 数据终结器

使用 `--data-finalizer` 可以控制如何将数据集行转换为请求（默认值为 `generative`）。

### 随机种子

使用 `--seed` 和配置字符串来控制序列的随机化，包括调度和合成数据生成。例如：`--seed kind=static,value=42`。

### 使用示例

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"test"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}' \
  --tokenizer '{"kind":"huggingface_auto","model":"path/to/processor","load_kwargs":{"use_fast":false}}' \
  --data-loader kind=pytorch,shuffle=true
```

## 数据集类型 {#datasets}

GuideLLM 支持多种数据集类型，每种类型都有各自的优势和适用场景。以下介绍 GuideLLM 支持的主要数据集类型，包括合成数据、Hugging Face 数据集、基于文件的数据集和内存数据集。

### 合成数据 {#synthetic-data}

合成数据集允许你使用可自定义的参数按需生成数据，适用于受控实验、压力测试和模拟特定场景。例如，你可能希望评估模型如何处理较长的提示词，或生成具有特定特征的输出。

GuideLLM 同时支持下文介绍的合成*文本*数据，以及用于视觉语言模型基准测试的合成*视觉*数据（图像和短视频）。请参阅[合成视觉数据](multimodal/synthetic_vision.md)，了解 `synthetic_image` 和 `synthetic_video` 这两种 `--data` 类型；它们可以与 `synthetic_text` 组合，用于多模态提示词。

#### 示例命令

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

或者使用 JSON 配置字符串：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128}'
```

#### 配置选项

- `prompt_tokens`：提示词中的平均 token 数。如果未指定其他参数，所有请求都将具有此 token 数。
- `prompt_tokens_stdev`：提示词 token 数的标准差。如果未提供且未指定最小值/最大值，则不应用偏差。如果未提供但指定了最小值/最大值，则使用均匀分布。
- `prompt_tokens_min`：提示词中的最小 token 数。如果未设置，且已设置标准差，则默认为 `max(1, prompt_tokens - 5 * prompt_tokens_stdev)`；否则默认为 `prompt_tokens`。
- `prompt_tokens_max`：提示词中的最大 token 数。如果未设置，且已设置标准差，则默认为 `prompt_tokens + 5 * prompt_tokens_stdev`；否则默认为 `prompt_tokens`。
- `output_tokens`：输出中的平均 token 数。可选；对不生成输出 token 的端点（例如 embeddings）进行基准测试时可省略。
- `output_tokens_stdev`：输出 token 数的标准差。如果未提供且未指定最小值/最大值，则不应用偏差。如果未提供但指定了最小值/最大值，则使用均匀分布。
- `output_tokens_min`：输出中的最小 token 数。如果未设置，且已设置标准差，则默认为 `max(1, output_tokens - 5 * output_tokens_stdev)`；否则默认为 `output_tokens`。
- `output_tokens_max`：输出中的最大 token 数。如果未设置，且已设置标准差，则默认为 `output_tokens + 5 * output_tokens_stdev`；否则默认为 `output_tokens`。

设置标准差后，token 数量会遵循经过配置边界截断的正态分布。未设置标准差时，如果最小值和最大值不同，则使用均匀整数分布。例如，`prompt_tokens=256,prompt_tokens_stdev=16` 的默认边界为 176 和 336 个 token。

合成文本按需生成。使用运行时约束（如 `--constraint kind=max_requests,count=1000`）停止基准测试。若要预先生成固定数量的数据行，并在多次数据集迭代中重复使用，请为数据加载器设置 `samples`：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --data-loader kind=pytorch,samples=100
```

默认值 `samples=-1` 会以流式方式读取数据行，而不是预先生成固定数量的数据。若要对现有文本文件中的提示词进行基准测试，请使用 `--data kind=text_file,path=prompts.txt`；请参阅[基于文件的数据集](#file-based-datasets)。

#### 说明

- 必须提供 tokenizer。默认使用后端中的模型。如果该模型不可用，请使用 `--tokenizer kind=huggingface_auto,model=<path-or-id>` 指定包含 tokenizer 文件的目录或 Hugging Face 模型 ID。

### Hugging Face 数据集

GuideLLM 支持从 Hugging Face Hub 或符合 `datasets` 库格式的本地目录加载数据集。这样便于使用丰富的数据集，对真实数据进行基准测试和评估。

#### 示例命令

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=huggingface,source=garage-bAInd/Open-Platypus
```

或者使用本地数据集目录：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=huggingface,source=path/to/dataset
```

通过 `load_kwargs` 传入 HuggingFace `load_dataset` 选项：

```bash
--data '{"kind":"huggingface","source":"my/dataset","load_kwargs":{"split":"train"}}'
```

#### 说明

- Hugging Face 数据集可以通过 ID、本地目录或本地 Python 文件路径指定。
- 支持的 Hugging Face 数据集格式是指可以使用 `datasets` 库中的 `load_dataset` 函数加载，因此可表示为 `Dataset`、`DatasetDict`、`IterableDataset` 或 `IterableDatasetDict`。关于支持的数据类型以及底层 `load_dataset` 可用的其他参数，请参阅 [Hugging Face datasets 文档](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub)。
- 仅当环境中设置了 `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` 或 `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"` 时，才需要 tokenizer。在这种情况下，必须使用 `--tokenizer kind=huggingface_auto,model=...` 指定 tokenizer。如果未设置这些环境变量，则会将传入的模型或从后端获取的模型设为 tokenizer。

### 基于文件的数据集 {#file-based-datasets}

GuideLLM 支持多种数据集文件格式，包括文本、CSV、JSON 等。这些数据集可用于基准测试和评估，让你可以根据用例使用熟悉且结构匹配的数据格式。

#### 支持的格式和示例

- **文本文件（`.txt`、`.text`）**：每行是一条要用作提示词的内容。

  ```
  Hello, how are you?
  What is your name?
  ```

- **CSV 文件（`.csv`）**：每行是一条数据集记录，首行包含列名。列中应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他列，列名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。

  ```csv
  prompt,output_tokens_count,additional_column,additional_column2
  Hello, how are you?,5,foo,bar
  What is your name?,3,baz,qux
  ```

- **JSON Lines 文件（`.jsonl`）**：每行是一个独立的 JSON 对象。对象中应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他字段，字段名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。

  ```json
  {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"}
  {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
  ```

- **Trace 文件（`.jsonl`、`.json`、`.csv` 或 `.parquet`，且采用支持的 trace 文件格式）**：用于回放的专用文件。与 `--profile kind=replay` 配合使用，根据每行中的时间戳和 token 长度重放 trace 事件。时间戳必须是以秒为单位的数字，并使用统一时间轴上的任意一致零点；`otel` 除外，它会将 ISO-8601 `start_time` 值（以及经 HuggingFace 解码的 `datetime` 对象）解析为纪元秒数。GuideLLM 会按时间戳排序，并在调度前将它们转换为相对于第一个事件的偏移量。请参阅[Trace 回放基准测试](../getting-started/benchmark.md#trace-replay-benchmarking)。

  ```json
  {"timestamp": 1234500.0, "input_length": 256, "output_length": 128}
  {"timestamp": 1234500.5, "input_length": 512, "output_length": 64}
  ```

  在此示例中，第二个请求会在第一个请求之后 0.5 秒调度。GuideLLM 会先按时间戳对 trace 行排序，再调度请求并生成合成负载。这样每个已调度事件都会与同一行中的提示词和输出 token 长度保持对应。

  使用支持的 [trace 文件格式](trace_replay.md#supported-formats)以启用 trace 加载：

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=trace_synthetic,source.kind=json_file,source.path=path/to/trace.jsonl,time_scale=1.0
  ```

  默认情况下，所有 trace 格式都会查找 "timestamp"、"input_length" 和 "output_length" 列。如果 trace 使用了不同的列名，请在数据配置中加入 `timestamp_column`、`prompt_tokens_column` 和 `output_tokens_column`：

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,timestamp_column=timestamp,prompt_tokens_column=input_length,output_tokens_column=output_length
  ```

  `otel` 使用相同的嵌套 `source` 字段。token 数来自 span 属性，而不是顶层列。使用 Hugging Face 数据集是最快的入门方式：

  ```bash
  guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
    --constraint kind=max_requests,count=30
  ```

  默认记录的内容是 `gen_ai.input.messages`，且 `history=trace`（每个 span 的完整输入，`history_context=new`）。传入 `history=runtime` 后，只发送新消息，并通过 DAG 附加实时补全结果。对于不包含 `gen_ai.input.messages` 的 span，应使用 `trace_synthetic`。本地 JSONL 使用 `source.kind=json_file,source.path=...`，并可设置相同的 `history` 选项。请参阅[Trace 文件格式](trace_replay.md#otel)。

  对于回放，`--data` 中的 `time_scale` 用于缩放应用等待和打包上限后的 trace 事件间隔。等待上限（`max_wait`、`max_session_wait`、`min_concurrent_sessions`）会在应用 `time_scale` 之前按原始 trace 秒数执行。建议使用这些打包选项提高并行度；如果数据集太短，无法满足基准测试时长，可使用 `copies` 顺序重放打包后的数据集。回放配置也接受 `--profile kind=replay` 中的调度器侧 `time_scale`。使用 `--data-loader kind=pytorch,samples=1000` 可限制加载和重放的 trace 行数。`--constraint kind=max_requests,count=<n>` 仅作为运行时完成条件；它不会限制从文件加载的 trace 行数。`--constraint kind=max_duration,seconds=<n>` 还会取消正在进行的等待，包括回放休眠。

- **JSON 文件（`.json`）**：整个数据集以 JSON 对象数组的形式存放在特定键下。要指定正确的键，必须传入 `--data-column-mapper` 参数，并设置数组所在位置的 `"field": "NAME"`。对象中应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他字段，字段名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。

  ```json
  {
    "version": "1.0",
    "data": [
      {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"},
      {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
    ]
  }
  ```

- **Parquet 文件（`.parquet`）** 示例：一种二进制列式存储格式，适用于高效数据处理。有关支持的格式，请参阅[说明](#notes)部分中链接的 Hugging Face 数据集文档。

- **Arrow 文件（`.arrow`）** 示例：一种跨语言开发平台，用于内存中的数据。有关支持的格式，请参阅[说明](#notes)部分中链接的 Hugging Face 数据集文档。

- **HDF5 文件（`.hdf5`）** 示例：一种用于存储大量数据的分层数据格式。有关支持的格式，请参阅[说明](#notes)部分中链接的 Hugging Face 数据集文档。

#### 示例命令

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=1000 \
  --data kind=json_file,path=path/to/dataset.json \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt"}}'
```

对于其他格式，请将 `json_file` 替换为 `csv_file`、`text_file`、`parquet_file`、`arrow_file` 或 `hdf5_file`。

#### 说明 {#notes}

- 确保文件格式符合数据集的预期结构，并且属于支持的格式。
- `--data-column-mapper` 参数可用于指定解析数据集时的其他参数，例如提示词列名或要使用的数据集分割。
- 关于支持的格式以及底层 `load_dataset` 可用的其他参数，请参阅 [Hugging Face datasets 文档](https://huggingface.co/docs/datasets/en/loading#local-and-remote-files)。

### 内存数据集

内存数据集允许直接传入 Python 对象中的数据，适合快速原型开发和测试，无需先将数据保存到磁盘。

#### 支持的格式和示例

- **列名和取值组成的字典**：每个键是列名，对应的值是数据点列表。键中应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他列，列名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。
  ```python
  {"column1": ["value1", "value2"], "column2": ["value3", "value4"]}
  ```
- **字典列表**：每个字典代表一个数据点，并包含键值对。字典中应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他字段，字段名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。
  ```python
  [{"column1": "value1", "column2": "value3"}, {"column1": "value2", "column2": "value4"}]
  ```
- **条目列表**：每个条目是一个数据点。条目应包含 `prompt` 或其他常见提示词列名，GuideLLM 会将其用作提示词列。也可以包含其他字段，字段名可使用前文提到的 `--data-column-mapper` 参数所支持的别名。
  ```python
  ["value1", "value2", "value3"]
  ```

#### 使用示例

```python
from guidellm.benchmark import benchmark_generative_text

data = [
    {"prompt": "Hello", "output": "Hi"},
    {"prompt": "How are you?", "output": "I'm fine."}
]

benchmark_generative_text(data=data, ...)
```

#### 说明

- 确保数据格式一致，并符合某种受支持的结构。
- 对于字典，所有列的样本数量必须相同。
- 对于字典列表，所有条目必须具有相同的键。
- 对于条目列表，所有元素必须属于相同类型。
- 仅当环境中设置了 `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` 或 `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"` 时，才需要 tokenizer。在这种情况下，必须使用 `--tokenizer kind=huggingface_auto,model=...` 指定 tokenizer。如果未设置这些环境变量，则会将传入的模型或从服务器获取的模型设为 tokenizer。

## 数据集预处理

GuideLLM 提供了预处理命令，可处理数据集，使其提示词和输出具有指定的 token 数量。当你需要为基准测试统一数据集，或数据集中的提示词不符合目标 token 数量时，这项功能尤其有用。

预处理命令可以：

- 将提示词调整为目标 token 长度
- 使用由注册表支持的策略处理短于或长于目标长度的提示词
- 将数据集中的列映射到 GuideLLM 预期的列名
- 根据策略配置生成输出 token 数量
- 以多种格式保存处理后的数据集

### 基本用法

```bash
guidellm preprocess dataset \
    kind=<DATA_TYPE>,key=value,... \
    <OUTPUT_PATH> \
    --tokenizer kind=huggingface_auto,model=<MODEL> \
    --strategy kind=<STRATEGY>,prompt_tokens=<N>,output_tokens=<N>
```

### 必需参数

| 参数          | 说明                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`        | 指定要处理的数据集。支持[数据集配置](#datasets)中记录的所有数据集格式。通过此描述符中的 `load_kwargs` 传递加载器选项。              |
| `OUTPUT_PATH` | 保存处理后数据集的路径，需包含文件后缀（例如 `processed_dataset.jsonl`、`output.csv`）。                                            |
| `--tokenizer` | **必需。** 使用注册表语法配置分词器，例如 `--tokenizer kind=huggingface_auto,model=gpt2`。可选的构造函数参数放在 `load_kwargs` 中。 |
| `--strategy`  | **必需。** 预处理策略，包含 token 目标数量和短提示词处理方式。参见[预处理策略](#preprocess-strategy)。                              |

### 示例

```bash
guidellm preprocess dataset \
    kind=json_file,path=path/to/input_dataset.jsonl \
    "path/to/processed_dataset.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256,prefix_tokens_max=100
```

### 预处理策略 {#preprocess-strategy}

`--strategy` 选项使用与 GuideLLM 其他选项相同的注册表格式（`kind=<name>,...` 或 JSON）。它将目标 token 数量设置与短提示词处理类型组合在一起。

**所有类型共用的字段：**

- `prompt_tokens`：提示词的平均 token 数量。若未指定其他设置，所有提示词都会调整为此 token 数量。
- `prompt_tokens_stdev`：提示词 token 数量的标准差。若未提供且未指定最小值/最大值，则不应用偏差；若未提供但指定了最小值/最大值，则使用均匀分布。
- `prompt_tokens_min`：提示词的最小 token 数量。若未设置但设置了 `prompt_tokens_stdev`，最小值为 1。
- `prompt_tokens_max`：提示词的最大 token 数量。若未设置但设置了 `prompt_tokens_stdev`，最大值为标准差的 5 倍。
- `output_tokens`：输出的平均 token 数量。若未指定其他设置，所有输出都会具有此 token 数量。
- `output_tokens_stdev`：输出 token 数量的标准差。若未提供且未指定最小值/最大值，则不应用偏差；若未提供但指定了最小值/最大值，则使用均匀分布。
- `output_tokens_min`：输出的最小 token 数量。若未设置但设置了 `output_tokens_stdev`，最小值为 1。
- `output_tokens_max`：输出的最大 token 数量。若未设置但设置了 `output_tokens_stdev`，最大值为标准差的 5 倍。
- `prefix_tokens_max`：保留的前缀 token 最大数量。设置后，前缀会被截断至此长度。
- `count_prefix`：为 `true` 时，将前缀 token 计入提示词 token 预算（默认值：`false`）。

**策略类型：**

| 类型          | 说明                                        | 该类型专用字段                                     |
| ------------- | ------------------------------------------- | -------------------------------------------------- |
| `ignore`      | 丢弃短于目标长度的提示词                    | —                                                  |
| `concatenate` | 将多个短提示词拼接，直到达到目标长度        | `delimiter`（拼接提示词之间的字符串；默认值 `""`） |
| `pad`         | 用字符/字符串填充短提示词，直到达到目标长度 | `pad`（必填的非空填充字符串）                      |
| `error`       | 如果提示词短于目标长度，则报错              | —                                                  |

**策略示例：**

```bash
# Ignore short prompts (default style)
--strategy kind=ignore,prompt_tokens=512,output_tokens=256

# Concatenate short prompts
--strategy kind=concatenate,prompt_tokens=512,output_tokens=256,delimiter="\n\n"

# Pad short prompts
--strategy kind=pad,prompt_tokens=512,output_tokens=256,pad=" "

# JSON form with count_prefix
--strategy '{"kind":"ignore","prompt_tokens":512,"output_tokens":256,"count_prefix":true}'
```

`--tokenizer` 参数用于指定计算 token 数量时使用的分词器（注册表格式与 `guidellm run --tokenizer` 相同）。此参数为必需项，因为预处理命令需要对提示词进行分词，以确保其符合目标 token 数量。有关在基准测试中使用分词器的信息，请参见[数据参数概览](#data-arguments-overview)。

### 列映射

当数据集使用非标准列名时，可以使用 `--data-column-mapper` 将列映射到 GuideLLM 预期的列名。以下情况尤其适用：

1. **数据集使用了不同的列名**（例如使用 `question` 而不是 `prompt`，或使用 `instruction` 而不是 `text_column`）
2. **你有多个数据集**，需要指定要使用哪个数据集的列
3. **数据集将 system prompt 或前缀放在单独的列中**

**列映射格式：**`--data-column-mapper` 选项使用注册表语法（与 `guidellm run` 相同）：

```bash
--data-column-mapper kind=generative_column_mapper,column_mappings.text_column=question,column_mappings.prefix_column=system_prompt

# Or as JSON:
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {
  "text_column": "question",
  "prefix_column": "system_prompt",
  "prompt_tokens_count_column": "input_tokens",
  "output_tokens_count_column": "completion_tokens"
}}'
```

**支持的列类型：**

- `text_column`：主要提示词文本（默认查找：`prompt`、`instruction`、`question`、`input`、`context`、`content`、`text`）
- `raw_messages_column`：当前轮次的 Chat Completions 消息（`[{role, content}, …]`；默认查找：`messages`、`chat_messages`）。`/v1/chat/completions` 会原样发送这些消息。其他请求格式如果无法根据这些消息构建 `prompt`/`input`，则会中止。
- `prefix_column`：系统提示词或前缀（默认查找：`system_prompt`、`system`、`prefix`）
- `prompt_tokens_count_column`：包含提示词 token 数量的列（默认查找：`prompt_tokens_count`、`input_tokens_count`）
- `output_tokens_count_column`：包含输出 token 数量的列（默认查找：`output_tokens_count`、`completion_tokens_count`）
- `image_column`：图像数据列
- `video_column`：视频数据列
- `audio_column`：音频数据列

**示例：映射自定义列名**

假设你的 CSV 文件包含 `user_query` 和 `system_message` 两列：

```csv
user_query,system_message
"What is AI?","You are a helpful assistant."
"How does ML work?","You are a technical expert."
```

可使用以下命令：

```bash
guidellm preprocess dataset \
    "kind=csv_file,path=dataset.csv" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=user_query,column_mappings.prefix_column=system_message
```

**示例：多个数据集**

如果你正在处理多个数据集，并需要指定要使用哪个数据集的列，可以使用 `<dataset_index>.<column_name>` 或 `<dataset_name>.<column_name>` 格式：

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"text_column": "0.prompt", "prefix_column": "1.system"}}'
```

### 其他选项

| 选项                    | 说明                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--data-column-mapper`  | 列映射预处理器配置（默认值：`kind=generative_column_mapper`）。                                                              |
| `--data-loader`         | 数据加载器配置（默认值：`kind=pytorch`）。使用 `samples` 限制写入的处理后行数。预处理期间会忽略 `shuffle` 和 `num_workers`。 |
| `--seed`                | 用于可复现 token 采样的随机种子配置（默认值：`kind=static,value=42`）。                                                      |
| `--push-to-hub`         | 将处理后的数据集推送到 Hugging Face Hub（标志）。                                                                            |
| `--hub-dataset-id <ID>` | 上传所需的 Hugging Face Hub 数据集 ID（设置 `--push-to-hub` 时必需）。                                                       |

### 完整示例

**示例 1：使用自定义列名进行基本预处理**

```bash
guidellm preprocess dataset \
    kind=csv_file,path=my_dataset.csv \
    "processed_dataset.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=user_question,column_mappings.prefix_column=system_instruction
```

**示例 2：使用分布和 concatenate 策略进行预处理**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=concatenate,prompt_tokens=512,prompt_tokens_stdev=50,output_tokens=256,output_tokens_stdev=25,delimiter="\n\n" \
    --seed kind=static,value=123
```

**示例 3：预处理时指定分词器加载参数、前缀限制和 count_prefix**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2,load_kwargs.use_fast=false \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256,prefix_tokens_max=100,count_prefix=true
```

**示例 4：使用 --data-loader 限制处理后的行数**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=large_dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --data-loader kind=pytorch,samples=1000
```

**示例 5：预处理并上传到 Hugging Face Hub**

```bash
guidellm preprocess dataset \
    "kind=json_file,path=my_dataset.jsonl" \
    "processed.jsonl" \
    --tokenizer kind=huggingface_auto,model=gpt2 \
    --strategy kind=ignore,prompt_tokens=512,output_tokens=256 \
    --push-to-hub \
    --hub-dataset-id "username/processed-dataset"
```

### 说明

- `--strategy` 选项接受 `PreprocessStrategyArgs` 注册表类型，其中包含 token 数量字段（`prompt_tokens`、`output_tokens` 及相关分布字段）、`prefix_tokens_max`、`count_prefix` 和短提示词处理 `kind`。参见[预处理策略](#preprocess-strategy)。
- 分词器为必需项，因为预处理命令需要对提示词进行分词，以确保它们符合目标 token 数量。有关基准测试中的分词器用法，请参见[数据参数概览](#data-arguments-overview)。
- 数据集加载器参数应写在位置参数 `DATA` 描述符中（`load_kwargs=...`），而不是单独的 `--data-args` 标志中。
- 使用 `--data-loader kind=pytorch,samples=<N>` 可在写入 `N` 行处理结果后停止。为了与 `guidellm run` 的 CLI 保持一致，`shuffle` 和 `num_workers` 参数可以接受，但在预处理中会被忽略。
- 仅当数据集使用非标准列名时才需要列映射。未提供映射时，GuideLLM 会自动尝试常见列名。
- 输出文件格式由 `OUTPUT_PATH` 中的文件后缀决定（例如 `.json`、`.jsonl`、`.csv`、`.parquet`）。
