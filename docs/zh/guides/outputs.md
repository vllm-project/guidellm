---
weight: 2
source_sha256: !!str "a92596617cad147aa021099b7ce2be22fc83304a85246db2e031f478dd401ad5"
---

# 输出类型

GuideLLM 提供灵活的基准测试结果输出方式，既支持在控制台查看摘要，也支持将详细报告保存到文件。本文介绍支持的输出类型及其配置方法，并说明如何有效使用这些输出。

## CLI 输出配置

如果没有指定任何 `--output` 选项，`guidellm run` 会在[默认结果目录](#configuring-file-outputs)中写入 `benchmarks.json` 和 `benchmarks.csv`。

输出配置遵循基于类型注册表的 CLI 模式。指定任意 `--output` 后，默认的 JSON 和 CSV 输出都会被替换，因此每种需要的文件格式都要重复指定。以下示例保留两种默认格式，并额外生成 HTML：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json \
  --output kind=csv \
  --output kind=html
```

请从[支持的文件格式](#supported-file-formats)中选择格式，并参阅[配置文件输出](#configuring-file-outputs)设置路径。控制台输出单独控制，具体说明见下文。

## 控制台输出

默认情况下，GuideLLM 会直接在控制台显示基准测试结果和进度。控制台输出是隐式默认项，不会被显式指定的 `--output` 选项替换；指定 `--output kind=console` 也不会产生额外效果。控制台进度和输出分为以下几个部分：

1. **初始设置进度**：显示初始设置进度，包括服务器连接和数据准备情况。
2. **基准测试进度**：显示基准测试的运行进度，包括已完成的请求数和当前速率。
3. **最终结果**：汇总基准测试结果，包括平均延迟、吞吐量和其他关键指标。
   1. **基准测试元数据**：汇总本次基准测试的信息，包括服务器详情、数据配置和 profile 参数。
   2. **基准测试信息**：概述每项基准测试，包括请求状态、token 数量和耗时。
   3. **基准测试统计数据**：展示每项基准测试的详细统计信息，例如请求速率、并发数、延迟和 token 级指标。

### 禁用控制台输出

要禁用交互式进度更新，请使用 `--disable-console-interactive`（别名 `--disable-progress`）：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console-interactive
```

要禁用所有控制台输出，请使用 `--disable-console`（别名 `--disable-console-outputs`）：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console
```

## 文件输出

GuideLLM 支持将基准测试结果保存为多种文件格式，包括 JSON、YAML 和 CSV。这些文件可用于后续分析和报告，也可以重新载入 Python 进行详细探索。

### 支持的文件格式 {#supported-file-formats}

| 格式（`kind`） | 默认文件名        | 内容                                                                                         |
| -------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| `json`         | `benchmarks.json` | 配置、元数据、基准测试统计数据，以及为调试或重新载入 Python 而保留的请求数据。               |
| `yaml`         | `benchmarks.yaml` | 与 JSON 相同的详细基准测试数据，以便于阅读的 YAML 格式呈现。                                 |
| `csv`          | `benchmarks.csv`  | 吞吐量、延迟、token 数量和速率的表格摘要，适用于电子表格和结果对比。不包含详细的请求级数据。 |
| `html`         | `benchmarks.html` | 自包含的报告，包含吞吐量和延迟图表及表格。CSS 和 JavaScript 均内嵌，可离线分享。             |
| `plot`         | `benchmarks.png`  | 静态基准测试性能图表。                                                                       |

对于 `plot`，`path` 的扩展名决定图像格式：PNG、JPG/JPEG、SVG 或 PDF。例如，`--output kind=plot,path=benchmarks.pdf` 会生成 PDF。未指定扩展名的路径默认使用 `.png`；不支持的扩展名会导致错误。`dpi` 参数（默认值为 `100`）控制图像分辨率，例如 `--output kind=plot,path=benchmarks.jpg,dpi=72`。

### 配置文件输出 {#configuring-file-outputs}

省略 `path` 时，文件将使用对应格式的默认文件名，并保存到 `GUIDELLM__DEFAULT_RESULTS_DIR` 配置的目录中；如果未设置该环境变量，则保存到当前目录。指定 `path=` 可以设置文件名或目标路径。

例如，以下命令会将 JSON 写入自定义路径，同时仍会独立输出控制台信息：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json
```

### 控制输出文件大小

运行时间较长、请求数达到数千次的基准测试可能会生成很大的 JSON 和 YAML 文件，因为默认情况下会保留每个请求的完整数据（提示文本、输出文本和工具调用）。`--metrics` 选项使用蓄水池抽样来限制保留的请求数据量；延迟、token 数量和计时等轻量统计信息仍会为每个请求保留。

使用 `sample_size` 设置每种状态组（已完成、出错、未完成）中保留完整数据的请求数上限：

| 值                | 行为                                  |
| ----------------- | ------------------------------------- |
| 未设置（默认值）  | 保留所有请求的完整数据                |
| `0`               | 删除所有请求数据（仅保留统计数据）    |
| `N`（例如 `100`） | 每组随机抽样保留 `N` 个请求的完整数据 |

例如，如需每种状态组仅保留 100 个随机抽样请求的完整数据，可以运行：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_requests,count=10000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics kind=generative,sample_size=100 \
  --output kind=json,path=results/benchmark.json
```

`--metrics` 还接受 `prefer_response_metrics` 参数（默认值为 `true`），用于控制在服务器返回的 token 数量和客户端计算的数量同时可用时，是否优先使用服务器报告的数值。通常无需更改此项。

### 重新载入结果

可以使用 `GenerativeBenchmarksReport` 类将 JSON 文件重新载入 Python，以便进一步分析。以下示例展示如何载入结果：

```python
from guidellm.benchmark import GenerativeBenchmarksReport

report = GenerativeBenchmarksReport.load_file(
    path="benchmarks.json",
)
benchmarks = report.benchmarks

for benchmark in benchmarks:
    print(benchmark.config.id_)
```

有关 `GenerativeBenchmarksReport` 类及其方法的更多信息，请参阅[源代码](https://github.com/vllm-project/guidellm/blob/main/src/guidellm/benchmark/schemas/report.py)。
