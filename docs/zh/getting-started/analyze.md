---
weight: -4
---

# 分析结果

完成[基准测试](../../getting-started/benchmark.md)后，GuideLLM 会提供全面的结果，帮助你了解大语言模型部署的性能。本指南介绍如何解读控制台输出和保存到文件的结果。

## 了解控制台输出

基准测试完成后，GuideLLM 会自动在控制台显示结果，分为三个主要部分：

### 1. 基准测试元数据

这一部分概述基准测试运行情况，包括：

- **服务器配置：** 目标 URL、模型名称和后端详情
- **数据配置：** 数据来源、token 数量和数据集属性
- **负载模式参数：** 请求速率类型、最长运行时间、请求数量限制等
- **附加信息：** 通过 `--output-extras` 参数提供的其他元数据

例如：

```
Benchmarks Metadata
------------------
Args:        {"backend_type": "openai", "target": "http://localhost:8000", "model": "Meta-Llama-3.1-8B-Instruct-quantized", ...}
Worker:      {"type_": "generative", "backend_type": "openai", "backend_args": {"timeout": 120.0, ...}, ...}
Request Loader: {"type_": "generative", "data_args": {"prompt_tokens": 256, "output_tokens": 128, ...}, ...}
Extras:      {}
```

### 2. 基准测试概况

这一部分以表格汇总每次基准测试运行的关键信息，包含以下列：

- **类型：** 基准测试类型，例如同步、恒定速率、泊松分布等
- **开始/结束时间：** 基准测试的开始和结束时间
- **持续时间：** 基准测试的总时长，以秒为单位
- **请求数：** 成功、未完成及出错的请求数量
- **Token 统计：** 输入提示和输出的平均 token 数及 token 总数

通过这一部分，你可以了解实际执行了哪些测试，并快速掌握结果概况。

### 3. 基准测试统计数据

这是性能分析中最关键的部分，展示了各项指标的详细统计数据：

- **吞吐量指标：**

  - 每秒请求数（RPS）
  - 并发请求数
  - 每秒输出 token 数
  - 每秒总 token 数

- **延迟指标：**

  - 请求延迟（平均值、中位数、p99）
  - 首 token 延迟（TTFT；平均值、中位数、p99）
  - Token 间延迟（ITL；平均值、中位数、p99）
  - 每个输出 token 的生成耗时（平均值、中位数、p99）

p99（第 99 百分位数）对服务级别目标（SLO）分析尤为重要：99% 的请求在该指标上的数值不超过 p99。

## 分析保存的结果

为了便于深入分析，GuideLLM 默认将详细结果保存为以下文件：

- `benchmarks.json`：JSON 格式的完整基准测试数据
- `benchmarks.csv`：CSV 格式的关键指标汇总

文件会写入 `GUIDELLM__DEFAULT_RESULTS_DIR` 指定的目录；如果未设置该变量，则写入当前目录。若要指定文件名和保存位置，请参阅英文版[配置文件输出](../../guides/outputs.md#configuring-file-outputs)。

### 文件格式

可用的报告格式请参阅英文版[支持的文件格式](../../guides/outputs.md#supported-file-formats)；指定格式的示例请参阅英文版[输出配置](../../guides/outputs.md#cli-output-configuration)。显式指定 `--output` 选项后，默认的 JSON 和 CSV 输出配置将被替换。控制台显示选项请参阅英文版[控制台输出](../../guides/outputs.md#console-output)。

### 通过程序分析

如需自定义分析，可以在 Python 中重新加载结果：

```python
from guidellm.benchmark import GenerativeBenchmarksReport

# Load results from file
report = GenerativeBenchmarksReport.load_file("benchmarks.json")

# Access individual benchmarks
for benchmark in report.benchmarks:
    # Print basic info
    print(f"Benchmark: {benchmark.id_}")
    print(f"Type: {benchmark.type_}")

    # Access metrics
    print(f"Avg RPS: {benchmark.metrics.requests_per_second.successful.mean}")
    print(f"p99 latency: {benchmark.metrics.request_latency.successful.percentiles.p99}")
    print(f"TTFT (p99): {benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99}")
```

## 关键性能指标

分析结果时，重点关注以下指标：

### 1. 吞吐量与容量

- **最大 RPS：** 服务器能承受的最高请求速率是多少？
- **并发量：** 服务器能同时处理多少个请求？
- **Token 吞吐量：** 服务器每秒能生成多少个 token？

### 2. 延迟与响应速度

- **首 token 延迟（TTFT）：** 模型需要多久才开始生成输出？
- **Token 间延迟（ITL）：** 模型生成后续 token 的速度是否平稳？
- **总请求延迟：** 一个请求从开始到完成需要多长时间？

### 3. 可靠性与错误率

- **成功率：** 成功完成的请求占比是多少？
- **错误分布：** 出现了哪些类型的错误？各占多少？

## 其他分析方法

### 比较不同模型或硬件

使用不同的模型或硬件配置运行基准测试，然后比较结果：

```bash
guidellm run \
  --backend kind=openai_http,target=http://server1:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=model1/benchmarks.json

guidellm run \
  --backend kind=openai_http,target=http://server2:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=model2/benchmarks.json
```

### 优化成本

分析以下指标，衡量成本效益：

- 单位硬件成本对应的每秒 token 数
- 不同硬件配置下的最大吞吐量
- 最优批处理大小与延迟之间的取舍

### 确定扩容需求

根据基准测试结果规划：

- 处理预期负载需要多少台服务器
- 应在何时根据需求自动扩容或缩容
- 哪种硬件能为当前工作负载提供最优性价比
