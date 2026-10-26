---
weight: 4
source_sha256: !!str "7def85ce965d16e808c8314fc6615d528967e3dc2c58a24265c40d894965245d"
---

# 过饱和停止

GuideLLM 提供过饱和检测（Over-Saturation Detection，OSD），当模型进入过饱和状态时可自动停止基准测试。此功能可检测响应速率已无法跟上请求速率的情况，从而避免浪费计算资源，并确保基准测试结果有效。

## 什么是过饱和？

当大语言模型（LLM）推理服务器收到请求的速度超过其处理速度时，就会发生过饱和，导致请求队列不断增长。随着队列变长，服务器开始处理每个请求所需的时间也会逐渐增加，性能指标随之下降。如果性能基准测试工具使 LLM 推理服务器过载，测得的指标就会严重失真，失去参考价值。

这就像收银员突然遇到客流高峰而应接不暇：队伍（负载）不断变长，收银员跟不上，队伍越排越长，也没有空间再容纳更多顾客。通过自动检测过饱和并停止基准测试，可以避免浪费宝贵的机器运行时间。

## 工作原理

GuideLLM 的过饱和检测（OSD）算法使用统计斜率检测，判断模型何时进入过饱和状态。该算法会持续跟踪以下两项指标：

1. **并发请求数**：同时处理中的请求数量
2. **首个 token 延迟（TTFT）**：生成响应中第一个 token 所需的延迟

对于每项指标，算法会：

- 维护近期数据点的滑动窗口
- 使用在线统计量计算线性回归斜率
- 使用 t 分布置信区间计算误差范围（MOE）
- 检测误差范围较小的正斜率，以判断性能是否正在退化

满足以下条件时，系统会判定发生过饱和：

- 并发请求数和 TTFT 的斜率均显著为正
- 已达到最短持续时间阈值
- 数据点数量足够，可以可靠地估算斜率

检测到过饱和后，该约束会自动停止请求继续排队，并可选择停止处理已有请求，从而避免进一步浪费资源。

## 用法

在约束配置中使用 `kind=over_saturation` 启用过饱和检测：

```bash
--constraint kind=over_saturation,...
```

### 基本用法

使用默认设置启用过饱和检测：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=throughput,max_concurrency=10 \
  --constraint kind=over_saturation
```

### 高级配置

在约束配置字符串中设置检测参数：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=16 \
  --constraint kind=over_saturation,mode=enforce,min_seconds=60,max_window_seconds=300,moe_threshold=1.5
```

也可以使用 JSON 配置：

```bash
--constraint '{"kind":"over_saturation","mode":"enforce","min_seconds":60,"max_window_seconds":300,"moe_threshold":1.5}'
```

## 配置选项

启用过饱和检测时，可以配置以下参数：

- **`mode`**（`"enforce"` | `"monitor"`，默认：`"enforce"`）：控制检测到过饱和时的行为。`"enforce"` 会停止基准测试；`"monitor"` 只进行监控并报告结果。
- **`min_seconds`**（浮点数，默认：`30.0`）：开始检查过饱和前的最短等待秒数，避免在初始预热阶段误报。
- **`max_window_seconds`**（浮点数，默认：`120.0`）：保留数据的最大时间窗口（秒）。系统会自动删除较早的数据点，以限制内存使用量。
- **`moe_threshold`**（浮点数，默认：`2.0`）：用于斜率检测的误差范围阈值。数值越低，对性能退化越敏感。
- **`minimum_ttft`**（浮点数，默认：`2.5`）：统计违规情况的最小 TTFT 阈值（秒）。只有超过此阈值的 TTFT 值才会计为违规。
- **`maximum_window_ratio`**（浮点数，默认：`0.75`）：窗口大小占请求总数的最大比例。此参数通过限制跟踪的请求数量来控制内存使用量。
- **`minimum_window_size`**（整数，默认：`5`）：估算斜率所需的最少数据点数，确保作出检测判断前有足够的统计依据。
- **`confidence`**（浮点数，默认：`0.95`）：t 分布计算使用的置信水平（0–1）。数值越高，检测过饱和所需的证据越充分。

## 适用场景

以下场景尤其适合使用过饱和检测：

### 压力测试与容量规划

测试系统处理不断增加的负载的能力时，过饱和检测会在系统无法继续跟上负载时自动停止基准测试，避免无效运行继续消耗计算资源。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep,sweep_size=5 \
  --constraint kind=over_saturation
```

### 降低基准测试成本

对多个模型、GPU 和配置运行大规模基准测试矩阵时，过饱和检测可以提前终止无效运行，显著降低成本。

### 寻找安全运行区间

使用过饱和检测找出部署可持续支持的最大吞吐量，帮助设置合适的速率限制并规划容量。

## 解读结果

无论使用 `"enforce"` 还是 `"monitor"` 模式，启用过饱和检测后，基准测试输出都会包含检测状态的元数据。这些元数据位于调度器动作元数据中，包括：

- **`is_over_saturated`**（布尔值）：评估时是否检测到过饱和
- **`concurrent_slope`**（浮点数）：并发请求数的计算斜率
- **`concurrent_slope_moe`**（浮点数）：并发请求数斜率的误差范围
- **`concurrent_n`**（整数）：计算并发请求数斜率所用的数据点数量
- **`ttft_slope`**（浮点数）：TTFT 的计算斜率
- **`ttft_slope_moe`**（浮点数）：TTFT 斜率的误差范围
- **`ttft_n`**（整数）：计算 TTFT 斜率所用的数据点数量
- **`ttft_violations`**（整数）：超过最小阈值的 TTFT 值数量

这些指标有助于理解触发过饱和检测的原因，并据此微调检测参数。

## 示例：启用过饱和检测的完整基准测试

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=16 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=300 \
  --constraint kind=over_saturation,mode=enforce,min_seconds=30,max_window_seconds=120 \
  --output kind=json,path=benchmark.json \
  --output kind=html,path=benchmark.html
```

此示例会：

- 运行包含 16 个并发请求的基准测试
- 使用包含 256 个提示 token 和 128 个输出 token 的合成数据
- 使用自定义时间参数启用过饱和检测
- 将最长运行时间设为 300 秒，作为后备限制
- 同时以 JSON 和 HTML 格式输出结果

## 更多资源

如需深入了解过饱和检测，包括算法开发、评估指标和实现细节，请参阅以下 Red Hat Developer 博文：

- [通过过饱和检测降低 LLM 基准测试成本](https://developers.redhat.com/articles/2025/11/18/reduce-llm-benchmarking-costs-oversaturation-detection)：介绍过饱和问题及其对 LLM 基准测试的重要性
- [定义成功：过饱和检测的评估指标与数据增强](https://developers.redhat.com/articles/2025/11/20/oversaturation-detection-evaluation-metrics)：介绍如何通过自定义指标、数据集标注和负载增强技术评估 OSD 算法
- [通过迭代误差分析构建过饱和检测器](https://developers.redhat.com/articles/2025/11/24/building-oversaturation-detector-iterative-error-analysis)：详细介绍 OSD 算法的构建过程
