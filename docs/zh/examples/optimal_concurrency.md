---
source_sha256: !!str "49e2398539a6eaf15d86dcd21b13f3037b248a22a951c608f6c73c38b9f003e8"
---

# 找到服务器的最佳并发上限 {#finding-your-servers-optimal-concurrency-limit}

确定 LLM 服务器在延迟开始恶化前能够处理的最大并发用户数，并找出适合当前部署的最佳运行点。

## 前提条件

- 正在运行的兼容 OpenAI 服务器（[服务器设置指南](../getting-started/server.md)）
- 已安装 GuideLLM（[安装指南](../getting-started/install.md)）
- 了解如何运行基准测试（[基准测试指南](../getting-started/benchmark.md)）

## 第 1 步：运行 Sweep 测试

`sweep` 配置档会自动测试逐渐增加的负载：从单个顺序请求开始，逐步提升到最大吞吐量，并在中间插入多个速率点。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=sweep_results.json
```

该命令最多运行 10 种策略（同步基线、吞吐量上限和 8 个插值速率），每种策略运行 120 秒。

调整 `prompt_tokens` 和 `output_tokens`，使其符合实际工作负载。聊天应用可能使用 `prompt_tokens=256,output_tokens=512`，摘要生成流水线则可能使用 `prompt_tokens=2048,output_tokens=128`。

## 第 2 步：查看结果

打开控制台输出或 JSON 报告。针对每种策略，查看以下指标：

| 指标                               | 含义                                      |
| ---------------------------------- | ----------------------------------------- |
| `output_tokens_per_second`（均值） | 服务器吞吐量，即服务器完成工作的速度      |
| `time_to_first_token_ms`（p50）    | 用户收到响应前等待时间的中位数            |
| `time_to_first_token_ms`（p99）    | 最坏情况下的等待时间，对尾延迟 SLO 很重要 |
| `inter_token_latency_ms`（p50）    | 流式输出体验是否顺畅                      |
| `request_latency`（p50）           | 每个请求端到端的总耗时                    |

## 第 3 步：识别三个区域

随着 Sweep 测试中的负载逐渐增加，服务器会依次经过以下三个区域：

**利用不足区**：吞吐量随负载线性增长，延迟保持稳定。增加并发用户数会让服务器按比例完成更多工作，同时不会带来额外损失。此时服务器容量尚未充分利用。

**最佳区间**：吞吐量仍在上升，但增幅逐渐缩小。TTFT p50 开始上升，但仍在可接受范围内。服务器正在高效地批处理请求。**这就是理想的运行区间。**

**过饱和区**：吞吐量趋于平台或开始下降，TTFT p99 急剧上升。请求队列的增长速度超过服务器的处理能力。GuideLLM 的过饱和检测器会在此处停止 Sweep 测试。

观察第一个区域到第二个区域的转折点：此时吞吐量仍在上升，但 TTFT p99 开始明显增加。该转折点之前的策略就是最佳运行点。

## 第 4 步：通过定点运行验证

确定目标并发量后（例如，Sweep 结果显示为 32 个并发流），在该并发量下运行更长时间的基准测试，以确认指标稳定：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --profile kind=concurrent,streams=32 \
  --constraint kind=max_duration,seconds=300 \
  --seed kind=static,value=42 \
  --output kind=json,path=validation_run.json
```

固定并发量运行 5 分钟，比 Sweep 中较短的策略测试能提供更可靠的指标。将验证结果与 Sweep 的预测进行比较，二者应保持一致。如果 TTFT p99 高于预期，就将并发量降低一个级别并重新验证。

## 第 5 步：测试临界值附近的并发量

要获得更精确的结果，请围绕候选并发量测试几个并发级别：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --profile kind=concurrent \
  --override profile.streams 24,28,32,36,40 \
  --constraint kind=max_duration,seconds=180 \
  --seed kind=static,value=42 \
  --output kind=json,path=boundary_test.json
```

这会在间隔较小的并发级别下运行 5 次子基准测试。比较这些测试的吞吐量和 TTFT p99，即可准确找出性能开始下降的位置。

## 如何确定运行值

最终值取决于你的部署目标：

**对延迟敏感（聊天、实时应用）**：选择 TTFT p99 保持在 SLO 以内的最高并发量。如果 SLO 要求 TTFT 为 500 毫秒，32 个并发流时 TTFT p99 为 480 毫秒，而 36 个并发流时升至 720 毫秒，则应使用 32。

**优化吞吐量（批处理、离线任务）**：选择 `output_tokens_per_second` 开始趋于平台时的并发量。此场景对延迟要求较低，因此可以进一步提高负载，进入最佳区间的更高位置。

**生产环境预留余量**：将选定的并发量乘以 0.8。如果最佳区间在 40 个并发流左右，则以 32 运行。这样可以为流量峰值留出余量，避免过饱和。

## 示例：解读 Sweep 结果

以下是一次实际 Sweep 测试的结果。

测试使用单张 **NVIDIA A100 80GB** GPU，通过启用了分块预填充的 vLLM 提供 **meta-llama/Llama-3.1-8B-Instruct** 服务。工作负载为 1000 个输入 token 和 1000 个输出 token。

| 策略        | 并发量（均值） | 请求/秒（均值） | 输出 Token/秒 | TTFT p50（毫秒） | TTFT p95（毫秒） | ITL p50（毫秒） | ITL p95（毫秒） | 区域     |
| ----------- | -------------: | --------------: | ------------: | ---------------: | ---------------: | --------------: | --------------: | -------- |
| synchronous |            1.0 |             0.1 |          90.2 |             79.5 |            115.2 |            11.0 |            11.1 | 利用不足 |
| constant    |            3.8 |             0.3 |         327.7 |             95.0 |            100.1 |            11.5 |            11.6 | 利用不足 |
| constant    |            7.1 |             0.5 |         569.7 |             97.7 |            102.8 |            12.3 |            12.4 | 利用不足 |
| constant    |           10.8 |             0.8 |         809.4 |            100.4 |            105.8 |            13.3 |            13.4 | 利用不足 |
| constant    |           14.7 |             1.0 |        1047.9 |            100.4 |            108.4 |            14.0 |            14.1 | 最佳区间 |
| constant    |           21.0 |             1.2 |        1274.9 |            107.8 |            115.9 |            16.6 |            16.7 | 最佳区间 |
| constant    |           27.8 |             1.4 |        1496.5 |            117.7 |            126.6 |            19.2 |            19.3 | 最佳区间 |
| constant    |           33.7 |             1.6 |        1728.6 |            123.5 |            133.9 |            19.8 |            19.9 | 最佳区间 |
| constant    |           46.0 |             1.7 |        1911.7 |            132.6 |            146.6 |            25.0 |            25.5 | 过饱和   |
| throughput  |          509.8 |             2.1 |        3217.6 |           9947.7 |          21961.3 |            84.1 |           106.8 | 过饱和   |

在此示例中，并发量从 1 增加到约 11 时，吞吐量线性增长，服务器处于利用不足区。并发量在 15 到 34 之间时，吞吐量仍在上升，同时 TTFT 和 ITL 也逐渐增加，服务器处于最佳区间。并发量达到 46 时，吞吐量增幅趋平而延迟仍在增加；吞吐量策略中的 TTFT 更是飙升至约 10 秒，证实系统已过饱和。

本例的最佳运行点约为**并发量 28 到 34**：吞吐量接近 1500 到 1700 tok/s，TTFT p50 仍低于 135 毫秒，ITL 低于 20 毫秒。生产环境若要预留余量，建议将目标并发量设为 24 到 28。
