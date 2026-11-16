---
weight: 3
source_sha256: !!str "336cf7e8d307c68bcb79b48a1d402b7e4fb23e14ed681d1ee5c47609eb252b4d"
---

# 数据采集器

数据采集器会将 GuideLLM 之外的数据加入基准测试报告。内置的 `prometheus` 采集器会在基准测试运行期间抓取服务器的 Prometheus 端点，因此你可以将服务器的队列长度、KV 缓存使用量和延迟，与 GuideLLM 自身的测量结果一并查看。例如，它可以帮助判断延迟上升是因为请求排队，还是因为 KV 缓存已满。

## 快速开始

使用服务器的指标端点传入 `--collector`：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=60 \
  --collector kind=prometheus,url=http://localhost:8000/metrics
```

JSON 或 YAML 报告中的每项基准测试都会包含一个 `collectors` 列表，列表中的结果与传入采集器的顺序一致，每个采集器对应一项结果（`jq '.benchmarks[].collectors' benchmarks.json`）。每项结果都包含 `kind` 字段，用于标识生成该结果的采集器。控制台、CSV 和 HTML 输出不会报告采集器结果。

## 采集器的运行方式

所有采集器共用一个独立进程，因此不会占用发送请求的进程的时间。每项基准测试的请求开始和结束时，该进程都会收到通知；随后，系统会要求它返回每个采集器在基准测试测量时间窗口内的结果（不包括预热和冷却阶段）。

如果采集器失败，例如其端点不可用，基准测试仍会继续运行。该采集器会改为报告其 `kind` 和 `error`，因此列表仍会与传入的采集器一一对应。

## Prometheus 采集器

### 选项

| 选项       | 默认值          | 说明                                               |
| ---------- | --------------- | -------------------------------------------------- |
| `url`      | 必填            | 要抓取的 Prometheus 指标端点。                     |
| `interval` | `1.0`           | 基准测试运行期间两次抓取之间的秒数。               |
| `timeout`  | `5.0`           | 将一次抓取计为失败前的等待秒数。                   |
| `metrics`  | vLLM 的主要指标 | 要保留的指标族。若端点未提供某个指标族，则会跳过。 |

`url` 与后端目标地址分开设置，因为后端可能指向网关或负载均衡器。要覆盖多台服务器，请为每台服务器分别传入一次 `--collector`。如果将其指向负载均衡器，每次抓取可能会命中不同的副本。

默认的 `metrics` 包含 vLLM 的队列、KV 缓存、token、请求结果和延迟指标，也包括旧版 vLLM 中的 `vllm:gpu_cache_usage_perc` 和 `vllm:time_per_output_token_seconds`。若要自定义指标，请以 JSON 格式传入采集器：

```bash
--collector '{"kind": "prometheus", "url": "http://localhost:8000/metrics", "interval": 0.5, "metrics": ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"]}'
```

### 抓取

系统会在基准测试的第一个请求发出前抓取端点，在运行期间每隔 `interval` 秒抓取一次，并在最后一个请求结束后再抓取一次。抓取失败时会记录日志并计入 `scrape_errors`，基准测试仍会继续。如果 `scrapes` 为 0，说明从未成功访问该端点，通常是 URL 有误。无法解析的行会被跳过。

### 结果

指标会在测量时间窗口内汇总，并按标签集分别生成序列（例如 `model_name`，或数据并行部署中的 `engine`）。

- **Gauge（瞬时值）**，例如 `vllm:num_requests_running` 和 `vllm:kv_cache_usage_perc`，会报告 `summary`（均值、最小值、最大值和百分位数）以及原始 `samples`，格式为 `(timestamp, value)` 数值对。
- **Counter（计数器）**，例如 `vllm:request_success` 和 `vllm:generation_tokens`，会报告该时间窗口内的 `increase`，以及每秒的 `rate`。抓取时间很少恰好落在窗口边界上，因此边界处的值会通过插值计算；计数值下降则视为计数器重置，处理方式与 Prometheus 的 `increase()` 类似。
- **Histogram（直方图）**，例如 `vllm:time_to_first_token_seconds`，会报告 `count`、`sum`、`mean` 和 `quantiles`（`p50`、`p90`、`p95`、`p99`），并像 Prometheus 的 `histogram_quantile()` 一样在桶之间插值。vLLM 的延迟桶粒度较粗，因此这些结果仅是近似值；精确数值请使用 GuideLLM 自身的百分位数。

结果示例如下（内容已缩短）：

```json
"collectors": [
  {
    "kind": "prometheus",
    "source": "http://localhost:8000/metrics",
    "scrapes": 62,
    "scrape_errors": 0,
    "gauges": {
      "vllm:num_requests_waiting": [
        {
          "labels": {"engine": "0", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "summary": {"mean": 12.4, "max": 31.0, "...": "..."},
          "samples": [[1760000000.5, 0.0], [1760000001.5, 4.0], "..."]
        }
      ]
    },
    "counters": {
      "vllm:request_success": [
        {
          "labels": {"engine": "0", "finished_reason": "length", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "increase": 584.0,
          "rate": 9.73
        }
      ]
    },
    "histograms": {
      "vllm:time_to_first_token_seconds": [
        {
          "labels": {"engine": "0", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
          "count": 584.0,
          "sum": 146.2,
          "mean": 0.25,
          "quantiles": {"p50": 0.21, "p90": 0.42, "p95": 0.61, "p99": 0.93}
        }
      ]
    }
  }
]
```

### 限制

- 如果测量时间窗口短于两倍抓取间隔，窗口内只有少量抓取，因此 GuideLLM 会记录警告。运行时间较短时，请调低 `interval`。
- 服务器指标会统计它处理的所有请求，而不只是 GuideLLM 发出的请求。因此，只有在没有其他流量使用该服务器时，服务器指标才会与客户端测量结果吻合。
- 端点必须提供 Prometheus 文本格式；vLLM 默认使用该格式。
