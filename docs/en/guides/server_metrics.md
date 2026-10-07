# Server Metrics

GuideLLM measures every request from the client side. Server metrics add the server's own view of the same run, such as how many requests it was queuing, how full its KV cache was, and how long it measured each request to take. With both in one report, you can see why a result changed, for example whether latency rose because requests started queuing or because the KV cache filled up.

GuideLLM scrapes a Prometheus `/metrics` endpoint, such as the one every vLLM server exposes, while each benchmark runs, and summarizes what it collected over that benchmark's measurement window.

## Quick Start

Point `--server-metrics` at the server's metrics endpoint:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=60 \
  --server-metrics kind=prometheus,url=http://localhost:8000/metrics
```

Each benchmark in the JSON or YAML report then has a `server_metrics` entry with one summary per source. Server metrics are not shown in the console, CSV or HTML outputs.

## Configuration

| Option     | Default             | Description                                                                                       |
| ---------- | ------------------- | ------------------------------------------------------------------------------------------------- |
| `url`      | required            | The Prometheus metrics endpoint to scrape.                                                        |
| `interval` | `1.0`               | Seconds between scrapes while a benchmark runs.                                                   |
| `timeout`  | `5.0`               | Seconds to wait for one scrape before it counts as failed.                                        |
| `metrics`  | vLLM's main metrics | Metric family names to collect. Families the endpoint does not expose are skipped without errors. |

Repeat `--server-metrics` to scrape several servers, such as each replica behind a load balancer. Scraping the load balancer's address instead would sample a different replica on each scrape.

The default `metrics` cover vLLM's queue, KV cache, token, request outcome and latency metrics. They include the names older vLLM releases used, `vllm:gpu_cache_usage_perc` and `vllm:time_per_output_token_seconds`, so the same configuration works across versions. To collect a different set, pass a JSON list:

```bash
--server-metrics '{"kind": "prometheus", "url": "http://localhost:8000/metrics", "interval": 0.5, "metrics": ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"]}'
```

## How Metrics Are Collected

Scraping runs in the benchmark's own process and does not touch the worker processes that send requests, so it does not change request timing. Each benchmark is scraped:

1. once just before its first request,
2. every `interval` seconds while it runs,
3. once more right after its last request.

A failed scrape is counted and logged, and the benchmark continues. Each summary reports `scrapes` (successful) and `scrape_errors` (failed), so a summary with `scrapes` at 0 means the endpoint could not be reached, for example because the URL is wrong or the server does not expose metrics there. Lines the parser cannot read are skipped rather than failing the scrape.

## What the Report Contains

Every metric is summarized over the benchmark's measurement window, which excludes any configured warmup and cooldown. Series are kept separately for each label set, such as `model_name` or `engine` on a data-parallel deployment.

### Gauges

Gauges such as `vllm:num_requests_running`, `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` report:

- `summary`: the mean, min, max and percentiles of the values scraped within the window.
- `samples`: each scraped `(timestamp, value)` pair within the window, for plotting the gauge over time alongside request timings.

### Counters

Counters such as `vllm:request_success`, `vllm:generation_tokens` and `vllm:num_preemptions` report:

- `increase`: how much the counter grew over the window.
- `rate`: `increase` per second.

Scrapes rarely land exactly on the window's edges, so the counter is interpolated linearly between the scrapes on either side of each edge. If the server restarts and a counter drops, the drop is treated as a reset, the same way Prometheus's `increase()` handles it.

### Histograms

Histograms such as `vllm:time_to_first_token_seconds` and `vllm:e2e_request_latency_seconds` report the observations recorded within the window:

- `count`, `sum` and `mean`.
- `quantiles`: estimates for `p50`, `p90`, `p95` and `p99`, interpolated within buckets the same way as Prometheus's `histogram_quantile()`.

> **Note:** Histogram quantiles are only as precise as the server's bucket boundaries. vLLM's latency buckets are coarse, so compare these estimates with GuideLLM's own request-level percentiles rather than reading them as exact values.

### Example

A trimmed summary from a run against a vLLM server:

```json
"server_metrics": [
  {
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

## Limitations

- **Short benchmarks:** a measurement window shorter than twice the scrape interval has few scrapes inside it, so GuideLLM logs a warning and its server metrics are approximate. Lower `interval` for short runs.
- **Shared servers:** the server's metrics include every request it served during the window, not just GuideLLM's. Compare them with the client-side results only when GuideLLM is the server's only client.
- **Text format only:** the endpoint must serve the Prometheus text exposition format, which vLLM and most Prometheus clients do by default.
