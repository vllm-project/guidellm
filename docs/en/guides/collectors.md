# Collectors

Collectors add data from outside GuideLLM to the benchmark report. The built-in `prometheus` collector scrapes a server's Prometheus endpoint while each benchmark runs, so you can read the server's queue length, KV cache usage and latencies next to GuideLLM's own measurements. For example, it can tell you whether latency went up because requests were queuing or because the KV cache was full.

## Quick Start

Pass `--collector` with the server's metrics endpoint:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_duration,seconds=60 \
  --collector kind=prometheus,url=http://localhost:8000/metrics
```

Each benchmark in the JSON or YAML report gets a `collectors` list with one result per collector, in the order they were passed (`jq '.benchmarks[].collectors' benchmarks.json`). Each result contains a `kind` field naming the collector that produced it. The console, CSV and HTML outputs don't report results from collectors.

## How Collectors Run

All collectors share one separate process, so they don't take time away from the process that sends requests. That process is told when each benchmark's requests start and end, and then asked for each collector's result over the benchmark's measurement window (warmup and cooldown are excluded).

If a collector fails, for example because its endpoint is down, the benchmark keeps going and the collector reports its `kind` and an `error` instead, so the list still matches the collectors you passed.

## Prometheus Collector

### Options

| Option     | Default             | Description                                                                |
| ---------- | ------------------- | -------------------------------------------------------------------------- |
| `url`      | required            | Prometheus metrics endpoint to scrape.                                     |
| `interval` | `1.0`               | Seconds between scrapes while a benchmark runs.                            |
| `timeout`  | `5.0`               | Seconds to wait for a scrape before counting it as failed.                 |
| `metrics`  | vLLM's main metrics | Metric families to keep. Families the endpoint doesn't expose are skipped. |

The `url` is separate from the backend target because the backend may point at a gateway or load balancer. To cover several servers, pass `--collector` once per server. Pointing it at a load balancer would hit a different replica on each scrape.

The default `metrics` cover vLLM's queue, KV cache, token, request outcome and latency metrics, including `vllm:gpu_cache_usage_perc` and `vllm:time_per_output_token_seconds` from older vLLM releases. To pick your own, pass the collector as JSON:

```bash
--collector '{"kind": "prometheus", "url": "http://localhost:8000/metrics", "interval": 0.5, "metrics": ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"]}'
```

### Scraping

The endpoint is scraped right before a benchmark's first request, every `interval` seconds while it runs, and once more after its last request. A failed scrape is logged and counted in `scrape_errors`, and the benchmark continues. If `scrapes` is 0, the endpoint was never reached, which usually means the URL is wrong. Lines that can't be parsed are skipped.

### Result

Metrics are summarized over the measurement window, with a separate series for each label set (for example `model_name`, or `engine` on a data-parallel deployment).

- **Gauges** such as `vllm:num_requests_running` and `vllm:kv_cache_usage_perc` report a `summary` (mean, min, max and percentiles) and the raw `samples` as `(timestamp, value)` pairs.
- **Counters** such as `vllm:request_success` and `vllm:generation_tokens` report their `increase` over the window and the `rate` per second. Scrapes rarely land exactly on the window's edges, so the value at each edge is interpolated, and a drop is treated as a counter reset, like Prometheus's `increase()`.
- **Histograms** such as `vllm:time_to_first_token_seconds` report `count`, `sum`, `mean` and `quantiles` (`p50`, `p90`, `p95`, `p99`), interpolated within buckets like Prometheus's `histogram_quantile()`. vLLM's latency buckets are coarse, so treat these as rough estimates and use GuideLLM's own percentiles for exact values.

The result looks like this (shortened):

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

### Limitations

- A measurement window shorter than two scrape intervals only has a few scrapes in it, so GuideLLM logs a warning. Lower `interval` for short runs.
- The server's metrics count every request it handled, not only GuideLLM's, so they only line up with the client-side results when nothing else is using the server.
- The endpoint has to serve the Prometheus text format, which vLLM does by default.

## vLLM Server Info Collector

Use `vllm_server_info` to record the structured configuration of a specific vLLM server:

```bash
--collector kind=vllm_server_info,url=http://localhost:8000/server_info
```

The collector requests `config_format=json` once after each sub-benchmark's requests finish, in the shared collector process. It does not make a startup snapshot or sample configuration changes during the run. Each result is stored in that sub-benchmark's `collectors` list:

```json
{
  "kind": "vllm_server_info",
  "source": "http://localhost:8000/server_info",
  "collected_at": 1760000000.5,
  "vllm_config": {
    "model_config": {"model": "example/model", "max_model_len": 4096},
    "parallel_config": {"tensor_parallel_size": 2}
  }
}
```

`collected_at` is the Unix timestamp when the capture request begins. Only the structured `vllm_config` object is retained. `vllm_env`, `system_env` and legacy string configurations are not captured.

| Option         | Default  | Description                                                                                                                 |
| -------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `url`          | required | Full HTTP(S) server-info endpoint, independent of the backend target.                                                       |
| `redact_hosts` | `true`   | Filter remote host fields and URL authorities in captured data, including `source`. Credential filtering is always enabled. |

Pass one collector per server, pointing directly at each replica. Results retain collector order even when hostnames are filtered. To retain hosts for comparison, use `redact_hosts=false`. Canonical local hosts (`localhost`, `127.0.0.1`, `::1`) are preserved either way.

The collector filters recognized credential fields, URL user information and sensitive assignments or flags. Free-form text containing a recognized credential is omitted as a whole. This is best-effort filtering, not a guarantee that arbitrary configuration strings contain no secrets. The host option affects collected data, not the endpoint URL in the report's run arguments. Review the report before sharing it.

Each capture has a five-second total timeout and a 1 MiB decoded response limit. Redirects are not followed. An unavailable, unauthorized or unsupported endpoint produces an `error` instead of `vllm_config`; subsequent benchmarks still run and retry the capture. Errors contain only the exception type, never the response body or exception message.

The endpoint must allow this metadata request without authentication. Backend credentials are not inherited, and collector URLs cannot contain user information, query parameters or fragments. The collector adds the required `config_format=json` parameter itself. Protected endpoints report an error; support for explicit collector authentication can be added separately.
