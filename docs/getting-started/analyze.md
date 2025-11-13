---
weight: -4
---

# Analyze Results

After [running a benchmark](benchmark.md), GuideLLM provides comprehensive results that help you understand your LLM deployment's performance. This guide explains how to interpret both console output and file-based results.

## Understanding Console Output

Upon benchmark completion, GuideLLM automatically displays results in the console, divided into three main sections:

### 1. Benchmarks Metadata

This section provides a high-level summary of the benchmark run, including:

- **Server configuration**: Target URL, model name, and backend details
- **Data configuration**: Data source, token counts, and dataset properties
- **Profile arguments**: Rate type, maximum duration, request limits, etc.
- **Extras**: Any additional metadata provided via the `--output-extras` argument

Example:

```
Benchmarks Metadata
------------------
Args:        {"backend_type": "openai", "target": "http://localhost:8000", "model": "Meta-Llama-3.1-8B-Instruct-quantized", ...}
Worker:      {"type_": "generative", "backend_type": "openai", "backend_args": {"timeout": 120.0, ...}, ...}
Request Loader: {"type_": "generative", "data_args": {"prompt_tokens": 256, "output_tokens": 128, ...}, ...}
Extras:      {}
```

### 2. Benchmarks Info

This section summarizes the key information about each benchmark run, presented as a table with columns:

- **Type**: The benchmark type (e.g., synchronous, constant, poisson, etc.)
- **Start/End Time**: When the benchmark started and ended
- **Duration**: Total duration of the benchmark in seconds
- **Requests**: Count of successful, incomplete, and errored requests
- **Token Stats**: Average token counts and totals for prompts and outputs

This section helps you understand what was executed and provides a quick overview of the results.

### 3. Benchmarks Stats

This is the most critical section for performance analysis. It displays detailed statistics for each metric:

- **Throughput Metrics**:

  - Requests per second (RPS)
  - Request concurrency
  - Output tokens per second
  - Total tokens per second

- **Latency Metrics**:

  - Request latency (mean, median, p99)
  - Time to first token (TTFT) (mean, median, p99)
  - Inter-token latency (ITL) (mean, median, p99)
  - Time per output token (mean, median, p99)

The p99 (99th percentile) values are particularly important for SLO analysis, as they represent the worst-case performance for 99% of requests.

## Analyzing the Results File

For deeper analysis, GuideLLM saves detailed results to a file (default: `benchmarks.json`). This file contains all metrics with more comprehensive statistics and individual request data.

### File Formats

GuideLLM supports multiple output formats:

- **JSON**: Complete benchmark data in JSON format (default)
- **YAML**: Complete benchmark data in human-readable YAML format
- **CSV**: Summary of key metrics in CSV format

To specify the format, use the `--output-path` argument with the appropriate extension:

```bash
guidellm benchmark --target "http://localhost:8000" --output-path results.yaml
```

### Programmatic Analysis

For custom analysis, you can reload the results into Python:

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

## Key Performance Indicators

When analyzing your results, focus on these key indicators:

### 1. Throughput and Capacity

- **Maximum RPS**: What's the highest request rate your server can handle?
- **Concurrency**: How many concurrent requests can your server process?
- **Token Throughput**: How many tokens per second can your server generate?

### 2. Latency and Responsiveness

- **Time to First Token (TTFT)**: How quickly does the model start generating output?
- **Inter-Token Latency (ITL)**: How smoothly does the model generate subsequent tokens?
- **Total Request Latency**: How long do complete requests take end-to-end?

### 3. Reliability and Error Rates

- **Success Rate**: What percentage of requests completes successfully?
- **Error Distribution**: What types of errors occur and at what rates?

## Additional Analysis Techniques

### Comparing Different Models or Hardware

Run benchmarks with different models or hardware configurations, then compare:

```bash
guidellm benchmark --target "http://server1:8000" --output-path model1.json
guidellm benchmark --target "http://server2:8000" --output-path model2.json
```

### Cost Optimization

Calculate cost-effectiveness by analyzing:

- Tokens per second per dollar of hardware cost
- Maximum throughput for different hardware configurations
- Optimal batch size vs. latency tradeoffs

### Determining Scaling Requirements

Use your benchmark results to plan:

- How many servers you need to handle your expected load
- When to automatically scale up or down based on demand
- What hardware provides the best price/performance for your workload
