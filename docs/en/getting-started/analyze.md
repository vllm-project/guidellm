---
weight: -4
---

# Analyze Results

After [running a benchmark](benchmark.md), GuideLLM provides comprehensive results that help you understand your LLM deployment's performance. This guide explains how to interpret both console output and file-based results.

## Understanding Console Output

GuideLLM displays configuration details during setup. Upon benchmark completion, it prints a series of result tables, followed by knee detection results when available and any post-benchmark warnings.

### Initial Setup Output

Before benchmark execution, the setup output summarizes the resolved configuration, including:

- **Backend configuration**: Target URL, model name, and backend details
- **Tokenizer configuration**: Tokenizer and load arguments
- **Data configuration**: Data source, token counts, and dataset properties
- **Preprocessor configuration**: Selected preprocessors
- **Finalizer configuration**: A finalizer processes requests
- **Request loader configuration**: Show the data loader parameters
- **Profile configuration**: Show the settings of the benchmark run profile
- **Output format configuration**: Confirm the output formats and paths selected.

Example:

```
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-0.6B
  {'kind': 'openai_http', 'target': 'http://localhost:8000', 'model': 'Qwen/Qwen3-0.6B', 'request_format': '/v1/chat/completions', 'api_key': None, 'api_routes': {'/health': 'health',
  '/v1/models': 'v1/models', '/v1/completions': 'v1/completions', '/v1/chat/completions': 'v1/chat/completions', '/v1/embeddings': 'v1/embeddings', '/v1/responses': 'v1/responses', '/v1/audio/transcriptions':
  'v1/audio/transcriptions', '/v1/audio/translations': 'v1/audio/translations', '/pooling': 'pooling'}, 'timeout': None, 'timeout_connect': 5.0, 'http2': True, 'follow_redirects': True, 'verify': False, 'validate_backend': True,
  'stream': True, 'extras': None, 'max_tokens': None, 'server_history': False, 'tool_call_missing_behavior': 'error_stop', 'multiturn_reasoning': False}
✔ Tokenizer resolved, using model 'Qwen/Qwen3-0.6B' as tokenizer
  {'kind': 'huggingface_auto', 'model': 'Qwen/Qwen3-0.6B', 'load_kwargs': {}}
✔ 1 datasets resolved
  [{'kind': 'synthetic_text', 'load_kwargs': {}, 'prompt_tokens': 128, 'prompt_tokens_stdev': 4, 'prompt_tokens_min': None, 'prompt_tokens_max': None, 'output_tokens': 128, 'output_tokens_stdev': None, 'output_tokens_min': None,
  'output_tokens_max': None, 'first_prompt_tokens': None, 'first_prompt_tokens_stdev': None, 'first_prompt_tokens_min': None, 'first_prompt_tokens_max': None, 'first_output_tokens': None, 'first_output_tokens_stdev': None,
  'first_output_tokens_min': None, 'first_output_tokens_max': None, 'delay': None, 'delay_stdev': None, 'delay_min': None, 'delay_max': None, 'turns': 1, 'tool_call_turns': [], 'tools': None, 'tool_response_tokens': None,
  'tool_response_tokens_stdev': None, 'tool_response_tokens_min': None, 'tool_response_tokens_max': None, 'server_tool_call_turns': [], 'branches': [], 'prefix_buckets': None}]
✔ 2 preprocessors resolved
  [{'kind': 'generative_column_mapper', 'column_mappings': None}, {'kind': 'encode_media', 'audio_kwargs': {}, 'image_kwargs': {}, 'video_kwargs': {}}]
✔ Finalizer resolved
  {'kind': 'generative', 'tool_call_mode': 'client'}
✔ Request loader resolved with inf unique requests
  {'kind': 'pytorch', 'samples': -1, 'shuffle': False, 'num_workers': 1, 'prefetch_factor': 4096}
✔ ConcurrentProfile profile resolved
  {'kind': 'concurrent', 'rampup_duration': 0.0, 'prefetch': 0, 'warmup': {'percent': None, 'value': None, 'mode': 'prefer_duration'}, 'cooldown': {'percent': None, 'value': None, 'mode': 'prefer_duration'}, 'streams': [20]}
✔ Output formats resolved
  [{'kind': 'json', 'path': '/home/dbutenho/ai/cleanup/benchmarks.json'}, {'kind': 'csv', 'path': '/home/dbutenho/ai/cleanup/benchmarks.csv'}]
```

### Run Summary Info

The `Run Summary Info` table has one row per benchmark (strategy) and includes:

- **Benchmark**: The strategy that ran, such as `synchronous`, `throughput`, or `constant@10.00`
- **Timings**: Measurement start and end timestamps and duration, excluding warmup and cooldown; warmup and cooldown durations are reported separately
- **Requests**: Totals for completed, incomplete, and errored requests
- **Input Tokens / Output Tokens**: Token totals broken down by request status

### Modality Statistics

When the run includes text, image, video, audio, or tool-call data, GuideLLM prints a statistics table for each modality that has data (for example, `Text Metrics Statistics (Completed Requests)`). These tables report per-request values and per-second rates for modality-specific counts such as tokens, words, characters, image pixels and bytes, video frames and seconds, audio samples and seconds, and tool calls.

### Request Token Statistics

`Request Token Statistics (Completed Requests)` reports per-request input, output, and total token counts, plus streaming iterations and output tokens per streaming iteration.

### Request Latency Statistics

`Request Latency Statistics (Completed Requests)` reports the distribution of request latency, time to first token (TTFT), time to first output token (TTFOT), inter-token latency (ITL), and time per output token (TPOT). The table shows the mean, median, and p95. Request latency, TTFT, and TTFOT include a mean margin of error when available; ITL and TPOT do not. When confidence intervals are enabled, a marker identifies p95 estimates whose interval cannot be bounded by the sample. The console does not display percentile interval endpoints. JSON and CSV reports contain the available confidence intervals and additional percentiles, including p99.

The p99 (99th percentile) values are particularly important for SLO analysis, as they represent the worst-case performance for 99% of requests.

### Server Throughput Statistics

`Server Throughput Statistics (All Requests)` reports request concurrency, requests per second, and input, output, and total tokens per second. When SLO objectives are configured with `--metrics`, this table also reports SLO attainment and goodput, and a separate `Per-Objective SLO Attainment` table reports attainment for each configured objective.

The console tables summarize every strategy in the run. Profile conclusions, including knee detection and goodput search results, are stored in the report's `conclusions` list. The console prints knee detection results after the tables; goodput search conclusions are available in the saved JSON and YAML reports. Any triggered warning rules appear in a final Warnings section.

JSON and YAML reports contain per-request records and metric summaries. CSV contains one row per benchmark with aggregate statistics and percentile summaries. `metrics.sample_size` controls retention of full request payloads; lightweight per-request statistics remain available in JSON and YAML regardless of that setting.

## Analyzing Saved Results

For deeper analysis, GuideLLM saves detailed results to these files by default:

- `benchmarks.json`: Complete benchmark data in JSON format
- `benchmarks.csv`: Summary of key metrics in CSV format

The files are written to the directory configured by `GUIDELLM__DEFAULT_RESULTS_DIR`, or to the current directory when the variable is not set. See [configuring file outputs](../guides/outputs.md#configuring-file-outputs) to choose filenames and destinations.

### File Formats

See [supported file formats](../guides/outputs.md#supported-file-formats) for the available reports and [output configuration](../guides/outputs.md#cli-output-configuration) for examples of selecting them. Explicit `--output` options replace the default JSON and CSV selection. Console display controls are described in [console output](../guides/outputs.md#console-output).

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
    print(
        f"p99 latency: {benchmark.metrics.request_latency.successful.percentiles.p99}"
    )
    print(
        f"TTFT (p99): {benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99}"
    )
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
guidellm run \
  --backend kind=openai_http,target=http://server1:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=60 \
  --output kind=json,path=model1/benchmarks.json

guidellm run \
  --backend kind=openai_http,target=http://server2:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=60 \
  --output kind=json,path=model2/benchmarks.json
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
