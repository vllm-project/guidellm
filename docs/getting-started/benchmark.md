---
weight: -6
---

# Run a Benchmark

1. [Install GuideLLM](install.md)
2. You can run GuideLLM two ways:
   1. Targeting a running OpenAI-compatible LLM server
      - The most common setup.
   2. Using the vLLM Python backend, with vLLM running in the same process
      - Requires knowledge on how to setup vLLM in addition to the knowledge on how to run GuideLLM.
      - Simplifies orchestration due to the lack of need for a separate server.

> [!NOTE]\
> Everything in this guide applies to both backends except the backend-specific inputs.
>
> This guide assumes you're using the OpenAI HTTP backend with an OpenAI-compatible LLM server. For information on using the vLLM Python backend see [vLLM Python backend](../guides/vllm-python-backend.md)

After [starting a server](server.md), you're ready to run benchmarks to evaluate your LLM deployment's performance.

Running a GuideLLM benchmark is straightforward. The basic command structure is:

```bash
guidellm benchmark --target <server-url> [options]
```

### Basic Example

To run a benchmark against your local vLLM server with default settings:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=256,output_tokens=128" \
  --max-seconds 60
```

This command:

- Connects to your vLLM server running at `http://localhost:8000`
- Uses synthetic data with 256 prompt tokens and 128 output tokens per request
- Automatically determines the available model on the server
- Runs a "sweep" profile (default) to find optimal performance points

During the benchmark, you'll see a progress display similar to this:

![Benchmark Progress](../assets/sample-benchmarks.gif)

Learn more about dataset options in the [Datasets documentation](../guides/datasets.md) and backend configuration in the [Backends documentation](../guides/backends.md).

## Understanding Benchmark Options

GuideLLM offers a wide range of configuration options to customize your benchmarks. Here are the most important parameters you should know:

### Key Parameters

| Parameter              | Description                                    | Example                                                            |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| `--target`             | URL of the OpenAI-compatible server            | `--target "http://localhost:8000"`                                 |
| `--model`              | Model name to benchmark                        | `--model "Meta-Llama-3.1-8B-Instruct"`                             |
| `--data`               | Data configuration for benchmarking            | `--data "kind=synthetic_text,prompt_tokens=256,output_tokens=128"` |
| `--profile`            | Type of benchmark profile to run               | `--profile kind=sweep`                                             |
| `--rate`               | Request rate or number of benchmarks for sweep | `--rate 10`                                                        |
| `--images-per-request` | Number of images per request for vision benchmarks | `--images-per-request "1,2,5"`                                 |
| `--random-seed`        | Random seed for reproducibility                | `--random-seed 42`                                                 |
| `--max-seconds`        | Duration for each benchmark in seconds         | `--max-seconds 30`                                                 |
| `--max-requests`       | Maximum number of requests for each benchmark  | `--max-requests 1000`                                              |
| `--data-samples`       | Maximum number of dataset rows to load         | `--data-samples 1000`                                              |
| `--output-dir`         | Directory path to save output files            | `--output-dir results/`                                            |
| `--outputs`            | Output formats to generate                     | `--outputs json csv html`                                          |

### Random Seed (`--random-seed`)

The random seed is used for any operation in GuideLLM that involves randomness such as synthetic data generation or poisson strategy scheduling calculations. By default it is a fixed value, so that rerunning GuideLLM with the same arguments should produce the same results.

### Strategy Constraints

The strategy constraints, `--max-requests` and `--max-seconds`, apply individually to each strategy in a profile. Profiles with multiple strategies include `sweep` and any profile with `--rate` set to a list of values.

For example, setting `--max-requests 1000` with `--profile kind=sweep` will run 1000 synchronous requests, 1000 throughput requests, and 1000 `constant` requests at each interpolated rate, while `--max-seconds 30` will run each strategy for 30 seconds. Similarly, using `--max-seconds 30` with `--profile kind=concurrent --rate 10,20` will run 10 concurrent streams for 30 seconds and then 20 concurrent streams for 30 seconds.

### Benchmark Profiles (`--profile`)

GuideLLM supports several benchmark profiles and strategies, which are described in detail below.

#### Synchronous Profile

Runs requests one at a time (sequential).

```bash
guidellm benchmark --profile kind=synchronous
```

| Strategy parameter | Profile parameter | Description  | Example |
| ------------------ | ----------------- | ------------ | ------- |
| `--rate`           | —                 | Not allowed. |         |

#### Throughput Profile

Attempts to discover the server's maximum throughput by continually making requests in parallel.

```bash
guidellm benchmark --profile kind=throughput,max_concurrency=10
```

| Strategy parameter | Profile parameter | Description                                             | Example                                                         |
| ------------------ | ----------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| `--rate`           | `max_concurrency` | Number of concurrent request streams.                   | `--rate 10` or `--profile kind=throughput,max_concurrency=10`   |
| `--rampup`         | `rampup_duration` | Number of seconds to ramp up to the maximum throughput. | `--rampup 10` or `--profile kind=throughput,rampup_duration=10` |

#### Concurrent Profile

Runs a fixed number of parallel request streams.

```bash
guidellm benchmark --profile kind=concurrent,streams=10
```

| Strategy parameter | Profile parameter | Description                                                                                                                | Example                                                                  |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `--rate`           | `streams`         | Number of concurrent requests to maintain. May be a single value or a list of values to run successive concurrent streams. | `--rate 10`, `--profile kind=concurrent,streams=16,32`                   |
| `--rampup`         | `rampup_duration` | Duration in seconds to spread initial requests up to target rate.                                                          | `--rampup 10`, `--profile kind=concurrent,streams=10,rampup_duration=10` |
|                    | `max_concurrency` | Maximum concurrent requests to schedule.                                                                                   | `--profile kind=concurrent,streams=10,max_concurrency=10`                |

#### Constant Profile

Sends asynchronous requests at a fixed rate per second.

(The profile name `async` is an alias for the `constant` profile.)

```bash
guidellm benchmark --profile '{"kind": "constant", "rate": [16,32]}'
```

| Strategy parameter | Profile parameter | Description                                                                                        | Example                                                                                 |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `--rate`           | `rate`            | Number of requests to send per second. May be a list of values to run successive constant streams. | `--rate 10`, `--profile '{"kind": "constant","rate": [16,32]}'  `                       |
| `--rampup`         | `rampup_duration` | Duration in seconds to linearly ramp up from 0 to target rate.                                     | `--rampup 10`, `--profile '{"kind": "constant","rate": [16,32],"rampup_duration": 10}'` |

#### Poisson Profile

Sends asynchronous requests at varying rates using a Poisson distribution around the specified target rate(s). This probabilistic pattern is useful for simulating more realistic real-world traffic patterns.

```bash
guidellm benchmark --profile kind=poisson,rate=16,random_seed=42
```

| Strategy parameter | Profile parameter | Description                                                                                      | Example                                                     |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `--rate`           | `rate`            | Maximum asynchronous requests to run. May be a list of values to run successive Poisson streams. | `--rate 10`, `--profile kind=poisson,rate=16`               |
| `--random-seed`    | `random_seed`     | Random seed for the Poisson distribution.                                                        | `--random-seed 42`, `--profile kind=poisson,random_seed=42` |

#### Sweep Profile

The sweep profile applies a sequence of benchmark strategies to find the optimal performance points for the given model and data.

1. It runs a `synchronous` strategy to measure the baseline rate,
2. then runs a `throughput` strategy to determine peak throughput,
3. and finally runs a series of asynchronous strategies with rates interpolated between the baseline and maximum throughput. (The number of interpolated strategies is `sweep_size` (or `rate`) minus 2.) The asynchronous strategy type is determined by the `strategy_type` profile parameter. The default strategy type is `constant`.

For example, to run a sweep with 10 strategies, 10 seconds of rampup, and a strategy type of `poisson`:

```bash
guidellm benchmark --profile kind=sweep,sweep_size=10,rampup_duration=10,strategy_type=poisson
```

| Strategy parameter | Profile parameter | Description                                                                      | Example                                                                 |
| ------------------ | ----------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--rate`           | `sweep_size`      | Number of strategies to run in the sweep (including synchronous and throughput). | `--rate 10`, `--profile kind=sweep,sweep_size=10`                       |
| `--rampup`         | `rampup_duration` | Rate rampup duration in seconds for throughput and constant strategy steps.      | `--rampup 10`, `--profile kind=sweep,sweep_size=10,rampup_duration=10`  |
| `--strategy-type`  | `strategy_type`   | Strategy type to use for the interpolated strategies.                            | `--strategy-type poisson`, `--profile kind=sweep,strategy_type=poisson` |
| `--random-seed`    | `random_seed`     | Random seed for the Poisson distribution.                                        | `--random-seed 42`, `--profile kind=sweep,random_seed=42`               |
|                    | `max_concurrency` | Maximum concurrent requests to schedule.                                         | `--profile kind=sweep,max_concurrency=10`                               |

#### Replay Profile

Replays trace events using timestamps from a `trace_synthetic` dataset. See [Trace Replay Benchmarking](#trace-replay-benchmarking-beta) below for data setup.

```bash
guidellm benchmark --profile kind=replay
```

| Strategy parameter | Profile parameter | Description                                                              | Example                                              |
| ------------------ | ----------------- | ------------------------------------------------------------------------ | ---------------------------------------------------- |
| `--rate`           | `time_scale`      | Time scale for intervals between trace events (not requests per second). | `--rate 1.0`, `--profile kind=replay,time_scale=2.0` |

## Data Options

### Synthetic Data Options

For synthetic data, pass `kind=synthetic_text` with the desired parameters to the `--data` argument. Some key options include:

- `prompt_tokens`: Average number of tokens for prompts
- `output_tokens`: Average number of tokens for outputs

For example, to benchmark with a prompt length of 100 tokens and an output length of 50 tokens:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=synthetic_text,prompt_tokens=100,output_tokens=50" \
  --profile kind=constant \
  --rate 5
```

You can customize synthetic data generation with additional parameters such as standard deviation, minimum, and maximum values. See the [Datasets Synthetic data documentation](../guides/datasets.md#synthetic-data) for more details.

### Trace Replay Benchmarking (beta)

For realistic load testing, replay trace events using each row's timestamp and token lengths. Trace files must be JSONL and are loaded with the `trace_synthetic` data type. By default, each row uses `timestamp`, `input_length`, and `output_length` fields. Timestamps may be absolute or monotonic values; GuideLLM sorts them and converts them to offsets from the first event before scheduling:

```json
{"timestamp": 1234500.0, "input_length": 256, "output_length": 128}
{"timestamp": 1234500.5, "input_length": 512, "output_length": 64}
```

In this example, the second request is scheduled 0.5 seconds after the first request.

Run with the `replay` profile:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=trace_synthetic,path=path/to/trace.jsonl" \
  --profile kind=replay \
  --rate 1.0
```

The `--rate` parameter (profile field `time_scale`) acts as a time scale for the intervals between trace events, not requests per second: `1.0` preserves the original timing, `2.0` doubles the intervals and runs twice as long, and `0.5` halves the intervals and runs twice as fast.

GuideLLM orders trace rows by timestamp before scheduling and payload generation, so each scheduled event uses the token lengths from the same sorted row. Use `--data-samples` to limit how many trace rows are loaded and replayed. `--max-requests` remains a runtime completion constraint; it does not truncate the trace dataset.

If your trace uses different column names, include `timestamp_column`, `prompt_tokens_column`, and `output_tokens_column` directly in the `--data` argument:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=trace_synthetic,path=replay.jsonl,timestamp_column=timestamp,prompt_tokens_column=input_length,output_tokens_column=output_length" \
  --profile kind=replay \
  --rate 1.0
```

For very small prompts (roughly under 15 tokens, depending on the tokenizer), GuideLLM may not have enough room to include the full per-row unique prefix. Different rows can then produce similar or identical prompts, which reduces cache resistance in replay benchmarks.

### Working with Real Data

While synthetic data is convenient for quick tests, you can benchmark with real-world data:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=json_file,path=/path/to/your/dataset.json" \
  --profile kind=constant \
  --rate 5
```

You can also use datasets from HuggingFace:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "kind=huggingface,source=garage-bAInd/Open-Platypus" \
  --profile kind=constant \
  --rate 5
```

### Multi-Image Benchmarking

When benchmarking vision-language models with multiple images per request, use `--images-per-request` to measure latency impact. This is useful for understanding how TTFT and ITL scale with increasing frame/image counts:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "prompt_tokens=256,output_tokens=128" \
  --images-per-request 1,2,5 \
  --profile constant \
  --rate 10 \
  --max-seconds 30
```

This runs three sequential benchmarks (1, 2, and 5 images per request) with synthetic 720p images and outputs comparative latency metrics in the report.

**Single image count:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --images-per-request 3 \
  --profile constant \
  --rate 5
```

**Programmatic usage:**

```python
from guidellm.benchmark import MultiImageBenchmark

# Create multi-image benchmark configuration
bench = MultiImageBenchmark(
    image_counts=[1, 2, 5],
    prompt_tokens=256,
    output_tokens=128,
)

# Get configs for each image count
configs = bench.get_configs()  # {1: config, 2: config, 5: config}

# Get image statistics
for img_count in [1, 2, 5]:
    stats = bench.get_image_stats(img_count)
    print(f"{img_count} images: {stats['total_bytes']} bytes total")
```

**Note:** Multi-image benchmarking requires the vision dependencies (`pip install guidellm[vision]`).

## Output Options

By default, complete results are saved to `benchmarks.json`, `benchmarks.csv`, and `benchmarks.html` in your current directory. Use the `--output-dir` parameter to specify a different location and `--outputs` to control which formats are generated.

Learn more about output options in the [Outputs documentation](../guides/outputs.md).

## Authentication

When benchmarking against servers that require authentication (such as OpenAI's API), you'll need to provide an API key via the `--backend-kwargs` parameter. See the [API Key Configuration](../guides/backends.md#api-key-configuration) section in the Backends documentation for details.
