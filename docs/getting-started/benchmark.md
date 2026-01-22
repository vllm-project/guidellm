---
weight: -6
---

# Run a Benchmark

After [installing GuideLLM](install.md) and [starting a server](server.md), you're ready to run benchmarks to evaluate your LLM deployment's performance.

Running a GuideLLM benchmark is straightforward. The basic command structure is:

```bash
guidellm benchmark --target <server-url> [options]
```

### Basic Example

To run a benchmark against your local vLLM server with default settings:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "prompt_tokens=256,output_tokens=128" \
  --max-seconds 60
```

This command:

- Connects to your vLLM server running at `http://localhost:8000`
- Uses synthetic data with 256 prompt tokens and 128 output tokens per request
- Automatically determines the available model on the server
- Runs a "sweep" profile (default) to find optimal performance points

During the benchmark, you'll see a progress display similar to this:

![Benchmark Progress](../assets/sample-benchmarks.gif)

## Understanding Benchmark Options

GuideLLM offers a wide range of configuration options to customize your benchmarks. Here are the most important parameters you should know:

### Key Parameters

| Parameter       | Description                                    | Example                                        |
| --------------- | ---------------------------------------------- | ---------------------------------------------- |
| `--target`      | URL of the OpenAI-compatible server            | `--target "http://localhost:8000"`             |
| `--model`       | Model name to benchmark (optional)             | `--model "Meta-Llama-3.1-8B-Instruct"`         |
| `--data`        | Data configuration for benchmarking            | `--data "prompt_tokens=256,output_tokens=128"` |
| `--profile`     | Type of benchmark profile to run               | `--profile sweep`                              |
| `--rate`        | Request rate or number of benchmarks for sweep | `--rate 10`                                    |
| `--max-seconds` | Duration for each benchmark in seconds         | `--max-seconds 30`                             |
| `--output-dir`  | Directory path to save output files            | `--output-dir results/`                        |
| `--outputs`     | Output formats to generate                     | `--outputs json csv html`                      |

### Benchmark Profiles (`--profile`)

GuideLLM supports several benchmark profiles and strategies:

- `synchronous`: Runs requests one at a time (sequential)
- `throughput`: Tests maximum throughput by running requests in parallel
- `concurrent`: Runs a fixed number of parallel request streams
- `constant`: Sends requests at a fixed rate per second
- `poisson`: Sends requests following a Poisson distribution
- `sweep`: Automatically determines optimal performance points (default)

#### Sweep Profile Configuration

The sweep profile includes advanced configuration options for optimizing benchmarks on CPU-based deployments. These parameters help manage saturation detection and prevent graph artifacts:

**Available Parameters:**

| Parameter                       | Description                                           | Default | Environment Variable                     |
| ------------------------------- | ----------------------------------------------------- | ------- | ---------------------------------------- |
| `--exclude-throughput-target`   | Stop constant-rate tests before throughput level      | `false` | `GUIDELLM__EXCLUDE_THROUGHPUT_TARGET`    |
| `--exclude-throughput-result`   | Exclude throughput benchmark from saved results       | `false` | `GUIDELLM__EXCLUDE_THROUGHPUT_RESULT`    |
| `--saturation-threshold`        | Efficiency threshold for stopping sweep (0.0-1.0)     | `0.98`  | `GUIDELLM__SATURATION_THRESHOLD`         |

**When to Use:**

- **CPU Deployments**: Enable `exclude-throughput-target` and `exclude-throughput-result` to prevent anomalous data points in performance graphs (TTFT spikes, inter-token latency anomalies)
- **GPU Deployments**: Use default settings (all disabled)

**Example for CPU-optimized benchmarking:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --exclude-throughput-target true \
  --exclude-throughput-result true \
  --saturation-threshold 0.98 \
  --data "prompt_tokens=256,output_tokens=128" \
  --max-seconds 300
```

**Using Environment Variables:**

```bash
export GUIDELLM__EXCLUDE_THROUGHPUT_TARGET=true
export GUIDELLM__EXCLUDE_THROUGHPUT_RESULT=true
export GUIDELLM__SATURATION_THRESHOLD=0.98

guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --data "prompt_tokens=256,output_tokens=128"
```

**How It Works:**

- `exclude-throughput-target`: Prevents generating a constant-rate test at the throughput level, avoiding "elbow" artifacts in graphs
- `exclude-throughput-result`: Removes the throughput benchmark from saved results, eliminating visual anomalies caused by burst capacity measurements
- `saturation-threshold`: Automatically stops the sweep when achieved rate falls below this percentage of target rate (e.g., 0.98 = 98%)

These parameters are particularly useful for CPU deployments where saturation occurs at lower rates than burst capacity, creating misleading graph patterns if throughput measurements are included.

**Important Note:**

Do not set `--max-concurrency` or `GUIDELLM__MAX_CONCURRENCY` when running sweep tests. The sweep profile uses the throughput test to discover the server's true capacity, and artificially limiting concurrency will result in an underestimated throughput measurement. This causes the constant-rate tests to run at rates far below the actual server capacity, preventing proper saturation detection and producing misleading results where TTFT may decrease instead of increase.

### Data Options

For synthetic data, some key options include, among others:

- `prompt_tokens`: Average number of tokens for prompts
- `output_tokens`: Average number of tokens for outputs
- `samples`: Number of samples to generate (default: 1000)

For a complete list of options, run:

```bash
guidellm benchmark run --help
```

## Working with Real Data

While synthetic data is convenient for quick tests, you can benchmark with real-world data:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "/path/to/your/dataset.json" \
  --profile constant \
  --rate 5
```

You can also use datasets from HuggingFace or customize synthetic data generation with additional parameters such as standard deviation, minimum, and maximum values.

**Note:** When benchmarking against servers that require authentication (such as OpenAI's API), you'll need to provide an API key. See the [API Key Configuration](../guides/backends.md#api-key-configuration) section in the Backends documentation for details.

By default, complete results are saved to `benchmarks.json`, `benchmarks.csv`, and `benchmarks.html` in your current directory. Use the `--output-dir` parameter to specify a different location and `--outputs` to control which formats are generated.

Learn more about dataset options in the [Datasets documentation](../guides/datasets.md) and backend configuration in the [Backends documentation](../guides/backends.md).
