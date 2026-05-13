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

Learn more about dataset options in the [Datasets documentation](../guides/datasets.md) and backend configuration in the [Backends documentation](../guides/backends.md).

## Understanding Benchmark Options

GuideLLM offers a wide range of configuration options to customize your benchmarks. Here are the most important parameters you should know:

### Key Parameters

| Parameter        | Description                                    | Example                                        |
| ---------------- | ---------------------------------------------- | ---------------------------------------------- |
| `--target`       | URL of the OpenAI-compatible server            | `--target "http://localhost:8000"`             |
| `--model`        | Model name to benchmark                        | `--model "Meta-Llama-3.1-8B-Instruct"`         |
| `--data`         | Data configuration for benchmarking            | `--data "prompt_tokens=256,output_tokens=128"` |
| `--profile`      | Type of benchmark profile to run               | `--profile sweep`                              |
| `--rate`         | Request rate or number of benchmarks for sweep | `--rate 10`                                    |
| `--random-seed`  | Random seed for reproducibility                | `--random-seed 42`                             |
| `--max-seconds`  | Duration for each benchmark in seconds         | `--max-seconds 30`                             |
| `--max-requests` | Maximum number of requests for each benchmark  | `--max-requests 1000`                          |
| `--output-dir`   | Directory path to save output files            | `--output-dir results/`                        |
| `--outputs`      | Output formats to generate                     | `--outputs json csv html`                      |

### Random Seed (`--random-seed`)

The random seed is used for any operation in GuideLLM that involves randomness such as synthetic data generation or poisson strategy scheduling calculations. By default it is a fixed value, so that rerunning GuideLLM with the same arguments should produce the same results.

### Strategy Constraints

The strategy constraints, `--max-requests` and `--max-seconds`, apply individually to each strategy in a profile. Profiles with multiple strategies include `sweep` and any profile with `--rate` set to a list of values.

For example, setting `--max-requests 1000` with `--profile sweep` will run 1000 synchronous requests, 1000 throughput requests, and 1000 `constant` requests at each interpolated rate, while `--max-seconds 30` will run each strategy for 30 seconds. Similarly, using `--max-seconds 30` with `--profile concurrent --rate 10,20` will run 10 concurrent streams for 30 seconds and then 20 concurrent streams for 30 seconds.

### Benchmark Profiles (`--profile`)

GuideLLM supports several benchmark profiles and strategies, which are described in detail below.

#### Synchronous Profile

Runs requests one at a time (sequential).

```bash
guidellm benchmark --profile synchronous
```

| Strategy parameter | Description  | Example |
| ------------------ | ------------ | ------- |
| `--rate`           | Not allowed. |         |
| `--rampup`         | Not used.    |         |

#### Throughput Profile

Attempts to discover the server's maximum throughput by continually making requests in parallel.

```bash
guidellm benchmark --profile throughput
```

| Strategy parameter | Description                                              | Example       |
| ------------------ | -------------------------------------------------------- | ------------- |
| `--rate`           | Number of concurrent request streams.                    | `--rate 10`   |
| `--rampup`         | Number of sectonds to ramp up to the maximum throughput. | `--rampup 10` |

#### Concurrent Profile

Runs a fixed number of parallel request streams.

```bash
guidellm benchmark --profile concurrent
```

| Strategy parameter | Description                                                                                                                | Example                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `--rate`           | Number of concurrent requests to maintain. May be a single value or a list of values to run successive concurrent streams. | `--rate 10`, `--rate 16,32` |
| `--rampup`         | Duration in seconds to spread initial requests up to target rate.                                                          | `--rampup 10`               |

#### Constant Profile

Sends asynchronous requests at a fixed rate per second.

(The profile name `async` is an alias for the `constant` profile.)

```bash
guidellm benchmark --profile constant
```

| Strategy parameter | Description                                                                                        | Example                     |
| ------------------ | -------------------------------------------------------------------------------------------------- | --------------------------- |
| `--rate`           | Number of requests to send per second. May be a list of values to run successive constant streams. | `--rate 10`, `--rate 16,32` |
| `--rampup`         | Duration in seconds to linearly ramp up from 0 to target rate.                                     | `--rampup 10`               |

#### Poisson Profile

Sends asynchronous requests at varying rates using a Poisson distribution around the specified target rate(s). This probabilistic pattern is useful for simulating more realistic real-world traffic patterns.

```bash
guidellm benchmark --profile poisson
```

| Strategy parameter | Description                                                                                      | Example                     |
| ------------------ | ------------------------------------------------------------------------------------------------ | --------------------------- |
| `--rate`           | Maximum asynchronous requests to run. May be a list of values to run successive Poisson streams. | `--rate 10`, `--rate 16,32` |
| `--rampup`         | Not used.                                                                                        |                             |

#### Sweep Profile

The sweep profile applies a sequence of benchmark strategies to find the optimal performance points for the given model and data.

1. It runs a `synchronous` strategy to measure the baseline rate,
2. then runs a `throughput` strategy to determine peak throughput,
3. and finally runs a series of asynchronous `constant` strategies with rates interpolated between the baseline and maximum throughput. (The number of interpolated strategies is the value of the `--rate` parameter minus 2.)

```bash
guidellm benchmark --profile sweep
```

| Strategy parameter | Description                                                                      | Example       |
| ------------------ | -------------------------------------------------------------------------------- | ------------- |
| `--rate`           | Number of strategies to run in the sweep (including synchronous and throughput). | `--rate 10`   |
| `--rampup`         | Rate rampup duration in seconds for throughput and constant strategy steps.      | `--rampup 10` |

##### Sweep Profile Configuration

The sweep profile includes advanced configuration options for optimizing benchmarks on CPU-based deployments. These parameters help manage saturation detection and prevent graph artifacts:

**Available Parameters:**

| Parameter                     | Description                                       | Default | Environment Variable                  |
| ----------------------------- | ------------------------------------------------- | ------- | ------------------------------------- |
| `--exclude-throughput-target` | Stop constant-rate tests before throughput level  | `false` | `GUIDELLM__EXCLUDE_THROUGHPUT_TARGET` |
| `--exclude-throughput-result` | Exclude throughput benchmark from saved results   | `false` | `GUIDELLM__EXCLUDE_THROUGHPUT_RESULT` |
| `--saturation-threshold`      | Efficiency threshold for stopping sweep (0.0-1.0) | `0.98`  | `GUIDELLM__SATURATION_THRESHOLD`      |

**When to Use:**

- **CPU based system under test**: Enable `exclude-throughput-target` and `exclude-throughput-result` to prevent anomalous data points in performance graphs (TTFT spikes, inter-token latency anomalies)
- **GPU based system under test**: Use default settings (all disabled)

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

The sweep profile runs tests in this order:

1. **Synchronous test**: Measures baseline single-request performance
2. **Throughput test**: Discovers maximum server capacity with parallel requests
3. **Constant-rate tests**: Tests at interpolated rates between synchronous and throughput

Each parameter optimizes a different aspect:

- **`exclude-throughput-target`**: Prevents generating a constant-rate test at the throughput level itself

  - **Why**: The highest constant-rate test would target the same rate as the throughput test, creating redundant "elbow" artifacts in graphs
  - **Effect**: Stops constant-rate tests just before reaching throughput rate

- **`exclude-throughput-result`**: Removes the throughput benchmark from saved results

  - **Why**: Throughput tests measure burst capacity with severe queuing (e.g., 23+ second TTFT), creating extreme outliers in graphs
  - **Effect**: Graphs only show sustainable performance metrics from constant-rate tests

- **`saturation-threshold`**: Stops the sweep when efficiency drops below threshold

  - **Why**: Once saturation is detected (achieved rate < target rate × threshold), further tests provide diminishing returns
  - **Effect**: Saves time by stopping early when the server can no longer meet target rates

**Why use all three together?**

For CPU based system under test, all three parameters work synergistically:

- `saturation-threshold` stops the sweep efficiently when saturation is detected
- `exclude-throughput-target` prevents testing at the unsustainable throughput rate
- `exclude-throughput-result` removes the anomalous throughput spike from graphs

This combination produces clean, efficient benchmarks that focus on sustainable performance ranges.

**Important Note:**

Do not set `--max-concurrency` or `GUIDELLM__MAX_CONCURRENCY` when running sweep tests. The sweep profile uses the throughput test to discover the server's true capacity, and artificially limiting concurrency will result in an underestimated throughput measurement. This causes the constant-rate tests to run at rates far below the actual server capacity, preventing proper saturation detection and producing misleading results where TTFT may decrease instead of increase.

## Data Options

### Synthetic Data Options

For synthetic data, parameters for random data generation are passed as key=value pairs to the `--data` parameter. Some key options include:

- `prompt_tokens`: Average number of tokens for prompts
- `output_tokens`: Average number of tokens for outputs

For example, to generate 1000 samples with a prompt length of 100 tokens and an output length of 50 tokens:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "prompt_tokens=100,output_tokens=50,samples=1000" \
  --profile constant \
  --rate 5
```

You can customize synthetic data generation with additional parameters such as standard deviation, minimum, and maximum values. See the [Datasets Synthetic data documentation](../guides/datasets.md#synthetic-data) for more details.

### Working with Real Data

While synthetic data is convenient for quick tests, you can benchmark with real-world data:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data "/path/to/your/dataset.json" \
  --profile constant \
  --rate 5
```

You can also use datasets from HuggingFace:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --data garage-bAInd/Open-Platypus \
  --profile constant \
  --rate 5
```

## Output Options

By default, complete results are saved to `benchmarks.json`, `benchmarks.csv`, and `benchmarks.html` in your current directory. Use the `--output-dir` parameter to specify a different location and `--outputs` to control which formats are generated.

Learn more about output options in the [Outputs documentation](../guides/outputs.md).

## Authentication

When benchmarking against servers that require authentication (such as OpenAI's API), you'll need to provide an API key via the `--backend-kwargs` parameter. See the [API Key Configuration](../guides/backends.md#api-key-configuration) section in the Backends documentation for details.
