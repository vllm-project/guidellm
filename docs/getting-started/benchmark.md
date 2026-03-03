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

| Parameter        | Description                                    | Example                                        |
| ---------------- | ---------------------------------------------- | ---------------------------------------------- |
| `--target`       | URL of the OpenAI-compatible server            | `--target "http://localhost:8000"`             |
| `--model`        | Model name to benchmark (optional)             | `--model "Meta-Llama-3.1-8B-Instruct"`         |
| `--data`         | Data configuration for benchmarking            | `--data "prompt_tokens=256,output_tokens=128"` |
| `--profile`      | Type of benchmark profile to run               | `--profile sweep`                              |
| `--rate`         | Request rate or number of benchmarks for sweep | `--rate 10`                                    |
| `--max-seconds`  | Duration for each benchmark in seconds         | `--max-seconds 30`                             |
| `--max-requests` | Maximum number of requests for each benchmark  | `--max-requests 1000`                          |
| `--output-dir`   | Directory path to save output files            | `--output-dir results/`                        |
| `--outputs`      | Output formats to generate                     | `--outputs json csv html`                      |

### Benchmark Profiles (`--profile`)

GuideLLM supports several benchmark profiles and strategies, which are described in detail below.

*Strategy Constraints*

The strategy constraints, `--max-requests` and `--max-seconds`, apply individually to each strategy step. Profiles with multiple strategies include `sweep`, and any profile with `--rate` set to a list of values.

For example, setting `--max-requests 1000` with `--profile sweep` will run 1000 synchronous requests, 1000 throughput requests, and 1000 `constant` requests at each interpolated rate, while `--max-seconds 30` will run each strategy for 30 seconds. Similarly, using `--max-seconds 30` with `--profile concurrent --rate 10,20` will run 10 concurrent streams for 30 seconds and then 20 concurrent streams for 30 seconds.

#### Synchronous Profile

Runs requests one at a time (sequential).

```bash
guidellm benchmark --profile synchronous
```

| Strategy parameter | Description  | Example |
| ------------------ | ------------ | ------- |
| `--rate`           | Not allowed. |         |
| `--random-seed`    | Not used.    |         |

#### Throughput Profile

Tests maximum throughput by running requests in parallel.

```bash
guidellm benchmark --profile throughput
```

| Strategy parameter | Description                         | Example     |
| ------------------ | ----------------------------------- | ----------- |
| `--rate`           | Maximum concurrent requests to run. | `--rate 10` |
| `--random-seed`    | Not used.                           |             |

#### Concurrent Profile

Runs a fixed number of parallel request streams.

```bash
guidellm benchmark --profile concurrent
```

| Strategy parameter | Description                                                                                                          | Example                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `--rate`           | Number of concurrent streams to run. May be a single value or a list of values to run successive concurrent streams. | `--rate 10`, `--rate 16,32` |
| `--random-seed`    | Not used.                                                                                                            |                             |

#### Constant Profile

Sends asynchronous requests at a fixed rate per second.

(The profile name `async` is an alias for the `constant` profile.)

```bash
guidellm benchmark --profile constant
```

| Strategy parameter | Description                                                                                       | Example                     |
| ------------------ | ------------------------------------------------------------------------------------------------- | --------------------------- |
| `--rate`           | Maximum asynchronous requests to run. May be a list of values to run successive constant streams. | `--rate 10`, `--rate 16,32` |
| `--random-seed`    | Not used.                                                                                         |                             |

#### Poisson Profile

Sends asynchronous requests at varying rates using a Poisson distribution around the specified target rate(s).

```bash
guidellm benchmark --profile poisson
```

| Strategy parameter | Description                                                                                      | Example                     |
| ------------------ | ------------------------------------------------------------------------------------------------ | --------------------------- |
| `--rate`           | Maximum asynchronous requests to run. May be a list of values to run successive Poisson streams. | `--rate 10`, `--rate 16,32` |
| `--random-seed`    | Used to generate random rates.                                                                   | `--random-seed 42`          |

#### Sweep Profile

The sweep profile applies a sequence of benchmark strategies to find the optimal performance points for the given model and data.

1. It runs a `synchronous` strategy to measure the baseline rate,
2. then runs a `throughput` strategy to determine peak throughput,
3. and finally runs a series of asynchronous `constant` strategies with rates interpolated between the baseline and maximum throughput. (The number of interpolated strategies is the value of the `--rate` parameter minus 2.)

```bash
guidellm benchmark --profile sweep
```

| Strategy parameter | Description                                                                      | Example     |
| ------------------ | -------------------------------------------------------------------------------- | ----------- |
| `--rate`           | Number of strategies to run in the sweep (including synchronous and throughput). | `--rate 10` |
| `--random-seed`    | Not used.                                                                        |             |

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
