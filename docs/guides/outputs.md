# Output Types

GuideLLM provides flexible options for outputting benchmark results, catering to both console-based summaries and file-based detailed reports. This document outlines the supported output types, their configurations, and how to utilize them effectively.

## CLI output configuration

Outputs use the single-parameter CLI pattern. Repeat `--output` for each format:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json \
  --output kind=csv,path=results/benchmark.csv \
  --output kind=html,path=results/benchmark.html
```

Supported output types: `json`, `yaml`, `csv`, `html`, and `console`. Each accepts a `path` parameter (defaults vary by type; for example `benchmarks.json` for JSON).

## Console Output

By default, GuideLLM displays benchmark results and progress directly in the console. The console progress and outputs are divided into multiple sections:

1. **Initial Setup Progress**: Displays the progress of the initial setup, including server connection and data preparation.
2. **Benchmark Progress**: Shows the progress of the benchmark runs, including the number of requests completed and the current rate.
3. **Final Results**: Summarizes the benchmark results, including average latency, throughput, and other key metrics.
   1. **Benchmarks Metadata**: Summarizes the benchmark run, including server details, data configurations, and profile arguments.
   2. **Benchmarks Info**: Provides a high-level overview of each benchmark, including request statuses, token counts, and durations.
   3. **Benchmarks Stats**: Displays detailed statistics for each benchmark, such as request rates, concurrency, latency, and token-level metrics.

### Disabling Console Output

To disable interactive progress updates, use `--disable-console-interactive` (alias `--disable-progress`):

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console-interactive
```

To disable all console output, use `--disable-console` (alias `--disable-console-outputs`):

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --disable-console
```

## File-Based Outputs

GuideLLM supports saving benchmark results to files in various formats, including JSON, YAML, and CSV. These files can be used for further analysis, reporting, or reloading into Python for detailed exploration.

### Supported File Formats

1. **JSON**: Contains all benchmark results, including full statistics and request data. This format is ideal for reloading into Python for in-depth analysis.
2. **YAML**: Contains all benchmark results, including full statistics and request data, in YAML format which is human-readable and easy to work with in various tools.
3. **CSV**: Provides a summary of the benchmark data, focusing on key metrics and statistics. Note that CSV does not include detailed request-level data.
4. **HTML**: Interactive HTML report with tables and visualizations of benchmark results.
5. **Console**: Terminal output displayed during execution (can be disabled).

### Configuring File Outputs

- **Output path**: Pass `path=` in each `--output` config. Use an explicit filename to control the destination, or a directory-style path as supported by each output type.
- **Multiple formats**: Repeat `--output` with different types.

#### Example commands to save results in specific formats:

```bash
# JSON, CSV, and HTML to a results directory
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json \
  --output kind=csv,path=results/benchmark.csv \
  --output kind=html,path=results/benchmark.html
```

**Example: Single output format**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json
```

### Reloading Results

JSON files can be reloaded into Python for further analysis using the `GenerativeBenchmarksReport` class. Below is a sample code snippet for reloading results:

```python
from guidellm.benchmark import GenerativeBenchmarksReport

report = GenerativeBenchmarksReport.load_file(
    path="benchmarks.json",
)
benchmarks = report.benchmarks

for benchmark in benchmarks:
    print(benchmark.id_)
```

For more details on the `GenerativeBenchmarksReport` class and its methods, refer to the [source code](https://github.com/vllm-project/guidellm/blob/main/src/guidellm/benchmark/schemas/generative/report.py).
