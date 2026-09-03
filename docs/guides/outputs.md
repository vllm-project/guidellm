# Output Types

GuideLLM provides flexible options for outputting benchmark results, catering to both console-based summaries and file-based detailed reports. This document outlines the supported output types, their configurations, and how to utilize them effectively.

## CLI Output Configuration

Without any `--output` options, GuideLLM writes `benchmarks.json` and `benchmarks.csv`. These files use the directory configured by `GUIDELLM__DEFAULT_RESULTS_DIR`, or the current directory when the variable is not set.

Output configuration follows the typed registry-backed CLI pattern. Specifying any `--output` replaces the default JSON and CSV outputs, so repeat the option for every file format you want. This example keeps both default formats and adds HTML:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json \
  --output kind=csv \
  --output kind=html
```

Supported file output types are `json`, `yaml`, `csv`, `html`, and `plot`. Each supplies a default filename, so `path` is only needed to change the name or destination. Console output is configured separately as described below.

## Console Output

By default, GuideLLM displays benchmark results and progress directly in the console. Console output is an implicit default: it is not replaced by explicit `--output` options, and specifying `--output kind=console` has no additional effect. The console progress and outputs are divided into multiple sections:

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
4. **HTML**: Self-contained static HTML report with throughput/latency charts and tables (no CDN or external assets).
5. **PLOT**: Static image chart of benchmark metrics. The image format is selected from the `path` file extension — supported formats are PNG, JPG/JPEG, SVG, and PDF. A path with no extension defaults to `.png`, and an unsupported extension raises an error. The `dpi` parameter (default `100`) sets the output image resolution in dots per inch — for example, `--output kind=plot,path=plot.png,dpi=72`.

### Configuring File Outputs

- **Default destination**: File outputs use `GUIDELLM__DEFAULT_RESULTS_DIR` when set and the current directory otherwise.
- **Output path**: Each file type has a default filename. Pass `path=` to control the name or destination.
- **Multiple formats**: Repeat `--output` with different types.
- **Explicit selection**: Any `--output` replaces the default JSON and CSV selection. Include `--output kind=json` and `--output kind=csv` explicitly when you want to preserve them.

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

This command writes only JSON (in addition to the independent console output):

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json,path=results/benchmark.json
```

### Controlling Output File Size

Long benchmark runs with thousands of requests can produce large JSON and YAML output files because, by default, every request's full data (prompt text, output text, tool calls) is retained. The `--metrics` option lets you limit how much request data is kept using reservoir sampling, while lightweight stats (latency, token counts, timing) are always preserved for every request.

Use `sample_size` to set the maximum number of requests **per status group** (completed, errored, incomplete) that retain their full data:

| Value             | Behavior                                                   |
| ----------------- | ---------------------------------------------------------- |
| Not set (default) | Keep full data for all requests                            |
| `0`               | Strip all request data (stats only)                        |
| `N` (e.g. `100`)  | Retain full data for N randomly sampled requests per group |

```bash
# Keep full data for only 100 sampled requests per group
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_requests,count=10000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics kind=generative,sample_size=100 \
  --output kind=json,path=results/benchmark.json
```

The `--metrics` option also accepts `prefer_response_metrics` (default `true`), which controls whether server-reported token counts are preferred over client-calculated counts when both are available. This rarely needs to be changed.

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
