# Output Types

GuideLLM provides flexible options for outputting benchmark results, catering to both console-based summaries and file-based detailed reports. This document outlines the supported output types, their configurations, and how to utilize them effectively.

For all of the output formats, `--output-extras` can be used to include additional information. This could include tags, metadata, hardware details, and other relevant information that can be useful for analysis. This must be supplied as a JSON encoded string. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-extras '{"tag": "my_tag", "metadata": {"key": "value"}}'
```

## Console Output

By default, GuideLLM displays benchmark results and progress directly in the console. The console progress and outputs are divided into multiple sections:

1. **Initial Setup Progress**: Displays the progress of the initial setup, including server connection and data preparation.
2. **Benchmark Progress**: Shows the progress of the benchmark runs, including the number of requests completed and the current rate.
3. **Final Results**: Summarizes the benchmark results, including average latency, throughput, and other key metrics.
   1. **Benchmarks Metadata**: Summarizes the benchmark run, including server details, data configurations, and profile arguments.
   2. **Benchmarks Info**: Provides a high-level overview of each benchmark, including request statuses, token counts, and durations.
   3. **Benchmarks Stats**: Displays detailed statistics for each benchmark, such as request rates, concurrency, latency, and token-level metrics.

### Disabling Console Output

To disable the progress outputs to the console, use the `disable-progress` flag when running the `guidellm benchmark` command. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --disable-progress
```

To disable console output, use the `--disable-console-outputs` flag when running the `guidellm benchmark` command. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --disable-console-outputs
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

- **Output Directory**: Use the `--output-dir` argument to specify the directory for saving the results. By default, files are saved in the current directory.
- **Output Formats**: Use the `--outputs` argument to specify which formats or exact file names (with supported file extensions, e.g. `benchmarks.json`) to generate. By default, JSON, CSV, and HTML are generated.
- **Sampling**: To limit the size of the output files and number of detailed request samples included, you can configure sampling options using the `--sample-requests` argument.

#### Example commands to save results in specific formats:

The `--outputs` parameter accepts output formats in the following ways:

**Command Line:**

```bash
# Comma-separated format aliases (recommended)
--outputs json,csv,html

# Or using multiple flags
--outputs json --outputs csv --outputs html
```

**Environment Variables:**

```bash
# Comma-separated values for environment variables
-e GUIDELLM_OUTPUTS=json,csv,html
```

#### Examples

**Example 1: Comma-separated format aliases (recommended)**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-dir "results/" \
  --outputs json,csv,html \
  --sample-requests 20
```

**Example 2: Multiple flags for each format**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-dir "results/" \
  --outputs json \
  --outputs csv \
  --outputs html \
  --sample-requests 20
```

**Example 3: Using environment variables with Docker/Podman**

```bash
podman run --rm -it --network=host \
  -v "/tmp/results:/results:z" \
  -e GUIDELLM_TARGET=http://localhost:8000 \
  -e GUIDELLM_PROFILE=sweep \
  -e GUIDELLM_MAX_SECONDS=30 \
  -e GUIDELLM_DATA="prompt_tokens=256,output_tokens=128" \
  -e GUIDELLM_OUTPUT_DIR=/results \
  -e GUIDELLM_OUTPUTS=json,csv,html \
  -e GUIDELLM_SAMPLE_REQUESTS=20 \
  ghcr.io/vllm-project/guidellm:latest
```

**Example 4: Single output format**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-dir "results/" \
  --outputs json \
  --sample-requests 20
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

For more details on the `GenerativeBenchmarksReport` class and its methods, refer to the [source code](https://github.com/vllm-project/guidellm/blob/main/src/guidellm/benchmark/schemas/generative/reports.py).
