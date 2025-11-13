# Output Types

GuideLLM provides flexible options for outputting benchmark results, catering to both console-based summaries and file-based detailed reports. This document outlines the supported output types, their configurations, and how to utilize them effectively.

For all of the output formats, `--output-extras` can be used to include additional information. This could include tags, metadata, hardware details, and other relevant information that can be useful for analysis. This must be supplied as a JSON encoded string. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type sweep \
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
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --disable-progress
```

To disable console output, use the `--disable-console-outputs` flag when running the `guidellm benchmark` command. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --disable-console-outputs
```

### Enabling Extra Information

GuideLLM includes the option to display extra information during the benchmark runs to monitor the overheads and performance of the system. This can be enabled by using the `--display-scheduler-stats` flag when running the `guidellm benchmark` command. For example:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --display-scheduler-stats
```

The above command will display an additional row for each benchmark within the progress output, showing the scheduler overheads and other relevant information.

## File-Based Outputs

GuideLLM supports saving benchmark results to files in various formats, including JSON, YAML, and CSV. These files can be used for further analysis, reporting, or reloading into Python for detailed exploration.

### Supported File Formats

1. **JSON**: Contains all benchmark results, including full statistics and request data. This format is ideal for reloading into Python for in-depth analysis.
2. **YAML**: Similar to JSON, YAML files include all benchmark results and are human-readable.
3. **CSV**: Provides a summary of the benchmark data, focusing on key metrics and statistics. Note that CSV does not include detailed request-level data.

### Configuring File Outputs

- **Output Path**: Use the `--output-path` argument to specify the file path or directory for saving the results. If a directory is provided, the results will be saved as `benchmarks.json` by default. The file type is determined by the file extension (e.g., `.json`, `.yaml`, `.csv`).
- **Sampling**: To limit the size of the output files, you can configure sampling options for the dataset using the `--output-sampling` argument.

Example command to save results in YAML format:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-path "results/benchmarks.csv" \
  --output-sampling 20
```

### Reloading Results

JSON and YAML files can be reloaded into Python for further analysis using the `GenerativeBenchmarksReport` class. Below is a sample code snippet for reloading results:

```python
from guidellm.benchmark import GenerativeBenchmarksReport

report = GenerativeBenchmarksReport.load_file(
    path="benchmarks.json",
)
benchmarks = report.benchmarks

for benchmark in benchmarks:
    print(benchmark.id_)
```

For more details on the `GenerativeBenchmarksReport` class and its methods, refer to the [source code](https://github.com/neuralmagic/guidellm/blob/main/src/guidellm/benchmark/output.py).
