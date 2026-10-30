# Output Types

GuideLLM provides flexible options for outputting benchmark results, catering to both console-based summaries and file-based detailed reports. This document outlines the supported output types, their configurations, and how to utilize them effectively.

## CLI Output Configuration

Without any `--output` options, `guidellm run` writes `benchmarks.json` and `benchmarks.csv` to the [default results directory](#configuring-file-outputs).

Output configuration follows the typed registry-backed CLI pattern. Specifying any `--output` replaces the default JSON and CSV outputs, so repeat the option for every file format you want. This example keeps both default formats and adds HTML:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=json \
  --output kind=csv \
  --output kind=html
```

Choose from the [supported file formats](#supported-file-formats) and see [configuring file outputs](#configuring-file-outputs) to set paths. Console output is controlled separately as described below.

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

| Format (`kind`) | Default filename  | Contents                                                                                                                                          |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `json`          | `benchmarks.json` | Configuration, metadata, benchmark statistics, and retained request data for debugging or reloading into Python.                                  |
| `yaml`          | `benchmarks.yaml` | The same detailed benchmark data as JSON in a human-readable YAML representation.                                                                 |
| `csv`           | `benchmarks.csv`  | A tabular summary of throughput, latency, token counts, and rates for spreadsheets and comparisons. Does not include detailed request-level data. |
| `html`          | `benchmarks.html` | A self-contained report with throughput and latency charts and tables. CSS and JavaScript are embedded for offline sharing.                       |
| `plot`          | `benchmarks.png`  | Static benchmark performance graphs.                                                                                                              |

For PLOT, the `path` extension selects the image format: PNG, JPG/JPEG, SVG, or PDF. For example, `--output kind=plot,path=benchmarks.pdf` creates a PDF. A path without an extension defaults to `.png`; unsupported extensions raise an error. The `dpi` parameter (default `100`) controls image resolution, for example `--output kind=plot,path=benchmarks.jpg,dpi=72`.

### Configuring File Outputs

Omit `path` to use the format's default filename in the directory configured by `GUIDELLM__DEFAULT_RESULTS_DIR`, or in the current directory when the variable is not set. Specify `path=` to set the filename or destination.

For example, this command writes JSON to a custom path, in addition to the independent console output:

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

To retain full request data for at most 100 randomly sampled requests per status group, use:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_requests,count=10000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics kind=generative,sample_size=100 \
  --output kind=json,path=results/benchmark.json
```

The `--metrics` option also accepts `prefer_response_metrics` (default `true`), which controls whether server-reported token counts are preferred over client-calculated counts when both are available. This rarely needs to be changed.

### Embedded Media in JSON and YAML

JSON and YAML outputs preserve embedded media by default (`include_media=true`) so retained requests can be reconstructed for replay. To omit recognized media payloads from an individual output, set `include_media=false`. This replaces multipart file contents, media data URLs, and explicit inline audio/image/file payloads with metadata. It applies to retained request arguments and recognized media in the saved configuration.

For example, re-export a report to two smaller sharing copies and a lossless copy:

```bash
guidellm export benchmarks.json \
  --output kind=json,path=share.json,include_media=false \
  --output kind=yaml,path=share.yaml,include_media=false \
  --output kind=json,path=replay.json,include_media=true
```

The same output options work with `guidellm run`. Each output has an independent policy: omitting media does not change the in-memory report, another output, or the requests sent to the server. Transcriptions, ordinary prompt/output text, and lightweight metrics remain available.

Replacement metadata includes `omitted: true`, a serialized character count, filename/MIME type when available, and decoded byte count when the Base64 encoding is explicit. Multipart argument serialization does not distinguish original bytes from text, so its metadata reports the serialized character count without guessing a decoded size.

Scalar dataset values retain their string type and contain JSON-encoded omission metadata. This keeps compact reports loadable for analysis and re-export. Ordinary prompt and output text remain unchanged, including literal data URLs.

An omitted-media copy cannot reconstruct the removed payloads for replay. Keep a lossless copy when replay is required. This option is not general secret or personal-data redaction: opaque/free-form encodings, ordinary text, headers, filenames, and remote URLs can still contain sensitive information. It does not fetch or remove external files and does not reduce payload retention during benchmark execution.

When saving through Python, use `report.save_file("share.json", include_media=False)` for the same output-time policy.

### Reloading Results

JSON files can be reloaded into Python for further analysis using the `GenerativeBenchmarksReport` class. Below is a sample code snippet for reloading results:

```python
from guidellm.benchmark import GenerativeBenchmarksReport

report = GenerativeBenchmarksReport.load_file(
    path="benchmarks.json",
)
benchmarks = report.benchmarks

for benchmark in benchmarks:
    print(benchmark.config.id_)
```

For more details on the `GenerativeBenchmarksReport` class and its methods, refer to the [source code](https://github.com/vllm-project/guidellm/blob/main/src/guidellm/benchmark/schemas/report.py).
