# CLI Reference

This page provides a reference for the `guidellm` command-line interface. For more advanced configuration, including environment variables and `.env` files, see the [Configuration Guide](./configuration.md).

## `guidellm benchmark run`

This command is the primary entrypoint for running benchmarks. It has many options that can be specified on the command line or in a scenario file.

### Scenario Configuration

| Option                      | Description                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `--scenario <PATH or NAME>` | The name of a builtin scenario or path to a scenario configuration file. Options specified on the command line will override the scenario file. |

### Target and Backend Configuration

These options configure how `guidellm` connects to the system under test.

| Option                  | Description                                                                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--target <URL>`        | **Required.** The endpoint of the target system, e.g., `http://localhost:8080`. Can also be set with the `GUIDELLM__OPENAI__BASE_URL` environment variable.                                                   |
| `--backend-type <TYPE>` | The type of backend to use. Defaults to `openai_http`.                                                                                                                                                        |
| `--backend-args <JSON>` | A JSON string for backend-specific arguments. For example: `--backend-args '{"headers": {"Authorization": "Bearer my-token"}, "verify": false}'` to pass custom headers and disable certificate verification. |
| `--model <NAME>`        | The ID of the model to benchmark within the backend.                                                                                                                                                          |

### Data and Request Configuration

These options define the data to be used for benchmarking and how requests will be generated.

| Option                    | Description                                                                                                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--data <SOURCE>`         | The data source. This can be a HuggingFace dataset ID, a path to a local data file, or a synthetic data configuration. See the [Data Formats Guide](./data_formats.md) for more details. |
| `--rate-type <TYPE>`      | The type of request generation strategy to use (e.g., `constant`, `poisson`, `sweep`).                                                                                                   |
| `--rate <NUMBER>`         | The rate of requests per second for `constant` or `poisson` strategies, or the number of steps for a `sweep`.                                                                            |
| `--max-requests <NUMBER>` | The maximum number of requests to run for each benchmark.                                                                                                                                |
| `--max-seconds <NUMBER>`  | The maximum number of seconds to run each benchmark for.                                                                                                                                 |

## `guidellm preprocess dataset`

The `preprocess dataset` command processes datasets to have specific prompt and output token sizes. This is useful for standardizing datasets before benchmarking, especially when your source data has variable-length prompts or doesn't match your target token requirements.

### Basic Syntax

```bash
guidellm preprocess dataset <DATA> <OUTPUT_PATH> --processor <PROCESSOR> --config <CONFIG>
```

### Required Arguments

| Argument           | Description                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`             | Path to the input dataset or Hugging Face dataset ID. Supports all dataset formats documented in the [Dataset Configurations](../datasets.md). |
| `OUTPUT_PATH`      | Path to save the processed dataset, including file suffix (e.g., `processed_dataset.jsonl`, `output.csv`).                                    |
| `--processor`      | **Required.** Processor or tokenizer name/path for calculating token counts. Can be a Hugging Face model ID or local path.                    |
| `--config`         | **Required.** Configuration specifying target token sizes. Can be a JSON string, key=value pairs, or file path (.json, .yaml, .yml, .config).  |

### Configuration and Processor Options

The `--config` parameter uses the same format as synthetic data configuration. It accepts a JSON string, key=value pairs, or a configuration file path. For detailed information about available configuration parameters (such as `prompt_tokens`, `output_tokens`, `prompt_tokens_stdev`, etc.), see the [Synthetic Data Configuration Options](../datasets.md#configuration-options) in the Dataset Configurations guide.

The `--processor` argument specifies the tokenizer to use for calculating token counts. This is required because the preprocessing command needs to tokenize prompts to ensure they match the target token sizes. For information about using processors, including Hugging Face model IDs, local paths, and processor arguments, see the [Data Arguments Overview](../datasets.md#data-arguments-overview) section.

### Column Mapping

When your dataset uses non-standard column names, you can use `--data-column-mapper` to map your columns to GuideLLM's expected column names. This is particularly useful when:

1. **Your dataset uses different column names** (e.g., `question` instead of `prompt`, `instruction` instead of `text_column`)
2. **You have multiple datasets** and need to specify which dataset's columns to use
3. **Your dataset has system prompts or prefixes** in a separate column

**Column mapping format:**
The `--data-column-mapper` accepts a JSON string mapping column types to column names:

```json
{
  "text_column": "question",
  "prefix_column": "system_prompt",
  "prompt_tokens_count_column": "input_tokens",
  "output_tokens_count_column": "completion_tokens"
}
```

**Supported column types:**
- `text_column`: The main prompt text (defaults: `prompt`, `instruction`, `question`, `input`, `context`, `content`, `text`)
- `prefix_column`: System prompt or prefix (defaults: `system_prompt`, `system`, `prefix`)
- `prompt_tokens_count_column`: Column containing prompt token counts (defaults: `prompt_tokens_count`, `input_tokens_count`)
- `output_tokens_count_column`: Column containing output token counts (defaults: `output_tokens_count`, `completion_tokens_count`)
- `image_column`: Image data column
- `video_column`: Video data column
- `audio_column`: Audio data column

**Example: Mapping custom column names**

If your dataset has a CSV file with columns `user_query` and `system_message`:

```csv
user_query,system_message
"What is AI?","You are a helpful assistant."
"How does ML work?","You are a technical expert."
```

You would use:
```bash
guidellm preprocess dataset \
    "dataset.csv" \
    "processed.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,output_tokens=256" \
    --data-column-mapper '{"text_column": "user_query", "prefix_column": "system_message"}'
```

**Example: Multiple datasets**

If you're working with multiple datasets and need to specify which dataset's columns to use, you can use the format `<dataset_index>.<column_name>` or `<dataset_name>.<column_name>`:

```bash
--data-column-mapper '{"text_column": "0.prompt", "prefix_column": "1.system"}'
```

### Handling Short Prompts

When prompts are shorter than the target token length, you can specify how to handle them using `--short-prompt-strategy`:

| Strategy      | Description                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------- |
| `ignore`      | Skip prompts that are shorter than the target length (default)                                 |
| `concatenate` | Concatenate multiple short prompts together until the target length is reached                 |
| `pad`         | Pad short prompts with a specified character to reach the target length                      |
| `error`       | Raise an error if a prompt is shorter than the target length                                  |

**Example: Concatenating short prompts**
```bash
guidellm preprocess dataset \
    "dataset.jsonl" \
    "processed.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,output_tokens=256" \
    --short-prompt-strategy "concatenate" \
    --concat-delimiter "\n\n"
```

**Example: Padding short prompts**
```bash
guidellm preprocess dataset \
    "dataset.jsonl" \
    "processed.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,output_tokens=256" \
    --short-prompt-strategy "pad" \
    --pad-char " "
```

### Additional Options

| Option                           | Description                                                                                                                             |
| -------------------------------- |-----------------------------------------------------------------------------------------------------------------------------------------|
| `--data-args <JSON>`             | JSON string of arguments to pass to dataset loading. See [Data Arguments Overview](../datasets.md#data-arguments-overview) for details. |
| `--prefix-tokens <NUMBER>`       | Single prefix token count (alternative to `prefix_tokens` in config).                                                                   |
| `--include-prefix-in-token-count` | Include prefix tokens in prompt token count calculation (flag). When enabled, prefix trimming is disabled and the prefix is kept as-is. |
| `--random-seed <NUMBER>`         | Random seed for reproducible token sampling (default: 42).                                                                              |
| `--push-to-hub`                  | Push the processed dataset to Hugging Face Hub (flag).                                                                                  |
| `--hub-dataset-id <ID>`          | Hugging Face Hub dataset ID for upload (required if `--push-to-hub` is set).                                                            |

### Complete Examples

**Example 1: Basic preprocessing with custom column names**
```bash
guidellm preprocess dataset \
    "my_dataset.csv" \
    "processed_dataset.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,output_tokens=256" \
    --data-column-mapper '{"text_column": "user_question", "prefix_column": "system_instruction"}'
```

**Example 2: Preprocessing with distribution and short prompt handling**
```bash
guidellm preprocess dataset \
    "dataset.jsonl" \
    "processed.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,prompt_tokens_stdev=50,output_tokens=256,output_tokens_stdev=25" \
    --short-prompt-strategy "concatenate" \
    --concat-delimiter "\n\n" \
    --random-seed 123
```

**Example 3: Preprocessing with processor arguments and prefix tokens**
```bash
guidellm preprocess dataset \
    "dataset.jsonl" \
    "processed.jsonl" \
    --processor "gpt2" \
    --processor-args '{"use_fast": false}' \
    --config "prompt_tokens=512,output_tokens=256" \
    --prefix-tokens 100 \
    --include-prefix-in-token-count
```

**Example 4: Preprocessing and uploading to Hugging Face Hub**
```bash
guidellm preprocess dataset \
    "my_dataset.jsonl" \
    "processed.jsonl" \
    --processor "gpt2" \
    --config "prompt_tokens=512,output_tokens=256" \
    --push-to-hub \
    --hub-dataset-id "username/processed-dataset"
```

### Notes

- The `--config` parameter uses the same format as synthetic data configuration. See the [Synthetic Data Configuration Options](../datasets.md#configuration-options) for all available parameters.
- The processor/tokenizer is required because the preprocessing command needs to tokenize prompts to ensure they match target token sizes. See the [Data Arguments Overview](../datasets.md#data-arguments-overview) for processor usage details.
- Column mappings are only needed when your dataset uses non-standard column names. GuideLLM will automatically try common column names if no mapping is provided.
- When using `--short-prompt-strategy concatenate`, ensure your dataset has enough samples to concatenate, or some prompts may be skipped.
- The output format is determined by the file extension of `OUTPUT_PATH` (e.g., `.jsonl`, `.csv`, `.parquet`).
- The prefix handling only trims prefixes. It doesn't expand them. Prefix buckets, if specified, only trim the given prefixes by bucket weighting. It doesn't generate unique prefixes for each bucket.
