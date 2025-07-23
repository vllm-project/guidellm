# Preprocess Commands

GuideLLM provides preprocessing capabilities to transform and prepare data for benchmarking workflows. The preprocess module includes tools for creating datasets from existing benchmark results, enabling "apples-to-apples" comparisons and reusable benchmark datasets.

## Overview

The `guidellm preprocess` command provides utilities to:

- **Extract datasets from benchmark results**: Convert completed benchmark reports into reusable datasets with known prompt and output token counts for consistent comparisons


## Commands

### `dataset-from-file`

Extracts prompts and their corresponding output token counts from saved benchmark report files to create datasets for future benchmarking runs.

#### Purpose

When you run a benchmark with GuideLLM, you get detailed results about how a model performed with specific prompts. The `dataset-from-file` command allows you to extract those successful prompt-response pairs and convert them into a standardized dataset format. This enables:

1. **Consistent Comparisons**: Use the exact same prompts across different models or configurations
2. **Known Expectations**: Each prompt comes with its expected output token count
3. **Reproducible Benchmarks**: Eliminate variability from different prompts when comparing models

#### Syntax

```bash
guidellm preprocess dataset-from-file [OPTIONS] BENCHMARK_FILE
```

#### Arguments

- `BENCHMARK_FILE`: Path to the saved benchmark report file (JSON format)

#### Options

- `-o, --output-path PATH`: Output dataset file path (default: `dataset_from_benchmark.json`)
- `--show-stats`: Show dataset statistics after creation
- `--disable-console-outputs`: Disable console output for silent operation
- `--help`: Show help message and exit

#### Example Usage

##### Basic Usage

```bash
# Convert a benchmark report to a dataset
guidellm preprocess dataset-from-file benchmark-results.json

# Specify custom output path
guidellm preprocess dataset-from-file benchmark-results.json -o my_dataset.json

# Show statistics about the created dataset
guidellm preprocess dataset-from-file benchmark-results.json --show-stats
```

#### Input File Requirements

The input benchmark file must be a valid GuideLLM benchmark report containing:

- **Valid JSON format**: The file must be properly formatted
- **Benchmark report structure**: Must contain the expected benchmark report schema
- **Successful requests**: Must contain at least one successful request to extract data from

##### Supported Input Formats

```json
{
  "benchmarks": [
    {
      "requests": {
        "successful": [
          {
            "prompt": "What is the capital of France?",
            "output_tokens": 5,
            "... other request fields ..."
          }
        ],
        "errored": [],
        "incomplete": []
      }
    }
  ]
}
```

#### Output Format

The generated dataset follows this structure:

```json
{
  "version": "1.0",
  "description": "Dataset created from benchmark results for apples-to-apples comparisons",
  "data": [
    {
      "prompt": "What is the capital of France?",
      "output_tokens_count": 5,
      "prompt_tokens_count": 12
    },
    {
      "prompt": "Explain quantum computing in simple terms.",
      "output_tokens_count": 45,
      "prompt_tokens_count": 8
    }
  ]
}
```


Each data item contains:
- `prompt`: The original prompt text
- `output_tokens_count`: The number of tokens in the model's response
- `prompt_tokens_count`: The number of tokens in the original prompt

#### Statistics Output

When using `--show-stats`, you'll see detailed information about the created dataset:

```
Dataset Statistics:
==================
Total items: 95
Prompt length statistics:
  Min: 8 characters
  Max: 245 characters  
  Mean: 87.3 characters
Output tokens statistics:
  Min: 1 tokens
  Max: 512 tokens
  Mean: 124.8 tokens
```

