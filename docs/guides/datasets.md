# Datasets

GuideLLM supports various dataset configurations to enable benchmarking and evaluation of large language models (LLMs). This document provides a comprehensive guide to configuring datasets for different use cases, along with detailed examples and rationale for choosing specific pathways.

## Data Arguments Overview

The following arguments can be used to configure datasets and their processing:

- `--data`: Specifies the dataset source. This can be a file path, Hugging Face dataset ID, synthetic data configuration, or in-memory data.
- `--data-args`: A JSON string or dictionary argument that allows you to control how datasets are parsed and prepared. This includes specific aliases for GuideLLM flows, such as:
  - `prompt_column`: Specifies the column name for the prompt. By default, GuideLLM will try the most common column names (e.g., `prompt`, `text`, `input`).
  - `prompt_tokens_count_column`: Specifies the column name for the prompt token count. These are used to set the request prompt token count for counting metrics. By default, GuideLLM assumes no token count is provided.
  - `output_tokens_count_column`: Specifies the column name for the output token count. These are used to set the request output token count for the request and counting metrics. By default, GuideLLM assumes no token count is provided.
  - `split`: Specifies the dataset split to use (e.g., `train`, `val`, `test`). By default, GuideLLM will try the most common split names (e.g., `train`, `validation`, `test`) if the dataset has splits, otherwise it will use the entire dataset.
  - Any remaining arguments are passed directly into the dataset constructor as kwargs.
- `--data-sampler`: Specifies the sampling strategy for datasets. By default, no sampling is applied. When set to `random`, it enables random shuffling of the dataset, which can be useful for creating diverse batches during benchmarking.
- `--processor`: Specifies the processor or tokenizer to use. This is only required for synthetic data generation or when local calculations are specified through configuration settings. By default, the processor is set to the `--model` argument. If `--model` is not supplied, it defaults to the model retrieved from the backend.
- `--processor-args`: A JSON string containing any arguments to pass to the processor or tokenizer constructor. These arguments are passed as a dictionary of kwargs.

### Example Usage

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data "path/to/dataset|dataset_id" \
    --data-args '{"prompt_column": "prompt", "split": "train"}' \
    --processor "path/to/processor" \
    --processor-args '{"arg1": "value1"}' \
    --data-sampler "random"
```

## Dataset Types

GuideLLM supports several types of datasets, each with its own advantages and use cases. Below are the main dataset types supported by GuideLLM, including synthetic data, Hugging Face datasets, file-based datasets, and in-memory datasets.

### Synthetic Data

Synthetic datasets allow you to generate data on the fly with customizable parameters. This is useful for controlled experiments, stress testing, and simulating specific scenarios. For example, you might want to evaluate how a model handles long prompts or generates outputs with specific characteristics.

#### Example Commands

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data "prompt_tokens=256,output_tokens=128"
```

Or using a JSON string:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data '{"prompt_tokens": 256, "output_tokens": 128}'
```

#### Configuration Options

- `prompt_tokens`: Average number of tokens in prompts. If nothing else is specified, all requests will have this number of tokens.
- `prompt_tokens_stdev`: Standard deviation for prompt tokens. If not supplied and min/max are not specified, no deviation is applied. If not supplied and min/max are specified, a uniform distribution is used.
- `prompt_tokens_min`: Minimum number of tokens in prompts. If unset and `prompt_tokens_stdev` is set, the minimum is 1.
- `prompt_tokens_max`: Maximum number of tokens in prompts. If unset and `prompt_tokens_stdev` is set, the maximum is 5 times the standard deviation.
- `output_tokens`: Average number of tokens in outputs. If nothing else is specified, all requests will have this number of tokens.
- `output_tokens_stdev`: Standard deviation for output tokens. If not supplied and min/max are not specified, no deviation is applied. If not supplied and min/max are specified, a uniform distribution is used.
- `output_tokens_min`: Minimum number of tokens in outputs. If unset and `output_tokens_stdev` is set, the minimum is 1.
- `output_tokens_max`: Maximum number of tokens in outputs. If unset and `output_tokens_stdev` is set, the maximum is 5 times the standard deviation.
- `samples`: Number of samples to generate (default: 1000). More samples will increase the time taken to generate the dataset before benchmarking, but will also decrease the likelihood of caching requests.
- `source`: Source text for generation (default: `data:prideandprejudice.txt.gz`). This can be any text file, URL containing a text file, or a compressed text file. The text is used to sample from at a word and punctuation granularity and then combined into a single string of the desired lengths.

#### Notes

- A processor/tokenizer is required. By default, the model passed in or retrieved from the server is used. If unavailable, use the `--processor` argument to specify a directory or Hugging Face model ID containing the processor/tokenizer files.

### Hugging Face Datasets

GuideLLM supports datasets from the Hugging Face Hub or local directories that follow the `datasets` library format. This allows you to easily leverage a wide range of datasets for benchmarking and evaluation with real-world data.

#### Example Commands

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data "garage-bAInd/Open-Platypus"
```

Or using a local dataset:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data "path/to/dataset"
```

#### Notes

- Hugging Face datasets can be specified by ID, a local directory, or a path to a local Python file.
- A supported Hugging Face datasets format is defined as one that can be loaded using the `datasets` library with the `load_dataset` function and therefore it is representable as a `Dataset`, `DatasetDict`, `IterableDataset`, or `IterableDatasetDict`. More information on the supported data types and additional args for the underlying use of `load_dataset` can be found in the [Hugging Face datasets documentation](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub).
- A processor/tokenizer is only required if `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` or `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"` is set in the environment. In this case, the processor/tokenizer must be specified using the `--processor` argument. If not set, the processor/tokenizer will be set to the model passed in or retrieved from the server.

### File-Based Datasets

GuideLLM supports various file formats for datasets, including text, CSV, JSON, and more. These datasets can be used for benchmarking and evaluation, allowing you to work with structured data in a familiar format that matches your use case.

#### Supported Formats with Examples

- **Text files (`.txt`, `.text`)**: Where each line is a separate prompt to use.
  ```
  Hello, how are you?
  What is your name?
  ```
- **CSV files (`.csv`)**: Where each row is a separate dataset entry and the first row contains the column names. The columns should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional columns can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```csv
  prompt,output_tokens_count,additional_column,additional_column2
  Hello, how are you?,5,foo,bar
  What is your name?,3,baz,qux
  ```
- **JSON Lines files (`.jsonl`)**: Where each line is a separate JSON object. The objects should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional fields can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```json
  {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"}
  {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
  ```
- **JSON files (`.json`)**: Where the entire dataset is represented as a JSON array of objects nested under a specific key. To surface the correct key to use, a `--data-args` argument must be passed in of `"field": "NAME"` for where the array exists. The objects should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional fields can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```json
  {
    "version": "1.0",
    "data": [
      {"prompt": "Hello, how are you?", "output_tokens_count": 5, "additional_column": "foo", "additional_column2": "bar"},
      {"prompt": "What is your name?", "output_tokens_count": 3, "additional_column": "baz", "additional_column2": "qux"}
    ]
  }
  ```
- **Parquet files (`.parquet`)** Example: A binary columnar storage format for efficient data processing. For more information on the supported formats, see the Hugging Face dataset documentation linked in the [Notes](#notes) section.
- **Arrow files (`.arrow`)** Example: A cross-language development platform for in-memory data. For more information on the supported formats, see the Hugging Face dataset documentation linked in the [Notes](#notes) section.
- **HDF5 files (`.hdf5`)** Example: A hierarchical data format for storing large amounts of data. For more information on the supported formats, see the Hugging Face dataset documentation linked in the [Notes](#notes) section.

#### Example Commands

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --rate-type "throughput" \
    --max-requests 1000 \
    --data "path/to/dataset.ext" \
    --data-args '{"prompt_column": "prompt", "split": "train"}'
```

Where `.ext` can be any of the supported file formats listed above.

#### Notes

- Ensure the file format matches the expected structure for the dataset and is listed as a supported format.
- The `--data-args` argument can be used to specify additional parameters for parsing the dataset, such as the prompt column name or the split to use.
- A processor/tokenizer is only required if `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` or `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"` is set in the environment. In this case, the processor/tokenizer must be specified using the `--processor` argument. If not set, the processor/tokenizer will be set to the model passed in or retrieved from the server.
- More information on the supported formats and additional args for the underlying use of `load_dataset` can be found in the [Hugging Face datasets documentation](https://huggingface.co/docs/datasets/en/loading#local-and-remote-files).

### In-Memory Datasets

In-memory datasets allow you to directly pass data as Python objects, making them ideal for quick prototyping and testing without the need to save data to disk.

#### Supported Formats with Examples

- **Dictionary of columns and values**: Where each key is a column name and the values are lists of data points. The keys should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional columns can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```python
  {
      "column1": ["value1", "value2"],
      "column2": ["value3", "value4"]
  }
  ```
- **List of dictionaries**: Where each dictionary represents a single data point with key-value pairs. The dictionaries should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional fields can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```python
  [
      {"column1": "value1", "column2": "value3"},
      {"column1": "value2", "column2": "value4"}
  ]
  ```
- **List of items**: Where each item is a single data point. The items should include `prompt` or other common names for the prompt which will be used as the prompt column. Additional fields can be included based on the previously mentioned aliases for the `--data-args` argument.
  ```python
  [
      "value1",
      "value2",
      "value3"
  ]

  ```

#### Example Usage

```python
from guidellm.benchmark import benchmark_generative_text

data = [
    {"prompt": "Hello", "output": "Hi"},
    {"prompt": "How are you?", "output": "I'm fine."}
]

benchmark_generative_text(data=data, ...)
```

#### Notes

- Ensure that the data format is consistent and adheres to one of the supported structures.
- For dictionaries, all columns must have the same number of samples.
- For lists of dictionaries, all items must have the same keys.
- For lists of items, all elements must be of the same type.
- A processor/tokenizer is only required if `GUIDELLM__PREFERRED_PROMPT_TOKENS_SOURCE="local"` or `GUIDELLM__PREFERRED_OUTPUT_TOKENS_SOURCE="local"` is set in the environment. In this case, the processor/tokenizer must be specified using the `--processor` argument. If not set, the processor/tokenizer will be set to the model passed in or retrieved from the server.
