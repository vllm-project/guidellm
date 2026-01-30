# Image Benchmarking

This guide demonstrates how to benchmark Vision-Language Models (VLMs) for tasks like Visual Question Answering (VQA) (`chat/completions`) and Image Captioning (`chat/completions`).

## Setup

First, ensure you have a running inference server and model compatible with the OpenAI Chat API. GuideLLM supports any OpenAI-compatible server that can handle image inputs through chat/completions endpoint. For the benchmarking examples below, we’ll use vLLM serving the Qwen3-VL model.

```bash
# Qwen2-VL VQA/Captioning
vllm serve Qwen/Qwen3-VL-2B-Instruct
```

Next, either on the same instance or another machine that can reach your server (recommended), install GuideLLM with vision support:

```bash
pip install guidellm[vision,recommended]
```

Finally, ensure you have a dataset with supported image files for benchmarking. GuideLLM can handle image data from Hugging Face datasets, local files, URLs, etc. For the examples below, we’ll use the `lmms-lab/MMBench_EN` dataset.

## Processing Options

All of the standard arguments for benchmarking apply to image tasks as well, such as `--profile`, `--rate`, and `--max-requests`, among others. There are a few additional options that help control image-specific data handling and request formatting.

### Data Loading

GuideLLM supports multiple methods for loading image data. First, the overall data source must be deserializable by GuideLLM into a Hugging Face dataset. This includes local files, Hugging Face datasets, JSON files, etc.

Next, the desired image column within the deserializable data source must be supported by GuideLLM’s image data decoder/encoder. Supported formats include:

- Hugging Face Image feature (preferred)
- Local file paths (e.g., .jpg, .png)
- URLs pointing to image files
- Base64-encoded image data
- Numpy arrays or PIL Images

### Data Column Mapping

When specifying the dataset, generally, you will want to map the specific image column to GuideLLM’s `image_column` so it knows which data to process as images. If nothing is specified, GuideLLM will attempt to auto-detect an image column based on commonly used names such as image, picture, etc.

To specify the mapping, use the `--data-column-mapper` argument with a JSON string that specifies an existing column name for `image_column`. For example, if your dataset has an image column named `photo`, you would use:

```bash
--data-column-mapper '{"image_column": "photo"}'
```

If you are combining multiple datasets (e.g., for prompts and images), prepend the column name with the dataset index (starting at 0) or the dataset alias followed by a dot. For example, if the image column is in the second dataset (index 1):

```bash
--data-column-mapper '{"1.image_column": "photo"}'
```

### Request Formatting

Across the supported endpoints, a request formatter encodes image data and formats the request payload. This uses reasonable defaults out of the box, but can be customized as needed. The following options are available for image request formatting via the `--request-formatter-kwargs` argument, provided as a JSON string.

#### "encode_kwargs"

A dictionary of arguments passed to the image encoder that controls how image data is preprocessed before being included in the request.

**Note on Nesting:**

- For **Chat Completions** (`chat_completions`), image arguments must be nested under an `image` key within `encode_kwargs`.

Supported arguments include:

- "max_size": Maximum size of the longest edge of the image (in pixels). Useful for downscaling large images while maintaining aspect ratio.
- "max_width": Maximum width of the image (in pixels). Maintain aspect ratio unless height is also specified.
- "max_height": Maximum height of the image (in pixels). Maintain aspect ratio unless width is also specified.
- "width": Force resize to specific width (in pixels).
- "height": Force resize to specific height (in pixels).
- "encode_type": How to include the image in the request.
    - "base64" (default): Downloads (if URL) or reads the image, processes it (resize), and encodes it as a base64 string.
    - "url": Sends the image URL directly in the request. This requires the inference server to have network access to the image URL. Resizing options cannot be used with this type.

**Examples:**

For **Chat Completions**, resizing images to a maximum of 512 pixels:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"image": {"max_size": 512}}}'
```

#### "extras"

A dictionary of extra arguments to include directly in the request. Within extras, you can specify where to include the extra arguments:

- "headers": Include in request headers
- "params" / "body": Include in request parameters or body

For example, to specify a specific system prompt or other body parameter:

```bash
--request-formatter-kwargs '{"extras": {"body": {"temperature": 0.7}}}'
```

#### "stream"

Turn streaming responses on or off (if supported by the server) using a boolean value. By default, streaming is enabled.

```bash
--request-formatter-kwargs '{"stream": false}'
```

## Expected Results

GuideLLM captures comprehensive metrics across the entire request lifecycle, stored in `GenerativeRequestStats` and aggregated into `GenerativeMetrics`. Results are displayed in the console and saved to local files for further analysis.

### Output Files

- **`benchmarks.json`**: The complete hierarchical statistics object containing scheduler timings, request distributions, and detailed metric summaries for text and images.
- **`benchmarks.csv`**: A row-per-request export of `GenerativeRequestStats`, useful for analyzing individual request performance, latency, and specific input/output configurations.
- **`benchmarks.html`**: A visual report summarizing performance.

### Captured Metrics

In addition to standard performance metrics like Latency, Time to First Token (TTFT), and Inter-Token Latency (ITL), image benchmarks track specific usage metrics across Input and Output:

- **Image Count**: Number of images processed.
- **Image Pixels**: Total number of pixels.
- **Image Bytes**: Size of image payloads in bytes.
- **Image Tokens**: Number of tokens used to represent the images (if available).
- **Text Metrics**: Standard counts for Tokens, Words, and Characters are also tracked for prompts and responses.

### Statistical Analysis

For each metric above, GuideLLM calculates statistical distributions including:

- **Values**: Mean, Median, P95, P99, Min, Max.
- **Rates**: Throughput per second (e.g., `image_pixels/sec`).
- **Concurrency**: Measures of concurrent active processing.

These use the `StatusDistributionSummary` structure to track Successful, Incomplete, and Errored requests separately.

## Examples

### 1. Visual Question Answering (VQA)

This benchmark tests Vision-Language Models for their ability to answer questions about images. We use the `lmms-lab/MMBench_EN` dataset which includes images and text questions.

**Command:**

```bash
guidellm benchmark \
  --target "http://192.168.4.12:8000" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --request-type chat_completions \
  --profile synchronous \
  --max-requests 20 \
  --data "lmms-lab/MMBench_EN" \
  --data-args "{\"split\": \"test\"}" \
  --data-column-mapper '{"image_column": "image", "text_column": "question"}'
```

**Key Parameters**

- `--target`: The base URL of the inference server.
- `--model`: The model name to use for requests.
- `--request-type`: chat_completions, supporting multimodal inputs.
- `--data`: The dataset identifier (lmms-lab/MMBench_EN).
- `--data-column-mapper`: Maps the dataset’s `image` column to `image_column` and `question` to `text_column`.

The above command benchmarks the chat/completions endpoint on the target server using the prompt text and image from the MMBench_EN dataset. It will result in an output similar to the following:

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```

### 2. Image Captioning

This benchmark tests the model's ability to describe an image without a specific question (or with a generic prompt if implied). Here we omit the text column, sending only the image.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --request-type chat_completions \
  --profile synchronous \
  --max-requests 20 \
  --data "lmms-lab/MMBench_EN" \
  --data-args "{\"split\": \"test\"}" \
  --data-column-mapper '{"image_column": "image"}'
```

**Key Parameters:**

- `--data-column-mapper`: Only maps `image_column`, implying an image-only request (or one where the model generates a caption/description).

The above command benchmarks the chat/completions endpoint on the target server using the prompt image from the MMBench_EN dataset. It will result in an output similar to the following:

```bash
✔ OpenAIHTTPBackend backend validated with model Qwen/Qwen3-VL-2B-Instruct

......
......

✔ Setup complete, starting benchmarks...

......
......

ℹ Image Metrics Statistics (Completed Requests)
|=============|=========|==========|=========|=========|=========|=========|========|=========|
| Benchmark   | Input Pixels                        |||| Input Bytes                       ||||
| Strategy    | Per Request       || Per Second       || Per Request      || Per Second      ||
|             | Mdn     | p95      | Mdn     | Mean    | Mdn     | p95     | Mdn    | Mean    |
|-------------|---------|----------|---------|---------|---------|---------|--------|---------|
| synchronous | 70064.0 | 120000.0 | 39775.8 | 46992.0 | 11656.0 | 37716.0 | 8779.7 | 10634.5 |
|=============|=========|==========|=========|=========|=========|=========|========|=========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```
