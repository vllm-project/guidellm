# Video Benchmarking

This guide demonstrates how to benchmark Video-Language Models (Video-LLMs) for tasks like Video Question Answering (Video QA) (`chat/completions`) and Video Captioning (`chat/completions`).

## Setup

First, ensure you have a running inference server and model compatible with the OpenAI Chat API. GuideLLM supports any OpenAI-compatible server that can handle video inputs through the `chat/completions` endpoint. For the benchmarking examples below, we’ll use vLLM serving the Qwen3-VL model.

```bash
# Qwen3-VL Video QA/Captioning
vllm serve Qwen/Qwen3-VL-2B-Instruct
```

Next, either on the same instance or another machine that can reach your server (recommended), install GuideLLM with vision support:

```bash
pip install guidellm[vision,recommended]
```

Finally, ensure you have a dataset with supported video files for benchmarking. GuideLLM can handle video data from Hugging Face datasets, local files, URLs, etc. For the examples below, we’ll use the `lmms-lab/Video-MME` dataset.

## Processing Options

All of the standard arguments for benchmarking apply to video tasks as well, such as `--profile`, `--rate`, and `--max-requests`, among others. There are a few additional options that help control video-specific data handling and request formatting.

### Data Loading

GuideLLM supports multiple methods for loading video data. First, the overall data source must be deserializable by GuideLLM into a Hugging Face dataset. This includes local files, Hugging Face datasets, JSON files, etc.

Next, the desired video column within the deserializable data source must be supported by GuideLLM’s video data decoder/encoder. Supported formats include:

- Hugging Face Video feature
- Local file paths
- URLs pointing to video files
- Base64-encoded video data
- Raw video bytes

### Data Column Mapping

When specifying the dataset, generally, you will want to map the specific video column to GuideLLM’s `video_column` so it knows which data to process as video. If nothing is specified, GuideLLM will attempt to auto-detect a video column based on commonly used names such as video, clip, etc.

To specify the mapping, use the `--data-column-mapper` argument with a JSON string that specifies an existing column name for `video_column`. For example, if your dataset has a video column named `url`, you would use:

```bash
--data-column-mapper '{"video_column": "url"}'
```

If you are combining multiple datasets (e.g., for prompts and video), prepend the column name with the dataset index (starting at 0) or the dataset alias followed by a dot. For example, if the video column is in the second dataset (index 1):

```bash
--data-column-mapper '{"1.video_column": "url"}'
```

### Request Formatting

Across the supported endpoints, a request formatter encodes video data and formats the request payload. This uses reasonable defaults out of the box, but can be customized as needed. The following options are available for video request formatting via the `--request-formatter-kwargs` argument, provided as a JSON string.

#### "encode_kwargs"

A dictionary of arguments passed to the video encoder that controls how video data is preprocessed before being included in the request.

**Note on Nesting:**

- For **Chat Completions** (`chat_completions`), video arguments must be nested under a `video` key within `encode_kwargs`.

Supported arguments include:

- "encode_type": How to include the video in the request.
  - "base64" (default): Downloads (if URL) or reads the video and encodes it as a base64 string.
  - "url": Sends the video URL directly in the request. This requires the inference server to have network access to the video URL.

**Examples:**

For **Chat Completions**, sending video URLs directly:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"video": {"encode_type": "url"}}}'
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

- **`benchmarks.json`**: The complete hierarchical statistics object containing scheduler timings, request distributions, and detailed metric summaries for text and videos.
- **`benchmarks.csv`**: A row-per-request export of `GenerativeRequestStats`, useful for analyzing individual request performance, latency, and specific input/output configurations.
- **`benchmarks.html`**: A visual report summarizing performance.

### Captured Metrics

In addition to standard performance metrics like Latency, Time to First Token (TTFT), and Inter-Token Latency (ITL), video benchmarks track specific usage metrics across Input and Output:

- **Video Frames**: Number of frames detected/processed.
- **Video Seconds**: Duration of the video content.
- **Video Bytes**: Size of the video payload.
- **Video Tokens**: If available from the model/server, the number of video tokens processed.
- **Text Metrics**: Standard counts for Tokens, Words, and Characters are also tracked for prompts and responses.

### Statistical Analysis

For each metric above, GuideLLM calculates statistical distributions including:

- **Values**: Mean, Median, P95, P99, Min, Max.
- **Rates**: Throughput per second (e.g., `video_frames/sec`).
- **Concurrency**: Measures of concurrent active processing.

These use the `StatusDistributionSummary` structure to track Successful, Incomplete, and Errored requests separately.

## Examples

### 1. Video Question Answering (Video QA)

This benchmark tests Video-Language Models for their ability to answer questions about videos. We use the `lmms-lab/Video-MME` dataset which includes videos and text questions.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --request-type chat_completions \
  --profile synchronous \
  --max-requests 50 \
  --data "lmms-lab/Video-MME" \
  --data-args "{\"split\": \"test\"}" \
  --data-column-mapper '{"video_column": "url", "text_column": "question"}'
```

**Key Parameters**

- `--target`: The base URL of the inference server.
- `--model`: The model name to use for requests.
- `--request-type`: chat_completions, supporting multimodal inputs.
- `--data`: The dataset identifier (lmms-lab/Video-MME). 
- `--data-args`: Configuration for the dataset loading, selecting the "test" split. See [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset) for full list of valid options.
- `--data-column-mapper`: Maps the dataset’s `url` column (containing the video link) to `video_column` and `question` to `text_column`.

### 2. Video Captioning

This benchmark tests the model's ability to describe a video without a specific question (or with a generic prompt if implied). Here we omit the text column, sending only the video.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --request-type chat_completions \
  --profile synchronous \
  --max-requests 50 \
  --data "lmms-lab/Video-MME" \
  --data-args "{\"split\": \"test\"}" \
  --data-column-mapper '{"video_column": "url"}'
```

**Key Parameters:**

- `--data-column-mapper`: Only maps `video_column`, implying a video-only request (or one where the model generates a caption/description).
