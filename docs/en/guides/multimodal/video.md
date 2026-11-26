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

The examples use [`ngqtrung/IMAVB`](https://huggingface.co/datasets/ngqtrung/IMAVB), which hosts individual MP4 files alongside its annotations. Use the `standard_vision` split for visual questions.

### Prepare IMAVB Prompts and URLs

IMAVB's `video` column contains repository-relative paths. This short script converts them to direct MP4 URLs and combines each question with its six answer options. It downloads only the annotations and writes `imavb.jsonl`; no video download or extraction is needed during preparation.

```python
import json

from datasets import load_dataset
from huggingface_hub import hf_hub_url

repo_id = "ngqtrung/IMAVB"
annotations = load_dataset(repo_id, split="standard_vision")
with open("imavb.jsonl", "w", encoding="utf-8") as output:
    for row in annotations:
        options = "\n".join(
            f"{letter.upper()}. {row[f'option_{letter}']}" for letter in "abcdef"
        )
        record = {
            "video": hf_hub_url(repo_id, row["video"], repo_type="dataset"),
            "prompt": f"{row['question']}\n{options}",
            "caption": "Describe the events in this video.",
            "output_tokens_count": 128,
        }
        output.write(json.dumps(record) + "\n")
```

The examples below use `encode_type=url`, so the inference server fetches the MP4 files directly from Hugging Face. It must support remote video URLs and have network access to them. With the default `encode_type=base64`, GuideLLM downloads the videos and embeds their bytes in requests instead.

## Processing Options

All of the standard arguments for benchmarking apply to video tasks as well, such as `--profile`, profile rate parameters, and `--constraint kind=max_requests,count=<n>`, among others. There are a few additional options that help control video-specific data handling and request formatting.

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
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"video_column": "url"}}'
```

If you are combining multiple datasets (e.g., for prompts and video), prepend the column name with the dataset index (starting at 0) or the dataset alias followed by a dot. For example, if the video column is in the second dataset (index 1):

```bash
--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"1.video_column": "url"}}'
```

### Media Encoding and Request Options

Video data is encoded by the `encode_media` data preprocessor, which is part of the default data pipeline. Pass `video_kwargs` to control how videos are preprocessed before they are included in requests. To send a direct MP4 URL, use `encode_type=url`. This single-clip example uses an MP4 hosted by IMAVB and can be run without the preparation step above:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=1 \
  --data '{"kind":"in_memory_dict_list","data":[{"video":"https://huggingface.co/datasets/ngqtrung/IMAVB/resolve/main/videos/--aqjaJyZLk.mp4","prompt":"Describe the events in this video.","output_tokens_count":128}]}' \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

Or using JSON:

```bash
--data-preprocessor '{"kind":"encode_media","video_kwargs":{"encode_type":"url"}}'
```

Supported `video_kwargs` include:

- `encode_type`: `base64` (default) reads local files or downloads URL inputs and encodes the video bytes; `url` sends the URL directly. The inference server must have network access to the URL when using `url`. Use direct media URLs, not YouTube watch-page links or ZIP archive URLs.

> [!WARNING]\
> Specifying any `--data-preprocessor` replaces the default preprocessors except for the column mapper. Include every preprocessor the run needs, for example `--data-preprocessor kind=encode_media,video_kwargs.encode_type=url`.

Request body parameters and headers are configured on the backend `extras` field. For example, to set `temperature` in the request body:

```bash
--backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"temperature":0.7}}}'
```

`extras` supports the `body`, `content`, `headers`, `params`, and `files` fields; see [Structured Chat Content Payloads](../backends.md#structured-chat-content-payloads).

To disable streaming responses (when the backend supports it), pass `stream=false` in the backend configuration:

```bash
--backend kind=openai_http,target=http://localhost:8000,stream=false
```

## Expected Results

GuideLLM captures comprehensive metrics across the entire request lifecycle, stored in `GenerativeRequestStats` and aggregated into `GenerativeMetrics`. Results are displayed in the console and saved to local files for further analysis.

### Output Files

- **`benchmarks.json`**: The complete hierarchical statistics object containing scheduler timings, request distributions, and detailed metric summaries for text and videos.
- **`benchmarks.csv`**: A one-row-per-benchmark export of summary metrics and distributions, useful for comparing strategies and sharing results.
- **`benchmarks.html`**: An optional visual report summarizing performance, written when `--output kind=html` is requested.

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

This benchmark measures performance while answering IMAVB's visual questions. Use `imavb.jsonl` from the preparation step above. GuideLLM measures inference performance; it does not score the answers against the dataset's answer key.

**Command:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data kind=json_file,path=imavb.jsonl \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

**Key Parameters**

- `--backend`: Server URL, model, and chat completions endpoint
- `--data`: Loads the prepared JSONL file, with direct MP4 URLs, questions and options, and a requested output limit of 128 tokens
- Column mapping is automatic: `video` maps to `video_column`, `prompt` to `text_column`, and `output_tokens_count` to `output_tokens_count_column`
- `--data-preprocessor`: Passes each MP4 URL to the inference server

### 2. Video Captioning

This benchmark uses the same `imavb.jsonl` file, selecting the generic `caption` prompt instead of the question and options.

**Command:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-VL-2B-Instruct,request_format=/v1/chat/completions \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=50 \
  --data kind=json_file,path=imavb.jsonl \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"video_column":"video","text_column":"caption","output_tokens_count_column":"output_tokens_count"}}' \
  --data-preprocessor kind=encode_media,video_kwargs.encode_type=url
```

**Key Parameters:**

- `--data-column-mapper`: Selects the video URL, generic description prompt, and requested output limit from each row
- `--data-preprocessor`: Passes each MP4 URL to the inference server
