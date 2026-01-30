# Audio Benchmarking

This guide demonstrates how to benchmark audio models for tasks like Automatic Speech Recognition (ASR) (`audio/transcriptions`), Translation (`audio/translations`), and Audio Chat (`chat/completions`).

## Setup

First, ensure you have a running inference server and model compatible with the desired audio APIs. GuideLLM supports any OpenAI-compatible server that can handle audio inputs through chat/completions, audio/transcriptions, or audio/translations endpoints. For the benchmarking examples below, we’ll use vLLM serving a Whisper model for transcription and translation tasks and Ultravox for chat. Here are sample commands to start each of these servers:

```bash
# Whisper ASR/Translation
vllm serve openai/whisper-small

# Ultravox Audio Chat
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

Next, either on the same instance or another machine that can reach your server (recommended), install GuideLLM with audio support:

```bash
pip install guidellm[audio,recommended]
```

Finally, ensure you have a dataset with supported audio files for benchmarking. GuideLLM can handle audio data from Hugging Face datasets, local files, URLs, etc. For the examples below, we’ll use the `openslr/librispeech_asr` dataset.

## Processing Options

All of the standard arguments for benchmarking apply to audio tasks as well, such as `--profile`, `--rate`, and `--max-requests`, among others. There are a few additional options that help control audio-specific data handling and request formatting.

### Data Loading

GuideLLM supports multiple methods for loading audio data. First, the overall data source must be deserializable by GuideLLM into a Hugging Face dataset. This includes local files, Hugging Face datasets, JSON files, etc.

Next, the desired audio column within the deserializable data source must be supported by GuideLLM’s audio data decoder/encoder. Supported formats include:

- Hugging Face Audio feature (preferred)
- Local file paths (e.g., .wav, .mp3, .flac)
- URLs pointing to audio files
- Base64-encoded audio data
- Numpy or PyTorch arrays with raw audio samples

### Data Column Mapping

When specifying the dataset, generally, you will want to map the specific audio column to GuideLLM’s expected audio_column so it knows which data to process as audio. If nothing is specified, GuideLLM will attempt to auto-detect an audio column based on commonly used names such as audio, speech, wav, etc.

To specify the mapping, use the `--data-column-mapper` argument with a JSON string that specifies an existing column name for audio_column. For example, if your dataset has an audio column named speech_data, you would use:

```bash
--data-column-mapper '{"audio_column": "speech_data"}'
```

If you are combining multiple datasets (e.g., for prompts and audio), prepend the column name with the dataset index (starting at 0) or the dataset alias followed by a dot. For example, if the audio column is in the second dataset (index 1):

```bash
--data-column-mapper '{"1.audio_column": "speech_data"}'
```

### Request Formatting

Across the supported audio endpoints, a request formatter encodes audio data and formats the request payload. This uses reasonable defaults out of the box, but can be customized as needed. The following options are available for audio request formatting via the `--request-formatter-kwargs` argument, provided as a JSON string.

#### "encode_kwargs"

A dictionary of arguments passed to the audio encoder that controls how audio data is preprocessed before being included in the request.

**Note on Nesting:**

- For **Chat Completions** (`chat_completions`), audio arguments must be nested under an `audio` key within `encode_kwargs`.
- For **Transcription/Translation** (`audio_transcriptions`, `audio_translations`), arguments are provided at the top level of `encode_kwargs`.

Supported arguments include:

- "sample_rate": The sample rate of the input audio data. Only required if it cannot be inferred (e.g., for raw numpy/torch arrays).
- "encode_sample_rate": Target sample rate for the audio sent to the API. (default: 16000 Hz).
- "audio_format": File format for the payload. Supported formats are "wav", "mp3", and "flac" (default: "mp3").
- "bitrate": Bitrate for lossy formats like mp3 (default: "64k").
- "max_duration": If specified, audio longer than this duration (in seconds) will be truncated.
- "mono": Whether to convert audio to mono (default: True).
- "file_name": Optional file name to include in the request metadata (useful for endpoints that rely on filename extensions). Default is "audio.wav".

**Examples:**

For **Audio Transcription** (flat structure), converting to 16kHz WAV:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio_format": "wav", "encode_sample_rate": 16000}}'
```

For **Audio Chat** (nested structure), truncating to 30 seconds:

```bash
--request-formatter-kwargs '{"encode_kwargs": {"audio": {"max_duration": 30.0}}}'
```

#### "extras"

A dictionary of extra arguments to include directly in the request, enabling direct control over endpoint-specific parameters, such as language for Whisper models. Within extras, you can specify where to include the extra arguments:

- "headers": Include in request headers
- "params" / "body": Include in request parameters or body (auto-detected based on endpoint)

For example, to specify French as the target language for an audio translation request:

```bash
--request-formatter-kwargs '{"extras": {"body": {"language": "fr"}}}'
```

#### "stream"

Turn streaming responses on or off (if supported by the server) using a boolean value. By default, streaming is enabled.

```bash
--request-formatter-kwargs '{"stream": false}'
```

## Expected Results

GuideLLM captures comprehensive metrics across the entire request lifecycle, stored in `GenerativeRequestStats` and aggregated into `GenerativeMetrics`. Results are displayed in the console and saved to local files for further analysis.

### Output Files

- **`benchmarks.json`**: The complete hierarchical statistics object containing scheduler timings, request distributions, and detailed metric summaries for text and audio.
- **`benchmarks.csv`**: A row-per-request export of `GenerativeRequestStats`, useful for analyzing individual request performance, latency, and specific input/output configurations.
- **`benchmarks.html`**: A visual report summarizing performance.

### Captured Metrics

In addition to standard performance metrics like Latency, Time to First Token (TTFT), and Inter-Token Latency (ITL), audio benchmarks track specific usage metrics across Input and Output:

- **Audio Tokens**: Number of audio tokens processed (if supported by the model).
- **Audio Samples**: Count of raw audio samples.
- **Audio Seconds**: Total duration of audio content in seconds.
- **Audio Bytes**: Size of the audio payload in bytes.
- **Text Metrics**: Standard counts for Tokens, Words, and Characters are also tracked for transcriptions or chat responses.

### Statistical Analysis

For each metric above, GuideLLM calculates statistical distributions including:

- **Values**: Mean, Median, P95, P99, Min, Max.
- **Rates**: Throughput per second (e.g., `audio_seconds/sec`).
- **Concurrency**: Measures of concurrent active processing.

These use the `StatusDistributionSummary` structure to track Successful, Incomplete, and Errored requests separately.

## Examples

### 1. Audio Transcription (ASR)

This benchmark tests Automatic Speech Recognition (ASR) models, such as Whisper, for converting audio to text. Use the Whisper vLLM serving command above or a similar model that supports the audio transcription endpoint. For this example, we use only the audio data from the LibriSpeech dataset; however, a prompt can also be provided if desired, as shown in the Audio Chat example.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --request-type audio_transcriptions \
  --profile synchronous \
  --max-requests 20 \
  --data openslr/librispeech_asr \
  --data-args "{\"name\": \"clean\", \"split\": \"test\"}" \
  --data-column-mapper "{\"audio_column\": \"audio\"}"
```

**Key Parameters**

- `--target`: The base URL of the inference server (e.g., [http://localhost:8000](http://localhost:8000/)).
- `--request-type`: Specifies the API endpoint type, here audio_transcriptions for ASR.
- `--profile`: The load generation profile. synchronous runs requests sequentially.
- `--max-requests`: Limits the benchmark to 20 total requests.
- `--data`: The dataset identifier (openslr/librispeech_asr) to load from Hugging Face.
- `--data-args`: Configuration for the dataset loading, selecting the "clean" config and "test" split. See [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset) for full list of valid options.
- `--data-column-mapper`: Maps the dataset’s audio column to GuideLLM’s audio_column to ensure correct processing.

The above command benchmarks the audio/transcriptions endpoint on the target server using audio from the LibriSpeech dataset for ASR. It will result in an output similar to the following:

```bash
✔ OpenAIHTTPBackend backend validated with model openai/whisper-small

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7565.1 | 7329.1 | 16000.0 | 16000.0 | 129722.1 | 141848.5 | 6.4  | 16.8  | 75.3 | 72.9 | 52172.0 | 135692.0 | 610195.0 | 592749.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```

### 2. Audio Translation

This benchmark tests audio translation models like Whisper at converting audio in one language to text in another. Use the Whisper vLLM serving command above or a similar model that supports the audio translation endpoint. For this example, we use only the audio data from the LibriSpeech dataset; however, a prompt can also be provided if desired, as shown in the Audio Chat example.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --request-type audio_translations \
  --request-formatter-kwargs '{"extras": {"body": {"language": "fr"}}}' \
  --profile synchronous \
  --max-requests 20 \
  --data openslr/librispeech_asr \
  --data-args "{\"name\": \"clean\", \"split\": \"test\"}" \
  --data-column-mapper "{\"audio_column\": \"audio\"}"
```

**Key Parameters:**

- `--target`: The URL of the inference server.
- `--request-type`: audio_translations for the translation endpoint.
- `--request-formatter-kwargs`: Injects additional parameters into the request body. Here, it sets the target language to French (fr).
- `--profile`: synchronous execution mode.
- `--max-requests`: Limits the test to 20 requests.
- `--data`: Uses openslr/librispeech_asr as the source.
- `--data-args`: Selects the "clean" configuration and "test" split. See [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v4.5.0/en/package_reference/loading_methods#datasets.load_dataset) for full list of valid options.
- `--data-column-mapper`: Identifies the audio column for audio processing.

The above command benchmarks the audio/translations endpoint on the target server using audio from the LibriSpeech dataset and requesting translations to French. It will result in an output similar to the following:

```bash
✔ OpenAIHTTPBackend backend validated with model openai/whisper-small

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7483.6 | 7563.5 | 16000.0 | 16000.0 | 133404.0 | 146385.0 | 6.4  | 16.8  | 74.5 | 75.2 | 52172.0 | 135692.0 | 603620.5 | 611706.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

......
......

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html

```

### 3. Audio Chat Completions

This benchmark tests models that can handle audio inputs in a conversational format, such as Ultravox. Use the Ultravox vLLM serving command above, or a similar model that supports audio formats in chat-completion pathways. In addition to the LibriSpeech dataset, the following example adds a synthetic dataset for text prompts. Replace the datasets and column mappings as needed for your use case.

**Command:**

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --request-type chat_completions \
  --profile synchronous \
  --max-requests 20 \
  --data "prompt_tokens=256,output_tokens=128" \
  --data openslr/librispeech_asr \
  --data-args "{}" \
  --data-args "{\"name\": \"clean\", \"split\": \"test\"}" \
  --data-column-mapper "{\"audio_column\": \"1.audio\", \"text_column\": \"0.prompt\"}"
```

**Key Parameters**

- `--target`: The server URL.
- `--request-type`: chat_completions, supporting multimodal inputs (audio + text).
- `--profile`: synchronous execution.
- `--max-requests`: Limits to 20 requests.
- `--data`: Specified twice: first for synthetic prompt configuration (`prompt_tokens=256,output_tokens=128`), second for real audio from `openslr/librispeech_asr`.
- `--data-args`: Dataset arguments corresponding to the order of `--data` inputs (empty `{}` for synthetic prompts, LibriSpeech config second).
- `--data-column-mapper`: Maps audio from dataset index 1 (`"1.audio"`, LibriSpeech) and text from dataset index 0 (`"0.prompt"`, synthetic prompts) into each request.

The above command benchmarks the chat/completions endpoint on the target server using the prompt text from the synthetic dataset and audio from the LibriSpeech dataset. It will result in an output similar to the following:

```bash
✔ OpenAIHTTPBackend backend validated with model fixie-ai/ultravox-v0_5-llama-3_2-1b

......
......
✔ Setup complete, starting benchmarks...

......
......

ℹ Audio Metrics Statistics (Completed Requests)
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|
| Benchmark   | Input Tokens                  |||| Input Samples                        |||| Input Seconds           |||| Input Bytes                           ||||
| Strategy    | Per Request   || Per Second     || Per Request      || Per Second         || Per Request || Per Second || Per Request       || Per Second         ||
|             | Mdn   | p95    | Mdn    | Mean   | Mdn     | p95     | Mdn      | Mean     | Mdn  | p95   | Mdn  | Mean | Mdn     | p95      | Mdn      | Mean     |
|-------------|-------|--------|--------|--------|---------|---------|----------|----------|------|-------|------|------|---------|----------|----------|----------|
| synchronous | 642.0 | 1688.0 | 7565.1 | 7329.1 | 16000.0 | 16000.0 | 129722.1 | 141848.5 | 6.4  | 16.8  | 75.3 | 72.9 | 52172.0 | 135692.0 | 610195.0 | 592749.4 |
|=============|=======|========|========|========|=========|=========|==========|==========|======|=======|======|======|=========|==========|==========|==========|

ℹ GuideLLM Request Metrics Statistics (Completed Requests)
|=============|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|
| Benchmark   | Request Latency (ms)          ||||| Output Tokens / Sec          ||||| Time to First Token (ms)      ||||| Time per Output Token (ms)    |||||
| Strategy    | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 | Mdn   | Mean  | p50   | p90   | p95  | p99 |
|-------------|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|-------|-------|-------|-------|------|-----|
| synchronous | 125.4 | 130.2 | 125.4 | 145.1 | 150.2| 160.5| 45.2  | 44.8  | 45.2  | 42.1  | 41.5 | 40.2| 25.1  | 26.5  | 25.1  | 30.2  | 32.5 | 35.1| 22.1  | 22.3  | 22.1  | 23.7  | 24.1 | 24.8|
|=============|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|=======|=======|=======|=======|======|=====|

✔ Benchmarking complete, generated 1 benchmark(s)
…   json    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.json
…   csv     : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.csv
…   html    : /Users/markkurtz/code/github/vllm-project/guidellm/benchmarks.html
```
