# Embeddings Benchmarking

GuideLLM supports benchmarking embeddings models that process text and return vector representations. This guide covers how to benchmark embeddings endpoints using GuideLLM.

## Overview

Embeddings benchmarking in GuideLLM:

- Tests `/v1/embeddings` endpoints (OpenAI-compatible)
- Measures input token processing performance (no output tokens)
- Tracks encoding format usage (float vs base64)
- Supports synthetic and real text datasets
- Reports latency, throughput, and token processing metrics

## Quick Start

The simplest way to run an embeddings benchmark:

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "prompt_tokens=100" \
  --max-requests 50
```

This command:

- Targets an embeddings endpoint at `http://localhost:8000`
- Uses synthetic data with 100 tokens per prompt
- Runs 50 requests
- Outputs results to console and JSON

## Data Sources

### Synthetic Data

Generate synthetic text data with specific token counts:

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "prompt_tokens=256" \
  --max-requests 100
```

You can also use a YAML config file for more control:

```yaml
# embeddings_config.yaml
prompt_tokens: 512
prompt_tokens_stdev: 50
prompt_tokens_min: 400
prompt_tokens_max: 600
output_tokens: 0
turns: 1
```

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data embeddings_config.yaml \
  --max-requests 100
```

### Real Datasets

Use actual text data from JSON or JSONL files:

```json
[
  {"text": "What is artificial intelligence?"},
  {"text": "Explain machine learning"},
  {"text": "How does deep learning work?"}
]
```

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data texts.json \
  --max-requests 100
```

### HuggingFace Datasets

Load datasets from HuggingFace Hub:

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "hf://sentence-transformers/embedding-training-data" \
  --max-requests 100
```

## Load Profiles

Control how requests are sent to the embeddings endpoint:

### Constant Rate

Send requests at a constant rate (requests per second):

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "prompt_tokens=100" \
  --profile constant \
  --rate 10 \
  --max-duration 30
```

### Sweep Profile

Test multiple request rates to find performance limits:

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "prompt_tokens=100" \
  --profile sweep \
  --rate 5 \
  --max-requests 100
```

This runs 5 test points at increasing rates to map throughput vs latency.

### Synchronous Mode

Send one request at a time (for baseline measurements):

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model text-embedding-3-small \
  --data "prompt_tokens=100" \
  --profile synchronous \
  --max-requests 20
```

## Metrics

Embeddings benchmarks report the following metrics:

### Request Metrics

- **request_latency**: End-to-end request latency distribution
- **request_concurrency**: Number of concurrent requests over time
- **requests_per_second**: Throughput (requests/sec)

### Token Metrics

- **input_tokens_count**: Total input tokens processed
- **input_tokens_per_second**: Input token throughput

Note: Embeddings have **no output tokens** (only input processing).

### Encoding Format

- **encoding_format_breakdown**: Count of requests by encoding format

### Example Output

```json
{
  "metrics": {
    "request_totals": {
      "successful": 100,
      "errored": 0,
      "incomplete": 0
    },
    "request_latency": {
      "successful": {
        "mean": 0.085,
        "p50": 0.082,
        "p95": 0.120,
        "p99": 0.145
      }
    },
    "input_tokens_count": {
      "successful": 10000
    },
    "input_tokens_per_second": {
      "successful": {
        "mean": 11764.7
      }
    },
    "encoding_format_breakdown": {
      "float": 100
    }
  }
}
```

## Complete Example

Here's a complete example benchmarking an embeddings endpoint:

### 1. Start an Embeddings Server

Using vLLM:

```bash
vllm serve BAAI/bge-base-en-v1.5 \
  --port 8000 \
  --task embed
```

### 2. Prepare Test Data

Create `test_embeddings.json`:

```json
[
  {"text": "Natural language processing techniques"},
  {"text": "Machine learning model optimization"},
  {"text": "Deep learning architectures"},
  {"text": "Transformer models for NLP"}
]
```

### 3. Run Benchmark

```bash
guidellm benchmark run-embeddings \
  --target http://localhost:8000 \
  --model BAAI/bge-base-en-v1.5 \
  --data test_embeddings.json \
  --profile sweep \
  --rate 5 \
  --max-requests 100 \
  --encoding-format float \
  --outputs json,html \
  --output-dir ./results
```

### 4. Review Results

Check the generated report:

```bash
cat ./results/embeddings_benchmarks.json
```

Or open the HTML report in a browser:

```bash
open ./results/embeddings_benchmarks.html
```

## Comparison with Generative Benchmarks

| Feature   | Embeddings        | Generative                                |
| --------- | ----------------- | ----------------------------------------- |
| Endpoint  | `/v1/embeddings`  | `/v1/chat/completions`, `/v1/completions` |
| Output    | Vector embeddings | Generated text                            |
| Tokens    | Input only        | Input + output                            |
| Streaming | No                | Optional                                  |
| Encoding  | Float or base64   | N/A                                       |
| TTFT      | N/A               | Measured                                  |
| ITL       | N/A               | Measured                                  |

## See Also

- [Architecture Guide](architecture.md) - Understanding GuideLLM's design
- [Metrics Guide](metrics.md) - Detailed metrics explanations
- [Datasets Guide](datasets.md) - Working with different data sources
- [Getting Started](../getting-started/index.md) - Basic GuideLLM usage
