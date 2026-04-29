# Embeddings Benchmarking Guide

GuideLLM supports benchmarking OpenAI-compatible embeddings endpoints to measure performance characteristics like throughput, latency, and concurrency.

## Overview

Embeddings models convert text into dense vector representations used for semantic search, retrieval, and similarity tasks. Unlike generative models that produce text output, embeddings models:

- Process input text and return vector embeddings
- Do not support streaming (single response per request)
- Track only input tokens (no output tokens)
- Measure request latency and throughput

## Quick Start

### Basic Embeddings Benchmark

```bash
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings \
  --data "prompt_tokens=128" \
  --max-requests 100 \
  --rate 10
```

### Using a Dataset

```bash
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --request-format embeddings \
  --data dataset.jsonl \
  --max-requests 50
```

## Dataset Format

Embeddings benchmarks accept datasets in JSONL format with text fields:

```jsonl
{"text": "What is the capital of France?"}
{"text": "Explain quantum computing in simple terms"}
{"text": "How does photosynthesis work?"}
```

Or use synthetic data generation:

```bash
# Fixed token count
--data "prompt_tokens=128"

# Random token counts
--data "prompt_tokens=64:256"
```

## Configuration Options

### Model Parameters

```bash
# Specify embedding model
--model "BAAI/bge-large-en-v1.5"

# Set encoding format (float or base64)
--backend-extra '{"encoding_format": "base64"}'

# Limit embedding dimensions
--backend-extra '{"dimensions": 512}'
```

### Benchmark Profiles

Embeddings support all standard benchmark profiles:

```bash
# Constant rate
--profile constant --rate 10

# Sweep multiple rates
--profile sweep --rate 1 5 10 20

# Maximum throughput
--profile throughput

# Synchronous (one request at a time)
--profile synchronous

# Poisson distribution
--profile poisson --rate 10
```

## Output Formats

### Console Output

```
ℹ Benchmark Summary
|===========|==========|==========|=====|=======|
| Strategy  | Requests | Latency  | Throughput   ||
|           | Total    | Mean (s) | Req/s | Tok/s |
|-----------|----------|----------|-------|-------|
| constant  | 100      | 0.125    | 8.5   | 1088  |
|===========|==========|==========|=====|=======|
```

### JSON Output

```bash
--outputs results.json
```

```json
{
  "benchmarks": [{
    "config": {
      "strategy": {"type_": "constant", "rate": 10}
    },
    "metrics": {
      "request_latency": {
        "mean": 0.125,
        "median": 0.120,
        "p95": 0.150
      },
      "requests_per_second": {
        "mean": 8.5
      },
      "input_tokens_per_second": {
        "mean": 1088
      }
    }
  }]
}
```

## Supported Embeddings Endpoints

GuideLLM works with any OpenAI-compatible embeddings API:

### OpenAI

```bash
guidellm benchmark \
  --target https://api.openai.com/v1 \
  --model "text-embedding-3-small" \
  --request-format embeddings \
  --backend-extra '{"api_key": "sk-..."}'
```

### vLLM

```bash
# Start vLLM server
vllm serve BAAI/bge-small-en-v1.5 --port 8000

# Benchmark
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings \
  --data "prompt_tokens=128" \
  --max-requests 100
```

### TEI (Text Embeddings Inference)

```bash
# Start TEI
docker run -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-small-en-v1.5

# Benchmark
guidellm benchmark \
  --target http://localhost:8080/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings
```

## Metrics Explained

### Request Latency

Time from sending request to receiving response (seconds).

```
Mean: Average latency across all requests
Median: Middle value (50th percentile)
p95: 95% of requests completed within this time
p99: 99% of requests completed within this time
```

### Throughput

```
Requests/second: Number of requests processed per second
Tokens/second: Input tokens processed per second
```

### Concurrency

Number of requests being processed simultaneously.

```
Mean: Average concurrent requests
Median: Typical concurrency level
p95: Peak concurrency during benchmark
```

## Example Workflows

### Capacity Planning

Find maximum sustainable throughput:

```bash
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings \
  --profile throughput \
  --max-requests 200
```

### Latency Analysis

Measure latency across different request rates:

```bash
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings \
  --profile sweep \
  --rate 1 5 10 20 50 \
  --max-requests 100
```

### Production Simulation

Simulate realistic production traffic with Poisson distribution:

```bash
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format embeddings \
  --profile poisson \
  --rate 15 \
  --max-duration 300
```

## Comparison with Generative Benchmarks

| Feature         | Embeddings          | Generative            |
| --------------- | ------------------- | --------------------- |
| Output          | Vector embeddings   | Generated text        |
| Streaming       | No                  | Yes                   |
| Output tokens   | 0 (not applicable)  | Variable              |
| TTFT            | N/A                 | Measured              |
| Token latency   | N/A                 | Measured              |
| Primary metrics | Latency, throughput | TTFT, ITL, throughput |

## Tips & Best Practices

1. **Use realistic token counts** - Match your production input sizes
2. **Warm up the server** - Run a small benchmark first to load the model
3. **Test multiple rates** - Use sweep profile to find optimal throughput
4. **Monitor resource usage** - Check CPU/GPU/memory during benchmarks
5. **Compare models** - Benchmark different embedding models with same data

## Troubleshooting

### "Handler not found for /v1/embeddings"

Ensure you're using:

```bash
--request-format embeddings
```

### Empty or invalid responses

Check that your endpoint supports the OpenAI embeddings API format:

- Request: `{"input": "text", "model": "..."}`
- Response: `{"data": [{"embedding": [...]}], "usage": {...}}`

### Low throughput

Try:

- Increasing batch size on server side
- Using a smaller model
- Checking server resource utilization
- Reducing input token counts

## See Also

- [Benchmark Profiles](benchmark-profiles.md) - Detailed explanation of all profile types
- [Datasets Guide](datasets.md) - Creating and using custom datasets
- [Metrics Guide](metrics.md) - Understanding performance metrics
