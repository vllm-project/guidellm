# Embeddings Benchmarking Guide

GuideLLM supports benchmarking OpenAI-compatible embeddings endpoints to measure performance characteristics like throughput, latency, and concurrency.

## Overview

Embeddings models convert text into dense vector representations used for semantic search, retrieval, and similarity tasks. Unlike generative models that produce text output, embeddings models:

- Process input text and return vector embeddings
- Do not support streaming (single response per request)
- Track only input tokens (no output tokens)
- Measure request latency and throughput

## Quick Start

```bash
# Start vLLM server with an embeddings model
vllm serve BAAI/bge-small-en-v1.5 --port 8000

# Run benchmark
guidellm benchmark \
  --target http://localhost:8000/v1 \
  --model "BAAI/bge-small-en-v1.5" \
  --request-format /v1/embeddings \
  --data "prompt_tokens=128" \
  --max-requests 100
```

## Key Differences from Generative Benchmarks

| Feature         | Embeddings          | Generative            |
| --------------- | ------------------- | --------------------- |
| Output          | Vector embeddings   | Generated text        |
| Streaming       | No                  | Yes                   |
| Output tokens   | Not applicable      | Variable              |
| TTFT            | Not applicable      | Measured              |
| Token latency   | Not applicable      | Measured              |
| Primary metrics | Latency, throughput | TTFT, ITL, throughput |

## See Also

- [Benchmark Profiles](benchmark-profiles.md) - Detailed explanation of all profile types
- [Datasets Guide](datasets.md) - Creating and using custom datasets
- [Metrics Guide](metrics.md) - Understanding performance metrics
