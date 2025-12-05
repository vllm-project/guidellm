# Embeddings Benchmarking

GuideLLM supports benchmarking embedding models through the `/v1/embeddings` endpoint. This guide covers how to set up and run benchmarks for text embedding models, which are commonly used for semantic search, clustering, and other ML tasks.

## Overview

Embedding models convert text into dense vector representations that capture semantic meaning. Benchmarking these models helps you:

- Measure throughput and latency for embedding generation
- Test performance under different load conditions
- Compare different embedding model deployments
- Optimize your embedding service configuration

## Supported Backends

### vLLM

vLLM supports embedding models starting from version 0.4.0. To serve an embedding model with vLLM:

```bash
vllm serve "BAAI/bge-small-en-v1.5"
```

Popular embedding models supported by vLLM:

- **BAAI/bge-small-en-v1.5**: Lightweight English embedding model (384 dimensions)
- **BAAI/bge-base-en-v1.5**: Base English embedding model (768 dimensions)
- **BAAI/bge-large-en-v1.5**: Large English embedding model (1024 dimensions)
- **sentence-transformers/all-MiniLM-L6-v2**: Compact multilingual model (384 dimensions)
- **intfloat/e5-large-v2**: High-performance English model (1024 dimensions)

For the latest list of supported models, see the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).

### OpenAI API

GuideLLM can also benchmark OpenAI's embedding endpoints:

```bash
guidellm benchmark \
    --target "https://api.openai.com" \
    --request-type embeddings \
    --model "text-embedding-3-small" \
    --rate 5 \
    --max-requests 50 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "gpt2"
```

Note: You'll need to set your OpenAI API key as an environment variable or in the request headers.

## Basic Benchmarking

### Simple Concurrent Benchmark (Recommended)

For embeddings, concurrent testing is the most relevant approach. To run a basic concurrent benchmark with synthetic data:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 32 \
    --max-requests 100 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

This command:

- Tests with 32 concurrent requests (parallel processing)
- Stops after 100 total requests
- Uses synthetic text with ~256 tokens per request
- Uses the bge-small tokenizer for token counting
- **Note**: `output_tokens=1` is required when using synthetic data, even though embeddings don't generate output. This is a current limitation of the synthetic data generator.

## Benchmark Profiles for Embeddings

Different benchmark profiles serve different purposes when testing embedding models:

- **Concurrent** (Recommended): Tests parallel request handling - the most common production pattern for embeddings
- **Throughput**: Finds maximum sustainable request rate - useful for capacity planning
- **Synchronous**: Sequential baseline testing - useful for measuring per-request latency without concurrency effects
- **Constant**: Fixed-rate testing - less relevant for embeddings since they have predictable processing times
- **Sweep**: Not recommended for embeddings (designed for optimizing generative model parameters)

For most embedding benchmarks, use **concurrent** or **throughput** profiles.

## Advanced Usage

### Variable Input Lengths

Test performance across different input lengths:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --rate 10 \
    --max-requests 200 \
    --data "prompt_tokens=256,prompt_tokens_min=128,prompt_tokens_max=500,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

This creates requests with uniformly distributed lengths between 128 and 500 tokens.

### Using Real Data

Benchmark with actual text data from a file or Hugging Face dataset:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --rate 10 \
    --max-requests 100 \
    --data "path/to/your/data.jsonl" \
    --data-args '{"prompt_column": "text"}' \
    --processor "BAAI/bge-small-en-v1.5"
```

Or from Hugging Face:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --rate 10 \
    --max-requests 100 \
    --data "sentence-transformers/stsb" \
    --data-args '{"prompt_column": "sentence1", "split": "test"}' \
    --processor "BAAI/bge-small-en-v1.5"
```

### Load Testing Scenarios

#### Testing Concurrent Request Handling (Recommended)

The concurrent profile is the most relevant for embeddings, as it simulates how production systems typically use embedding models (parallel batch processing):

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 32 \
    --max-requests 500 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

The `--rate` parameter specifies the number of concurrent streams (e.g., 32 parallel requests).

#### Finding Maximum Throughput

Use the throughput profile to find the maximum sustainable request rate for capacity planning:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile throughput \
    --max-requests 500 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

## Metrics and Analysis

When benchmarking embeddings, GuideLLM tracks:

- **Request Latency**: Time from request start to completion
- **Time to First Token (TTFT)**: For embeddings, this is effectively the processing time
- **Throughput**: Requests processed per second
- **Token Throughput**: Input tokens processed per second
- **Success Rate**: Percentage of successful requests
- **Error Rate**: Percentage of failed requests

### Example Output

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --rate 10 \
    --max-requests 100 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5" \
    --output-path embeddings_report.json
```

The JSON report will include:

- Per-request timing and token counts
- Aggregate statistics (mean, median, percentiles)
- Request success/failure breakdown
- Overall benchmark metadata

## Best Practices

1. **Match the Processor**: Use the same tokenizer as your embedding model for accurate token counting

2. **Account for Model Context Length**:

   - **Check your model's limit**: Query the models endpoint to find `max_model_len`:

     ```bash
     curl -s http://localhost:8000/v1/models | python3 -m json.tool | grep "max_model_len"
     ```

     This will show something like: `"max_model_len": 512`

   - **Synthetic data overhead**: The generator adds 2-5 tokens per request to ensure uniqueness

   - **Leave headroom**: Subtract ~10 tokens from `max_model_len` for safety

   - **Examples**:

     - 512-token model → use `prompt_tokens=500` or `prompt_tokens_max=500`
     - 8192-token model → use up to `prompt_tokens=8180`

   - **Error symptom**: "maximum context length exceeded" errors mean your tokens + prefix > model limit

3. **Start with Low Rates**: Begin with conservative request rates and gradually increase

4. **Use Realistic Data**: Test with data similar to your production workload

5. **Test Multiple Scenarios**: Vary input lengths, batch sizes, and request patterns

6. **Monitor System Resources**: Watch CPU, memory, and GPU utilization during benchmarks

7. **Run Multiple Iterations**: Execute benchmarks several times to account for variance

## Examples

### Short Context Embeddings (128-512 tokens)

Typical for BERT-style models with concurrent processing:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 32 \
    --max-requests 500 \
    --data "prompt_tokens=256,prompt_tokens_min=128,prompt_tokens_max=500,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

This tests with 32 concurrent streams, which matches common production patterns. Using `prompt_tokens_max=500` instead of 512 leaves headroom for the synthetic data generator's unique request prefix.

### Long Context Embeddings (8k-32k tokens)

For newer long-context embedding models (lower concurrency due to larger context):

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 8 \
    --max-requests 100 \
    --data "prompt_tokens=16384,prompt_tokens_min=8192,prompt_tokens_max=32768,output_tokens=1" \
    --processor "jinaai/jina-embeddings-v3"
```

### Production Simulation

Simulate realistic production workload with variable input lengths:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 16 \
    --max-requests 1000 \
    --data "prompt_tokens=256,prompt_tokens_stdev=100,output_tokens=1,samples=1000" \
    --data-sampler random \
    --processor "BAAI/bge-base-en-v1.5" \
    --output-path production_simulation.json
```

This runs a comprehensive benchmark with 1000 requests and variable-length inputs (using standard deviation), closely mimicking real-world usage patterns.
