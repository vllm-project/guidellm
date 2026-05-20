# vLLM Offline Backend

The **vLLM Offline backend** (`vllm_offline`) provides synchronous batch
processing using vLLM's `LLM` class. It collects requests into micro-batches
and processes them together for maximum throughput, making it ideal for
offline benchmarking scenarios where batching efficiency is prioritized over
per-request latency.

## When to Use the Offline Backend

**Use `vllm_offline` when:**
- Running offline batch inference on large datasets
- Maximizing throughput is more important than individual request latency
- You have a known dataset size and want optimal batch processing
- Benchmarking pure model throughput without HTTP overhead
- Processing datasets for evaluation or ETL pipelines

**Use `vllm_python` (AsyncLLMEngine) when:**
- You need streaming token-by-token responses
- Simulating production-like continuous request arrival
- Measuring realistic latency characteristics
- Need async request handling

**Use OpenAI HTTP backend when:**
- Testing against a production vLLM server
- Measuring end-to-end latency including network overhead
- Benchmarking a deployed service

## Installation

The offline backend requires vLLM to be installed. See the [vLLM Python
Backend installation guide](vllm-python-backend.md#installation) for
recommended installation methods.

## Basic Usage

```bash
guidellm benchmark run \
  --backend vllm_offline \
  --model "Qwen/Qwen3-0.6B" \
  --backend-kwargs '{"batch_size": 64}' \
  --data "prompt_tokens=256,output_tokens=128" \
  --max-requests 1000
```

## Backend Options

Configure the offline backend via `--backend-kwargs` with JSON:

```bash
--backend-kwargs '{
  "model": "meta-llama/Llama-2-7b-hf",
  "batch_size": 64,
  "vllm_config": {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9
  }
}'
```

### Key Parameters

- **`model`** (required): Model identifier or path
- **`batch_size`**: Number of requests to collect before processing (default: 32)
  - Larger batches = higher throughput but more latency
  - Smaller batches = lower latency but less throughput
  - Recommended: 32-128 for most use cases
- **`vllm_config`**: Dictionary of vLLM EngineArgs parameters
  - `tensor_parallel_size`: Number of GPUs for tensor parallelism
  - `gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0)
  - `max_model_len`: Maximum sequence length
  - See [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
    for all options (use Python parameter names)
- **`request_format`**: How to format prompts
  - `"default-template"` (default): Use tokenizer's chat template
  - `"plain"`: No chat template, plain text concatenation
  - Path or string: Custom Jinja2 chat template
- **`image_placeholder`**: Placeholder for images (default: `"<image>"`)
- **`audio_placeholder`**: Placeholder for audio (default: `"<|audio|>"`)

## How Micro-Batching Works

The offline backend uses a **micro-batching** approach:

1. **Buffering**: As requests arrive via `resolve()`, they're added to a buffer
2. **Batch Detection**: When buffer reaches `batch_size`, trigger processing
3. **Batch Processing**: Process entire batch with one `LLM.generate()` call
4. **Result Distribution**: Return cached results to waiting requests
5. **Flush on Shutdown**: Remaining requests processed when backend shuts down

This gives you 10-100x fewer model forward passes compared to per-request
processing while working within GuideLLM's scheduler architecture.

## Examples

### Basic Throughput Benchmark

```bash
guidellm benchmark run \
  --backend vllm_offline \
  --model "Qwen/Qwen3-0.6B" \
  --data "prompt_tokens=512,output_tokens=256" \
  --profile throughput \
  --max-seconds 60
```

### Large Batch Processing

```bash
guidellm benchmark run \
  --backend vllm_offline \
  --backend-kwargs '{"batch_size": 128}' \
  --model "meta-llama/Llama-2-7b-hf" \
  --data path/to/dataset.csv \
  --max-requests -1  # Process entire dataset
```

### Multi-GPU Configuration

```bash
guidellm benchmark run \
  --backend vllm_offline \
  --backend-kwargs '{
    "model": "meta-llama/Llama-2-70b-hf",
    "batch_size": 64,
    "vllm_config": {
      "tensor_parallel_size": 4,
      "gpu_memory_utilization": 0.95
    }
  }' \
  --data "prompt_tokens=1024,output_tokens=512"
```

### HuggingFace Dataset

```bash
guidellm benchmark run \
  --backend vllm_offline \
  --model "meta-llama/Llama-2-7b-hf" \
  --backend-kwargs '{"batch_size": 32}' \
  --data "hf:cnn_dailymail" \
  --data-args '{"name": "3.0.0"}' \
  --data-column-mapper '{"column_mappings": {"text_column": "article"}}'
```

## Performance Tuning

### Choosing Batch Size

| Batch Size | Throughput | Latency | Memory | When to Use |
|------------|------------|---------|--------|-------------|
| 8-16 | Low | Low | Low | Small models, limited memory |
| 32-64 | Good | Medium | Medium | General use, balanced |
| 128-256 | High | High | High | Large GPUs, max throughput |

**Rule of thumb**: Start with 32, increase until GPU utilization >90% or OOM.

### Memory Optimization

```bash
# Reduce memory usage
--backend-kwargs '{
  "batch_size": 16,
  "vllm_config": {
    "gpu_memory_utilization": 0.8,
    "max_model_len": 2048
  }
}'
```

### Maximizing Throughput

```bash
# Maximize throughput
--backend-kwargs '{
  "batch_size": 128,
  "vllm_config": {
    "gpu_memory_utilization": 0.95,
    "enable_prefix_caching": true
  }
}'
```

## Comparison: Offline vs Python vs HTTP

| Feature | `vllm_offline` | `vllm_python` | OpenAI HTTP |
|---------|----------------|---------------|-------------|
| **Batching** | Micro-batching | Continuous | Continuous |
| **Throughput** | Highest | High | Good |
| **Latency** | Higher (batched) | Lower | Lowest† |
| **Streaming** | No | Yes | Yes |
| **Overhead** | None | None | HTTP/network |
| **Processes** | 1 | 1 | Multiple |
| **Use Case** | Offline eval | Research | Production |

*† Subject to network conditions*

## Troubleshooting

### "Backend not started up for process"

The backend wasn't initialized. Ensure your benchmark calls the backend
lifecycle correctly (this should happen automatically).

### Out of Memory (OOM)

Reduce `batch_size` or `gpu_memory_utilization`:

```bash
--backend-kwargs '{"batch_size": 16, "vllm_config": {"gpu_memory_utilization": 0.7}}'
```

### Batch Processing Too Slow

Increase `batch_size` for better GPU utilization:

```bash
--backend-kwargs '{"batch_size": 64}'
```

### Wrong Prompt Format

Specify `request_format` explicitly:

```bash
--backend-kwargs '{"request_format": "plain"}'
```

## Limitations

1. **No Streaming**: Results returned after entire batch completes
2. **Single Process**: Limited to 1 worker process for batch coordination
3. **Fixed Batch Window**: Batches based on count, not time
4. **Multi-turn Not Supported**: Conversation history not yet implemented

## See Also

- [vLLM Python Backend](vllm-python-backend.md) - AsyncLLMEngine-based backend
- [Backends Guide](backends.md) - Overview of all backends
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) - Full configuration options
- [vLLM LLM Class](https://docs.vllm.ai/en/stable/offline_inference/llm.html) - Underlying API documentation
