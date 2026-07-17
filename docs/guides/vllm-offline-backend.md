# vLLM Offline Backend

The **vLLM offline backend** (`vllm_offline`) runs batch inference in the same process as GuideLLM using vLLM's synchronous [LLM](https://docs.vllm.ai/) engine. Requests are queued and dispatched in configurable batches via `LLM.generate()`, removing per-request scheduling overhead. This is ideal for throughput benchmarking where latency per individual request is less important than aggregate throughput.

Like the `vllm_python` backend, no HTTP server is involved. You do **not** pass a `target`; you **must** pass `model` in the backend configuration.

For all engine options and supported models, see vLLM's [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) and the [vLLM documentation](https://docs.vllm.ai/).

## Installation

Installation is the same as for the [vLLM Python backend](vllm-python-backend.md#installation). The offline backend uses the same vLLM package.

## Basic example

Run a benchmark with the vLLM offline backend:

```bash
guidellm run \
  --backend kind=vllm_offline,model=Qwen/Qwen3-0.6B,batch_size=32 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=constant,rate=3 \
  --constraint kind=max_duration,seconds=20
```

## Offline vs Python backend

| Feature   | `vllm_python`                | `vllm_offline`             |
| --------- | ---------------------------- | -------------------------- |
| Engine    | `AsyncLLMEngine` (async)     | `LLM` (synchronous, batch) |
| Streaming | Supported                    | Not supported              |
| Batching  | Per-request async scheduling | Configurable micro-batches |
| Best for  | Latency profiling, streaming | Throughput benchmarking    |

## Backend options

- **`batch_size`** (default: `32`)\
  Maximum number of requests to accumulate before dispatching a single `LLM.generate()` call. Larger batches amortize engine overhead but increase per-request latency.

- **`model`** (required)\
  Hugging Face model identifier or filesystem path for vLLM to load.

- **`request_format`**\
  Controls how chat prompts are built. Same options as `vllm_python`: `plain`, `default-template`, or a Jinja2 template path/string.

- **`vllm_config`**\
  Engine options passed as a nested dict. Uses vLLM's `EngineArgs` parameter names (Python form, not CLI form). See the [vLLM Python backend docs](vllm-python-backend.md#request-format-and-backend-options) for details on `vllm_config`.

  Example with JSON:

  ```bash
  --backend '{"kind":"vllm_offline","model":"Qwen/Qwen3-0.6B","batch_size":64,"vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

> [!IMPORTANT]
>
> The `model` field in the backend configuration is required for `vllm_offline`. If `model` is also set inside `vllm_config`, the top-level `model` field takes precedence.

## See also

- [vLLM Python Backend](vllm-python-backend.md) -- Async per-request backend.
- [Backends](backends.md) -- Overview of supported backends.
- [Run a benchmark](../getting-started/benchmark.md) -- General benchmark options.
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) -- CLI-oriented docs; use Python names in `vllm_config`.
- [vLLM documentation](https://docs.vllm.ai/)
