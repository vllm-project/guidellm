# vLLM Python Batch Backend

The **vLLM Python batch backend** (`vllm_python_batch`) runs batch inference in the same process as GuideLLM using vLLM's synchronous [LLM](https://docs.vllm.ai/) engine. Requests are queued and dispatched in configurable batches via `LLM.generate()`, removing per-request scheduling overhead. This is ideal for throughput benchmarking where latency per individual request is less important than aggregate throughput.

Like the `vllm_python_async` backend, no HTTP server is involved. You do **not** pass a `target`; you **must** pass `model` in the backend configuration.

For all engine options and supported models, see vLLM's [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) and the [vLLM documentation](https://docs.vllm.ai/).

## Installation

Installation is the same as for the [vLLM Python backend](vllm-python-backend.md#installation). The batch backend uses the same vLLM package.

## Basic example

Run a benchmark with the vLLM Python batch backend:

```bash
guidellm run \
  --backend kind=vllm_python_batch,model=Qwen/Qwen3-0.6B,batch_size=32 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=throughput,max_concurrency=20 \
  --constraint kind=max_requests,count=100
```

## Async vs batch backend

| Feature   | `vllm_python_async`                | `vllm_python_batch`             |
| --------- | ---------------------------- | -------------------------- |
| Engine    | `AsyncLLMEngine` (async)     | `LLM` (synchronous, batch) |
| Streaming | Supported                    | Not supported              |
| Batching  | Per-request async scheduling | Configurable micro-batches |
| Best for  | Latency profiling, streaming | Throughput benchmarking    |

## Backend options

- **`batch_size`** (default: `32`)\
  Maximum number of requests to accumulate before dispatching a single `LLM.generate()` call. When the batch fills to this size, it is dispatched immediately. Partial batches (fewer requests than `batch_size`) are flushed after `batch_timeout` seconds. The effective batch size is therefore `min(batch_size, requests arriving within the timeout window)`. Larger values amortize engine overhead but increase per-request latency.

- **`batch_timeout`** (default: `0.01`)\
  Seconds to wait for more requests before flushing a partial batch. Full batches bypass this delay entirely. Increase this value when higher concurrency allows more requests to accumulate per batch; decrease it (or leave at the default) for latency-sensitive workloads.

- **`model`** (required)\
  Hugging Face model identifier or filesystem path for vLLM to load.

- **`request_format`**\
  Controls how chat prompts are built. Same options as `vllm_python_async`: `plain`, `default-template`, or a Jinja2 template path/string.

- **`vllm_config`**\
  Engine options passed as a nested dict. Uses vLLM's `EngineArgs` parameter names (Python form, not CLI form). See the [vLLM Python backend docs](vllm-python-backend.md#request-format-and-backend-options) for details on `vllm_config`.

  Example with JSON:

  ```bash
  --backend '{"kind":"vllm_python_batch","model":"Qwen/Qwen3-0.6B","batch_size":64,"vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

> [!IMPORTANT]
>
> The `model` field in the backend configuration is required for `vllm_python_batch`. If `model` is also set inside `vllm_config`, the top-level `model` field takes precedence.

## Engine lifecycle

The vLLM `LLM` engine is never loaded during `process_startup()`. Engine creation is controlled by a worker-process check that distinguishes the main (preflight) process from scheduler workers:

- **Main preflight** (`resolve_backend`): `validate()` runs in the main process (`multiprocessing.parent_process()` is `None`) and performs a cheap readiness check only — no model weights are loaded.
- **Worker process**: when `multiprocessing.parent_process()` is set (true for both `fork` and `spawn` workers; see `GUIDELLM__MP_CONTEXT_TYPE`), `validate()` calls `_ensure_engine()` to **preload** the engine so the cold-start time is excluded from the timed benchmark phase.
- **Inference-time safety net**: as requests are generated from the dataset, `_ensure_engine()` is called as an idempotent fallback, so inference works correctly even if `validate()` was skipped.

## See also

- [vLLM Python Async Backend](vllm-python-backend.md) -- Async per-request backend.
- [Backends](backends.md) -- Overview of supported backends.
- [Run a benchmark](../getting-started/benchmark.md) -- General benchmark options.
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) -- CLI-oriented docs; use Python names in `vllm_config`.
- [vLLM documentation](https://docs.vllm.ai/)
