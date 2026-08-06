# Benchmarking Different Workload Shapes

Not all LLM workloads are the same. A chatbot, a summarization pipeline, and a code generator stress your server in completely different ways. This guide shows how to configure GuideLLM for each specific workload type and explains why metrics behave differently.

## Prerequisites

- A running OpenAI-compatible server ([setup guide](../getting-started/server.md))
- GuideLLM installed ([install guide](../getting-started/install.md))

## Why Workload Shape Matters

LLM requests have two phases:

- **Prefill**: the server processes all input tokens at once. This determines time-to-first-token (TTFT).
- **Decode**: the server generates output tokens one at a time. This determines inter-token latency (ITL) and overall throughput.

The ratio of input to output tokens shifts which phase dominates. A workload with long prompts and short responses is prefill-bound. A workload with short prompts and long responses is decode-bound. Each has different bottlenecks and different metrics to watch.

## The Four Common Shapes

### 1. Chat / Conversational

A user sends a short message and expects a medium-length response **(decode-bound)**. Typical of chatbots, customer support, and Q&A interfaces.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=512 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=chat_workload.json
```

**What to watch:** Both TTFT and ITL matter. Users are waiting for the response to start streaming (TTFT) and then reading it as it arrives (ITL). If either is too slow, the experience feels laggy.

### 2. Summarization / RAG

A retrieval-augmented generation pipeline stuffs a large context into the prompt and expects a concise answer. Long input, short output. **(prefill-bound)**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=2048,output_tokens=128 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=summarization_workload.json
```

**What to watch:** TTFT dominates. The server spends most of its time processing the long prompt before generating a short answer. ITL matters less because there are so few output tokens. Throughput (tokens/sec) will look lower than chat because prefill is the bottleneck.

### 3. Code Generation

A developer gives a short instruction and expects a long block of generated code. Short input, long output. **(prefill-bound)**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=1024 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=codegen_workload.json
```

**What to watch:** ITL dominates. TTFT will be fast (short prompt to prefill), but the server spends most of its time decoding a long output. High ITL means the code takes forever to stream in. Throughput (tokens/sec) is the key metric here.

### 4. Balanced / General Purpose

Equal input and output lengths **(balanced)**. A good starting point when you are not sure what your workload looks like, or when you want a general performance baseline.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=balanced_workload.json
```

**What to watch:** Both phases contribute roughly equally. This gives the most balanced view of your server's capabilities but may not reflect your actual production workload.

## Comparing Results Across Shapes

Run all four shapes on the same server and compare the results side by side.

| Workload | Tokens (in/out) | Throughput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) | Bottleneck |
|----------|-----------------|-------------------|---------------|-------------|------------|
| Chat | 256 / 512 | 1850 | 35 | 12 | Mixed |
| Summarization | 2048 / 128 | 920 | 180 | 9 | Prefill (TTFT) |
| Code gen | 128 / 1024 | 2100 | 18 | 14 | Decode (ITL) |
| Balanced | 1000 / 1000 | 1350 | 95 | 13 | Mixed |

Key observations:

- **Summarization has the highest TTFT** because the server processes 2048 input tokens before generating the first output token **(prefill-bound)**. But ITL is the lowest because there are very few tokens to decode.
- **Code gen has the highest throughput** because the short prompt prefills quickly, and the server spends most of its time in the efficient decode phase. But requests take the longest end-to-end because of 1024 output tokens.
- **Chat is the most balanced** — neither phase dominates, so both TTFT and ITL need to be within SLO.
- **Throughput numbers are not directly comparable** across shapes because total tokens per request differ. Use `output_tokens_per_second` for an apples-to-apples decode throughput comparison.

## Choosing Your Benchmark Configuration

If you know your workload:

| Your application | Recommended config | Primary metric |
|-----------------|-------------------|----------------|
| Chatbot, Q&A | `prompt_tokens=256,output_tokens=512` | TTFT p99 and ITL p50 |
| RAG, summarization, search | `prompt_tokens=2048,output_tokens=128` | TTFT p99 |
| Code gen, writing, translation | `prompt_tokens=128,output_tokens=1024` | ITL p50 and throughput |
| Unknown or mixed | `prompt_tokens=1000,output_tokens=1000` | All metrics |

If you have real production data, use it instead of synthetic. GuideLLM supports custom datasets via JSONL files or HuggingFace datasets — see the [Datasets guide](../guides/datasets.md) for details.

## Tuning Your Server for a Workload Shape

The benchmark results can guide server configuration:

**Prefill-bound (high TTFT):**
- Enable chunked prefill (`--enable-chunked-prefill`) to overlap prefill with decode
- Increase tensor parallelism to spread the prefill computation across GPUs
- Consider a shorter `--max-model-len` if your prompts don't need the full context window

**Decode-bound (high ITL):**
- Check if you are memory-bandwidth limited — larger batch sizes help divide memory reads
- Try quantized models (FP8, W4A16) to reduce memory bandwidth pressure
- Consider speculative decoding for latency-sensitive workloads

## Next Steps

- [Finding Your Server's Optimal Concurrency Limit](optimal_concurrency.md) to optimize the load level for your chosen workload shape