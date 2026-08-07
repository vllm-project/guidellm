# Finding Your Server's Optimal Concurrency Limit

Determine the maximum number of concurrent users your LLM server can handle before latency degrades, and identify the optimal operating point for your deployment.

## Prerequisites

- A running OpenAI-compatible server ([setup guide](../getting-started/server.md))
- GuideLLM installed ([install guide](../getting-started/install.md))
- How to run a benchmark ( [benchmark guide](../getting-started/benchmark.md))

## Step 1: Run a Sweep

The [`sweep`](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) profile automatically tests increasing load levels — starting from a single sequential request, ramping up to maximum throughput, and interpolating several rates in between. It stops automatically when it detects the server is over-saturated.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 # change this to your desired
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=sweep_results.json
```

This runs up to 10 strategies (synchronous baseline, throughput ceiling, and 8 interpolated rates) for 120 seconds each. The over-saturation detector will skip remaining strategies once the server can no longer keep up.

Adjust `prompt_tokens` and `output_tokens` to match your actual workload. A chat application might use `prompt_tokens=256,output_tokens=512`, while a summarization pipeline might use `prompt_tokens=2048,output_tokens=128`.

## Step 2: Read the Results

Open the console output or JSON report. For each strategy, look at these metrics:

| Metric                            | What it tells you                                          |
| --------------------------------- | ---------------------------------------------------------- |
| `output_tokens_per_second` (mean) | Server throughput — how much work is getting done          |
| `time_to_first_token_ms` (p50)    | Typical user-perceived wait before they receive a response |
| `time_to_first_token_ms` (p99)    | Worst-case wait — important for tail latency SLOs          |
| `inter_token_latency_ms` (p50)    | How smooth the streaming experience feels                  |
| `request_latency` (p50)           | Total end-to-end time per request                          |

## Step 3: Identify the Three Zones

As load increases across the sweep strategies, your server will pass through three zones:

**Underutilized**: Throughput scales linearly with load. Latency stays flat. Adding more concurrent users makes the server do proportionally more work with no penalty. You are leaving capacity on the table.

**Sweet spot**: Throughput is still climbing but gains are shrinking. TTFT p50 is creeping up but still acceptable. The server is efficiently batching requests. **This is where you want to operate.**

**Over-saturated**: Throughput plateaus or drops. TTFT p99 spikes sharply. The request queue is growing faster than the server can handle. GuideLLM's over-saturation detector will stop the sweep here.

Look for the transition between the first and second zone — the point where TTFT p99 starts rising noticeably while throughput is still increasing. The strategy just before that transition is your optimal operating point.

## Step 4: Validate with a Focused Run

Once you have identified a target concurrency (for example, 32 streams from the sweep results), run a longer benchmark at that specific level to confirm the metrics are stable:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --profile kind=concurrent,streams=32 \
  --constraint kind=max_duration,seconds=300 \
  --seed kind=static,value=42 \
  --output kind=json,path=validation_run.json
```

A 5-minute run at fixed concurrency gives more reliable metrics than the shorter sweep strategies. Compare the validation results to what the sweep predicted — they should be consistent. If TTFT p99 is higher than expected, drop down one concurrency level and re-validate.

## Step 5: Test Around the Boundary

To get a precise answer, run a few concurrency levels around your candidate:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --profile '{"kind":"concurrent","streams":[24,28,32,36,40]}' \
  --constraint kind=max_duration,seconds=180 \
  --seed kind=static,value=42 \
  --output kind=json,path=boundary_test.json
```

This runs 5 sub-benchmarks at tightly spaced concurrency levels. Compare throughput and TTFT p99 across them to pinpoint exactly where the performance degrades.

## Making the Decision

How you pick the final number depends on your deployment goal:

**Latency-sensitive (chat, real-time)**: Choose the highest concurrency where TTFT p99 stays under your SLO. If your SLO is 500ms TTFT, and TTFT p99 at 32 streams is 480ms but at 36 streams is 720ms, run at 32.

**Throughput-optimized (batch, offline)**: Choose the concurrency where `output_tokens_per_second` plateaus. Latency matters less here, so you can push further into the sweet spot.

**Production with headroom**: Take your chosen concurrency and multiply by 0.8. If your sweet spot is 40 streams, operate at 32. This gives room for traffic spikes without hitting over-saturation.

## Example: Interpreting a Sweep

Here is what a real sweep looks like.

Results collected using **meta-llama/Llama-3.1-8B-Instruct** on a single **NVIDIA A100 80GB** GPU, served by vLLM with chunked prefill enabled. Workload: 1000 input tokens, 1000 output tokens.

| Strategy    | Concurrency (mean) | Req/s (mean) | Output Tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | Zone           |
| ----------- | ------------------ | ------------ | --------------- | ------------- | ------------- | ------------ | ------------ | -------------- |
| synchronous | 1.0                | 0.1          | 90.2            | 79.5          | 115.2         | 11.0         | 11.1         | Under-utilized |
| constant    | 3.8                | 0.3          | 327.7           | 95.0          | 100.1         | 11.5         | 11.6         | Under-utilized |
| constant    | 7.1                | 0.5          | 569.7           | 97.7          | 102.8         | 12.3         | 12.4         | Under-utilized |
| constant    | 10.8               | 0.8          | 809.4           | 100.4         | 105.8         | 13.3         | 13.4         | Under-utilized |
| constant    | 14.7               | 1.0          | 1047.9          | 100.4         | 108.4         | 14.0         | 14.1         | Sweet spot     |
| constant    | 21.0               | 1.2          | 1274.9          | 107.8         | 115.9         | 16.6         | 16.7         | Sweet spot     |
| constant    | 27.8               | 1.4          | 1496.5          | 117.7         | 126.6         | 19.2         | 19.3         | Sweet spot     |
| constant    | 33.7               | 1.6          | 1728.6          | 123.5         | 133.9         | 19.8         | 19.9         | Sweet spot     |
| constant    | 46.0               | 1.7          | 1911.7          | 132.6         | 146.6         | 25.0         | 25.5         | Over-saturated |
| throughput  | 509.8              | 2.1          | 3217.6          | 9947.7        | 21961.3       | 84.1         | 106.8        | Over-saturated |

In this example, throughput scales linearly from concurrency 1 through ~11 (under-utilized). Between concurrency 15 and 34, throughput is still climbing but TTFT and ITL are creeping up (sweet spot). At concurrency 46, throughput gains flatten while latency continues rising — and the throughput strategy shows TTFT exploding to ~10 seconds, confirming over-saturation.

The optimal operating point here is around **concurrency 28-34** — throughput is near 1500-1700 tok/s with TTFT p50 still under 135ms and ITL under 20ms. For production with headroom, target concurrency 24-28.
