---
weight: -3
---

# Benchmark a governed OpenAI-compatible endpoint

GuideLLM can benchmark any OpenAI-compatible target, including a governed
endpoint such as Tuning Engines. This is useful when you want to measure not
just raw model latency, but the full production path your application will use:
routing, approvals, policy checks, usage attribution, and trace correlation.

## When this example is useful

Use this setup when you want to answer questions like:

- What is the latency overhead of the governed path compared with a direct model
  endpoint?
- Does the endpoint still meet the team's TTFT, ITL, and throughput goals under
  realistic traffic?
- How does the endpoint behave when you benchmark the same model route with
  different traffic profiles or tool-calling payloads?

## Example setup

Point GuideLLM at the public OpenAI-compatible Tuning Engines inference URL and
authenticate with a tenant inference key:

```bash
export TE_INFERENCE_KEY=sk-te-your-inference-key

guidellm run \
  --backend kind=openai_http,target=https://api.tuningengines.com/v1,api_key=$TE_INFERENCE_KEY \
  --profile kind=sweep,sweep_size=8 \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --output kind=html,path=guidellm-tuning-engines.html \
  --output kind=json,path=guidellm-tuning-engines.json
```

This benchmark keeps GuideLLM in its normal OpenAI HTTP mode. The governed
endpoint stays responsible for model routing and enforcement, while GuideLLM
measures user-visible performance and exports the same latency and throughput
artifacts you would produce for any other OpenAI-compatible deployment.

## Suggested comparison workflow

For a realistic deployment evaluation, run the same GuideLLM profile against:

1. a direct provider or inference server endpoint
2. the governed endpoint
3. any alternate governed route or model deployment you are considering

This lets you compare the operational tradeoff directly:

- latency distributions
- throughput ceilings
- error rates under load
- output consistency across benchmark profiles

## Correlating GuideLLM runs with runtime traces

If the governed endpoint supports request and run correlation metadata, keep the
same identifiers across benchmark runs so benchmark artifacts can be joined with
runtime traces, policy decisions, and usage records later.

For Tuning Engines specifically, this means pairing the benchmark with:

- OpenAI-compatible inference calls through `https://api.tuningengines.com/v1`
- runtime trace ingest at `POST /api/v1/traces` when you want workflow-level or
  tool-level context alongside benchmark results
- optional state or memory references at `POST /api/v1/runtime_state_references`
  when the benchmark belongs to a larger orchestrated workflow

## Why this matters

Many teams benchmark the raw model server but deploy a more complex production
path. Benchmarking the governed endpoint directly gives a better picture of the
latency and throughput users will actually experience in production.
