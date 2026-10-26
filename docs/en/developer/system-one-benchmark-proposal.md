# Proposal: Native System One Decision Benchmarks

**Status: proposal, not an implemented backend.** This document defines a reviewable contribution to GuideLLM. It does not add a CLI option or claim Basal performance results.

## Problem and value

[Basal](https://github.com/rkinas/basal) answers typed questions about a shared state in one or more forward passes. Its [System One interface](https://github.com/rkinas/basal/blob/main/docs/API.md) accepts `state` and named `questions`, and returns probability-bearing `answers`. It is not an OpenAI chat completion. Its native usage reports `output_tokens: 0`.

GuideLLM can already schedule offered-rate, Poisson, concurrency, and replay workloads, measure dispatch delay and scheduled latency, and evaluate latency objectives. Basal's [load tester](https://github.com/rkinas/basal/blob/main/basal/loadtest.py) already provides sequential latency, fixed-concurrency throughput, and probability comparisons alone versus under load. The contribution is controlled service-load characterization, not a replacement for Basal's calibration, identity, or engine benchmarks.

Changing `openai_http.api_routes` is insufficient: it changes the URL, not the request formatter, response parser, or model-discovery format.

## Proposed scope

1. A registered `systemone_http` backend that communicates over HTTP without importing Basal or loading model weights.
2. A native request finalizer that preserves one entire request object per dataset row.
3. Explicit completed-question accounting alongside existing request latency and throughput.
4. CPU-only protocol and report verification, followed by optional real Basal serving measurements.

Do not add an arbitrary-HTTP expression language, server retries, calibration evaluation, live tool execution, or a new scheduler. Native media can remain inside the request object; the first benchmark examples should use text.

## Request contract

The dataset has a configurable payload column, defaulting to `request`. Each value is one complete native JSON object, for example:

```json
{
  "request": {
    "state": {"ticket": "I cannot sign in after resetting my password."},
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"support": "Account support", "billing": "Billing"}
      },
      "urgent": {"type": "noul", "instructions": "Is immediate action required?"}
    }
  }
}
```

The proposed `systemone` finalizer reads that column directly from deserialized rows. It emits one standard request node per row, with the payload in `GenerationRequest.columns["native_request_column"]`. It does not pass the payload through the generative column mapper, chat-template formatting, or media encoding. The existing `GenerationRequest.columns` accepts arbitrary names, but `GenerativeDatasetColumnType` does not; adding a name to the generative mapper is not sufficient or necessary for this separate finalizer.

Reuse existing scheduling metadata and `GenerativeConversationGraph` construction. The first milestone supports independent requests, not implicit chat-history concatenation. A dataset that models growing state must carry the full intended state in every request. Preserve multiple named questions in a single HTTP request so their state can be shared by the server.

## Backend contract

Register the implementation through `Backend` and its configuration through `BackendArgs`:

- `target`: server base URL.
- `request_path`: `/v1/systemone` by default; `/v1/basal` is an existing Basal alias.
- `model`: optional explicit report identity; otherwise discover from the native `models` array and `name` field of `/v1/models`.
- HTTP connection/read timeouts and optional bearer authentication follow existing backend conventions.
- No streaming flag: this interface returns one complete decision response.

Keep native discovery, payload formatting, response interpretation, and validation in the backend implementation. Do not put `if backend.kind == "systemone_http"` in the scheduler, generic HTTP backend, or benchmark entrypoint.

Set request start/end timestamps around actual HTTP I/O, as existing non-streaming backends do. Do not record token iterations for the JSON response. Preserve request identity and let the existing scheduler classify HTTP failures; do not retry them implicitly. Basal documents 422 for invalid requests, 503 for resource exhaustion, and 500 for other server failures.

A successful response must contain the requested named answers. Keep the original structured answer data: probabilities are not generated text, and a selected `tool` is not an OpenAI function call to execute.

## Response and metrics contract

The current seams are `GenerationResponse`, `UsageMetrics`, retained `GenerativeRequestStats`, and the benchmark accumulator/report. `response_metrics` retains server data but does not automatically aggregate arbitrary counters.

For the first implementation:

- Retain the native response as serialized JSON in the existing sampled `output` field, explicitly documented as structured output, not generated prose. Set text word/character counts to unavailable rather than counting JSON syntax as model output.
- Retain native usage in namespaced `response_metrics`, including server `latency_ms`. Client end-to-end and scheduled latency remain distinct measurements.
- Set `output_metrics.text_tokens = 0` explicitly. Existing zero-versus-unavailable token semantics preserve this value. Do not tokenize answer JSON or substitute one token per answer.
- Leave TTFT, TTFOT, ITL, and TPOT unavailable. Verify both per-request values and compiled distributions; coordinate with the existing [missing-token-timing work](https://github.com/vllm-project/guidellm/pull/1406).
- Add an optional completed-question count to the response/statistics contract and corresponding aggregate count/rate fields. Use one completed named answer as the unit; a multi-label set still answers one named question. Require agreement between returned answer names and the requested questions before counting the request as successful.
- Derive question throughput from actual successful question counts over the same measurement interval as request throughput. Do not multiply request rate by an assumed fixed questions-per-request value. Failed and incomplete requests contribute no completed questions.
- Keep `branches`, `second_order_branches`, and truncation information in raw server metrics until their semantics are stable. Do not label branches GPU forwards or input tokens unique physical prefill work.

Prefer explicit question count/rate fields for this concrete interface over introducing an unrestricted aggregate-metric plugin system. Confirm these public report field names with maintainers before implementation. Keep all counts available even when full request/answer payload sampling is disabled.

For latency SLOs, request goodput counts conforming requests. A future question-goodput metric would weight those requests by their completed questions and must be named separately. Latency conformity does not establish answer quality or calibration.

## Implementation locations

| Existing seam                                                                 | Proposed change                                                                                            |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `src/guidellm/backends/backend.py` and `schemas/backends/`                    | Register native backend and configuration; reuse lifecycle and worker interfaces.                          |
| `src/guidellm/data/finalizers/` and `schemas/data/`                           | Register native payload finalizer with a configurable dataset column.                                      |
| `src/guidellm/schemas/base/response.py`, `request.py`, and `request_stats.py` | Preserve structured output/usage and completed-question counts without fake token measurements.            |
| `src/guidellm/benchmark/schemas/accumulator.py` and `metrics.py`              | Aggregate successful question counts and rates over existing measurement windows.                          |
| `src/guidellm/benchmark/outputs/`                                             | Expose units in JSON and a human-readable report; respect configured payload retention.                    |
| `docs/en/guides/`                                                             | Document supported native requests, timing semantics, and a real CPU Basal smoke command once implemented. |

The existing registry-backed `kind` convention should select each implementation. No compatibility aliases or changes to OpenAI request formatting are needed.

## Acceptance scenarios

| Scenario                                                  | Observable result                                                                                 |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| One and five questions per request, then a mixed workload | Exact completed-question totals and question throughput; requests remain separately counted.      |
| Choice, yes/no, score, and multi-label questions          | Complete typed answers preserved; no fabricated function execution or generated tokens.           |
| HTTP 422, 503, and 500; missing answer names              | Failures are visible and do not inflate successful question counts.                               |
| Synthetic native server with known delay                  | Client request latency measures the exchange; server latency remains a separate raw metric.       |
| Zero generated tokens                                     | Token throughput is zero; token-generation timing is unavailable, not a zero-latency observation. |
| Sampling disabled or bounded                              | All lightweight counts remain correct; retained answers follow the configured sample policy.      |
| Native model discovery and explicit identity              | Correct model identity with no assumption that native `/v1/models` has OpenAI's `data/id` shape.  |
| Lossless and media-omitting outputs together              | Each output follows its own retention policy; the transmitted native request remains unchanged.   |

Run protocol scenarios against a CPU HTTP peer and exercise a real Basal CPU/eager server before claiming interoperability. Protocol fixtures alone are not real model-serving evidence.

## Basal performance experiment after interoperability

Use matched workloads across engine configurations: unique states; repeated exact states with different questions; growing states; one versus five questions; and mixed short/long states. Measure requests/s, questions/s, client latency tails, scheduled latency, errors, and request SLO attainment.

Use fresh seeds between phases to avoid accidentally measuring caches populated by warm-up. Do not infer cache-hit rates from input identity or latency. Preserve probability comparisons and calibration checks in Basal's existing tools; a faster service is not automatically an equivalent decision model.

A GPU is unnecessary for backend/accounting development. CUDA batching, cache, SOAM, or probability-stability claims require the actual target GPU and model. Hopper-specific claims require Hopper hardware; Apple/MLX claims require the relevant Apple hardware.

## Coordination before coding

- Ask GuideLLM maintainers to approve the native backend, structured-output representation, and explicit question accounting together; a backend that silently misreports work is not a useful partial cutover.
- [Basal #45](https://github.com/rkinas/basal/pull/45) proposes `/v1/decisions`. It is not the current native contract and is not a prerequisite for this proposal.
- [Basal #48](https://github.com/rkinas/basal/pull/48) owns Prometheus instrumentation. Reuse it through existing GuideLLM collectors if it lands rather than creating competing server telemetry.
- [GuideLLM #566](https://github.com/vllm-project/guidellm/issues/566) concerns live public response delivery. It can support downstream decision analysis but is independent of native request replay.
