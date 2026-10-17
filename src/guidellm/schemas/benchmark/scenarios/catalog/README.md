# Inference-perf workload catalog

These scenarios reproduce the runnable YAML configurations in the [inference-perf workload catalog](https://github.com/kubernetes-sigs/inference-perf/tree/76e2959b6a8ed9ded7af9c7398d9025a04da195f/workload-catalog) at revision `76e2959b6a8ed9ded7af9c7398d9025a04da195f`, using GuideLLM's existing scenario schema. Where upstream `config.json` and `inference-perf.yaml` differ, the YAML takes precedence. Each scenario records its source and original model in metadata; neither the backend target nor model is configured.

Supply the endpoint and served model when running a scenario:

```bash
uv run guidellm run \
  --scenario catalog/interactive-chat \
  --backend kind=openai_http,target=http://localhost:8000,model=YOUR_SERVED_MODEL
```

## Workloads and stages

Each row describes five successive benchmark runs. All scenarios use seed 42 and streaming responses. Synthetic scenarios use `/v1/completions`; trace replay uses `/v1/chat/completions`.

| Scenario                          | Concurrent streams         | Minimum processed requests | Turns per conversation |
| --------------------------------- | -------------------------- | -------------------------- | ---------------------- |
| `interactive-chat`                | 10, 20, 30, 40, 50         | 300 each                   | 4                      |
| `code-generation`                 | 5, 10, 20, 30, 40          | 100, 200, 400, 600, 800    | 540                    |
| `deep-research`                   | 1, 2, 3, 4, 5              | 50 each                    | 15                     |
| `reasoning`                       | 2, 4, 6, 8, 10             | 20 each                    | 1                      |
| `batch-summarization-rag`         | 6, 12, 18, 24, 30          | 450 each                   | 1                      |
| `batch-synthetic-data-generation` | 8, 16, 24, 32, 40          | 160 each                   | 1                      |
| `agentic-trace-replay`            | Recorded timing; see below | Dataset exhaustion         | Recorded               |

## Synthetic workload approximations

- **Distributions:** GuideLLM uses bounded normal sampling for upstream normal and lognormal distributions, retaining the configured mean, standard deviation, and bounds. Lognormal tails are therefore not reproduced, and clipping can change realized moments substantially. Random samples also differ between harnesses. Batch synthetic generation uses uniform integer output lengths from 500 to 8000 by omitting `output_tokens_stdev`. Like upstream's uniform sampler, this uses the bounds rather than its nominal mean of 4000 and standard deviation of 2500.
- **First-turn context:** One shared prefix represents the fixed system prompt. For multi-turn workloads, the dynamic system suffix and first user input are combined into the first prompt: means and bounds are summed, and standard deviations are combined by rounded root-sum-of-squares. The shared prefix is additional to these token counts. This approximates the sum of independently clipped distributions and does not preserve the system/user role distinction. Later prompts use the per-turn input distribution and retain conversation history, including live model outputs.
- **Turns and delays:** Turn counts are fixed at the upstream mean. Think-time means, standard deviations, and bounds are retained, but GuideLLM samples one delay per conversation and reuses it between turns; upstream samples each turn separately and rounds delays to integer seconds. No additional tool-call or branching requests are introduced: the synthetic upstream YAML simulates tool latency through sequential completion requests.
- **Load and stopping:** Upstream request totals become `min_requests` thresholds, allowing GuideLLM to continue admitting conversations even when a conversation has more turns than a stage's request count. Stopping may leave conversations incomplete and counts may overshoot. Concurrent streams limit requests, not sessions; delayed conversations can overlap with new conversations. Upstream worker affinity, worker counts, fixed recycled conversation pools, and exact shared-prefix reuse across stages are not reproduced.
- **Context limits:** GuideLLM does not reproduce inference-perf's rolling context truncation (262144 tokens for code generation; a 225000-token generator default for the other synthetic workloads). In particular, code generation can produce a first prompt exceeding one million tokens including its shared prefix, and later turns accumulate history. The server context limit must accommodate the chosen workload, or token/turn settings must be overridden. Backend and tokenizer differences can also affect actual token lengths.

## Agentic trace replay

The scenario loads `Exgentic/agent-llm-traces-v2` using the existing OTEL reader. Parquet loading filters sessions to `max_tokens < 25000`, matching the upstream context filter, then selects the first 50, 100, 200, 300, and 400 filtered sessions for successive runs. These deterministic slices are not upstream's session sampling. The dataset is fetched from its current revision, as in upstream; override the source's `load_kwargs.revision` to pin a dataset revision.

- **Timing and concurrency:** The replay profile honors recorded relative request timestamps and dependencies, with `max_wait: 15`. GuideLLM caps gaps between request starts, whereas upstream caps tool waits after a completion. OTEL sessions start at relative time zero, so this does not reproduce upstream's closed-loop limits of 5, 10, 20, 30, and 40 concurrent sessions. Each stage ends at dataset exhaustion without a request-count constraint.
- **History and tools:** `history: trace` preserves recorded full inputs, including independent calls whose context does not extend the preceding call. Runtime history cannot represent all such transitions. Required tool choice and the existing reader's tool-result injection are retained, but general substitution of live outputs into subsequent recorded inputs is not equivalent to inference-perf. GuideLLM can insert placeholder injection requests when a recorded continuation cannot be matched.
- **Errors and output lengths:** The OTEL reader excludes failed spans, and `on_bad_files: skip` skips unreadable Parquet files. This does not reproduce upstream's handling of every invalid session or its recorded-output fallback for malformed live tool calls. Upstream's tiered tool-output token allowances are also not reproduced; the existing GuideLLM tool-call behavior applies.

The original percentile SLOs in upstream descriptive configs are not enforced: they are not part of the runnable YAML, and GuideLLM's per-request goodput thresholds are not equivalent percentile objectives.
