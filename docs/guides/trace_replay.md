# Trace File Formats

Many trace files are formatted in ways that need to be specially handled to create an accurate replay. This guide covers all trace file formats currently supported by GuideLLM, along with the format-agnostic and format-specific data arguments.

Detailed use of the replay profile and file-based datasets as a whole is explained in [Trace Replay Benchmarking](../getting-started/benchmark.md#trace-replay-benchmarking).

## Supported Formats

These are passed to the `--data` argument as `kind=format`:

- `trace_synthetic`: A trace format that does the bare minimum needed to complete a fully functioning trace replay benchmark with synthetic prompt generation
- `mooncake`: The trace format used by the serving platform *Mooncake*, as defined in [https://doi.org/10.48550/arXiv.2407.00079](https://doi.org/10.48550/arXiv.2407.00079)
- `weka`: The trace format used by WEKA's *Augmented Memory Grid*, as specified [in the original research repository](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md)
- `otel` (alias `opentelemetry`): OpenTelemetry GenAI spans. GuideLLM keeps successful LLM spans and replays each `trace_id` as one conversation. It sends recorded `gen_ai.input.messages` with each span's full input (`history=trace`) unless `history=runtime` is set.

## Loading Trace Data

Trace replay always uses `--profile kind=replay`. Choose a **format** (`trace_synthetic`, `mooncake`, or `weka`) and a **source** type from one of the other deserializer kinds (`json_file`, `huggingface`, etc). For example:

**`trace_synthetic` with local `json_file`:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,time_scale=2.0 \
  --constraint kind=max_requests,count=30
```

**WEKA dataset from `huggingface`:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=weka,source.kind=huggingface,source.source=semianalysisai/cc-traces-weka-no-subagents-051226 \
  --constraint kind=max_requests,count=30
```

**Mooncake dataset from `huggingface`**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=mooncake,source.kind=huggingface,source.source=valeriol29/mooncake-traces,load_kwargs.name=mooncake \
  --constraint kind=max_requests,count=30
```

**OTEL dataset from `huggingface`:**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
  --constraint kind=max_requests,count=30
```

## Format-Agnostic Data Arguments

All trace formats can accept the following optional data arguments:

| Argument                  | Default         | Description                                                                                     |
| ------------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| `timestamp_column`        | "timestamp"     | Column name for timestamps in the trace file                                                    |
| `prompt_tokens_column`    | "input_length"  | Column name for prompt token counts in the trace file                                           |
| `output_tokens_column`    | "output_length" | Column name for output token counts in the trace file                                           |
| `time_scale`              | 1.0             | Scale remaining relative timestamps after wait and pack caps                                    |
| `max_wait`                | unset           | Maximum gap in original trace seconds between consecutive requests in one session               |
| `max_session_wait`        | unset           | Maximum idle in original trace seconds from the previous session's last request to this session |
| `min_concurrent_sessions` | unset           | Pack sessions so at least this many overlap during steady state                                 |

These are passed through the `--data` argument like below:

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data "kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,timestamp_column=ts,prompt_tokens_column=input_tokens,output_tokens_column=generated_tokens,time_scale=1.0,max_session_wait=30"
```

`trace_synthetic` can be thought of as the format-agnostic option, only looking for the timestamp, prompt token count and output token count columns and ignoring all other features contained in a dataset. While primarily used for testing, `trace_synthetic` may be used as a fallback for trace formats not currently supported by GuideLLM.

`trace_synthetic` and `mooncake` replay each row as an independent, single-request conversation. Rows are sorted by timestamp and keep their offsets from the first request in the trace. Prompts are generated as rows are consumed, and Mooncake hash IDs remain shared across rows. Use `max_session_wait` to cap gaps between these independent requests; `max_wait` only caps gaps within multi-request conversations, such as WEKA sessions.

## Format-Specific Data Arguments

### `mooncake`

The Mooncake format expects an additional column for prefix-based cache hash IDs. During prompt generation, hash IDs sharing the same previous ID are required to represent distinct blocks of token ids.

| Argument             | Default    | Description                                         |
| -------------------- | ---------- | --------------------------------------------------- |
| `hash_ids_column`    | "hash_ids" | Column name for lists of hash IDs in the trace file |
| `hash_id_block_size` | 512        | Amount of tokens represented by one hash ID         |

### `weka`

**NOTE:** Warm `tool_tokens`/`system_tokens` prefixes and hash-id LCP splitting of flattened agents are not implemented. Declared `type: "subagent"` groups and tool-call events (`stop: tool_use`, `input_types: ["tool_result"]`) are replayed.

The WEKA format expects a column with conversation UUIDs that is not wrapped within another column. The timestamp, input token length, output token length and hash IDs columns must all be wrapped inside one JSON column (ex. "requests"), in the form of a list of JSON objects.

Similar to Mooncake, WEKA uses prefix-based cache hash IDs. The original [specification](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md) for the trace requires hash IDs to be 1 or greater, and for trailing hash IDs to be dropped if there are not enough input tokens to fill the hash ID block size. To accommodate for datasets which may not follow the specification exactly (ex. [semianalysisai/cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226)), GuideLLM will accept any non-negative integer as a valid hash ID, and will drop partially filled hash IDs if they exist.

GuideLLM will generate prompts starting from the first conversation. When the conversation ends, the next conversation will be used. Relative timestamps are local to the conversation and return to 0.0 after each conversation ends.

Hash IDs follow the per-row `hash_id_scope` field:

- `"global"` or omitted: hash IDs share one token-block table across conversations, matching Mooncake. The same hash ID in a later conversation reuses the earlier token block so prefix-cache hit rate stays close to the original trace.
- `"local"`: hash IDs apply only within that conversation. The table is discarded after the conversation is emitted.

Declared `type: "subagent"` entries become isolated child chains. Each child spawns from the preceding parent API turn with a fresh history (`history_context="new"`) and the following parent turn waits for every sibling spawned since that turn (`history_context="last"`). Multiple subagents listed between the same parent turns therefore run in parallel; the parent resumes only after all of them complete. Request-list order is preserved at every nesting level (it is the spawn/join topology) and is not sorted by timestamp.

Inner request timestamps follow the spec when they are relative to spawn, and published Hugging Face corpora when they are already absolute: if the first inner `t` is less than the subagent entry's spawn `t`, inner times are treated as `spawn_t + inner_t`; otherwise they are left as-is. Conversation-relative timestamps are then `absolute_t - min_t` across all API requests in that conversation.

A single agent's consecutive turns are still serialized. If those turns overlap in time (`t[i] + api_time[i] > t[i+1]`, or `t[i+1] <= t[i]` when `api_time` is absent), GuideLLM logs a debug message. Overlap between different subagents is intended parallelism and is not warned.

Tool-call events map onto GuideLLM's existing client tool-call pipeline. A request with `stop: "tool_use"` and user text input becomes a `client_tool_call` turn. The following request with `input_types: ["tool_result"]` (or, if `input_types` is absent, the next request after `stop: "tool_use"` on the same agent chain) becomes a `tool_response_injection`. When that injection row also has `stop: "tool_use"`, it still sends tool results and keeps `tools` so the model may emit further tool calls. Traces do not contain real tool schemas or results. Pass `tools` and optionally `tool_response_tokens` the same way as [synthetic data](tool_calling.md#providing-tool-definitions); otherwise GuideLLM uses the default synthetic tool definition and placeholder tool response. Chat handlers do not send the hash-id prompt as a user message on injection turns.

| Argument                     | Default    | Description                                                                                           |
| ---------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| `conversation_id_column`     | "id"       | Column name for conversation UUIDs in the trace file                                                  |
| `hash_ids_column`            | "hash_ids" | Column name for lists of hash IDs in the trace file                                                   |
| `hash_id_block_size`         | 64         | Amount of tokens represented by one hash ID                                                           |
| `tools`                      | `None`     | OpenAI-format tool definitions for tool-call turns. When unset, the built-in placeholder tool is used |
| `tool_response_tokens`       | `None`     | Average tokens for mocked tool results. When unset, a short placeholder (`{"status": "ok"}`) is used  |
| `tool_response_tokens_stdev` | `None`     | Standard deviation for tool response token count                                                      |
| `tool_response_tokens_min`   | `None`     | Minimum number of tokens for tool response                                                            |
| `tool_response_tokens_max`   | `None`     | Maximum number of tokens for tool response                                                            |

Modified defaults:

| Argument               | New Default |
| ---------------------- | ----------- |
| `timestamp_column`     | "t"         |
| `prompt_tokens_column` | "in"        |
| `output_tokens_column` | "out"       |

### `otel`

OpenTelemetry GenAI traces are replayed as timed conversations. Two file layouts are accepted:

- **Session-per-line**: each JSONL row is `{ "trace_id": ..., "spans": [ ... ] }`
- **Span-per-line**: each JSONL row is one span; adjacent rows with the same `trace_id` become one conversation. Grouping is streaming and consecutive only, so an interleaved `a, b, a` dump is three conversations, not two. Published replay corpora write each `trace_id` contiguously; live collector exports of concurrent traces may not.

Only successful LLM spans are replayed (`gen_ai.operation.name` of `chat`, `generate`, or `text_completion`, or any span that already has usage token attributes). `invoke_agent` and failed spans (`status.code` error) are dropped. `execute_tool` spans are not sent as HTTP requests.

Each LLM span is a full API snapshot: `gen_ai.input.messages` re-records the whole transcript, and `prompt_tokens` / `input_tokens` is **that call's full input size**, not a delta. `history` controls how that snapshot is turned into a request:

| `history=trace` (default)                                                                                                                                                                                                              | `history=runtime`                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Send the span's full recorded messages. `history_context=new`, so live completions are never substituted into the next prompt. Most accurate. Prefix cache breaks after the live completion diverges from the recorded assistant text. | DAG `full` history of *live* outputs; current turn is only the **new** messages. Requires `input[i] == input[i-1] + output[i-1] + delta`. Cache-friendly; accuracy suffers when the live model diverges. |

Output length always uses the span's completion token count (`max_tokens` + `ignore_eos`).

Spans without `gen_ai.input.messages` cannot be replayed as OTEL. Flatten token-count-only dumps to `kind=trace_synthetic` instead.

Replay against `/v1/chat/completions` (the backend default). OTEL replay is chat-completions only for now: `raw_messages_column` is always chat-completions format and that handler sends it as a `messages` array. `/v1/completions` is a prompt string with no tool loop and does not read `raw_messages_column`.

Recorded tool loops are pre-split onto GuideLLM's client tool-call pipeline. An LLM span whose output messages contain `tool_calls` (or `gen_ai.response.finish_reasons` of `tool_calls` / `tool_call` / `tool_use` / `function_call`) becomes `client_tool_call`. The next span is consumed as `tool_response_injection` when its new messages after `input[i] + output[i]` are only `role=tool` results. Recorded result strings are rebound to **live** `tool_call_id`s by the chat handler. `gen_ai.tool.definitions` supplies `tools_column` on those turns (otherwise the default synthetic tool is used); definitions alone do not classify a turn. Injection parents always use `history_context=full`, including `history=trace`. Missing-tool policy stays `--backend tool_call_missing_behavior=...`.

If the next span cannot be parsed as tool results, a placeholder injection is synthesized (using a following `execute_tool` span's result when present) and that next span is still replayed.

Completed request stats merge response usage over the request's expected token counts, so a dedicated expected-vs-actual MAE is not reported. Compare `request.input_metrics` / `output_metrics` (span counts) with response usage before that merge if you need the deviation.

ISO-8601 `start_time` values (naive, `Z`, or offset) and HuggingFace-decoded `datetime` objects are converted to epoch seconds before scheduling. Token counts are read from span `attributes`, trying current GenAI names first and then the deprecated aliases. OTel `parts` (`text`, `tool_call`, `tool_call_response`) are converted to OpenAI chat dicts.

| Argument                   | Default                                                            | Description                                                                                                   |
| -------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `spans_column`             | "spans"                                                            | Column name for nested span lists in session-per-line files                                                   |
| `trace_id_column`          | "trace_id"                                                         | Column used to group span-per-line files into conversations                                                   |
| `span_timestamp_field`     | "start_time"                                                       | Span field holding the request start time                                                                     |
| `input_tokens_attributes`  | `["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"]`      | Attribute keys tried in order for prompt token counts                                                         |
| `output_tokens_attributes` | `["gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"]` | Attribute keys tried in order for output token counts                                                         |
| `history`                  | `trace`                                                            | `trace` resends each span's full input; `runtime` sends only new messages with DAG history                    |
| `tool_choice`              | `required`                                                         | `required` or `auto` on client tool-call turns. Pair `auto` with `tool_call_missing_behavior=ignore_continue` |

Start from Hugging Face. IBM traces have 30–50 LLM calls each, so `--constraint kind=max_requests` is a useful bound on first runs.

**Default (`history=trace`):** send each span's recorded messages in full. Later turns wait on the DAG but use `history_context=new`, so live completions are not spliced into the next prompt.

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
    --constraint kind=max_requests,count=30
```

**Recorded messages with DAG history (`history=runtime`):** send only the new messages; prior turns come from live completions (`history_context=full`). Requires each span's input to continue the previous span's input plus output.

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces,history=runtime \
    --constraint kind=max_requests,count=30
```

Local JSONL uses the same `history` switch with `source.kind=json_file,source.path=...`.

Public Hugging Face corpora:

- [ibm-research/synthetic-conversations-traces](https://huggingface.co/datasets/ibm-research/synthetic-conversations-traces) — multi-turn chats, session-per-line JSONL, `prompt_tokens` / `completion_tokens`
- [Exgentic/agent-llm-traces-v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2) — agent sessions, nested `spans`, `input_tokens` / `output_tokens`
- [ibm-research/lmcache-agentic-traces_Otel](https://huggingface.co/datasets/ibm-research/lmcache-agentic-traces_Otel) — agentic sessions, session-per-line JSONL, deprecated token keys

Related corpora: [DiscoPosse/agent-llm-traces](https://huggingface.co/datasets/DiscoPosse/agent-llm-traces) (Exgentic v1 schema) and [lenadan/otel-test-snippet-jsonl](https://huggingface.co/datasets/lenadan/otel-test-snippet-jsonl) (small span-per-line snippet). [ibm-research/codex_swebenchpro_traces_Otel](https://huggingface.co/datasets/ibm-research/codex_swebenchpro_traces_Otel) omits usage token attributes and is not usable for token-count replay. Spans without `gen_ai.input.messages` should use `trace_synthetic`.
