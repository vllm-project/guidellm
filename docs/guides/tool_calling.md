# Tool Calling

GuideLLM supports benchmarking multi-turn tool calling workloads. Tool call turns are **pre-anticipated**: the data pipeline decides upfront which turns expect a tool call and which expect plain text. GuideLLM does not dynamically create follow-up turns at runtime. Instead, the full conversation structure is planned during data generation, and the worker executes each turn in order, with each tool call being scheduled like any other turn by the profile.

When a tool-call turn completes, GuideLLM appends a tool result to the conversation history and proceeds to the next pre-planned turn. The tool result content comes from one of three sources (in priority order): the dataset's tool response column, synthetic data configured via `tool_response_tokens`, or a short placeholder (`{"status": "ok"}`). Tool definitions are only included in the request body on turns that have a `tools_column` in their data; non-tool turns are sent as plain chat completions without tools or `tool_choice`.

## Mocked client-side tool calls

GuideLLM currently supports mocked client-side tool calls. This means that the inference server runs the model and may return real `tool_calls`, but GuideLLM **does not execute** those functions against live APIs or other runtimes. The benchmark worker acts as a **mock client**: after each tool-call turn it injects the next `role: "tool"` message into client-side chat history for the following request. This allows measuring LLM throughput with tool-call handling, not external tool latency or side effects.

## Server Setup

Tool calling requires server-side support. For vLLM, enable auto tool choice and a parser matching your model:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Common parsers: `hermes` (Qwen/Hermes), `llama3_json` (Llama 3.x), `mistral` (Mistral). Without these flags, vLLM will reject tool call output with grammar errors.

## Providing Tool Definitions

Tool definitions are always provided through the data pipeline rather than as a global CLI flag. There are three ways to supply them:

**1. Synthetic data** -- set `tool_call_turns` (and optionally `tools`) in the data configuration:

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-0.6B" \
  --request-format /v1/chat/completions \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 3, "tool_call_turns": 2}' \
  --max-requests 30 \
  --profile constant \
  --rate 1
```

To specify non-contiguous tool-call turns, pass a list of 0-based turn indices:

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-0.6B" \
  --request-format /v1/chat/completions \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 4, "tool_call_turns": [0, 2]}' \
  --max-requests 30 \
  --profile constant \
  --rate 1
```

Synthetic data configuration fields for tool calling:

| Field                        | Type               | Default | Description                                                                                                                                                                                                                                                 |
| ---------------------------- | ------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool_call_turns`            | `int \| list[int]` | `0`     | Which turns include tool definitions and expect tool-call responses. An int N means "the first N turns"; a list of ints specifies explicit 0-based turn indices (e.g. `[0, 2]`). Normalized to a sorted list internally. When `0` or `[]`, no tool calling. |
| `tools`                      | `list`             | `None`  | Tool definitions in OpenAI format. When `None`, a built-in placeholder tool is used. Custom definitions can be provided inline: `"tools": [{"type": "function", ...}]`.                                                                                     |
| `tool_response_tokens`       | `int`              | `None`  | Average number of tokens for synthetic tool call responses. When `None`, a short placeholder (`{"status": "ok"}`) is used.                                                                                                                                  |
| `tool_response_tokens_stdev` | `int`              | `None`  | Standard deviation for tool response token count.                                                                                                                                                                                                           |
| `tool_response_tokens_min`   | `int`              | `None`  | Minimum number of tokens for tool response.                                                                                                                                                                                                                 |
| `tool_response_tokens_max`   | `int`              | `None`  | Maximum number of tokens for tool response.                                                                                                                                                                                                                 |

Note: The token count is for the content of a field of the mock tool call response. The JSON structure adds ~5 tokens to the mock tool call response.

**Configuring tool response content** -- by default, tool results use a short placeholder (`{"status": "ok"}`). This default can be changed via the `GUIDELLM__DEFAULT_SYNTHETIC_TOOL_RESPONSE` environment variable. For more realistic benchmarks, set `tool_response_tokens` to generate variable-length JSON responses:

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "Qwen/Qwen3-0.6B" \
  --request-format /v1/chat/completions \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 3, "tool_call_turns": 2, "tool_response_tokens": 50}' \
  --max-requests 30 \
  --profile constant \
  --rate 1
```

The `tool_response_tokens_stdev`, `tool_response_tokens_min`, and `tool_response_tokens_max` fields work identically to the corresponding `prompt_tokens_*` / `output_tokens_*` variance parameters.

**2. Datasets with a tools column** -- datasets that already contain tool definitions (e.g. `madroid/glaive-function-calling-openai`) work directly. The column mapper auto-detects columns named `tools`, `functions`, or `tool_definitions`:

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --data "madroid/glaive-function-calling-openai" \
  --data-column-mapper '{"text_column": "messages", "tools_column": "tools"}' \
  --data-preprocessors "tool_calling_message_extractor,encode_media" \
  --max-requests 50 \
  --profile constant \
  --rate 1
```

The `tool_calling_message_extractor` preprocessor must be explicitly enabled via `--data-preprocessors` (it is not included by default). It parses each row's `messages` array and extracts prompts, system messages, and tool results into the appropriate columns. If the dataset has no tool result messages, the placeholder (`{"status": "ok"}`) is used as a fallback.

## Tool Choice and Missing Tool Call Behavior

Two backend settings control how tool-call turns are handled at runtime. Both are configured via `--backend-kwargs`:

| Setting                      | Values                                         | Default      | Description                                                                                                                                      |
| ---------------------------- | ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `extras.body.tool_choice`    | `required`, `auto`, `none`                     | `required`   | Sent as the `tool_choice` API parameter on turns that expect a tool call. Non-tool turns omit tools and `tool_choice` from the request entirely. |
| `tool_call_missing_behavior` | `ignore_continue`, `ignore_stop`, `error_stop` | `error_stop` | What the backend does when a tool call was expected but the model produced plain text instead.                                                   |

**Setting `tool_choice` via `--backend-kwargs`:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 3, "tool_call_turns": 2}' \
  --backend-kwargs '{"extras": {"body": {"tool_choice": "auto"}}}' \
  --max-requests 30
```

**Setting `tool_call_missing_behavior` via `--backend-kwargs`:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 3, "tool_call_turns": 2}' \
  --backend-kwargs '{"tool_call_missing_behavior": "ignore_continue", "extras": {"body": {"tool_choice": "auto"}}}' \
  --max-requests 30
```

**`tool_choice` implications:**

- `required` (default) -- the model **must** produce a tool call. This gives the most predictable benchmarks and the fewest errors, since the server constrains the output to valid tool call JSON. Use this when you don't want to rely on the model choosing to use tools. However, it may slow down the server due to forcing the server to choose low-probability options.
- `auto` -- the model decides whether to call a tool. Useful for testing how often a model chooses to invoke tools, but increases the chance of missing tool calls (see `tool_call_missing_behavior`).
- `none` -- tools are present in the request but the model cannot call them. This value is not set automatically by the pipeline (non-tool turns omit tools entirely); it is only useful when explicitly configured via `--backend-kwargs` alongside a global `tools` definition in extras.

Note that `required` vs `auto` can also result in different model behavior. For example, the Qwen models only show their pre-tool-call thinking with `auto`.

**`tool_call_missing_behavior` implications:**

This setting only matters when `tool_choice` is `auto` (or `required` and the server doesn't enforce it):

- `error_stop` (default) -- the current turn is marked as **errored** and all remaining turns are cancelled. Surfaces problems immediately. Best for validating that the model and server are correctly configured.
- `ignore_stop` -- the current turn is marked as **cancelled** (incomplete) and all remaining turns are cancelled. The model's response is preserved in the output but the turn's status reflects that the expected tool call was not produced. Use this when a missing tool call means the conversation can't continue meaningfully but shouldn't be treated as an error.
- `ignore_continue` -- the current turn is treated as **completed** and the conversation continues to the next turn. Each future tool-call turn is evaluated independently. Use this when you want to measure how many tool calls actually happen under `auto` mode without aborting the conversation.

### Recommended scenarios

| Tool Choice | Missing Behavior  | Current Turn Status | Description                                                                                                  |
| ----------- | ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `required`  | `error_stop`      | errored             | (default) Good for consistent and predictable behavior with synthetic data. May slow down the server.        |
| `auto`      | `ignore_continue` | completed           | Good for testing `auto` behavior without the model choosing to not use a tool call causing errors.           |
| `auto`      | `ignore_stop`     | cancelled           | Good for testing `auto` behavior but ends the conversation early once the model creates a non-tool response. |

## Output Token Limits on Tool-Call Turns

When `output_tokens` is configured (either via synthetic data or a dataset column), GuideLLM normally sets `ignore_eos=True` and clears stop sequences to force the model to generate exactly N tokens. On tool-call turns, these settings are **automatically removed** because they are incompatible with vLLM's constrained decoding grammar:

- **`ignore_eos`** conflicts with the grammar's terminal state. Constrained decoding guides token selection via a finite-state machine that marks EOS as the only valid token once the JSON is complete. `ignore_eos` masks out EOS, creating an impossible state with no valid tokens — causing server errors or runaway generation.
- **`stop=None`** removes stop sequences that the tool-call parser may rely on internally (e.g. `<|eot_id|>` for Llama models).
- **`max_tokens` / `max_completion_tokens`** would truncate mid-JSON, producing invalid tool call arguments that corrupt the conversation history on follow-up turns.

As a result, tool-call turns generate output whose length is determined by the model and tool schema rather than the configured `output_tokens`. The model stops as soon as it produces valid JSON for the function name and arguments. This typically results in shorter output (20–80 tokens) compared to the configured target.

Plain-text turns are unaffected and continue to respect `output_tokens` / `ignore_eos` as normal. These turns do not include tool definitions or `tool_choice` in the request.

## Edge Cases

- **Single-turn tool calling** (`turns=1, tool_call_turns=1` or `tool_call_turns=[0]`) is supported. The conversation has one turn that expects a tool call and no plain-text response.
- **All-tool conversations** (e.g. `tool_call_turns=3` with `turns=3`, or `tool_call_turns=[0,1,2]`) are supported. Every turn is a tool-call turn and the model never produces a final plain-text response. The `output` field in `benchmarks.json` will be `None` for every request; use the `tool_calls` field to inspect model output.
- **Non-contiguous tool turns** (e.g. `tool_call_turns=[0, 2]` with `turns=4`) are supported. Only the specified turns expect tool calls; other turns produce plain text.
- **Tool definitions on non-tool turns** are not included in the request. The data pipeline only attaches `tools_column` to turns that expect a tool call, so non-tool turns are sent as plain chat completions without tools or `tool_choice`.
- **Mixed datasets** where only some rows have a `tools_column` work correctly. Rows without tools are treated as plain text conversations; rows with tools follow the tool-call flow.
- **Rate-limited profiles** (e.g. `--profile constant --rate 1`) pace follow-up tool turns through the same scheduler as any other request. The follow-up turn is requeued and waits for the next available scheduling slot, so the effective delay between turns is determined by the profile, not by the tool calling logic.
