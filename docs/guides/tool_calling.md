# Tool Calling

GuideLLM supports benchmarking multi-turn tool calling workloads. Client tool call turns are **pre-anticipated**: the data pipeline decides upfront which user turns expect a tool call and which expect plain text. Each client tool call user turn automatically generates an additional `tool_response_injection` request, so the total request count per conversation is `turns + len(tool_call_turns)`. For example, `turns=3, tool_call_turns=[0, 1]` produces 5 requests: tool call, injection, tool call, injection, standard.

When a client tool call turn completes, GuideLLM sends the tool response back to the server in a separate injection turn and waits for a text response before proceeding to the next user turn. The tool response content comes from one of three sources (in priority order): the dataset's tool response column, synthetic data configured via `tool_response_tokens`, or a short placeholder (`{"status": "ok"}`). Tool definitions are only included in the request body on turns that have a `tools_column` in their data; non-tool turns are sent as plain chat completions without tools or `tool_choice`.

## Supported Request Formats

Tool calling is supported with both `/v1/chat/completions` and `/v1/responses`. The CLI examples in this guide default to `/v1/chat/completions`, but all features work identically with `request_format=/v1/responses` in the backend configuration. The only difference is in how tool call history is represented in follow-up requests:

- **`/v1/chat/completions`** uses `role: "assistant"` messages with a `tool_calls` array, followed by `role: "tool"` messages carrying tool results.
- **`/v1/responses`** uses `function_call` items in the `input` array, followed by `function_call_output` items.

For the full wire format of tool call messages, see the [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling).

## Mocked client-side tool calls

GuideLLM currently supports mocked client-side tool calls (`turn_type="client_tool_call"`). This means that the inference server runs the model and may return real `tool_calls`, but GuideLLM **does not execute** those functions against live APIs or other runtimes. After each client tool call turn, the benchmark sends a separate tool response injection request (`turn_type="tool_response_injection"`) containing the mocked tool output, then waits for the server's text response before proceeding to the next user turn. This allows measuring LLM throughput with tool-call handling, not external tool latency or side effects.

## Server-side tool calls

For servers that handle tool execution internally (e.g. OpenAI Responses API with `container_auto`, or [OGX](https://github.com/ogx-ai/ogx) with MCP tool groups), use `server_tool_call` turns. These behave like standard turns from GuideLLM's perspective -- one request in, one response out -- but they prevent the backend from overriding `tool_choice` to `"none"`, so server-configured tools remain usable.

No injection turn is created, and GuideLLM does not mock any tool responses. The server runs the full tool-calling loop internally and returns the final text answer. Latency metrics include the server-side tool execution time.

### Synthetic data with server-side tool handling

For synthetic data, use `server_tool_call_turns` to mark user turns as server-managed. You can pass an int N (the first N turns), a list of indices, or `"all"` to mark every turn:

```bash
# All 3 turns are server_tool_call — the server decides per-turn whether to use tools
guidellm benchmark run \
  --target "http://localhost:8000" \
  --data '{"prompt_tokens": 200, "output_tokens": 100, "turns": 3, "server_tool_call_turns": "all"}' \
  --backend-kwargs '{"extras": {"body": {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}}}' \
  --max-requests 30
```

The `"all"` shorthand is equivalent to setting `server_tool_call_turns=N` where N equals `turns`, but avoids having to keep them in sync. It also works for `tool_call_turns` (client-side tool calling).

### Real datasets with server-side tool handling

For real datasets that contain tool definitions (e.g. a `tools` column), the finalizer's `tool_call_mode` setting controls whether those turns are treated as client-side or server-side tool calls:

- `tool_call_mode="client"` (default) -- turns with tool definitions become `client_tool_call` + `tool_response_injection` pairs. GuideLLM mocks tool responses and sends them back to the server.
- `tool_call_mode="server"` -- turns with tool definitions become `server_tool_call`. No injection turn is created. Tool definitions from the dataset are stripped; tools are expected to be configured at the backend level via `--backend` or on the server itself.

**OpenAI Responses API** -- tools are passed in the request body via `extras.body.tools` in the backend configuration:

```bash
guidellm run \
  --backend '{"kind": "openai_http", "target": "https://api.openai.com", "request_format": "/v1/responses", "extras": {"body": {"tools": [{"type": "shell", "environment": {"type": "container_auto"}}]}}}' \
  --data '{"kind": "huggingface", "source": "madroid/glaive-function-calling-openai", "load_kwargs": {"split": "train"}}' \
  --data-column-mapper '{"kind": "generative_column_mapper", "column_mappings": {"text_column": "messages", "tools_column": "tools"}}' \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer '{"kind": "generative", "tool_call_mode": "server"}' \
  --constraint '{"kind": "max_requests", "count": 50}' \
  --profile '{"kind": "constant", "rate": 1}'
```

**OGX (Llama Stack)** -- an open-source, OpenAI-compatible agentic server that handles tool execution server-side via its `/v1/responses` orchestration loop. Tools are configured on the OGX server (MCP tool groups, built-in tools), so no `extras.body.tools` is needed in the GuideLLM command:

```bash
guidellm run \
  --backend '{"kind": "openai_http", "target": "http://localhost:8321", "request_format": "/v1/responses"}' \
  --data '{"kind": "huggingface", "source": "madroid/glaive-function-calling-openai", "load_kwargs": {"split": "train"}}' \
  --data-column-mapper '{"kind": "generative_column_mapper", "column_mappings": {"text_column": "messages", "tools_column": "tools"}}' \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer '{"kind": "generative", "tool_call_mode": "server"}' \
  --constraint '{"kind": "max_requests", "count": 50}' \
  --profile '{"kind": "constant", "rate": 1}'
```

Tools for server-side tool calls are configured either in the request body via `--backend` (e.g. OpenAI `extras.body.tools`) or on the server itself (e.g. OGX MCP tool groups). The model decides per-turn whether to invoke them.

## Server Setup

Tool calling requires server-side support.

**vLLM** -- enable auto tool choice and a parser matching your model:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Common parsers: `hermes` (Qwen/Hermes), `llama3_json` (Llama 3.x), `mistral` (Mistral). Without these flags, vLLM will reject tool call output with grammar errors.

**OGX (Llama Stack)** -- for server-side tool execution, [OGX](https://github.com/ogx-ai/ogx) provides an OpenAI-compatible `/v1/responses` endpoint that handles the full tool-calling loop internally. OGX can use vLLM as its inference backend and supports MCP tool groups for server-side tool execution. See the [OGX documentation](https://ogx-ai.github.io/docs) for setup and tool group configuration.

## Providing Tool Definitions

Tool definitions are always provided through the data pipeline rather than as a global CLI flag. There are three ways to supply them:

**1. Synthetic data** -- set `tool_call_turns` (and optionally `tools`) in the data configuration:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-0.6B,request_format=/v1/chat/completions \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

To specify non-contiguous tool-call turns, pass a list of 0-based turn indices:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-0.6B,request_format=/v1/chat/completions \
  --data '{"kind":"synthetic_text","prompt_tokens":200,"output_tokens":100,"turns":4,"tool_call_turns":[0,2]}' \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

Synthetic data configuration fields for tool calling:

| Field                        | Type                        | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------- | --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool_call_turns`            | `int \| list[int] \| "all"` | `0`     | Which user turns include tool definitions and expect tool-call responses. Indices are 0-based into user turns (not the expanded request list). An int N means "the first N user turns"; a list specifies explicit indices (e.g. `[0, 2]`); `"all"` means every turn. Each tool-calling user turn generates an additional injection request, so `tool_call_turns=[0,1]` with `turns=3` produces 5 total requests. When `0` or `[]`, no tool calling.                               |
| `server_tool_call_turns`     | `int \| list[int] \| "all"` | `0`     | Which user turns use server-side tool calling. These turns are marked as `server_tool_call` so `tool_choice="none"` is not applied, letting the server use its configured tools. No injection turn is created. Must not overlap with `tool_call_turns`. An int N means "the first N user turns"; a list specifies explicit indices; `"all"` means every turn.                                                                                                                     |
| `tools`                      | `list`                      | `None`  | Tool definitions in either [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools) format (nested `function` key) or [Responses API](https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools) format (flat). GuideLLM auto-converts to match the `--request-format` at request time. Using the format that matches your target endpoint is recommended. When `None`, a built-in placeholder tool is used. |
| `tool_response_tokens`       | `int`                       | `None`  | Average number of tokens for synthetic tool call responses. When `None`, a short placeholder (`{"status": "ok"}`) is used.                                                                                                                                                                                                                                                                                                                                                        |
| `tool_response_tokens_stdev` | `int`                       | `None`  | Standard deviation for tool response token count.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `tool_response_tokens_min`   | `int`                       | `None`  | Minimum number of tokens for tool response.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `tool_response_tokens_max`   | `int`                       | `None`  | Maximum number of tokens for tool response.                                                                                                                                                                                                                                                                                                                                                                                                                                       |

Note: The token count is for the content of a field of the mock tool call response. The JSON structure adds ~5 tokens to the mock tool call response.

**Configuring tool response content** -- by default, tool results use a short placeholder (`{"status": "ok"}`). This default can be changed via the `GUIDELLM__DEFAULT_SYNTHETIC_TOOL_RESPONSE` environment variable. For more realistic benchmarks, set `tool_response_tokens` to generate variable-length JSON responses:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen3-0.6B,request_format=/v1/chat/completions \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2,tool_response_tokens=50 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

The `tool_response_tokens_stdev`, `tool_response_tokens_min`, and `tool_response_tokens_max` fields work identically to the corresponding `prompt_tokens_*` / `output_tokens_*` variance parameters.

**2. Datasets with a tools column** -- datasets that already contain tool definitions (e.g. `madroid/glaive-function-calling-openai`) work directly. The column mapper auto-detects columns named `tools`, `functions`, or `tool_definitions`.

**JSON-wrapped datasets** -- some HuggingFace datasets store all fields inside a single JSON string column (e.g. `madroid/glaive-function-calling-openai` has a `json` column containing `{"messages": [...], "tools": [...]}`). The column mapper automatically detects this pattern and unwraps the JSON to find the inner columns:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"messages","tools_column":"tools"}}' \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

The `tool_calling_message_extractor` preprocessor must be explicitly enabled via `--data-preprocessor` (it is not included by default). It parses each row's `messages` array and extracts prompts, system messages, and tool results into the appropriate columns. If the dataset has no tool result messages, the placeholder (`{"status": "ok"}`) is used as a fallback.

## Tool Choice and Missing Tool Call Behavior

Two backend settings control how tool-call turns are handled at runtime. Both are configured in the `--backend` config:

| Setting                      | Values                                         | Default      | Description                                                                                                                                                                    |
| ---------------------------- | ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `extras.body.tool_choice`    | `required`, `auto`, `none`                     | `required`   | Sent as the `tool_choice` API parameter on `client_tool_call` turns. The handler automatically sets `tool_choice="none"` on non-tool turns when tools are present in the body. |
| `tool_call_missing_behavior` | `ignore_continue`, `ignore_stop`, `error_stop` | `error_stop` | What the backend does when a tool call was expected but the model produced plain text instead.                                                                                 |

**Setting `tool_choice` via backend config:**

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"tool_choice":"auto"}}}' \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30
```

**Setting `tool_call_missing_behavior` via backend config:**

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","tool_call_missing_behavior":"ignore_continue","extras":{"body":{"tool_choice":"auto"}}}' \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30
```

**`tool_choice` implications:**

- `required` (default) -- the model **must** produce a tool call. This gives the most predictable benchmarks and the fewest errors, since the server constrains the output to valid tool call JSON. Use this when you don't want to rely on the model choosing to use tools. However, it may slow down the server due to forcing the server to choose low-probability options.
- `auto` -- the model decides whether to call a tool. Useful for testing how often a model chooses to invoke tools, but increases the chance of missing tool calls (see `tool_call_missing_behavior`).
- `none` -- tools are present in the request but the model cannot call them. This value is not set automatically by the pipeline (non-tool turns omit tools entirely); it is only useful when explicitly configured in the backend config alongside a global `tools` definition in extras.

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
- **Rate-limited profiles** (e.g. `--profile kind=constant,rate=1`) pace follow-up tool turns through the same scheduler as any other request. The follow-up turn is requeued and waits for the next available scheduling slot, so the effective delay between turns is determined by the profile, not by the tool calling logic.
