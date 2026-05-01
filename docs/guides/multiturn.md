# Multiturn Conversation Benchmarking

This guide demonstrates how to utilize GuideLLM to orchestrate multi-turn benchmarks for simulating user conversations in a multi-request/response pattern.

## Setup

First, ensure you have a running inference server and compatible model. GuideLLM supports any OpenAI-compatible server that can handle conversational interactions. For the benchmarking examples below, we'll use vLLM serving a conversational model.

```bash
# Example: vLLM with a conversational model
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Next, either on the same instance or another machine that can reach your server, install GuideLLM:

```bash
pip install guidellm[recommended]
```

## Understanding Multiturn Data Structure

Multiturn benchmarking in GuideLLM uses indexed columns to represent conversational exchanges. Each turn in a conversation is represented by a set of columns with a numeric suffix indicating the turn index. Turn indexes can be any numerical value but we recommend ascending from 0 for simplicity.

### Turn-indexed Column Format

For a 3-turn conversation, the dataset could contain the following columns:

- **prefix**: Optional system prompt that becomes a system message (applies to first turn)
- **prompt_0**, **output_tokens_count_0**: Turn 1 prompt and requested output tokens
- **prompt_1**, **output_tokens_count_1**: Turn 2 prompt and requested output tokens
- **prompt_2**, **output_tokens_count_2**: Turn 3 prompt and requested output tokens

### How Multiturn Orchestration Works

When executing a multiturn benchmark, GuideLLM:

1. **Sends a turn** (prefix + prompt_0) to the model and captures the response
2. **Return the request** Store the request/response in the aggregator as a single request
3. **Builds conversation history** by combining the requests and the model's responses
4. **Sends the next turn** (prompt_i) along with the conversation history
5. **Repeat from (2)** for the `n` given turns

For `/v1/chat/completions`, the conversation history is passed as a messages array with alternating user and assistant roles. For `/v1/responses`, the history is either passed as alternating user and assistant roles, or as a previous request ID. For `/v1/completions`, the history is concatenated as a single prompt string.

For more information see [Request Formatting](#request-formatting) and [Server-Side Conversation History](#server-side-conversation-history-v1responses-only).

### Prefix Columns and System Prompts

Prefix columns (if present) are treated specially:

- In `/v1/chat/completions`, the prefix becomes a system message in the conversation array
- In `/v1/responses`, the prefix becomes the `instructions` field
- In `/v1/completions`, the prefix is prepended to the turn's prompt
- Prefixes can be specified with a turn index if desired; however the recommended use-case is a single prefix for the first turn
- Synthetic data only supports a prefix on the first turn

## Processing Options

All standard benchmarking arguments apply to multiturn tasks, such as `--profile`, `--rate`, and `--max-requests`. Any options that operate on "requests" will treat each turn as a separate request (e.g. A dataset row with 3 turns will count as 3 requests).

### Synthetic Data Configuration

GuideLLM can automatically generate multiturn synthetic data using the `turns` parameter in the synthetic data configuration.

#### Basic Synthetic Multiturn

To generate multiturn synthetic data, use the `--data` argument with `turns` specified:

```bash
--data "prompt_tokens=256,output_tokens=128,turns=3"
```

This creates a 3-turn conversation where each turn has 256 prompt tokens and requests 128 output tokens.

#### Synthetic Data with Prefixes

You can add system prompts (prefixes) to synthetic conversations using two approaches:

**Simple Prefix Configuration:**

```bash
--data "prompt_tokens=256,output_tokens=128,turns=3,prefix_count=5,prefix_tokens=50"
```

This generates 5 unique prefixes of 50 tokens. Every conversation will select one of these 5 at random as the system message.

**Advanced Prefix Configuration:**

For more complex scenarios, use `prefix_buckets` to create weighted distributions of different prefix configurations. This requires passing a JSON configuration:

```bash
--data '{
  "prompt_tokens": 256,
  "output_tokens": 128,
  "turns": 3,
  "prefix_buckets": [
    {"bucket_weight": 60, "prefix_count": 10, "prefix_tokens": 100},
    {"bucket_weight": 40, "prefix_count": 1, "prefix_tokens": 50}
  ]
}'
```

For this configuration:

- 60% of conversations use one of the 10 prefixes which are 100 tokens each
- 40% of conversations use the prefix of 50 tokens

### Request Formatting

Multiturn conversations are formatted differently depending on the request format:

#### Chat Completions (`/v1/chat/completions`)

For chat completions, GuideLLM creates a `messages` array with the conversation history:

```json
{
  "messages": [
    {"role": "system", "content": "prefix content"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_2 content"}]}
  ]
}
```

#### Responses API (`/v1/responses`)

For the Responses API with `server_history` disabled, GuideLLM creates an `input` array with the conversation history and sets the prefix as `instructions`:

```json
{
  "instructions": "prefix content",
  "input": [
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_2 content"}]}
  ]
}
```

#### Text Completions (`/v1/completions`)

For text completions, the conversation history is concatenated:

```text
prefix content prompt_0 content response to prompt_0 prompt_1 content response to prompt_1 prompt_2 content
```

### Server-Side Conversation History (`/v1/responses` only)

By default, GuideLLM replays the full conversation history in each request (client-side history). For the Responses API, you can instead use **server-side history** via the `previous_response_id` field, where the server stores and manages conversation context.

Enable it with `--backend-kwargs`:

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --request-format /v1/responses \
  --backend-kwargs '{"server_history": true}' \
  --data "prompt_tokens=200,output_tokens=100,turns=3"
```

When enabled, GuideLLM sends only the current turn's input and references the previous response by ID. The server reconstructs the full conversation context internally.

**Requirements:**

- The server must support `previous_response_id` with response storage enabled. For vLLM, set the `VLLM_ENABLE_RESPONSES_API_STORE=1` environment variable when starting the server.
- If the server does not support response storage, requests on turn 2+ will fail with an error (typically a 404).
- This option is only valid with `/v1/responses`. Using it with other request formats raises an error at startup.

## Tool Calling

GuideLLM supports benchmarking multi-turn tool calling workloads. Tool call turns are **pre-anticipated**: the data pipeline decides upfront which turns expect a tool call and which expect plain text. GuideLLM does not dynamically create follow-up turns at runtime. Instead, the full conversation structure is planned during data generation, and the worker executes each turn in order, with each tool call being scheduled like any other turn by the profile.

When a tool-call turn completes, GuideLLM appends a tool result to the conversation history and proceeds to the next pre-planned turn. The tool result content comes from one of three sources (in priority order): the dataset's tool response column, synthetic data configured via `tool_response_tokens`, or a short placeholder (`{"status": "ok"}`). All turns where a tool call is not anticipated have `tool_choice` overridden to `"none"` for predictability.

### Mocked client-side tool calls

GuideLLM currently supports mocked client-side tool calls. This means that the inference server runs the model and may return real `tool_calls`, but GuideLLM **does not execute** those functions against live APIs or other runtimes. The benchmark worker acts as a **mock client**: after each tool-call turn it injects the next `role: "tool"` message into client-side chat history for the following request. This allows measuring LLM throughput with tool-call handling, not external tool latency or side effects.

### Server Setup

Tool calling requires server-side support. For vLLM, enable auto tool choice and a parser matching your model:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Common parsers: `hermes` (Qwen/Hermes), `llama3_json` (Llama 3.x), `mistral` (Mistral). Without these flags, vLLM will reject tool call output with grammar errors.

### Providing Tool Definitions

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

Synthetic data configuration fields for tool calling:

| Field                        | Type   | Default | Description                                                                                                                                                                                                            |
| ---------------------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool_call_turns`            | `int`  | `0`     | Number of turns (from the start) that include tool definitions and expect tool-call responses. Must be `<= turns`. When equal to `turns`, every turn is a tool-call turn and no final plain-text response is produced. |
| `tools`                      | `list` | `None`  | Tool definitions in OpenAI format. When `None`, a built-in placeholder tool is used. Custom definitions can be provided inline: `"tools": [{"type": "function", ...}]`.                                                |
| `tool_response_tokens`       | `int`  | `None`  | Average number of tokens for synthetic tool call responses. When `None`, a short placeholder (`{"status": "ok"}`) is used.                                                                                             |
| `tool_response_tokens_stdev` | `int`  | `None`  | Standard deviation for tool response token count.                                                                                                                                                                      |
| `tool_response_tokens_min`   | `int`  | `None`  | Minimum number of tokens for tool response.                                                                                                                                                                            |
| `tool_response_tokens_max`   | `int`  | `None`  | Maximum number of tokens for tool response.                                                                                                                                                                            |

Note: The token count is for the content of a field of the mock tool call response. The JSON structure adds ~5 tokens to the mock tool call response.

**Configuring tool response content** -- by default, tool results use a short placeholder (`{"status": "ok"}`). For more realistic benchmarks, set `tool_response_tokens` to generate variable-length JSON responses:

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

### Tool Choice and Missing Tool Call Behavior

Two CLI options control how tool-call turns are handled at runtime:

| Option                         | Values                                         | Default      | Description                                                                                                                     |
| ------------------------------ | ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `--tool-choice`                | `required`, `auto`, `none`                     | `required`   | Sent as the `tool_choice` API parameter on turns that expect a tool call. On non-tool turns, it is always overridden to `none`. |
| `--tool-call-missing-behavior` | `ignore_continue`, `ignore_stop`, `error_stop` | `error_stop` | What the worker does when a tool call was expected but the model produced plain text instead.                                   |

**`--tool-choice` implications:**

- `required` (default) -- the model **must** produce a tool call. This gives the most predictable benchmarks and the fewest errors, since the server constrains the output to valid tool call JSON. Use this unless you specifically want to test the model's autonomous tool-use decisions.
- `auto` -- the model decides whether to call a tool. Useful for testing how often a model chooses to invoke tools, but increases the chance of missing tool calls (see `--tool-call-missing-behavior`).
- `none` -- tools are present in the request but the model cannot call them. This is primarily set automatically on the final (plain-text) turn; setting it globally disables tool calling entirely.

Note that `required` vs `auto` can also result in different model behavior. For example, the Qwen models only show their pre-tool-call thinking with `auto`.

**`--tool-call-missing-behavior` implications:**

This setting only matters when `--tool-choice` is `auto` (or `required` and the server doesn't enforce it):

- `error_stop` (default) -- the current turn is marked as errored and all remaining turns are cancelled. Surfaces problems immediately. Best for validating that the model and server are correctly configured.
- `ignore_stop` -- the current turn is treated as successful (the response is kept), but all remaining turns are cancelled. Use this when a missing tool call means the conversation can't continue meaningfully but isn't an error per se.
- `ignore_continue` -- the current turn is treated as successful and the conversation continues to the next turn. Each future tool-call turn is evaluated independently. Use this when you want to measure how many tool calls actually happen under `auto` mode without aborting the conversation.

#### Recommended scenarios

| Tool Choice | Missing Behavior  | Description                                                                                                  |
| ----------- | ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `required`  | `error_stop`      | (default) Good for consistent and predictable behavior.                                                      |
| `auto`      | `ignore_continue` | Good for testing `auto` behavior without the model choosing to not use a tool call causing errors.           |
| `auto`      | `ignore_stop`     | Good for testing `auto` behavior but ends the conversation early once the model creates a non-tool response. |

### Edge Cases

- **Single-turn tool calling** (`turns=1, tool_call_turns=1`) is supported. The conversation has one turn that expects a tool call and no plain-text response.
- **All-tool conversations** (`tool_call_turns == turns`) are supported. Every turn is a tool-call turn and the model never produces a final plain-text response. The `output` field in `benchmarks.json` will be `None` for every request; use the `tool_calls` field to inspect model output.
- **Tool definitions on non-tool turns** are still sent in the request (they're part of the data), but `tool_choice` is forced to `none` so the model produces text. This matches real-world agentic patterns where the tools remain available but the model is instructed to respond in natural language.
- **Mixed datasets** where only some rows have a `tools_column` work correctly. Rows without tools are treated as plain text conversations; rows with tools follow the tool-call flow.
- **Rate-limited profiles** (e.g. `--profile constant --rate 1`) pace follow-up tool turns through the same scheduler as any other request. The follow-up turn is requeued and waits for the next available scheduling slot, so the effective delay between turns is determined by the profile, not by the tool calling logic.

## The TurnPivot Preprocessor

GuideLLM supports passing multiple `--data` options, each pointing to a separate dataset. If there are matches for the same column type across multiple datasets, they are treated as separate batches. Normally this is useful for layering columns from different datasets within the same request. For example adding a text column from one dataset to another with images or combining multiple normally-distributed synthetic datasets into a multimodal distribution. We can use the **TurnPivot** preprocessor to transpose turn columns and dataset batches.

For instance, given the following datasets:

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_0 prompt_0 | dataset_0 prompt_1 | 11              | 12              |

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_1 prompt_0 | dataset_1 prompt_1 | 21              | 22              |

without **TurnPivot** the second turn will be:

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_0"}
        ]
    },
    {"role": "assistant", "content": "32 token response to prompt_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_1"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 34,
  ...
}
```

with **TurnPivot** the second turn will be:

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_0 prompt_1"}
        ]
    },
    {"role": "assistant", "content": "23 token response to dataset_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_1 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 43,
  ...
}
```

### Usage

To use TurnPivot in the CLI, specify it as a data preprocessor:

```bash
--data "dataset0.jsonl" \
--data "dataset1.jsonl" \
--data-preprocessors "encode_media,turn_pivot"
```

> [!WARNING]\
> In the current CLI design, setting `--data-preprocessors` overrides **all** preprocessors, *except for the column mapper*, so take care to specify any preprocessor required for your use-case.

## Examples

### 1. Basic Multiturn with Synthetic Data

This example demonstrates a simple 3-turn conversation benchmark using synthetic data.

**Command:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --request-format /v1/chat/completions \
  --profile concurrent \
  --rate 6 \
  --max-requests 30 \
  --data "prompt_tokens=200,output_tokens=100,turns=3"
```

**Key Parameters:**

- `--target`: The base URL of the inference server
- `--model`: The model name to use for requests
- `--request-format`: Request format endpoint (`/v1/chat/completions`, `/v1/completions`, etc)
- `--profile`: Benchmark execution profile (concurrent maintains a fixed number of concurrent requests)
- `--rate`: 6 - Maintain 6 concurrent requests
- `--max-requests`: Maximum number of requests to send (30 requests ~= 10 conversations with 3 turns each)
- `--data`: Synthetic data configuration with 200 prompt tokens, 100 output tokens, and 3 turns

This command benchmarks 10 three-turn conversations, where each turn has 200 input tokens and generates 100 output tokens. The model maintains conversation history across all three turns.

### 2. Multiturn with System Prompts (Prefixes)

This example shows how to include system prompts in multiturn conversations, useful for many case including emulating agentic systems.

**Command:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --request-format /v1/chat/completions \
  --profile constant \
  --rate 2.0 \
  --max-requests 100 \
  --data "prompt_tokens=150,output_tokens=75,turns=4,prefix_tokens=100"
```

**Key Parameters:**

- `--profile`: constant - Send requests at a constant rate
- `--rate`: 2.0 - Send 2 requests per second
- `--max-requests`: Maximum number of requests (100 requests ~= 25 conversations with 4 turns each)
- `--data`: Added `prefix_tokens=100` to generate a system prompt of 100 tokens

In this benchmark, each conversation includes a system message (prefix) at the beginning, followed by 4 turns of user-assistant interaction. In real use-cases the system prompt establishes context or instructions that apply to the entire conversation and is often common to all users.

> [!NOTE]\
> `turns=4` + `--max-requests 100` will result in 25 **or more** conversations. Follow-up turns can only be scheduled when the previous turn completes. When a turn is complete the conversation is placed at the front of the queue for the current worker.

### 3. Advanced Prefix Distribution

This example demonstrates using multiple prefix configurations with weighted distributions, useful for testing various system prompt scenarios.

**Command:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --request-format /v1/chat/completions \
  --profile constant \
  --rate 1.5 \
  --max-seconds 60 \
  --data '{
    "prompt_tokens": 180,
    "output_tokens": 90,
    "turns": 3,
    "prefix_buckets": [
      {"bucket_weight": 60, "prefix_count": 5, "prefix_tokens": 100},
      {"bucket_weight": 40, "prefix_count": 1, "prefix_tokens": 0}
    ]
  }'
```

**Key Parameters:**

- `--profile`: constant - Send requests at a constant rate
- `--rate`: 1.5 - Send 1.5 requests per second
- `--max-seconds`: 60 - Run the benchmark for up to 60 seconds
- `--data`: JSON configuration with `prefix_buckets` defining two prefix distributions

This creates a distribution where 60% of conversations have one of 5 prefixes (100 tokens each) and 40% have a prefix of 0 tokens (effectively no prefix).

### 4. File-based Dataset with Multiturn

This example shows how to use an existing dataset file with multiturn structure.

**Example JSONL File** (multiturn_conversations.jsonl):

```json
{"prefix": "You are a helpful assistant.", "prompt_0": "What is Python?", "output_tokens_count_0": 50, "prompt_1": "How do I install it?", "output_tokens_count_1": 40}
{"prefix": "You are a coding expert.", "prompt_0": "Explain functions", "output_tokens_count_0": 60, "prompt_1": "Give me an example", "output_tokens_count_1": 45}
```

**Command:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --request-format /v1/chat/completions \
  --profile concurrent \
  --rate 10 \
  --max-requests 200 \
  --data "multiturn_conversations.jsonl"
```

**Key Parameters:**

- `--profile`: concurrent - Maintain a fixed number of concurrent requests
- `--rate`: 10 - Maintain 10 concurrent requests
- `--max-requests`: Maximum number of requests (200 requests ~= 100 conversations with 2 turns each)
- `--data`: Path to JSONL file with turn-indexed columns

### 5. Using TurnPivot with Multiple Datasets

This example demonstrates using the TurnPivot preprocessor to build a synthetic dataset where each turn follows a different distribution.

**Command:**

```bash
guidellm benchmark run \
  --target "http://localhost:8000" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --request-format /v1/chat/completions \
  --profile concurrent \
  --rate 10 \
  --max-requests 150 \
  --data "prefix_tokens=512,prompt_tokens=128,output_tokens=256" \      # Turn 1
  --data "prompt_tokens=256,prompt_token_stdev=32,output_tokens=128" \  # Turn 2
  --data "prompt_tokens=64,output_tokens=128,output_tokens_stdev=16" \  # Turn 3
  --data-preprocessors "turn_pivot"
```

**Key Parameters:**

- `--max-requests`: Maximum number of requests (150 requests ~= 50 conversations with 3 turns each)
- `--data`: Specified separately for each dataset; can also be specified once as an array
- `--data-preprocessors`: Specify `"turn_pivot"` in the preprocessor list to transpose datasets and turn columns

> [!WARNING]\
> In the current CLI design, setting `--data-preprocessors` overrides **all** preprocessors, *except for the column mapper*, so take care to specify any preprocessor required for your use-case.

## Limitations and Considerations

### Supported Request Formats

Multiturn benchmarking is currently supported for:

- `/v1/chat/completions` - Utilizing chat template formatting
- `/v1/responses` - Using the OpenAI Responses API input format
- `/v1/completions` - With basic concatenated history

Audio endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`) do not support multiturn benchmarking.

### Column Naming Requirements

Turn-indexed columns must follow the naming conventions:

- Column mapping applies to the base name. For example, `--data-column-mapper '{"column_mappings": {"text_column": "prompt"}}'`
- Turn indices can be in the form of `-0` or `_0`. Exact numbering does not matter, turns will be re-numbered to avoid holes.
- All turn columns must use the same base name. E.g. `prompt_0`, `prompt_2`, etc.

### Model Context Considerations

Multiturn conversations accumulate conversation history, which increases memory usage:

- Each turn includes the full conversation history from all previous turns
- Longer conversations (more turns) result in larger prompt sizes
- Token counts grow with each turn as history accumulates
- Consider the model's context window when configuring the number of turns and token counts

For example, `--data prefix_tokens=50,prompt_tokens=100,output_tokens=200,turns=5` will have:

- Turn 1: 150 tokens in; 200 tokens out
- Turn 2: (150 + 200) + 100 = 450 tokens in; 200 tokens out
- Turn 3: (450 + 200) + 100 = 750 tokens in; 200 tokens out
- Turn 4: (750 + 200) + 100 = 1050 tokens in; 200 tokens out
- Turn 5: (1050 + 200) + 100 = 1350 tokens in; 200 tokens out

### Additional Considerations

Multi-turn benchmarking has additional characteristics to consider when compared to single-turn:

- Expect high request-based metric variance due to the wide distribution of turn sizes
- Any error will end the entire conversation (if one turn fails the rest of the conversation is canceled)
