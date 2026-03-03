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
- **prompt_2**, **output_tokens_count_2**: Turn 2 prompt and requested output tokens

### How Multiturn Orchestration Works

When executing a multiturn benchmark, GuideLLM:

1. **Sends a turn** (prefix + prompt_0) to the model and captures the response
2. **Return the request** Store the request/response in the aggregator as a single request
3. **Builds conversation history** by combining the requests and the model's responses
4. **Sends the next turn** (prompt_i) along with the conversation history
5. **Repeat from (2)** for the `n` given turns

For `/v1/chat/completions`, the conversation history is passed as a messages array with alternating user and assistant roles. For `/v1/completions`, the history is concatenated as a single prompt string.

### Prefix Columns and System Prompts

Prefix columns (if present) are treated specially:

- In `/v1/chat/completions`, the prefix becomes a system message in the conversation array
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

For chat completions, GuideLLM creates a messages array with the conversation history:

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

#### Text Completions (`/v1/completions`)

For text completions, the conversation history is concatenated:

```text
prefix content prompt_0 content response to prompt_0 prompt_1 content response to prompt_1 prompt_2 content
```

## The TurnPivot Preprocessor

GuideLLM supports passing multiple `--data` argument, each pointing to a separate dataset. Normally this is useful for layering columns from different datasets within the same request. For example adding a text column from one dataset to another with images or combining multiple normally-distributed synthetic datasets into a multimodal distribution. In multiturn we can use the **TurnPivot** preprocessor to transpose each matched dataset with our turn columns.

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

> [!WARNING] In the current CLI design, setting `--data-preprocessors` overrides **all** preprocessors, *except for the column mapper*, so take care to specify any preprocessor required for your use-case.

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

> [!NOTE] `turns=4` + `--max-requests 100` will result in 25 **or more** conversations. Follow-up turns can only be scheduled when the previous turn completes. When a turn is complete the conversation is placed at the front of the queue for the current worker.

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

> [!WARNING] In the current CLI design, setting `--data-preprocessors` overrides **all** preprocessors, *except for the column mapper*, so take care to specify any preprocessor required for your use-case.

## Limitations and Considerations

### Supported Request Formats

Multiturn benchmarking is currently supported for:

- `/v1/chat/completions` - Utilizing chat template formatting
- `/v1/completions` - with basic concatenated history

Audio endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`) do not support multiturn benchmarking.

### Column Naming Requirements

Turn-indexed columns must follow the naming convention:

- All turn columns must use the same base name. E.g. `<prompt>_0`, `<prompt>_2`, etc.
- Turn indices can be in the form of `-0` or `_0`. Exact numbering does not matter, turns will be re-numbered to avoid holes.
- Column mapping applies to the base name. For example, `--data-column-mapper '{"text_column": "prompt"}'`

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

Multi-turn benchmarking has a couple extra characteristics to consider when compared to single-turn:

- Expect high request-based metric variance due to the wide distribution of turn sizes
- Any error will end the entire conversation (if one turn fails the rest of the conversation is canceled)
