# Trace File Formats

Many trace files are formatted in ways that need to be specially handled to create an accurate replay. This guide covers all trace file formats currently supported by GuideLLM, along with the format-agnostic and format-specific data arguments.

Detailed use of the replay profile and file-based datasets as a whole is explained in [Trace Replay Benchmarking](../getting-started/benchmark.md#trace-replay-benchmarking).

## Supported Formats

These are passed to the `--data` argument as `kind=format`:

- `trace_synthetic`: A trace format that does the bare minimum needed to complete a fully functioning trace replay benchmark with synthetic prompt generation
- `mooncake`: The trace format used by the serving platform *Mooncake*, as defined in [https://doi.org/10.48550/arXiv.2407.00079](https://doi.org/10.48550/arXiv.2407.00079)
- `weka`: The trace format used by WEKA's *Augmented Memory Grid*, as specified [in the original research repository](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md)

## Format-Agnostic Data Arguments

All trace formats can accept the following optional data arguments:

| Argument               | Default         | Description                                           |
| ---------------------- | --------------- | ----------------------------------------------------- |
| `timestamp_column`     | "timestamp"     | Column name for timestamps in the trace file          |
| `prompt_tokens_column` | "input_length"  | Column name for prompt token counts in the trace file |
| `output_tokens_column` | "output_length" | Column name for output token counts in the trace file |

These are passed through the `--data` argument like below:

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data "kind=trace_synthetic,path=replay.jsonl,timestamp_column=ts,prompt_tokens_column=input_tokens,output_tokens_column=generated_tokens"
```

`trace_synthetic` can be thought of as the format-agnostic option, only looking for the timestamp, prompt token count and output token count columns and ignoring all other features contained in a dataset. While primarily used for testing, `trace_synthetic` may be used as a fallback for trace formats not currently supported by GuideLLM.

## Format-Specific Data Arguments

### `mooncake`

The Mooncake format expects an additional column for prefix-based cache hash IDs. During prompt generation, hash IDs sharing the same previous ID are required to represent distinct blocks of token ids.

| Argument             | Default    | Description                                         |
| -------------------- | ---------- | --------------------------------------------------- |
| `hash_ids_column`    | "hash_ids" | Column name for lists of hash IDs in the trace file |
| `hash_id_block_size` | 512        | Amount of tokens represented by one hash ID         |

### `weka`

**NOTE:** :construction: While the format is accepted, some features such as subagent conversations, tool call events and non-linear histories are still in active development. The results from datasets including these features will be unreliable.

The WEKA format expects a column with conversation UUIDs that is not wrapped within another column. The timestamp, input token length, output token length and hash IDs columns may be wrapped inside another column (ex. "requests"): if so, ensure these four virtual columns are wrapped in the same column. GuideLLM will handle the unwrapping and parsing of these trace sessions internally. If these columns are not all wrapped inside another column, ensure that none of them are wrapped inside another column.

Similar to Mooncake, WEKA uses prefix-based cache hash IDs. The original [specification](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md) for the trace requires hash IDs to be 1 or greater, and for trailing hash IDs to be dropped if there are not enough input tokens to fill the hash ID block size. To accommodate for datasets which may not follow the specification exactly (ex. [semianalysisai/cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226)), GuideLLM will accept any non-negative integer as a valid hash ID, and will drop partially filled hash IDs if they exist.

GuideLLM will generate prompts starting from the first conversation. When the conversation ends, the next conversation will be used. Hash IDs and relative timestamps are local to the conversation. After a conversation ends, the hash ID tree is reset and the relative timestamp returns to 0.0.

| Argument                 | Default    | Description                                          |
| ------------------------ | ---------- | ---------------------------------------------------- |
| `conversation_id_column` | "id"       | Column name for conversation UUIDs in the trace file |
| `hash_ids_column`        | "hash_ids" | Column name for lists of hash IDs in the trace file  |
| `hash_id_block_size`     | 64         | Amount of tokens represented by one hash ID          |
