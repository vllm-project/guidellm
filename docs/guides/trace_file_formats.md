# Trace File Formats

Many trace files are formatted in ways that need to be specially handled to create an accurate replay. This guide covers all trace file formats currently supported by GuideLLM, along with the format-agnostic and format-specific data arguments.

Detailed use of the replay profile and file-based datasets as a whole is explained in [Trace Replay Benchmarking](../getting-started/benchmark.md#trace-replay-benchmarking).

## Supported Formats

These are passed to the `--data` argument as `kind=format`:

- `trace_minimal`: A trace format that does the bare minimum needed to complete a fully functioning trace replay benchmark with synthetic prompt generation
- `mooncake`: The trace format used by the serving platform Mooncake, as defined in [https://doi.org/10.48550/arXiv.2407.00079](https://doi.org/10.48550/arXiv.2407.00079)

## Format-Agnostic Data Arguments

All trace formats can accept the following optional data arguments:

| Argument               | Default         | Description                                           |
| ---------------------- | --------------- | ----------------------------------------------------- |
| `timestamp_column`     | "timestamp"     | Column name for timestamps in the trace file          |
| `prompt_tokens_column` | "input_length"  | Column name for prompt token counts in the trace file |
| `output_tokens_column` | "output_length" | Column name for output token counts in the trace file |

These are passed through the `--data` argument like below:

```bash
guidellm benchmark \
    --target http://localhost:8000 \
    --profile kind=replay \
    --data "kind=trace_minimal,path=replay.jsonl,timestamp_column=ts,prompt_tokens_column=input_tokens,output_tokens_column=generated_tokens"
```

`trace_minimal` can be thought of as the format-agnostic option, only looking for the timestamp, prompt token count and output token count columns and ignoring all other features contained in a dataset. While primarily used for testing, `trace_minimal` may be used as a fallback for trace formats not currently supported by GuideLLM.

## Format-Specific Data Arguments

### `mooncake`

The Mooncake format expects an additional column for hash IDs. During prompt generation, hash IDs sharing the same previous ID are required to represent dinstinct blocks of token ids.

| Argument             | Default    | Description                                         |
| -------------------- | ---------- | --------------------------------------------------- |
| `hash_ids_column`    | "hash_ids" | Column name for lists of hash IDs in the trace file |
| `hash_id_block_size` | 512        | Amount of tokens represented by one hash ID         |
