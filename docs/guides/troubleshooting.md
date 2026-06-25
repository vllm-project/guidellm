---
weight: 15
---

# Troubleshooting

Find your symptom below, then follow the linked fix. For CLI syntax, see [Run a Benchmark](../getting-started/benchmark.md#cli-option-format).

| Symptom                                                     | Section                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Requests fail or results look wrong                         | [Debug logging](#debug-logging)                              |
| `trust_remote_code=True` when loading a tokenizer           | [Tokenizer: trust_remote_code](#tokenizer-trust_remote_code) |
| `Worker process ... died unexpectedly (signal 11)` on macOS | [macOS worker crash](#macos-worker-crash-signal-11)          |

## Debug logging

Enable verbose output to inspect request handling and worker startup:

```bash
GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG guidellm run ...
```

Run `guidellm env` to confirm settings. For all logging options (file output, log levels), see [Logging](../developer/developing.md#logging) in the development guide.

## Tokenizer: trust_remote_code

### Symptom

```text
The repository moonshotai/Kimi-K2.6 contains custom code which must be executed
to correctly load the model. You can inspect the repository content at
https://hf.co/moonshotai/Kimi-K2.6.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
```

### Fix

Pass `trust_remote_code` through `--tokenizer` `load_kwargs`:

```bash
--tokenizer '{"kind":"huggingface_auto","load_kwargs":{"trust_remote_code":true}}'
```

See [Datasets: Tokenizer](datasets.md#tokenizer) for other tokenizer options. Only use `trust_remote_code` with models you trust.

## macOS worker crash (signal 11)

### Symptom

```text
RuntimeError: Worker process group startup failed: Worker process <pid> died
unexpectedly (signal 11). Check system logs for details
```

You may also see repeated log lines such as:

```text
Worker process <pid> died unexpectedly (signal 11)
```

### Fix

GuideLLM defaults to `fork` multiprocessing, which can segfault on macOS. Use `spawn` instead:

```bash
GUIDELLM__MP_CONTEXT_TYPE=spawn guidellm run ...
```