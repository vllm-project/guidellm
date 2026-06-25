---
weight: 15
---

# Troubleshooting

Find your symptom below, then follow the linked fix. For CLI syntax, see [Run a Benchmark](../getting-started/benchmark.md#cli-option-format).

| Symptom                                                     | Section                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Requests fail or results look wrong                         | [Debug logging](#debug-logging)                              |
| Custom code error when loading a model's tokenizer.         | [Tokenizer: trust_remote_code](#tokenizer-trust_remote_code) |
| `Worker process ... died unexpectedly (signal 11)` on macOS | [macOS worker crash](#macos-worker-crash-signal-11)          |

## Debug logging

Enable debig output to inspect request handling and worker startup:

```bash
GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG guidellm run ... --disable-progress
```

Run `guidellm env` to confirm the settings are being applied. The `--disable-progress` call is optional, but the interactive progress console can overwrite console log messages. Alternatively, you can use a file log as mentioned in the [logging guide](../developer/developing.md#logging) .

For all logging options (file output, log levels), see [Logging](../developer/developing.md#logging) in the development guide.

## Tokenizer: trust_remote_code

### Symptom

You get an error that looks like:

```text
The repository moonshotai/Kimi-K2.6 contains custom code which must be executed
to correctly load the model. You can inspect the repository content at
https://hf.co/moonshotai/Kimi-K2.6.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
```

### Fix

If you fully trust the model, pass `trust_remote_code` through `--tokenizer` `load_kwargs`:

```bash
--tokenizer '{"kind":"huggingface_auto","load_kwargs":{"trust_remote_code":true}}'
```

Do not use this if you do not trust the model, as this allows code execution on your machine.

See [Datasets: Tokenizer](datasets.md#tokenizer) for other tokenizer options.

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
