# vLLM Python Backend

The **vLLM Python backend** (`vllm_python`) runs inference in the **same process** as GuideLLM using vLLM's [AsyncLLMEngine](https://docs.vllm.ai/). No HTTP server is involved, reducing overheat and variables. This is useful for isolating performance bottlenecks or simplifying your benchmark setup. You do **not** pass `--target`; you **must** pass `--model`.

For all engine options and supported models, see vLLM's [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) and the [vLLM documentation](https://docs.vllm.ai/).

## Installation

### Recommended methods

- **Official GuideLLM + vLLM image**  
  Build and run the image that uses the vLLM base image (e.g. [Containerfile.vllm](https://github.com/vllm-project/guidellm/blob/main/Containerfile.vllm)). It is based on `vllm/vllm-openai` and installs GuideLLM on top, giving a known-good vLLM + GuideLLM stack with hardware support as provided by the base image.
  
  **Note:** This method will result in the preference for vllm's requirements as opposed to GuideLLM's requirements. Since vLLM is the more complex project, this is the recommended configuration, but this may result in an older Python or dependency version, resulting in sub-optimal GuideLLM performance and behavior in some scenarios.

- **Existing vLLM installation**  
  Install vLLM first for your environment (GPU/CPU, CUDA, etc.), then install GuideLLM in the same environment (e.g. `pip install guidellm` or with extras). You avoid a duplicate vLLM install and reuse your existing acceleration setup.
  
  **Note:** Using [uv](https://github.com/astral-sh/uv) is not recommended for the vLLM Python backend because of potentially incompatible requirements between the two projects. Prefer pip or the container / existing vLLM environment.


It is also possible to install GuideLLM and vLLM via pip using `pip install guidellm[vllm]`. This method may make **hardware acceleration** (e.g. CUDA) harder to get working. See [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation) and GPU/hardware-specific docs there. For production or GPU use, the container or existing-install path is recommended.


## Basic example

Run a benchmark with the vLLM Python backend (no `--target`):

```bash
guidellm benchmark run \
  --backend vllm_python \
  --model "Qwen/Qwen3-0.6B" \
  --data "prompt_tokens=256,output_tokens=128" \
  --max-seconds 20 \
  --rate 3
```

Engine behavior (device, memory, etc.) follows vLLM defaults unless you override it via `--backend-kwargs` (e.g. `vllm_config`). When running without a GPU (e.g. the GuideLLM + vLLM container without GPU access), the backend automatically uses the CPU device unless you set `device` in `vllm_config`. For engine configuration options, see vLLM's [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/).

## Request format and backend options

- **`--request-format`**  
  Controls how chat prompts are built. Options: `plain` (no chat template; message content is concatenated as plain text), `default-template` (use the tokenizer’s default chat template), or a file path / single-line template string per vLLM’s supported options. The value is passed through to vLLM's chat template handling. For details, see vLLM's [Chat templates](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) documentation.

- **`--backend-kwargs`**  
  Backend-specific options are passed here as a JSON object: pass a `vllm_config` key whose value is a dict of engine option names and values. You can also pass `request_format` here as an alternative to `--request-format`.

  **Using Engine Arguments in `vllm_config`:** The [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) documentation describes options in **CLI form** (e.g. `--gpu-memory-utilization`, `--max-model-len`). For `vllm_config` you must use the **Python parameter names** instead: strip the leading `--` and replace dashes with underscores (e.g. `gpu_memory_utilization`, `max_model_len`). The keys are the same as the field names on vLLM's `EngineArgs` and `AsyncEngineArgs` dataclasses; for the exact list of allowed keys and types, see the [vLLM source: `vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) (search for `class EngineArgs`).

  Example — limit GPU memory use and context length:

  ```bash
  --backend-kwargs '{"vllm_config": {"gpu_memory_utilization": 0.8, "max_model_len": 4096}}'
  ```

  For the full list of options and their types, see vLLM's [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) (CLI form) and the [EngineArgs source](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) (Python field names for `vllm_config`).

## See also

- [Backends](backends.md) — Overview of supported backends.
- [Run a benchmark](../getting-started/benchmark.md) — General benchmark options.
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) — CLI-oriented docs; use Python names (e.g. `gpu_memory_utilization`) in `vllm_config`.
- [vLLM source: `vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) — `EngineArgs` / `AsyncEngineArgs` field names and types for `vllm_config` keys.
- [vLLM AsyncEngineArgs API](https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.AsyncEngineArgs) — API reference for the class that receives these options.
- [vLLM Chat templates](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) — For `--request-format` behavior.
- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — When using the HTTP server instead of the Python backend.
