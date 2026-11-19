# Backends

GuideLLM is designed to work with OpenAI-compatible HTTP servers, enabling seamless integration with a variety of generative AI backends. This compatibility ensures that users can evaluate and optimize their large language model (LLM) deployments efficiently. While the current focus is on OpenAI-compatible servers, we welcome contributions to expand support for other backends, including additional server implementations and Python interfaces.

## CLI Backend Configuration

Backends are configured using the `--backend` option. You can only specify one backend per command. Select a registered backend type with `kind=<TYPE>` and configure parameters with key=value pairs:

```bash
guidellm run --backend kind=<TYPE>,key=value,...
```

For HTTP servers, pass `kind=openai_http` with the target URL and other connection settings:

```bash
--backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

Flat settings can be specified using comma-separated key=value pairs; for nested settings use serialized JSON or YAML. Common `openai_http` parameters include `target`, `model`, `request_format`, `api_key`, `stream`, `verify`, `timeout`, and nested `extras` for request body, headers, and query parameters:

```bash
--backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}'
```

## Supported Backends

### OpenAI-Compatible HTTP Servers

GuideLLM supports OpenAI-compatible HTTP servers, which provide a standardized API for interacting with LLMs. This includes popular implementations such as [vLLM](https://github.com/vllm-project/vllm) and [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference). These servers allow GuideLLM to perform evaluations, benchmarks, and optimizations with minimal setup.

### vLLM Python Backend

GuideLLM supports running inference in the same process using the **vLLM Python backend** (`vllm_python_async`). This backend runs inference in the same process as GuideLLM's using vLLM's python API (AsyncLLMEngine), without an HTTP server. For setup, installation options (container, existing vLLM, pip), and examples, see [vLLM Python backend](vllm-python-backend.md).

### vLLM Python Batch Backend

The **vLLM Python batch backend** (`vllm_python_batch`) uses vLLM's synchronous `LLM` engine for batch-oriented inference. Requests are queued and dispatched in configurable batches, removing per-request scheduling overhead. This is ideal for throughput benchmarking. For setup and examples, see [vLLM Python batch backend](vllm-python-batch-backend.md).

## Examples for Spinning Up Compatible Servers

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-performance OpenAI-compatible server designed for efficient LLM inference. It supports a variety of models and provides a simple interface for deployment.

First ensure you have vLLM installed (`pip install vllm`), and then run the following command to start a vLLM server with a Llama 3.1 8B quantized model:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

### 2. Text Generation Inference (TGI)

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is another OpenAI-compatible server that supports a wide range of models, including those hosted on Hugging Face. TGI is optimized for high-throughput and low-latency inference.

To start a TGI server with a Llama 3.1 8B model using Docker, run the following command:

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

For more information on starting a TGI server, see the [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index).

### 3. llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) provides lightweight, OpenAI-compatible server through its [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server) tool.

To start a llama.cpp server with the gpt-oss-20b model, you can use the following command:

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --alias gpt-oss-20b --ctx-size 0 --jinja -ub 2048 -b 2048
```

Note that we are providing an alias `gpt-oss-20b` for the model name because GuideLLM is using it to retrieve model metadata in JSON format and such metadata is not included in GGUF model repositories. A simple workaround is to download the metadata files from the safetensors repository and place them in a local directory named after the alias:

```bash
huggingface-cli download openai/gpt-oss-20b --include "*.json" --local-dir gpt-oss-20b/
```

Now you can run `guidellm` as usual and it will be able to fetch the model metadata from the local directory.

## API Key Configuration

Some OpenAI-compatible servers require authentication via an API key. This is typically needed when:

- Connecting to OpenAI's API directly
- Using hosted or cloud-based inference services that require authentication
- Connecting to servers that have authentication enabled

Local servers like vLLM typically don't require an API key unless you've explicitly configured authentication.

### Configuring the API Key

To provide an API key when running benchmarks, pass it in the backend configuration:

```bash
guidellm run \
  --backend kind=openai_http,target=https://api.openai.com/v1,api_key=sk-...,model=gpt-3.5-turbo \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

Or with JSON:

```bash
--backend '{"kind":"openai_http","target":"https://api.openai.com/v1","api_key":"sk-...","model":"gpt-3.5-turbo"}'
```

The API key is used to set the `Authorization: Bearer {api_key}` header in HTTP requests to the backend server.

> [!IMPORTANT]\
> For security, avoid hardcoding API keys in scripts. Consider using environment variables or secure credential management tools when passing API keys via `--backend`.

## Passing Sampling Parameters

By default, GuideLLM does not set sampling parameters such as `temperature`, `top_p`, or `top_k` in its requests to the backend server. If you need to control these parameters during benchmarking, pass them through the backend `extras` field.

The `extras` field accepts a `body` key whose values are merged directly into the API request body sent to the backend server. This means any parameter supported by the OpenAI completions or chat completions API (or your backend's extensions) can be passed through.

### Example: Setting temperature, top_p, and top_k

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

This will include `temperature`, `top_p`, and `top_k` in every request body sent to the server.

## Controlling End-of-Sequence Behavior

To measure throughput and latency for an exact output length, GuideLLM asks the server to generate precisely the requested number of tokens. For the `openai_http` backend, when you request a specific output token count each request sets `ignore_eos: true` (alongside `max_tokens`/`max_completion_tokens` and `stop: null`) so the server keeps generating instead of stopping when the model emits an end-of-sequence token.

Some models should not have their end-of-sequence token suppressed. Formats such as Harmony / `gpt-oss` expect the model to stop on its own end-of-turn token; forcing generation past it makes the server reject the trailing tokens and fail the request. For these (or similar) models, disable `ignore_eos` by passing `false` through the backend `extras.body` field:

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"openai/gpt-oss-20b","extras":{"body":{"ignore_eos":false}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

Or using GuideLLM's compact `key=value` parser:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=openai/gpt-oss-20b,extras.body.ignore_eos=false \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

Values in `extras.body` are merged into the request body and take precedence over the defaults, so `ignore_eos: false` overrides the built-in `true`. Leave `ignore_eos` unset (the default) for models where suppressing the end-of-sequence token is safe, which lets GuideLLM control the exact output length.

> [!NOTE] Removing `ignore_eos: true` allows the model to stop generating at its discretion which can result in significantly shorter sequence lengths than desired.

## Structured Chat Content Payloads

Some chat templates require metadata alongside the text in each structured content object. Pass these fields through `extras.content` in the `openai_http` backend configuration. GuideLLM adds them to every generated text content object for Chat Completions and Responses API requests.

```bash
guidellm run \
  --backend '{
    "kind": "openai_http",
    "target": "http://localhost:8000",
    "model": "google/translategemma-12b-it",
    "request_format": "/v1/chat/completions",
    "extras": {
      "content": {
        "source_lang_code": "en",
        "target_lang_code": "es"
      }
    }
  }' \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --constraint kind=max_duration,seconds=60
```

### How It Works

The `--backend` config is parsed into keyword arguments for the backend constructor. The `extras` field within that config maps to a `GenerationRequestArguments` object that supports the following sub-fields:

- `body`: A dictionary of key-value pairs merged into the HTTP request body. Use this for sampling parameters like `temperature`, `top_p`, `top_k`, `repetition_penalty`, etc.
- `content`: A dictionary of fields merged into each generated text content object.
- `headers`: A dictionary of additional HTTP headers to include in requests.
- `params`: A dictionary of query parameters to append to the request URL.

### Example: Combining Sampling Parameters with Other Backend Options

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","api_key":"sk-...","extras":{"body":{"temperature":0.8,"top_p":0.9}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

## Custom Backends and Plugins

You can add custom backends without modifying GuideLLM itself by registering a `BackendArgs` schema and a `Backend` subclass and exposing them as a plugin.

> [!WARNING]\
> The plugin interfaces are experimental. The `Backend`, `BackendArgs` and other base classes that plugins subclass may change their method signatures between GuideLLM releases, so pin the GuideLLM version your plugin is tested against.

A plugin is a Python package that mirrors GuideLLM's top-level packages. When one of `guidellm.schemas`, `guidellm.backends`, `guidellm.scheduler`, `guidellm.data`, `guidellm.benchmark` or `guidellm.collectors` finishes importing, GuideLLM imports the module with the same name from each plugin, if the plugin has one. `import guidellm` alone loads no plugin code, so a plugin only adds import time to the processes that use the matching part of GuideLLM, including `spawn` and `forkserver` workers.

### 1. Define and Register the Backend

Register a `BackendArgs` subclass and a `Backend` subclass under the same `kind` name, each in the plugin module that matches where GuideLLM defines the base class:

```text
my_guidellm_plugin/
├── __init__.py          # keep it empty: it is imported to find the modules below
├── schemas/
│   ├── __init__.py      # from .backends import MyCustomBackendArgs
│   └── backends.py      # MyCustomBackendArgs, loaded with guidellm.schemas
└── backends.py          # MyCustomBackend, loaded with guidellm.backends
```

`guidellm run --help` only imports `guidellm.schemas`, so it registers the `--backend` kind without importing the backend implementation.

Put every registered schema class (`BackendArgs`, `ProfileArgs`, `ConstraintArgs`, `DataArgs` and other `*Args` subclasses) under `my_guidellm_plugin/schemas/`. Those modules load before any GuideLLM model that contains these types is built. A schema registered from another plugin module can come too late: a model such as `BenchmarkArgs` that was already built keeps its old schema and rejects the new `kind`, and the CLI never sees it.

`my_guidellm_plugin/schemas/backends.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.backends import BackendArgs


@BackendArgs.register("my_custom_backend")
class MyCustomBackendArgs(BackendArgs):
    kind: Literal["my_custom_backend"] = Field(default="my_custom_backend")
    target: str = Field(description="Base URL of the inference server.")
    model: str = Field(
        default="",
        description="Served or Hugging Face model id. Empty queries /v1/models.",
    )
```

`my_guidellm_plugin/backends.py`:

```python
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx2

from guidellm.backends import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    UsageMetrics,
)
from my_guidellm_plugin.schemas.backends import MyCustomBackendArgs


@Backend.register("my_custom_backend")
class MyCustomBackend(Backend):
    _args: MyCustomBackendArgs

    def __init__(self, arguments: MyCustomBackendArgs) -> None:
        super().__init__(arguments)
        self._client: httpx2.AsyncClient | None = None

    async def process_startup(self) -> None:
        self._client = httpx2.AsyncClient()

    async def process_shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def validate(self) -> None:
        assert self._client is not None
        response = await self._client.get(f"{self._args.target.rstrip('/')}/health")
        response.raise_for_status()

    async def default_model(self) -> str:
        if self._args.model:
            return self._args.model
        assert self._client is not None
        response = await self._client.get(f"{self._args.target.rstrip('/')}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        self._args.model = models[0]["id"] if models else ""
        return self._args.model

    async def resolve(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse | None]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        assert self._client is not None
        columns = request.columns
        parts = [*columns.get("prefix_column", []), *columns.get("text_column", [])]
        prompt = " ".join(parts) or "hello"
        body = {
            "model": await self.default_model(),
            "prompt": prompt,
            "max_tokens": request.output_metrics.text_tokens or 16,
            "stream": False,
        }
        request_info.timings.request_start = time.time()
        response = await self._client.post(
            f"{self._args.target.rstrip('/')}/v1/completions",
            json=body,
        )
        request_info.timings.request_end = time.time()
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["text"]
        usage = payload.get("usage", {})
        yield (
            GenerationResponse(
                request_id=request.request_id,
                request_args=json.dumps(body),
                text=text,
                input_metrics=UsageMetrics(text_tokens=usage.get("prompt_tokens")),
                output_metrics=UsageMetrics(text_tokens=usage.get("completion_tokens")),
            ),
            request_info,
        )
```

> [!NOTE]\
> Your `Backend` and `BackendArgs` classes must be pickleable because GuideLLM's scheduler runs them inside `spawn` or `forkserver` worker processes. Define them in the plugin package, not in a `__main__` script.

### 2. Expose the Plugin

You can load plugins via Python package entry points or the `GUIDELLM__PLUGINS` environment variable:

#### Option A: Package Entry Point (`guidellm.plugins`)

In your plugin's `pyproject.toml`, register an entry point under the `guidellm.plugins` group that names the plugin package:

```toml
[project.entry-points."guidellm.plugins"]
my_custom_backend = "my_guidellm_plugin"
```

After installing your package (`pip install -e .`), `guidellm run --help` will list `my_custom_backend` under `--backend`.

#### Option B: `GUIDELLM__PLUGINS` Environment Variable

For local development or uninstalled packages on `PYTHONPATH`, set `GUIDELLM__PLUGINS` to a comma-separated list (or JSON array) of plugin package names. A package listed here that fails to import raises an error, while a broken entry-point plugin only logs a warning.

`model` defaults to empty. `default_model` then uses the first id from `GET /v1/models`, and the completion body sends that id. Set `model=<served-or-hf-id>` to choose one. GuideLLM copies that id into the tokenizer when `--tokenizer` is omitted and loads it with `AutoTokenizer.from_pretrained`. Pass `--tokenizer` when the id is not an existing local path or a Hugging Face repo:

```bash
GUIDELLM__PLUGINS=my_guidellm_plugin \
guidellm run \
  --backend kind=my_custom_backend,target=http://localhost:8000,model=<served-or-hf-id> \
  --tokenizer kind=huggingface_auto,model=<local-path-or-hf-repo> \
  --profile kind=synchronous \
  --data kind=synthetic_text,prompt_tokens=16,output_tokens=8 \
  --constraint kind=max_requests,count=5
```

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md) file.
