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

## Recording vLLM Server Configuration

To include the server's configuration alongside benchmark results, select the desired `capture_server_config` sections on the HTTP backend:

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","capture_server_config":["vllm_config"]}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_requests,count=10 \
  --output kind=json,path=benchmark.json
```

During setup, GuideLLM requests `/server_info?config_format=json` once and saves the selected sections under `benchmarks[].config.backend.server_info` in the JSON report. This includes settings such as tensor parallelism, scheduler limits and cache configuration that help explain differences between benchmark runs. The snapshot is reused across worker processes and benchmark strategies; collection is outside the measured generation requests. Captured server details are retained in the report and omitted from the initialization console output. It also works with `validate_backend=false`.

The selector accepts a set of section names (a JSON array in CLI input), `"all"`, or `null`:

| Section       | Contents                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------- |
| `vllm_config` | Model, scheduler, parallelism and cache configuration                                         |
| `vllm_env`    | vLLM environment settings reported by the server                                              |
| `system_env`  | System diagnostics such as library, CUDA and operating-system versions reported by the server |

For example, use `"capture_server_config":["vllm_env","system_env"]` in the backend JSON to capture only environment information. Use `capture_server_config=all` with the key/value CLI syntax to capture all three supported sections. `all` does not include unknown response fields. These names select fields from the server response, not environment variables on the GuideLLM client. The exact contents depend on the server version; `system_env` is not necessarily a dump of process environment variables.

Capture is disabled by default (`null`); an empty set/array also disables it. Booleans and unknown section names are rejected. The optional request uses the configured API key and `extras.headers`, has a five-second total deadline and a 1 MiB response limit, and does not follow redirects. A missing or denied endpoint, network failure, invalid response or unsupported format produces a warning and leaves the benchmark running without server metadata. Custom routes can be supplied through `api_routes`, for example `{"/server_info": "proxy/server_info"}`.

Each selected section must be a JSON object. Missing or unsupported sections produce a warning and are omitted independently, preserving other valid selected sections. Empty objects are retained. Legacy text configurations are skipped because their credential fields cannot be reliably redacted; structured environment sections can still be captured when available. Depending on the vLLM version, `/server_info` may require `VLLM_SERVER_DEV_MODE=1`. This enables development endpoints beyond server information; consult [vLLM's security documentation](https://docs.vllm.ai/en/latest/usage/security/) before enabling it, and use an isolated benchmark deployment.

GuideLLM applies best-effort filtering to every selected section:

- Recognizable credential fields, including API keys, passwords, private keys, tokens, cookies and credential aliases, are redacted recursively. Additional exact names and suffixes include `ssh_key`, `encryption_key`, `signing_key`, `license_key`, `client_key`, `passphrase` and `pwd`. Related settings such as `signing_key_algorithm` and `client_key_file` remain intact. Unset values and boolean switches are preserved.
- In environment records such as `{"name": "VLLM_API_KEY", "value": "..."}`, a recognizable sensitive name causes the `value` to be redacted. Records for ordinary settings, such as `CUDA_VISIBLE_DEVICES`, are retained.
- User information in recognized scheme-based URLs (such as `postgres://user:password@host/db`) is replaced with `[REDACTED]`, retaining the host and path. Strings with ambiguous URL user information are omitted in full: for example, `redis://user:pa/ss@cache:6379/0`. This conservative rule can also omit a URL with a port or IPv6 authority followed by `@` in its path, query or fragment.
- A sensitive flag in an argument list (such as `--api-key`) has its following value redacted. Free-form strings containing sensitive assignments, flags or `Authorization:`/`Proxy-Authorization:` headers are omitted in full rather than attempting to parse shell quoting or multiline credentials. This includes recognizable signature assignments such as `sig=` and `X-Amz-Signature=` in URLs. For example, an `env_vars` string containing `VLLM_API_KEY=...` is redacted as a whole.
- Fields named `host`, `hostname`, `node_ip` or `master_addr`, or ending in `_host` (case-insensitive, treating hyphens as underscores), are redacted unless their value is exactly `localhost`, `127.0.0.1`, `::1`, an empty string or `null`.

This does not guarantee that configuration or environment information is safe to publish. Hardware details, model paths, deployment names, URL hosts/paths and unrecognized secret formats may remain. Choose only the sections needed, leave capture disabled for confidential environments, and review the saved report before sharing it. `all` does not bypass filtering.

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

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md) file.
