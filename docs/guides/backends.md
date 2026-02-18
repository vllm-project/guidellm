# Backends

GuideLLM is designed to work with OpenAI-compatible HTTP servers, enabling seamless integration with a variety of generative AI backends. This compatibility ensures that users can evaluate and optimize their large language model (LLM) deployments efficiently. While the current focus is on OpenAI-compatible servers, we welcome contributions to expand support for other backends, including additional server implementations and Python interfaces.

## Supported Backends

### OpenAI-Compatible HTTP Servers

GuideLLM supports OpenAI-compatible HTTP servers, which provide a standardized API for interacting with LLMs. This includes popular implementations such as [vLLM](https://github.com/vllm-project/vllm) and [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference). These servers allow GuideLLM to perform evaluations, benchmarks, and optimizations with minimal setup.

### vLLM Python backend

GuideLLM supports running inference in the same process using the **vLLM Python backend** (`vllm_python`). This backend runs inference in the same process as GuideLLM's using vLLM's python API (AsyncLLMEngine), without an HTTP server. For setup, installation options (container, existing vLLM, pip), and examples, see [vLLM Python backend](vllm-python-backend.md).

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

To provide an API key when running benchmarks, use the `--backend-kwargs` option:

```bash
guidellm benchmark \
  --target "https://api.openai.com/v1" \
  --backend-kwargs '{"api_key": "sk-..."}' \
  --model "gpt-3.5-turbo" \
  --data "prompt_tokens=256,output_tokens=128"
```

The API key is used to set the `Authorization: Bearer {api_key}` header in HTTP requests to the backend server.

> [!IMPORTANT]\
> For security, avoid hardcoding API keys in scripts. Consider using environment variables or secure credential management tools when passing API keys via `--backend-kwargs`.

## Passing Sampling Parameters

By default, GuideLLM does not set sampling parameters such as `temperature`, `top_p`, or `top_k` in its requests to the backend server. If you need to control these parameters during benchmarking, you can pass them using the `--backend-kwargs` option with the `extras` field.

The `extras` field accepts a `body` key whose values are merged directly into the API request body sent to the backend server. This means any parameter supported by the OpenAI completions or chat completions API (or your backend's extensions) can be passed through.

### Example: Setting temperature, top_p, and top_k

```bash
guidellm benchmark \
  --target "http://localhost:8000/v1" \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --data "prompt_tokens=256,output_tokens=128" \
  --backend-kwargs '{"extras": {"body": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}}'
```

This will include `temperature`, `top_p`, and `top_k` in every request body sent to the server.

### How It Works

The `--backend-kwargs` option accepts a JSON string that is passed as keyword arguments to the backend constructor. The `extras` field within that JSON maps to a `GenerationRequestArguments` object that supports the following sub-fields:

- `body`: A dictionary of key-value pairs merged into the HTTP request body. Use this for sampling parameters like `temperature`, `top_p`, `top_k`, `repetition_penalty`, etc.
- `headers`: A dictionary of additional HTTP headers to include in requests.
- `params`: A dictionary of query parameters to append to the request URL.

### Example: Combining sampling parameters with other backend options

```bash
guidellm benchmark \
  --target "http://localhost:8000/v1" \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --data "prompt_tokens=256,output_tokens=128" \
  --backend-kwargs '{"api_key": "sk-...", "extras": {"body": {"temperature": 0.8, "top_p": 0.9}}}'
```

> [!NOTE]\
> The `--backend-args` flag is a legacy alias for `--backend-kwargs`. Both are accepted.

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](../../CONTRIBUTING.md) file.
