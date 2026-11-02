---
weight: -3
source_sha256: !!str "deaedda951b7d2072069ddf8a4503af48d3b57751728f840d334a7c9ff91e971"
---

# 后端

GuideLLM 可与兼容 OpenAI 的 HTTP 服务器配合使用，方便接入各种生成式 AI 后端。这种兼容性确保用户能够高效评估和优化大语言模型（LLM）部署。目前，GuideLLM 主要支持兼容 OpenAI 的服务器；我们也欢迎社区贡献，扩展对其他后端的支持，包括更多服务器实现和 Python 接口。

## CLI 后端配置

使用 `--backend` 选项配置后端。每条命令只能指定一个后端。通过 `kind=<TYPE>` 选择已注册的后端类型，并使用 key=value 对配置参数：

```bash
guidellm run --backend kind=<TYPE>,key=value,...
```

对于 HTTP 服务器，使用 `kind=openai_http`，并指定目标 URL 和其他连接设置：

```bash
--backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

简单设置可使用逗号分隔的 key=value 对；嵌套设置则使用序列化的 JSON 或 YAML。常见的 `openai_http` 参数包括 `target`、`model`、`request_format`、`api_key`、`stream`、`verify`、`timeout`，以及用于设置请求体、请求头和查询参数的嵌套字段 `extras`：

```bash
--backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}'
```

## 支持的后端

### 兼容 OpenAI 的 HTTP 服务器

GuideLLM 支持提供标准化 LLM 交互 API 的兼容 OpenAI 的 HTTP 服务器，其中包括 [vLLM](https://github.com/vllm-project/vllm) 和 [Text Generation Inference（TGI）](https://github.com/huggingface/text-generation-inference) 等常见实现。借助这些服务器，只需少量设置，就可以使用 GuideLLM 进行评估、基准测试和优化。

### vLLM Python 后端

GuideLLM 支持通过 **vLLM Python 后端**（`vllm_python_async`）在同一进程中运行推理。该后端使用 vLLM 的 Python API（`AsyncLLMEngine`），与 GuideLLM 在同一进程中执行推理，无需 HTTP 服务器。有关设置方法、安装选项（容器、现有 vLLM 环境或 pip）和示例，请参阅英文版 [vLLM Python 后端指南](../../guides/vllm-python-backend.md)。

### vLLM Python 批处理后端

**vLLM Python 批处理后端**（`vllm_python_batch`）使用 vLLM 的同步 `LLM` 引擎执行面向批处理的推理。请求会进入队列，并按可配置的批次分发，从而减少逐个请求调度的开销。这适合吞吐量基准测试。有关设置方法和示例，请参阅英文版 [vLLM Python 批处理后端指南](../../guides/vllm-python-batch-backend.md)。

## 启动兼容服务器的示例

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm) 是一种高性能、兼容 OpenAI 的服务器，专为高效的大语言模型推理而设计。它支持多种模型，并提供简便的部署接口。

先确保已安装 vLLM（`pip install vllm`），然后运行以下命令，使用量化的 Llama 3.1 8B 模型启动 vLLM 服务器：

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

有关启动 vLLM 服务器的更多信息，请参阅 [vLLM 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)。

### 2. Text Generation Inference（TGI）

[Text Generation Inference（TGI）](https://github.com/huggingface/text-generation-inference) 是另一种兼容 OpenAI 的服务器，支持包括 Hugging Face 托管模型在内的多种模型。TGI 针对高吞吐量、低延迟推理进行了优化。

要使用 Docker 启动运行 Llama 3.1 8B 模型的 TGI 服务器，请执行以下命令：

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

有关启动 TGI 服务器的更多信息，请参阅 [TGI 文档](https://huggingface.co/docs/text-generation-inference/index)。

### 3. llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) 通过其 [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server) 工具提供轻量级、兼容 OpenAI 的服务器。

可以使用以下命令启动运行 gpt-oss-20b 模型的 llama.cpp 服务器：

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --alias gpt-oss-20b --ctx-size 0 --jinja -ub 2048 -b 2048
```

这里将模型别名设为 `gpt-oss-20b`，因为 GuideLLM 使用该名称读取 JSON 格式的模型元数据，而 GGUF 模型仓库中不包含这些元数据。一个简单的解决方法是从 safetensors 仓库下载元数据文件，并将其放在以该别名命名的本地目录中：

```bash
huggingface-cli download openai/gpt-oss-20b --include "*.json" --local-dir gpt-oss-20b/
```

之后便可以照常运行 `guidellm`，它将能够从本地目录读取模型元数据。

## API Key 配置

一些兼容 OpenAI 的服务器要求使用 API key 进行身份验证，常见情况包括：

- 直接连接 OpenAI API。
- 使用要求身份验证的托管或云端推理服务。
- 连接已启用身份验证的服务器。

除非显式配置了身份验证，否则 vLLM 等本地服务器通常不需要 API key。

### 配置 API Key {#api-key-configuration}

运行基准测试时，可通过后端配置提供 API key：

```bash
guidellm run \
  --backend kind=openai_http,target=https://api.openai.com/v1,api_key=sk-...,model=gpt-3.5-turbo \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

也可以使用 JSON：

```bash
--backend '{"kind":"openai_http","target":"https://api.openai.com/v1","api_key":"sk-...","model":"gpt-3.5-turbo"}'
```

API key 用于设置发送给后端服务器的 HTTP 请求头 `Authorization: Bearer {api_key}`。

> [!IMPORTANT]\
> 为保障安全，请避免在脚本中写死 API key。通过 `--backend` 传递 API key 时，可考虑使用环境变量或安全的凭据管理工具。

## 传递采样参数

默认情况下，GuideLLM 向后端服务器发送请求时不会设置 `temperature`、`top_p` 或 `top_k` 等采样参数。如果需要在基准测试中控制这些参数，可通过后端配置中的 `extras` 字段传递。

`extras` 字段接受一个 `body` 键，其中的值会直接合并到发送给后端服务器的 API 请求体。这意味着，你可以通过它传递 OpenAI 补全或聊天补全 API（或后端扩展）支持的任何参数。

### 示例：设置 temperature、top_p 和 top_k

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","extras":{"body":{"temperature":0.6,"top_p":0.95,"top_k":20}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

这样，发送给服务器的每个请求体中都会包含 `temperature`、`top_p` 和 `top_k`。

## 控制序列结束行为

为测量指定输出长度下的吞吐量和延迟，GuideLLM 会要求服务器准确生成所需数量的 token。对于 `openai_http` 后端，指定输出 token 数量时，每个请求都会设置 `ignore_eos: true`（同时设置 `max_tokens`/`max_completion_tokens` 和 `stop: null`），使服务器在模型输出序列结束 token 后继续生成，而不立即停止。

某些模型不适合忽略序列结束 token。Harmony / `gpt-oss` 等格式要求模型在输出回合结束 token 时自行停止；如果强制它继续生成，服务器可能会拒绝后续 token，导致请求失败。对于这类模型，可通过后端的 `extras.body` 字段传入 `false` 来禁用 `ignore_eos`：

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"openai/gpt-oss-20b","extras":{"body":{"ignore_eos":false}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

也可以使用 GuideLLM 的简写 `key=value` 解析格式：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=openai/gpt-oss-20b,extras.body.ignore_eos=false \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

`extras.body` 中的值会合并到请求体，并覆盖默认值，因此 `ignore_eos: false` 会覆盖内置的 `true`。对于可以安全忽略序列结束 token 的模型，保持 `ignore_eos` 未设置（默认值），让 GuideLLM 控制准确的输出长度。

> [!NOTE] 移除 `ignore_eos: true` 后，模型可以自行决定何时停止生成，因此实际序列长度可能明显短于预期。

## 结构化聊天内容载荷

某些聊天模板要求每个结构化内容对象在文本之外还包含元数据。可通过 `openai_http` 后端配置中的 `extras.content` 传递这些字段。GuideLLM 会将其添加到 Chat Completions 和 Responses API 请求中每个生成的文本内容对象。

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

### 工作原理

`--backend` 配置会被解析为传给后端构造函数的关键字参数。配置中的 `extras` 字段映射到 `GenerationRequestArguments` 对象，该对象支持以下子字段：

- `body`：合并到 HTTP 请求体的键值对字典，可用于设置 `temperature`、`top_p`、`top_k`、`repetition_penalty` 等采样参数。
- `content`：合并到每个生成的文本内容对象中的字段字典。
- `headers`：随请求发送的额外 HTTP 请求头字典。
- `params`：附加到请求 URL 的查询参数字典。

### 示例：结合采样参数与其他后端选项

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","api_key":"sk-...","extras":{"body":{"temperature":0.8,"top_p":0.9}}}' \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

## 扩展后端支持

GuideLLM 是一个开放的平台，欢迎社区贡献，扩展其后端支持。无论是新增服务器实现、集成基于 Python 的后端，还是增强现有功能，都欢迎参与。有关贡献方式的详细信息，请参阅 [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md)。
