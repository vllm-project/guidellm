---
source_sha256: !!str "0742ac4ba47d98aef1e5436bd7aea0e2e89ecf2939e58e382dbae2680667c652"
---

# vLLM Python 后端 {#vllm-python-backend}

**vLLM Python 后端**（`vllm_python_async`）使用 vLLM 的 [AsyncLLMEngine](https://docs.vllm.ai/)，在与 GuideLLM 相同的进程中运行推理。它不需要 HTTP 服务器，因此消除了网络开销。这适用于隔离性能瓶颈或简化基准测试配置。你**无需**传入 `target`；必须在后端配置中传入 `model`，之后 vLLM 会下载并托管该模型。

有关所有引擎选项和支持的模型，请参阅 vLLM 的 [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) 和 [vLLM 文档](https://docs.vllm.ai/)。

## 安装 {#installation}

### 推荐安装方式 {#recommended-methods}

- **官方 GuideLLM + vLLM 镜像**\
  构建并运行使用 vLLM 基础镜像的镜像（例如 [Containerfile.vllm](https://github.com/vllm-project/guidellm/blob/main/Containerfile.vllm)）。该镜像基于 `vllm/vllm-openai`，并在其上安装 GuideLLM，从而提供经过验证的 vLLM + GuideLLM 软件栈以及基础镜像提供的硬件支持。

  **注意：** 这种方式会优先采用 vLLM 的依赖要求，而非 GuideLLM 的要求。由于 vLLM 是更复杂的项目，因此推荐采用这种配置；但这可能导致使用较旧的 Python 或依赖版本，从而在某些场景下影响 GuideLLM 的性能和行为。

- **已有 vLLM 安装**\
  先根据你的环境（GPU/CPU、CUDA 等）安装 vLLM，再在相同环境中安装 GuideLLM（例如运行 `pip install guidellm` 或安装 extras）。这样可以避免重复安装 vLLM，并复用现有的加速配置。

  **注意：** 为 vLLM Python 后端从 lockfile 安装时，可能无法安装适用于硬件加速的正确依赖项。

## 基本示例 {#basic-example}

使用 vLLM Python 后端运行基准测试：

```bash
guidellm run \
  --backend kind=vllm_python_async,model=Qwen/Qwen3-0.6B \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=constant,rate=3 \
  --constraint kind=max_duration,seconds=20
```

除非通过后端配置中的 `vllm_config` 覆盖，否则引擎行为（设备、内存等）遵循 vLLM 默认设置。在没有 GPU 的情况下运行时（例如 GuideLLM + vLLM 容器无法访问 GPU），后端会自动使用 CPU 设备；除非你在 `vllm_config` 中设置 `device`。有关引擎配置选项，请参阅 vLLM 的 [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)。

## 请求格式与后端选项 {#request-format-and-backend-options}

- **`request_format`**\
  控制如何构建聊天提示词。可选值：`plain`（不使用 chat template；将消息内容拼接为纯文本）、`default-template`（使用 tokenizer 的默认 chat template），或 vLLM 支持的文件路径/单行模板字符串。该值会传递给 vLLM 的 chat template 处理逻辑。详情请参阅 vLLM 的 [Chat templates](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) 文档。

- **`vllm_config`**\
  特定于后端的引擎选项以嵌套 dict 的形式传入后端配置。传入一个 `vllm_config` 键，其值为由引擎选项名称和值组成的 dict。

  **在 `vllm_config` 中使用 Engine Arguments：** [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) 文档以 **CLI 形式**介绍选项（例如 `--gpu-memory-utilization`、`--max-model-len`）。在 `vllm_config` 中必须改用 **Python 参数名称**：去掉开头的 `--`，并将连字符替换为下划线（例如 `gpu_memory_utilization`、`max_model_len`）。键名与 vLLM 的 `EngineArgs` 和 `AsyncEngineArgs` dataclass 字段名相同。允许使用的键及其类型的完整列表，请参阅 [vLLM 源码：`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)（搜索 `class EngineArgs`）。

  示例——限制 GPU 内存用量和上下文长度：

  ```bash
  --backend '{"kind":"vllm_python_async","model":"Qwen/Qwen3-0.6B","vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

  有关所有选项及其类型，请参阅 vLLM 的 [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)（CLI 形式）和 [EngineArgs 源码](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)（`vllm_config` 使用的 Python 字段名）。

> [!IMPORTANT]
>
> `vllm_python_async` 的后端配置必须包含 `model` 字段。如果 `vllm_config` 中也设置了 `model`，则优先使用顶层的 `model` 字段。

## 另请参阅 {#see-also}

- [后端](backends.md) — 支持的后端概览。
- [运行基准测试](../getting-started/benchmark.md) — 基准测试通用选项。
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) — 面向 CLI 的文档；在 `vllm_config` 中请使用 Python 名称（例如 `gpu_memory_utilization`）。
- [vLLM 源码：`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) — `vllm_config` 键对应的 `EngineArgs` / `AsyncEngineArgs` 字段名和类型。
- [vLLM AsyncEngineArgs API](https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.AsyncEngineArgs) — 接收这些选项的类的 API 参考。
- [vLLM Chat templates](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/chat_templates/) — `request_format` 行为说明。
- [vLLM 文档](https://docs.vllm.ai/)
- [vLLM 安装](https://docs.vllm.ai/en/latest/getting_started/installation)
- [vLLM OpenAI 兼容服务器](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — 使用 HTTP 服务器而非 Python 后端时请参阅。
