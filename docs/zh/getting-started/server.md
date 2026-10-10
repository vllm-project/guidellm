---
weight: -8
---

# 启动服务器

运行 GuideLLM 基准测试前，需要先准备一个 OpenAI 兼容服务器作为测试目标。本指南将介绍如何快速配置服务器。

## **推荐选项：vLLM**

由于性能和兼容性出色，vLLM 是运行 GuideLLM 基准测试的推荐后端。

### 安装 vLLM

```bash
pip install vllm
```

### 启动 vLLM 服务器

运行以下命令，启动一个使用量化版 Llama 3.1 8B 模型的 vLLM 服务器：

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

该命令会在 `http://localhost:8000` 启动一个 OpenAI 兼容服务器。

有关更多配置选项，请参阅 [vLLM 官方文档（英文）](https://docs.vllm.ai/en/latest/)。

## **其他服务器**

GuideLLM 支持任何 OpenAI 兼容服务器，例如 TGI、SG Lang 等。有关所有受支持后端的详细信息，请参阅英文版[后端文档](../../guides/backends.md)。

## **验证服务器**

服务器启动后，你可以通过简单的 curl 命令验证其是否正常运行，以及运行基准测试的服务器能否访问它。如果服务器运行在另一台机器上，请将 `localhost` 替换为服务器的 IP 地址：

```bash
curl http://localhost:8000/v1/models
```

你应该会看到一个响应，其中列出了服务器上可用的模型。
