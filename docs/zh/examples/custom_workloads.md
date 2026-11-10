---
source_sha256: !!str "07427f1cdf4c08efbab62fa102e570707b170851135c5f07c68aee9cd66103f3"
---

# 对不同工作负载形状进行基准测试 {#benchmarking-different-workload-shapes}

并非所有 LLM 工作负载都一样。聊天机器人、摘要流水线和代码生成器会以完全不同的方式给服务器带来压力。本指南介绍如何针对特定工作负载类型配置 GuideLLM，并解释指标为何会有不同表现。

## 前提条件

- 正在运行的兼容 OpenAI 服务器（[服务器设置指南](../getting-started/server.md)）
- 已安装 GuideLLM（[安装指南](../getting-started/install.md)）

## 工作负载形状为何重要

LLM 请求分为两个阶段：

- **预填充（Prefill）**：服务器一次处理所有输入 token。这一阶段决定首 token 时间（TTFT）。
- **解码（Decode）**：服务器逐个生成输出 token。这一阶段决定 token 间延迟（ITL）和整体吞吐量。

输入 token 与输出 token 的比例会改变占主导地位的阶段。提示词较长、回复较短的工作负载受预填充阶段限制；提示词较短、回复较长的工作负载受解码阶段限制。两种工作负载的瓶颈不同，需要关注的指标也不同。

## 四种常见工作负载形状

**注意：** _当前工作负载偏向重预填充（输入 token 多），这是因为存在多文件、更长的系统提示词、多轮对话等情况，因此这些工作负载可能无法反映真实使用场景。如果你有自定义工作负载，请针对具体用例配置相应形状。_

### 1. 聊天 / 对话

用户发送一条简短消息，并期待得到中等长度的回复 **（受解码阶段限制）**。聊天机器人、客户支持和问答界面通常属于这种情况。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=512 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=chat_workload.json
```

**关注指标：** TTFT 和 ITL 都很重要。用户既要等待响应开始流式传输（TTFT），也会在内容逐步到达时阅读（ITL）。其中任何一项过慢，都会让体验显得迟滞。

### 2. 摘要 / RAG

检索增强生成流水线会将大量上下文填入提示词，并期望得到简洁的答案。输入较长，输出较短。**（受预填充阶段限制）**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=2048,output_tokens=128 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=summarization_workload.json
```

**关注指标：** TTFT 是主要指标。服务器会将大部分时间用于处理较长的提示词，然后才生成简短答案。输出 token 很少，因此 ITL 的影响较小。由于瓶颈在预填充阶段，吞吐量（token/秒）会低于聊天场景。

### 3. 代码生成

开发者给出简短指令，并期待得到一大段生成的代码。输入较短，输出较长。**（受预填充阶段限制）**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=1024 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=codegen_workload.json
```

**关注指标：** ITL 是主要指标。提示词较短，预填充速度快，因此 TTFT 较低；但服务器会将大部分时间用于解码较长的输出。ITL 较高时，代码会以很慢的速度逐步输出。吞吐量（token/秒）是这里的关键指标。

### 4. 均衡 / 通用

输入和输出长度相同 **（均衡型）**。如果不确定工作负载的具体形态，或需要通用的性能基线，这是一个很好的起点。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=120 \
  --seed kind=static,value=42 \
  --output kind=json,path=balanced_workload.json
```

**关注指标：** 两个阶段的影响大致相同。这种形状可以较均衡地展现服务器能力，但可能无法反映实际生产工作负载。

## 比较不同工作负载形状的结果

在同一台服务器上运行这四种形状，并并排比较结果。

**建议：**_请将这些形状调整为你的自定义工作负载。_

| 工作负载 | Token（输入/输出） | 吞吐量（tok/s） | TTFT p50（毫秒） | ITL p50（毫秒） | 瓶颈           |
| -------- | ------------------ | --------------- | ---------------- | --------------- | -------------- |
| 聊天     | 256 / 512          | 1850            | 35               | 12              | 混合           |
| 摘要     | 2048 / 128         | 920             | 180              | 9               | 预填充（TTFT） |
| 代码生成 | 128 / 1024         | 2100            | 18               | 14              | 解码（ITL）    |
| 均衡     | 1000 / 1000        | 1350            | 95               | 13              | 混合           |

主要观察结果：

- **摘要的 TTFT 最高**，因为服务器要先处理 2048 个输入 token，才会生成第一个输出 token **（受预填充阶段限制）**。但由于需要解码的 token 很少，ITL 是最低的。
- **代码生成的吞吐量最高**，因为较短的提示词可以快速完成预填充，服务器大部分时间都处于效率较高的解码阶段。不过，由于每个请求需要生成 1024 个输出 token，其端到端耗时最长。
- **聊天场景最为均衡**：两个阶段都不占主导，因此 TTFT 和 ITL 都需要满足 SLO。
- **不同形状的吞吐量不能直接比较**，因为每个请求的总 token 数不同。若要在同一口径下比较解码吞吐量，请使用 `output_tokens_per_second`。

## 选择基准测试配置

如果你了解自己的工作负载，可以参考以下配置：

| 应用场景             | 推荐配置                                | 主要指标            |
| -------------------- | --------------------------------------- | ------------------- |
| 聊天机器人、问答     | `prompt_tokens=256,output_tokens=512`   | TTFT p99 和 ITL p50 |
| RAG、摘要、搜索      | `prompt_tokens=2048,output_tokens=128`  | TTFT p99            |
| 代码生成、写作、翻译 | `prompt_tokens=128,output_tokens=1024`  | ITL p50 和吞吐量    |
| 未知或混合工作负载   | `prompt_tokens=1000,output_tokens=1000` | 所有指标            |

如果你有真实生产数据，请使用真实数据代替合成数据。GuideLLM 支持通过 JSONL 文件或 HuggingFace 数据集加载自定义数据集。详情请参阅[数据集指南](../guides/datasets.md)。

## 针对工作负载形状调整服务器

基准测试结果可以帮助你调整服务器配置：

**受预填充阶段限制（TTFT 较高）：**

- 启用分块预填充（`--enable-chunked-prefill`），使预填充与解码阶段重叠
- 增加张量并行度，将预填充计算分摊到多张 GPU 上
- 如果提示词不需要完整上下文窗口，可考虑缩短 `--max-model-len`

**受解码阶段限制（ITL 较高）：**

- 检查是否受内存带宽限制；较大的批次有助于分摊内存读取开销
- 尝试量化模型（FP8、W4A16），以减轻内存带宽压力
- 对延迟敏感的工作负载，可考虑推测解码

## 后续步骤

- 使用[找到服务器的最佳并发上限](optimal_concurrency.md)为所选工作负载形状优化负载水平
