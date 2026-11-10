---
weight: 1
source_sha256: !!str "fc93b1a254019f1a14bc25d4374530d3d5ace1127a68a7ae3d4a166bf8d36ab3"
---

# 多轮对话基准测试

本指南演示如何使用 GuideLLM 编排多轮基准测试，以多请求/响应模式模拟用户对话。

## 环境准备

首先，确保已有正在运行的推理服务器和兼容的模型。GuideLLM 支持任何能够处理对话交互的 OpenAI 兼容服务器。以下基准测试示例将使用 vLLM 提供对话模型服务。

```bash
# Example: vLLM with a conversational model
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

接下来，在同一实例或另一台能够访问服务器的机器上安装 GuideLLM：

```bash
pip install guidellm[recommended]
```

## 了解多轮数据结构

GuideLLM 的多轮基准测试使用带索引的列表示对话轮次。对话中的每一轮由一组列表示，列名带有表示轮次索引的数字后缀。轮次索引可以是任意数值，但为简单起见，建议从 0 开始递增。

### 按轮次索引的列格式

对于一个包含 3 轮的对话，数据集可以包含以下列：

- **prefix**：可选的系统提示词，会成为系统消息（仅应用于第一轮）
- **prompt_0**、**output_tokens_count_0**：第 1 轮提示词及请求的输出 token 数
- **prompt_1**、**output_tokens_count_1**：第 2 轮提示词及请求的输出 token 数
- **prompt_2**、**output_tokens_count_2**：第 3 轮提示词及请求的输出 token 数

对于文件数据集和 HuggingFace 数据集，带索引的列仍是数据编写格式。映射后，带索引的行和图结构负载会共用同一条最终处理流程：先规范化为 `ConversationGraphData`，再将客户端工具调用轮次展开为工具调用节点和注入节点，最后构建运行时对话图。

### 合成数据始终生成图结构

合成多轮数据（包括单轮数据和带分支的子代理配置）会生成 `conversation_turns` JSON 列，而不是 `prompt_0` / `prompt_1` 这类带索引的列。同一条最终处理流程会处理该图结构负载。

### 多轮编排的工作方式

执行多轮基准测试时，GuideLLM 会：

1. **发送一轮**：将 prefix 和 prompt_0 发送给模型并获取响应
2. **返回请求**：在聚合器中将请求/响应作为单个请求存储
3. **构建对话历史**：组合各个请求和模型响应
4. **发送下一轮**：连同对话历史一起发送 prompt_i
5. 对给定的 `n` 轮从第 (2) 步开始重复

对于 `/v1/chat/completions`，对话历史会作为 user 和 assistant 角色交替排列的 messages 数组传递。对于 `/v1/responses`，历史会以 user 和 assistant 角色交替排列的形式传递，或通过前一个请求 ID 传递。对于 `/v1/completions`，历史会拼接成一个提示词字符串。

更多信息请参阅[请求格式](#request-formatting)和[服务端对话历史](#server-side-conversation-history-v1responses-only)。

### Prefix 列和系统提示词

如果存在 prefix 列，其处理方式如下：

- 在 `/v1/chat/completions` 中，prefix 会成为对话数组中的系统消息
- 在 `/v1/responses` 中，prefix 会成为 `instructions` 字段
- 在 `/v1/completions` 中，prefix 会添加到该轮提示词之前
- 如有需要，可以为 prefix 指定轮次索引；不过建议只为第一轮设置一个 prefix
- 合成数据仅支持在第一轮设置 prefix

## 处理选项

所有标准基准测试参数都适用于多轮任务，例如 `--profile`、profile 速率参数以及 `--constraint kind=max_requests,count=<n>`。所有以“请求”为单位的选项都会将每一轮视为单独的请求（例如，包含 3 轮的数据集行会计为 3 个请求）。

### 合成数据配置

GuideLLM 可以使用合成数据配置中的 `turns` 参数自动生成多轮合成数据。

#### 基本的多轮合成数据

要生成多轮合成数据，请在 `--data` 参数中指定 `turns`：

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=3
```

这会创建一个 3 轮对话，其中每轮都会根据已配置的 `prompt_tokens` / `output_tokens` 分布独立采样（此处未设置标准差，因此分别固定为 256 和 128）。

#### 第一条消息的大小

默认情况下，每一轮（包括第一轮）都使用相同的 `prompt_tokens` / `output_tokens` 分布。如果开场消息需要采用不同的大小，可以设置可选参数 `first_prompt_tokens` / `first_output_tokens`（以及常用的 `_stdev` / `_min` / `_max` 参数）。例如，可模拟以较大的文件或文章开场供模型审阅，之后再提出较短的问题：

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=4,first_prompt_tokens=4096
```

这样，后续轮次的提示词会保持在约 256 个 token，而第一条用户消息的平均长度为 4096 个 token。子代理分支的第一轮会继承父级的 `first_*` 设置，除非分支另行覆盖（参见[子代理分支](#sub-agent-branches)）。

#### 带 Prefix 的合成数据

可以通过以下两种方式为合成对话添加系统提示词（prefix）：

**简单的 Prefix 配置：**

```bash
--data kind=synthetic_text,prompt_tokens=256,output_tokens=128,turns=3,prefix_count=5,prefix_tokens=50
```

这会生成 5 个各含 50 个 token 的唯一 prefix。每个对话都会随机选择其中一个作为系统消息。

**高级 Prefix 配置：**

对于更复杂的场景，可以使用 `prefix_buckets` 为不同的 prefix 配置创建加权分布。这需要传入 JSON 配置：

```bash
--data '{
  "kind": "synthetic_text",
  "prompt_tokens": 256,
  "output_tokens": 128,
  "turns": 3,
  "prefix_buckets": [
    {
      "bucket_weight": 60,
      "prefix_count": 10,
      "prefix_tokens": 100
    },
    {
      "bucket_weight": 40,
      "prefix_count": 1,
      "prefix_tokens": 50
    }
  ]
}'
```

对于此配置：

- 60% 的对话会使用 10 个各含 100 个 token 的 prefix 之一
- 40% 的对话会使用含 50 个 token 的 prefix

#### 子代理分支 {#sub-agent-branches}

GuideLLM 支持模拟多代理工作负载：编排器会在对话过程中启动并行子代理。使用 `branches` 参数指定子代理分支，设置其从主对话的哪一轮分出，以及之后在哪一轮合并回来。

每个分支都会：

- 在 `at_turn` 处**启动**，并使用全新上下文（子代理看不到主对话历史）
- 独立运行 `turns` 轮
- 在 `at_turn + merge_after` 处**合并**回来（默认 `merge_after=1`），此时主对话会收到子代理的最终输出

支持在同一轮启动多个分支，且这些分支可以有不同的长度。在主对话越过合并点继续执行之前，所有分支都必须完成。

**基本示例：**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":2,"turns":3}]}'
```

这会创建一个 5 轮的主对话，并启动一个子代理分支：它在第 2 轮启动，运行 3 轮，然后在第 3 轮合并回来。

**延迟合并：**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":1,"turns":2,"merge_after":2}]}'
```

这会在第 1 轮启动一个运行 2 轮的子代理，并在第 3 轮（`at_turn + merge_after`）合并回来；与此同时，主对话会并行继续执行到第 2 轮。

**多个分支：**

```bash
--data '{"kind":"synthetic_text","prompt_tokens":256,"output_tokens":128,"turns":5,"branches":[{"at_turn":2,"turns":3},{"at_turn":2,"turns":1,"agent_id":"reviewer"}]}'
```

这会在第 2 轮启动两个子代理：一个运行 3 轮，另一个运行 1 轮。两者都会在第 3 轮合并回来。主对话第 3 轮会收到第 0 至 2 轮的完整历史，以及每个子代理的最终输出。

**结果中的历史长度和轮次索引：**

`benchmarks.json` 中每个请求条目都包含：

- `info.history_len`：随该请求发送到服务器的已组装历史中，之前消息的数量。由于历史可能包含通过 `last` 边合并的分叉路径中的消息，因此 `history_len` 在合并点可能会跳增，而不是每次只增加 1。以全新（`new`）上下文启动的分支根节点从 `0` 开始。
- `info.turn_index`：更简单的路径深度计数器。在 `new` 边处重置为 `0`，沿 `full` 边每次加 1；每条 `last` 边最多按增加 `1` 计算，且不会继续遍历该父节点的祖先节点。节点有多个父节点时，`turn_index` 取这些贡献中的最大值（即最长路径）。
- `info.preceding_nodes`：按拓扑执行顺序排在此请求之前的图节点数量（从 0 开始）。与 `turn_index` 和 `history_len` 不同，它不受 `history_context` 影响。回放使用 `history_context=new`（例如 OTEL `history=trace`）时可以使用此值，此时每个请求的 `turn_index` 都保持为 `0`。记录的工具循环会增加额外图节点（调用 + 注入），因此 `preceding_nodes` 统计的是 DAG 节点，而不是原始 LLM span。并行子代理同级节点的值可能会因拓扑排序中的平局处理而不同。

在子代理分支合并之后，`history_len` 通常会大于 `turn_index`：合并的 `last` 输出会计入组装后的历史，但对路径深度只会以非递归的 `1` 计入；如果 `full` 路径已经更长，也不会增加 `turn_index`。

**分支配置字段：**

| 字段                  | 类型          | 默认值     | 说明                                                           |
| --------------------- | ------------- | ---------- | -------------------------------------------------------------- |
| `at_turn`             | `int`         | （必填）   | 分支启动时所在的主链轮次索引                                   |
| `turns`               | `int`         | （必填）   | 此分支的轮次数                                                 |
| `agent_id`            | `str`         | `"worker"` | 分支节点的代理标识                                             |
| `merge_after`         | `int`         | `1`        | 在 `at_turn` 之后经过多少个主链轮次再合并                      |
| `first_prompt_tokens` | `int \| null` | `null`     | 第一轮提示词大小覆盖值（默认使用父级的 `first_prompt_tokens`） |
| `first_output_tokens` | `int \| null` | `null`     | 第一轮输出大小覆盖值（默认使用父级的 `first_output_tokens`）   |

分支轮次根据主对话的 `prompt_tokens` / `output_tokens` 分布采样。未设置的分支 `first_*` 字段会继承父对话的 `first_*` 设置（如果父对话也未设置，则回退到主分布）。分支 `first_*` 字段也支持相同的 `_stdev` / `_min` / `_max` 分布参数。

> [!NOTE]\
> `at_turn + merge_after` 必须小于 `turns`，这样才会存在合并点。不允许分支在主对话结束时或结束之后才合并。

### 请求格式 {#request-formatting}

多轮对话会根据请求格式采用不同的格式：

#### 聊天补全（`/v1/chat/completions`）

对于聊天补全，GuideLLM 会创建一个包含对话历史的 `messages` 数组：

```json
{
  "messages": [
    {"role": "system", "content": "prefix content"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "text", "text": "prompt_2 content"}]}
  ]
}
```

#### Responses API（`/v1/responses`）

对于禁用了 `server_history` 的 Responses API，GuideLLM 会创建一个包含对话历史的 `input` 数组，并将前缀设置为 `instructions`：

```json
{
  "instructions": "prefix content",
  "input": [
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_0 content"}]},
    {"role": "assistant", "content": "response to prompt_0"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_1 content"}]},
    {"role": "assistant", "content": "response to prompt_1"},
    {"role": "user", "content": [{"type": "input_text", "text": "prompt_2 content"}]}
  ]
}
```

#### 文本补全（`/v1/completions`）

对于文本补全，对话历史会拼接在一起：

```text
prefix content prompt_0 content response to prompt_0 prompt_1 content response to prompt_1 prompt_2 content
```

### 服务端对话历史（仅适用于 `/v1/responses`） {#server-side-conversation-history-v1responses-only}

默认情况下，GuideLLM 会在每个请求中重放完整的对话历史（客户端历史）。对于 Responses API，也可以通过 `previous_response_id` 字段使用**服务端历史**，由服务器存储并管理对话上下文。

在后端配置中启用服务端历史：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/responses,server_history=true \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3
```

启用后，GuideLLM 只会发送当前轮次的输入，并通过 ID 引用上一条响应。服务器会在内部重建完整的对话上下文。

**要求：**

- 服务器必须支持 `previous_response_id`，并启用响应存储。对于 vLLM，启动服务器时设置环境变量 `VLLM_ENABLE_RESPONSES_API_STORE=1`。
- 如果服务器不支持响应存储，第 2 轮及之后的请求会失败并报错（通常为 404）。
- 此选项仅适用于 `/v1/responses`。在其他请求格式中使用会导致启动时报错。

## 工具调用

多轮工具调用支持作为多轮基准测试的一部分使用。有关服务器设置、工具定义、工具选择配置和边界情况的完整文档，请参阅专门的[工具调用指南（英文）](../../guides/tool_calling.md)。

## TurnPivot 预处理器

GuideLLM 支持传入多个 `--data` 选项，每个选项指向一个单独的数据集。如果多个数据集包含相同列类型的匹配项，它们会被视为不同的批次。通常，这可用于在同一请求中叠加来自不同数据集的列。例如，将一个数据集中的文本列与另一个包含图像的数据集组合，或将多个正态分布的合成数据集组合成多模态分布。我们可以使用 **TurnPivot** 预处理器转置轮次列和数据集批次。

例如，给定以下数据集：

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_0 prompt_0 | dataset_0 prompt_1 | 11              | 12              |

| prompt_0           | prompt_1           | output_tokens_0 | output_tokens_1 |
| ------------------ | ------------------ | --------------- | --------------- |
| dataset_1 prompt_0 | dataset_1 prompt_1 | 21              | 22              |

不使用 **TurnPivot** 时，第二轮如下：

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_0"}
        ]
    },
    {"role": "assistant", "content": "32 token response to prompt_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_1"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 34,
  ...
}
```

使用 **TurnPivot** 时，第二轮如下：

```json
{
  "messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_0 prompt_0"},
            {"type": "text", "text": "dataset_0 prompt_1"}
        ]
    },
    {"role": "assistant", "content": "23 token response to dataset_0"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "dataset_1 prompt_0"},
            {"type": "text", "text": "dataset_1 prompt_1"}
        ]
    }
  ],
  "output_completion_tokens": 43,
  ...
}
```

### 用法

在 CLI 中使用 TurnPivot 时，将其指定为数据预处理器：

```bash
--data kind=json_file,path=dataset0.jsonl \
--data kind=json_file,path=dataset1.jsonl \
--data-preprocessor kind=encode_media \
--data-preprocessor kind=turn_pivot
```

> [!WARNING]\
> 设置 `--data-preprocessor` 会覆盖**所有**预处理器，*列映射器除外*，因此请务必指定用例所需的其他预处理器。

## 示例

### 1. 使用合成数据进行基础多轮测试

此示例展示如何使用合成数据，对一个简单的三轮对话进行基准测试。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=6 \
  --constraint kind=max_requests,count=30 \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3
```

**关键参数：**

- `--backend`：服务器 URL、模型和请求格式
- `--profile`：基准测试执行配置（`concurrent` 会保持固定数量的并发请求）
- `--constraint kind=max_requests,count=30`：要发送的最大请求数（30 个请求约等于 10 个各含 3 轮的对话）
- `--data`：合成数据配置，包含 200 个提示 token、100 个输出 token 和 3 轮对话

此命令会对 10 个三轮对话进行基准测试；每一轮使用所配置的 token 数量（未设置标准差时，提示为 200 个 token，输出为 100 个 token）。模型会在三轮对话中保留对话历史。

### 2. 使用系统提示（前缀）进行多轮测试

此示例展示如何在多轮对话中包含系统提示，这适用于许多场景，例如模拟智能体系统。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=constant,rate=2.0 \
  --constraint kind=max_requests,count=100 \
  --data kind=synthetic_text,prompt_tokens=150,output_tokens=75,turns=4,prefix_tokens=100
```

**关键参数：**

- `--profile kind=constant,rate=2.0`：以每秒 2 个请求的固定速率发送请求
- `--constraint kind=max_requests,count=100`：请求数上限为 100（100 个请求约等于 25 个各含 4 轮的对话）
- `--data`：添加 `prefix_tokens=100`，以生成包含 100 个 token 的系统提示

在此基准测试中，每个对话开头都包含一条系统消息（前缀），之后进行 4 轮用户与助手的交互。在实际使用场景中，系统提示用于设定适用于整个对话的上下文或指令，而且通常对所有用户都相同。

> [!NOTE]\
> `turns=4` 与 `--constraint kind=max_requests,count=100` 配合使用时，将产生 25 个或更多对话。只有上一轮完成后，才能安排后续轮次。某一轮完成后，该对话会被放到当前工作线程队列的最前面。

### 3. 高级前缀分布

此示例展示如何使用多个带权重分布的前缀配置，这对于测试不同的系统提示场景很有用。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=constant,rate=1.5 \
  --constraint kind=max_duration,seconds=60 \
  --data '{
  "kind": "synthetic_text",
  "prompt_tokens": 180,
  "output_tokens": 90,
  "turns": 3,
  "prefix_buckets": [
    {
      "bucket_weight": 60,
      "prefix_count": 5,
      "prefix_tokens": 100
    },
    {
      "bucket_weight": 40,
      "prefix_count": 1,
      "prefix_tokens": 0
    }
  ]
}'
```

**关键参数：**

- `--profile kind=constant,rate=1.5`：以每秒 1.5 个请求的固定速率发送请求
- `--constraint kind=max_duration,seconds=60`：每种策略最多运行 60 秒
- `--data`：定义两种前缀分布的 `prefix_buckets` JSON 配置

此配置会生成如下分布：60% 的对话使用 5 个前缀之一（每个 100 个 token），40% 的对话使用 0 个 token 的前缀（相当于没有前缀）。

### 4. 使用多轮文件数据集

此示例展示如何使用包含多轮结构的现有数据集文件。

**JSONL 文件示例**（multiturn_conversations.jsonl）：

```json
{"prefix": "You are a helpful assistant.", "prompt_0": "What is Python?", "output_tokens_count_0": 50, "prompt_1": "How do I install it?", "output_tokens_count_1": 40}
{"prefix": "You are a coding expert.", "prompt_0": "Explain functions", "output_tokens_count_0": 60, "prompt_1": "Give me an example", "output_tokens_count_1": 45}
```

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=200 \
  --data kind=json_file,path=multiturn_conversations.jsonl
```

**关键参数：**

- `--profile kind=concurrent,streams=10`：保持 10 个并发请求
- `--constraint kind=max_requests,count=200`：请求数上限为 200（200 个请求约等于 100 个各含 2 轮的对话）
- `--data`：包含按轮次索引列的 JSONL 文件路径

### 5. 使用 TurnPivot 处理多个数据集

此示例展示如何使用 TurnPivot 预处理器构建合成数据集，让每一轮遵循不同的分布。

**命令：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=meta-llama/Llama-3.1-8B-Instruct,request_format=/v1/chat/completions \
  --profile kind=concurrent,streams=10 \
  --constraint kind=max_requests,count=150 \
  --data kind=synthetic_text,prefix_tokens=512,prompt_tokens=128,output_tokens=256 \
  --data kind=synthetic_text,prompt_tokens=256,prompt_token_stdev=32,output_tokens=128 \
  --data kind=synthetic_text,prompt_tokens=64,output_tokens=128,output_tokens_stdev=16 \
  --data-preprocessor kind=turn_pivot
```

**关键参数：**

- `--constraint kind=max_requests,count=150`：请求数上限为 150（150 个请求约等于 50 个各含 3 轮的对话）
- `--data`：为每个数据集分别指定；也可以一次性将它们指定为数组
- `--data-preprocessor kind=turn_pivot`：对数据集和轮次列进行转置

> [!WARNING]\
> 在当前 CLI 设计中，设置 `--data-preprocessor` 会覆盖**所有**预处理器，*列映射器除外*。请确保指定用例所需的其他预处理器。

## 历史记录中的推理

推理模型会在最终答案前生成思维链 token。默认情况下，这些内容不会保留在后续轮次的对话历史中。若要将其纳入历史记录，请在后端配置中启用 `multiturn_reasoning`：

- `false`（默认值）——历史记录中不包含推理内容
- `true`——使用 `<think>...</think>` 标签包裹（等同于 `"<think>{reasoning}</think>"`）
- 包含 `{reasoning}` 的格式字符串——自定义包裹格式

### Think 标签（适用于大多数模型）

```bash
--backend kind=openai_http,target=http://localhost:8000,multiturn_reasoning=true
```

对话历史中发送的内容如下：

```json
{"role": "assistant", "content": "<think>step-by-step reasoning...</think>Final answer."}
```

### Granite 格式

```bash
--backend 'kind=openai_http,target=http://localhost:8000,multiturn_reasoning="Here is my thought process:{reasoning}Here is my response:"'
```

### 原始格式（无分隔符）

```bash
--backend 'kind=openai_http,target=http://localhost:8000,multiturn_reasoning="{reasoning}"'
```

### 常见模型搭配

| 模型家族                                                           | 推荐值                                                         |
| ------------------------------------------------------------------ | -------------------------------------------------------------- |
| DeepSeek R1, QwQ, Qwen3, Gemma 4, GLM-4.5, Holo2, Cohere Command A | `true`                                                         |
| IBM Granite 3.2                                                    | `"Here is my thought process:{reasoning}Here is my response:"` |
| OpenAI o-series (o1, o3)                                           | `false`（推理内容无法通过 API 重放）                           |
| Custom / no delimiters                                             | `"{reasoning}"`                                                |

### 关于推理的其他说明

只有在客户端收到 `reasoning` 片段时，GuideLLM 才会识别推理内容。通常只有在包含推理解析器时才会发送这些片段。否则，GuideLLM 会将它们视为非推理 token。关于 KVCache，推理解析器会重新格式化推理内容；因此，即使发送了推理 token，KVCache 也很可能无法完全匹配，从而导致部分缓存未命中。截至本文撰写时，用于从 vLLM 缓存中逐出推理内容的设置仍在审核中。

建议研究所用模型和模型服务器的设计，确保推理历史记录配置正确。

## 限制与注意事项

### 支持的请求格式

目前支持以下格式的多轮基准测试：

- `/v1/chat/completions` - 使用聊天模板格式
- `/v1/responses` - 使用 OpenAI Responses API 输入格式
- `/v1/completions` - 使用基础的历史记录拼接方式

音频端点（`/v1/audio/transcriptions`、`/v1/audio/translations`）不支持多轮基准测试。

### 列命名要求

按轮次索引的列必须遵循以下命名约定：

- 列映射作用于基础名称。例如：`--data-column-mapper '{"kind":"generative_column_mapper","column_mappings": {"text_column": "prompt"}}'`
- 轮次索引可以采用 `-0` 或 `_0` 的形式。具体编号并不重要；系统会重新编号轮次以填补空缺。
- 所有轮次列必须使用相同的基础名称。例如：`prompt_0`、`prompt_2` 等。

### 模型上下文注意事项

多轮对话会不断累积对话历史，从而增加内存用量：

- 每一轮都会包含此前所有轮次的完整对话历史
- 对话轮数越多，提示越长
- 随着历史累积，token 数量会逐轮增长
- 配置轮数和 token 数量时，请考虑模型的上下文窗口大小

例如，`--data kind=synthetic_text,prefix_tokens=50,prompt_tokens=100,output_tokens=200,turns=5` 将产生以下数据：

- 第 1 轮：输入 150 个 token；输出 200 个 token
- 第 2 轮：（150 + 200）+ 100 = 输入 450 个 token；输出 200 个 token
- 第 3 轮：（450 + 200）+ 100 = 输入 750 个 token；输出 200 个 token
- 第 4 轮：（750 + 200）+ 100 = 输入 1050 个 token；输出 200 个 token
- 第 5 轮：（1050 + 200）+ 100 = 输入 1350 个 token；输出 200 个 token

### 其他注意事项

与单轮基准测试相比，多轮基准测试还有以下特点需要考虑：

- 由于各轮大小分布范围较广，按请求统计的指标预计会有较大波动
- 任何错误都会终止整个对话（如果某一轮失败，该对话剩余轮次都会取消）
