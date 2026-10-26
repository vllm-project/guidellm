---
weight: 7
source_sha256: !!str "0dce93800de781aa287cbe1a28696ec8011cbf171a7fb709942f1229562827c5"
---

# 跟踪文件格式

许多跟踪文件的格式需要经过特殊处理，才能准确重放。本指南介绍 GuideLLM 当前支持的所有跟踪文件格式，以及通用和格式专属的数据参数。

有关重放 profile 和基于文件的数据集的完整用法，请参阅[跟踪重放基准测试](../getting-started/benchmark.md#trace-replay-benchmarking)。

## 支持的格式

通过 `--data` 参数以 `kind=format` 的形式指定以下格式：

- `trace_synthetic`：仅包含使用合成提示词完成完整跟踪重放基准测试所需的最基本数据的格式。
- `mooncake`：服务平台 *Mooncake* 使用的跟踪格式，定义见[相关论文](https://doi.org/10.48550/arXiv.2407.00079)。
- `weka`：WEKA 的 *Augmented Memory Grid* 使用的跟踪格式，详见[原始研究仓库中的规范](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md)。
- `otel`（别名 `opentelemetry`）：OpenTelemetry GenAI spans。GuideLLM 会保留成功的 LLM span，并将每个 `trace_id` 作为一段对话进行重放。除非设置 `history=runtime`，否则会发送记录的 `gen_ai.input.messages`，并将每个 span 的完整输入作为请求内容（`history=trace`）。

## 加载跟踪数据

跟踪重放始终使用 `--profile kind=replay`。选择一种**格式**（`trace_synthetic`、`mooncake` 或 `weka`）以及一种来源类型，来源类型可从其他反序列化器类型中选择（例如 `json_file`、`huggingface`）。例如：

**使用本地 `json_file` 的 `trace_synthetic`：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,time_scale=2.0 \
  --constraint kind=max_requests,count=30
```

**从 `huggingface` 加载 WEKA 数据集：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay,prefetch=all \
  --data kind=weka,source.kind=huggingface,source.source=semianalysisai/cc-traces-weka-no-subagents-051226 \
  --constraint kind=max_requests,count=30
```

**从 `huggingface` 加载 Mooncake 数据集：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=mooncake,source.kind=huggingface,source.source=valeriol29/mooncake-traces,load_kwargs.name=mooncake \
  --constraint kind=max_requests,count=30
```

**从 `huggingface` 加载 OTEL 数据集：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
  --constraint kind=max_requests,count=30
```

## 通用数据参数

所有跟踪格式都可以使用以下可选数据参数：

| 参数                      | 默认值          | 说明                                                                                              |
| ------------------------- | --------------- | ------------------------------------------------------------------------------------------------- |
| `timestamp_column`        | "timestamp"     | 跟踪文件中时间戳所在的列名。                                                                      |
| `prompt_tokens_column`    | "input_length"  | 跟踪文件中提示词 token 数所在的列名。                                                             |
| `output_tokens_column`    | "output_length" | 跟踪文件中输出 token 数所在的列名。                                                               |
| `duration_column`         | "duration"      | 可选列，表示每个请求的运行时长（秒）。由 `schedule_turn=idle_gap` 使用。                          |
| `time_scale`              | 1.0             | 等待时间和打包上限处理后，对剩余相对时间戳进行缩放。                                              |
| `max_wait`                | 未设置          | 同一会话中连续请求在原始跟踪时间内的最大间隔（秒）。                                              |
| `max_session_wait`        | 未设置          | 从上一个会话的最后一个请求到当前会话之间，原始跟踪时间内允许的最大空闲时长（秒）。                |
| `min_concurrent_sessions` | 未设置          | 对会话进行打包，使稳态期间至少有指定数量的会话重叠。                                              |
| `copies`                  | 1               | 按顺序完整重放数据集的次数；第 k+1 次重放从第 k 次最后一个已调度请求的位置开始。                  |
| `copy_offset`             | 1.0             | 下一轮重放相对于上一轮时间跨度的起始位置：0 表示从头开始，1 表示从尾部开始，大于 1 表示留出间隔。 |

通过 `--data` 参数传入这些设置，例如：

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data "kind=trace_synthetic,source.kind=json_file,source.path=replay.jsonl,timestamp_column=ts,prompt_tokens_column=input_tokens,output_tokens_column=generated_tokens,time_scale=1.0,max_session_wait=30"
```

可以将 `trace_synthetic` 视为通用格式选项：它只查找时间戳、提示词 token 数和输出 token 数这几列，忽略数据集中的其他所有特性。`trace_synthetic` 主要用于测试，也可以作为 GuideLLM 尚未支持的跟踪格式的后备方案。

重放 profile 的 `schedule_turn` 参数决定如何应用这些时间戳。默认值 `idle_gap` 会在 `duration_column` 的请求时长之后保留空闲间隔：例如，某请求记录的时长为 1 秒，下一条时间戳在 5 秒后，则下一个请求会在当前请求实际完成 4 秒后启动。当服务器过载时，`idle_gap` 更宽容，因为延迟的请求会在前面轮次结束后立即发送。如果缺少 `duration_column`，加载器会记录一条警告，并将每个请求视为瞬时完成，因此实际间隔会略长。`timestamp` 则按照跟踪中的时间启动每个请求；只有前一个轮次仍在运行时才会等待。

`trace_synthetic` 和 `mooncake` 会将每一行作为一段独立的单请求对话重放。各行按时间戳排序，并保留相对于跟踪中首个请求的时间偏移。提示词会随着数据行被读取而生成；在一次 `copies` 重放中，Mooncake 哈希 ID 会在各行之间共享。使用 `max_session_wait` 可限制这些独立请求之间的间隔；`max_wait` 只会限制多请求对话内部的间隔，例如 WEKA 会话。

应优先使用 `min_concurrent_sessions`、等待上限和 `time_scale` 来增加并行度。只有打包后的一轮重放对基准测试（`max_duration` / `max_requests`）来说太短时，才使用 `copies`。默认值 `copy_offset=1` 会将完整打包数据集连续重放：下一轮从上一轮最后一个请求的时间戳开始。`copy_offset=0` 会从上一轮首个请求的时间戳开始；介于 0 和 1 之间的值会在首尾之间插值；大于 1 的值会增加间隔。对于哈希 ID 格式（`mooncake`、`weka`），每轮重放都会使用单独加盐的全局 token 块表，避免后续轮次复用先前 token 并虚增前缀缓存命中率。

## 格式专属数据参数

### `mooncake`

Mooncake 格式需要额外提供一列基于前缀的缓存哈希 ID。生成提示词时，共享同一个前序 ID 的哈希 ID 必须表示不同的 token ID 块。

| 参数                 | 默认值     | 说明                               |
| -------------------- | ---------- | ---------------------------------- |
| `hash_ids_column`    | "hash_ids" | 跟踪文件中哈希 ID 列表所在的列名。 |
| `hash_id_block_size` | 512        | 每个哈希 ID 表示的 token 数量。    |

### `weka`

**注意：**

目前尚未实现对 `tool_tokens` / `system_tokens` 前缀进行预热，也未实现对扁平化 agent 使用哈希 ID 的最长公共前缀（LCP）拆分。声明为 `type: "subagent"` 的分组和工具调用事件（`stop: tool_use`、`input_types: ["tool_result"]`）会被重放。

WEKA 格式要求提供一列对话 UUID，且该列不能再嵌套在其他列中。时间戳、输入 token 长度、输出 token 长度和哈希 ID 列则必须一并包装在一个 JSON 列中（例如 `"requests"`），其形式为 JSON 对象列表。

与 Mooncake 类似，WEKA 使用基于前缀缓存的哈希 ID。原始[规范](https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md)要求哈希 ID 大于或等于 1；如果输入 token 不足以填满哈希 ID 块，则应丢弃末尾的哈希 ID。为兼容可能不完全符合规范的数据集（例如 [semianalysisai/cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226)），GuideLLM 接受任意非负整数作为有效哈希 ID，并会丢弃不完整的哈希 ID。

GuideLLM 会从第一段对话开始生成提示词。当前对话结束后，再处理下一段对话。相对时间戳表示相对于数据集中最早请求的时间偏移，因此后续对话的开始时间可以晚于第一段对话。

一般来说，使用 WEKA 跟踪时建议设置 `prefetch=all`。只有当跟踪中的会话顺序正确且彼此不重叠时，才可以不设置。否则，数据反序列化器可能会落后，导致会话在其相对跟踪时间之后才被调度。当大多数或所有会话中的请求都在基准测试开始时或接近开始时启动，这个问题会更加明显。必须预先获取所有请求，才能确保每个会话都按时开始。使用 `--profile kind=replay,prefetch=all`，让这些请求图在基准测试计时开始前完成加载。参阅[请求加载时间晚于其应启动时间](troubleshooting.md#requests-load-after-they-are-due)。

哈希 ID 的处理方式由每行的 `hash_id_scope` 字段决定：

- `"global"` 或未设置：所有对话共享一张 token 块表，与 Mooncake 一致。后续对话中的相同哈希 ID 会复用之前的 token 块，使前缀缓存命中率接近原始跟踪。每次 `copies` 重放都会使用单独加盐的全局表。
- `"local"`：哈希 ID 只在当前对话中生效。对话输出后，该表即被丢弃。每次 `copies` 重放也会分别进行本地隔离。

声明为 `type: "subagent"` 的条目会转换为隔离的子链。每个子链从父级上一轮 API 调用派生，并使用新的历史（`history_context="new"`）；父级的下一轮则会等待自该轮以来启动的所有同级子链（`history_context="last"`）完成。因此，同一对父级轮次之间列出的多个 subagent 会并行运行；只有全部完成后，父级才会继续。每一级都保留请求列表的顺序（该顺序定义了启动和汇合拓扑），不会按时间戳重新排序。

如果内部请求时间戳是相对于启动时间的，GuideLLM 会按规范处理；如果它们是绝对时间，则按已发布的 Hugging Face 数据集处理：当首个内部 `t` 小于 subagent 条目的启动 `t` 时，内部时间会转换为 `spawn_t + inner_t`；否则保持原值。之后，对话时间戳会计算为 `absolute_t` 减去数据集中最早 API 请求的时间。

同一 agent 的连续轮次仍会串行执行。如果这些轮次在时间上重叠（`t[i] + api_time[i] > t[i+1]`；若缺少 `api_time`，则以 `t[i+1] <= t[i]` 为准），GuideLLM 会记录一条 debug 消息。不同 subagent 之间的重叠属于预期并行行为，不会触发警告。

工具调用事件会映射到 GuideLLM 现有的客户端工具调用流水线。若某请求包含 `stop: "tool_use"` 且输入为用户文本，则该轮会成为 `client_tool_call`。紧随其后的请求若包含 `input_types: ["tool_result"]`（或缺少 `input_types` 时，同一 agent 链上位于 `stop: "tool_use"` 后的下一个请求），则会成为 `tool_response_injection`。如果该注入行也包含 `stop: "tool_use"`，它仍会发送工具结果并保留 `tools`，让模型可以继续调用工具。跟踪中不包含真实的工具 schema 或结果。请像配置[合成数据](tool_calling.md#providing-tool-definitions)一样传入 `tools`，并可选传入 `tool_response_tokens`；否则 GuideLLM 会使用默认的合成工具定义和占位工具响应。注入轮次中，聊天处理器不会将哈希 ID 提示作为用户消息发送。

| 参数                         | 默认值     | 说明                                                                          |
| ---------------------------- | ---------- | ----------------------------------------------------------------------------- |
| `conversation_id_column`     | "id"       | 跟踪文件中对话 UUID 所在的列名。                                              |
| `hash_ids_column`            | "hash_ids" | 跟踪文件中哈希 ID 列表所在的列名。                                            |
| `hash_id_block_size`         | 64         | 每个哈希 ID 表示的 token 数量。                                               |
| `tools`                      | `None`     | 工具调用轮次使用的 OpenAI 格式工具定义。未设置时使用内置占位工具。            |
| `tool_response_tokens`       | `None`     | 模拟工具结果的平均 token 数。未设置时使用简短占位内容（`{"status": "ok"}`）。 |
| `tool_response_tokens_stdev` | `None`     | 工具响应 token 数量的标准差。                                                 |
| `tool_response_tokens_min`   | `None`     | 工具响应的最小 token 数。                                                     |
| `tool_response_tokens_max`   | `None`     | 工具响应的最大 token 数。                                                     |

以下参数的默认值会被修改：

| 参数                   | 新默认值 |
| ---------------------- | -------- |
| `timestamp_column`     | "t"      |
| `prompt_tokens_column` | "in"     |
| `output_tokens_column` | "out"    |

### `otel`

OpenTelemetry GenAI 跟踪会被重放为带时间信息的对话。支持以下两种文件布局：

- **每行一个会话**：每条 JSONL 记录为 `{ "trace_id": ..., "spans": [ ... ] }`。
- **每行一个 span**：每条 JSONL 记录是一个 span；相邻且 `trace_id` 相同的记录会组成一段对话。分组以流式方式进行，且只合并连续记录，因此交错的 `a, b, a` 会形成三段对话，而不是两段。已发布的重放语料会将每个 `trace_id` 连续写入；并发跟踪的实时 collector 导出则可能不会。

只会重放成功的 LLM span（`gen_ai.operation.name` 为 `chat`、`generate` 或 `text_completion`，或者 span 已包含 usage token 属性）。`invoke_agent` 和失败的 span（`status.code` 为 error）会被丢弃。`execute_tool` span 不会作为 HTTP 请求发送。

每个 LLM span 都是完整的 API 快照：`gen_ai.input.messages` 会重新记录完整对话历史，`prompt_tokens` / `input_tokens` 表示**本次调用的完整输入大小**，而不是增量。`history` 决定如何将该快照转换为请求：

| `history=trace`（默认值）                                                                                                                                                         | `history=runtime`                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 发送 span 中记录的完整消息。设置 `history_context=new`，因此实时生成结果不会替换下一条提示词中的内容。准确度最高。当实时生成的内容与记录的 assistant 文本不同时，前缀缓存会失效。 | 使用 DAG 中所有**实时**输出构成的 `full` 历史；当前轮次只包含**新增**消息。要求 `input[i] == input[i-1] + output[i-1] + delta`。有利于缓存；当实时模型输出出现偏差时，准确度会降低。 |

输出长度始终使用 span 的 completion token 数（`max_tokens` + `ignore_eos`）。

缺少 `gen_ai.input.messages` 的 span 无法作为 OTEL 重放。对于仅含 token 数的扁平化数据转储，请改用 `kind=trace_synthetic`。

通过后端默认使用的 `/v1/chat/completions` 进行重放。OTEL 会在 `raw_messages_column` 中存储 Chat Completions 消息字典，处理器会将其作为 `messages` 数组发送。`/v1/completions` 和 `/v1/responses` 不会读取此列；如果缺少 `prompt` 或 `input`，它们会中止运行并报错，而不会发送空请求体。

记录的工具调用循环会预先拆分到 GuideLLM 的客户端工具调用流水线中。如果某个 LLM span 的输出消息包含 `tool_calls`（或 `gen_ai.response.finish_reasons` 为 `tool_calls` / `tool_call` / `tool_use` / `function_call`），该 span 会成为 `client_tool_call`。如果下一个 span 在剔除 `input[i] + output[i]` 之后新增的消息仅包含 `role=tool` 结果，则会作为 `tool_response_injection` 消费。聊天处理器会将记录的结果字符串重新绑定到**实时的** `tool_call_id`。`gen_ai.tool.definitions` 会在这些轮次提供 `tools_column`（否则使用默认的合成工具）；仅有工具定义不会被判定为工具调用轮次。无论 `history` 设置为何值，注入请求的父级始终使用 `history_context=full`。缺少工具调用时的策略仍由 `--backend tool_call_missing_behavior=...` 设置。

如果下一个 span 无法解析为工具结果，则会合成一个占位注入请求，同时仍会重放该下一个 span。如果 `execute_tool` span 是工具调用 LLM span 的后代，或是与其共享非空 `parent_span_id`、且位于下一个相同父级 LLM span 之前的同级 span，则该 `execute_tool` span 会提供注入文本和节奏信息。否则会在工具调用时间戳处使用合成占位响应。

已完成请求的统计数据会将响应 usage 与请求的预期 token 数合并，因此不会单独报告预期值与实际值的 MAE。如需查看偏差，应在合并前对比 `request.input_metrics` / `output_metrics`（span 中记录的 token 数）和响应 usage。

ISO-8601 格式的 `start_time`（不带时区、带 `Z` 或时区偏移）以及 Hugging Face 解码出的 `datetime` 对象，都会在调度前转换为 epoch 秒。token 数会从 span 的 `attributes` 中读取，优先尝试当前 GenAI 字段名，再尝试已弃用的别名。OTel 的 `parts`（`text`、`tool_call`、`tool_call_response`）会转换为 OpenAI Chat 字典。

| 参数                       | 默认值                                                             | 说明                                                                                                             |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `spans_column`             | "spans"                                                            | 每行一个会话的文件中，嵌套 span 列表所在的列名。                                                                 |
| `trace_id_column`          | "trace_id"                                                         | 用于将每行一个 span 的文件分组为对话的列名。                                                                     |
| `span_timestamp_field`     | "start_time"                                                       | 存储请求开始时间的 span 字段。                                                                                   |
| `input_tokens_attributes`  | `["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"]`      | 按顺序尝试读取提示词 token 数的属性键。                                                                          |
| `output_tokens_attributes` | `["gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"]` | 按顺序尝试读取输出 token 数的属性键。                                                                            |
| `history`                  | `trace`                                                            | `trace` 会重新发送每个 span 的完整输入；`runtime` 只发送新增消息，并使用 DAG 历史。                              |
| `tool_choice`              | `required`                                                         | 客户端工具调用轮次使用 `required` 或 `auto`。使用 `auto` 时请搭配 `tool_call_missing_behavior=ignore_continue`。 |

建议从 Hugging Face 数据开始。IBM 跟踪中每段对话有 30–50 次 LLM 调用，因此首次运行时可使用 `--constraint kind=max_requests` 限制请求数。

**默认模式（`history=trace`）：**

完整发送每个 span 中记录的消息。后续轮次会等待 DAG 依赖，但使用 `history_context=new`，因此不会将实时生成结果插入下一条提示词。

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces \
    --constraint kind=max_requests,count=30
```

**使用 DAG 历史发送记录消息（`history=runtime`）：**

只发送新增消息；此前轮次的内容来自实时生成结果（`history_context=full`）。要求每个 span 的输入都由前一个 span 的输入与输出延续而来。

```bash
guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=replay \
    --data kind=otel,source.kind=huggingface,source.source=ibm-research/synthetic-conversations-traces,history=runtime \
    --constraint kind=max_requests,count=30
```

本地 JSONL 文件也可以通过 `source.kind=json_file,source.path=...` 使用相同的 `history` 设置。

可公开获取的 Hugging Face 语料：

- [ibm-research/synthetic-conversations-traces](https://huggingface.co/datasets/ibm-research/synthetic-conversations-traces)：多轮聊天，每行一个会话的 JSONL，包含 `prompt_tokens` / `completion_tokens`。
- [Exgentic/agent-llm-traces-v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2)：agent 会话，包含嵌套的 `spans` 和 `input_tokens` / `output_tokens`。
- [ibm-research/lmcache-agentic-traces_Otel](https://huggingface.co/datasets/ibm-research/lmcache-agentic-traces_Otel)：agent 会话，每行一个会话的 JSONL，使用已弃用的 token 键。

相关语料：[DiscoPosse/agent-llm-traces](https://huggingface.co/datasets/DiscoPosse/agent-llm-traces)（Exgentic v1 schema）和 [lenadan/otel-test-snippet-jsonl](https://huggingface.co/datasets/lenadan/otel-test-snippet-jsonl)（小型的每行一个 span 示例）。[ibm-research/codex_swebenchpro_traces_Otel](https://huggingface.co/datasets/ibm-research/codex_swebenchpro_traces_Otel) 不包含 usage token 属性，不能用于 token 数重放。缺少 `gen_ai.input.messages` 的 span 应使用 `trace_synthetic`。
