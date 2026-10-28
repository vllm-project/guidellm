---
weight: 6
source_sha256: !!str "a97dddd6c736eea4bf3a2782284166890cfd441d7eb2bdd7c0cb4e4eb75c4e9e"
---

# 工具调用

GuideLLM 支持对多轮工具调用工作负载进行基准测试。客户端工具调用轮次会**预先确定**：数据流水线会提前判断哪些用户轮次预期调用工具，哪些轮次只生成普通文本。每个客户端工具调用用户轮次都会自动生成一个额外的 `tool_response_injection` 请求，因此每段对话的总请求数为 `turns + len(tool_call_turns)`。例如，`turns=3, tool_call_turns=[0, 1]` 会产生 5 个请求：工具调用、注入、工具调用、注入、普通请求。

客户端工具调用轮次完成后，GuideLLM 会通过单独的注入轮次将工具响应发回服务器，并等待服务器返回文本响应，然后再继续下一个用户轮次。工具响应内容按以下优先级取自三个来源之一：数据集的工具响应列、通过 `tool_response_tokens` 配置的合成数据，或简短占位内容（`{"status": "ok"}`）。只有数据中包含 `tools_column` 的轮次，请求体才会包含工具定义；不调用工具的轮次会作为普通聊天补全请求发送，不包含 `tools` 或 `tool_choice`。

## 支持的请求格式

工具调用支持 `/v1/chat/completions` 和 `/v1/responses`。本指南中的 CLI 示例默认使用 `/v1/chat/completions`，但在后端配置中设置 `request_format=/v1/responses` 后，所有功能也都能以相同方式运行。两种格式的唯一区别是后续请求中工具调用历史的表示方式：

- **`/v1/chat/completions`** 使用 `role: "assistant"` 消息及其 `tool_calls` 数组，随后使用携带工具结果的 `role: "tool"` 消息。
- **`/v1/responses`** 在 `input` 数组中使用 `function_call` 项，随后使用 `function_call_output` 项。

有关工具调用消息在线路上的完整格式，请参阅 [OpenAI 函数调用指南](https://developers.openai.com/api/docs/guides/function-calling)。

## 模拟客户端工具调用

GuideLLM 目前支持模拟的客户端工具调用（`turn_type="client_tool_call"`）。也就是说，推理服务器会运行模型，并可能返回真实的 `tool_calls`，但 GuideLLM **不会**调用实时 API 或其他运行时来执行这些函数。每个客户端工具调用轮次结束后，基准测试都会发送一个独立的工具响应注入请求（`turn_type="tool_response_injection"`），其中包含模拟的工具输出，然后等待服务器返回文本响应，再继续下一个用户轮次。这样可以测量包含工具调用处理环节的 LLM 吞吐量，但不包括外部工具的延迟或副作用。

## 服务端工具调用

对于在内部处理工具执行的服务器（例如使用 `container_auto` 的 OpenAI Responses API，或使用 MCP 工具组的 [OGX](https://github.com/ogx-ai/ogx)），请使用 `server_tool_call` 轮次。从 GuideLLM 的角度看，这类轮次与普通轮次类似，但发送请求时不会将 `tool_choice` 覆盖为 `"none"`，因此服务器配置的工具仍可使用。

此模式不会创建注入轮次，GuideLLM 也不会模拟工具响应。服务器会在内部完成完整的工具调用循环并返回最终文本答案。延迟指标会包含服务端工具执行时间。

### 使用合成数据进行服务端工具处理

对于合成数据，可以通过 `server_tool_call_turns` 将用户轮次标记为由服务器处理。参数可以是整数 N（表示前 N 个轮次）、索引列表，或 `-1`（表示所有轮次）。以下示例将三个轮次均标记为 `server_tool_call`，由服务器决定每轮是否调用工具：

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000","extras":{"body":{"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}}}' \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,server_tool_call_turns=-1 \
  --constraint kind=max_requests,count=30
```

简写 `-1` 等同于将 `server_tool_call_turns` 设为 `N`，其中 N 等于 `turns`，这样就不必手动保持两个值一致。`tool_call_turns`（客户端工具调用）也支持此简写。

### 使用真实数据集进行服务端工具处理

注意：目前将常见工具调用数据集映射到请求时存在一些限制。

对于包含工具定义（例如 `tools` 列）的真实数据集，finalizer 的 `tool_call_mode` 设置决定这些轮次按客户端还是服务端工具调用处理：

- `tool_call_mode="client"`（默认值）：包含工具定义的轮次会变为 `client_tool_call` 和 `tool_response_injection` 配对轮次。GuideLLM 会模拟工具响应并发回服务器。
- `tool_call_mode="server"`：包含工具定义的轮次会变为 `server_tool_call`。不会创建注入轮次。数据集中的工具定义会被移除；工具应通过 `--backend` 在后端级别配置，或直接在服务器上配置。

**OpenAI Responses API**——在后端配置中通过 `extras.body.tools` 将工具传入请求体：

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"https://api.openai.com","request_format":"/v1/responses","extras":{"body":{"tools":[{"type":"shell","environment":{"type":"container_auto"}}]}}}' \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai,load_kwargs.split=train \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer kind=generative,tool_call_mode=server \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

**OGX（Llama Stack）**——这是一个开源且兼容 OpenAI 的智能体服务器，通过其 `/v1/responses` 编排循环在服务端处理工具执行。OGX 服务器上配置工具（MCP 工具组、内置工具），因此 GuideLLM 命令中无需设置 `extras.body.tools`：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8321,request_format=/v1/responses \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai,load_kwargs.split=train \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --data-finalizer kind=generative,tool_call_mode=server \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

服务端工具调用所需的工具可以通过 `--backend` 配置在请求体中（例如 OpenAI 的 `extras.body.tools`），也可以直接配置在服务器上（例如 OGX MCP 工具组）。模型会逐轮决定是否调用工具。

## 服务器设置

工具调用需要服务器端支持。

**vLLM**——启用自动工具选择，并指定与模型匹配的解析器：

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

常见解析器包括：`hermes`（Qwen/Hermes）、`llama3_json`（Llama 3.x）和 `mistral`（Mistral）。如果不添加这些参数，vLLM 会因语法错误拒绝工具调用输出。

**OGX（Llama Stack）**——如需在服务端执行工具，[OGX](https://github.com/ogx-ai/ogx) 提供兼容 OpenAI 的 `/v1/responses` 端点，可在内部处理完整的工具调用循环。OGX 可以使用 vLLM 作为推理后端，并支持通过 MCP 工具组在服务端执行工具。设置方法和工具组配置请参阅 [OGX 文档](https://ogx-ai.github.io/docs)。

## 提供工具定义

工具定义始终通过数据流水线提供，而不是使用全局 CLI 参数。共有三种提供方式：

**1. 合成数据**——在数据配置中设置 `tool_call_turns`（以及可选的 `tools`）：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

若要指定不连续的工具调用轮次，请传入从 0 开始的轮次索引列表：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,request_format=/v1/chat/completions \
  --data '{"kind":"synthetic_text","prompt_tokens":200,"output_tokens":100,"turns":4,"tool_call_turns":[0,2]}' \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

合成数据中的工具调用配置字段：

| 字段                         | 类型               | 默认值 | 说明                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------- | ------------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool_call_turns`            | `int \| list[int]` | `0`    | 哪些用户轮次包含工具定义并预期收到工具调用响应。索引以用户轮次为准，从 0 开始（不是展开后的请求列表索引）。整数 N 表示前 N 个用户轮次；列表用于指定索引（例如 `[0, 2]`）；`-1` 表示所有轮次。每个工具调用用户轮次都会额外生成一个注入请求，因此 `turns=3` 且 `tool_call_turns=[0,1]` 时总共会产生 5 个请求。设为 `0` 或 `[]` 时不调用工具。                                                               |
| `server_tool_call_turns`     | `int \| list[int]` | `0`    | 哪些用户轮次使用服务端工具调用。这些轮次会标记为 `server_tool_call`，因此不会设置 `tool_choice="none"`，服务器可以使用其已配置的工具。不会创建注入轮次。不得与 `tool_call_turns` 重叠。整数 N 表示前 N 个用户轮次；列表用于指定索引；`-1` 表示所有轮次。                                                                                                                                                  |
| `tools`                      | `list`             | `None` | 工具定义，格式可以是 [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)（包含嵌套的 `function` 键）或 [Responses API](https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools)（扁平格式）。GuideLLM 会在发送请求时自动转换为与 `request_format` 匹配的格式。建议使用与目标端点相同的格式。若为 `None`，则使用内置占位工具。 |
| `tool_response_tokens`       | `int`              | `None` | 合成工具调用响应的平均 token 数。若为 `None`，则使用简短占位内容（`{"status": "ok"}`）。                                                                                                                                                                                                                                                                                                                  |
| `tool_response_tokens_stdev` | `int`              | `None` | 工具响应 token 数量的标准差。                                                                                                                                                                                                                                                                                                                                                                             |
| `tool_response_tokens_min`   | `int`              | `None` | 工具响应的最小 token 数。                                                                                                                                                                                                                                                                                                                                                                                 |
| `tool_response_tokens_max`   | `int`              | `None` | 工具响应的最大 token 数。                                                                                                                                                                                                                                                                                                                                                                                 |

注意：这里的 token 数量只计算模拟工具调用响应中某个字段的内容。JSON 结构本身还会为模拟工具调用响应增加约 5 个 token。

**配置工具响应内容**——默认情况下，工具结果使用简短占位内容（`{"status": "ok"}`）。可以通过 `GUIDELLM__DEFAULT_SYNTHETIC_TOOL_RESPONSE` 环境变量修改此默认值。若要进行更真实的基准测试，可设置 `tool_response_tokens` 来生成长度可变的 JSON 响应：

以下示例将 `tool_response_tokens` 设为 50。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2,tool_response_tokens=50 \
  --constraint kind=max_requests,count=30 \
  --profile kind=constant,rate=1
```

`tool_response_tokens_stdev`、`tool_response_tokens_min` 和 `tool_response_tokens_max` 字段的作用方式与相应的 `prompt_tokens_*` / `output_tokens_*` 方差参数相同。

**2. WEKA 跟踪**——工具调用轮次来自文件（`stop` / `input_types`），但工具 schema 不包含在文件中。在 `--data kind=weka` 上像合成数据一样传入 `tools`，并可选传入 `tool_response_tokens`。哪些轮次调用工具不可配置；`tool_call_turns` 不适用。

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data '{"kind":"weka","source":{"kind":"json_file","path":"trace.jsonl"},"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_response_tokens":50}' \
  --constraint kind=max_requests,count=30
```

如果省略 `tools`，则与合成数据一样使用内置占位工具。

**3. OTEL 跟踪**——工具调用轮次来自文件：span 输出中的 assistant `tool_call` 部分，或 `gen_ai.response.finish_reasons` 中的 `tool_calls` / `tool_call` / `tool_use` / `function_call`。如果存在 `gen_ai.tool.definitions`，工具 schema 会从中读取；否则使用默认的合成工具。记录的工具结果字符串会被注入，并重新绑定到实时的 `tool_call_id`。在 `--data kind=otel` 上设置 `tool_choice=required`（默认值）或 `tool_choice=auto`。记录的消息采用 Chat Completions 字典格式；请通过 `/v1/chat/completions` 重放。如果使用其他请求格式，程序会中止，因为无法根据 `raw_messages_column` 构造 `prompt` / `input`。后端的 `tool_call_missing_behavior` 设置不变。参阅[跟踪重放](trace_replay.md#otel)。

**4. 含有 tools 列的数据集**——已有工具定义的数据集（例如 `madroid/glaive-function-calling-openai`）可以直接使用。列映射器会自动检测名为 `tools`、`functions` 或 `tool_definitions` 的列：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=huggingface,source=madroid/glaive-function-calling-openai \
  --data-column-mapper kind=generative_column_mapper,column_mappings.text_column=messages,column_mappings.tools_column=tools \
  --data-preprocessor kind=tool_calling_message_extractor \
  --data-preprocessor kind=encode_media \
  --constraint kind=max_requests,count=50 \
  --profile kind=constant,rate=1
```

必须通过 `--data-preprocessor` 显式启用 `tool_calling_message_extractor` 预处理器（默认不包含）。它会解析每行的 `messages` 数组，并将提示词、系统消息和工具结果提取到相应列中。如果数据集没有工具结果消息，则会使用占位内容（`{"status": "ok"}`）作为回退值。

## 工具选择和缺少工具调用时的行为

有两个后端设置会控制运行时如何处理工具调用轮次，二者都在 `--backend` 配置中设置：

| 设置                         | 取值                                           | 默认值       | 说明                                                                                                                                       |
| ---------------------------- | ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `extras.body.tool_choice`    | `required`、`auto`、`none`                     | `required`   | 在 `client_tool_call` 轮次中作为 `tool_choice` API 参数发送。当请求体中存在工具时，处理器会自动在非工具调用轮次设置 `tool_choice="none"`。 |
| `tool_call_missing_behavior` | `ignore_continue`、`ignore_stop`、`error_stop` | `error_stop` | 预期发生工具调用但模型返回普通文本时，后端的处理方式。                                                                                     |

**通过后端配置设置 `tool_choice`：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,extras.body.tool_choice=auto \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30 \
  --profile kind=synchronous
```

**通过后端配置设置 `tool_call_missing_behavior`：**

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,tool_call_missing_behavior=ignore_continue,extras.body.tool_choice=auto \
  --data kind=synthetic_text,prompt_tokens=200,output_tokens=100,turns=3,tool_call_turns=2 \
  --constraint kind=max_requests,count=30
```

**`tool_choice` 的影响：**

- `required`（默认值）：模型**必须**生成工具调用。服务器会将输出限制为有效的工具调用 JSON，因此基准测试最可预测、错误最少。若不希望依赖模型自行决定是否使用工具，请使用此选项。不过，由于服务器被迫选择概率较低的选项，速度可能会变慢。
- `auto`：由模型决定是否调用工具。适合测试模型选择调用工具的频率，但缺少工具调用的可能性也更高（参见 `tool_call_missing_behavior`）。
- `none`：请求中包含工具，但模型不能调用它们。数据流水线不会自动设置此值（非工具调用轮次会完全省略工具）；只有在后端配置中通过 extras 定义了全局 `tools` 时，显式设置它才有用。

请注意，`required` 和 `auto` 也可能导致模型行为不同。例如，Qwen 模型只有在 `auto` 模式下才会显示工具调用前的思考过程。

**`tool_call_missing_behavior` 的影响：**

此设置仅在 `tool_choice` 为 `auto` 时生效（或者 `tool_choice` 为 `required` 但服务器未强制执行时）：

- `error_stop`（默认值）：当前轮次标记为**出错**，并取消所有剩余轮次。这样可以立即发现问题，适合验证模型和服务器是否配置正确。
- `ignore_stop`：当前轮次标记为**已取消**（未完成），并取消所有剩余轮次。模型响应会保留在输出中，但轮次状态会表明模型未生成预期的工具调用。适用于缺少工具调用意味着对话无法有意义地继续，但又不应视为错误的情况。
- `ignore_continue`：当前轮次视为**已完成**，对话继续到下一轮。之后每个预期工具调用的轮次都会单独评估。适用于统计 `auto` 模式下实际发生的工具调用次数，同时不中止对话的情况。

### 推荐场景

| 工具选择   | 缺少调用时的行为  | 当前轮次状态 | 说明                                                                     |
| ---------- | ----------------- | ------------ | ------------------------------------------------------------------------ |
| `required` | `error_stop`      | errored      | （默认）适合使用合成数据获得一致且可预测的行为，但可能会降低服务器速度。 |
| `auto`     | `ignore_continue` | completed    | 适合测试 `auto` 模式，避免模型选择不调用工具而导致错误。                 |
| `auto`     | `ignore_stop`     | cancelled    | 适合测试 `auto` 模式；模型生成普通文本而非工具调用时，对话会提前结束。   |

## 工具调用轮次的输出 token 限制

配置 `output_tokens` 时（无论通过合成数据还是数据集列设置），GuideLLM 通常会设置 `ignore_eos=True` 并清除停止序列，以强制模型生成恰好 N 个 token。但在工具调用轮次中，系统会**自动移除**这些设置，因为它们与 vLLM 的受限解码语法不兼容：

- **`ignore_eos`** 与语法的终止状态冲突。受限解码通过有限状态机引导 token 选择：JSON 完成后，EOS 是唯一有效 token。`ignore_eos` 会屏蔽 EOS，导致状态中没有任何有效 token，从而引发服务器错误或生成失控。
- **`stop=None`** 会移除工具调用解析器可能依赖的停止序列（例如 Llama 模型中的 `<|eot_id|>`）。
- **`max_tokens` / `max_completion_tokens`** 可能会在 JSON 中途截断输出，产生无效的工具调用参数，并破坏后续轮次的对话历史。

因此，工具调用轮次的输出长度由模型和工具 schema 决定，而不受配置的 `output_tokens` 控制。模型在生成有效的函数名称和参数 JSON 后即会停止。与配置的目标相比，输出通常较短（20–80 个 token）。

普通文本轮次不受影响，仍会像往常一样遵循 `output_tokens` / `ignore_eos` 设置。这些轮次的请求中不包含工具定义或 `tool_choice`。

## 边界情况

- **单轮工具调用**（`turns=1, tool_call_turns=1` 或 `tool_call_turns=[0]`）受支持。对话只有一个预期调用工具的轮次，不会再有普通文本响应。
- **全工具调用对话**（例如 `turns=3` 且 `tool_call_turns=3`，或 `tool_call_turns=[0,1,2]`）受支持。每个轮次都会调用工具，模型不会生成最终的普通文本响应。`benchmarks.json` 中每个请求的 `output` 字段均为 `None`；请使用 `tool_calls` 字段检查模型输出。
- **不连续的工具调用轮次**（例如 `turns=4` 且 `tool_call_turns=[0, 2]`）受支持。只有指定轮次预期调用工具；其他轮次生成普通文本。
- **非工具调用轮次中的工具定义**不会被放入请求。数据流水线只会将 `tools_column` 附加到预期调用工具的轮次，因此其他轮次会作为不含工具或 `tool_choice` 的普通聊天补全请求发送。
- **混合数据集**（只有部分行包含 `tools_column`）可以正常工作。不含工具的行会按普通文本对话处理；含工具的行则按工具调用流程处理。
- **受速率限制的 profile**（例如 `--profile kind=constant,rate=1`）会通过与其他请求相同的调度器控制后续工具轮次的发送节奏。后续轮次会重新加入队列，并等待下一个可用的调度时隙，因此轮次之间的实际延迟由 profile 决定，而不是由工具调用逻辑决定。
