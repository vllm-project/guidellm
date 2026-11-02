---
weight: 15
source_sha256: !!str "19294b49d173bdf267cb1c483547a7e4e6ef18e914ec92a96c62a50fa02b14d5"
---

# 故障排查

在下方找到你的问题症状，然后按照链接查看相应修复方法。CLI 语法请参阅[运行基准测试](../getting-started/benchmark.md)。

| 症状                                                        | 章节                                                         |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| 请求失败或结果异常                                          | [调试日志](#debug-logging)                                   |
| 请求到达时间晚于预期，或 trace 未完全加载                   | [请求延迟到达](#requests-load-after-they-are-due)            |
| 加载模型 tokenizer 时出现自定义代码错误                     | [Tokenizer: trust_remote_code](#tokenizer-trust_remote_code) |
| `Worker process ... died unexpectedly (signal 11)`（macOS） | [macOS worker 崩溃](#macos-worker-crash-signal-11)           |

## 调试日志 {#debug-logging}

启用调试输出，以检查请求处理和 worker 启动过程：

```bash
GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG guidellm run ... --disable-progress
```

运行 `guidellm env`，确认设置已生效。`--disable-progress` 是可选参数，但交互式进度控制台可能会覆盖控制台日志消息。或者，也可以使用[开发指南的日志章节（英文）](https://github.com/vllm-project/guidellm/blob/main/DEVELOPING.md#logging)中提到的文件日志。

有关所有日志选项（文件输出、日志级别），请参阅[开发指南的日志章节（英文）](https://github.com/vllm-project/guidellm/blob/main/DEVELOPING.md#logging)。

## Tokenizer: trust_remote_code {#tokenizer-trust_remote_code}

### 症状

你可能会看到类似以下错误：

```text
The repository moonshotai/Kimi-K2.6 contains custom code which must be executed
to correctly load the model. You can inspect the repository content at
https://hf.co/moonshotai/Kimi-K2.6.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
```

### 修复方法

如果你完全信任该模型，可以通过 `--tokenizer` 的 `load_kwargs` 传入 `trust_remote_code`：

```bash
--tokenizer '{"kind":"huggingface_auto","load_kwargs":{"trust_remote_code":true}}'
```

如果你不信任该模型，请勿这样做，因为这会允许它在你的机器上执行代码。

其他 tokenizer 选项请参阅[数据集：Tokenizer](datasets.md#tokenizer)。

## 请求延迟到达 {#requests-load-after-they-are-due}

### 症状

基准测试警告称数据生成正在拖慢运行、对话加载过晚，或 trace 数据集未完全加载。

### 修复方法

在运行计时开始前，使用 `--profile` 上的 `prefetch` 参数加载请求图：

- `0`（默认值）会在运行过程中按需加载请求图。
- 指定数量会预先加载相应数量的请求节点。
- `start` 会匹配策略的启动并发数（如果该策略定义了启动并发数）。
- `all` 会从有限数据集中加载所有请求。如果数据源无限，且未通过 `--data-loader kind=pytorch,samples=<n>` 限定样本数，则会失败。

所有基准测试配置都支持 `prefetch` 参数。下面是将其用于 `concurrent` profile 的示例：

```bash
--profile kind=concurrent,prefetch=all
```

WEKA trace 应使用 `prefetch=all`。每个会话都有请求从 0 或接近 0 的时间开始，因此在运行期间加载请求会导致数据处理落后，进而使对话启动延迟。

## macOS worker 崩溃（signal 11） {#macos-worker-crash-signal-11}

### 症状

```text
RuntimeError: Worker process group startup failed: Worker process <pid> died
unexpectedly (signal 11). Check system logs for details
```

你也可能会看到重复出现的日志行，例如：

```text
Worker process <pid> died unexpectedly (signal 11)
```

### 修复方法

GuideLLM 默认使用 `fork` 多进程启动方式，这可能会在 macOS 上触发段错误。请改用 `spawn`：

```bash
GUIDELLM__MP_CONTEXT_TYPE=spawn guidellm run ...
```
