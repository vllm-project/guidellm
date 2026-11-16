---
source_sha256: !!str "3173be306b7a6b99f40a838280668c97666d50e655bef1c97ba2b994bdf323d0"
---

# vLLM Python 批处理后端 {#vllm-python-batch-backend}

**vLLM Python 批处理后端**（`vllm_python_batch`）使用 vLLM 的同步 [LLM](https://docs.vllm.ai/) 引擎，在与 GuideLLM 相同的进程中运行批量推理。请求会排队，并通过 `LLM.generate()` 按可配置的批次分发，从而消除逐请求调度开销。这适用于吞吐量基准测试，因为此时单个请求的延迟不如总体吞吐量重要。

与 `vllm_python_async` 后端一样，此后端不需要 HTTP 服务器。你**无需**传入 `target`；必须在后端配置中传入 `model`。

有关所有引擎选项和支持的模型，请参阅 vLLM 的 [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) 和 [vLLM 文档](https://docs.vllm.ai/)。

## 安装 {#installation}

安装方式与 [vLLM Python 后端](vllm-python-backend.md#installation)相同。批处理后端使用相同的 vLLM 软件包。

## 基本示例 {#basic-example}

使用 vLLM Python 批处理后端运行基准测试：

```bash
guidellm run \
  --backend kind=vllm_python_batch,model=Qwen/Qwen3-0.6B,batch_size=8 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --profile kind=throughput,max_concurrency=20 \
  --constraint kind=max_requests,count=100
```

## 异步与批处理后端对比 {#async-vs-batch-backend}

| 特性     | `vllm_python_async`      | `vllm_python_batch`   |
| -------- | ------------------------ | --------------------- |
| 引擎     | `AsyncLLMEngine`（异步） | `LLM`（同步、批处理） |
| 流式传输 | 支持                     | 不支持                |
| 批处理   | 按请求异步调度           | 可配置的微批次        |
| 适用场景 | 延迟分析、流式传输       | 吞吐量基准测试        |

## 后端选项 {#backend-options}

- **`batch_size`**（默认值：`32`）\
  单次 `LLM.generate()` 调用前累积的最大请求数。达到该批次大小时会立即分发。未满的批次（请求数少于 `batch_size`）会在 `batch_timeout` 秒后刷新。因此，有效批次大小为 `min(batch_size, requests arriving within the timeout window)`。较大的值可以分摊引擎开销，但会增加单个请求的延迟。

- **`batch_timeout`**（默认值：`0.01`）\
  刷新未满批次前等待更多请求的秒数。已满的批次不会受到此延迟影响。如果更高的并发量能让每批次累积更多请求，可以增大此值；对于对延迟敏感的工作负载，可以减小此值（或保留默认值）。

- **`model`**（必填）\
  要由 vLLM 加载的 Hugging Face 模型标识符或文件系统路径。

- **`request_format`**\
  控制如何构建聊天提示词。选项与 `vllm_python_async` 相同：`plain`、`default-template`，或 Jinja2 模板路径/字符串。

- **`vllm_config`**\
  引擎选项以嵌套 dict 的形式传入。使用 vLLM 的 `EngineArgs` 参数名称（Python 形式，而非 CLI 形式）。有关 `vllm_config` 的详情，请参阅 [vLLM Python 后端文档](vllm-python-backend.md#request-format-and-backend-options)。

  JSON 示例：

  ```bash
  --backend '{"kind":"vllm_python_batch","model":"Qwen/Qwen3-0.6B","batch_size":64,"vllm_config":{"gpu_memory_utilization":0.8,"max_model_len":4096}}'
  ```

> [!IMPORTANT]
>
> `vllm_python_batch` 的后端配置必须包含 `model` 字段。如果 `vllm_config` 中也设置了 `model`，则优先使用顶层的 `model` 字段。

## 引擎生命周期 {#engine-lifecycle}

vLLM `LLM` 引擎不会在 `process_startup()` 期间加载。引擎的创建由 worker 进程检查控制，该检查用于区分主（预检）进程和调度器 worker：

- **主预检进程**（`resolve_backend`）：`validate()` 在主进程中运行（`multiprocessing.parent_process()` 为 `None`），且只执行开销很低的就绪状态检查，不会加载模型权重。
- **Worker 进程**：当 `multiprocessing.parent_process()` 已设置时（`fork` 和 `spawn` worker 都满足此条件；参见 `GUIDELLM__MP_CONTEXT_TYPE`），`validate()` 会调用 `_ensure_engine()`，以便**预加载**引擎，将冷启动时间排除在计时的基准测试阶段之外。
- **推理时的安全兜底**：从数据集生成请求时，会调用 `_ensure_engine()` 作为幂等的后备处理，因此即使跳过了 `validate()`，推理仍可正常运行。

## 另请参阅 {#see-also}

- [vLLM Python 异步后端](vllm-python-backend.md) — 逐请求异步后端。
- [后端](backends.md) — 支持的后端概览。
- [运行基准测试](../getting-started/benchmark.md) — 基准测试通用选项。
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) — 面向 CLI 的文档；在 `vllm_config` 中请使用 Python 名称。
- [vLLM 文档](https://docs.vllm.ai/)
