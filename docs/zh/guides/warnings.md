---
source_sha256: !!str "42e4062e74a51707176ea6347a362c2e480d11c215aa3d13cc821c9d21a9045f"
---

# 警告 {#warnings}

每次基准测试结束后，如果某个指标或两个指标的比值超过阈值，GuideLLM 都可以报告警告。运行仍会完成。警告会在控制台表格后打印，并存储在 JSON 报告中每个基准测试对应的记录里。

规则位于生成式指标配置的 `metrics.warnings` 下。可以在场景文件中设置，也可以通过 `--metrics` 设置。省略该字段时会保留内置规则；设置该字段则会替换内置规则，因此请包含所有仍需使用的规则。

## 内置规则 {#built-in-rules}

省略 `metrics.warnings` 时会运行以下三条规则：

- `generation_delay_ttft`：
  - 当 `generation_delay` 的平均值超过 `time_to_first_token_ms.total` 平均值的 1% 时发出警告。
  - 这表示数据生成的延迟相对于服务器的 TTFT 已达到足以拖慢基准测试的程度。请参阅[请求延迟到达](troubleshooting.md#requests-load-after-they-are-due)。
- `root_late`：
  - 当 `root_dispatch_delay` 的第 95 百分位数大于 0.75 秒，或平均值大于 0.25 秒时发出警告。
  - 这表示对话的第一轮到达时间晚于调度器安排的时间，通常是数据延迟造成的。请参阅[请求延迟到达](troubleshooting.md#requests-load-after-they-are-due)。
- `dataset_incomplete`：
  - 在 trace 策略运行期间，如果 `dataset_incomplete` 为 true，则发出警告。
  - 这表示 trace 数据集未完全加载，因此 trace 对话可能迟到或缺失，因为许多 trace 对话的轮次原本应在基准测试开始时启动。请参阅[请求延迟到达](troubleshooting.md#requests-load-after-they-are-due)。

默认规则的定义位置见 `src/guidellm/schemas/benchmark/warnings.py` 中的 `default_warning_rules()`。

## 规则字段 {#rule-fields}

`metrics.warnings` 中的每一项都包含以下字段：

- `code`：存储在警告中的自由格式标签。任何字符串均可使用。通常使用简短的 snake_case 标签，且标签不必与指标名称一致。该字段便于在机器可读输出中识别警告。
- `metric.name`：指向已编译调度器指标或生成式指标的点分路径。
- `metric.statistic`：`mean`、`max`、`p95` 或 `sum`。默认值为 `mean`。
- `relative_to`：可选的第二个指标。设置后，观测值为 `metric / relative_to`，且 `threshold` 表示比例；省略时，观测值就是指标本身，`threshold` 使用该指标的单位。
- `threshold`：当观测值大于此值时发出警告。
- `scale`：比较前对统计值进行乘法缩放。默认值为 `1`。将毫秒指标与秒指标比较时使用 `0.001`。
- `enabled`：设为 `false` 后，可在文件中保留该规则但不报告它。
- `note`：在测量结果说明下方打印的补充信息。可用于提示需要更改的内容或提供链接。
- `when`：可选的路径和值。仅当该路径等于指定值时才运行该规则。比较使用该值本身，因此可以匹配策略类型之类的文本。

布尔指标按 true 或 false 进行比较。`True` 大于阈值 `0`，`False` 则不大于。警告单位为 `boolean`。

状态细分并非单一分布。`time_to_first_token_ms` 会分别为 `successful`、`incomplete`、`errored` 和 `total` 提供摘要。路径片段 `.total` 选择包含所有状态的摘要；随后 `statistic: mean` 选择该摘要的平均值。二者相互独立，因此首 token 延迟的平均值由 `time_to_first_token_ms.total` 和 `statistic: mean` 指定。`request_latency.total` 也采用相同结构。`generation_delay` 已经是单一分布，因此其路径不包含状态片段。

如果路径不在已编译的 schema 中，系统会记录该路径，并在进度显示结束后与其他警告一起打印。字段存在但值为 null 时会跳过。`root_dispatch_delay` 只统计每段对话的第一个请求。如果这些请求都没有调度延迟，该值为 null；不定义到达时间表的 profile 就会出现这种情况。

## 常用指标路径 {#common-metric-paths}

- `generation_delay`：等待数据集提供下一段对话所花费的秒数。调度器指标。
- `request_latency.total`：所有状态下的请求时长，单位为秒。
- `time_to_first_token_ms.total`：首 token 延迟，单位为毫秒。
- `time_to_first_output_token_ms.total`：首个内容 token 的延迟，单位为毫秒。
- `time_per_output_token_ms.total`：每个输出 token 的平均耗时，单位为毫秒。
- `inter_token_latency_ms.total`：token 间延迟，单位为毫秒。
- `request_dispatch_delay.total`：请求相对于预定启动时间的延迟，单位为秒。如果 profile 未定义到达时间表，则值为 null。
- `root_dispatch_delay`：每段对话的第一个请求的延迟，单位为秒；不包括后续轮次。如果没有首轮请求具有调度延迟，则值为 null。
- `dataset_incomplete`：当调度器在请求迭代器耗尽前停止入队时为 `true`。只有加载器交付所有对话后，调度器才会记录迭代器已完成。因此，`max_requests` 或 `max_duration` 停止会将数据集标记为未完整加载；如果运行原本就要提前停止，内置 trace 规则也可能触发。`data_loader.samples` 会限制迭代器本身的样本数；如果运行消耗了所有已交付的样本，数据集会被视为完整，规则不会触发。
- `strategy_type`：策略类型标识符，例如 `trace` 或 `concurrent`。

trace 运行可以在数据集未完全加载时发出警告，同时对其他所有策略类型保持安静：

```yaml
- code: dataset_incomplete
  metric:
    name: dataset_incomplete
  threshold: 0
  when:
    name: strategy_type
    equals: trace
  note: The trace dataset was not fully loaded.
```

## 场景文件 {#scenario-file}

```yaml
spec:
  metrics:
    kind: generative
    warnings:
      - code: generation_delay
        metric:
          name: generation_delay
          statistic: mean
        relative_to:
          name: request_latency.total
          statistic: mean
        threshold: 0.25
        note: >-
          Dataset generation is a large share of request time.
          https://github.com/vllm-project/guidellm/blob/main/docs/en/guides/warnings.md
      - code: slow_ttft
        metric:
          name: time_to_first_token_ms.total
          statistic: mean
        threshold: 250
        note: Time to first token is above 250 ms.
```

将其与场景的其他内容一起加载：

```bash
guidellm run \
  --config scenario.yaml \
  --backend kind=openai_http,target=http://localhost:8000
```

`metrics.warnings` 会替换[内置规则](#built-in-rules)。请包含所有仍需使用的规则。

也可以通过命令行传入相同的列表。与传入服务级目标（service-level objective）的方式相同，实际使用时可以采用 JSON 对象：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256  --metrics '{"kind":"generative","warnings":[{"code":"slow_ttft","metric":{"name":"time_to_first_token_ms.total"},"threshold":250,"note":"Time to first token is above 250 ms."}]}'
```

## 环境变量 {#environment-variable}

`GUIDELLM__SPEC__METRICS__WARNINGS` 用于为单次运行设置相同的规则列表。该值是一个 JSON 列表。

要使其生效，还必须设置 `GUIDELLM__SPEC__METRICS__KIND=generative`。

```bash
GUIDELLM__SPEC__METRICS__KIND=generative GUIDELLM__SPEC__METRICS__WARNINGS='[{"code":"generation_delay","metric":{"name":"generation_delay","statistic":"mean"},"relative_to":{"name":"request_latency.total","statistic":"mean"},"threshold":0.25,"note":"Dataset generation is a large share of request time."}]' \
  guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256
```

场景文件或 `--metrics` 中明确设置的 `metrics.warnings` 会覆盖环境变量。省略 `warnings` 时会保留内置规则。

## 警告的显示位置 {#where-warnings-appear}

控制台会按策略分组打印警告。每条警告以警告图标开头，测量结果说明和备注显示在图标右侧的列中：

```text
Warnings (constant@2.00)
⚠ mean generation delay was 40% of mean request latency.total (0.040s / 0.100s; threshold 25%).
  Dataset generation is a large share of request time.
```

JSON 报告会在每个基准测试的 `warnings` 字段下保存相同记录，包括 `code`、`message`、`observed`、`threshold`、`unit`、`sample_count` 和 `note`。

HTML 报告会在页眉下方以卡片显示每条警告。卡片会显示消息；如果设置了备注，也会显示备注。报告包含多次运行时，卡片还会标注对应运行的策略。

CSV 报告会将警告列表存储在单个 `Warnings` 列中。单元格内容是相同记录组成的 JSON 列表；如果该次运行没有警告，则为 `[]`。
