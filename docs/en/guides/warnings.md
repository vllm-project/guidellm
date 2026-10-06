# Warnings

After each benchmark, GuideLLM can report a warning when a metric, or the ratio of two metrics, crosses a threshold. The run still completes. Warnings are printed after the console tables and stored on each benchmark in the JSON report.

Rules live on the benchmark spec under `warnings`. A scenario file is the place to keep them. There is no `--warnings` flag.

## Rule fields

Each entry in `warnings.rules` has:

- `code`: stable identifier stored on the warning.
- `metric.name`: dotted path into the compiled scheduler metrics or generative metrics.
- `metric.statistic`: `mean`, `max`, `p95`, or `sum`. The default is `mean`.
- `relative_to`: optional second metric. When set, the observed value is `metric / relative_to` and `threshold` is a fraction. When omitted, the observed value is the metric itself and `threshold` is in that metric's units.
- `threshold`: warn when the observed value is greater than this.
- `scale`: multiply the statistic before comparing. The default is `1`. Use `0.001` when a millisecond metric is compared with a seconds metric.
- `enabled`: set to `false` to keep the rule in the file without reporting it.
- `note`: extra context printed under the measured sentence. Use it for what to change, or for a link.
- `when`: optional path and value. The rule runs only when that path equals the value. The comparison uses the value itself, so it can match text such as a strategy type.

A boolean metric compares as true or false. `True` is greater than a threshold of `0`, and `False` is not. The warning unit is `boolean`.

A status breakdown is not one distribution. `time_to_first_token_ms` has a separate summary for `successful`, `incomplete`, `errored`, and `total`. The path segment `.total` selects the summary that includes every status. `statistic: mean` then selects the mean of that summary. The two are independent, so the mean of time to first token is `time_to_first_token_ms.total` with `statistic: mean`. `request_latency.total` has the same shape. `generation_delay` is already one distribution, so its path has no status segment.

A path that is not on the compiled schemas produces an `unknown_metric` warning. A field that exists but is null is skipped. `root_dispatch_delay` covers only the first request of each conversation, the requests with no preceding nodes. It is null when none of those requests have a dispatch delay, which is the case for a profile with no arrival schedule.

## Common metric paths

- `generation_delay`: seconds spent waiting for the dataset to yield the next conversation. Scheduler metric.
- `request_latency.total`: request duration in seconds, across every status.
- `time_to_first_token_ms.total`: time to first token, in milliseconds.
- `time_to_first_output_token_ms.total`: time to the first content token, in milliseconds.
- `time_per_output_token_ms.total`: average time per output token, in milliseconds.
- `inter_token_latency_ms.total`: inter-token latency, in milliseconds.
- `request_dispatch_delay.total`: lateness against the scheduled start, in seconds. Null when the profile does not define an arrival schedule.
- `root_dispatch_delay`: lateness of the first request of each conversation, in seconds. Later turns are omitted. Null when no first turn has a dispatch delay.
- `dataset_incomplete`: `true` when the request iterator stopped before it was exhausted.
- `strategy_type`: strategy type identifier, such as `trace` or `concurrent`.

A trace run can warn when the dataset was not fully loaded, and stay quiet for every other strategy type:

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

## Scenario file

```yaml
spec:
  warnings:
    rules:
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

Load it with the rest of the scenario:

```bash
guidellm run \
  --config scenario.yaml \
  --backend kind=openai_http,target=http://localhost:8000
```

`spec.warnings` replaces the built-in rules. Include every rule you still want.

## Environment variable

`GUIDELLM__SPEC__WARNINGS` sets the same object for one run. The value is JSON, the same pattern as `GUIDELLM__SPEC__BACKEND`.

```bash
GUIDELLM__SPEC__WARNINGS='{"rules":[{"code":"generation_delay","metric":{"name":"generation_delay","statistic":"mean"},"relative_to":{"name":"request_latency.total","statistic":"mean"},"threshold":0.25,"note":"Dataset generation is a large share of request time."}]}' \
  guidellm run \
  --backend kind=openai_http,target=http://localhost:8000
```

An explicit scenario value overrides the environment variable when both set `warnings`.

## Where warnings appear

The console prints a group per strategy. Each warning starts with a warning icon, and the measured sentence and note sit in the column to the right of that icon:

```text
Warnings (constant@2.00)
⚠ mean generation delay was 40% of mean request latency.total (0.040s / 0.100s; threshold 25%).
  Dataset generation is a large share of request time.
```

The JSON report stores the same record on each benchmark under `warnings`, including `code`, `message`, `observed`, `threshold`, `unit`, `sample_count`, and `note`.
