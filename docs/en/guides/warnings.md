# Warnings

After each benchmark, GuideLLM can report a warning when a metric, or the ratio of two metrics, crosses a threshold. The run still completes. Warnings are printed after the console tables and stored on each benchmark in the JSON report.

Rules live on the generative metrics configuration, under `metrics.warnings`. Set them in a scenario file or with `--metrics`. Omitting the field keeps the built-in rules. Setting the field replaces those rules, so include every rule you still want.

## Built-in rules

These three rules run when `metrics.warnings` is omitted:

- `generation_delay_ttft`:
  - Warn when the mean `generation_delay` is more than 1% of mean \`time_to_first_token_ms.total
  - Means that the data generation is lagging by a great enough fraction of the server's TTFT that it's likely slowing down the benchmark. See [Requests load after they are due](troubleshooting.md#requests-load-after-they-are-due).
- `root_late`:
  - Warn when the 95th percentile of `root_dispatch_delay` is greater than 0.2 seconds.
  - This means that the first turn of conversations arrived later than the scheduler scheduled them. Often due to data lag. See [Requests load after they are due](troubleshooting.md#requests-load-after-they-are-due).
- `dataset_incomplete`:
  - Warn when `dataset_incomplete` is true during trace strategies.
  - This means that the trace dataset was not fully loaded, which can result in late and missing arrivals of trace conversations, as many trace conversations have turns that are supposed to start at the beginning of the benchmark. See [Requests load after they are due](troubleshooting.md#requests-load-after-they-are-due).

See `default_warning_rules()` in `src/guidellm/schemas/benchmark/warnings.py` for where the defaults are set.

## Rule fields

Each entry in `metrics.warnings` has:

- `code`: free-form tag stored on the warning. Any string is accepted. Short snake_case tags are the convention, and the tag does not have to match the metric name. Intended for easy identification in the machine readable output.
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

A path that is not on the compiled schemas is logged and printed with the other warnings after the progress display finishes. A field that exists but is null is skipped. `root_dispatch_delay` covers only the first request of each conversation. It is null when none of those requests have a dispatch delay, which is the case for a profile with no arrival schedule.

## Common metric paths

- `generation_delay`: seconds spent waiting for the dataset to yield the next conversation. Scheduler metric.
- `request_latency.total`: request duration in seconds, across every status.
- `time_to_first_token_ms.total`: time to first token, in milliseconds.
- `time_to_first_output_token_ms.total`: time to the first content token, in milliseconds.
- `time_per_output_token_ms.total`: average time per output token, in milliseconds.
- `inter_token_latency_ms.total`: inter-token latency, in milliseconds.
- `request_dispatch_delay.total`: lateness against the scheduled start, in seconds. Null when the profile does not define an arrival schedule.
- `root_dispatch_delay`: lateness of the first request of each conversation, in seconds. Later turns are omitted. Null when no first turn has a dispatch delay.
- `dataset_incomplete`: `true` when the scheduler stops queueing before the request iterator is exhausted. The scheduler records that the iterator finished only after the loader has yielded every conversation. A `max_requests` or `max_duration` stop therefore marks the dataset incomplete, and the built-in trace rule can fire on a run that was meant to stop early. `data_loader.samples` caps the iterator itself, so a run that consumes every yielded sample looks complete and does not fire.
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

Load it with the rest of the scenario:

```bash
guidellm run \
  --config scenario.yaml \
  --backend kind=openai_http,target=http://localhost:8000
```

`metrics.warnings` replaces the [built-in rules](#built-in-rules). Include every rule you still want.

The same list can be passed on the command line. A JSON object is the practical form, the same way a service-level objective is passed:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256  --metrics '{"kind":"generative","warnings":[{"code":"slow_ttft","metric":{"name":"time_to_first_token_ms.total"},"threshold":250,"note":"Time to first token is above 250 ms."}]}'
```

## Environment variable

`GUIDELLM__SPEC__METRICS__WARNINGS` sets the same list for one run. The value is a JSON list.

You must also set `GUIDELLM__SPEC__METRICS__KIND=generative` for this to work.

```bash
GUIDELLM__SPEC__METRICS__KIND=generative GUIDELLM__SPEC__METRICS__WARNINGS='[{"code":"generation_delay","metric":{"name":"generation_delay","statistic":"mean"},"relative_to":{"name":"request_latency.total","statistic":"mean"},"threshold":0.25,"note":"Dataset generation is a large share of request time."}]' \
  guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=5 \
  --constraint kind=max_duration,seconds=20 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=256
```

An explicit `metrics.warnings` value from the scenario file or `--metrics` overrides the environment variable. Omitting `warnings` keeps the built-in rules.

## Where warnings appear

The console prints a group per strategy. Each warning starts with a warning icon, and the measured sentence and note sit in the column to the right of that icon:

```text
Warnings (constant@2.00)
⚠ mean generation delay was 40% of mean request latency.total (0.040s / 0.100s; threshold 25%).
  Dataset generation is a large share of request time.
```

The JSON report stores the same record on each benchmark under `warnings`, including `code`, `message`, `observed`, `threshold`, `unit`, `sample_count`, and `note`.
