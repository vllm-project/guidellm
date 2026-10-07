# Target Margin of Error Stopping

A fixed duration or request count decides how long a benchmark runs, but not how precisely its results are measured. A short run reports a p95 with the same confidence as a long one, even when it rests on a few dozen requests. The `target_moe` constraint stops a benchmark once a chosen statistic has been measured to a requested relative precision, so each run lasts as long as that precision needs.

## How It Works

For every successfully completed request, the constraint records one value of the configured metric, computed the same way as in the benchmark report. Once `min_samples` values are collected, and then every `check_interval` new values, it computes a confidence interval for the configured statistic using the same estimators the report uses for its confidence intervals:

- **Mean**: a Student's t interval.
- **Percentiles**: a distribution-free interval whose bounds are themselves observations.

The relative margin of error is the larger distance from the estimate to either bound, divided by the estimate. When it is at or below `moe`, the constraint stops request queuing and processing for the current strategy. Because the estimators are shared, the margin a run stops at is the margin its report shows for the same requests.

A percentile cannot be bounded at all until the sample is large enough: at 95% confidence that takes 72 requests for p95, 368 for p99 and 3688 for p999. The constraint does not check a percentile before that point.

While the run is in progress, the constraint estimates how many samples the target still needs. The interval width shrinks with the square root of the sample count, so the estimate scales the current count by the squared ratio of the current margin to the target.

## Usage

```bash
--constraint kind=target_moe,metric=time_to_first_token_ms,statistic=p95,moe=0.05
```

Or with JSON:

```bash
--constraint '{"kind":"target_moe","metric":"time_to_first_token_ms","statistic":"p95","moe":0.05}'
```

A target that the run cannot reach never stops it, so combine `target_moe` with a duration or request limit:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=constant,rate=10 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=target_moe,statistic=p95,moe=0.05 \
  --constraint kind=max_duration,seconds=600 \
  --output kind=json,path=benchmark.json
```

Each strategy in a profile gets its own constraint, so every rate in a sweep collects its own samples and stops at its own precision.

## Configuration Options

- **`moe`** (float, required): Target relative margin of error, strictly between 0 and 1. For example, `0.05` stops once the statistic is known to within 5% of its estimate.
- **`metric`** (`"time_to_first_token_ms"` | `"request_latency"`, default: `"time_to_first_token_ms"`): Request-level metric to measure, named as in the benchmark report.
- **`statistic`** (`"mean"` | `"p25"` | `"p50"` | `"p75"` | `"p90"` | `"p95"` | `"p99"` | `"p999"`, default: `"mean"`): Statistic of the metric whose margin of error is targeted.
- **`confidence`** (float, default: `0.95`): Two-sided confidence level of the interval, between 0.5 and 0.999.
- **`min_samples`** (int, default: `30`): Minimum successful requests before the margin of error is checked.
- **`check_interval`** (int, default: `10`): Number of new successful requests between checks.
- **`stopping_scope`** (`"current"` | `"all"`, default: `"current"`): Whether reaching the target stops only the current benchmark or also escalation to subsequent rates or streams.

## Interpreting Results

When the target is reached, the constraint appears under `end_processing_constraints` in the scheduler state of the benchmark report, with this metadata:

- **`target_moe_reached`** (bool): Whether the target margin of error was reached
- **`samples`** (int): Number of values collected
- **`estimate`** (float): Point estimate of the statistic at the last check
- **`lower`** / **`upper`** (float): Confidence bounds at the last check
- **`relative_moe`** (float): Relative margin of error at the last check
- **`required_samples`** (int): Estimated number of samples needed to reach the target

## Limitations

- **Precision within a run, not across runs.** The interval treats requests as independent draws. Requests sent under load share queue state, so the interval describes how precisely a run located its own statistic, and is likely narrower than the variation between repeated runs. Reaching the target does not mean that a second run would report the same value within the same margin.
- **Stopping on the data.** Stopping the first time an interval looks narrow enough favours moments when the sample spread happens to be low, so the final interval covers the true value less often than its nominal level. `min_samples` and `check_interval` reduce this effect by avoiding early stops and repeated checks, but do not remove it.
- **Warmup and cooldown.** The constraint counts every completed request in the run, while the report summarizes only requests inside the measured window. With warmup or cooldown configured, the values the constraint stopped on can differ from the reported ones.
- **Supported metrics.** Only time to first token and request latency are supported. Token-weighted metrics such as inter-token latency have no confidence interval in the report.
