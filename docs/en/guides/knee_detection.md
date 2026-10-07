# Knee Profile and Adaptive Concurrency

GuideLLM's `knee` profile estimates where output throughput stops increasing substantially as concurrency rises. This transition is the throughput knee. The profile runs an initial set of concurrency points, reports the knee, and optionally selects additional points around it to refine the estimate.

Use `--profile` with `kind=knee` and either an `initial_streams` list or `min_streams`, `max_streams`, and `count` to enable knee detection. Adaptive refinement is disabled by default.

## Basic Usage

Measure a chosen set of concurrency points and report a knee:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile '{"kind":"knee","initial_streams":[1,5,10,20,40,80,160]}' \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --constraint kind=max_duration,seconds=60 \
  --output kind=json,path=knee-benchmark.json
```

To run additional concurrency points automatically, set the adaptive options on the same profile:

```bash
--profile '{"kind":"knee","initial_streams":[1,5,10,20,40,80,160],"adaptive":true,"points_each_side":5,"max_step":3}'
```

To generate an evenly spaced initial set instead, provide inclusive bounds and a point count:

```bash
--profile '{"kind":"knee","min_streams":1,"max_streams":9,"count":5}'
```

This example runs concurrency points `1, 3, 5, 7, 9`. Generated points are rounded to the nearest integer, with ties rounded upward. The bounds must contain enough integers to produce the requested number of distinct points. Use `initial_streams` when you need exact, unevenly spaced values.

The knee profile uses concurrent scheduling strategies and requires at least five distinct positive integer concurrency points in increasing order. At least five completed concurrency points with output-throughput measurements are required to fit a knee. Choose a range that covers both rising throughput and a plateau; a curve that is flat or approximately linear produces `status: no_knee`.

## Configuration

| Option             | Default | Description                                                                 |
| ------------------ | ------- | --------------------------------------------------------------------------- |
| `initial_streams`  | —       | Exact initial concurrency points, at least five in increasing order.        |
| `min_streams`      | —       | Lowest generated initial concurrency, inclusive.                            |
| `max_streams`      | —       | Highest generated initial concurrency, inclusive.                           |
| `count`            | —       | Number of generated points, at least five.                                  |
| `adaptive`         | `false` | Run additional concurrency points around the initial saturation estimate.   |
| `points_each_side` | `5`     | Maximum number of grid points selected below and above the adaptive anchor. |
| `max_step`         | `5`     | Largest integer spacing considered for the adaptive concurrency grid.       |

Set either `initial_streams` or all three of `min_streams`, `max_streams`, and `count` on the knee profile. Both forms run as one benchmark. `points_each_side` and `max_step` must be positive integers. All adaptive concurrency values are positive integers. GuideLLM removes concurrency points that were already measured before starting adaptive refinement. Standard profile timing options (`rampup_duration`, `warmup`, and `cooldown`) apply to both phases.

## How the Knee Is Calculated

GuideLLM uses successful output tokens per second as throughput and concurrency as load. It performs the following analysis:

1. Sort the measurements by concurrency and ignore measurements without throughput. The standalone fitting function averages duplicate concurrency measurements, but the knee profile requires distinct concurrency points because repeated saturation detector decisions can be ambiguous.
2. Normalize concurrency and throughput to comparable scales.
3. Fit one line to the complete throughput curve.
4. Try eligible breakpoints and fit a rising line before each breakpoint and a tail line after it.
5. Accept a breakpoint when the rising slope is positive, the tail slope is no more than 25% of the rising slope, the two-line fit reduces squared error by at least 50%, and breakpoint throughput is at least 85% of the observed peak.
6. Select the accepted fit with the lowest combined squared error and report the intersection of its two lines as the knee. If the intersection lies outside the measured points immediately neighboring the breakpoint, use the breakpoint itself.

The fitted intersection can be fractional even though concurrency is an integer. GuideLLM also reports `saturation_concurrency`, the first measured integer concurrency at or above the fitted knee, and `breakpoint_concurrency`, the measured point used by the best segmented fit.

## Adaptive Refinement

GuideLLM analyzes the initial benchmarks before selecting adaptive points. When a throughput knee is found, that fitted knee becomes the selection center. If there is no throughput knee but the over-saturation detector found a boundary, GuideLLM uses the midpoint between the last safe concurrency and the first over-saturated concurrency. If there is no safe point below the boundary, it uses the first over-saturated concurrency. Adaptive refinement is skipped when neither method finds saturation or when their results disagree.

The adaptive grid uses the largest step up to `max_step` that keeps the requested lower points positive. The selection center is rounded to the nearest multiple of that step to form the integer anchor, with ties rounding upward. If even a step of one cannot fit all lower points, non-positive candidates are omitted. The anchor is also a candidate, so a grid has up to `2 * points_each_side + 1` points before excluding measurements already taken.

For example, a selection center of `19.97` with `points_each_side=5` and `max_step=3` produces an anchor of `21` and this candidate grid:

```text
6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36
```

Use `max_step=1` for a denser grid: a center of `19.97` then produces anchor `20` and candidates from `15` through `25` with five points on each side.

Initial and adaptive points run through the same benchmark lifecycle, using the same backend, data loader, profile timing settings, and constraints. Duration limits apply to each point, so additional points increase total runtime. The grid can extend beyond the initial concurrency range. A constraint with `stopping_scope: all` halts the profile in either phase. If this happens during the initial sweep, adaptive refinement is skipped; if it happens during refinement, remaining planned points are not run.

The progress display reserves space for the initial sweep and up to `2 * points_each_side + 1` adaptive points. The profile may finish with fewer measurements when points are excluded, refinement is skipped, or a stopping constraint is triggered.

After the additional points finish, GuideLLM combines the initial and adaptive measurements and calculates the final knee. Adaptive selection happens once; the final knee does not start another refinement pass. If refinement is disabled or skipped, the final analysis uses only the initial measurements.

## Using Over-Saturation Detection

The throughput knee can be calculated without the over-saturation constraint. Adding the constraint provides independent temporal evidence based on concurrent requests and time to first token. Use `mode=monitor` when collecting the initial curve so detection metadata is recorded without stopping the remaining concurrency points:

```bash
--constraint kind=over_saturation,mode=monitor,min_seconds=30
```

See [Over-Saturation Stopping](over_saturation_stopping.md) for all detector settings.

## Results

Console output summarizes the final knee and adaptive refinement status. JSON and YAML reports store the full analysis in the standard profile `conclusions` list under an entry with `kind: knee_detection`. The entry contains:

- `initial`: Knee and over-saturation analysis from the configured concurrency points.
- `adaptive_plan`: Selection center, anchor, step, candidates, excluded points, and planned additional points. A stopping constraint can prevent some planned points from running.
- `final`: Analysis of the combined initial and adaptive measurements.

Initial and adaptive benchmark measurements are stored together in the report's `benchmarks` list, with the initial measurements first. These measurements show which points actually ran. The configuration is stored in `config.spec.profile`; each benchmark also records its profile and concurrent strategy.
