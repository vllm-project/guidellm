# Knee Detection and Adaptive Concurrency

GuideLLM can estimate where output throughput stops increasing substantially as concurrency rises. This transition is the throughput knee. GuideLLM can report the knee after a concurrent benchmark and optionally run additional concurrency points around it to refine the estimate.

Knee detection is disabled by default. It runs only when `enabled=true` is set with `--knee-detection` or in a scenario file.

## Basic Usage

Calculate and report a knee from an existing set of concurrency points:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1 \
  --profile '{"kind":"concurrent","streams":[1,5,10,20,40,80,160]}' \
  --data kind=synthetic_text,prompt_tokens=1000,output_tokens=1000 \
  --constraint kind=max_duration,seconds=60 \
  --knee-detection enabled=true \
  --output kind=json,path=knee-benchmark.json
```

Enable adaptive refinement to run additional concurrency points automatically:

```bash
--knee-detection enabled=true,adaptive=true,points_each_side=5,max_step=3
```

Knee detection requires a `concurrent` profile with distinct stream counts. At least five completed concurrency points with output-throughput measurements are required to fit a knee. Choose a range that covers both rising throughput and a plateau; a curve that is flat or approximately linear produces `status: no_knee`.

## Configuration

| Option             | Default | Description                                                                 |
| ------------------ | ------- | --------------------------------------------------------------------------- |
| `enabled`          | `false` | Calculate and report the throughput knee.                                   |
| `adaptive`         | `false` | Run additional concurrency points around the initial saturation estimate.   |
| `points_each_side` | `5`     | Maximum number of grid points selected below and above the adaptive anchor. |
| `max_step`         | `5`     | Largest integer spacing considered for the adaptive concurrency grid.       |

`adaptive=true` requires `enabled=true`. `points_each_side` and `max_step` must be positive integers. All adaptive concurrency values are positive integers. GuideLLM removes concurrency points that were already measured before starting the adaptive run.

## How the Knee Is Calculated

GuideLLM uses successful output tokens per second as throughput and concurrent streams as load. It performs the following analysis:

1. Sort the measurements by concurrency and ignore measurements without throughput. The standalone fitting function averages duplicate concurrency measurements, but the benchmark CLI requires distinct stream counts because repeated saturation detector decisions can be ambiguous.
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

Each adaptive point uses the same backend, data loader, profile timing settings, and constraints as the initial run. Duration limits apply to each point, so additional points increase total runtime. The grid can extend beyond the initial concurrency range. Adaptive refinement is skipped if an initial benchmark triggers a constraint with `stopping_scope: all`.

After the additional points finish, GuideLLM combines the initial and adaptive measurements and calculates the final knee. Adaptive selection happens once; the final knee does not start another refinement pass. If refinement is disabled or skipped, the final analysis uses only the initial measurements.

## Using Over-Saturation Detection

The throughput knee can be calculated without the over-saturation constraint. Adding the constraint provides independent temporal evidence based on concurrent requests and time to first token. Use `mode=monitor` when collecting the initial curve so detection metadata is recorded without stopping the remaining concurrency points:

```bash
--constraint kind=over_saturation,mode=monitor,min_seconds=30
```

See [Over-Saturation Stopping](over_saturation_stopping.md) for all detector settings.

## Scenario File

The complete [knee detection scenario](../examples/knee-detection.yaml) can be run from a GuideLLM checkout:

```bash
guidellm run --config docs/examples/knee-detection.yaml
```

Change `spec.backend.target`, the initial `spec.profile.streams`, dataset sizes, and duration for the deployment being tested. The server must be running, and its model tokenizer must be available to GuideLLM; set `--tokenizer kind=huggingface_auto,model=<tokenizer-name-or-path>` if needed.

`knee_detection` is a top-level scenario field alongside `spec`, because its analysis spans the concurrency points. CLI settings override the YAML values. To run this scenario with reporting only, add `--knee-detection adaptive=false`. To disable the feature in this example, add `--knee-detection enabled=false,adaptive=false`.

## Results

The console reports the completed knee analysis. JSON and YAML reports store the same information in the `conclusions` list under an entry with `kind: knee_detection`. The entry contains:

- `initial`: Knee and over-saturation analysis from the configured concurrency points.
- `adaptive_plan`: Selection center, anchor, step, candidates, excluded points, and points launched.
- `final`: Analysis of the combined initial and adaptive measurements.

Initial and adaptive benchmark measurements are stored together in the report's `benchmarks` list, with the initial measurements first. The configuration is stored in `config.knee_detection`. Output paths are relative to the working directory unless an absolute path is specified; the example writes `knee-benchmark.json` there. CSV and HTML retain their existing benchmark summaries and do not yet have dedicated knee summaries.

When knee detection is omitted or `enabled=false`, GuideLLM does not perform this analysis or launch adaptive concurrency points.
