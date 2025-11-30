# Over-Saturation Stopping

GuideLLM provides over-saturation detection (OSD) to automatically stop benchmarks when a model becomes over-saturated. This feature helps prevent wasted compute resources and ensures that benchmark results remain valid by detecting when the response rate can no longer keep up with the request rate.

## What is Over-Saturation?

Over-saturation occurs when an LLM inference server receives requests faster than it can process them, causing a queue to build up. As the queue grows, the server takes progressively longer to start handling each request, leading to degraded performance metrics. When a performance benchmarking tool oversaturates an LLM inference server, the metrics it measures become significantly skewed, rendering them useless.

Think of it like a cashier getting flustered during a sudden rush. As the line grows (the load), the cashier can't keep up, the line gets longer, and there is no room for additional customers. This waste of costly machine time can be prevented by automatically detecting and stopping benchmarks when over-saturation is detected.

## How It Works

GuideLLM's Over-Saturation Detection (OSD) algorithm uses statistical slope detection to identify when a model becomes over-saturated. The algorithm tracks two key metrics over time:

1. **Concurrent Requests**: The number of requests being processed simultaneously
2. **Time-to-First-Token (TTFT)**: The latency for the first token of each response

For each metric, the algorithm:
- Maintains a sliding window of recent data points
- Calculates the linear regression slope using online statistics
- Computes the margin of error (MOE) using t-distribution confidence intervals
- Detects positive slopes with low MOE, indicating degradation

Over-saturation is detected when:
- Both concurrent requests and TTFT show statistically significant positive slopes
- The minimum duration threshold has been met
- Sufficient data points are available for reliable slope estimation

When over-saturation is detected, the constraint automatically stops request queuing and optionally stops processing of existing requests, preventing further resource waste.

## Usage

### Basic Usage

Enable over-saturation detection with default settings:

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --profile throughput \
  --rate 10 \
  --over-saturation True
```

### Advanced Configuration

Configure detection parameters using a JSON dictionary:

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --profile concurrent \
  --rate 16 \
  --over-saturation '{"enabled": true, "min_seconds": 60, "max_window_seconds": 300, "moe_threshold": 1.5}'
```

### Using the Alias

You can also use the `--detect-saturation` alias:

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --profile throughput \
  --rate 10 \
  --detect-saturation True
```

## Configuration Options

The following parameters can be configured when enabling over-saturation detection:

- **`enabled`** (bool, default: `True`): Whether to stop the benchmark if over-saturation is detected
- **`min_seconds`** (float, default: `30.0`): Minimum seconds before checking for over-saturation. This prevents false positives during the initial warm-up phase.
- **`max_window_seconds`** (float, default: `120.0`): Maximum time window in seconds for data retention. Older data points are automatically pruned to maintain bounded memory usage.
- **`moe_threshold`** (float, default: `2.0`): Margin of error threshold for slope detection. Lower values make detection more sensitive to degradation.
- **`minimum_ttft`** (float, default: `2.5`): Minimum TTFT threshold in seconds for violation counting. Only TTFT values above this threshold are counted as violations.
- **`maximum_window_ratio`** (float, default: `0.75`): Maximum window size as a ratio of total requests. Limits memory usage by capping the number of tracked requests.
- **`minimum_window_size`** (int, default: `5`): Minimum data points required for slope estimation. Ensures statistical reliability before making detection decisions.
- **`confidence`** (float, default: `0.95`): Statistical confidence level for t-distribution calculations (0-1). Higher values require stronger evidence before detecting over-saturation.

## Use Cases

Over-saturation detection is particularly useful in the following scenarios:

### Stress Testing and Capacity Planning

When testing how your system handles increasing load, over-saturation detection automatically stops benchmarks once the system can no longer keep up, preventing wasted compute time on invalid results.

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --profile sweep \
  --rate 5 \
  --over-saturation True
```

### Cost-Effective Benchmarking

When running large-scale benchmark matrices across multiple models, GPUs, and configurations, over-saturation detection can significantly reduce costs by stopping invalid runs early.

### Finding Safe Operating Ranges

Use over-saturation detection to identify the maximum sustainable throughput for your deployment, helping you set appropriate rate limits and capacity planning targets.

## Interpreting Results

When over-saturation detection is enabled, the benchmark output includes metadata about the detection state. This metadata is available in the scheduler action metadata and includes:

- **`is_over_saturated`** (bool): Whether over-saturation was detected at the time of evaluation
- **`concurrent_slope`** (float): The calculated slope for concurrent requests
- **`concurrent_slope_moe`** (float): The margin of error for the concurrent requests slope
- **`concurrent_n`** (int): The number of data points used for concurrent requests slope calculation
- **`ttft_slope`** (float): The calculated slope for TTFT
- **`ttft_slope_moe`** (float): The margin of error for the TTFT slope
- **`ttft_n`** (int): The number of data points used for TTFT slope calculation
- **`ttft_violations`** (int): The count of TTFT values exceeding the minimum threshold

These metrics can help you understand why over-saturation was detected and fine-tune the detection parameters if needed.

## Example: Complete Benchmark with Over-Saturation Detection

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --profile concurrent \
  --rate 16 \
  --data "prompt_tokens=256,output_tokens=128" \
  --max-seconds 300 \
  --over-saturation '{"enabled": true, "min_seconds": 30, "max_window_seconds": 120}' \
  --outputs json,html
```

This example:
- Runs a concurrent benchmark with 16 simultaneous requests
- Uses synthetic data with 256 prompt tokens and 128 output tokens
- Enables over-saturation detection with custom timing parameters
- Sets a maximum duration of 300 seconds (as a fallback)
- Outputs results in both JSON and HTML formats

## Additional Resources

For more in-depth information about over-saturation detection, including the algorithm development, evaluation metrics, and implementation details, see the following Red Hat Developer blog posts:

- [Reduce LLM benchmarking costs with oversaturation detection](https://developers.redhat.com/articles/2025/11/18/reduce-llm-benchmarking-costs-oversaturation-detection) - An introduction to the problem of over-saturation and why it matters for LLM benchmarking
- [Defining success: Evaluation metrics and data augmentation for oversaturation detection](https://developers.redhat.com/articles/2025/11/20/oversaturation-detection-evaluation-metrics) - How to evaluate the performance of an OSD algorithm through custom metrics, dataset labeling, and load augmentation techniques
- [Building an oversaturation detector with iterative error analysis](https://developers.redhat.com/articles/2025/11/24/building-oversaturation-detector-iterative-error-analysis) - A detailed walkthrough of how the OSD algorithm was built
