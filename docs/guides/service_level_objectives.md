# Service Level Objectives

Service Level Objectives (SLOs) and Service Level Agreements (SLAs) are critical for ensuring the quality and reliability of large language model (LLM) deployments. They define measurable performance and reliability targets that a system must meet to satisfy user expectations and business requirements. Below, we outline the key concepts, tradeoffs, and examples of SLOs/SLAs for various LLM use cases.

## Definitions

### Service Level Objectives (SLOs)

SLOs are internal performance and reliability targets that guide the operation and optimization of a system. They are typically defined as measurable metrics, such as latency, throughput, or error rates, and serve as benchmarks for evaluating system performance.

### Service Level Agreements (SLAs)

SLAs are formal agreements between a service provider and its users or customers. They specify the performance and reliability guarantees that the provider commits to delivering. SLAs often include penalties or compensations if the agreed-upon targets are not met.

## Tradeoffs Between Latency and Throughput

When setting SLOs and SLAs for LLM deployments, it is essential to balance the tradeoffs between latency, throughput, and cost efficiency:

- **Latency**: The time taken to process individual requests, including metrics like Time to First Token (TTFT) and Inter-Token Latency (ITL). Low latency is critical for user-facing applications where responsiveness is key.
- **Throughput**: The number of requests processed per second. High throughput is essential for handling large-scale workloads efficiently.
- **Cost Efficiency**: The cost per request, which depends on the system's resource utilization and throughput. Optimizing for cost efficiency often involves increasing throughput, which may come at the expense of higher latency for individual requests.

For example, a chat application may prioritize low latency to ensure a smooth user experience, while a batch processing system for content generation may prioritize high throughput to minimize costs.

## Examples of SLOs/SLAs for Common LLM Use Cases

### Real-Time, Application-Facing Usage

This category includes use cases where low latency is critical for external-facing applications. These systems must ensure quick responses to maintain user satisfaction and meet stringent performance requirements.

#### 1. Chat Applications

**Enterprise Use Case**: A customer support chatbot for an e-commerce platform, where quick responses are critical to maintaining user satisfaction and resolving issues in real time.

- **SLOs**:
  - TTFT: ≤ 200ms for 99% of requests
  - ITL: ≤ 50ms for 99% of requests

#### 2. Retrieval-Augmented Generation (RAG)

**Enterprise Use Case**: A legal document search tool that retrieves and summarizes relevant case law in real time for lawyers during court proceedings.

- **SLOs**:
  - Request Latency: ≤ 3s for 99% of requests
  - TTFT: ≤ 300ms for 99% of requests (if iterative outputs are shown)
  - ITL: ≤ 100ms for 99% of requests (if iterative outputs are shown)

#### 3. Instruction Following / Agentic AI

**Enterprise Use Case**: A virtual assistant for scheduling meetings and managing tasks, where quick responses are essential for user productivity.

- **SLOs**:
  - Request Latency: ≤ 5s for 99% of requests

### Real-Time, Internal Usage

This category includes use cases where low latency is important but less stringent compared to external-facing applications. These systems are often used by internal teams within enterprises, but if provided as a service, they may require external-facing guarantees.

#### 4. Content Generation

**Enterprise Use Case**: An internal marketing tool for generating ad copy and social media posts, where slightly higher latencies are acceptable compared to external-facing applications.

- **SLOs**:
  - TTFT: ≤ 600ms for 99% of requests
  - ITL: ≤ 200ms for 99% of requests

#### 5. Code Generation

**Enterprise Use Case**: A developer productivity tool for generating boilerplate code and API integrations, used internally by engineering teams.

- **SLOs**:
  - TTFT: ≤ 500ms for 99% of requests
  - ITL: ≤ 150ms for 99% of requests

#### 6. Code Completion

**Enterprise Use Case**: An integrated development environment (IDE) plugin for auto-completing code snippets, improving developer efficiency.

- **SLOs**:
  - Request Latency: ≤ 2s for 99% of requests

### Offline, Batch Use Cases

This category includes use cases where maximizing throughput is the primary concern. These systems process large volumes of data in batches, often during off-peak hours, to optimize resource utilization and cost efficiency.

#### 7. Summarization

**Enterprise Use Case**: A tool for summarizing customer reviews to extract insights for product improvement, processed in large batches overnight.

- **SLOs**:
  - Maximize Throughput: ≥ 100 requests per second

#### 8. Analysis

**Enterprise Use Case**: A data analysis pipeline for generating actionable insights from sales data, used to inform quarterly business strategies.

- **SLOs**:
  - Maximize Throughput: ≥ 150 requests per second

## Measuring Against These Objectives

The targets above are stated as a threshold plus a share of requests, for example "TTFT ≤ 200ms for 99% of requests". GuideLLM measures both parts directly.

### Declaring Objectives

Objectives are set on `--metrics`. Each is a per-request threshold in milliseconds, and a request conforms only when it meets all of them:

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=concurrent,streams=32 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics '{"kind":"generative","slo":{"ttft_ms":200,"tpot_ms":50}}' \
  --constraint kind=max_duration,seconds=120
```

| Objective | Compared against    | Notes                                                                                                                                                                                                                                                                                                        |
| --------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ttft_ms` | Time to First Token | Requires a streaming backend                                                                                                                                                                                                                                                                                 |
| `tpot_ms` | Inter-Token Latency | Excludes the first token. This is not GuideLLM's Time Per Output Token, which includes it. It is the closest metric to vLLM's `tpot`, though not identical: vLLM measures to request completion, inter-token latency to the last token received. Requests producing one token or fewer are left undetermined |
| `e2el_ms` | Request Latency     | Works with streaming and non-streaming backends                                                                                                                                                                                                                                                              |

Two metrics are then reported: SLO attainment, the share of requests meeting every objective, and request goodput, the rate of those conforming requests. Errored requests count against attainment, since a request that failed did not deliver a response within its objective. Requests cancelled when the run hits its duration limit are excluded, since the run ended them rather than the server. An objective naming a metric the workload cannot measure, such as `ttft_ms` against a non-streaming backend, leaves every request undetermined and both metrics are reported as unset rather than zero.

### Finding the Load an Objective Supports

Declaring objectives tells you whether one load level meets them. The `goodput` profile searches for the highest level that does, which is the question behind capacity planning.

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=goodput,target_attainment=0.99 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --metrics '{"kind":"generative","slo":{"ttft_ms":200,"tpot_ms":50}}' \
  --constraint kind=max_duration,seconds=120
```

`target_attainment` is the "for 99% of requests" half of the objective. The search doubles concurrency until a level misses that target, then bisects between the highest passing and lowest failing level, stopping once the answer is known to within `tolerance` (10% by default).

The search varies concurrency rather than request rate. Every concurrency level settles into a steady state, whereas a rate above what the server can sustain grows an unbounded backlog, and measurements taken there describe the backlog rather than the server.

Each level is judged on attainment rather than on goodput. Attainment is a ratio over the requests actually measured, so it does not depend on how much of the measurement window the server spent filling its pipeline before the first request completed.

### Reading the Result

Past the saturation point, request rate flattens while goodput falls, because requests still complete but no longer complete quickly enough to count. Measured against the mock server bundled with this repository, configured with 16 concurrent slots and a 1500ms end-to-end objective:

| Concurrency | Requests/sec | Goodput/sec | Attainment |
| ----------- | ------------ | ----------- | ---------- |
| 16          | 24.6         | 24.6        | 100.0%     |
| 26          | 24.4         | 23.2        | 95.1%      |
| 32          | 24.3         | 19.3        | 79.6%      |

Request rate is the same at 16 and 32 concurrent, so a search driven by throughput alone cannot tell them apart. Goodput falls 22% between them.

The search writes its outcome to `profile_result` in the benchmark report: every probe with its attainment and interval, the highest passing and lowest failing concurrency, and why the search stopped.

Each probe's attainment is scored against a confidence interval. When that interval straddles the target, the probe ran too briefly to decide the question, and GuideLLM warns rather than reporting a number that looks precise. The same warning is issued when the search stops on its probe budget or stream ceiling, because the level it reports is then a lower bound rather than the highest passing one. If every level tested failed but the deciding intervals straddled the target, the result is reported as indeterminate rather than as objectives that cannot be met. Raising `--constraint kind=max_duration,seconds=...` collects more requests per probe and narrows the interval.

A probe that a constraint stops mid-run, such as enforced over-saturation, is recorded but never used as a search bound. Such a run cancels active requests, which are excluded from attainment, so the completed remainder can look conforming at a concurrency the server could not actually sustain.

## Conclusion

Setting appropriate SLOs and SLAs is essential for optimizing LLM deployments to meet user expectations and business requirements. By balancing latency, throughput, and cost efficiency, organizations can ensure high-quality service while minimizing operational costs. The examples provided above serve as a starting point for defining SLOs and SLAs tailored to specific use cases.
