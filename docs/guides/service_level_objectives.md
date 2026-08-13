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

## Conclusion

Setting appropriate SLOs and SLAs is essential for optimizing LLM deployments to meet user expectations and business requirements. By balancing latency, throughput, and cost efficiency, organizations can ensure high-quality service while minimizing operational costs. The examples provided above serve as a starting point for defining SLOs and SLAs tailored to specific use cases.
