# Home

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
SLO-Aware Benchmarking and Evaluation Platform for Optimizing Real-World LLM Inference
</h3>

**GuideLLM** is a platform for evaluating how language models perform under real workloads and configurations. It simulates end-to-end interactions with OpenAI-compatible and vLLM-native servers, generates workload patterns that reflect production usage, and produces detailed reports that help teams understand system behavior, resource needs, and operational limits. GuideLLM supports real and synthetic datasets, multimodal inputs, and flexible execution profiles, giving engineering and ML teams a consistent framework for assessing model behavior, tuning deployments, and planning capacity as their systems evolve.

## Key Features

- **Captures complete latency and token-level statistics for SLO-driven evaluation:** Including full distributions for TTFT, ITL, and end-to-end behavior.
- **Generates realistic, configurable traffic patterns:** Across synchronous, concurrent, and rate-based modes, including reproducible sweeps to identify safe operating ranges.
- **Supports both real and synthetic multimodal datasets:** Enabling controlled experiments and production-style evaluations in one framework with support for text, image, audio, and video inputs.
- **Produces standardized, exportable reports:** For dashboards, analysis, and regression tracking, ensuring consistency across teams and workflows.
- **Delivers high-throughput, extensible benchmarking:** With multiprocessing, threading, async execution, and a flexible CLI/API for customization or quickstarts.

## Key Sections

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } Getting Started

  ______________________________________________________________________

  Install GuideLLM, set up your first benchmark, and analyze the results to optimize your LLM deployment.

  [:octicons-arrow-right-24: Getting started](./getting-started/)

- :material-book-open-variant:{ .lg .middle } Guides

  ______________________________________________________________________

  Detailed guides covering backends, datasets, metrics, and service level objectives for effective LLM benchmarking.

  [:octicons-arrow-right-24: Guides](./guides/)

- :material-code-tags:{ .lg .middle } Examples

  ______________________________________________________________________

  Step-by-step examples demonstrating real-world benchmarking scenarios and optimization techniques.

  [:octicons-arrow-right-24: Examples](./examples/)

- :material-api:{ .lg .middle } API Reference

  ______________________________________________________________________

  Complete reference documentation for the GuideLLM API to integrate benchmarking into your workflow.

  [:octicons-arrow-right-24: API Reference](./api/)

</div>
