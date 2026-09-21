---
title: 简体中文
weight: 100
---

# 首页

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
面向真实世界大语言模型推理优化、感知 SLO 的基准测试与评估平台
</h3>

**GuideLLM** 是一个用于评估语言模型在真实工作负载和配置下性能的平台。它可以模拟与 OpenAI 兼容服务器及 vLLM 原生服务器的端到端交互，生成反映生产使用情况的工作负载模式，并生成详细报告，帮助团队了解系统行为、资源需求和运行限制。GuideLLM 支持真实及合成数据集、多模态输入和灵活的执行配置，为工程团队和机器学习团队提供一致的模型行为评估、部署调优及容量规划框架。

## 主要特性

- **记录完整的延迟和 token 级统计信息，以支持 SLO 驱动的评估：** 包括首 token 延迟（TTFT）、token 间延迟（ITL）及端到端行为的完整分布。
- **生成真实且可配置的流量模式：** 支持同步、并发和基于速率的模式，并可执行可复现的扫描以确定安全运行范围。
- **同时支持真实和合成的多模态数据集：** 支持文本、图像、音频和视频输入，可在同一框架中开展受控实验和生产环境风格的评估。
- **生成标准化且可导出的报告：** 可用于仪表盘、分析和回归跟踪，确保不同团队及工作流之间的一致性。
- **提供高吞吐量、可扩展的基准测试能力：** 支持多进程、多线程和异步执行，并提供灵活的 CLI/API，既便于定制，也适合快速上手。

## 主要内容

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } 快速开始

  ______________________________________________________________________

  安装 GuideLLM、运行首次基准测试并分析结果，以优化大语言模型部署。

  [:octicons-arrow-right-24: 安装 GuideLLM](./getting-started/install.md)

- :material-book-open-variant:{ .lg .middle } 使用指南

  ______________________________________________________________________

  深入了解后端、数据集、指标和服务级别目标等 GuideLLM 基准测试主题。

  [:octicons-arrow-right-24: 阅读英文使用指南](../guides/)

- :material-code-tags:{ .lg .middle } 示例

  ______________________________________________________________________

  通过分步示例了解真实场景下的基准测试和优化方法。

  [:octicons-arrow-right-24: 阅读英文示例](../examples/)

- :material-api:{ .lg .middle } API 参考

  ______________________________________________________________________

  查看完整的 GuideLLM API 参考文档，以便将基准测试集成到工作流中。

  [:octicons-arrow-right-24: 阅读英文 API 参考](../api/)

</div>
