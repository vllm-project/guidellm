---
weight: -5
source_sha256: !!str "aedbf7bede7abb84441ecae1889f53b8576d2b6123152fef89ff02dd2dcfb408"
---

# 使用指南

欢迎阅读 GuideLLM 使用指南！本节详细介绍 GuideLLM 的关键组件和概念，帮助你理解其内部工作方式、配置各项组件，并解读基准测试结果，从而优化大语言模型部署。

无论你想了解系统架构、探索支持的后端、配置数据集、分析指标，还是设定服务级别目标，都可以通过以下指南获取详细信息，为大语言模型部署做出更明智的决策。

## 主要指南

<div class="grid cards" markdown>

- :material-layers-outline:{ .lg .middle } 架构

  ______________________________________________________________________

  了解 GuideLLM 的模块化设计，以及核心组件如何协同评估大语言模型部署。

  [:octicons-arrow-right-24: 架构概览（英文）](../../guides/architecture.md)

- :material-server:{ .lg .middle } 后端

  ______________________________________________________________________

  了解支持的大语言模型后端，以及如何搭建兼容 OpenAI 的服务器以进行基准测试。

  [:octicons-arrow-right-24: 后端指南（英文）](../../guides/backends.md)

- :material-database:{ .lg .middle } 数据集

  ______________________________________________________________________

  配置并使用不同的数据源进行基准测试，包括合成数据、Hugging Face 数据集和基于文件的数据源。

  [:octicons-arrow-right-24: 数据集指南（英文）](../../guides/datasets.md)

- :material-chart-bar:{ .lg .middle } 指标

  ______________________________________________________________________

  了解 GuideLLM 提供的各项性能评估指标，包括延迟、吞吐量和 token 级分析。

  [:octicons-arrow-right-24: 指标指南（英文）](../../guides/metrics.md)

- :material-file-export:{ .lg .middle } 输出

  ______________________________________________________________________

  了解支持的输出格式，以及如何自定义基准测试结果报告。

  [:octicons-arrow-right-24: 输出指南（英文）](../../guides/outputs.md)

- :material-alert-outline:{ .lg .middle } 警告

  ______________________________________________________________________

  可在场景文件中、通过 `--metrics` 参数或环境变量，按指标路径、阈值和说明配置基准测试后的警告。

  [:octicons-arrow-right-24: 警告指南（英文）](../../guides/warnings.md)

- :material-target:{ .lg .middle } 服务级别目标

  ______________________________________________________________________

  为大语言模型部署定义并实施服务级别目标（SLO）和服务级别协议（SLA），保障可靠性和性能。

  [:octicons-arrow-right-24: SLO 指南（英文）](../../guides/service_level_objectives.md)

- :material-stop-circle-outline:{ .lg .middle } 过饱和停止策略

  ______________________________________________________________________

  当模型进入过饱和状态时，自动检测并停止基准测试，以免浪费计算资源，并确保结果有效。

  [:octicons-arrow-right-24: 过饱和停止策略指南（英文）](../../guides/over_saturation_stopping.md)

- :material-chart-bell-curve-cumulative:{ .lg .middle } 膝点配置

  ______________________________________________________________________

  找到吞吐量趋于饱和的位置，并可选择在检测到的膝点附近增加并发度测试点。

  [:octicons-arrow-right-24: 膝点配置指南（英文）](../../guides/knee_detection.md)

- :material-wrench:{ .lg .middle } 工具调用

  ______________________________________________________________________

  使用预设工具调用轮次、合成数据和数据集提供的工具定义，对多轮工具调用工作负载进行基准测试。

  [:octicons-arrow-right-24: 工具调用指南（英文）](../../guides/tool_calling.md)

- :material-image-multiple:{ .lg .middle } 多模态基准测试

  ______________________________________________________________________

  为文本与图像、视频和音频任务的多模态模型设置基准测试。

  [:octicons-arrow-right-24: 多模态指南（英文）](../../guides/multimodal/index.md)

- :material-lifebuoy:{ .lg .middle } 故障排除

  ______________________________________________________________________

  了解如何排查常见错误。

  [:octicons-arrow-right-24: 故障排除指南（英文）](../../guides/troubleshooting.md)

- :material-update:{ .lg .middle } v0.7.0 迁移指南

  ______________________________________________________________________

  了解如何将 GuideLLM CLI 命令从 v0.6.0 迁移到 v0.7.0。

  [:octicons-arrow-right-24: v0.7.0 迁移指南（英文）](../../guides/v0.7.0_migration_guide.md)

</div>
