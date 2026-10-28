---
weight: -4
source_sha256: !!str "6e34ec94343d15d0617d5cd5a77fbf5acf8d29952b65a993ea970ba517574f18"
---

# 示例 {#examples}

欢迎来到 GuideLLM 示例部分！这里展示了 GuideLLM 在各种真实场景中评估和优化大语言模型部署的实用方法。我们希望通过具体示例，帮助你将 GuideLLM 有效地用于自己的工作流程。

## 可用示例 {#available-examples}

- [在 vLLM 模拟器上练习](./practice_on_vllm_simulator.md) — 无需 GPU，从头开始使用 vLLM 模拟器运行首次基准测试
- [自定义 JSONL 数据集](./custom-jsonl-dataset.md) — 使用本地分词器和自定义 JSONL 提示词，对已部署的端点进行基准测试
- [Kubernetes / OpenShift 客户端作业](./kubernetes-openshift-job.md) — 将 GuideLLM 作为集群内的非 GPU 批处理作业运行，访问兼容 OpenAI 的服务，并将结果保存到 PVC
- [自适应拐点配置档](../guides/knee_detection.md) — 计算吞吐量拐点，并通过增加并发流测量来细化结果

## 征集示例贡献 {#call-for-contributions}

目前可用的示例还不多，我们欢迎社区成员贡献内容！如果你有使用 GuideLLM 解决实际问题或优化大语言模型部署的经验，欢迎将示例分享给大家。

贡献示例的步骤如下：

1. Fork [GuideLLM 仓库](https://github.com/vllm-project/guidellm)
2. 按照[贡献指南](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md)，在 `docs/en/examples/` 目录中创建示例
3. 提交 Pull Request

你的示例将帮助更多人有效地使用 GuideLLM，也能丰富有关大语言模型部署优化的知识库。
