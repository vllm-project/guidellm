---
weight: -1
source_sha256: !!str "8f3df750660e8a76f60c17da3efaf31e9577216bd329135d08f05bb8b1025c2f"
---

# 嵌入模型基准测试指南

GuideLLM 支持对兼容 OpenAI 的嵌入模型端点进行基准测试，以衡量吞吐量、延迟和并发等性能表现。

## 概览

嵌入模型将文本转换为用于语义搜索、检索和相似度计算的稠密向量表示。与生成文本的生成式模型不同，嵌入模型：

- 处理输入文本并返回嵌入向量
- 不支持流式传输（每个请求只返回一次响应）
- 只统计输入 token（不统计输出 token）
- 测量请求延迟和吞吐量

## 快速开始

```bash
# Start vLLM server with an embeddings model
vllm serve BAAI/bge-small-en-v1.5 --port 8000

# Run benchmark
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000/v1,model=BAAI/bge-small-en-v1.5,request_format=/v1/embeddings \
  --data kind=synthetic_text,prompt_tokens=128 \
  --constraint kind=max_requests,count=100
```

## 与生成式基准测试的主要区别

| 特性       | 嵌入模型     | 生成式模型        |
| ---------- | ------------ | ----------------- |
| 输出       | 嵌入向量     | 生成的文本        |
| 流式传输   | 否           | 是                |
| 输出 token | 不适用       | 数量可变          |
| TTFT       | 不适用       | 测量              |
| Token 延迟 | 不适用       | 测量              |
| 主要指标   | 延迟、吞吐量 | TTFT、ITL、吞吐量 |

## 另请参阅

- [基准测试配置（英文）](../../getting-started/benchmark.md#benchmark-profiles-profile)：详细介绍各种配置类型
- [数据集指南（英文）](../../guides/datasets.md)：创建和使用自定义数据集
- [指标指南（英文）](../../guides/metrics.md)：了解性能指标
