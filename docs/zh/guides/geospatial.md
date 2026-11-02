---
weight: 0
source_sha256: !!str "d3ef04ec9e360b125c48df8ed58b736a48be25be1ce4e610e914649cb93698c5"
---

# 地理空间模型基准测试

GuideLLM 支持对处理卫星影像和遥感影像的地理空间模型进行基准测试。本指南介绍如何通过 vLLM 的 `/pooling` 端点，对 IBM-NASA Prithvi 等 [TerraTorch](https://terrastackai.github.io/terratorch/stable/guide/vllm/intro/) 地理空间模型进行基准测试。

## 概览

GuideLLM 中的地理空间模型可以：

- 处理通过 URL 或 base64 提供的卫星影像
- 返回用于下游任务的嵌入向量或池化表示
- 统计图像相关指标，而非文本 token
- 支持多波段影像，例如 Sentinel-1 和 Sentinel-2

## 支持的模型

本指南已使用以下地理空间模型进行测试：

- **IBM-NASA Prithvi-EO-2.0-300M-TL-Sen1Floods11**：使用 Sentinel-1 数据训练的洪水检测模型

## 配置

### 基本用法

要对地理空间模型进行基准测试，请在后端配置中设置 `request_format=/pooling`：

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=10 \
  --output kind=json,path=results.json
```

## 数据集格式

通过 vLLM 提供服务的地理空间模型要求数据集文件采用特定的嵌套 JSON 结构：

### JSONL 格式

```json
{
  "prompt": {
    "data": {
      "data": "https://example.com/image.tif",
      "data_format": "url",
      "out_data_format": "b64_json",
      "indices": [1, 2, 3, 8, 11, 12]
    },
    "priority": 0
  }
}
```

### 字段说明

- **prompt**：请求数据的顶层容器
  - **data**：包含影像信息的嵌套数据对象
    - **data**：影像的 URL 或 base64 编码数据，例如 GeoTIFF 文件
    - **data_format**：输入数据格式（`"url"` 或 `"base64"`）
    - **out_data_format**：期望的输出格式，例如 `"b64_json"`
    - **indices**：要处理的波段或通道索引数组，具体取值取决于卫星传感器
  - **priority**：（可选）用于调度的请求优先级，默认为 0

### 波段索引

波段的选择取决于模型。[Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) 使用以下六个波段进行洪水测绘训练：蓝、绿、红、窄近红外（NIR）、短波红外（SWIR）和第二短波红外（SWIR 2）。推理时，需要指定 vLLM 应使用 GeoTIFF 中的哪些波段。[terratorch_segmentation](https://terrastackai.github.io/terratorch/stable/guide/vllm/vllm_io_plugins/) 插件组件负责预处理 GeoTIFF。请查阅所用模型的文档，确定合适的波段配置。

## 完整示例：使用 Prithvi 检测洪水

以下是对 Prithvi 洪水检测模型进行基准测试的完整示例：

### 1. 准备数据集

创建 `flood_detection_dataset.jsonl`，并记得将 `data` 字段中的 GeoTIFF URL 替换为可通过网络访问的地址。可以在 [Hugging Face 上的 Prithvi 仓库](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main/examples)中找到示例。

以下命令会创建一个包含 100 条相同记录的示例数据集，用于基准测试：

```bash
printf '{"prompt":{"data": {"data": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif","data_format": "url","out_data_format": "b64_json","indices": [1, 2, 3, 8, 11, 12]},"priority": 0}}\n%.0s' {1..100} > flood_detection_dataset.jsonl
```

### 2. 启动 vLLM 服务器

```bash
vllm serve \
  ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --enforce-eager \
  --skip-tokenizer-init \
  --enable-mm-embeds \
  --io-processor-plugin \
  terratorch_segmentation
```

有关在 vLLM 中提供 TerraTorch 模型服务的更多信息，请参阅[相关文档](https://terrastackai.github.io/terratorch/stable/guide/vllm/intro/)。

### 3. 运行基准测试

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=100 \
  --output kind=json,path=results.json
```

## 与 vLLM Bench 对比

GuideLLM 的地理空间模型支持与 vLLM 的基准测试工具兼容：

### vLLM Bench 命令

```bash
vllm bench serve \
  --base-url http://localhost:8000 \
  --dataset-name=custom \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --skip-tokenizer-init \
  --endpoint /pooling \
  --backend vllm-pooling \
  --percentile-metrics e2el \
  --metric-percentiles 25,75,99 \
  --num-prompts 10 \
  --dataset-path flood_detection_dataset.jsonl
```

### 等效的 GuideLLM 命令

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000,model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11,request_format=/pooling \
  --data kind=json_file,path=flood_detection_dataset.jsonl \
  --data-column-mapper kind=pooling_column_mapper \
  --constraint kind=max_requests,count=10 \
  --output kind=json,path=results.json
```

## 另请参阅

- [数据集指南（英文）](../../guides/datasets.md)：了解数据集准备方法和格式
- [指标指南（英文）](../../guides/metrics.md)：了解基准测试指标
- [多模态指南（英文）](../../guides/multimodal/index.md)：了解如何使用多模态数据
