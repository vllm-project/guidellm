# Geospatial Model Benchmarking

GuideLLM supports benchmarking geospatial models that process satellite and remote sensing imagery. This guide focuses on benchmarking [TerraTorch](<>) Geospatial models like IBM-NASA's Prithvi using vLLM's `/pooling` endpoint.

## Overview

Geospatial models in GuideLLM:

- Process satellite imagery (via URLs or base64)
- Return embeddings or pooled representations for downstream tasks
- Track image-related metrics instead of text tokens
- Support multi-band imagery (e.g., Sentinel-1, Sentinel-2)

## Supported Models

This guide has been tested with the following geospatial models:

- **IBM-NASA Prithvi-EO-2.0-300M-TL-Sen1Floods11**: Flood detection model trained on Sentinel-1 data

## Configuration

### Basic Usage

To benchmark a geospatial model, use the `--request-format /pooling` parameter:

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --backend openai_http \
  --data flood_detection_dataset.jsonl \
  --request-format /pooling \
  --data-column-mapper pooling_column_mapper \
  --max-requests 10 \
  --output-path results.json
```

## Dataset Format

Geospatial models served with vLLM expect a specific nested JSON structure in the dataset file:

### JSONL Format

```json
{
  "prompt": {
    "data": {
      "data": "https://example.com/image.tif",
      "data_format": "url",
      "out_data_format": "b64_json",
      "indices": [1, 2, 3, 8, 11, 12]
    },
  }
}
```

### Field Descriptions

- **prompt**: Top-level container for the request data
  - **data**: Nested data object containing image information
    - **data**: URL or base64-encoded image data (e.g., GeoTIFF file)
    - **data_format**: Format of input data (`"url"` or `"base64"`)
    - **out_data_format**: Desired output format (e.g., `"b64_json"`)
    - **indices**: Array of band/channel indices to process (specific to the satellite sensor)

### Band Indices

The selection of the bands depends on the model. [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) was trained on the following six bands for flood mapping: Blue, Green, Red, Narrow NIR, SWIR, SWIR 2. To use the correct input data at inference time we need to specify which of the bands available in the GeoTIFF vLLM should use. The preprocessing of the GeoTIFF is handled by the [terratorch_segmentation](https://terrastackai.github.io/terratorch/stable/guide/vllm/vllm_io_plugins/) plugin component. Consult your model's documentation for the appropriate band configuration.

## Complete Example: Flood Detection with Prithvi

Here's a complete example for benchmarking the Prithvi flood detection model:

### 1. Prepare Dataset

Create `flood_detection_dataset.jsonl` remembering to update the GeoTIFF url reference by `data` with one reachable over the network. You can find examples in the [Prithvi repository on HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main/examples).

The following command creates a sample dataset with 100 identical entries for benchmarking purposes:

```bash
printf '{"prompt":{"data": {"data": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif","data_format": "url","out_data_format": "b64_json","indices": [1, 2, 3, 8, 11, 12]},"priority": 0}}\n%.0s' {1..100} > flood_detection_dataset.jsonl
```

### 2. Start vLLM Server

```bash
vllm serve \
  ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --enforce-eager \
  --skip-tokenizer-init \
  --enable-mm-embeds \
  --io-processor-plugin \
  terratorch_segmentation
```

To know more about serving TerraTorch models in vLLM follow the available [documentation](https://terrastackai.github.io/terratorch/stable/guide/vllm/intro/)

### 3. Run Benchmark

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --backend openai_http \
  --data flood_detection_dataset.jsonl \
  --request-format /pooling \
  --data-column-mapper pooling_column_mapper \
  --max-requests 100 \
  --output-path results.json
```

## Comparison with vLLM Bench

GuideLLM's geospatial model support is compatible with vLLM's benchmark tool:

### vLLM Bench Command

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

### Equivalent GuideLLM Command

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --backend openai_http \
  --data flood_detection_dataset.jsonl \
  --request-format /pooling \
  --data-column-mapper pooling_column_mapper \
  --max-requests 10 \
  --output-path results.json
```

## See Also

- [Datasets Guide](datasets.md) - Dataset preparation and formats
- [Metrics Guide](metrics.md) - Understanding benchmark metrics
- [Multimodal Guide](multimodal/index.md) - Working with multimodal data
