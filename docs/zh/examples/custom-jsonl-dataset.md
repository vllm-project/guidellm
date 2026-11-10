---
source_sha256: !!str "b42aeb86e760ea508bcf54341af29ddb6ce00bcfd392f50ff66efada9fcd72ec"
---

# 使用本地分词器和自定义数据集进行 GuideLLM 基准测试 {#guidellm-benchmark-testing-with-a-local-tokenizer-and-custom-dataset}

使用 GuideLLM、本地分词器和自定义 JSONL 提示词数据集，对已部署的 OpenAI 兼容模型端点进行基准测试，无需在本地保存模型本身。本示例通过本地 `pip install` 安装并运行 GuideLLM。

## 开始使用

### 1. 准备分词器文件

模型已部署并运行，例如运行在 GPU 上，由 vLLM 或 OpenShift 提供服务，并通过 OpenAI 兼容端点访问。GuideLLM 不会加载或运行该模型，只会通过 HTTP 与其通信。GuideLLM 需要在本地获取分词器，因为它会用分词器计算提示词 token 数、输出 token 数和吞吐量等指标。

从模型仓库（例如 HuggingFace Hub 或保存模型文件的位置）复制特定模型所需的分词器文件。本示例使用 Mistral 系列模型，只需以下三个文件。这是许多分词器常见的配置，但并非通用要求；请将其作为起点，根据模型进行调整，不要将其视为每个模型都必须满足的固定要求：

```bash
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

将这些文件放在本地目录中，例如 `/home/<USERNAME>/mistral`。

______________________________________________________________________

### 2. 准备自定义 JSONL 数据集

在同一目录中添加一个或多个 JSONL 提示词文件。每行都是一个包含 `prompt` 字段的 JSON 对象，例如：

```json
{"prompt": "Translate the following text in full to German, French and Spanish. Provide a fluent translation of the entire text without summarizing or omitting any sentences. Respond with the translation in the same format as the original text, including line breaks and whitespace. Text: ..."}
```

你也可以按长度或类别将数据拆分到多个文件中，以满足基准测试场景的需要，例如：

```bash
short_translation_prompts.jsonl
medium_translation_prompts.jsonl
long_translation_prompts.jsonl
```

完成以上两步后，本地目录应如下所示：

```bash
ls /home/<USERNAME>/mistral
long_translation_prompts.jsonl  medium_translation_prompts.jsonl  short_translation_prompts.jsonl
special_tokens_map.json  tokenizer_config.json  tokenizer.json
```

验证 GuideLLM 是否已安装：

```bash
guidellm --version
guidellm version: 0.7.0
```

______________________________________________________________________

## 3. 运行基准测试

```bash
guidellm run \
  --backend kind=openai_http,target=http://<PREDICTOR_URL> \
  --data '{"kind":"json_file","path":"/home/<USERNAME>/mistral/long_translation_prompts.jsonl","load_kwargs":{"split":"train"}}' \
  --tokenizer '{"kind":"huggingface_auto","model":"/home/<USERNAME>/mistral"}' \
  --profile kind=concurrent,streams=100 \
  --output kind=json,path=/home/<USERNAME>/mistral/benchmarks.json
```

### 各参数的作用

| 参数                                                               | 说明                                                        |
| ------------------------------------------------------------------ | ----------------------------------------------------------- |
| `--backend kind=openai_http,target=...`                            | 将 GuideLLM 指向兼容 OpenAI 的预测端点。                    |
| `--data kind=json_file,path=...`                                   | 从本地 JSONL 文件加载提示词。                               |
| `--tokenizer kind=huggingface_auto,model=/home/<USERNAME>/mistral` | 从本地目录加载步骤 1 中的三个文件作为分词器，无需下载模型。 |
| `--profile kind=concurrent,streams=100`                            | 模拟 100 个“用户”同时访问该端点。                           |
| `--output kind=json,path=/home/<USERNAME>/mistral/benchmarks.json` | 指定输出文件的类型和路径。                                  |

______________________________________________________________________

## 4. 注意事项

- 本示例中的分词器路径（`/home/<USERNAME>/mistral`）只需包含三个分词器文件，无需模型权重。
- 数据集文件和分词器可以位于不同目录，也可以使用不同模型；本示例为了方便将它们放在同一目录中。
- 可将 `--profile kind=concurrent,streams=N` 替换为其他 GuideLLM 配置档，例如 `synchronous` 或 `throughput`，具体取决于要测试的负载模式。
