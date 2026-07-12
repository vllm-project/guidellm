# GuideLLM Benchmark Testing with a Local Tokenizer and Custom Dataset

Benchmark an already-deployed OpenAI-compatible model endpoint using GuideLLM with a local tokenizer and a custom JSONL prompt dataset, without needing a local copy of the model itself. This example runs GuideLLM via a local `pip install`.


## Getting Started

### 1. Prepare the Tokenizer Files

The model is already deployed and running, for example on GPU, served via vLLM or OpenShift, behind an OpenAI-compatible endpoint. GuideLLM never loads or runs the model, it only talks to it over HTTP. What GuideLLM does need locally is the tokenizer, since it uses it to compute token counts for its metrics such as prompt tokens, output tokens and throughput.

From the model's repo, for example the HuggingFace Hub or wherever the model artifacts live, copy the tokenizer files required by your specific model.
This example uses a Mistral-family model, which needs only these three files. A pattern that's typical of many tokenizers, but not universal, so treat it as a starting point rather than a fixed requirement for every model:

```bash
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Place them in a local directory, e.g. `/home/<USERNAME>/mistral`.

### 2. Prepare a Custom JSONL Dataset

In the same directory, add your JSONL prompt file(s). Each line is a JSON object with a `prompt` field, for example:

```json
{"prompt": "Translate the following text in full to German, French and Spanish. Provide a fluent translation of the entire text without summarizing or omitting any sentences. Respond with the translation in the same format as the original text, including line breaks and whitespace. Text: ..."}
```

You can split your data into multiple files by length or category if that's useful for your benchmarking scenario, for example:

```bash
short_translation_prompts.jsonl
medium_translation_prompts.jsonl
long_translation_prompts.jsonl
```

After both steps, your local directory should look like this:

```bash
ls /home/<USERNAME>/mistral
long_translation_prompts.jsonl  medium_translation_prompts.jsonl  short_translation_prompts.jsonl
special_tokens_map.json  tokenizer_config.json  tokenizer.json
```

Verify your GuideLLM install:

```bash
guidellm --version
guidellm version: 0.7.0
```

______________________________________________________________________

## 3. Running the Benchmark

```bash
guidellm run \
  --backend kind=openai_http,target=http://<PREDICTOR_URL> \
  --data '{"kind":"json_file","path":"/home/<USERNAME>/mistral/long_translation_prompts.jsonl","load_kwargs":{"split":"train"}}' \
  --tokenizer '{"kind":"huggingface_auto","model":"/home/<USERNAME>/mistral"}' \
  --profile kind=concurrent,streams=100 \
  --output kind=json,path=/home/<USERNAME>/mistral/benchmarks.json
```

### What Each Argument Does

| Argument | Purpose |
|---|---|
| `--backend kind=openai_http,target=...` | Points GuideLLM at your OpenAI-compatible predictor endpoint. |
| `--data kind=json_file,path=...` | Loads prompts from the local JSONL file. |
| `--tokenizer kind=huggingface_auto,model=/home/<USERNAME>/mistral` | Loads the tokenizer from the local directory, the 3 files from step 1, instead of downloading a model. |
| `--profile kind=concurrent,streams=100` | Simulates 100 concurrent "users" hitting the endpoint at once. |
| `--output kind=json,path=/home/<USERNAME>/mistral/benchmarks.json` | Output file type and path. |

______________________________________________________________________

## 5. Notes

- The tokenizer path (`/home/<USERNAME>/mistral` in this example) only needs the 3 tokenizer files, no model weights required.
- The dataset file path and the tokenizer path can point to different directories or models if you want to mix and match, though in this example they are the same folder for convenience.
- Swap `--profile kind=concurrent,streams=N` for other GuideLLM profiles, such as `synchronous` or `throughput`, depending on the load pattern you want to test.
