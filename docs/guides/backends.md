# Backends

GuideLLM is designed to work with OpenAI-compatible HTTP servers, enabling seamless integration with a variety of generative AI backends. This compatibility ensures that users can evaluate and optimize their large language model (LLM) deployments efficiently. While the current focus is on OpenAI-compatible servers, we welcome contributions to expand support for other backends, including additional server implementations and Python interfaces.

## Supported Backends

### OpenAI-Compatible HTTP Servers

GuideLLM supports OpenAI-compatible HTTP servers, which provide a standardized API for interacting with LLMs. This includes popular implementations such as [vLLM](https://github.com/vllm-project/vllm) and [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference). These servers allow GuideLLM to perform evaluations, benchmarks, and optimizations with minimal setup.

## Examples for Spinning Up Compatible Servers

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-performance OpenAI-compatible server designed for efficient LLM inference. It supports a variety of models and provides a simple interface for deployment.

First ensure you have vLLM installed (`pip install vllm`), and then run the following command to start a vLLM server with a Llama 3.1 8B quantized model:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

### 2. Text Generation Inference (TGI)

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is another OpenAI-compatible server that supports a wide range of models, including those hosted on Hugging Face. TGI is optimized for high-throughput and low-latency inference.

To start a TGI server with a Llama 3.1 8B model using Docker, run the following command:

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

For more information on starting a TGI server, see the [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index).

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](https://github.com/neuralmagic/guidellm/blob/main/CONTRIBUTING.md) file.
