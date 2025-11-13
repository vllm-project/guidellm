---
weight: -8
---

# Start a Server

# **Start a Server**

Before running GuideLLM benchmarks, you need an OpenAI-compatible server to test against. This guide will help you set up a server quickly.

## **Recommended Option: vLLM**

vLLM is the recommended backend for running GuideLLM benchmarks due to its performance and compatibility.

### Installing vLLM

```bash
pip install vllm
```

### Starting a vLLM Server

Run the following command to start a vLLM server with a quantized Llama 3.1 8B model:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

This will start an OpenAI-compatible server at `http://localhost:8000`.

For more configuration options, refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/).

## **Alternative Servers**

GuideLLM supports any OpenAI-compatible server, such as TGI, SG Lang, and more. For detailed information on all supported backends, see the [Backends documentation](../guides/backends/).

## **Verifying Your Server**

Once your server is running, you can verify it's working and accessible from your benchmarking server with a simple curl command (if it is running on another machine, replace `localhost` with the server's IP address):

```bash
curl http://localhost:8000/v1/models
```

You should see a response listing the available model on your server.
