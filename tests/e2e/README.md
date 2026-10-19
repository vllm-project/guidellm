# E2E tests

E2E benchmarks default to GuideLLM's built-in **MockServer** (`guidellm mock-server`). No Docker build or HuggingFace Hub access is required for the default path.

```shell
python -m pytest tests/e2e
```

A minimal tokenizer (trained for Faker-style synthetic text, ~60KB) is vendored under `tests/fixtures/tokenizers/minimal/`. Tests run with `HF_HUB_OFFLINE=1` so accidental Hub downloads fail fast. One dedicated offline test seeds a local hub-style cache to validate hub-id resolution without network.

## Audio unit coverage

Run the unit suite, including request handling, audio parsing and metrics, benchmark reporting, serialization, and mock-server coverage, with the system Python environment:

```shell
python -m pytest tests/unit
```

## Generated audio benchmark

Run the MockServer benchmark flow with the system Python environment:

```shell
python -m pytest tests/e2e/test_audio_omni_benchmark.py
```

This covers `/v1/audio/speech` raw streaming from request formatting through the serialized audio metric report. The test skips when `--e2e-server=llm-d` is selected because the simulator does not implement the speech endpoint.

## Optional live audio checks

The live checks are disabled by default and make network requests only when their individual environment flags are set. Configure the target and model for the OpenAI-compatible server; set the API key only when that server requires one:

```shell
export GUIDELLM_LIVE_AUDIO_TARGET=http://localhost:8000
export GUIDELLM_LIVE_AUDIO_MODEL=your-audio-model
export GUIDELLM_LIVE_AUDIO_API_KEY=your-api-key
```

Run the speech endpoint check (for example, against a Qwen TTS or Voxtral deployment):

```shell
GUIDELLM_RUN_LIVE_AUDIO_SPEECH_TESTS=1 \
python -m pytest tests/integration/backends/openai/test_audio_live.py -k speech
```

Run the chat audio check (for example, against a Qwen Omni deployment):

```shell
GUIDELLM_RUN_LIVE_AUDIO_CHAT_TESTS=1 \
python -m pytest tests/integration/backends/openai/test_audio_live.py -k chat
```

If the server uses different request fields, set `GUIDELLM_LIVE_AUDIO_SPEECH_BODY` or `GUIDELLM_LIVE_AUDIO_CHAT_BODY` to a JSON object before running the corresponding check. The test body always includes the benchmark prompt, and the configured model name is supplied to the endpoint.

## Optional: llm-d inference simulator

To run against the [vLLM simulator by llm-d](https://llm-d.ai/docs/architecture/Components/inference-simulator) instead of MockServer:

```shell
# Linux / CI (extracts Linux binary from GHCR; requires Docker Buildx for -o)
docker buildx build . -f tests/e2e/vllm-sim.Dockerfile -o type=local,dest=./

# macOS native binary (requires Docker Buildx + a running engine, e.g. Colima)
docker buildx build . -f tests/e2e/vllm-sim-macos.Dockerfile -o type=local,dest=./
```

If `docker build -o` fails with "unknown shorthand flag: 'o'", install/register buildx:

```shell
mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix docker-buildx)/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx
```

Podman may also work when it supports `-o type=local`:

```shell
podman build . -f tests/e2e/vllm-sim-macos.Dockerfile -o type=local,dest=./
```

Then:

```shell
python -m pytest tests/e2e --e2e-server=llm-d 
or
tox -e test-e2e -- --e2e-server=llm-d

```
