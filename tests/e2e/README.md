# E2E tests

E2E benchmarks default to GuideLLM's built-in **MockServer** (`guidellm mock-server`). No Docker build or HuggingFace Hub access is required for the default path.

```shell
tox -e test-e2e
```

Tokenizer files are vendored under `tests/fixtures/tokenizers/gpt2/`. Tests run with `HF_HUB_OFFLINE=1` so accidental Hub downloads fail fast. One dedicated offline test seeds a local hub-style cache to validate hub-id (`gpt2`) resolution without network.

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
tox -e test-e2e -- --e2e-server=llm-d
```
