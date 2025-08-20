# E2E tests

The E2E tests in GuideLLM use the [vLLM simulator by llm-d](https://llm-d.ai/docs/architecture/Components/inf-simulator), to run them run the following command:

```shell
docker build . -f tests/e2e/vllm-sim.Dockerfile -o type=local,dest=./
```

Then to run the tests:
```shell
tox -e test-e2e
```
