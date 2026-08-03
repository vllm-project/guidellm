# OpenTelemetry Tracing

GuideLLM can emit OpenTelemetry spans for benchmark runs, scheduling strategies, and individual inference requests. HTTP requests and WebSocket handshakes propagate the active W3C `traceparent` and `tracestate` context, allowing instrumented model servers and gateways to join the same distributed trace.

Tracing is disabled by default and the base installation does not require OpenTelemetry. Install the optional dependencies and configure an OTLP collector:

```bash
pip install 'guidellm[otel]'

export OTEL_SERVICE_NAME=guidellm
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_RESOURCE_ATTRIBUTES=deployment.environment.name=development
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=constant \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --constraint kind=max_requests,count=100
```

The OTLP endpoint is the activation signal for GuideLLM's automatic exporter setup. Both `grpc` and `http/protobuf` protocols are supported. Set `OTEL_SDK_DISABLED=true` to explicitly disable tracing.

Request spans cover the full streamed response lifecycle and record request IDs, backend and model metadata, outcome, error type, token usage, total request latency, and time to first token when available. GuideLLM never adds prompt or response content to spans.

Each worker process initializes tracing from the same environment, so multiprocess benchmarks export request spans without sharing unpickleable SDK objects through the scheduler. As with any high-volume benchmark, use an appropriate sampler and size the collector for the expected span rate.
