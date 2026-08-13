FROM ghcr.io/llm-d/llm-d-inference-sim:v0.3.0 AS base

FROM scratch
COPY --from=base /app/llm-d-inference-sim /bin/llm-d-inference-sim
