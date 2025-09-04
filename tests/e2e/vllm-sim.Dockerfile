FROM golang AS base

WORKDIR /app

RUN git clone https://github.com/llm-d/llm-d-inference-sim.git && \
    cd llm-d-inference-sim && \
    make build

WORKDIR /app/llm-d-inference-sim

FROM scratch
COPY --from=base /app/llm-d-inference-sim/bin/llm-d-inference-sim /bin/llm-d-inference-sim
