FROM golang AS base

WORKDIR /app

ARG BUILDOS
ARG BUILDARCH

RUN apt-get update && \
    apt-get install -y libzmq3-dev pkg-config && \
    git clone https://github.com/llm-d/llm-d-inference-sim.git && \
    cd llm-d-inference-sim && \
    git checkout v0.3.0 && \
    GOOS=${BUILDOS} GOARCH=${BUILDARCH} make build

WORKDIR /app/llm-d-inference-sim

FROM scratch
COPY --from=base /app/llm-d-inference-sim/bin /bin
