ARG BASE_IMAGE=docker.io/python:3.13-slim

# Use a multi-stage build to create a lightweight production image
FROM $BASE_IMAGE as builder

# Ensure files are installed as root
USER root

# Copy repository files
COPY / /opt/app-root/src

# Create a venv and install guidellm
RUN python3 -m venv /opt/app-root/guidellm \
    && /opt/app-root/guidellm/bin/pip install --no-cache-dir /opt/app-root/src

# Prod image
FROM $BASE_IMAGE

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/app-root/guidellm /opt/app-root/guidellm

# Add guidellm bin to PATH
# Argument defaults can be set with GUIDELLM_<ARG>
ENV HOME="/home/guidellm" \
    PATH="/opt/app-root/guidellm/bin:$PATH" \
    GUIDELLM_OUTPUT_PATH="/results/benchmarks.json"

# Create a non-root user
RUN useradd -Md $HOME -g root guidellm

# Switch to non-root user
USER guidellm

# Create the user home dir
WORKDIR $HOME

# Create a volume for results
VOLUME /results

# Metadata
LABEL org.opencontainers.image.source="https://github.com/vllm-project/guidellm" \
      org.opencontainers.image.description="GuideLLM Performance Benchmarking Container"

ENTRYPOINT [ "/opt/app-root/guidellm/bin/guidellm" ]
CMD [ "benchmark", "run" ]
