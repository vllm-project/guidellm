ARG BASE_IMAGE=docker.io/python:3.13-slim

# release: take the last version and add a post if build iteration
# candidate: increment to next minor, add 'rc' with build iteration
# nightly: increment to next minor, add 'a' with build iteration
# alpha: increment to next minor, add 'a' with build iteration
# dev: increment to next minor, add 'dev' with build iteration
ARG GUIDELLM_BUILD_TYPE=dev

# Use a multi-stage build to create a lightweight production image
FROM $BASE_IMAGE as builder

# Ensure files are installed as root
USER root

# Set correct build type for versioning
ENV GUIDELLM_BUILD_TYPE=$GUIDELLM_BUILD_TYPE

# Install build tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir -U pdm

# disable pdm update check
ENV PDM_CHECK_UPDATE=false

# Copy repository files
COPY / /opt/app-root/src

# Create a venv and install guidellm
RUN python3 -m venv /opt/app-root/guidellm \
    && pdm use -p /opt/app-root/src -f /opt/app-root/guidellm \
    && pdm install -p /opt/app-root/src --check --prod --no-editable

# Prod image
FROM $BASE_IMAGE

# Add guidellm bin to PATH
# Argument defaults can be set with GUIDELLM_<ARG>
ENV HOME="/home/guidellm" \
    PATH="/opt/app-root/guidellm/bin:$PATH" \
    GUIDELLM_OUTPUT_PATH="/results/benchmarks.json"

# Create a non-root user
RUN useradd -K UMASK=0002 -Md $HOME -g root guidellm

# Switch to non-root user
USER guidellm

# Create the user home dir
WORKDIR $HOME

# Create a volume for results
VOLUME /results

# Metadata
LABEL org.opencontainers.image.source="https://github.com/vllm-project/guidellm" \
      org.opencontainers.image.description="GuideLLM Performance Benchmarking Container"

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/app-root/guidellm /opt/app-root/guidellm

ENTRYPOINT [ "/opt/app-root/guidellm/bin/guidellm" ]
CMD [ "benchmark", "run" ]
