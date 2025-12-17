ARG BASE_IMAGE=vllm/vllm-openai

# Use a multi-stage build to create a lightweight production image
FROM $BASE_IMAGE as builder

# release: take the last version and add a post if build iteration
# candidate: increment to next minor, add 'rc' with build iteration
# nightly: increment to next minor, add 'a' with build iteration
# alpha: increment to next minor, add 'a' with build iteration
# dev: increment to next minor, add 'dev' with build iteration
ARG GUIDELLM_BUILD_TYPE=dev

# Switch to root for installing packages
USER root

RUN apt-get update || true \
    && apt-get install -y --no-install-recommends gnupg ca-certificates \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 871920D1991BC93C \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA6932366A755776 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Set the default venv for uv
# Copy instead of link files with uv
# Set correct build type for versioning
ENV VIRTUAL_ENV=/opt/app-root \
    UV_LINK_MODE="copy" \
    GUIDELLM_BUILD_TYPE=$GUIDELLM_BUILD_TYPE

# Copy repository files
# Do this as late as possible to leverage layer caching
COPY / /src

# Install guidellm and locked dependencies
# Have it use system packages (where vllm is installed)
RUN uv venv /opt/app-root --system-site-packages

RUN uv sync --active --project /src --frozen --no-dev --extra all --no-editable

# Prod image
FROM $BASE_IMAGE

# Switch to root for installing packages
USER root

# Install some helpful utilities and deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        vim tar rsync ffmpeg \
        && apt clean

# Switch back to unpriv user
# Root group for k8s
USER 1001:0

# Add guidellm bin to PATH
# Argument defaults can be set with GUIDELLM_<ARG>
ENV HOME="/home/guidellm" \
    GUIDELLM_OUTPUT_DIR="/results"

# Create the user home dir
WORKDIR $HOME

# Create a volume for results
VOLUME /results

# Metadata
LABEL io.k8s.display-name="GuideLLM" \
      org.opencontainers.image.description="GuideLLM Performance Benchmarking Container" \
      org.opencontainers.image.source="https://github.com/vllm-project/guidellm" \
      org.opencontainers.image.documentation="https://blog.vllm.ai/guidellm/stable" \
      org.opencontainers.image.license="Apache-2.0"

# Copy the virtual environment from the builder stage
# Do this as late as possible to leverage layer caching
COPY --chown=1001:0 --from=builder /opt/app-root /opt/app-root

ENTRYPOINT [ "/opt/app-root/bin/guidellm" ]
CMD [ "benchmark", "run" ]
