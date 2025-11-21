# TODO: Update to official python-3.13-minimal image when available
ARG BASE_IMAGE=quay.io/fedora/python-313-minimal:latest

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

# Install build tooling
RUN dnf install -y git \
    && /usr/bin/python3 -m venv /tmp/pdm \
    && /tmp/pdm/bin/pip install --no-cache-dir -U pdm \
    && ln -s /tmp/pdm/bin/pdm /usr/local/bin/pdm

# Disable pdm update check
# Set correct build type for versioning
ENV PDM_CHECK_UPDATE=false \
    GUIDELLM_BUILD_TYPE=$GUIDELLM_BUILD_TYPE

# Copy repository files
# Do this as late as possible to leverage layer caching
COPY / /src

# Install guidellm and locked dependencies
RUN pdm use -p /src -f /opt/app-root \
    && pdm install -p /src -G all --check --prod --no-editable

# Prod image
FROM $BASE_IMAGE

# Switch to root for installing packages
USER root

# Install some helpful utilities and deps
RUN dnf install -y --setopt=install_weak_deps=False \
        vi tar rsync ffmpeg-free \
    && dnf clean all

# Switch back to unpriv user
# Root group for k8s
USER 1001:0

# Add guidellm bin to PATH
# Argument defaults can be set with GUIDELLM_<ARG>
ENV HOME="/home/guidellm" \
    GUIDELLM_OUTPUT_PATH="/results/benchmarks.json"

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
