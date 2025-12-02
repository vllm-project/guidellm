# GuideLLM with uv, CUDA 13, and embedded dataset
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create symlink for python command
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create non-root user
RUN useradd -m -s /bin/bash guidellm
USER guidellm
WORKDIR /home/guidellm

# Copy source code
COPY --chown=guidellm:guidellm . /home/guidellm/guidellm-src

# Install guidellm with uv
WORKDIR /home/guidellm/guidellm-src
RUN uv venv --python python3.12 /home/guidellm/.venv && \
    . /home/guidellm/.venv/bin/activate && \
    uv pip install --no-cache -e .

# Pre-download LibriSpeech dataset (parquet revision)
RUN . /home/guidellm/.venv/bin/activate && \
    python -c "from datasets import load_dataset; load_dataset('hf://datasets/distil-whisper/librispeech_asr@refs%2Fconvert%2Fparquet', 'clean', split='validation', streaming=False, cache_dir='/home/guidellm/.cache/huggingface')"

# Add venv to PATH
ENV PATH="/home/guidellm/.venv/bin:$PATH"
ENV HF_HOME="/home/guidellm/.cache/huggingface"

# Create results volume
WORKDIR /home/guidellm
VOLUME /results

ENTRYPOINT ["guidellm"]
CMD ["benchmark", "run"]
