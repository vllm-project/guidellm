# Remote vLLM Server Testing for Embeddings

This directory contains tests and documentation for testing GuideLLM embeddings support against a remote vLLM server deployment.

## Remote Server Information

**Server Address:** `ec2-18-117-141-109.us-east-2.compute.amazonaws.com`
**SSH Access:** `ssh -i ~/mtahhan.pem ec2-user@ec2-18-117-141-109.us-east-2.compute.amazonaws.com`

## Server Setup

### Starting a vLLM Embeddings Server

SSH into the remote server and start vLLM with an embeddings model:

```bash
# SSH into server
ssh -i ~/mtahhan.pem ec2-user@ec2-18-117-141-109.us-east-2.compute.amazonaws.com

# Option 1: BAAI/bge-base-en-v1.5 (small, fast, good for testing)
vllm serve BAAI/bge-base-en-v1.5 --port 8000

# Option 2: intfloat/e5-mistral-7b-instruct (larger, higher quality)
vllm serve intfloat/e5-mistral-7b-instruct --port 8000

# Option 3: With specific settings
vllm serve BAAI/bge-base-en-v1.5 \
  --port 8000 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9
```

### Verifying Server is Running

```bash
# From local machine
curl http://ec2-18-117-141-109.us-east-2.compute.amazonaws.com:8000/health

# Test embeddings endpoint
curl http://ec2-18-117-141-109.us-east-2.compute.amazonaws.com:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "BAAI/bge-base-en-v1.5"
  }'
```

## Running Tests

### Environment Setup

```bash
# Set remote server URL
export GUIDELLM_REMOTE_URL=http://ec2-18-117-141-109.us-east-2.compute.amazonaws.com:8000

# Optional: Set baseline model for quality validation
export GUIDELLM_BASELINE_MODEL=BAAI/bge-base-en-v1.5
```

### Running Pytest Tests

```bash
# Run all remote tests
pytest tests/remote/test_embeddings_remote.py -v

# Run specific test
pytest tests/remote/test_embeddings_remote.py::test_remote_basic_embeddings -v

# Run with detailed output
pytest tests/remote/test_embeddings_remote.py -v -s

# Skip remote tests if server unavailable
pytest tests/remote/test_embeddings_remote.py -v -m "not slow"
```

### Running CLI Benchmarks

#### Basic Embeddings Benchmark

```bash
guidellm benchmark embeddings \
  --target $GUIDELLM_REMOTE_URL \
  --model BAAI/bge-base-en-v1.5 \
  --outputs csv,html,json \
  --max-requests 100 \
  --rate 10
```

#### With Quality Validation

```bash
guidellm benchmark embeddings \
  --target $GUIDELLM_REMOTE_URL \
  --model BAAI/bge-base-en-v1.5 \
  --enable-quality-validation \
  --baseline-model BAAI/bge-base-en-v1.5 \
  --quality-tolerance 0.01 \
  --outputs csv,html,json \
  --max-requests 100
```

#### With MTEB Benchmarks

```bash
guidellm benchmark embeddings \
  --target $GUIDELLM_REMOTE_URL \
  --model BAAI/bge-base-en-v1.5 \
  --enable-mteb \
  --mteb-tasks STS12 STS13 STSBenchmark \
  --outputs csv,html,json
```

#### Full Feature Test

```bash
guidellm benchmark embeddings \
  --target $GUIDELLM_REMOTE_URL \
  --model BAAI/bge-base-en-v1.5 \
  --enable-quality-validation \
  --baseline-model BAAI/bge-base-en-v1.5 \
  --enable-mteb \
  --mteb-tasks STS12 STS13 \
  --outputs csv,html,json \
  --max-requests 200 \
  --rate 20
```

## Test Coverage

The remote tests cover:

1. **Basic Functionality**
   - Connection to remote server
   - Basic embeddings generation
   - Request/response validation

2. **Quality Validation**
   - Cosine similarity against baseline model
   - Self-consistency checks
   - Tolerance thresholds (standard: 1e-2, MTEB: 5e-4)

3. **MTEB Integration**
   - Semantic Textual Similarity tasks
   - Score validation against published benchmarks
   - Task-specific metrics

4. **Encoding Formats**
   - Float array encoding
   - Base64 binary encoding
   - Format conversion and validation

5. **Request Parameters**
   - `truncate_prompt_tokens` parameter
   - `dimensions` parameter (for matryoshka models)
   - Model-specific options

## Expected Results

### Cosine Similarity Thresholds

When using the same model for baseline and target:
- **Expected:** > 0.99 (near-perfect similarity)
- **Acceptable:** > 0.95
- **Warning:** < 0.95 (indicates potential issue)

When using different models:
- **Expected:** > 0.85 (high semantic similarity)
- **Acceptable:** > 0.75
- **Variable:** Depends on model architectures

### MTEB Scores (BAAI/bge-base-en-v1.5)

Published benchmark scores for reference:
- **STS12:** ~72.3
- **STS13:** ~78.1
- **STSBenchmark:** ~81.2
- **Main Score:** ~75.5

Acceptable variance: ±2%

### Performance Metrics

Expected performance (BAAI/bge-base-en-v1.5):
- **Latency (p50):** 20-50ms
- **Latency (p95):** 50-100ms
- **Throughput:** 20-50 req/s (single GPU)

## Troubleshooting

### Server Connection Issues

```bash
# Check server is accessible
ping ec2-18-117-141-109.us-east-2.compute.amazonaws.com

# Check port is open
nc -zv ec2-18-117-141-109.us-east-2.compute.amazonaws.com 8000

# Check vLLM logs on server
ssh -i ~/mtahhan.pem ec2-user@ec2-18-117-141-109.us-east-2.compute.amazonaws.com
journalctl -u vllm -f
```

### Low Quality Scores

If cosine similarity is unexpectedly low:
1. Verify using same model for baseline and target
2. Check model was loaded correctly on server
3. Ensure no preprocessing differences
4. Check for version mismatches (vLLM, transformers)

### MTEB Test Failures

If MTEB scores differ significantly:
1. Check exact model version matches published benchmarks
2. Verify evaluation methodology matches MTEB standard
3. Consider statistical variance (±2% is normal)
4. Check for differences in tokenization/preprocessing

### Performance Issues

If latency is higher than expected:
1. Check GPU utilization on server
2. Verify no other processes competing for resources
3. Check batch size and concurrency settings
4. Monitor network latency between client and server

## Data Files

### Sample Embeddings Data

Create a test dataset for embeddings:

```json
[
  {"text": "This is a test sentence for embeddings."},
  {"text": "Machine learning models process text data."},
  {"text": "Semantic similarity measures text relatedness."},
  {"text": "Vector databases store embeddings efficiently."}
]
```

Save as `tests/remote/data/embeddings_test.json`

### Running with Custom Data

```bash
guidellm benchmark embeddings \
  --target $GUIDELLM_REMOTE_URL \
  --model BAAI/bge-base-en-v1.5 \
  --data tests/remote/data/embeddings_test.json \
  --outputs csv,html
```

## Continuous Integration

For automated remote testing in CI/CD:

```yaml
# .github/workflows/remote-embeddings-test.yml
name: Remote Embeddings Tests

on:
  workflow_dispatch:  # Manual trigger only

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Run remote tests
        env:
          GUIDELLM_REMOTE_URL: ${{ secrets.REMOTE_VLLM_URL }}
        run: |
          pytest tests/remote/test_embeddings_remote.py -v
```

## Security Notes

- SSH key (`~/mtahhan.pem`) should have restricted permissions (600)
- Remote server should use security groups to limit access
- Consider using VPN or bastion host for production deployments
- Don't commit SSH keys or credentials to repository
- Use environment variables for sensitive configuration

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [BGE Models](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [E5 Models](https://huggingface.co/intfloat/e5-mistral-7b-instruct)
