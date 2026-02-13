"""
Remote testing for embeddings support against a real vLLM server.

These tests require a running vLLM server and are designed to be run manually
or in a CI/CD environment with access to the remote server.

Set GUIDELLM_REMOTE_URL environment variable to the server URL before running.
Example: export GUIDELLM_REMOTE_URL=http://ec2-18-117-141-109.us-east-2.compute.amazonaws.com:8000
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest


@pytest.fixture(scope="module")
def remote_server_url() -> str:
    """
    Get remote server URL from environment and verify it's reachable.

    :return: The remote server URL
    :raises: pytest.skip if server is not configured or unreachable
    """
    url = os.getenv("GUIDELLM_REMOTE_URL")
    if not url:
        pytest.skip(
            "Remote server URL not configured. Set GUIDELLM_REMOTE_URL environment variable."
        )

    # Verify server is reachable
    try:
        response = httpx.get(f"{url}/health", timeout=10.0)
        if response.status_code != 200:
            pytest.skip(
                f"Remote server returned non-200 status: {response.status_code}"
            )
    except httpx.RequestError as e:
        pytest.skip(f"Remote server not reachable: {e}")
    except Exception as e:
        pytest.skip(f"Error checking remote server: {e}")

    return url


@pytest.fixture(scope="module")
def baseline_model() -> str:
    """
    Get baseline model for quality validation from environment.

    :return: The baseline model name
    """
    return os.getenv("GUIDELLM_BASELINE_MODEL", "ibm-granite/granite-embedding-english-r2")


@pytest.mark.remote
@pytest.mark.slow
def test_remote_server_health(remote_server_url: str):
    """Test that remote server health endpoint is accessible."""
    response = httpx.get(f"{remote_server_url}/health", timeout=10.0)
    assert response.status_code == 200


@pytest.mark.remote
@pytest.mark.slow
def test_remote_basic_embeddings(remote_server_url: str):
    """Test basic embeddings generation on remote server."""
    request_data = {
        "input": "This is a test sentence for embeddings.",
        "model": "ibm-granite/granite-embedding-english-r2",
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) == 1
    assert "embedding" in data["data"][0]
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) > 0
    assert "usage" in data


@pytest.mark.remote
@pytest.mark.slow
def test_remote_batch_embeddings(remote_server_url: str):
    """Test batch embeddings generation on remote server."""
    request_data = {
        "input": [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ],
        "model": "ibm-granite/granite-embedding-english-r2",
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Validate batch response
    assert "data" in data
    assert len(data["data"]) == 3

    # Check each embedding
    for i, embedding_obj in enumerate(data["data"]):
        assert "embedding" in embedding_obj
        assert "index" in embedding_obj
        assert embedding_obj["index"] == i
        assert isinstance(embedding_obj["embedding"], list)
        assert len(embedding_obj["embedding"]) > 0


@pytest.mark.remote
@pytest.mark.slow
def test_remote_float_encoding(remote_server_url: str):
    """Test float encoding format."""
    request_data = {
        "input": "Test sentence for float encoding.",
        "model": "ibm-granite/granite-embedding-english-r2",
        "encoding_format": "float",
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()

    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, list)
    assert all(isinstance(x, (int, float)) for x in embedding)


@pytest.mark.remote
@pytest.mark.slow
def test_remote_base64_encoding(remote_server_url: str):
    """Test base64 encoding format."""
    request_data = {
        "input": "Test sentence for base64 encoding.",
        "model": "ibm-granite/granite-embedding-english-r2",
        "encoding_format": "base64",
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()

    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, str)

    # Verify it's valid base64
    import base64

    try:
        decoded = base64.b64decode(embedding)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 embedding: {e}")


@pytest.mark.remote
@pytest.mark.slow
def test_remote_quality_validation(remote_server_url: str, baseline_model: str):
    """Test quality validation by comparing embeddings against baseline model."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Skip if sentence-transformers not installed
    try:
        baseline = SentenceTransformer(baseline_model)
    except Exception as e:
        pytest.skip(f"Could not load baseline model: {e}")

    test_text = "Machine learning models process text data efficiently."

    # Get baseline embedding
    baseline_embedding = baseline.encode(test_text)

    # Get remote server embedding
    request_data = {
        "input": test_text,
        "model": baseline_model,
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()
    remote_embedding = np.array(data["data"][0]["embedding"])

    # Compute cosine similarity
    cosine_sim = float(
        np.dot(baseline_embedding, remote_embedding)
        / (np.linalg.norm(baseline_embedding) * np.linalg.norm(remote_embedding))
    )

    # When using same model, should have very high similarity
    # Allow some tolerance for numerical differences
    assert cosine_sim > 0.95, f"Cosine similarity too low: {cosine_sim}"

    # Ideally should be > 0.99 for same model
    if cosine_sim < 0.99:
        pytest.warn(
            f"Cosine similarity is acceptable but lower than ideal: {cosine_sim}"
        )


@pytest.mark.remote
@pytest.mark.slow
def test_remote_self_consistency(remote_server_url: str):
    """Test that same input produces same embedding (self-consistency)."""
    import numpy as np

    test_text = "Semantic similarity measures text relatedness."
    request_data = {
        "input": test_text,
        "model": "ibm-granite/granite-embedding-english-r2",
    }

    # Get embedding twice
    embeddings = []
    for _ in range(2):
        response = httpx.post(
            f"{remote_server_url}/v1/embeddings",
            json=request_data,
            timeout=30.0,
        )
        assert response.status_code == 200
        data = response.json()
        embeddings.append(np.array(data["data"][0]["embedding"]))

    # Compute cosine similarity between the two embeddings
    cosine_sim = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )

    # Should be exactly 1.0 or very close (deterministic model)
    assert cosine_sim > 0.9999, f"Self-consistency check failed: {cosine_sim}"


@pytest.mark.remote
@pytest.mark.slow
def test_remote_truncation(remote_server_url: str):
    """Test truncate_prompt_tokens parameter."""
    # Create a long text that will need truncation
    long_text = " ".join(["test sentence"] * 100)

    request_data = {
        "input": long_text,
        "model": "ibm-granite/granite-embedding-english-r2",
        "truncate_prompt_tokens": 128,  # Truncate to 128 tokens
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Should succeed with truncation
    assert "data" in data
    assert len(data["data"]) == 1
    assert "embedding" in data["data"][0]

    # Usage should reflect truncation
    if "usage" in data:
        usage = data["usage"]
        if "prompt_tokens" in usage:
            # Tokens should be <= truncate limit (allowing for special tokens)
            assert usage["prompt_tokens"] <= 140  # Some buffer for special tokens


@pytest.mark.remote
@pytest.mark.slow
@pytest.mark.mteb
def test_remote_mteb_evaluation(remote_server_url: str, baseline_model: str):
    """Test MTEB benchmark evaluation on remote server (lightweight test)."""
    try:
        from sentence_transformers import SentenceTransformer
        import mteb
    except ImportError:
        pytest.skip("mteb or sentence-transformers not installed")

    # Use a very small subset for testing
    # Real MTEB evaluation would be more comprehensive
    test_texts = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
    ]

    # Get embeddings from remote server
    request_data = {
        "input": test_texts,
        "model": baseline_model,
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=60.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Verify we got embeddings for all texts
    assert len(data["data"]) == len(test_texts)

    # Compute simple semantic similarity checks
    import numpy as np

    embeddings = [np.array(item["embedding"]) for item in data["data"]]

    # Sentences 0 and 1 should be similar (both about eating)
    cos_01 = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )

    # Sentences 0 and 3 should be less similar (eating vs riding)
    cos_03 = float(
        np.dot(embeddings[0], embeddings[3])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[3]))
    )

    # Semantic similarity check: related sentences should be more similar
    assert cos_01 > cos_03, "Related sentences should have higher similarity"


@pytest.mark.remote
@pytest.mark.slow
def test_remote_performance_latency(remote_server_url: str):
    """Test that remote server latency is within acceptable bounds."""
    import time

    test_text = "Performance test sentence for latency measurement."
    request_data = {
        "input": test_text,
        "model": "ibm-granite/granite-embedding-english-r2",
    }

    # Warm-up request
    httpx.post(f"{remote_server_url}/v1/embeddings", json=request_data, timeout=30.0)

    # Measure latency over multiple requests
    latencies = []
    for _ in range(10):
        start_time = time.time()
        response = httpx.post(
            f"{remote_server_url}/v1/embeddings",
            json=request_data,
            timeout=30.0,
        )
        latency = time.time() - start_time
        assert response.status_code == 200
        latencies.append(latency)

    # Calculate statistics
    mean_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    # Check latency is reasonable (adjust thresholds based on your setup)
    assert (
        mean_latency < 1.0
    ), f"Mean latency too high: {mean_latency:.3f}s"  # Should be < 1s
    assert (
        p95_latency < 2.0
    ), f"P95 latency too high: {p95_latency:.3f}s"  # Should be < 2s

    print(f"\nLatency stats: mean={mean_latency:.3f}s, p95={p95_latency:.3f}s")


@pytest.mark.remote
@pytest.mark.slow
def test_remote_error_handling(remote_server_url: str):
    """Test that server properly handles invalid requests."""
    # Test missing required field
    request_data = {
        "model": "ibm-granite/granite-embedding-english-r2",
        # Missing "input" field
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    # Should return error status
    assert response.status_code >= 400

    # Test invalid encoding format
    request_data = {
        "input": "Test",
        "model": "ibm-granite/granite-embedding-english-r2",
        "encoding_format": "invalid_format",
    }

    response = httpx.post(
        f"{remote_server_url}/v1/embeddings",
        json=request_data,
        timeout=30.0,
    )

    # Should return error status
    assert response.status_code >= 400


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
