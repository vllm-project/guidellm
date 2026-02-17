from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    EmbeddingsRequestStats,
    RequestInfo,
    StandardBaseDict,
    UsageMetrics,
)
from tests.unit.testing_utils import async_timeout


class TestEmbeddingsRequestStats:
    """High-coverage, concise tests for EmbeddingsRequestStats."""

    @pytest.fixture(
        params=[
            "short_embedding",
            "long_embedding",
            "batch_embedding",
            "float_encoding",
            "base64_encoding",
            "with_cosine_similarity",
        ],
    )
    def valid_instances(
        self, request: pytest.FixtureRequest
    ) -> tuple[EmbeddingsRequestStats, dict[str, Any]]:
        """
        Generate realistic test instances for embeddings requests.

        Returns tuple of (EmbeddingsRequestStats instance, expected values dict).
        """
        case_id = request.param
        rng = np.random.default_rng(hash(case_id) % (2**32))

        # Define realistic scenarios based on common embeddings patterns
        if case_id == "short_embedding":
            # Quick embedding with few tokens
            prompt_tokens = 10
            request_start = 0.0
            # Embeddings are faster than generative (no output tokens)
            request_end = request_start + rng.uniform(0.05, 0.15)
            resolve_end = request_end
            encoding_format = "float"
            cosine_similarity = None

        elif case_id == "long_embedding":
            # Longer text embedding
            prompt_tokens = 512
            request_start = 5.0
            # Proportional to input size
            request_end = request_start + rng.uniform(0.3, 0.6)
            resolve_end = request_end
            encoding_format = "float"
            cosine_similarity = None

        elif case_id == "batch_embedding":
            # Batch processing
            prompt_tokens = 150
            request_start = 10.0
            request_end = request_start + rng.uniform(0.2, 0.4)
            resolve_end = request_end
            encoding_format = "float"
            cosine_similarity = None

        elif case_id == "float_encoding":
            # Float encoding (default)
            prompt_tokens = 50
            request_start = 0.0
            request_end = request_start + rng.uniform(0.1, 0.2)
            resolve_end = request_end
            encoding_format = "float"
            cosine_similarity = None

        elif case_id == "base64_encoding":
            # Base64 encoding
            prompt_tokens = 50
            request_start = 0.0
            request_end = request_start + rng.uniform(0.1, 0.2)
            resolve_end = request_end
            encoding_format = "base64"
            cosine_similarity = None

        else:  # with_cosine_similarity
            # With quality validation
            prompt_tokens = 25
            request_start = 0.0
            request_end = request_start + rng.uniform(0.08, 0.18)
            resolve_end = request_end
            encoding_format = "float"
            # Realistic cosine similarity (0.95-0.99 for good models)
            cosine_similarity = rng.uniform(0.95, 0.99)

        # Build timings object via RequestInfo
        info = RequestInfo(request_id=case_id, status="completed")
        info.timings.request_start = request_start
        info.timings.request_end = request_end
        info.timings.resolve_end = resolve_end

        stats = EmbeddingsRequestStats(
            request_id=case_id,
            info=info,
            input_metrics=UsageMetrics(text_tokens=prompt_tokens),
            cosine_similarity=cosine_similarity,
            encoding_format=encoding_format,
        )

        # Compute expected properties
        expected_latency = (
            request_end - request_start
            if request_start is not None
            else None
        )

        expected: dict[str, Any] = {
            "request_start_time": (
                request_start if request_start is not None else resolve_end
            ),
            "request_end_time": (
                request_end if request_end is not None else resolve_end
            ),
            "request_latency": expected_latency,
            "prompt_tokens": prompt_tokens,
            "cosine_similarity": cosine_similarity,
            "encoding_format": encoding_format,
        }
        return stats, expected

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface, inheritance, and key properties."""
        assert issubclass(EmbeddingsRequestStats, StandardBaseDict)
        assert hasattr(EmbeddingsRequestStats, "model_dump")
        assert hasattr(EmbeddingsRequestStats, "model_validate")

        # fields exposed
        fields = EmbeddingsRequestStats.model_fields
        for field_name in (
            "type_",
            "request_id",
            "request_args",
            "response_id",
            "info",
            "input_metrics",
            "cosine_similarity",
            "encoding_format",
        ):
            assert field_name in fields

        # computed properties
        for prop_name in (
            "request_start_time",
            "request_end_time",
            "request_latency",
            "prompt_tokens",
        ):
            assert hasattr(EmbeddingsRequestStats, prop_name)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Initialization from realistic inputs."""
        instance, expected = valid_instances
        assert isinstance(instance, EmbeddingsRequestStats)
        assert instance.type_ == "embeddings_request_stats"
        assert instance.request_id

        # Basic fields echo
        assert instance.prompt_tokens == expected["prompt_tokens"]
        assert instance.encoding_format == expected["encoding_format"]
        if expected["cosine_similarity"] is not None:
            assert instance.cosine_similarity == pytest.approx(
                expected["cosine_similarity"], rel=1e-6, abs=1e-6
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            EmbeddingsRequestStats()  # type: ignore[call-arg]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("request_id", None),
            ("request_id", 123),
            ("info", None),
            ("info", "not_request_info"),
            ("input_metrics", None),
            ("input_metrics", "not_usage_metrics"),
            ("cosine_similarity", "not_a_float"),
            ("encoding_format", 123),
        ],
    )
    def test_invalid_initialization_values(self, field_name: str, bad_value: Any):
        """Type/None mismatches should raise."""
        info = RequestInfo(request_id="bad-1", status="completed")
        info.timings.resolve_end = 1.0
        base = {
            "request_id": "ok",
            "info": info,
            "input_metrics": UsageMetrics(text_tokens=1),
        }
        base[field_name] = bad_value
        with pytest.raises(ValidationError):
            EmbeddingsRequestStats(**base)  # type: ignore[arg-type]

    @pytest.mark.regression
    def test_computed_properties_match_expected(self, valid_instances):
        """All computed properties should match precomputed expectations."""
        instance, expected = valid_instances

        # direct scalar comparisons
        for key in (
            "request_start_time",
            "request_end_time",
            "request_latency",
            "prompt_tokens",
        ):
            got = getattr(instance, key)
            exp = expected[key]
            if isinstance(exp, float):
                # tolerant float compare
                assert (got is None and exp is None) or pytest.approx(
                    exp, rel=1e-6, abs=1e-6
                ) == got
            else:
                assert got == exp

    @pytest.mark.sanity
    def test_none_paths_for_latency(self):
        """Ensure None is returned when required timing parts are missing."""
        info = RequestInfo(request_id="none-lat", status="completed")
        info.timings.resolve_end = 1.0  # minimal to avoid property error
        instance = EmbeddingsRequestStats(
            request_id="none-lat",
            info=info,
            input_metrics=UsageMetrics(text_tokens=10),
        )
        assert instance.request_latency is None

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """model_dump / model_validate round-trip."""
        instance, _ = valid_instances
        dumped = instance.model_dump()
        assert dumped["type_"] == "embeddings_request_stats"
        rebuilt = EmbeddingsRequestStats.model_validate(dumped)
        assert rebuilt.request_id == instance.request_id
        assert rebuilt.prompt_tokens == instance.prompt_tokens
        assert rebuilt.encoding_format == instance.encoding_format

    @pytest.mark.sanity
    def test_optional_fields(self):
        """Test optional fields request_args, cosine_similarity."""
        info = RequestInfo(request_id="opt-test", status="completed")
        info.timings.resolve_end = 10.0

        # Without optional fields
        instance = EmbeddingsRequestStats(
            request_id="opt-test",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
        )
        assert instance.request_args is None
        assert instance.cosine_similarity is None
        assert instance.encoding_format == "float"  # default

        # With optional fields
        instance_with_opts = EmbeddingsRequestStats(
            request_id="opt-test-2",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            request_args="dimensions=384",
            cosine_similarity=0.987,
            encoding_format="base64",
        )
        assert instance_with_opts.request_args == "dimensions=384"
        assert instance_with_opts.cosine_similarity == 0.987
        assert instance_with_opts.encoding_format == "base64"

    @pytest.mark.sanity
    def test_encoding_format_values(self):
        """Test valid encoding format values."""
        info = RequestInfo(request_id="enc-test", status="completed")
        info.timings.resolve_end = 10.0

        # Float encoding
        instance_float = EmbeddingsRequestStats(
            request_id="enc-float",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            encoding_format="float",
        )
        assert instance_float.encoding_format == "float"

        # Base64 encoding
        instance_base64 = EmbeddingsRequestStats(
            request_id="enc-base64",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            encoding_format="base64",
        )
        assert instance_base64.encoding_format == "base64"

    @pytest.mark.sanity
    def test_cosine_similarity_range(self):
        """Test cosine similarity values within expected range."""
        info = RequestInfo(request_id="cos-test", status="completed")
        info.timings.resolve_end = 10.0

        # Valid cosine similarity values (-1 to 1)
        for cos_val in [-1.0, -0.5, 0.0, 0.5, 0.99, 1.0]:
            instance = EmbeddingsRequestStats(
                request_id=f"cos-{cos_val}",
                info=info,
                input_metrics=UsageMetrics(text_tokens=5),
                cosine_similarity=cos_val,
            )
            assert instance.cosine_similarity == pytest.approx(cos_val, abs=1e-6)

    @pytest.mark.regression
    def test_zero_division_edge_cases(self):
        """Test edge cases that could cause zero division errors."""
        info = RequestInfo(request_id="zero-div", status="completed")
        info.timings.resolve_end = 10.0
        info.timings.request_start = 10.0  # Same as end
        info.timings.request_end = 10.0

        stats = EmbeddingsRequestStats(
            request_id="zero-div",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
        )

        # Zero latency should be returned as 0.0 (not None, no division error)
        assert stats.request_latency == 0.0

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(0.2)
    async def test_async_context_usage(self, valid_instances):
        """Light async smoke to satisfy async-timeout policy."""
        instance, expected = valid_instances
        await asyncio.sleep(0)  # yield
        assert instance.request_id
        assert instance.prompt_tokens == expected["prompt_tokens"]
        assert instance.encoding_format == expected["encoding_format"]
