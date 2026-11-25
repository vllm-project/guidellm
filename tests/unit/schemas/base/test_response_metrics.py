"""Tests for typed and extensible vLLM response metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    VLLMRequestMetrics,
    VLLMSpeculativeDecodingMetrics,
)


@pytest.mark.regression
@pytest.mark.parametrize("level", ["summary", "detailed"])
def test_captured_metrics_round_trip(level):
    """Preserve captured timing fields, nested counters, and detailed extras on replay.

    ## WRITTEN BY AI ##
    """
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "schemas"
        / "fixtures"
        / f"vllm_spec_{level}.json"
    )
    raw = json.loads(path.read_text())["metrics"]
    raw["future_timing"] = {"value": 12}
    raw["speculative_decoding"]["future_acceptance"] = [1, 2]
    metrics = VLLMRequestMetrics.model_validate(raw)
    assert metrics.time_to_first_token_ms == raw["time_to_first_token_ms"]
    assert metrics.generation_time_ms == raw["generation_time_ms"]
    assert metrics.queue_time_ms == raw["queue_time_ms"]
    assert metrics.mean_itl_ms == raw["mean_itl_ms"]
    assert metrics.tokens_per_second == raw["tokens_per_second"]
    assert isinstance(metrics.speculative_decoding, VLLMSpeculativeDecodingMetrics)
    assert (
        metrics.speculative_decoding.mean_acceptance_length
        == raw["speculative_decoding"]["mean_acceptance_length"]
    )
    assert (
        metrics.speculative_decoding.draft_acceptance_rate
        == raw["speculative_decoding"]["draft_acceptance_rate"]
    )
    assert metrics.model_dump(exclude_unset=True) == raw
    response = GenerationResponse(
        request_id="spec", request_args=None, response_metrics=metrics
    )
    assert response.response_metrics is metrics
    stats = response.compile_stats(
        GenerationRequest(request_id="spec"),
        RequestInfo(
            request_id="spec",
            status="completed",
            timings=RequestTimings(resolve_start=1, resolve_end=2),
        ),
    )
    assert stats.response_metrics is metrics
    restored = GenerativeRequestStats.model_validate_json(stats.model_dump_json())
    assert isinstance(restored.response_metrics, VLLMRequestMetrics)
    # Default serialization includes optional fields as null; reloading marks them
    # as supplied. Compare typed values rather than the original fields-set.
    assert restored.response_metrics == metrics


@pytest.mark.regression
def test_timing_only_and_legacy_dictionary_inputs():
    """Accept old dictionary-shaped saved data and timing-only responses.

    ## WRITTEN BY AI ##
    """
    raw = {"queue_time_ms": 0, "future_metric": 42}
    response = GenerationResponse.model_validate(
        {"request_id": "timing", "request_args": None, "response_metrics": raw}
    )
    assert isinstance(response.response_metrics, VLLMRequestMetrics)
    assert response.response_metrics.queue_time_ms == 0
    assert response.response_metrics.speculative_decoding is None
    assert response.response_metrics.time_to_first_token_ms is None
    assert response.response_metrics.model_dump(exclude_unset=True) == raw
    assert (
        GenerationResponse.model_validate_json(
            response.model_dump_json()
        ).response_metrics
        == response.response_metrics
    )
    stats = GenerativeRequestStats.model_validate(
        {
            "request_id": "legacy",
            "info": {"request_id": "legacy"},
            "input_metrics": {},
            "output_metrics": {},
            "response_metrics": raw,
        }
    )
    assert isinstance(stats.response_metrics, VLLMRequestMetrics)
    assert stats.response_metrics.model_dump(exclude_unset=True) == raw


@pytest.mark.regression
@pytest.mark.parametrize(
    "field",
    [
        "num_spec_steps",
        "num_accepted_draft_tokens",
        "num_draft_tokens",
        "num_spec_tokens",
    ],
)
@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", float("nan"), float("inf"), []])
def test_invalid_counters_rejected_by_schema(field, bad):
    """Reject malformed counters at ingestion rather than coercing their values.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError, match=field):
        VLLMRequestMetrics.model_validate({"speculative_decoding": {field: bad}})


@pytest.mark.regression
@pytest.mark.parametrize("bad", [[2, -1, 1], [True, 1], "bad"])
@pytest.mark.parametrize(
    "field", ["acceptance_histogram", "per_step_accepted", "per_step_drafted"]
)
def test_invalid_histogram_rejected_by_schema(bad, field):
    """Validate histogram and detailed array counter types before calculations.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError, match=field):
        VLLMRequestMetrics.model_validate({"speculative_decoding": {field: bad}})
