from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    StandardBaseDict,
    UsageMetrics,
)
from tests.unit.testing_utils import async_timeout


class TestGenerativeRequestStats:
    """High-coverage, concise tests for GenerativeRequestStats."""

    @pytest.fixture(
        params=[
            "short_completion",
            "long_chat",
            "single_token_output",
            "no_iterations_fallback",
        ],
    )
    def valid_instances(
        self, request: pytest.FixtureRequest
    ) -> tuple[GenerativeRequestStats, dict[str, Any]]:
        """
        Generate realistic test instances with statistically sampled metrics.

        Returns tuple of (GenerativeRequestStats instance, expected values dict).
        """
        case_id = request.param
        rng = np.random.default_rng(hash(case_id) % (2**32))

        # Define realistic scenarios based on common generative AI patterns
        if case_id == "short_completion":
            # Quick completion with moderate tokens
            request_type = "text_completions"
            prompt_tokens = 30
            output_tokens = 20
            request_start = 0.0
            # Time to first token: ~200ms (typical for small models)
            ttft_ms = rng.uniform(150, 250)
            first_iter = request_start + ttft_ms / 1000.0
            # Inter-token latency: ~50ms per token
            token_iters = 6
            inter_token_ms = rng.uniform(40, 60)
            last_iter = first_iter + (output_tokens - 1) * inter_token_ms / 1000.0
            request_end = last_iter + rng.uniform(0.05, 0.15)
            resolve_end = request_end

        elif case_id == "long_chat":
            # Long conversation with many tokens
            request_type = "chat_completions"
            prompt_tokens = 400
            output_tokens = 256
            request_start = 10.0
            # Longer TTFT for larger context
            ttft_ms = rng.uniform(2000, 3000)
            first_iter = request_start + ttft_ms / 1000.0
            # Consistent generation speed
            token_iters = 30
            inter_token_ms = rng.uniform(45, 55)
            last_iter = first_iter + (output_tokens - 1) * inter_token_ms / 1000.0
            request_end = last_iter + rng.uniform(0.1, 0.3)
            resolve_end = request_end

        elif case_id == "single_token_output":
            # Edge case: single token generation
            request_type = "text_completions"
            prompt_tokens = 5
            output_tokens = 1
            request_start = 5.0
            ttft_ms = rng.uniform(80, 120)
            first_iter = request_start + ttft_ms / 1000.0
            last_iter = first_iter
            token_iters = 1
            request_end = last_iter + rng.uniform(0.05, 0.1)
            resolve_end = request_end

        else:  # no_iterations_fallback
            # Fallback scenario with missing timing data
            request_type = "text_completions"
            prompt_tokens = 12
            output_tokens = 7
            request_start = None
            first_iter = None
            last_iter = None
            token_iters = 0
            request_end = 50.0
            resolve_end = 50.5

        # Build timings object via RequestInfo
        info = RequestInfo(request_id=case_id, status="completed")
        info.timings.request_start = request_start
        info.timings.resolve_start = resolve_end if request_start is None else None
        info.timings.first_token_iteration = first_iter
        info.timings.last_token_iteration = last_iter
        info.timings.token_iterations = token_iters
        info.timings.request_end = request_end
        info.timings.resolve_end = resolve_end

        stats = GenerativeRequestStats(
            request_id=case_id,
            request_type=request_type,
            info=info,
            input_metrics=UsageMetrics(text_tokens=prompt_tokens),
            output_metrics=UsageMetrics(text_tokens=output_tokens),
        )

        # Compute expected properties from the generated timings
        if request_start is not None and request_end is not None:
            expected_latency = request_end - request_start
        else:
            expected_latency = None

        expected_time_to_first_ms = (
            None
            if first_iter is None or request_start is None
            else (first_iter - request_start) * 1000.0
        )

        expected_time_per_output_ms = (
            None
            if request_start is None or last_iter is None or output_tokens in (None, 0)
            else (last_iter - request_start) * 1000.0 / output_tokens
        )

        if (
            first_iter is None
            or last_iter is None
            or output_tokens is None
            or output_tokens <= 1
        ):
            expected_inter_token_ms = None
        else:
            expected_inter_token_ms = (
                (last_iter - first_iter) * 1000.0 / (output_tokens - 1)
            )

        total_tokens = (prompt_tokens or 0) + (output_tokens or 0)
        expected_tokens_per_sec = (
            None
            if expected_latency is None or expected_latency == 0
            else total_tokens / expected_latency
        )
        expected_output_tokens_per_sec = (
            None
            if expected_latency is None
            or expected_latency == 0
            or output_tokens is None
            else output_tokens / expected_latency
        )

        expected_iter_tokens_per_iter = (
            None
            if output_tokens is None or output_tokens <= 1 or token_iters <= 1
            else (output_tokens - 1.0) / (token_iters - 1.0)
        )
        expected_output_tokens_per_iter = (
            None
            if output_tokens is None or token_iters < 1
            else output_tokens / token_iters
        )

        # Prompt timing chooses first token ts, or request_end_time fallback
        request_end_time_fallback = (
            request_end if request_end is not None else resolve_end
        )
        expected_prompt_timing = (
            None
            if resolve_end is None
            else (
                (first_iter if first_iter is not None else request_end_time_fallback),
                float(prompt_tokens or 0),
            )
        )

        # Output timings (first token + evenly spaced iterations) or single at end
        if resolve_end is None:
            expected_output_timings = None
        elif first_iter is None or last_iter is None or token_iters <= 1:
            expected_output_timings = [
                (
                    (last_iter if last_iter is not None else request_end_time_fallback),
                    float(output_tokens or 0),
                )
            ]
        else:
            # first token as 1, then spread the remaining (token_iters - 1) slots
            tok_per_iter = expected_iter_tokens_per_iter or 0.0
            iter_times = np.linspace(first_iter, last_iter, num=token_iters)[1:]
            expected_output_timings = [(first_iter, 1.0 * bool(output_tokens))] + [
                (float(tim), float(tok_per_iter)) for tim in iter_times
            ]

        expected_total_timings = (
            [] if expected_prompt_timing is None else [expected_prompt_timing]
        ) + ([] if expected_output_timings is None else list(expected_output_timings))

        expected: dict[str, Any] = {
            "request_start_time": (
                request_start if request_start is not None else resolve_end
            ),
            "request_end_time": (
                request_end if request_end is not None else resolve_end
            ),
            "request_latency": expected_latency,
            "prompt_tokens": prompt_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "time_to_first_token_ms": expected_time_to_first_ms,
            "time_per_output_token_ms": expected_time_per_output_ms,
            "inter_token_latency_ms": expected_inter_token_ms,
            "tokens_per_second": expected_tokens_per_sec,
            "output_tokens_per_second": expected_output_tokens_per_sec,
            "iter_tokens_per_iteration": expected_iter_tokens_per_iter,
            "output_tokens_per_iteration": expected_output_tokens_per_iter,
            "prompt_tokens_timing": expected_prompt_timing,
            "output_tokens_timings": (
                None
                if expected_output_timings is None
                else list(expected_output_timings)
            ),
            "total_tokens_timings": expected_total_timings,
            "first_token_iteration": first_iter,
            "last_token_iteration": last_iter,
            "token_iterations": token_iters,
        }
        return stats, expected

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface, inheritance, and key properties."""
        assert issubclass(GenerativeRequestStats, StandardBaseDict)
        assert hasattr(GenerativeRequestStats, "model_dump")
        assert hasattr(GenerativeRequestStats, "model_validate")

        # fields exposed
        fields = GenerativeRequestStats.model_fields
        for field_name in (
            "type_",
            "request_id",
            "request_type",
            "request_args",
            "output",
            "info",
            "input_metrics",
            "output_metrics",
        ):
            assert field_name in fields

        # computed properties
        for prop_name in (
            "request_start_time",
            "request_end_time",
            "request_latency",
            "prompt_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "time_to_first_token_ms",
            "time_per_output_token_ms",
            "inter_token_latency_ms",
            "tokens_per_second",
            "output_tokens_per_second",
            "iter_tokens_per_iteration",
            "output_tokens_per_iteration",
        ):
            assert hasattr(GenerativeRequestStats, prop_name)

        # regular properties
        for prop_name in (
            "first_token_iteration",
            "last_token_iteration",
            "token_iterations",
            "prompt_tokens_timing",
            "output_tokens_timings",
            "iter_tokens_timings",
            "total_tokens_timings",
        ):
            assert hasattr(GenerativeRequestStats, prop_name)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Initialization from realistic inputs."""
        instance, expected = valid_instances
        assert isinstance(instance, GenerativeRequestStats)
        assert instance.type_ == "generative_request_stats"
        assert instance.request_id
        assert instance.request_type in ("text_completions", "chat_completions")

        # Basic fields echo
        assert instance.prompt_tokens == expected["prompt_tokens"]
        assert instance.output_tokens == expected["output_tokens"]

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            GenerativeRequestStats()  # type: ignore[call-arg]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("request_id", None),
            ("request_id", 123),
            ("request_type", None),
            ("request_type", 456),
            ("info", None),
            ("info", "not_request_info"),
            ("input_metrics", None),
            ("input_metrics", "not_usage_metrics"),
            ("output_metrics", None),
            ("output_metrics", "not_usage_metrics"),
        ],
    )
    def test_invalid_initialization_values(self, field_name: str, bad_value: Any):
        """Type/None mismatches should raise."""
        info = RequestInfo(request_id="bad-1", status="completed")
        info.timings.resolve_end = 1.0
        base = {
            "request_id": "ok",
            "request_type": "text_completions",
            "info": info,
            "input_metrics": UsageMetrics(text_tokens=1),
            "output_metrics": UsageMetrics(text_tokens=1),
        }
        base[field_name] = bad_value
        with pytest.raises(ValidationError):
            GenerativeRequestStats(**base)  # type: ignore[arg-type]

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
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "time_to_first_token_ms",
            "time_per_output_token_ms",
            "inter_token_latency_ms",
            "tokens_per_second",
            "output_tokens_per_second",
            "iter_tokens_per_iteration",
            "output_tokens_per_iteration",
            "first_token_iteration",
            "last_token_iteration",
            "token_iterations",
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

        # prompt timing (all valid_instances have resolve_end set)
        got_ts, got_count = instance.prompt_tokens_timing
        exp_ts, exp_count = expected["prompt_tokens_timing"]
        assert pytest.approx(exp_ts, rel=1e-6, abs=1e-6) == got_ts
        assert got_count == exp_count

        # output timings (all valid_instances have resolve_end set)
        exp_output_timings = expected["output_tokens_timings"]
        got_output_timings = instance.output_tokens_timings
        assert len(got_output_timings) == len(exp_output_timings)
        for (got_t, got_v), (exp_t, exp_v) in zip(
            got_output_timings, exp_output_timings, strict=False
        ):
            assert pytest.approx(exp_t, rel=1e-6, abs=1e-6) == got_t
            assert pytest.approx(exp_v, rel=1e-6, abs=1e-6) == got_v

        # total timings (prompt + output)
        got_total_timings = instance.total_tokens_timings
        assert len(got_total_timings) == len(expected["total_tokens_timings"])

    @pytest.mark.sanity
    def test_error_when_no_resolve_end_for_end_time(self):
        """Accessing end-dependent fields without resolve_end should raise."""
        info = RequestInfo(request_id="no-end", status="completed")
        # request_end set but resolve_end missing -> property raises by design
        info.timings.request_end = 2.0
        instance = GenerativeRequestStats(
            request_id="no-end",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=3),
            output_metrics=UsageMetrics(text_tokens=2),
        )
        with pytest.raises(ValueError):
            _ = instance.request_end_time
        with pytest.raises(ValueError):
            _ = instance.prompt_tokens_timing
        with pytest.raises(ValueError):
            _ = instance.output_tokens_timings

    @pytest.mark.sanity
    def test_none_paths_for_latency_and_rates(self):
        """Ensure None is returned when required timing parts are missing."""
        info = RequestInfo(request_id="none-lat", status="completed")
        info.timings.resolve_end = 1.0  # minimal to avoid property error
        instance = GenerativeRequestStats(
            request_id="none-lat",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=4),
            output_metrics=UsageMetrics(text_tokens=6),
        )
        assert instance.request_latency is None
        assert instance.tokens_per_second is None
        assert instance.output_tokens_per_second is None
        assert instance.time_to_first_token_ms is None
        assert instance.time_per_output_token_ms is None
        assert instance.inter_token_latency_ms is None

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """model_dump / model_validate round-trip."""
        instance, _ = valid_instances
        dumped = instance.model_dump()
        assert dumped["type_"] == "generative_request_stats"
        rebuilt = GenerativeRequestStats.model_validate(dumped)
        assert rebuilt.request_id == instance.request_id
        assert rebuilt.request_type == instance.request_type

    @pytest.mark.sanity
    def test_optional_fields(self):
        """Test optional fields request_args and output."""
        info = RequestInfo(request_id="opt-test", status="completed")
        info.timings.resolve_end = 10.0

        # Without optional fields
        instance = GenerativeRequestStats(
            request_id="opt-test",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=10),
        )
        assert instance.request_args is None
        assert instance.output is None

        # With optional fields
        instance_with_opts = GenerativeRequestStats(
            request_id="opt-test-2",
            request_type="chat_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=10),
            request_args="temperature=0.7",
            output="Generated response text",
        )
        assert instance_with_opts.request_args == "temperature=0.7"
        assert instance_with_opts.output == "Generated response text"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("output_tokens_value", "token_iters_value", "expect_iter_tok_per_iter"),
        [
            (0, 5, None),
            (1, 5, None),
            (2, 1, None),
            (10, 2, (10 - 1) / (2 - 1)),
        ],
        ids=["zero_tokens", "single_token", "single_iter", "basic_case"],
    )
    def test_iter_token_formulas(
        self,
        output_tokens_value: int,
        token_iters_value: int,
        expect_iter_tok_per_iter: float | None,
    ):
        """Edge coverage for iteration-derived metrics."""
        info = RequestInfo(request_id="iter-calc", status="completed")
        info.timings.resolve_end = 9.0
        info.timings.token_iterations = token_iters_value
        stats = GenerativeRequestStats(
            request_id="iter-calc",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=3),
            output_metrics=UsageMetrics(text_tokens=output_tokens_value),
        )
        assert (
            stats.iter_tokens_per_iteration is None
            if expect_iter_tok_per_iter is None
            else pytest.approx(expect_iter_tok_per_iter, rel=1e-6, abs=1e-6)
            == stats.iter_tokens_per_iteration
        )

    @pytest.mark.sanity
    def test_iter_tokens_timings_property(self):
        """Test iter_tokens_timings property with various scenarios."""
        info = RequestInfo(request_id="iter-time", status="completed")
        info.timings.resolve_end = 10.0
        info.timings.first_token_iteration = 1.0
        info.timings.last_token_iteration = 5.0
        info.timings.token_iterations = 5

        # With valid iteration data
        stats = GenerativeRequestStats(
            request_id="iter-time",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=10),
        )
        iter_timings = stats.iter_tokens_timings
        assert len(iter_timings) == 4  # token_iterations - 1
        # Verify evenly spaced
        assert all(
            isinstance(tim, float) and isinstance(tok, float)
            for tim, tok in iter_timings
        )

        # Without iteration data (returns empty list)
        info_no_iter = RequestInfo(request_id="no-iter", status="completed")
        info_no_iter.timings.resolve_end = 10.0
        info_no_iter.timings.token_iterations = 0
        stats_no_iter = GenerativeRequestStats(
            request_id="no-iter",
            request_type="text_completions",
            info=info_no_iter,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=10),
        )
        assert stats_no_iter.iter_tokens_timings == []

    @pytest.mark.sanity
    def test_total_tokens_timings_property(self):
        """Test total_tokens_timings combines prompt and output timings."""
        info = RequestInfo(request_id="total-time", status="completed")
        info.timings.resolve_end = 10.0
        info.timings.request_start = 0.0
        info.timings.request_end = 10.0
        info.timings.first_token_iteration = 1.0
        info.timings.last_token_iteration = 5.0
        info.timings.token_iterations = 3

        stats = GenerativeRequestStats(
            request_id="total-time",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=6),
        )
        total_timings = stats.total_tokens_timings
        # Should have prompt timing + output timings
        assert len(total_timings) > 0
        # First should be prompt timing
        prompt_timing = stats.prompt_tokens_timing
        if prompt_timing:
            assert total_timings[0] == prompt_timing

    @pytest.mark.sanity
    def test_zero_division_edge_cases(self):
        """Test edge cases that could cause zero division errors."""
        info = RequestInfo(request_id="zero-div", status="completed")
        info.timings.resolve_end = 10.0
        info.timings.request_start = 10.0  # Same as end
        info.timings.request_end = 10.0

        stats = GenerativeRequestStats(
            request_id="zero-div",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=0),
        )

        # Zero latency should give None for rate calculations
        assert stats.request_latency == 0.0
        assert stats.tokens_per_second is None  # Division by zero avoided
        assert stats.output_tokens_per_second is None

        # Zero output tokens should give None for time_per_output_token_ms
        info_zero_tokens = RequestInfo(request_id="zero-tokens", status="completed")
        info_zero_tokens.timings.resolve_end = 10.0
        info_zero_tokens.timings.request_start = 0.0
        info_zero_tokens.timings.request_end = 10.0
        info_zero_tokens.timings.last_token_iteration = 5.0

        stats_zero_tokens = GenerativeRequestStats(
            request_id="zero-tokens",
            request_type="text_completions",
            info=info_zero_tokens,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(text_tokens=0),
        )
        assert stats_zero_tokens.time_per_output_token_ms is None

    @pytest.mark.regression
    def test_total_tokens_with_none_values(self):
        """Test total_tokens calculation when input or output tokens are None."""
        info = RequestInfo(request_id="none-tokens", status="completed")
        info.timings.resolve_end = 10.0

        # Both None returns None
        stats_both_none = GenerativeRequestStats(
            request_id="none-tokens-1",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(),
        )
        assert stats_both_none.total_tokens is None

        # One None, one with value
        stats_input_none = GenerativeRequestStats(
            request_id="none-tokens-2",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(text_tokens=10),
        )
        assert stats_input_none.total_tokens == 10

    @pytest.mark.regression
    def test_output_tokens_per_second_with_none_output(self):
        """Test output_tokens_per_second when output_tokens is None."""
        info = RequestInfo(request_id="none-output", status="completed")
        info.timings.resolve_end = 10.0
        info.timings.request_start = 0.0
        info.timings.request_end = 10.0

        stats = GenerativeRequestStats(
            request_id="none-output",
            request_type="text_completions",
            info=info,
            input_metrics=UsageMetrics(text_tokens=5),
            output_metrics=UsageMetrics(),  # No text_tokens
        )
        # Should return None when output_tokens is None, not crash
        assert stats.output_tokens_per_second is None

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(0.2)  # ensure no accidental indefinite waits if expanded later
    async def test_async_context_usage(self, valid_instances):
        """Light async smoke to satisfy async-timeout policy."""
        instance, expected = valid_instances
        await asyncio.sleep(0)  # yield
        assert instance.request_id
        # simple float compare in async path
        exp = expected["output_tokens_per_iteration"]
        got = instance.output_tokens_per_iteration
        if exp is None:
            assert got is None
        else:
            assert pytest.approx(exp, rel=1e-6, abs=1e-6) == got
