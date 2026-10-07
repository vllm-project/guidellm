"""Tests for throughput-knee and over-saturation analysis."""

import math
from types import SimpleNamespace
from typing import cast

import pytest

from guidellm.benchmark.analysis import (
    KneeAnalysis,
    KneeResult,
    SaturationAssessment,
    SaturationBoundary,
    analyze_knee,
    assess_saturation,
    extract_saturation_boundary,
    find_throughput_knee,
    generate_adaptive_concurrency_plan,
)
from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.scheduler import (
    ConcurrentStrategy,
    SchedulerState,
    SchedulerUpdateAction,
)

CONCURRENCIES = [1, 10, 20, 30, 40, 50, 60]

pytestmark = pytest.mark.sanity


def _benchmark(
    concurrency: int,
    throughput: float,
    is_over_saturated: bool | None,
    *,
    constraint_group: str = "scheduler_constraints",
) -> GenerativeBenchmark:
    state = SchedulerState()
    if is_over_saturated is not None:
        action = SchedulerUpdateAction(
            metadata={
                "is_over_saturated": is_over_saturated,
                "concurrent_slope": 0.25,
                "concurrent_slope_moe": 0.1,
                "concurrent_n": 20,
                "ttft_slope": 0.02,
                "ttft_slope_moe": 0.01,
                "ttft_n": 20,
                "ttft_violations": 2,
            }
        )
        if constraint_group == "scheduler_constraints":
            state.scheduler_constraints = {"over_saturation": action}
        elif constraint_group == "end_processing_constraints":
            state.end_processing_constraints = {"over_saturation": action}
        elif constraint_group == "end_queuing_constraints":
            state.end_queuing_constraints = {"over_saturation": action}
        else:
            raise ValueError(f"Unsupported constraint group: {constraint_group}")

    benchmark = SimpleNamespace(
        config=SimpleNamespace(strategy=ConcurrentStrategy(streams=concurrency)),
        scheduler_state=state,
        metrics=SimpleNamespace(
            output_tokens_per_second=SimpleNamespace(
                successful=SimpleNamespace(mean=throughput)
            )
        ),
    )
    return cast("GenerativeBenchmark", benchmark)


def test_finds_clear_throughput_plateau() -> None:
    """Find a clear plateau and report its measured boundary.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 30, 30, 30])

    assert result.status == "ok"
    assert result.knee == pytest.approx(30)
    assert result.breakpoint_concurrency == 30
    assert result.saturation_concurrency == 30
    assert result.slope_ratio == pytest.approx(0)


def test_allows_small_tail_gain() -> None:
    """Accept a tail whose gain is small relative to the rising segment.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 31, 32, 33])

    assert result.status == "ok"
    assert result.knee == pytest.approx(30)
    assert result.slope_ratio == pytest.approx(0.1, abs=0.02)


def test_finds_plateau_despite_tail_noise() -> None:
    """Retain the knee when saturated throughput contains small noise.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 31, 29, 30])

    assert result.status == "ok"
    assert result.knee == pytest.approx(30.29, abs=0.01)
    assert result.breakpoint_concurrency == 30
    assert result.saturation_concurrency == 40


def test_rejects_linear_curve() -> None:
    """Do not claim a knee for an approximately linear curve.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 40, 50, 60])

    assert result.status == "no_knee"
    assert "linear" in result.reason


def test_requires_five_distinct_valid_points() -> None:
    """Require enough distinct concurrency points to fit both segments.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee([1, 2, 3, 4], [10, 20, 20, 20])

    assert result.status == "no_knee"
    assert "at least 5" in result.reason


def test_averages_duplicate_concurrency_measurements() -> None:
    """Average repeated measurements before fitting the curve.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(
        [1, 10, 20, 30, 30, 40, 50, 60],
        [1, 10, 20, 29, 31, 30, 30, 30],
    )

    assert result.status == "ok"
    assert result.breakpoint_concurrency == 30


def test_rejects_mismatched_input_lengths() -> None:
    """Reject curves whose concurrency and throughput counts differ.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError, match="same length"):
        find_throughput_knee([1, 2, 3], [10, 20])


@pytest.mark.parametrize(
    ("concurrencies", "throughputs", "message"),
    [
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], "Concurrency"),
        ([1, 2, 3, 4, 5], [1, 2, -1, 4, 5], "Throughput"),
        ([1, 2, 3, 4, 5], [1, 2, math.nan, 4, 5], "Throughput"),
    ],
)
def test_rejects_invalid_measurements(
    concurrencies: list[int],
    throughputs: list[float],
    message: str,
) -> None:
    """Reject non-positive concurrency and invalid throughput measurements.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError, match=message):
        find_throughput_knee(concurrencies, throughputs)


def test_short_rising_curve_does_not_claim_saturation() -> None:
    """Do not mistake a still-rising production curve for saturation.

    ## WRITTEN BY AI ##
    """
    result = find_throughput_knee(
        [1, 2, 5, 10, 25, 50, 75, 100, 200, 300],
        [
            209.63,
            419.93,
            843.22,
            1411.77,
            2911.77,
            4488.20,
            5400.58,
            6190.11,
            7829.66,
            8328.99,
        ],
    )

    assert result.status == "no_knee"
    assert "thresholds" in result.reason


def test_extracts_first_oversaturated_and_previous_safe_concurrency() -> None:
    """Order native detector snapshots and identify their transition.

    ## WRITTEN BY AI ##
    """
    result = extract_saturation_boundary(
        [
            _benchmark(100, 30, True),
            _benchmark(50, 25, False),
            _benchmark(200, 30, True),
            _benchmark(75, 29, False),
        ]
    )

    assert result.status == "detected"
    assert result.previous_safe_concurrency == 75
    assert result.first_oversaturated_concurrency == 100
    assert [point.concurrency for point in result.points] == [50, 75, 100, 200]
    assert result.points[0].ttft_slope == 0.02


def test_reports_when_saturation_is_not_detected() -> None:
    """Report that configured detector snapshots remained safe.

    ## WRITTEN BY AI ##
    """
    result = extract_saturation_boundary(
        [_benchmark(1, 1, False), _benchmark(2, 2, False)]
    )

    assert result.status == "not_detected"
    assert result.first_oversaturated_concurrency is None


def test_reports_when_detector_metadata_is_unavailable() -> None:
    """Distinguish an absent detector from a detector that stayed safe.

    ## WRITTEN BY AI ##
    """
    result = extract_saturation_boundary(
        [_benchmark(1, 1, None), _benchmark(2, 2, None)]
    )

    assert result.status == "unavailable"
    assert all(point.is_over_saturated is None for point in result.points)


def test_reads_end_processing_constraint_snapshot() -> None:
    """Use a terminal detector snapshot when it is stored at processing end.

    ## WRITTEN BY AI ##
    """
    result = extract_saturation_boundary(
        [
            _benchmark(
                100,
                30,
                True,
                constraint_group="end_processing_constraints",
            )
        ]
    )

    assert result.status == "detected"
    assert result.first_oversaturated_concurrency == 100


def test_rejects_duplicate_saturation_concurrency() -> None:
    """Reject ambiguous detector decisions for the same concurrency.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError, match="Duplicate benchmark concurrency"):
        extract_saturation_boundary(
            [_benchmark(50, 20, False), _benchmark(50, 21, True)]
        )


def test_rejects_malformed_detector_metadata() -> None:
    """Require the detector decision to retain its native boolean type.

    ## WRITTEN BY AI ##
    """
    benchmark = _benchmark(50, 20, False)
    benchmark.scheduler_state.scheduler_constraints["over_saturation"].metadata[
        "is_over_saturated"
    ] = "false"

    with pytest.raises(ValueError, match="boolean"):
        extract_saturation_boundary([benchmark])


def test_rejects_non_monotonic_saturation_states() -> None:
    """Mark a safe result above an overloaded result as inconsistent.

    ## WRITTEN BY AI ##
    """
    result = extract_saturation_boundary(
        [
            _benchmark(50, 20, False),
            _benchmark(75, 30, True),
            _benchmark(100, 31, False),
        ]
    )

    assert result.status == "inconsistent"


def test_corroborates_knee_inside_detector_boundary() -> None:
    """Corroborate a throughput knee inside the safe-to-overloaded bracket.

    ## WRITTEN BY AI ##
    """
    result = assess_saturation(
        KneeResult(status="ok", reason="test", knee=90),
        SaturationBoundary(
            status="detected",
            reason="test",
            points=(),
            previous_safe_concurrency=75,
            first_oversaturated_concurrency=100,
        ),
    )

    assert result.status == "corroborated"
    assert result.selection_center == 90
    assert result.refinement_lower == 75
    assert result.refinement_upper == 100


def test_keeps_throughput_knee_without_detector_boundary() -> None:
    """Keep a detected throughput knee when temporal evidence stays safe.

    ## WRITTEN BY AI ##
    """
    result = assess_saturation(
        KneeResult(status="ok", reason="test", knee=90),
        SaturationBoundary(status="not_detected", reason="test", points=()),
    )

    assert result.status == "throughput_only"
    assert result.selection_center == 90


def test_uses_detector_boundary_when_throughput_has_no_knee() -> None:
    """Expose the temporal boundary when throughput has no accepted knee.

    ## WRITTEN BY AI ##
    """
    result = assess_saturation(
        KneeResult(status="no_knee", reason="test"),
        SaturationBoundary(
            status="detected",
            reason="test",
            points=(),
            previous_safe_concurrency=75,
            first_oversaturated_concurrency=100,
        ),
    )

    assert result.status == "oversaturation_boundary"
    assert result.refinement_lower == 75
    assert result.refinement_upper == 100


def test_rejects_detector_boundary_below_throughput_knee() -> None:
    """Report disagreement when temporal saturation precedes the knee.

    ## WRITTEN BY AI ##
    """
    result = assess_saturation(
        KneeResult(status="ok", reason="test", knee=90),
        SaturationBoundary(
            status="detected",
            reason="test",
            points=(),
            first_oversaturated_concurrency=75,
        ),
    )

    assert result.status == "disagreement"
    assert result.selection_center is None


def test_combines_native_throughput_and_detector_evidence() -> None:
    """Build one analysis directly from completed GuideLLM benchmarks.

    ## WRITTEN BY AI ##
    """
    benchmarks = [
        _benchmark(1, 1, False),
        _benchmark(10, 10, False),
        _benchmark(20, 20, False),
        _benchmark(30, 30, False),
        _benchmark(40, 30, True),
        _benchmark(50, 30, True),
        _benchmark(60, 30, True),
    ]

    result = analyze_knee(benchmarks)

    assert result.throughput.status == "ok"
    assert result.saturation.status == "detected"
    assert result.assessment.status == "corroborated"


def _analysis_with_assessment(assessment: SaturationAssessment) -> KneeAnalysis:
    return KneeAnalysis(
        throughput=KneeResult(status="no_knee", reason="test"),
        saturation=SaturationBoundary(
            status="unavailable",
            reason="test",
            points=(),
        ),
        assessment=assessment,
    )


def test_generates_unmeasured_concurrencies_around_knee() -> None:
    """Generate a five-step grid and exclude previously measured points.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(
            status="throughput_only",
            reason="test",
            selection_center=100,
        )
    )

    result = generate_adaptive_concurrency_plan(
        analysis,
        [1, 2, 5, 10, 25, 50, 75, 100, 200, 300],
    )

    assert result.status == "ready"
    assert result.anchor == 100
    assert result.step == 5
    assert result.candidate_concurrencies == tuple(range(75, 126, 5))
    assert result.excluded_concurrencies == (75, 100)
    assert result.concurrencies == (80, 85, 90, 95, 105, 110, 115, 120, 125)


def test_reduces_adaptive_step_around_low_knee() -> None:
    """Reduce the step near one so every selected concurrency remains positive.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(
            status="throughput_only",
            reason="test",
            selection_center=5,
        )
    )

    result = generate_adaptive_concurrency_plan(
        analysis,
        [1, 2, 5, 10],
    )

    assert result.step == 1
    assert result.candidate_concurrencies == tuple(range(1, 11))
    assert result.concurrencies == (3, 4, 6, 7, 8, 9)


def test_uses_detector_boundary_midpoint_for_adaptive_center() -> None:
    """Use the midpoint of a detector boundary when no knee was accepted.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(
            status="oversaturation_boundary",
            reason="test",
            refinement_lower=75,
            refinement_upper=100,
        )
    )

    result = generate_adaptive_concurrency_plan(analysis, [75, 100])

    assert result.status == "ready"
    assert result.selection_center == 87.5
    assert result.anchor == 90


def test_skips_adaptive_plan_without_safe_saturation_center() -> None:
    """Skip adaptive execution when the two signals disagree.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(status="disagreement", reason="test")
    )

    result = generate_adaptive_concurrency_plan(analysis, [1, 2, 3])

    assert result.status == "skipped"
    assert result.concurrencies == ()


def test_skips_adaptive_plan_when_grid_is_already_measured() -> None:
    """Avoid rerunning concurrency points already present in the report.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(
            status="throughput_only",
            reason="test",
            selection_center=5,
        )
    )

    result = generate_adaptive_concurrency_plan(analysis, list(range(1, 11)))

    assert result.status == "skipped"
    assert result.excluded_concurrencies == tuple(range(1, 11))


def test_rejects_invalid_adaptive_plan_inputs() -> None:
    """Reject non-positive measured points and selection settings.

    ## WRITTEN BY AI ##
    """
    analysis = _analysis_with_assessment(
        SaturationAssessment(
            status="throughput_only",
            reason="test",
            selection_center=100,
        )
    )

    with pytest.raises(ValueError, match="Measured concurrencies"):
        generate_adaptive_concurrency_plan(analysis, [0])
    with pytest.raises(ValueError, match="points_each_side"):
        generate_adaptive_concurrency_plan(
            analysis,
            [1],
            points_each_side=0,
        )
