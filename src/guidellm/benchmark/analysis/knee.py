"""Throughput-knee and over-saturation analysis for concurrent benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.scheduler import ConcurrentStrategy, SchedulerUpdateAction
from guidellm.schemas import StandardBaseModel

MIN_POINTS = 5
MAX_TAIL_SLOPE_RATIO = 0.25
MIN_FIT_IMPROVEMENT = 0.50
MIN_THROUGHPUT_FRACTION = 0.85
FIT_EPSILON = 1e-12

FitCandidate = tuple[float, int, float, float, float, float, float, float, float]


class KneeResult(StandardBaseModel):
    """Result of fitting rising and saturation segments to a throughput curve."""

    status: Literal["ok", "no_knee"] = Field(description="Whether a knee was detected")
    reason: str = Field(description="Human-readable explanation of the result")
    knee: float | None = Field(
        default=None, description="Estimated segment intersection"
    )
    saturation_concurrency: float | None = Field(
        default=None,
        description="First measured concurrency at or above the estimated knee",
    )
    breakpoint_concurrency: float | None = Field(
        default=None,
        description="Measured concurrency used as the best fit breakpoint",
    )
    pre_slope: float | None = Field(
        default=None, description="Normalized throughput slope before the breakpoint"
    )
    tail_slope: float | None = Field(
        default=None, description="Normalized throughput slope after the breakpoint"
    )
    slope_ratio: float | None = Field(
        default=None, description="Tail slope divided by the pre-breakpoint slope"
    )
    fit_improvement: float | None = Field(
        default=None,
        description="Error reduction from the segmented fit versus one line",
    )
    throughput_fraction: float | None = Field(
        default=None,
        description="Breakpoint throughput divided by peak observed throughput",
    )


class SaturationPoint(StandardBaseModel):
    """Final over-saturation detector snapshot for one concurrency."""

    concurrency: float = Field(description="Measured concurrent stream count")
    is_over_saturated: bool | None = Field(
        description="Detector decision, or None when no detector was configured"
    )
    concurrent_slope: float | None = None
    concurrent_slope_moe: float | None = None
    concurrent_n: float | None = None
    ttft_slope: float | None = None
    ttft_slope_moe: float | None = None
    ttft_n: float | None = None
    ttft_violations: float | None = None


class SaturationBoundary(StandardBaseModel):
    """Over-saturation boundary inferred from completed concurrent benchmarks."""

    status: Literal["detected", "not_detected", "unavailable", "inconsistent"]
    reason: str
    points: tuple[SaturationPoint, ...]
    previous_safe_concurrency: float | None = None
    first_oversaturated_concurrency: float | None = None


class SaturationAssessment(StandardBaseModel):
    """Relationship between a throughput knee and temporal saturation evidence."""

    status: Literal[
        "corroborated",
        "compatible",
        "throughput_only",
        "oversaturation_boundary",
        "no_saturation",
        "disagreement",
    ]
    reason: str
    selection_center: float | None = None
    refinement_lower: float | None = None
    refinement_upper: float | None = None


class KneeAnalysis(StandardBaseModel):
    """Combined throughput-knee and temporal saturation analysis."""

    throughput: KneeResult
    saturation: SaturationBoundary
    assessment: SaturationAssessment


class AdaptiveConcurrencyPlan(StandardBaseModel):
    """Additional concurrent stream counts selected around saturation."""

    status: Literal["ready", "skipped"]
    reason: str
    concurrencies: tuple[int, ...] = ()
    selection_center: float | None = None
    anchor: int | None = None
    step: int | None = None
    candidate_concurrencies: tuple[int, ...] = ()
    excluded_concurrencies: tuple[int, ...] = ()


class KneeDetectionConclusion(StandardBaseModel):
    """Initial and final knee analyses plus the adaptive execution plan."""

    kind: Literal["knee_detection"] = "knee_detection"
    initial: KneeAnalysis
    adaptive_plan: AdaptiveConcurrencyPlan
    final: KneeAnalysis


def _line_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Return the intercept, slope, and squared error of an OLS line."""
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope = numerator / denominator if denominator else 0.0
    intercept = y_mean - slope * x_mean
    error = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return intercept, slope, error


def _prepare_points(
    concurrencies: Sequence[int | float],
    output_throughputs: Sequence[int | float | None],
) -> tuple[list[float], list[float]]:
    if len(concurrencies) != len(output_throughputs):
        raise ValueError("Concurrency and throughput lists must have the same length")

    by_concurrency: dict[float, list[float]] = {}
    for concurrency, throughput in zip(concurrencies, output_throughputs, strict=True):
        if throughput is None:
            continue
        concurrency_value = float(concurrency)
        throughput_value = float(throughput)
        if not math.isfinite(concurrency_value) or concurrency_value <= 0:
            raise ValueError(
                f"Concurrency must be a positive finite number: {concurrency!r}"
            )
        if not math.isfinite(throughput_value) or throughput_value < 0:
            raise ValueError(
                f"Throughput must be a non-negative finite number: {throughput!r}"
            )
        by_concurrency.setdefault(concurrency_value, []).append(throughput_value)

    prepared_concurrencies = sorted(by_concurrency)
    prepared_throughputs = [
        sum(by_concurrency[concurrency]) / len(by_concurrency[concurrency])
        for concurrency in prepared_concurrencies
    ]
    return prepared_concurrencies, prepared_throughputs


def _fit_candidates(
    normalized_xs: Sequence[float],
    normalized_ys: Sequence[float],
    throughputs: Sequence[float],
    peak_throughput: float,
    single_line_error: float,
) -> list[FitCandidate]:
    candidates: list[FitCandidate] = []
    candidate_stop = (
        len(normalized_xs) - 1
        if len(normalized_xs) == MIN_POINTS
        else len(normalized_xs) - 2
    )
    for index in range(2, candidate_stop):
        pre_intercept, pre_slope, pre_error = _line_fit(
            normalized_xs[: index + 1], normalized_ys[: index + 1]
        )
        tail_intercept, tail_slope, tail_error = _line_fit(
            normalized_xs[index:], normalized_ys[index:]
        )
        if pre_slope <= 0:
            continue

        total_error = pre_error + tail_error
        slope_ratio = tail_slope / pre_slope
        fit_improvement = 1.0 - total_error / single_line_error
        throughput_fraction = (
            throughputs[index] / peak_throughput if peak_throughput > 0 else 0.0
        )

        if slope_ratio > MAX_TAIL_SLOPE_RATIO:
            continue
        if fit_improvement < MIN_FIT_IMPROVEMENT:
            continue
        if throughput_fraction < MIN_THROUGHPUT_FRACTION:
            continue

        candidates.append(
            (
                total_error,
                index,
                pre_intercept,
                pre_slope,
                tail_intercept,
                tail_slope,
                slope_ratio,
                fit_improvement,
                throughput_fraction,
            )
        )
    return candidates


def find_throughput_knee(
    concurrencies: Sequence[int | float],
    output_throughputs: Sequence[int | float | None],
) -> KneeResult:
    """Find where output throughput changes from rising to substantially flatter.

    Duplicate concurrency measurements are averaged. Missing throughput values
    are ignored. A knee is accepted only when segmented regression improves the
    fit, the tail is substantially flatter, and the breakpoint has reached most
    of the observed peak throughput.

    :param concurrencies: Concurrent stream counts for the measurements
    :param output_throughputs: Successful output-token throughput measurements
    :return: Detected knee and fit diagnostics, or a no-knee explanation
    :raises ValueError: If the inputs have different lengths or invalid values
    """
    xs, ys = _prepare_points(concurrencies, output_throughputs)
    if len(xs) < MIN_POINTS:
        return KneeResult(
            status="no_knee",
            reason=(
                f"Need at least {MIN_POINTS} distinct concurrency points; "
                f"found {len(xs)}"
            ),
        )

    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min or y_max == y_min:
        return KneeResult(
            status="no_knee", reason="The throughput curve has no usable range"
        )

    normalized_xs = [(x - x_min) / (x_max - x_min) for x in xs]
    normalized_ys = [(y - y_min) / (y_max - y_min) for y in ys]

    _, _, single_line_error = _line_fit(normalized_xs, normalized_ys)
    if single_line_error <= FIT_EPSILON:
        return KneeResult(
            status="no_knee",
            reason="The throughput curve is approximately linear",
        )

    candidates = _fit_candidates(
        normalized_xs,
        normalized_ys,
        ys,
        y_max,
        single_line_error,
    )

    if not candidates:
        return KneeResult(
            status="no_knee",
            reason="No breakpoint satisfied the saturation thresholds",
        )

    (
        _,
        best_index,
        pre_intercept,
        pre_slope,
        tail_intercept,
        tail_slope,
        slope_ratio,
        fit_improvement,
        throughput_fraction,
    ) = min(candidates)

    slope_delta = pre_slope - tail_slope
    knee_normalized = (
        (tail_intercept - pre_intercept) / slope_delta
        if abs(slope_delta) > FIT_EPSILON
        else normalized_xs[best_index]
    )
    if not (
        normalized_xs[best_index - 1]
        <= knee_normalized
        <= normalized_xs[best_index + 1]
    ):
        knee_normalized = normalized_xs[best_index]

    knee = x_min + knee_normalized * (x_max - x_min)
    epsilon = (x_max - x_min) * 1e-9
    saturation_concurrency = next((x for x in xs if x >= knee - epsilon), xs[-1])

    return KneeResult(
        status="ok",
        reason="A rising segment followed by a substantially flatter tail was detected",
        knee=knee,
        saturation_concurrency=saturation_concurrency,
        breakpoint_concurrency=xs[best_index],
        pre_slope=pre_slope,
        tail_slope=tail_slope,
        slope_ratio=slope_ratio,
        fit_improvement=fit_improvement,
        throughput_fraction=throughput_fraction,
    )


def benchmark_concurrency(benchmark: GenerativeBenchmark) -> int:
    """Read the concurrent stream count from a completed benchmark.

    :param benchmark: Completed GuideLLM benchmark
    :return: Positive integer concurrent stream count
    :raises ValueError: If the benchmark did not use a concurrent strategy
    """
    strategy = benchmark.config.strategy
    if not isinstance(strategy, ConcurrentStrategy):
        raise ValueError(
            "Knee analysis requires concurrent benchmark strategies; "
            f"received {strategy.type_!r}"
        )
    return strategy.streams


def _constraint_action(benchmark: GenerativeBenchmark) -> SchedulerUpdateAction | None:
    state = benchmark.scheduler_state
    for actions in (
        state.scheduler_constraints,
        state.end_processing_constraints,
        state.end_queuing_constraints,
    ):
        action = actions.get("over_saturation")
        if action is not None:
            return action
    return None


def _optional_finite_number(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Over-saturation metadata {key!r} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"Over-saturation metadata {key!r} must be finite")
    return float(value)


def extract_saturation_boundary(
    benchmarks: Sequence[GenerativeBenchmark],
) -> SaturationBoundary:
    """Infer the safe-to-over-saturated boundary from native benchmark state.

    :param benchmarks: Completed concurrent GuideLLM benchmarks
    :return: Ordered detector points and the inferred saturation boundary
    :raises ValueError: If concurrency or detector metadata is malformed
    """
    points: list[SaturationPoint] = []
    seen_concurrencies: set[float] = set()
    for benchmark in benchmarks:
        concurrency = float(benchmark_concurrency(benchmark))
        if concurrency in seen_concurrencies:
            raise ValueError(f"Duplicate benchmark concurrency: {concurrency:g}")
        seen_concurrencies.add(concurrency)

        action = _constraint_action(benchmark)
        if action is None:
            points.append(
                SaturationPoint(concurrency=concurrency, is_over_saturated=None)
            )
            continue

        metadata: dict[str, object] = action.metadata
        is_over_saturated = metadata.get("is_over_saturated")
        if not isinstance(is_over_saturated, bool):
            raise ValueError(
                "Over-saturation metadata must contain boolean 'is_over_saturated'"
            )
        points.append(
            SaturationPoint(
                concurrency=concurrency,
                is_over_saturated=is_over_saturated,
                concurrent_slope=_optional_finite_number(metadata, "concurrent_slope"),
                concurrent_slope_moe=_optional_finite_number(
                    metadata, "concurrent_slope_moe"
                ),
                concurrent_n=_optional_finite_number(metadata, "concurrent_n"),
                ttft_slope=_optional_finite_number(metadata, "ttft_slope"),
                ttft_slope_moe=_optional_finite_number(metadata, "ttft_slope_moe"),
                ttft_n=_optional_finite_number(metadata, "ttft_n"),
                ttft_violations=_optional_finite_number(metadata, "ttft_violations"),
            )
        )

    sorted_points = tuple(sorted(points, key=lambda point: point.concurrency))
    available_points = [
        point for point in sorted_points if point.is_over_saturated is not None
    ]
    if not available_points:
        return SaturationBoundary(
            status="unavailable",
            reason="The benchmarks contain no over-saturation detector metadata",
            points=sorted_points,
        )

    oversaturated_points = [
        point for point in available_points if point.is_over_saturated is True
    ]
    if not oversaturated_points:
        return SaturationBoundary(
            status="not_detected",
            reason="The detector did not report over-saturation at any concurrency",
            points=sorted_points,
        )

    first_oversaturated = oversaturated_points[0].concurrency
    safe_before = [
        point.concurrency
        for point in available_points
        if point.is_over_saturated is False and point.concurrency < first_oversaturated
    ]
    previous_safe = max(safe_before) if safe_before else None
    safe_after = [
        point.concurrency
        for point in available_points
        if point.is_over_saturated is False and point.concurrency > first_oversaturated
    ]
    if safe_after:
        return SaturationBoundary(
            status="inconsistent",
            reason=(
                "The detector reported a safe concurrency above its first "
                "over-saturated concurrency"
            ),
            points=sorted_points,
            previous_safe_concurrency=previous_safe,
            first_oversaturated_concurrency=first_oversaturated,
        )

    return SaturationBoundary(
        status="detected",
        reason="The detector reported an over-saturated concurrency",
        points=sorted_points,
        previous_safe_concurrency=previous_safe,
        first_oversaturated_concurrency=first_oversaturated,
    )


def assess_saturation(
    throughput: KneeResult,
    saturation: SaturationBoundary,
) -> SaturationAssessment:
    """Combine a throughput knee with temporal over-saturation evidence.

    :param throughput: Result of fitting the throughput curve
    :param saturation: Boundary inferred from the over-saturation detector
    :return: Combined assessment and optional refinement bounds
    """
    if saturation.status == "inconsistent":
        return SaturationAssessment(
            status="disagreement",
            reason="Over-saturation states are not monotonic across concurrency",
        )

    if throughput.status == "ok":
        return _assess_detected_knee(throughput, saturation)

    if saturation.status == "detected":
        return SaturationAssessment(
            status="oversaturation_boundary",
            reason="No throughput knee was found; refine the detector boundary instead",
            refinement_lower=saturation.previous_safe_concurrency,
            refinement_upper=saturation.first_oversaturated_concurrency,
        )

    return SaturationAssessment(
        status="no_saturation",
        reason="Neither detector found saturation in the measured range",
    )


def _assess_detected_knee(
    throughput: KneeResult,
    saturation: SaturationBoundary,
) -> SaturationAssessment:
    knee = throughput.knee
    if knee is None:
        raise ValueError("A successful throughput knee result must contain a knee")
    if saturation.status != "detected":
        return SaturationAssessment(
            status="throughput_only",
            reason="Throughput saturated without an over-saturation boundary",
            selection_center=knee,
        )

    first_oversaturated = saturation.first_oversaturated_concurrency
    if first_oversaturated is None:
        raise ValueError(
            "A detected saturation boundary must contain its first boundary"
        )
    if first_oversaturated < knee:
        return SaturationAssessment(
            status="disagreement",
            reason="The detector reported over-saturation below the throughput knee",
        )

    previous_safe = saturation.previous_safe_concurrency
    if previous_safe is not None and previous_safe <= knee:
        return SaturationAssessment(
            status="corroborated",
            reason="The throughput knee falls inside the safe-to-overloaded bracket",
            selection_center=knee,
            refinement_lower=previous_safe,
            refinement_upper=first_oversaturated,
        )
    return SaturationAssessment(
        status="compatible",
        reason="Over-saturation occurs above the throughput knee",
        selection_center=knee,
        refinement_upper=first_oversaturated,
    )


def analyze_knee(benchmarks: Sequence[GenerativeBenchmark]) -> KneeAnalysis:
    """Analyze concurrent benchmarks using throughput and temporal evidence.

    :param benchmarks: Completed concurrent GuideLLM benchmarks
    :return: Throughput fit, detector boundary, and their combined assessment
    """
    concurrencies = [benchmark_concurrency(benchmark) for benchmark in benchmarks]
    throughputs = [
        benchmark.metrics.output_tokens_per_second.successful.mean
        for benchmark in benchmarks
    ]
    throughput_result = find_throughput_knee(concurrencies, throughputs)
    saturation_result = extract_saturation_boundary(benchmarks)
    return KneeAnalysis(
        throughput=throughput_result,
        saturation=saturation_result,
        assessment=assess_saturation(throughput_result, saturation_result),
    )


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validated_measured_concurrencies(
    measured_concurrencies: Sequence[int],
) -> set[int]:
    measured: set[int] = set()
    for concurrency in measured_concurrencies:
        _validate_positive_integer("Measured concurrencies", concurrency)
        measured.add(concurrency)
    return measured


def _adaptive_center(assessment: SaturationAssessment) -> float | None:
    if assessment.selection_center is not None:
        return assessment.selection_center
    lower = assessment.refinement_lower
    upper = assessment.refinement_upper
    if lower is not None and upper is not None:
        return (lower + upper) / 2
    return upper if upper is not None else lower


def _adaptive_grid(
    center: float,
    points_each_side: int,
    max_step: int,
) -> tuple[int, int, tuple[int, ...]]:
    step = 1
    anchor = max(1, math.floor(center + 0.5))
    for candidate_step in range(max_step, 0, -1):
        candidate_anchor = max(
            candidate_step,
            math.floor(center / candidate_step + 0.5) * candidate_step,
        )
        step = candidate_step
        anchor = candidate_anchor
        if anchor - points_each_side * step > 0:
            break
    candidates = tuple(
        sorted(
            {
                anchor + offset * step
                for offset in range(-points_each_side, points_each_side + 1)
                if anchor + offset * step > 0
            }
        )
    )
    return anchor, step, candidates


def generate_adaptive_concurrency_plan(
    analysis: KneeAnalysis,
    measured_concurrencies: Sequence[int],
    *,
    points_each_side: int = 5,
    max_step: int = 5,
) -> AdaptiveConcurrencyPlan:
    """Select unmeasured integer concurrency points around saturation.

    A clean ``max_step`` grid is preferred. The step is reduced near the bottom
    of the positive integer range so lower points remain valid.

    :param analysis: Knee and over-saturation assessment from the initial run
    :param measured_concurrencies: Concurrent stream counts already executed
    :param points_each_side: Maximum additional grid points on either side
    :param max_step: Largest concurrency grid step to consider
    :return: Ready plan with new concurrencies, or an explanation for skipping
    :raises ValueError: If configuration or measured concurrencies are invalid
    """
    _validate_positive_integer("points_each_side", points_each_side)
    _validate_positive_integer("max_step", max_step)
    measured = _validated_measured_concurrencies(measured_concurrencies)

    assessment = analysis.assessment
    if assessment.status in {"disagreement", "no_saturation"}:
        return AdaptiveConcurrencyPlan(
            status="skipped",
            reason=f"Cannot safely select adaptive concurrencies: {assessment.reason}",
        )

    center = _adaptive_center(assessment)
    if center is None:
        return AdaptiveConcurrencyPlan(
            status="skipped",
            reason="The saturation assessment did not provide a refinement center",
        )
    if not math.isfinite(center) or center <= 0:
        raise ValueError("Adaptive concurrency center must be positive and finite")

    anchor, step, candidates = _adaptive_grid(center, points_each_side, max_step)
    excluded = tuple(
        concurrency for concurrency in candidates if concurrency in measured
    )
    concurrencies = tuple(
        concurrency for concurrency in candidates if concurrency not in measured
    )
    if not concurrencies:
        return AdaptiveConcurrencyPlan(
            status="skipped",
            reason="All adaptive candidate concurrencies were already measured",
            selection_center=center,
            anchor=anchor,
            step=step,
            candidate_concurrencies=candidates,
            excluded_concurrencies=excluded,
        )

    return AdaptiveConcurrencyPlan(
        status="ready",
        reason="Selected unmeasured concurrencies around the saturation region",
        concurrencies=concurrencies,
        selection_center=center,
        anchor=anchor,
        step=step,
        candidate_concurrencies=candidates,
        excluded_concurrencies=excluded,
    )
