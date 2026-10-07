"""Tests for post-benchmark warning checks."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import cast

import pytest
from logot import Logot
from logot.logged import warning
from pydantic import BaseModel

from guidellm.benchmark.outputs.console import GenerativeBenchmarkerConsole
from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.benchmark.schemas.metrics import (
    GenerativeMetrics,
    SchedulerMetrics,
    root_dispatch_delay_distribution,
)
from guidellm.benchmark.schemas.warnings import BenchmarkWarning
from guidellm.benchmark.warnings import BenchmarkWarningAnalyzer
from guidellm.scheduler.strategies import (
    AsyncConstantStrategy,
    ConcurrentStrategy,
    TraceReplayStrategy,
)
from guidellm.schemas import (
    DistributionSummary,
    GenerativeRequestStats,
    RequestInfo,
    UsageMetrics,
)
from guidellm.schemas.benchmark.warnings import (
    MetricRef,
    WarningCondition,
    WarningRuleArgs,
)
from guidellm.utils.console import Console


def _request(start: float, targeted: float, preceding_nodes: int = 0):
    """
    Build a request whose dispatch delay is ``start - targeted``.

    ## WRITTEN BY AI ##
    """
    info = RequestInfo(
        request_id="req",
        status="completed",
        preceding_nodes=preceding_nodes,
    )
    info.timings.targeted_start = targeted
    info.timings.request_start = start
    info.timings.request_end = start + 1.0
    return GenerativeRequestStats(
        request_id="req",
        info=info,
        input_metrics=UsageMetrics(text_tokens=1),
        output_metrics=UsageMetrics(text_tokens=1),
    )


class _LatencyBreakdown(BaseModel):
    """Status breakdown with the one segment the tests read."""

    total: DistributionSummary


class _SchedulerFixture(BaseModel):
    """Scheduler metrics fields the warning walker is asked to read."""

    generation_delay: DistributionSummary | None = None
    dataset_incomplete: bool = False
    strategy_type: str = ""


class _MetricsFixture(BaseModel):
    """Generative metrics fields the warning walker is asked to read."""

    request_latency: _LatencyBreakdown | None = None
    time_to_first_token_ms: _LatencyBreakdown | None = None
    root_dispatch_delay: DistributionSummary | None = None


def _distribution(values: list[float]) -> DistributionSummary | None:
    """
    :param values: Samples, or an empty list when the metric does not apply
    :return: A summary, or None when there are no samples

    ## WRITTEN BY AI ##
    """
    if not values:
        return None
    return DistributionSummary.from_values(values)


def _generation_delay_rule(
    threshold: float,
    *,
    enabled: bool = True,
    note: str = "",
) -> WarningRuleArgs:
    """
    :param threshold: Fraction of mean request latency
    :param enabled: Whether the rule is evaluated
    :param note: Supplementary text stored on the warning
    :return: A generation-delay rule with an explicit threshold

    ## WRITTEN BY AI ##
    """
    return WarningRuleArgs(
        enabled=enabled,
        code="generation_delay",
        metric=MetricRef(name="generation_delay", statistic="mean"),
        relative_to=MetricRef(name="request_latency.total", statistic="mean"),
        threshold=threshold,
        note=note,
    )


def _analyze(
    rules: list[WarningRuleArgs],
    strategy,
    generation_delay: list[float],
    request_latency: list[float],
    requests: list[GenerativeRequestStats] | None = None,
    time_to_first_token_ms: list[float] | None = None,
    dataset_incomplete: bool = False,
):
    """
    Evaluate rules against the given generation and request-latency samples.

    Root dispatch delay is compiled the same way ``GenerativeMetrics.compile``
    compiles it, from the first turn of each conversation.

    ## WRITTEN BY AI ##
    """
    scheduler_metrics = _SchedulerFixture(
        generation_delay=_distribution(generation_delay),
        dataset_incomplete=dataset_incomplete,
        strategy_type=strategy.type_,
    )
    metrics = _MetricsFixture(
        request_latency=(
            None
            if not request_latency
            else _LatencyBreakdown(
                total=DistributionSummary.from_values(request_latency)
            )
        ),
        time_to_first_token_ms=(
            None
            if not time_to_first_token_ms
            else _LatencyBreakdown(
                total=DistributionSummary.from_values(time_to_first_token_ms)
            )
        ),
        root_dispatch_delay=root_dispatch_delay_distribution(requests or []),
    )
    return BenchmarkWarningAnalyzer(rules).analyze(
        scheduler_metrics=cast("SchedulerMetrics", scheduler_metrics),
        metrics=cast("GenerativeMetrics", metrics),
    )


@pytest.mark.sanity
def test_generation_delay_warns_as_a_fraction_of_request_latency():
    """
    Mean generation delay above the configured fraction of mean request latency warns.

    The same generation delay against a slow request does not. Repeating the
    samples does not change the ratio. The rule's note is copied onto the warning.

    ## WRITTEN BY AI ##
    """
    rule = _generation_delay_rule(
        0.25,
        note="Slow the dataset or raise the request rate. https://example.com/warnings",
    )
    fast = _analyze(
        [rule],
        ConcurrentStrategy(streams=4),
        [0.04, 0.04],
        [0.10, 0.10],
    )
    slow_request = _analyze(
        [rule],
        ConcurrentStrategy(streams=4),
        [0.04, 0.04],
        [10.0, 10.0],
    )
    long_run = _analyze(
        [rule],
        ConcurrentStrategy(streams=4),
        [0.04] * 20,
        [0.10] * 20,
    )

    assert len(fast) == 1
    warning = fast[0]
    assert warning.code == "generation_delay"
    assert warning.unit == "ratio"
    assert warning.observed == pytest.approx(0.4)
    assert warning.threshold == pytest.approx(0.25)
    assert warning.note == rule.note
    assert warning.sample_count == 2
    assert "40%" in warning.message
    assert "0.040s" in warning.message
    assert "0.100s" in warning.message
    assert slow_request == []
    assert len(long_run) == 1
    assert long_run[0].observed == pytest.approx(0.4)


@pytest.mark.sanity
def test_generation_delay_is_quiet_at_the_threshold_and_when_disabled():
    """
    A ratio equal to the threshold, and a disabled rule, produce no warning.

    ## WRITTEN BY AI ##
    """
    at_threshold = _analyze(
        [_generation_delay_rule(0.25)],
        ConcurrentStrategy(streams=1),
        [0.025],
        [0.10],
    )
    disabled = _analyze(
        [_generation_delay_rule(0.25, enabled=False)],
        ConcurrentStrategy(streams=1),
        [5.0],
        [0.10],
        [_request(start=10.0, targeted=0.0)],
    )

    assert at_threshold == []
    assert disabled == []


@pytest.mark.sanity
def test_unknown_metric_path_is_logged(logot: Logot):
    """
    A path that is not on the compiled schemas is logged and reported.

    A disabled rule stays quiet and does not log.

    ## WRITTEN BY AI ##
    """
    disabled = _analyze(
        [
            WarningRuleArgs(
                enabled=False,
                code="disabled_rule",
                metric=MetricRef(name="generation_dealy", statistic="mean"),
                threshold=0.1,
            )
        ],
        ConcurrentStrategy(streams=1),
        [0.01],
        [1.0],
    )

    assert disabled == []
    logot.assert_not_logged(
        warning(
            "Metric 'generation_dealy' in rule 'disabled_rule' was not found "
            "on the compiled scheduler or generative metrics."
        )
    )

    logged = _analyze(
        [
            WarningRuleArgs(
                code="generation_delay",
                metric=MetricRef(name="generation_dealy", statistic="mean"),
                threshold=0.1,
                note="Time to first token is above 5 ms.",
            )
        ],
        ConcurrentStrategy(streams=1),
        [0.01],
        [1.0],
    )

    assert len(logged) == 1
    assert logged[0].code == "unknown_metric"
    assert "generation_dealy" in logged[0].message
    assert "generation_delay" in logged[0].message
    assert logged[0].note == ""
    logot.assert_logged(
        warning(
            "Metric 'generation_dealy' in rule 'generation_delay' was not found "
            "on the compiled scheduler or generative metrics."
        )
    )


@pytest.mark.sanity
def test_generation_delay_warns_as_a_fraction_of_ttft():
    """
    Mean generation delay above 1% of mean time to first token warns.

    Time to first token is stored in milliseconds and scaled to seconds.

    ## WRITTEN BY AI ##
    """
    warned = _analyze(
        [
            WarningRuleArgs(
                code="generation_delay_ttft",
                metric=MetricRef(name="generation_delay", statistic="mean"),
                relative_to=MetricRef(
                    name="time_to_first_token_ms.total",
                    statistic="mean",
                    scale=0.001,
                ),
                threshold=0.01,
            )
        ],
        ConcurrentStrategy(streams=1),
        [0.002],
        [],
        time_to_first_token_ms=[100.0],
    )
    quiet = _analyze(
        [
            WarningRuleArgs(
                code="generation_delay_ttft",
                metric=MetricRef(name="generation_delay", statistic="mean"),
                relative_to=MetricRef(
                    name="time_to_first_token_ms.total",
                    statistic="mean",
                    scale=0.001,
                ),
                threshold=0.01,
            )
        ],
        ConcurrentStrategy(streams=1),
        [0.0005],
        [],
        time_to_first_token_ms=[100.0],
    )

    assert len(warned) == 1
    warning = warned[0]
    assert warning.code == "generation_delay_ttft"
    assert warning.unit == "ratio"
    assert warning.observed == pytest.approx(0.02)
    assert warning.threshold == pytest.approx(0.01)
    assert "2%" in warning.message
    assert quiet == []


@pytest.mark.sanity
def test_ratio_message_keeps_sub_millisecond_precision():
    """
    A duration below one millisecond is not rounded to 0.000s.

    ## WRITTEN BY AI ##
    """
    warnings = _analyze(
        [_generation_delay_rule(0.01)],
        ConcurrentStrategy(streams=1),
        [0.00024],
        [0.012],
    )

    assert len(warnings) == 1
    message = warnings[0].message
    assert "2%" in message
    assert "0.0002s" in message
    assert "0.012s" in message


def _dataset_incomplete_rule() -> WarningRuleArgs:
    """
    :return: The incomplete-dataset rule limited to trace replay

    ## WRITTEN BY AI ##
    """
    return WarningRuleArgs(
        code="dataset_incomplete",
        metric=MetricRef(name="dataset_incomplete"),
        threshold=0,
        when=WarningCondition(name="strategy_type", equals="trace"),
        note="The trace dataset was not fully loaded.",
    )


@pytest.mark.sanity
def test_dataset_incomplete_warns_only_for_trace():
    """
    A true dataset_incomplete value warns for trace and stays quiet otherwise.

    ## WRITTEN BY AI ##
    """
    warned = _analyze(
        [_dataset_incomplete_rule()],
        TraceReplayStrategy(),
        [],
        [],
        dataset_incomplete=True,
    )
    other_profile = _analyze(
        [_dataset_incomplete_rule()],
        ConcurrentStrategy(streams=1),
        [],
        [],
        dataset_incomplete=True,
    )
    finished = _analyze(
        [_dataset_incomplete_rule()],
        TraceReplayStrategy(),
        [],
        [],
        dataset_incomplete=False,
    )

    assert len(warned) == 1
    warning = warned[0]
    assert warning.code == "dataset_incomplete"
    assert warning.unit == "boolean"
    assert warning.observed == pytest.approx(1.0)
    assert "true" in warning.message
    assert warning.note == "The trace dataset was not fully loaded."
    assert other_profile == []
    assert finished == []


@pytest.mark.sanity
def test_trace_root_lateness_ignores_later_turns():
    """
    Only the first request of a trace conversation is compared with its target.

    ## WRITTEN BY AI ##
    """
    warnings = _analyze(
        [
            WarningRuleArgs(
                code="trace_root_late",
                metric=MetricRef(name="root_dispatch_delay", statistic="max"),
                threshold=0.1,
            )
        ],
        TraceReplayStrategy(),
        [0.01],
        [1.0],
        [
            _request(start=0.4, targeted=0.0, preceding_nodes=0),
            _request(start=20.0, targeted=0.0, preceding_nodes=1),
        ],
    )

    assert [warning.code for warning in warnings] == ["trace_root_late"]
    warning = warnings[0]
    assert warning.unit == "seconds"
    assert warning.observed == pytest.approx(0.4)
    assert warning.sample_count == 1
    assert "max root dispatch delay" in warning.message


@pytest.mark.sanity
def test_trace_root_lateness_uses_the_configured_statistic():
    """
    p95 is read from the root delays rather than the maximum.

    ## WRITTEN BY AI ##
    """
    requests = [_request(start=0.0, targeted=0.0) for _ in range(19)]
    requests.append(_request(start=5.0, targeted=0.0))
    p95_rule = WarningRuleArgs(
        code="trace_root_late",
        metric=MetricRef(name="root_dispatch_delay", statistic="p95"),
        threshold=0.1,
    )
    max_rule = WarningRuleArgs(
        code="trace_root_late",
        metric=MetricRef(name="root_dispatch_delay", statistic="max"),
        threshold=0.1,
    )

    quiet = _analyze(
        [p95_rule],
        TraceReplayStrategy(),
        [],
        [],
        requests,
    )
    warned = _analyze(
        [max_rule],
        TraceReplayStrategy(),
        [],
        [],
        requests,
    )

    assert quiet == []
    assert len(warned) == 1
    assert warned[0].observed == pytest.approx(5.0)
    assert warned[0].sample_count == 20


@pytest.mark.sanity
def test_first_turn_dispatch_delay_includes_every_profile():
    """
    First-turn dispatch delay is measured for any profile, not only trace replay.

    ## WRITTEN BY AI ##
    """
    warnings = _analyze(
        [
            WarningRuleArgs(
                code="trace_root_late",
                metric=MetricRef(name="root_dispatch_delay", statistic="max"),
                threshold=0.1,
            )
        ],
        ConcurrentStrategy(streams=1),
        [],
        [],
        [
            _request(start=10.0, targeted=0.0, preceding_nodes=0),
            _request(start=20.0, targeted=0.0, preceding_nodes=1),
        ],
    )

    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.observed == pytest.approx(10.0)
    assert warning.sample_count == 1


@pytest.mark.smoke
def test_warning_serializes_its_measurement():
    """
    The warning model dumps the fields needed to explain why it fired.

    ## WRITTEN BY AI ##
    """
    dumped = BenchmarkWarning(
        code="generation_delay",
        message="mean generation delay was 40% of mean request latency.",
        observed=0.4,
        threshold=0.25,
        unit="ratio",
        sample_count=4,
    ).model_dump()

    assert dumped == {
        "code": "generation_delay",
        "message": "mean generation delay was 40% of mean request latency.",
        "observed": 0.4,
        "threshold": 0.25,
        "unit": "ratio",
        "sample_count": 4,
        "note": "",
    }


@pytest.mark.sanity
def test_console_prints_warnings_grouped_by_strategy():
    """
    Console output lists each warning under the strategy that produced it.

    ## WRITTEN BY AI ##
    """
    buffer = StringIO()
    output = GenerativeBenchmarkerConsole(
        console=Console(
            file=buffer,
            width=120,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
        )
    )
    warning = BenchmarkWarning(
        code="generation_delay",
        message="mean generation delay was 40% of mean request latency.",
        observed=0.4,
        threshold=0.25,
        unit="ratio",
        sample_count=1,
        note="See https://example.com/warnings",
    )
    report = SimpleNamespace(
        benchmarks=[
            SimpleNamespace(
                warnings=[warning],
                config=SimpleNamespace(strategy=AsyncConstantStrategy(rate=2.0)),
            )
        ]
    )

    output.print_warnings(cast("GenerativeBenchmarksReport", report))

    printed = buffer.getvalue()
    assert "Warnings" in printed
    assert "\x1b[38;2;253;181;22mWarnings" in printed
    assert "constant@2.00" in printed
    assert "\x1b[38;2;253;181;22m⚠" in printed
    assert "mean generation delay was 40% of mean request latency." in printed
    assert "See https://example.com/warnings" in printed
