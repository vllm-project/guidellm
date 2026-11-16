"""Unit tests for warmup and cooldown transient phase boundaries."""

from __future__ import annotations

import pytest

from guidellm.scheduler import SchedulerProgress, SchedulerState
from guidellm.schemas import RequestInfo, RequestTimings
from guidellm.schemas.benchmark import TransientPhaseConfig


def _run_synchronous(
    total_requests: int,
    warmup: TransientPhaseConfig,
    cooldown: TransientPhaseConfig,
) -> tuple[float | None, float | None, list[tuple[float, float]]]:
    """
    Feed a synchronous run through the phase configs the way the accumulator does.

    Each request sends an in_progress update and then a completed update, and
    request i runs over [10 * i, 10 * i + 5]. The scheduler only counts a request
    as processed on its terminal update. As in
    GenerativeBenchmarkTimings.update_estimate, the most recent transition time
    returned for each phase wins.

    :return: Tuple of (measure start, measure end, (start, end) of every request)
    """
    state = SchedulerState(
        start_time=0.0,
        progress=SchedulerProgress(
            total_requests=total_requests, remaining_requests=total_requests
        ),
    )
    measure_start: float | None = None
    measure_end: float | None = None
    requests: list[tuple[float, float]] = []

    for index in range(1, total_requests + 1):
        start, end = 10.0 * index, 10.0 * index + 5.0
        requests.append((start, end))
        updates = [
            RequestInfo(
                status="in_progress", timings=RequestTimings(request_start=start)
            ),
            RequestInfo(
                status="completed",
                timings=RequestTimings(request_start=start, request_end=end),
            ),
        ]
        for info in updates:
            if info.status == "completed":
                state.processed_requests += 1
                state.progress.remaining_requests = (
                    total_requests - state.processed_requests
                )
            _, start_time = warmup.compute_transition_time(info, state, "start")
            if start_time is not None:
                measure_start = start_time
            _, end_time = cooldown.compute_transition_time(info, state, "end")
            if end_time is not None:
                measure_end = end_time

    return measure_start, measure_end, requests


def _count_measured(
    measure_start: float | None,
    measure_end: float | None,
    requests: list[tuple[float, float]],
) -> int:
    lower = measure_start if measure_start is not None else float("-inf")
    upper = measure_end if measure_end is not None else float("inf")
    return sum(1 for start, end in requests if start >= lower and end <= upper)


class TestTransientPhaseConfigRequestTransitions:
    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("warmup", "cooldown", "expected_measured"),
        [
            (None, None, 10),
            (2, None, 8),
            (None, 2, 8),
            (2, 2, 6),
            (1, 1, 8),
        ],
    )
    def test_request_phases_exclude_exact_count(
        self, warmup, cooldown, expected_measured
    ):
        """
        Exclude exactly k requests for a request-count warmup or cooldown of k.

        Regression test for issue #1306, where each phase excluded k + 1.

        ## WRITTEN BY AI ##
        """
        measure_start, measure_end, requests = _run_synchronous(
            total_requests=10,
            warmup=TransientPhaseConfig(mode="requests", value=warmup),
            cooldown=TransientPhaseConfig(mode="requests", value=cooldown),
        )

        assert _count_measured(measure_start, measure_end, requests) == (
            expected_measured
        )

    @pytest.mark.regression
    def test_warmup_starts_at_first_measured_request(self):
        """
        Set the warmup boundary to the start of request k + 1 and keep it there.

        The in_progress update of request k + 2 arrives while the processed count
        is still k + 1 and must not move the boundary.

        ## WRITTEN BY AI ##
        """
        measure_start, _, requests = _run_synchronous(
            total_requests=10,
            warmup=TransientPhaseConfig(mode="requests", value=2),
            cooldown=TransientPhaseConfig(),
        )

        assert measure_start == requests[2][0]

    @pytest.mark.regression
    def test_cooldown_ends_at_last_measured_request(self):
        """
        Set the cooldown boundary to the end of request total - k.

        ## WRITTEN BY AI ##
        """
        _, measure_end, requests = _run_synchronous(
            total_requests=10,
            warmup=TransientPhaseConfig(),
            cooldown=TransientPhaseConfig(mode="requests", value=2),
        )

        assert measure_end == requests[7][1]

    @pytest.mark.sanity
    def test_warmup_ignores_non_terminal_updates(self):
        """
        Return no warmup transition for an in-progress update at the threshold.

        ## WRITTEN BY AI ##
        """
        config = TransientPhaseConfig(mode="requests", value=2)
        state = SchedulerState(
            start_time=0.0,
            processed_requests=3,
            progress=SchedulerProgress(total_requests=10, remaining_requests=7),
        )
        info = RequestInfo(
            status="in_progress", timings=RequestTimings(request_start=40.0)
        )

        active, transition_time = config.compute_transition_time(info, state, "start")

        assert active is True
        assert transition_time is None

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("total_requests", "warmup", "cooldown", "expected_measured"),
        [
            (10, 0, None, 10),
            (10, None, 0, 10),
            (10, 0, 0, 10),
            (1, 0, 0, 1),
            (3, 1, 1, 1),
            (3, 2, None, 1),
            (3, None, 2, 1),
        ],
    )
    def test_zero_phases_and_single_measured_request(
        self, total_requests, warmup, cooldown, expected_measured
    ):
        """
        Keep all requests for zero phases and retain a one-request window.

        ## WRITTEN BY AI ##
        """
        measure_start, measure_end, requests = _run_synchronous(
            total_requests=total_requests,
            warmup=TransientPhaseConfig(mode="requests", value=warmup),
            cooldown=TransientPhaseConfig(mode="requests", value=cooldown),
        )

        assert _count_measured(measure_start, measure_end, requests) == (
            expected_measured
        )
        if warmup is not None:
            assert measure_start == requests[warmup][0]
        if cooldown is not None:
            assert measure_end == requests[total_requests - cooldown - 1][1]

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("warmup", "cooldown", "expected_measured"),
        [(2, None, 8), (None, 2, 8), (2, 2, 6)],
    )
    def test_default_mode_falls_back_to_request_count(
        self, warmup, cooldown, expected_measured
    ):
        """
        Apply the same boundaries without a duration budget in the default mode.

        ## WRITTEN BY AI ##
        """
        measure_start, measure_end, requests = _run_synchronous(
            total_requests=10,
            warmup=TransientPhaseConfig(value=warmup),
            cooldown=TransientPhaseConfig(value=cooldown),
        )

        assert _count_measured(measure_start, measure_end, requests) == (
            expected_measured
        )
