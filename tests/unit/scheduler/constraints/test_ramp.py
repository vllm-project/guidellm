"""Unit tests for the ramp saturation constraint."""

from __future__ import annotations

import math

import pytest

from guidellm.scheduler import RampSaturationConstraint, SchedulerState
from guidellm.scheduler.constraints import ramp as ramp_module
from guidellm.schemas import RequestInfo, RequestTimings

RUN_START = 1000.0


def _started(request_id: int, sent: float) -> RequestInfo:
    return RequestInfo(
        request_id=str(request_id),
        status="in_progress",
        timings=RequestTimings(resolve_start=sent),
    )


def _completed(request_id: int, sent: float, done: float) -> RequestInfo:
    return RequestInfo(
        request_id=str(request_id),
        status="completed",
        timings=RequestTimings(request_start=sent, request_end=done),
    )


def _ramp_sends(start_rate: float, interval: float, max_rate: float) -> list[float]:
    """Send times on a doubling ramp, up to the point it reaches `max_rate`."""
    until = interval * math.log2(max_rate / start_rate)
    scale = start_rate * interval / math.log(2)
    sends, n = [], 0
    while (offset := interval * math.log2(1 + n / scale)) < until:
        sends.append(RUN_START + offset)
        n += 1
    return sends


def _fifo_completions(sends: list[float], capacity: float, latency: float):
    """A server that takes `latency` per request and finishes at most
    `capacity` requests per second, in order."""
    completions, last = [], -math.inf
    for sent in sends:
        last = max(sent + latency, last + 1 / capacity)
        completions.append(last)
    return completions


def _replay(constraint, sends, completions):
    """Feed events in time order. Once queuing stops, no more requests are sent,
    so their events are skipped. Returns every action produced."""
    state = SchedulerState(start_time=RUN_START, start_requests_time=RUN_START)
    events = sorted(
        [(sent, 0, i) for i, sent in enumerate(sends)]
        + [(done, 1, i) for i, done in enumerate(completions)]
    )
    actions, stopped_at = [], None
    for _, kind, i in events:
        if stopped_at is not None and sends[i] >= stopped_at:
            continue
        info = (
            _started(i, sends[i])
            if kind == 0
            else _completed(i, sends[i], completions[i])
        )
        action = constraint(state, info)
        actions.append(action)
        if stopped_at is None and action.request_queuing == "stop":
            stopped_at = sends[i] if kind == 0 else completions[i]
    return actions


class TestRampSaturationConstraint:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_seconds": 0.0},
            {"window_seconds": 1.0, "shortfall": 0.0},
            {"window_seconds": 1.0, "shortfall": 1.0},
            {"window_seconds": 1.0, "drain_latencies": -1.0},
            {"window_seconds": 1.0, "min_sent": 0},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        """
        Reject out of range configuration.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError):
            RampSaturationConstraint(**kwargs)

    @pytest.mark.sanity
    def test_keeps_ramping_while_server_keeps_up(self):
        """
        No saturation is detected while every request completes one latency later.

        ## WRITTEN BY AI ##
        """
        sends = _ramp_sends(start_rate=2.0, interval=2.0, max_rate=128.0)
        completions = _fifo_completions(sends, capacity=10_000.0, latency=0.5)
        constraint = RampSaturationConstraint(window_seconds=2.0)

        actions = _replay(constraint, sends, completions)

        assert constraint.phase == "ramping"
        assert all(action.request_queuing == "continue" for action in actions)
        assert actions[-1].metadata["saturated_at"] is None

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("capacity", "latency", "interval"),
        [(20.0, 0.5, 2.0), (150.0, 2.0, 8.0)],
    )
    def test_detects_saturation_and_measures_peak(self, capacity, latency, interval):
        """
        Saturation is detected after the send rate passes capacity, within two
        doubling intervals, and the peak lands within 10% of capacity.

        ## WRITTEN BY AI ##
        """
        start_rate = 1 / latency
        sends = _ramp_sends(start_rate, interval, max_rate=8 * capacity)
        completions = _fifo_completions(sends, capacity, latency)
        constraint = RampSaturationConstraint(window_seconds=interval)

        actions = _replay(constraint, sends, completions)

        crossing = interval * math.log2(capacity / start_rate)
        saturated_at = actions[-1].metadata["saturated_at"]
        assert saturated_at is not None
        assert crossing <= saturated_at <= crossing + 2 * interval
        assert actions[-1].metadata["peak_rate"] == pytest.approx(capacity, rel=0.1)

    @pytest.mark.sanity
    def test_keeps_processing_while_draining(self):
        """
        Saturation stops queuing but lets requests already sent keep running.

        ## WRITTEN BY AI ##
        """
        sends = _ramp_sends(start_rate=2.0, interval=2.0, max_rate=160.0)
        completions = _fifo_completions(sends, capacity=20.0, latency=0.5)
        constraint = RampSaturationConstraint(window_seconds=2.0)

        actions = _replay(constraint, sends, completions)

        draining = [a for a in actions if a.metadata["phase"] == "draining"]
        assert draining
        assert all(a.request_queuing == "stop" for a in draining)
        assert all(a.request_processing == "continue" for a in draining)

    @pytest.mark.sanity
    def test_poll_ends_drain(self, monkeypatch):
        """
        A poll with no request still ends the drain once its time has passed, so a
        server that stops completing anything cannot hold the run open.

        ## WRITTEN BY AI ##
        """
        sends = _ramp_sends(start_rate=2.0, interval=2.0, max_rate=160.0)
        completions = _fifo_completions(sends, capacity=20.0, latency=0.5)
        constraint = RampSaturationConstraint(window_seconds=2.0)
        state = SchedulerState(start_time=RUN_START, start_requests_time=RUN_START)
        for i, sent in enumerate(sends):
            constraint(state, _started(i, sent))
            constraint(state, _completed(i, sent, completions[i]))
            if constraint.phase == "draining":
                break
        assert constraint.phase == "draining"
        assert constraint.drain_until is not None

        monkeypatch.setattr(ramp_module.time, "time", lambda: constraint.drain_until)
        action = constraint(state, None)

        assert action.metadata["phase"] == "done"
        assert action.request_processing == "stop_local"

    @pytest.mark.regression
    def test_counts_each_request_once(self):
        """
        Repeated updates for the same request do not inflate sends or the peak.

        ## WRITTEN BY AI ##
        """
        constraint = RampSaturationConstraint(window_seconds=1.0)
        state = SchedulerState(start_time=RUN_START, start_requests_time=RUN_START)
        for i in range(5):
            sent = RUN_START + 0.2 * i
            for _ in range(3):
                constraint(state, _started(i, sent))
                constraint(state, _completed(i, sent, sent + 1.0))

        assert constraint.peak_rate == pytest.approx(5.0)

    @pytest.mark.regression
    def test_ignores_requests_cancelled_by_another_constraint(self, monkeypatch):
        """
        When another constraint ends the ramp, the requests it cancels never
        complete, but that is not saturation.

        ## WRITTEN BY AI ##
        """
        latency = 0.5
        sends = _ramp_sends(start_rate=2.0, interval=2.0, max_rate=16.0)
        cutoff = sends[-1]
        constraint = RampSaturationConstraint(window_seconds=2.0)
        state = SchedulerState(start_time=RUN_START, start_requests_time=RUN_START)
        for i, sent in enumerate(sends):
            constraint(state, _started(i, sent))
            if sent + latency <= cutoff:
                constraint(state, _completed(i, sent, sent + latency))

        state.end_queuing_time = cutoff
        monkeypatch.setattr(ramp_module.time, "time", lambda: cutoff + latency)
        action = constraint(state, None)

        assert constraint.phase == "ramping"
        assert action.metadata["saturated_at"] is None
