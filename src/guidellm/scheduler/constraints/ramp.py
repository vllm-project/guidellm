"""
Saturation detection and peak measurement for request rate ramps.

A server keeping up with a ramp completes requests at the rate they were sent,
one request latency later. Once it saturates, completions fall short of what
was sent and requests start piling up. This constraint detects that shortfall,
stops sending, and measures the completion rate while the server works through
the requests it already has, which is when it runs at full capacity.
"""

from __future__ import annotations

import bisect
import statistics
import time
from collections import deque
from typing import Any, Literal

from guidellm.scheduler.constraints.constraint import Constraint
from guidellm.scheduler.schemas import SchedulerState, SchedulerUpdateAction
from guidellm.schemas import RequestInfo

__all__ = ["RampSaturationConstraint"]


class RampSaturationConstraint(Constraint):
    """
    Detect when a rate ramp saturates the server and measure its peak.

    While ramping, compares the requests completed over the last
    ``window_seconds`` with the requests sent over the same length of time one
    measured latency earlier. When completions fall more than ``shortfall``
    below that, the server is saturated: queuing stops, and requests already
    sent keep running for ``drain_latencies`` request latencies before local
    processing stops too. The peak is the highest completion rate over any
    ``window_seconds`` window up to that point.

    Example:
    ::
        constraint = RampSaturationConstraint(window_seconds=8.0)
        action = constraint(state, request_info)
        peak = action.metadata["peak_rate"]
    """

    def __init__(
        self,
        window_seconds: float,
        shortfall: float = 0.25,
        drain_latencies: float = 3.0,
        min_sent: int = 5,
    ):
        """
        :param window_seconds: Window for comparing sends and completions, and for
            measuring the peak completion rate
        :param shortfall: Fraction by which completions must fall below sends to
            count as saturation
        :param drain_latencies: Request latencies to keep measuring after
            saturation, while already sent requests complete
        :param min_sent: Minimum requests sent in the compared window before
            saturation can be detected
        :raises ValueError: If any argument is out of range
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if not 0 < shortfall < 1:
            raise ValueError(f"shortfall must be between 0 and 1, got {shortfall}")
        if drain_latencies < 0:
            raise ValueError(
                f"drain_latencies must not be negative, got {drain_latencies}"
            )
        if min_sent < 1:
            raise ValueError(f"min_sent must be at least 1, got {min_sent}")

        self.window_seconds = window_seconds
        self.shortfall = shortfall
        self.drain_latencies = drain_latencies
        self.min_sent = min_sent

        self.phase: Literal["ramping", "draining", "done"] = "ramping"
        self.peak_rate: float | None = None
        self.saturated_at: float | None = None
        self.drain_until: float | None = None
        self._sends: list[float] = []
        self._completions: deque[tuple[float, float]] = deque()
        self._started_ids: set[str] = set()
        self._completed_ids: set[str] = set()

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: Constraint configuration for reporting
        """
        return {
            "type_": "ramp_saturation",
            "window_seconds": self.window_seconds,
            "shortfall": self.shortfall,
            "drain_latencies": self.drain_latencies,
            "min_sent": self.min_sent,
        }

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo | None
    ) -> SchedulerUpdateAction:
        """
        Record the request, then advance through ramping, draining and done.

        :param state: Current scheduler state, used for the run's start time
        :param request_info: Individual request information, or ``None`` on poll
        :return: Action that stops queuing once saturated and stops processing
            once the drain period ends, carrying the peak rate so far
        """
        run_start = state.start_requests_time or state.start_time
        now = time.time()
        if request_info is not None and self.phase != "done":
            self._record(request_info)
            if request_info.status == "completed" and request_info.completed_at:
                now = request_info.completed_at

        # Requests another constraint cancels never complete; that isn't saturation.
        if (
            self.phase == "ramping"
            and state.end_queuing_time is None
            and self._is_saturated(now, run_start)
        ):
            self.phase = "draining"
            self.saturated_at = now
            latency = statistics.median(self._window_latencies(now))
            self.drain_until = now + self.drain_latencies * latency
        if (
            self.phase == "draining"
            and self.drain_until is not None
            and now >= self.drain_until
        ):
            self.phase = "done"

        return SchedulerUpdateAction(
            request_queuing="continue" if self.phase == "ramping" else "stop",
            request_processing="stop_local" if self.phase == "done" else "continue",
            metadata={
                "phase": self.phase,
                "peak_rate": self.peak_rate,
                "saturated_at": (
                    self.saturated_at - run_start if self.saturated_at else None
                ),
                "window_seconds": self.window_seconds,
            },
        )

    def _record(self, request_info: RequestInfo) -> None:
        started_at = request_info.started_at
        if (
            request_info.status == "in_progress"
            and started_at is not None
            and request_info.request_id not in self._started_ids
        ):
            self._started_ids.add(request_info.request_id)
            bisect.insort(self._sends, started_at)
            return

        completed_at = request_info.completed_at
        if (
            request_info.status != "completed"
            or completed_at is None
            or started_at is None
            or request_info.request_id in self._completed_ids
        ):
            return

        self._completed_ids.add(request_info.request_id)
        self._completions.append((completed_at, completed_at - started_at))
        window_start = completed_at - self.window_seconds
        while self._completions and self._completions[0][0] <= window_start:
            self._completions.popleft()

        rate = len(self._completions) / self.window_seconds
        if self.peak_rate is None or rate > self.peak_rate:
            self.peak_rate = rate

    def _window_latencies(self, now: float) -> list[float]:
        window_start = now - self.window_seconds
        return [
            latency
            for completed_at, latency in self._completions
            if window_start < completed_at <= now
        ]

    def _is_saturated(self, now: float, run_start: float) -> bool:
        latencies = self._window_latencies(now)
        if not latencies or now - self.window_seconds < run_start:
            return False

        # Requests sent one latency earlier should have completed in this window.
        lagged_end = now - statistics.median(latencies)
        lagged_start = lagged_end - self.window_seconds
        if lagged_start < run_start:
            return False

        sent = bisect.bisect_right(self._sends, lagged_end) - bisect.bisect_right(
            self._sends, lagged_start
        )

        return sent >= self.min_sent and len(latencies) < (1 - self.shortfall) * sent
