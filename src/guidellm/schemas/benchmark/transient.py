from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from guidellm.schemas import RequestInfo, StandardBaseModel

if TYPE_CHECKING:
    from guidellm.scheduler import SchedulerState

__all__ = ["TransientPhaseConfig"]


class TransientPhaseConfig(StandardBaseModel):
    """Configure warmup and cooldown phases for benchmark execution.

    Supports flexible phase definition through percentage or absolute value
    specifications with multiple interpretation modes. Phases can be bounded
    by duration, request count, or both, enabling precise control over transient
    periods that should be excluded from final benchmark metrics.
    """

    @classmethod
    def create_from_value(
        cls, value: int | float | dict | TransientPhaseConfig | None
    ) -> TransientPhaseConfig:
        """
        Create configuration from flexible input formats.

        :param value: Configuration as int/float (percent if <1.0, absolute
            otherwise), dict (validated to model), TransientPhaseConfig instance,
            or None for defaults
        :return: Configured TransientPhaseConfig instance
        :raises ValueError: If value type is unsupported
        """
        if value is None:
            return TransientPhaseConfig()

        if isinstance(value, TransientPhaseConfig):
            return value

        if isinstance(value, dict):
            return TransientPhaseConfig.model_validate(value)

        if isinstance(value, int | float):
            kwargs: dict[str, Any] = {
                "percent": value if value < 1.0 else None,
                "value": value if value >= 1.0 else None,
            }
            return TransientPhaseConfig.model_validate(kwargs)

        raise ValueError(f"Unsupported type for TransientPhaseConfig: {type(value)}")

    percent: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Phase size as percentage (0.0-1.0) of total duration/requests; "
            "interpretation depends on mode. Takes precedence over value when target "
            "mode is available, otherwise falls back to value"
        ),
        examples=[0.0, 0.5],
        lt=1.0,
    )
    value: NonNegativeInt | NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Phase size as absolute duration (seconds) or request count; "
            "interpretation depends on mode. Used when percent is unset or "
            "target mode unavailable"
        ),
        examples=[1.0, 2.0],
    )
    mode: Literal[
        "duration", "requests", "prefer_duration", "prefer_requests", "both"
    ] = Field(
        default="prefer_duration",
        description=(
            "Interpretation mode: 'duration' for time-based phases, 'requests' for "
            "count-based phases, 'prefer_duration'/'prefer_requests' for fallback "
            "behavior, 'both' requires satisfying both conditions"
        ),
    )

    def compute_limits(
        self,
        max_requests: int | float | None,
        max_seconds: float | None,
        enforce_preference: bool = True,
    ) -> tuple[float | None, int | None]:
        """
        Calculate phase boundaries from benchmark constraints.

        :param max_requests: Total request budget for benchmark execution
        :param max_seconds: Total duration budget for benchmark execution
        :param enforce_preference: Whether to enforce preferred mode when both
            duration and request constraints are available
        :return: Tuple of (phase duration in seconds, phase request count)
        """
        duration: float | None = None
        requests: int | None = None

        if self.mode != "requests" and max_seconds is not None:
            if self.percent is not None:
                duration = self.percent * max_seconds
            elif self.value is not None:
                duration = float(self.value)

        if self.mode != "duration" and max_requests is not None:
            if self.percent is not None:
                requests = int(self.percent * max_requests)
            elif self.value is not None:
                requests = int(self.value)

        if enforce_preference:
            if self.mode == "prefer_duration" and duration is not None:
                requests = None
            elif self.mode == "prefer_requests" and requests is not None:
                duration = None

        return duration, requests

    def compute_transition_time(
        self,
        info: RequestInfo,
        state: SchedulerState,
        period: Literal["start", "end"],
    ) -> tuple[bool, float | None]:
        """
        Determine transition timestamp for entering or exiting phase.

        :param info: RequestInfo for current request to calculate against
        :param state: SchedulerState with current progress metrics and scheduler info
        :param period: Phase period, either "start" for warmup or "end" for cooldown
        :return: Tuple of (phase active flag, transition timestamp if applicable)
        """
        phase_duration, phase_requests = self.compute_limits(
            max_requests=state.progress.total_requests,
            max_seconds=state.progress.total_duration,
        )
        duration_transition_time: float | None = None
        request_transition_time: float | None = None

        # Calculate transition times for the phase based on phase limits and period
        # Potential phases: start (warmup) -> active -> end (cooldown)
        #   Warmup transition times: (start, start + duration)
        #   Active transition times: (start + duration, end - duration)
        #   Cooldown transition times: (end - duration, end)
        if period == "start":
            if phase_duration is not None:
                # Duration was set and caculating for "warmup" / start phase
                # Phase is active for [start, start + duration]
                duration_transition_time = state.start_time + phase_duration
            if phase_requests is not None:
                # Requests was set and calculating for "warmup" / start phase
                # Phase is active for requests [0, phase_requests]
                # Grab start time of the next request as transition time
                # (all requests up to and including phase_requests are in warmup)
                request_transition_time = (
                    info.started_at
                    if info.started_at is not None
                    and state.processed_requests == phase_requests + 1
                    else -1.0
                )
        elif period == "end":
            if phase_duration is not None:
                # Duration was set and calculating for "cooldown" / end phase
                # Phase is active for [end - duration, end]
                duration_transition_time = (
                    state.start_time + state.progress.total_duration - phase_duration
                    if state.progress.total_duration is not None
                    else -1.0
                )
            if phase_requests is not None:
                # Requests was set and calculating for "cooldown" / end phase
                # Phase is active for requests [total - phase_requests, total]
                # Grab completion time of the request right before cooldown starts
                # (all requests from that point onward are in cooldown)
                request_transition_time = (
                    info.completed_at
                    if info.completed_at is not None
                    and state.progress.remaining_requests is not None
                    and state.progress.remaining_requests == phase_requests + 1
                    else -1.0
                )

        transition_active: bool = False
        transition_time: float | None = None

        if request_transition_time == -1.0 or duration_transition_time == -1.0:
            # Transition defined but not yet reached or passed
            transition_active = True
            request_transition_time = None
        elif (
            request_transition_time is not None and duration_transition_time is not None
        ):
            # Both limits defined; need to satisfy both (min for end, max for start)
            transition_active = True
            transition_time = (
                min(request_transition_time, duration_transition_time)
                if period == "end"
                else max(request_transition_time, duration_transition_time)
            )
        elif (
            request_transition_time is not None or duration_transition_time is not None
        ):
            # One limit defined; satisfy that one
            transition_active = True
            transition_time = request_transition_time or duration_transition_time

        return transition_active, transition_time
