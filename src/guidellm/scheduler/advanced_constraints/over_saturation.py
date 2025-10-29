import math
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import Field

from guidellm.scheduler.constraints import (
    Constraint,
    ConstraintsInitializerFactory,
    PydanticConstraintInitializer,
)
from guidellm.scheduler.schemas import (
    RequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.settings import settings


class OverSaturationDetectorBase(ABC):
    @abstractmethod
    def add_finished(self, request: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def add_started(self, request: dict[str, Any]) -> None:
        pass

    def update_duration(self, duration: float) -> None:
        self.duration = duration

    @abstractmethod
    def check_alert(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


def approx_t_ppf(p, df):
    """
    Approximates the percent point function (PPF) for the t-distribution.
    This provides a close but not exact value compared to scipy.stats.t.ppf,
    but is much faster.

    Reference:
        Milton Abramowitz and Irene A. Stegun (Eds.). (1965).
        Handbook of Mathematical Functions: with Formulas, Graphs,
        and Mathematical Tables. Dover Publications.

        An electronic version of this book is available at:
        https://personal.math.ubc.ca/~cbm/aands/.

    Args:
        p (float): The probability (e.g., 0.975 for a 95% CI).
        df (float): The degrees of freedom.
    """
    dof = df
    if dof <= 0:
        return float("nan")

    # 1. Approximate the PPF of the Normal distribution (z-score)
    # Uses Abramowitz & Stegun formula 26.2.23.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    numerical_stability_threshold = 0.5
    if p < numerical_stability_threshold:
        t = math.sqrt(-2.0 * math.log(p))
        z = -(
            t
            - ((c[2] * t + c[1]) * t + c[0])
            / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
        )
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        z = t - ((c[2] * t + c[1]) * t + c[0]) / (
            ((d[2] * t + d[1]) * t + d[0]) * t + 1.0
        )

    # 2. Convert the z-score to a t-score
    # Uses the Cornish-Fisher expansion (first few terms).
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z

    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z4 + 16.0 * z3 + 3.0 * z2) / 96.0

    # Adjust z using the degrees of freedom (dof)
    return z + g1 / dof + g2 / (dof * dof)


class SlopeChecker:
    def __init__(
        self, moe_threshold: float = 1.0, confidence: float = 0.95, eps: float = 1e-12
    ) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.moe_threshold = moe_threshold
        self.eps = eps
        self.confidence = confidence
        self.slope: float | None = None
        self.margin_of_error: float | None = None

    def add_data_point(self, x_new: float, y_new: float) -> None:
        """
        Integrates a new data point into the accumulated statistics.
        This operation is O(1).

        Args:
            x_new (float): The new x-coordinate.
            y_new (float): The new y-coordinate.
        """
        self.n += 1
        self.sum_x += x_new
        self.sum_y += y_new
        self.sum_xy += x_new * y_new
        self.sum_x2 += x_new**2
        self.sum_y2 += y_new**2

    def remove_data_point(self, x_old: float, y_old: float) -> None:
        """
        Remove a data point from the accumulated statistics.
        This operation is O(1).

        Args:
            x_old (float): The x-coordinate to remove.
            y_old (float): The y-coordinate to remove.
        """
        self.n -= 1
        self.sum_x -= x_old
        self.sum_y -= y_old
        self.sum_xy -= x_old * y_old
        self.sum_x2 -= x_old**2
        self.sum_y2 -= y_old**2

    def check_slope(self, effective_n: float) -> bool:
        minimal_n_for_slope_estimation = 3
        if effective_n < minimal_n_for_slope_estimation:
            return False

        # Calculate sums of squares and cross-products
        # These formulas are numerically stable for online calculation.
        centered_sum_xx = self.sum_x2 - (self.sum_x**2) / self.n
        centered_sum_xy = self.sum_xy - (self.sum_x * self.sum_y) / self.n
        centered_sum_yy = self.sum_y2 - (self.sum_y**2) / self.n

        # Safeguard against division by zero for SS_xx
        centered_sum_xx_safe = max(centered_sum_xx, self.eps)

        slope = centered_sum_xy / centered_sum_xx_safe

        # Calculate Residual Sum of Squares (RSS)
        # This is a direct calculation using the sums of squares.
        residual_sum_of_squares = centered_sum_yy - (
            centered_sum_xy**2 / centered_sum_xx_safe
        )

        # Ensure RSS is non-negative due to potential floating point inaccuracies
        residual_sum_of_squares = max(residual_sum_of_squares, 0.0)

        # Degrees of freedom for standard error (n - 2 for simple linear regression)
        dof = effective_n - 2

        residual_variance = residual_sum_of_squares / dof
        standard_error = (residual_variance / centered_sum_xx_safe) ** 0.5

        # t-critical value
        alpha = 1 - self.confidence
        t_crit = approx_t_ppf(1 - alpha / 2, df=dof)

        # Margin Of Error
        margin_of_error = t_crit * standard_error / max(slope, self.eps)

        self.slope = slope
        self.margin_of_error = margin_of_error
        return (slope > 0) and (margin_of_error < self.moe_threshold)


class OverSaturationDetector(OverSaturationDetectorBase):
    def __init__(
        self,
        minimum_duration: float = 30.0,
        minimum_ttft: float = 2.5,
        maximum_window_seconds: float = 120.0,
        moe_threshold: float = 2.0,
        maximum_window_ratio: float = 0.75,
        minimum_window_size: int = 5,
        confidence: float = 0.95,
        eps: float = 1e-12,
    ) -> None:
        self.minimum_duration = minimum_duration
        self.minimum_ttft = minimum_ttft
        self.maximum_window_seconds = maximum_window_seconds
        self.maximum_window_ratio = maximum_window_ratio
        self.minimum_window_size = minimum_window_size
        self.moe_threshold = moe_threshold
        self.confidence = confidence
        self.eps = eps
        self.reset()

    def add_finished(self, request: dict[str, Any]) -> None:
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft is not None:
            self.total_finished_ever += 1
            self.finished_requests.append(request)
            if ttft > self.minimum_ttft:
                self.ttft_violations_counter += 1
            self.ttft_slope_checker.add_data_point(duration, ttft)

    def remove_finished(self, request: dict[str, Any]) -> None:
        del self.finished_requests[0]
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft > self.minimum_ttft:
            self.ttft_violations_counter -= 1
        self.ttft_slope_checker.remove_data_point(duration, ttft)

    def add_started(self, request: dict[str, Any]) -> None:
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        if concurrent is not None:
            self.total_started_ever += 1
            self.started_requests.append(request)
            self.concurrent_slope_checker.add_data_point(duration, concurrent)

    def remove_started(self, request: dict[str, Any]) -> None:
        del self.started_requests[0]
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        self.concurrent_slope_checker.remove_data_point(duration, concurrent)

    def update_duration(self, duration: float) -> None:
        self.duration = duration

        maximum_finished_window_size = int(
            self.total_finished_ever * self.maximum_window_ratio
        )
        while len(self.finished_requests) > maximum_finished_window_size:
            self.remove_finished(self.finished_requests[0])

        while (len(self.finished_requests) > 0) and (
            (
                time_since_earliest_request := duration
                - self.finished_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self.remove_finished(self.finished_requests[0])

        maximum_started_window_size = int(
            self.total_started_ever * self.maximum_window_ratio
        )
        while len(self.started_requests) > maximum_started_window_size:
            self.remove_started(self.started_requests[0])

        while (len(self.started_requests) > 0) and (
            (
                time_since_earliest_request := duration  # noqa: F841
                - self.started_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self.remove_started(self.started_requests[0])

    def check_alert(self) -> bool:
        # Use duration as the maximum n value since requests from the
        # same second are highly correlated, this is simple and good enough
        # given that the MOE has a custom threshold anyway.
        concurrent_n = min(self.duration, self.concurrent_slope_checker.n)
        ttft_n = min(self.duration, self.ttft_slope_checker.n)

        if (
            (self.duration < self.minimum_duration)
            or (self.ttft_slope_checker.n > self.ttft_violations_counter * 2)
            or (self.duration < self.minimum_ttft)
            or (concurrent_n < self.minimum_window_size)
        ):
            return False

        is_concurrent_slope_positive = self.concurrent_slope_checker.check_slope(
            concurrent_n
        )

        if ttft_n < self.minimum_window_size:
            return is_concurrent_slope_positive

        is_ttft_slope_positive = self.ttft_slope_checker.check_slope(ttft_n)

        return is_concurrent_slope_positive and is_ttft_slope_positive

    def reset(self) -> None:
        self.duration = 0.0
        self.started_requests: list[dict[str, Any]] = []
        self.finished_requests: list[dict[str, Any]] = []
        self.ttft_violations_counter = 0
        self.total_finished_ever = 0
        self.total_started_ever = 0
        self.concurrent_slope_checker = SlopeChecker(
            moe_threshold=self.moe_threshold, confidence=self.confidence, eps=self.eps
        )
        self.ttft_slope_checker = SlopeChecker(
            moe_threshold=self.moe_threshold, confidence=self.confidence, eps=self.eps
        )


class OverSaturationConstraint:  # type: ignore[misc]
    """
    Constraint that limits execution based on over-saturation detection.

    Stops request queuing when over-saturation is detected (i.e response-rate
    doesn't keep up with the request-rate).
    """

    def __init__(
        self,
        over_saturation_detector: OverSaturationDetector,
        stop_over_saturated: bool,
    ) -> None:
        self.over_saturation_detector = over_saturation_detector
        self.stop_over_saturated = stop_over_saturated

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state.

        :param state: Current scheduler state.
        :param request_info: Individual request information.
        :return: Action indicating whether to continue or stop operations.
        """
        duration = time.time() - state.start_time

        if request_info.status == "in_progress":
            concurrent_requests = state.processing_requests
            self.over_saturation_detector.add_started(
                {"concurrent_requests": concurrent_requests, "duration": duration}
            )
        elif (
            request_info.status == "completed"
            and request_info.timings
            and request_info.timings.first_iteration
        ):
            ttft = (
                request_info.timings.first_iteration
                - request_info.timings.request_start
            )
            self.over_saturation_detector.add_finished(
                {"ttft": ttft, "duration": duration}
            )

        self.over_saturation_detector.update_duration(duration)
        is_over_saturated = self.over_saturation_detector.check_alert()

        ttft_slope = self.over_saturation_detector.ttft_slope_checker.slope
        ttft_slope_moe = (
            self.over_saturation_detector.ttft_slope_checker.margin_of_error
        )
        ttft_n = self.over_saturation_detector.ttft_slope_checker.n
        ttft_violations = self.over_saturation_detector.ttft_violations_counter
        concurrent_slope = self.over_saturation_detector.concurrent_slope_checker.slope
        concurrent_slope_moe = (
            self.over_saturation_detector.concurrent_slope_checker.margin_of_error
        )
        concurrent_n = self.over_saturation_detector.concurrent_slope_checker.n

        should_stop = is_over_saturated and self.stop_over_saturated
        return SchedulerUpdateAction(
            request_queuing="stop" if should_stop else "continue",
            request_processing="stop_all" if should_stop else "continue",
            metadata={
                "ttft_slope": ttft_slope,
                "ttft_slope_moe": ttft_slope_moe,
                "ttft_n": ttft_n,
                "ttft_violations": ttft_violations,
                "concurrent_slope": concurrent_slope,
                "concurrent_slope_moe": concurrent_slope_moe,
                "concurrent_n": concurrent_n,
                "is_over_saturated": is_over_saturated,
            },
        )


@ConstraintsInitializerFactory.register(
    ["stop_over_saturated", "stop_over_sat", "stop_osd"]
)
class OverSaturationConstraintInitializer(PydanticConstraintInitializer):
    """Factory for creating OverSaturationConstraint instances from configuration."""

    type_: Literal["stop_over_saturated"] = "stop_over_saturated"  # type: ignore[assignment]
    stop_over_saturated: bool = Field(
        description="Whether to stop the benchmark if the model is over-saturated",
    )
    min_seconds: int | float = Field(
        default_factory=lambda: settings.constraint_over_saturation_min_seconds,
        ge=0,
        description="Minimum seconds before checking for over-saturation",
    )
    max_window_seconds: int | float = Field(
        default_factory=lambda: settings.constraint_over_saturation_max_window_seconds,
        ge=0,
        description="Maximum over-saturation checking window size in seconds",
    )

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a OverSaturationConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured OverSaturationConstraint instance.
        """
        over_saturation_detector = OverSaturationDetector(
            minimum_duration=self.min_seconds,
            maximum_window_seconds=self.max_window_seconds,
        )
        return OverSaturationConstraint(
            over_saturation_detector=over_saturation_detector,
            stop_over_saturated=self.stop_over_saturated,
        )

    @classmethod
    def validated_kwargs(cls, stop_over_saturated: bool, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for OverSaturationConstraint creation.

        :param stop_over_saturated: Whether to stop the benchmark if over-saturated
        :param kwargs: Supports stop_over_saturated, stop_over_sat, stop_osd
        :return: Validated dictionary with stop_over_saturated field
        """
        aliases = ["stop_over_saturated", "stop_over_sat", "stop_osd"]
        for alias in aliases:
            stop_over_saturated = stop_over_saturated or kwargs.get(alias)

        return {"stop_over_saturated": stop_over_saturated}
