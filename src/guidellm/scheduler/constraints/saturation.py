"""
Over-saturation detection constraint implementation.

This module implements the Over-Saturation Detection (OSD) algorithm for detecting
when a model becomes over-saturated during benchmarking. Over-saturation occurs when
the response rate doesn't keep up with the request rate, leading to degraded
performance.

Algorithm Overview:
-------------------
The OSD algorithm uses statistical slope detection to identify over-saturation:

1. **Slope Detection**: The algorithm tracks two key metrics over time:
   - Concurrent requests: Number of requests being processed simultaneously
   - Time-to-first-token (TTFT): Latency for the first token of each response

2. **Statistical Analysis**: For each metric, the algorithm:
   - Maintains a sliding window of recent data points
   - Calculates the linear regression slope using online statistics
   - Computes the margin of error (MOE) using t-distribution confidence intervals
   - Detects positive slopes with low MOE, indicating degradation

3. **Detection Criteria**: Over-saturation is detected when:
   - Both concurrent requests and TTFT show statistically significant positive slopes
   - The minimum duration threshold has been met
   - Sufficient data points are available for reliable slope estimation

4. **Window Management**: The algorithm maintains bounded memory by:
   - Limiting window size by time (maximum_window_seconds)
   - Limiting window size by ratio of total requests (maximum_window_ratio)
   - Automatically pruning old data points

5. **Constraint Integration**: When over-saturation is detected, the constraint:
   - Stops request queuing to prevent further degradation
   - Stops processing of existing requests (if enabled)
   - Provides detailed metadata about detection state

Key Parameters:
---------------
- minimum_duration: Minimum seconds before checking for over-saturation (default: 30.0)
- minimum_ttft: Minimum TTFT threshold for violation counting (default: 2.5)
- maximum_window_seconds: Maximum time window for data retention (default: 120.0)
- moe_threshold: Margin of error threshold for slope detection (default: 2.0)
- maximum_window_ratio: Maximum window size as ratio of total requests (default: 0.75)
- minimum_window_size: Minimum data points required for slope estimation (default: 5)
- confidence: Statistical confidence level for t-distribution (default: 0.95)

The constraint integrates with the scheduler by evaluating each request update and
providing scheduler actions (continue/stop) based on the current over-saturation state.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

from pydantic import Field

from guidellm.scheduler.schemas import (
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo

from .base import PydanticConstraintInitializer
from .factory import ConstraintsInitializerFactory
from .protocols import Constraint

__all__ = [
    "OverSaturationConstraint",
    "OverSaturationConstraintInitializer",
    "SlopeChecker",
    "approx_t_ppf",
]


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
    """
    Helper class for online slope detection using linear regression.

    Maintains running statistics for efficient O(1) updates and provides
    statistical slope detection with margin of error calculation.
    """

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
        """
        Check if there is a statistically significant positive slope.

        Args:
            effective_n: Effective sample size for slope estimation.

        Returns:
            True if positive slope detected with low margin of error.
        """
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


class OverSaturationConstraint:  # type: ignore[misc]
    """
    Constraint that detects and stops execution when over-saturation is detected.

    This constraint implements the Over-Saturation Detection (OSD) algorithm to
    identify when a model becomes over-saturated (response rate doesn't keep up with
    request rate). When over-saturation is detected, the constraint stops request
    queuing and optionally stops processing of existing requests.

    The constraint maintains internal state for tracking concurrent requests and
    time-to-first-token (TTFT) metrics, using statistical slope detection to identify
    performance degradation patterns.
    """

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
        enabled: bool = True,
    ) -> None:  # noqa: PLR0913
        """
        Initialize the over-saturation constraint.

        Args:
            minimum_duration: Minimum seconds before checking for over-saturation.
            minimum_ttft: Minimum TTFT threshold for violation counting.
            maximum_window_seconds: Maximum time window for data retention.
            moe_threshold: Margin of error threshold for slope detection.
            maximum_window_ratio: Maximum window size as ratio of total requests.
            minimum_window_size: Minimum data points required for slope estimation.
            confidence: Statistical confidence level for t-distribution.
            eps: Epsilon for numerical stability.
            enabled: Whether to actually stop when over-saturation is detected.
        """
        self.minimum_duration = minimum_duration
        self.minimum_ttft = minimum_ttft
        self.maximum_window_seconds = maximum_window_seconds
        self.maximum_window_ratio = maximum_window_ratio
        self.minimum_window_size = minimum_window_size
        self.moe_threshold = moe_threshold
        self.confidence = confidence
        self.eps = eps
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        """Reset all internal state to initial values."""
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

    def _add_finished(self, request: dict[str, Any]) -> None:
        """Add a finished request to tracking."""
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft is not None:
            self.total_finished_ever += 1
            self.finished_requests.append(request)
            if ttft > self.minimum_ttft:
                self.ttft_violations_counter += 1
            self.ttft_slope_checker.add_data_point(duration, ttft)

    def _remove_finished(self, request: dict[str, Any]) -> None:
        """Remove a finished request from tracking."""
        del self.finished_requests[0]
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft > self.minimum_ttft:
            self.ttft_violations_counter -= 1
        self.ttft_slope_checker.remove_data_point(duration, ttft)

    def _add_started(self, request: dict[str, Any]) -> None:
        """Add a started request to tracking."""
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        if concurrent is not None:
            self.total_started_ever += 1
            self.started_requests.append(request)
            self.concurrent_slope_checker.add_data_point(duration, concurrent)

    def _remove_started(self, request: dict[str, Any]) -> None:
        """Remove a started request from tracking."""
        del self.started_requests[0]
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        self.concurrent_slope_checker.remove_data_point(duration, concurrent)

    def _update_duration(self, duration: float) -> None:
        """Update duration and prune old data points."""
        self.duration = duration

        maximum_finished_window_size = int(
            self.total_finished_ever * self.maximum_window_ratio
        )
        while len(self.finished_requests) > maximum_finished_window_size:
            self._remove_finished(self.finished_requests[0])

        while (len(self.finished_requests) > 0) and (
            (
                time_since_earliest_request := duration
                - self.finished_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self._remove_finished(self.finished_requests[0])

        maximum_started_window_size = int(
            self.total_started_ever * self.maximum_window_ratio
        )
        while len(self.started_requests) > maximum_started_window_size:
            self._remove_started(self.started_requests[0])

        while (len(self.started_requests) > 0) and (
            (
                time_since_earliest_request := duration  # noqa: F841
                - self.started_requests[0]["duration"]
            )
            > self.maximum_window_seconds
        ):
            self._remove_started(self.started_requests[0])

    def _check_alert(self) -> bool:
        """
        Check if over-saturation is currently detected.

        Returns:
            True if over-saturation is detected, False otherwise.
        """
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
            self._add_started(
                {"concurrent_requests": concurrent_requests, "duration": duration}
            )
        elif (
            request_info.status == "completed"
            and request_info.timings
            and request_info.timings.first_token_iteration
            and request_info.timings.request_start
        ):
            ttft = (
                request_info.timings.first_token_iteration
                - request_info.timings.request_start
            )
            self._add_finished({"ttft": ttft, "duration": duration})

        self._update_duration(duration)
        is_over_saturated = self._check_alert()

        ttft_slope = self.ttft_slope_checker.slope
        ttft_slope_moe = self.ttft_slope_checker.margin_of_error
        ttft_n = self.ttft_slope_checker.n
        ttft_violations = self.ttft_violations_counter
        concurrent_slope = self.concurrent_slope_checker.slope
        concurrent_slope_moe = self.concurrent_slope_checker.margin_of_error
        concurrent_n = self.concurrent_slope_checker.n

        should_stop = is_over_saturated and self.enabled
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


@ConstraintsInitializerFactory.register(  # type: ignore[arg-type]
    ["over_saturation", "detect_saturation"]
)
class OverSaturationConstraintInitializer(PydanticConstraintInitializer):
    """
    Factory for creating OverSaturationConstraint instances from configuration.

    Supports both boolean and dictionary inputs:
    - bool: Enable/disable with default parameters
    - dict: Provide configuration parameters (min_seconds, max_window_seconds, etc.)
    """

    type_: Literal["over_saturation"] = "over_saturation"  # type: ignore[assignment]
    enabled: bool = Field(
        default=True,
        description="Whether to stop the benchmark if the model is over-saturated",
    )
    min_seconds: int | float = Field(
        default=30.0,
        ge=0,
        description="Minimum seconds before checking for over-saturation",
    )
    max_window_seconds: int | float = Field(
        default=120.0,
        ge=0,
        description="Maximum over-saturation checking window size in seconds",
    )
    moe_threshold: float = Field(
        default=2.0,
        ge=0,
        description="Margin of error threshold for slope detection",
    )
    minimum_ttft: float = Field(
        default=2.5,
        ge=0,
        description="Minimum TTFT threshold for violation counting",
    )
    maximum_window_ratio: float = Field(
        default=0.75,
        ge=0,
        le=1.0,
        description="Maximum window size as ratio of total requests",
    )
    minimum_window_size: int = Field(
        default=5,
        ge=0,
        description="Minimum data points required for slope estimation",
    )
    confidence: float = Field(
        default=0.95,
        ge=0,
        le=1.0,
        description="Statistical confidence level for t-distribution",
    )

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create an OverSaturationConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured OverSaturationConstraint instance.
        """
        return OverSaturationConstraint(  # type: ignore[return-value]
            minimum_duration=self.min_seconds,
            minimum_ttft=self.minimum_ttft,
            maximum_window_seconds=self.max_window_seconds,
            moe_threshold=self.moe_threshold,
            maximum_window_ratio=self.maximum_window_ratio,
            minimum_window_size=self.minimum_window_size,
            confidence=self.confidence,
            enabled=self.enabled,
        )

    @classmethod
    def validated_kwargs(
        cls, over_saturation: bool | dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for OverSaturationConstraint creation.

        Supports both bool and dict inputs:
        - bool: Enable/disable with defaults
        - dict: Provide configuration parameters

        :param over_saturation: Boolean to enable/disable, or dict with configuration
        :param kwargs: Additional keyword arguments (supports aliases)
        :return: Validated dictionary with constraint configuration
        """
        # Check for aliases in kwargs
        aliases = ["over_saturation", "detect_saturation"]
        result: bool | dict[str, Any] | None = over_saturation

        for alias in aliases:
            alias_value = kwargs.get(alias)
            if alias_value is not None:
                result = alias_value
                break

        if result is None:
            return {}

        if isinstance(result, bool):
            return {"enabled": result}
        elif isinstance(result, dict):
            # Extract configuration from dict
            return {
                "enabled": result.get("enabled", True),
                "min_seconds": result.get("min_seconds", 30.0),
                "max_window_seconds": result.get("max_window_seconds", 120.0),
                "moe_threshold": result.get("moe_threshold", 2.0),
                "minimum_ttft": result.get("minimum_ttft", 2.5),
                "maximum_window_ratio": result.get("maximum_window_ratio", 0.75),
                "minimum_window_size": result.get("minimum_window_size", 5),
                "confidence": result.get("confidence", 0.95),
            }
        else:
            # Convert to bool if it's truthy
            return {"enabled": bool(result)}
