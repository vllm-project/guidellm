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

from guidellm.scheduler.constraints.constraint import (
    Constraint,
    PydanticConstraintInitializer,
)
from guidellm.scheduler.constraints.factory import ConstraintsInitializerFactory
from guidellm.scheduler.schemas import (
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo

__all__ = [
    "OverSaturationConstraint",
    "OverSaturationConstraintInitializer",
    "SlopeChecker",
    "approx_t_ppf",
]


def approx_t_ppf(p: float, df: float) -> float:
    """
    Approximate the percent point function (PPF) for the t-distribution.

    Provides a fast approximation of the t-distribution PPF using numerical
    methods from Abramowitz & Stegun. This function is significantly faster
    than scipy.stats.t.ppf while providing sufficient accuracy for statistical
    slope detection in over-saturation detection. Used internally by SlopeChecker
    for calculating confidence intervals and margin of error.

    Reference:
        Milton Abramowitz and Irene A. Stegun (Eds.). (1965).
        Handbook of Mathematical Functions: with Formulas, Graphs,
        and Mathematical Tables. Dover Publications.

        An electronic version of this book is available at:
        https://personal.math.ubc.ca/~cbm/aands/.

    :param p: The probability value (e.g., 0.975 for a 95% confidence interval)
    :param df: The degrees of freedom for the t-distribution
    :return: Approximate t-distribution PPF value, or NaN if df <= 0
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
    statistical slope detection with margin of error calculation. Uses online
    algorithms to compute linear regression statistics incrementally without
    storing all data points, enabling memory-efficient slope detection for
    over-saturation detection. Supports adding and removing data points
    dynamically while maintaining accurate statistical measures.

    Example:
    ::
        checker = SlopeChecker(moe_threshold=2.0, confidence=0.95)
        checker.add_data_point(1.0, 2.0)
        checker.add_data_point(2.0, 3.0)
        checker.add_data_point(3.0, 4.0)
        is_positive = checker.check_slope(3.0)  # True for positive slope
    """

    def __init__(
        self, moe_threshold: float = 1.0, confidence: float = 0.95, eps: float = 1e-12
    ) -> None:
        """
        Initialize slope checker with statistical parameters.

        :param moe_threshold: Maximum margin of error threshold for slope detection
        :param confidence: Statistical confidence level for t-distribution (0-1)
        :param eps: Epsilon value for numerical stability in calculations
        """
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
        Integrate a new data point into the accumulated statistics.

        Updates running sums for linear regression calculation in O(1) time.
        The data point is incorporated into the statistical model without
        storing the individual value, enabling memory-efficient slope detection.

        :param x_new: The new x-coordinate (typically time or duration)
        :param y_new: The new y-coordinate (typically metric value like TTFT
            or concurrent requests)
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

        Updates running sums by subtracting the specified data point in O(1) time.
        Used for window management when pruning old data points to maintain
        bounded memory usage while preserving statistical accuracy.

        :param x_old: The x-coordinate to remove (typically time or duration)
        :param y_old: The y-coordinate to remove (typically metric value)
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

        Calculates linear regression slope and margin of error using online
        statistics. Returns True if the slope is positive and the margin of
        error is below the threshold, indicating statistically significant
        degradation. Updates internal slope and margin_of_error attributes
        for external inspection.

        :param effective_n: Effective sample size for slope estimation (may differ
            from actual n for correlation adjustment)
        :return: True if positive slope detected with margin of error below threshold
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


class OverSaturationConstraint(Constraint):
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

        Creates a new constraint instance with specified detection parameters.
        The constraint will track concurrent requests and TTFT metrics, using
        statistical slope detection to identify when the model becomes
        over-saturated. All parameters have sensible defaults suitable for
        most benchmarking scenarios.

        :param minimum_duration: Minimum seconds before checking for over-saturation
            (default: 30.0)
        :param minimum_ttft: Minimum TTFT threshold in seconds for violation counting
            (default: 2.5)
        :param maximum_window_seconds: Maximum time window in seconds for data retention
            (default: 120.0)
        :param moe_threshold: Margin of error threshold for slope detection
            (default: 2.0)
        :param maximum_window_ratio: Maximum window size as ratio of total requests
            (default: 0.75)
        :param minimum_window_size: Minimum data points required for slope estimation
            (default: 5)
        :param confidence: Statistical confidence level for t-distribution (0-1)
            (default: 0.95)
        :param eps: Epsilon for numerical stability in calculations
            (default: 1e-12)
        :param enabled: Whether to actually stop when over-saturation is detected
            (default: True)
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

    @property
    def info(self) -> dict[str, Any]:
        """
        Get current constraint configuration and state information.
        :return: Dictionary containing configuration parameters.
        """

        return {
            "type_": "over_saturation",
            "minimum_duration": self.minimum_duration,
            "minimum_ttft": self.minimum_ttft,
            "maximum_window_seconds": self.maximum_window_seconds,
            "maximum_window_ratio": self.maximum_window_ratio,
            "minimum_window_size": self.minimum_window_size,
            "moe_threshold": self.moe_threshold,
            "confidence": self.confidence,
            "enabled": self.enabled,
        }

    def reset(self) -> None:
        """
        Reset all internal state to initial values.

        Clears all tracked requests, resets counters, and reinitializes slope
        checkers. Useful for reusing constraint instances across multiple
        benchmark runs or resetting state after configuration changes.
        """
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
        """
        Add a finished request to tracking.

        :param request: Dictionary containing request data with 'ttft' and
            'duration' keys.
        """
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft is not None:
            self.total_finished_ever += 1
            self.finished_requests.append(request)
            if ttft > self.minimum_ttft:
                self.ttft_violations_counter += 1
            self.ttft_slope_checker.add_data_point(duration, ttft)

    def _remove_finished(self, request: dict[str, Any]) -> None:
        """
        Remove a finished request from tracking.

        :param request: Dictionary containing request data with 'ttft' and
            'duration' keys.
        """
        del self.finished_requests[0]
        ttft = request["ttft"]
        duration = request["duration"]
        if ttft > self.minimum_ttft:
            self.ttft_violations_counter -= 1
        self.ttft_slope_checker.remove_data_point(duration, ttft)

    def _add_started(self, request: dict[str, Any]) -> None:
        """
        Add a started request to tracking.

        :param request: Dictionary containing request data with
            'concurrent_requests' and 'duration' keys.
        """
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        if concurrent is not None:
            self.total_started_ever += 1
            self.started_requests.append(request)
            self.concurrent_slope_checker.add_data_point(duration, concurrent)

    def _remove_started(self, request: dict[str, Any]) -> None:
        """
        Remove a started request from tracking.

        :param request: Dictionary containing request data with
            'concurrent_requests' and 'duration' keys.
        """
        del self.started_requests[0]
        concurrent = request["concurrent_requests"]
        duration = request["duration"]
        self.concurrent_slope_checker.remove_data_point(duration, concurrent)

    def _update_duration(self, duration: float) -> None:
        """
        Update duration and prune old data points.

        Updates the current duration and removes data points that exceed the maximum
        window size (by ratio or time) to maintain bounded memory usage.

        :param duration: Current duration in seconds since benchmark start.
        """
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

        :return: True if over-saturation is detected, False otherwise.
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

    Provides a Pydantic-based initializer for over-saturation detection constraints
    with support for flexible configuration patterns. Supports detailed configuration
    dictionaries, enabling easy integration with CLI arguments, configuration files,
    and programmatic constraint creation.

    Example:
    ::
        # Configuration with defaults
        initializer = OverSaturationConstraintInitializer(enabled=True)
        constraint = initializer.create_constraint()

        # Detailed configuration
        initializer = OverSaturationConstraintInitializer(
            enabled=True,
            min_seconds=60.0,
            max_window_seconds=300.0,
            moe_threshold=1.5
        )
        constraint = initializer.create_constraint()

    :cvar type_: Always "over_saturation" to identify this constraint type
    :cvar enabled: Whether to stop the benchmark if over-saturation is detected
    :cvar min_seconds: Minimum seconds before checking for over-saturation
    :cvar max_window_seconds: Maximum time window for data retention
    :cvar moe_threshold: Margin of error threshold for slope detection
    :cvar minimum_ttft: Minimum TTFT threshold for violation counting
    :cvar maximum_window_ratio: Maximum window size as ratio of total requests
    :cvar minimum_window_size: Minimum data points required for slope estimation
    :cvar confidence: Statistical confidence level for t-distribution
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
        Create an OverSaturationConstraint instance from this initializer.

        Constructs a new OverSaturationConstraint with the configuration parameters
        specified in this initializer. The constraint will be ready for evaluation
        against scheduler state and requests.

        :param _kwargs: Additional keyword arguments (unused)
        :return: Configured OverSaturationConstraint instance ready for use
        """
        return OverSaturationConstraint(
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
        cls, over_saturation: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Validate and process arguments for OverSaturationConstraint creation.

        Processes flexible input formats to create validated constraint
        configuration. Supports dictionary inputs for detailed configuration, and
        alias parameters for compatibility. Handles parameter normalization and
        default value application.

        :param over_saturation: Dictionary with configuration parameters
            (min_seconds, max_window_seconds, etc.)
        :param kwargs: Additional keyword arguments supporting aliases like
            "detect_saturation" for compatibility, or unpacked dict values when
            dict is passed to factory
        :return: Validated dictionary with constraint configuration ready for
            initializer creation
        """
        # Check for aliases in kwargs
        aliases = ["over_saturation", "detect_saturation"]
        result: dict[str, Any] | None = over_saturation

        for alias in aliases:
            alias_value = kwargs.get(alias)
            if alias_value is not None:
                result = alias_value
                break

        # If over_saturation is None but kwargs contain constraint parameters,
        # treat kwargs as an unpacked dict (happens when dict is passed to factory)
        if result is None and kwargs:
            constraint_keys = {
                "enabled",
                "min_seconds",
                "max_window_seconds",
                "moe_threshold",
                "minimum_ttft",
                "maximum_window_ratio",
                "minimum_window_size",
                "confidence",
            }
            if any(key in kwargs for key in constraint_keys):
                # Reconstruct dict from kwargs
                result = {key: kwargs[key] for key in constraint_keys if key in kwargs}

        if result is None:
            return {"enabled": False}

        if isinstance(result, dict):
            # Return dict as-is, defaults come from fields above
            return result
        else:
            # Type signature only accepts dict or None, so this should never happen
            raise TypeError(
                f"over_saturation must be a dict or None, got {type(result).__name__}"
            )
