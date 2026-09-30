"""
Statistical primitives for quantifying measurement uncertainty.

Provides distribution quantile functions and interval estimators shared by the
benchmark reporting path and the scheduler constraints. Every estimator here
operates on plain numbers and returns plain tuples so that it can be used from
any layer without pulling in benchmark or scheduler types.

The module provides:
- Quantile functions for the normal and Student's t distributions
  (``normal_ppf``, ``t_ppf``, ``approx_t_ppf``)
- Interval estimators for a proportion, a sample mean, and a sample quantile
  (``wilson_interval``, ``mean_confidence_interval``,
  ``quantile_confidence_interval``)

Example:
::
    from guidellm.utils.statistics import mean_confidence_interval

    interval = mean_confidence_interval(
        count=250, mean=84.3, std_dev=19.8, confidence=0.95
    )
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

__all__ = [
    "approx_t_ppf",
    "mean_confidence_interval",
    "normal_ppf",
    "quantile_confidence_interval",
    "t_ppf",
    "wilson_interval",
]


_MIN_CONFIDENCE = 0.5
"""Lowest two-sided confidence level accepted by the interval estimators."""

_MAX_CONFIDENCE = 0.999
"""Highest two-sided confidence level accepted by the interval estimators."""

_NORMAL_LIMIT_DF = 1.0e7
"""Degrees of freedom above which the t quantile is taken as the normal one.

At this many degrees of freedom the two differ by 2.4e-7, an order of magnitude
below the tolerance ``t_ppf`` claims, so the inversion below is not worth
running. At 1e6 the gap is 2.4e-6, which would exceed it.
"""

_WILSON_NORMAL_DF = 1.0e9
"""Degrees of freedom at which ``approx_t_ppf`` stands in for the normal quantile.

``wilson_interval`` derives its quantile this way rather than from
``normal_ppf`` so that the goodput search keeps the bounds it has been
reporting. The two differ by 2.2e-4 relative, which is enough to move an upper
bound of exactly 1.0 to just below it and so to turn a probe that met a 100%
attainment target into one that missed it.
"""

_BETA_CONTINUED_FRACTION_ITERATIONS = 300
"""Iteration cap for the incomplete beta continued fraction."""

_BISECTION_ITERATIONS = 200
"""Iteration cap when inverting the t distribution CDF."""

_TINY = 1e-300
"""Floor used to keep the continued fraction away from division by zero."""

_CONTINUED_FRACTION_TOLERANCE = 1e-15
"""Relative change below which the continued fraction is taken as converged."""


def normal_ppf(probability: float) -> float:
    """
    Compute the quantile of the standard normal distribution.

    :param probability: Cumulative probability, strictly between 0 and 1
    :return: Value whose standard normal CDF equals ``probability``
    :raises ValueError: If ``probability`` is not strictly between 0 and 1
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"probability must be strictly between 0 and 1, got {probability}."
        )

    return NormalDist().inv_cdf(probability)


def approx_t_ppf(p: float, df: float) -> float:
    """
    Approximate the percent point function (PPF) for the t-distribution.

    Provides a fast approximation of the t-distribution PPF using numerical
    methods from Abramowitz & Stegun. This function is significantly faster
    than scipy.stats.t.ppf while providing sufficient accuracy for statistical
    slope detection in over-saturation detection. Used internally by SlopeChecker
    for calculating confidence intervals and margin of error.

    The approximation is within 0.5% of the exact quantile for ten or more
    degrees of freedom and degrades below that, reaching 49% low at one degree
    of freedom. Use ``t_ppf`` where small-sample accuracy matters.

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


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """
    Evaluate the continued fraction of the incomplete beta function.

    Uses the modified Lentz algorithm. Converges quickly for
    ``x < (a + 1) / (a + b + 2)``; callers use the symmetry of the incomplete
    beta function to stay in that range.

    :param a: First shape parameter, positive
    :param b: Second shape parameter, positive
    :param x: Evaluation point within the convergent range
    :return: Value of the continued fraction
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    result = d

    for iteration in range(1, _BETA_CONTINUED_FRACTION_ITERATIONS + 1):
        even = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        result *= d * c

        numerator = (
            -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        )
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        result *= step

        if abs(step - 1.0) < _CONTINUED_FRACTION_TOLERANCE:
            break

    return result


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """
    Compute the regularized incomplete beta function ``I_x(a, b)``.

    :param a: First shape parameter, positive
    :param b: Second shape parameter, positive
    :param x: Upper limit of integration, within [0, 1]
    :return: Value of the regularized incomplete beta function
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_prefactor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    prefactor = math.exp(log_prefactor)

    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _beta_continued_fraction(a, b, x) / a

    return 1.0 - prefactor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _t_cdf(value: float, degrees_of_freedom: float) -> float:
    """
    Compute the CDF of Student's t distribution.

    :param value: Point at which to evaluate the CDF
    :param degrees_of_freedom: Degrees of freedom, positive
    :return: Probability that a t-distributed variable is at most ``value``
    """
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)

    return 1.0 - tail if value > 0.0 else tail


def t_ppf(probability: float, degrees_of_freedom: float) -> float:
    """
    Compute the quantile of Student's t distribution.

    Inverts the CDF by bisection. Matches published quantile tables to within
    1e-6, their own stated precision, over one to one thousand degrees of
    freedom; round-tripping a quantile back through the CDF agrees to within
    1e-12. Prefer this over
    ``approx_t_ppf`` wherever the result is reported rather than compared
    against a tuned threshold, since the approximation is materially low below
    ten degrees of freedom.

    :param probability: Cumulative probability, strictly between 0 and 1
    :param degrees_of_freedom: Degrees of freedom, positive
    :return: Value whose t CDF equals ``probability``
    :raises ValueError: If ``probability`` is outside (0, 1) or the degrees of
        freedom are not positive
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"probability must be strictly between 0 and 1, got {probability}."
        )
    if degrees_of_freedom <= 0.0:
        raise ValueError(
            f"degrees_of_freedom must be positive, got {degrees_of_freedom}."
        )

    if degrees_of_freedom >= _NORMAL_LIMIT_DF:
        return normal_ppf(probability)

    if probability == 0.5:  # noqa: PLR2004
        return 0.0

    # Bracket the root. The t quantile is bounded by the normal quantile scaled
    # by the ratio of their spreads, which diverges as the degrees of freedom
    # approach one, so widen until the CDF straddles the target.
    bound = max(abs(normal_ppf(probability)), 1.0)
    while _t_cdf(bound, degrees_of_freedom) < probability:
        bound *= 2.0
    while _t_cdf(-bound, degrees_of_freedom) > probability:
        bound *= 2.0

    low, high = -bound, bound
    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (low + high)
        if _t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, abs(middle)):
            break

    return 0.5 * (low + high)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute a Wilson score interval for a binomial proportion.

    Preferred over the normal approximation when the proportion is measured
    near 0 or 1, where the normal interval extends outside [0, 1] and
    understates uncertainty for small trial counts.

    :param successes: Number of successes, clamped to [0, trials]
    :param trials: Number of trials
    :param confidence: Two-sided confidence level, clamped to [0.5, 0.999]
    :return: Tuple of (lower bound, upper bound), both within [0.0, 1.0]
    """
    if trials <= 0:
        return 0.0, 1.0

    # This is public API, so keep a caller that passes an out-of-range count or
    # confidence from reaching a domain error inside the formula below.
    successes = min(max(successes, 0), trials)
    confidence = min(max(confidence, _MIN_CONFIDENCE), _MAX_CONFIDENCE)

    z = approx_t_ppf((1.0 + confidence) / 2.0, _WILSON_NORMAL_DF)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
        )
        / denominator
    )

    return max(0.0, center - spread), min(1.0, center + spread)


def mean_confidence_interval(
    count: int, mean: float, std_dev: float, confidence: float = 0.95
) -> tuple[float, float] | None:
    """
    Compute a t confidence interval for the mean of a sample.

    Treats the observations as independent draws, so the interval describes how
    precisely this sample located its own mean. It does not describe how far the
    mean would move across repeated benchmark runs when successive observations
    share an underlying condition.

    :param count: Number of observations in the sample
    :param mean: Sample mean
    :param std_dev: Sample standard deviation, computed over ``count`` values
    :param confidence: Two-sided confidence level, clamped to [0.5, 0.999]
    :return: Tuple of (lower bound, upper bound), or None if fewer than two
        observations are available
    """
    if count < 2:  # noqa: PLR2004
        return None

    confidence = min(max(confidence, _MIN_CONFIDENCE), _MAX_CONFIDENCE)

    # The sample standard deviation carried by a DistributionSummary is the
    # population form, dividing by n. Rescale to the unbiased form so that the
    # interval matches the t distribution it is being read against.
    unbiased_std_dev = std_dev * math.sqrt(count / (count - 1))
    half_width = (
        t_ppf((1.0 + confidence) / 2.0, count - 1)
        * unbiased_std_dev
        / (math.sqrt(count))
    )

    return mean - half_width, mean + half_width


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """
    Compute ``P(X <= successes)`` for ``X`` binomial with the given parameters.

    :param successes: Success count threshold
    :param trials: Number of trials
    :param probability: Per-trial success probability
    :return: Cumulative probability at ``successes``
    """
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0

    return _regularized_incomplete_beta(
        trials - successes, successes + 1.0, 1.0 - probability
    )


def _largest_rank_below(trials: int, probability: float, target: float) -> int | None:
    """
    Find the largest ``k`` in [0, trials - 1] whose binomial CDF is at or below
    ``target``.

    :param trials: Number of trials
    :param probability: Per-trial success probability
    :param target: Cumulative probability ceiling
    :return: Largest qualifying ``k``, or None if even ``k = 0`` exceeds target
    """
    if _binomial_cdf(0, trials, probability) > target:
        return None

    low, high = 0, trials - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _binomial_cdf(middle, trials, probability) <= target:
            low = middle
        else:
            high = middle - 1

    return low


def _smallest_rank_above(trials: int, probability: float, target: float) -> int | None:
    """
    Find the smallest ``k`` in [0, trials - 1] whose binomial CDF is at or above
    ``target``.

    :param trials: Number of trials
    :param probability: Per-trial success probability
    :param target: Cumulative probability floor
    :return: Smallest qualifying ``k``, or None if no ``k`` below ``trials``
        reaches the target
    """
    if _binomial_cdf(trials - 1, trials, probability) < target:
        return None

    low, high = 0, trials - 1
    while low < high:
        middle = (low + high) // 2
        if _binomial_cdf(middle, trials, probability) >= target:
            high = middle
        else:
            low = middle + 1

    return low


def quantile_confidence_interval(
    sorted_values: Sequence[float],
    quantile: float,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """
    Compute a distribution-free confidence interval for a sample quantile.

    The bounds are themselves observations. The number of observations at or
    below the true quantile is binomial, so the ranks that bracket it at the
    requested confidence follow from that distribution and no assumption about
    the shape of the underlying distribution is needed.

    A two-sided interval only exists once the sample reaches
    ``log(alpha / 2) / log(quantile)`` observations, since below that every
    observation in the sample can fall below the true quantile with probability
    above ``alpha / 2``. Under that size there is no upper bound to report and
    this returns None rather than a bound the sample cannot support: at the
    default confidence, 72 observations are needed for p95, 368 for p99 and
    3688 for p999.

    :param sorted_values: Sample observations in ascending order
    :param quantile: Quantile to bound, strictly between 0 and 1
    :param confidence: Two-sided confidence level, clamped to [0.5, 0.999]
    :return: Tuple of (lower bound, upper bound) drawn from ``sorted_values``,
        or None when the sample cannot support a two-sided interval
    :raises ValueError: If ``quantile`` is not strictly between 0 and 1
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}.")

    trials = len(sorted_values)
    if trials < 2:  # noqa: PLR2004
        return None

    confidence = min(max(confidence, _MIN_CONFIDENCE), _MAX_CONFIDENCE)
    tail = (1.0 - confidence) / 2.0

    lower_rank = _largest_rank_below(trials, quantile, tail)
    upper_rank = _smallest_rank_above(trials, quantile, 1.0 - tail)

    if lower_rank is None or upper_rank is None:
        return None

    return sorted_values[lower_rank], sorted_values[upper_rank]
