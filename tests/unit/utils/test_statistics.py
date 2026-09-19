"""
Unit tests for the shared statistical primitives.

Reference quantiles are taken from published Student's t tables so that the
implementation is checked against an external source rather than against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from guidellm.utils.statistics import (
    approx_t_ppf,
    mean_confidence_interval,
    normal_ppf,
    quantile_confidence_interval,
    t_ppf,
    wilson_interval,
)

# Two-sided quantiles of Student's t distribution, keyed by
# (cumulative probability, degrees of freedom).
T_TABLE: dict[tuple[float, int], float] = {
    (0.90, 1): 3.077684,
    (0.90, 10): 1.372184,
    (0.90, 30): 1.310415,
    (0.95, 1): 6.313752,
    (0.95, 10): 1.812461,
    (0.95, 30): 1.697261,
    (0.975, 1): 12.706205,
    (0.975, 2): 4.302653,
    (0.975, 3): 3.182446,
    (0.975, 5): 2.570582,
    (0.975, 10): 2.228139,
    (0.975, 20): 2.085963,
    (0.975, 30): 2.042272,
    (0.975, 60): 2.000298,
    (0.975, 120): 1.979930,
    (0.975, 1000): 1.962339,
    (0.99, 1): 31.820516,
    (0.99, 10): 2.763769,
    (0.99, 30): 2.457262,
    (0.995, 1): 63.656741,
    (0.995, 10): 3.169273,
    (0.995, 30): 2.749996,
}

# Quantiles of the standard normal distribution.
NORMAL_TABLE: dict[float, float] = {
    0.90: 1.281552,
    0.95: 1.644854,
    0.975: 1.959964,
    0.99: 2.326348,
    0.995: 2.575829,
}


class TestNormalPpf:
    """Tests for the standard normal quantile function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(("probability", "expected"), NORMAL_TABLE.items())
    def test_matches_published_quantiles(self, probability: float, expected: float):
        """
        Quantiles agree with published normal tables.

        ## WRITTEN BY AI ##
        """
        assert normal_ppf(probability) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.sanity
    def test_symmetry(self):
        """
        Quantiles either side of the median are equal and opposite.

        ## WRITTEN BY AI ##
        """
        for probability in (0.6, 0.75, 0.9, 0.99):
            assert normal_ppf(probability) == pytest.approx(
                -normal_ppf(1.0 - probability), abs=1e-12
            )

    @pytest.mark.sanity
    @pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range(self, probability: float):
        """
        Probabilities outside the open unit interval raise.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            normal_ppf(probability)


class TestTPpf:
    """Tests for the Student's t quantile function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(("key", "expected"), T_TABLE.items())
    def test_matches_published_quantiles(self, key: tuple[float, int], expected: float):
        """
        Quantiles agree with published t tables to their stated precision.

        ## WRITTEN BY AI ##
        """
        probability, degrees_of_freedom = key
        assert t_ppf(probability, degrees_of_freedom) == pytest.approx(
            expected, abs=1e-6
        )

    @pytest.mark.sanity
    def test_converges_to_the_normal_quantile(self):
        """
        Large degrees of freedom reproduce the normal quantile.

        ## WRITTEN BY AI ##
        """
        assert t_ppf(0.975, 1.0e9) == pytest.approx(normal_ppf(0.975), abs=1e-9)

    @pytest.mark.sanity
    def test_decreases_with_degrees_of_freedom(self):
        """
        A given quantile shrinks as degrees of freedom grow.

        ## WRITTEN BY AI ##
        """
        values = [t_ppf(0.975, df) for df in (1, 2, 5, 10, 50, 200, 5000)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))

    @pytest.mark.sanity
    def test_increases_with_probability(self):
        """
        The quantile grows with the requested cumulative probability.

        ## WRITTEN BY AI ##
        """
        values = [t_ppf(p, 7) for p in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)]
        assert all(a < b for a, b in zip(values, values[1:], strict=False))

    @pytest.mark.sanity
    def test_symmetry_and_median(self):
        """
        The distribution is symmetric about a median of zero.

        ## WRITTEN BY AI ##
        """
        assert t_ppf(0.5, 7) == 0.0
        assert t_ppf(0.3, 7) == pytest.approx(-t_ppf(0.7, 7), abs=1e-9)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("probability", "degrees_of_freedom"),
        [(0.0, 10), (1.0, 10), (0.975, 0), (0.975, -1)],
    )
    def test_rejects_invalid_arguments(
        self, probability: float, degrees_of_freedom: float
    ):
        """
        Out-of-domain probabilities and degrees of freedom raise.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="must be"):
            t_ppf(probability, degrees_of_freedom)

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("degrees_of_freedom", "tolerance"),
        [(1, 0.6), (2, 0.2), (3, 0.1), (10, 0.01), (30, 0.001)],
    )
    def test_is_more_accurate_than_the_approximation(
        self, degrees_of_freedom: int, tolerance: float
    ):
        """
        The approximation it replaces is materially low at small samples.

        Pins the gap that motivates using this function for reported intervals:
        ``approx_t_ppf`` is around 49% low at one degree of freedom and closes
        to a fraction of a percent by thirty.

        ## WRITTEN BY AI ##
        """
        exact = T_TABLE[(0.975, degrees_of_freedom)]
        approximate = approx_t_ppf(0.975, degrees_of_freedom)

        assert t_ppf(0.975, degrees_of_freedom) == pytest.approx(exact, abs=1e-6)
        assert approximate < exact
        assert abs(approximate - exact) / exact == pytest.approx(
            tolerance, abs=tolerance
        )


class TestApproxTPpf:
    """Tests pinning the behaviour of the retained approximation."""

    @pytest.mark.smoke
    def test_returns_nan_for_non_positive_degrees_of_freedom(self):
        """
        Non-positive degrees of freedom yield NaN rather than raising.

        ## WRITTEN BY AI ##
        """
        assert math.isnan(approx_t_ppf(0.975, 0))
        assert math.isnan(approx_t_ppf(0.975, -1))

    @pytest.mark.regression
    def test_values_are_unchanged_by_the_move(self):
        """
        The approximation returns exactly what it did before relocation.

        Over-saturation detection is tuned against these values, so a change
        here would alter when a benchmark stops.

        ## WRITTEN BY AI ##
        """
        assert approx_t_ppf(0.975, 1) == pytest.approx(6.479055276561087, abs=1e-12)
        assert approx_t_ppf(0.975, 5) == pytest.approx(2.5209206582151387, abs=1e-12)
        assert approx_t_ppf(0.975, 30) == pytest.approx(2.041898987326509, abs=1e-12)
        assert approx_t_ppf(0.95, 10) == pytest.approx(1.8097534523414425, abs=1e-12)
        assert approx_t_ppf(0.99, 3) == pytest.approx(3.9921373886852733, abs=1e-12)


class TestWilsonInterval:
    """Tests for the binomial proportion interval."""

    @pytest.mark.smoke
    def test_brackets_the_proportion(self):
        """
        The interval contains the observed proportion.

        ## WRITTEN BY AI ##
        """
        lower, upper = wilson_interval(95, 100, 0.95)
        assert lower < 0.95 < upper

    @pytest.mark.sanity
    def test_stays_within_the_unit_interval(self):
        """
        Bounds remain valid probabilities at the extremes.

        ## WRITTEN BY AI ##
        """
        assert wilson_interval(40, 40)[1] == pytest.approx(1.0, abs=1e-12)
        assert wilson_interval(40, 40)[1] <= 1.0
        assert wilson_interval(0, 40)[0] == pytest.approx(0.0, abs=1e-12)
        assert wilson_interval(0, 40)[0] >= 0.0

    @pytest.mark.sanity
    def test_narrows_with_more_trials(self):
        """
        More trials at the same proportion give a tighter interval.

        ## WRITTEN BY AI ##
        """
        small = wilson_interval(95, 100)
        large = wilson_interval(9500, 10000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    @pytest.mark.sanity
    def test_widens_with_confidence(self):
        """
        A higher confidence level gives a wider interval.

        ## WRITTEN BY AI ##
        """
        narrow = wilson_interval(950, 1000, 0.90)
        wide = wilson_interval(950, 1000, 0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("successes", "trials", "confidence", "expected"),
        [
            (95, 100, 0.95, (0.8882316293959427, 0.9784601218241689)),
            (40, 40, 0.95, (0.9123432418992939, 1.0)),
            (0, 40, 0.95, (0.0, 0.08765675810070626)),
            (950, 1000, 0.99, (0.9290892050518871, 0.9649768851275418)),
            (19, 20, 0.9, (0.8039485463512348, 0.9887682993566453)),
        ],
    )
    def test_bounds_are_unchanged_by_the_move(
        self,
        successes: int,
        trials: int,
        confidence: float,
        expected: tuple[float, float],
    ):
        """
        The interval returns exactly what it did before relocation.

        The goodput search compares these bounds against a target attainment, so
        a shift of even one ulp can turn a probe that met a 100% target into one
        that missed it. That is why the quantile here stays with the existing
        approximation rather than moving to ``normal_ppf``.

        ## WRITTEN BY AI ##
        """
        lower, upper = wilson_interval(successes, trials, confidence)

        assert lower == pytest.approx(expected[0], abs=0.0, rel=0.0)
        assert upper == pytest.approx(expected[1], abs=0.0, rel=0.0)

    @pytest.mark.sanity
    def test_handles_degenerate_and_out_of_range_input(self):
        """
        Zero trials and clamped counts do not raise.

        ## WRITTEN BY AI ##
        """
        assert wilson_interval(0, 0) == (0.0, 1.0)
        assert wilson_interval(-5, 10) == wilson_interval(0, 10)
        assert wilson_interval(50, 10) == wilson_interval(10, 10)


class TestMeanConfidenceInterval:
    """Tests for the t interval on a sample mean."""

    @pytest.mark.smoke
    def test_brackets_the_mean_symmetrically(self):
        """
        The interval is centred on the mean.

        ## WRITTEN BY AI ##
        """
        interval = mean_confidence_interval(100, 10.0, 2.0, 0.95)
        assert interval is not None
        assert (interval[0] + interval[1]) / 2.0 == pytest.approx(10.0, abs=1e-12)

    @pytest.mark.sanity
    @pytest.mark.parametrize("count", [0, 1])
    def test_requires_two_observations(self, count: int):
        """
        Fewer than two observations yield no interval.

        ## WRITTEN BY AI ##
        """
        assert mean_confidence_interval(count, 10.0, 2.0) is None

    @pytest.mark.sanity
    def test_narrows_with_sample_size(self):
        """
        The interval tightens as the sample grows.

        ## WRITTEN BY AI ##
        """
        small = mean_confidence_interval(10, 10.0, 2.0)
        large = mean_confidence_interval(1000, 10.0, 2.0)
        assert small is not None
        assert large is not None
        assert (large[1] - large[0]) < (small[1] - small[0])

    @pytest.mark.sanity
    def test_uses_the_unbiased_standard_deviation(self):
        """
        The population standard deviation is rescaled before use.

        ``DistributionSummary`` divides by n, so using it directly would give an
        interval that is too narrow at small samples.

        ## WRITTEN BY AI ##
        """
        count, std_dev = 5, 2.0
        interval = mean_confidence_interval(count, 0.0, std_dev, 0.95)
        assert interval is not None

        unbiased = std_dev * math.sqrt(count / (count - 1))
        expected = t_ppf(0.975, count - 1) * unbiased / math.sqrt(count)
        assert interval[1] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.regression
    def test_covers_the_true_mean_at_the_nominal_rate(self):
        """
        Repeated samples are covered at close to the requested rate.

        ## WRITTEN BY AI ##
        """
        rng = np.random.default_rng(17)
        covered = 0
        trials = 2000

        for _ in range(trials):
            sample = rng.normal(10.0, 2.0, size=25)
            interval = mean_confidence_interval(
                25, float(sample.mean()), float(sample.std()), 0.95
            )
            assert interval is not None
            covered += interval[0] <= 10.0 <= interval[1]

        assert 0.93 <= covered / trials <= 0.97


class TestQuantileConfidenceInterval:
    """Tests for the distribution-free interval on a sample quantile."""

    @pytest.mark.smoke
    def test_bounds_are_observations_that_bracket_the_quantile(self):
        """
        Both bounds come from the sample and straddle the estimate.

        ## WRITTEN BY AI ##
        """
        values = [float(index) for index in range(1000)]
        interval = quantile_confidence_interval(values, 0.5, 0.95)
        assert interval is not None
        assert interval[0] in values
        assert interval[1] in values
        assert interval[0] <= 500.0 <= interval[1]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("quantile", "minimum_count"),
        [(0.5, 6), (0.75, 13), (0.9, 36), (0.95, 72), (0.99, 368)],
    )
    def test_minimum_sample_size_for_a_two_sided_interval(
        self, quantile: float, minimum_count: int
    ):
        """
        An interval appears exactly at the sample size theory requires.

        Below ``ceil(log(alpha / 2) / log(quantile))`` observations every value
        in the sample can fall below the true quantile with probability above
        alpha / 2, so no upper bound exists.

        ## WRITTEN BY AI ##
        """
        values = [float(index) for index in range(minimum_count)]
        assert quantile_confidence_interval(values, quantile, 0.95) is not None
        assert quantile_confidence_interval(values[:-1], quantile, 0.95) is None

    @pytest.mark.sanity
    def test_narrows_with_sample_size(self):
        """
        A larger sample brackets the quantile more tightly.

        ## WRITTEN BY AI ##
        """
        rng = np.random.default_rng(5)
        small = quantile_confidence_interval(
            np.sort(rng.normal(0.0, 1.0, 200)).tolist(), 0.5, 0.95
        )
        large = quantile_confidence_interval(
            np.sort(rng.normal(0.0, 1.0, 20000)).tolist(), 0.5, 0.95
        )
        assert small is not None
        assert large is not None
        assert (large[1] - large[0]) < (small[1] - small[0])

    @pytest.mark.sanity
    @pytest.mark.parametrize("quantile", [0.0, 1.0, -0.2, 1.4])
    def test_rejects_out_of_range_quantiles(self, quantile: float):
        """
        Quantiles outside the open unit interval raise.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            quantile_confidence_interval([1.0, 2.0, 3.0], quantile)

    @pytest.mark.sanity
    def test_requires_two_observations(self):
        """
        A sample of fewer than two values yields no interval.

        ## WRITTEN BY AI ##
        """
        assert quantile_confidence_interval([1.0], 0.5) is None

    @pytest.mark.regression
    def test_covers_the_true_quantile_at_or_above_the_nominal_rate(self):
        """
        Repeated samples cover the population quantile at least as often as
        requested, which is the guarantee a discrete order-statistic interval
        gives.

        ## WRITTEN BY AI ##
        """
        rng = np.random.default_rng(23)
        truth = math.exp(1.0 + 0.5 * normal_ppf(0.9))
        covered = 0
        trials = 1500

        for _ in range(trials):
            sample = np.sort(rng.lognormal(1.0, 0.5, size=400))
            interval = quantile_confidence_interval(sample.tolist(), 0.9, 0.95)
            assert interval is not None
            covered += interval[0] <= truth <= interval[1]

        assert covered / trials >= 0.94
