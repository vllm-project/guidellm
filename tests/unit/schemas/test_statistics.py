from __future__ import annotations

import math
from typing import Literal, TypeVar

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from guidellm.schemas import DistributionSummary, Percentiles, StatusDistributionSummary
from guidellm.schemas.statistics import FunctionObjT


def test_function_obj_type():
    """Test that FunctionObjT is filled out correctly as a TypeVar."""
    assert isinstance(FunctionObjT, type(TypeVar("test")))
    assert FunctionObjT.__name__ == "FunctionObjT"
    assert FunctionObjT.__bound__ is None
    assert FunctionObjT.__constraints__ == ()


def generate_pdf(
    distribution: str | None, distribution_args: dict, size: int
) -> np.ndarray:
    if distribution is None:
        return np.empty((0, 2))

    if distribution == "normal":
        mean = distribution_args.get("loc", 0.0)
        std_dev = distribution_args.get("scale", 1.0)
        x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, size)
        pdf_values = (1.0 / np.sqrt(2 * np.pi * std_dev**2)) * np.exp(
            -1.0 * ((x_values - mean) ** 2) / (2 * std_dev**2)
        )
    elif distribution == "uniform":
        low = distribution_args.get("low", 0.0)
        high = distribution_args.get("high", 1.0)
        x_values = np.linspace(low, high, size)
        pdf_values = np.full_like(x_values, 1.0 / (high - low))
    elif distribution == "exponential":
        scale = distribution_args.get("scale", 1.0)
        x_values = np.linspace(0, 10 * scale, size)
        pdf_values = (1 / scale) * np.exp(-x_values / scale)
    elif distribution == "poisson":
        lam = distribution_args.get("lam", 1.0)
        x_values = np.arange(0, 20)
        pdf_values = (lam**x_values * np.exp(-lam)) / np.array(
            [math.factorial(x) for x in x_values]
        )
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return np.column_stack((x_values, pdf_values / np.sum(pdf_values)))


@pytest.fixture(
    params=[
        {"distribution": None, "distribution_args": {}},
        {
            "distribution": "normal",
            "distribution_args": {"loc": 5.0, "scale": 1.0},
        },
        {
            "distribution": "normal",
            "distribution_args": {"loc": 100.0, "scale": 15.0},
        },
        {"distribution": "uniform", "distribution_args": {"low": 3.4, "high": 9.8}},
        {
            "distribution": "exponential",
            "distribution_args": {"scale": 1.0},
        },
        {
            "distribution": "poisson",
            "distribution_args": {"lam": 5.0},
        },
    ]
)
def probability_distributions(
    request,
) -> tuple[str | None, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Create various probability distributions for testing.

    :return: A tuple containing the distribution type, the generated values,
        the pdf, and the correct distribution statistics.
    """
    distribution_type: str | None = request.param["distribution"]
    distribution_args: dict[str, float] = request.param["distribution_args"]

    num_samples = 10000
    rng = np.random.default_rng(seed=42)
    percentile_probs = {
        "p001": 0.001,
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.1,
        "p25": 0.25,
        "p50": 0.5,
        "p75": 0.75,
        "p90": 0.9,
        "p95": 0.95,
        "p99": 0.99,
        "p999": 0.999,
    }

    if distribution_type is None:
        # Empty / 0's distribution
        return (
            None,
            [],
            np.empty((0, 2)),
            {
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "total_sum": 0.0,
                "percentiles": dict.fromkeys(percentile_probs.keys(), 0.0),
            },
        )

    rng = np.random.default_rng(seed=42)
    samples = getattr(rng, distribution_type)(**distribution_args, size=num_samples)
    pdf = np.column_stack(
        (np.sort(samples), np.zeros_like(samples) + 1.0 / num_samples)
    )

    return (
        distribution_type,
        samples,
        pdf,
        {
            "mean": float(np.mean(samples)),
            "median": float(np.median(samples)),
            "variance": float(np.var(samples)),
            "std_dev": float(np.std(samples)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "count": int(len(samples)),
            "total_sum": float(np.sum(samples)),
            "percentiles": {
                key: float(np.percentile(samples, per * 100))
                for key, per in percentile_probs.items()
            },
        },
    )


def concurrency_distributions(
    concurrency_type: Literal[
        "sequential",
        "parallel",
        "constant_rate",
        "burst",
        "triangular_ramp",
        "normal_dist",
    ],
    num_requests: int = 100,
    start_time: float = 0.0,
    end_time: float = 100.0,
) -> tuple[
    Literal["sequential", "parallel", "constant_rate", "burst", "triangular_ramp"],
    np.ndarray,
    dict[str, float],
]:
    if concurrency_type == "sequential":
        timings = np.linspace(start_time, end_time, num_requests + 1)
        requests = np.column_stack((timings[:-1], timings[1:]))

        return (
            concurrency_type,
            requests,
            {
                "start_time": None,
                "end_time": None,
                "mean_concurrency": 1.0,
                "median_concurrency": 1.0,
                "std_dev_concurrency": 0.0,
            },
        )

    if concurrency_type == "parallel":
        requests = np.column_stack(
            (np.ones(num_requests) * start_time, np.ones(num_requests) * end_time)
        )

        return (
            concurrency_type,
            requests,
            {
                "start_time": None,
                "end_time": None,
                "mean_concurrency": num_requests,
                "median_concurrency": num_requests,
                "std_dev_concurrency": 0.0,
            },
        )

    if concurrency_type == "constant_rate":
        request_duration = (end_time - start_time) / 10
        timings = np.linspace(start_time, end_time - request_duration, num_requests)
        requests = np.column_stack((timings, timings + request_duration))
        request_delay = timings[1] - timings[0]
        rate = 1 / request_delay
        concurrency = rate * request_duration

        return (
            concurrency_type,
            requests,
            {
                "start_time": request_delay * concurrency,
                "end_time": end_time - request_delay * concurrency,
                "mean_concurrency": concurrency,
                "median_concurrency": concurrency,
                "std_dev_concurrency": 0.0,
            },
        )

    if concurrency_type == "burst":
        request_length = (end_time - start_time) / 10
        requests = np.column_stack(
            (
                np.repeat(start_time, num_requests),
                np.repeat(start_time + request_length, num_requests),
            )
        )

        fraction_active = request_length / (end_time - start_time)
        mean_concurrency_windowed = num_requests * fraction_active
        median_concurrency_windowed = 0.0 if fraction_active < 0.5 else num_requests
        variance = (
            fraction_active * (num_requests - mean_concurrency_windowed) ** 2
            + (1 - fraction_active) * mean_concurrency_windowed**2
        )
        std_dev_concurrency_windowed = variance**0.5

        return (
            concurrency_type,
            requests,
            {
                "start_time": start_time,
                "end_time": end_time,
                "mean_concurrency": mean_concurrency_windowed,
                "median_concurrency": median_concurrency_windowed,
                "std_dev_concurrency": std_dev_concurrency_windowed,
            },
        )

    if concurrency_type == "triangular_ramp":
        max_concurrency = num_requests
        ramp_up_time = (end_time - start_time) / 2
        request_duration = ramp_up_time
        timings = np.linspace(start_time, start_time + ramp_up_time, max_concurrency)
        requests = np.column_stack((timings, timings + request_duration))

        return (
            concurrency_type,
            requests,
            {
                "start_time": None,
                "end_time": None,
                "mean_concurrency": max_concurrency / 2,
                "median_concurrency": max_concurrency / 2,
                "std_dev_concurrency": max_concurrency / (2 * math.sqrt(3)),
            },
        )

    return None


class TestPercentiles:
    @pytest.fixture
    def valid_instances(
        self,
        probability_distributions: tuple[
            str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
    ) -> tuple[Percentiles, str | None, np.ndarray, np.ndarray, dict[str, float]]:
        dist_type, samples, pdf, stats = probability_distributions
        instance = Percentiles(
            p001=stats["percentiles"]["p001"],
            p01=stats["percentiles"]["p01"],
            p05=stats["percentiles"]["p05"],
            p10=stats["percentiles"]["p10"],
            p25=stats["percentiles"]["p25"],
            p50=stats["percentiles"]["p50"],
            p75=stats["percentiles"]["p75"],
            p90=stats["percentiles"]["p90"],
            p95=stats["percentiles"]["p95"],
            p99=stats["percentiles"]["p99"],
            p999=stats["percentiles"]["p999"],
        )
        return instance, dist_type, samples, pdf, stats

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(Percentiles, BaseModel)
        assert "p001" in Percentiles.model_fields
        assert "p01" in Percentiles.model_fields
        assert "p05" in Percentiles.model_fields
        assert "p10" in Percentiles.model_fields
        assert "p25" in Percentiles.model_fields
        assert "p50" in Percentiles.model_fields
        assert "p75" in Percentiles.model_fields
        assert "p90" in Percentiles.model_fields
        assert "p95" in Percentiles.model_fields
        assert "p99" in Percentiles.model_fields
        assert "p999" in Percentiles.model_fields
        assert hasattr(Percentiles, "from_pdf")

    @pytest.mark.smoke
    def test_initialization(
        self,
        valid_instances: tuple[
            DistributionSummary, Percentiles, np.ndarray, np.ndarray, dict[str, float]
        ],
    ):
        instance, _dist_type, _samples, _pdf, stats = valid_instances
        assert isinstance(instance, Percentiles)
        assert instance.p001 == stats["percentiles"]["p001"], "p001 percentile mismatch"
        assert instance.p01 == stats["percentiles"]["p01"], "p01 percentile mismatch"
        assert instance.p05 == stats["percentiles"]["p05"], "p05 percentile mismatch"
        assert instance.p10 == stats["percentiles"]["p10"], "p10 percentile mismatch"
        assert instance.p25 == stats["percentiles"]["p25"], "p25 percentile mismatch"
        assert instance.p50 == stats["percentiles"]["p50"], "p50 percentile mismatch"
        assert instance.p75 == stats["percentiles"]["p75"], "p75 percentile mismatch"
        assert instance.p90 == stats["percentiles"]["p90"], "p90 percentile mismatch"
        assert instance.p95 == stats["percentiles"]["p95"], "p95 percentile mismatch"
        assert instance.p99 == stats["percentiles"]["p99"], "p99 percentile mismatch"
        assert instance.p999 == stats["percentiles"]["p999"], "p999 percentile mismatch"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "missing_field",
        ["p001", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p999"],
    )
    def test_invalid_initialization(self, missing_field):
        test_kwargs = {
            "p001": 0.1,
            "p01": 1.0,
            "p05": 5.0,
            "p10": 10.0,
            "p25": 25.0,
            "p50": 50.0,
            "p75": 75.0,
            "p90": 90.0,
            "p95": 95.0,
            "p99": 99.0,
            "p999": 99.9,
        }
        del test_kwargs[missing_field]

        with pytest.raises(ValidationError):
            Percentiles(**test_kwargs)

    @pytest.mark.smoke
    def test_from_pdf(self, valid_instances):
        _instance, _dist_type, _values, pdf, stats = valid_instances

        tolerance = 0.1 * abs(stats["std_dev"])  # within 10% of standard deviation
        percentiles = Percentiles.from_pdf(pdf)
        assert percentiles.p001 == pytest.approx(
            stats["percentiles"]["p001"], abs=tolerance
        ), "p001 percentile mismatch"
        assert percentiles.p01 == pytest.approx(
            stats["percentiles"]["p01"], abs=tolerance
        ), "p01 percentile mismatch"
        assert percentiles.p05 == pytest.approx(
            stats["percentiles"]["p05"], abs=tolerance
        ), "p05 percentile mismatch"
        assert percentiles.p10 == pytest.approx(
            stats["percentiles"]["p10"], abs=tolerance
        ), "p10 percentile mismatch"
        assert percentiles.p25 == pytest.approx(
            stats["percentiles"]["p25"], abs=tolerance
        ), "p25 percentile mismatch"
        assert percentiles.p50 == pytest.approx(
            stats["percentiles"]["p50"], abs=tolerance
        ), "p50 percentile mismatch"
        assert percentiles.p75 == pytest.approx(
            stats["percentiles"]["p75"], abs=tolerance
        ), "p75 percentile mismatch"
        assert percentiles.p90 == pytest.approx(
            stats["percentiles"]["p90"], abs=tolerance
        ), "p90 percentile mismatch"
        assert percentiles.p95 == pytest.approx(
            stats["percentiles"]["p95"], abs=tolerance
        ), "p95 percentile mismatch"
        assert percentiles.p99 == pytest.approx(
            stats["percentiles"]["p99"], abs=tolerance
        ), "p99 percentile mismatch"
        assert percentiles.p999 == pytest.approx(
            stats["percentiles"]["p999"], abs=(tolerance * 2)
        ), "p999 percentile mismatch"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("values", "expected_count"),
        [
            ([], 0),
            ([5.0], 1),
            ([3.0, 7.0], 2),
        ],
    )
    def test_from_pdf_small_distributions(self, values, expected_count):
        """Test Percentiles.from_pdf with edge cases of 0, 1, 2 values."""
        if len(values) == 0:
            pdf = np.empty((0, 2))
        else:
            pdf = np.column_stack((values, np.ones(len(values)) / len(values)))

        percentiles = Percentiles.from_pdf(pdf)
        assert isinstance(percentiles, Percentiles)

        if expected_count == 0:
            # All percentiles should be 0 for empty distribution
            assert percentiles.p001 == 0.0
            assert percentiles.p50 == 0.0
            assert percentiles.p999 == 0.0
        elif expected_count == 1:
            # All percentiles should be the single value
            assert percentiles.p001 == values[0]
            assert percentiles.p50 == values[0]
            assert percentiles.p999 == values[0]
        elif expected_count == 2:
            # Percentiles should be one of the two values
            assert percentiles.p001 in values
            assert percentiles.p50 in values
            assert percentiles.p999 in values

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("pdf", "error_match"),
        [
            (np.array([1, 2, 3]), "must be a 2D array"),
            (np.array([[1, 2, 3]]), "must be a 2D array"),
            (np.array([[1.0, -0.5], [2.0, 0.5]]), "must be non-negative"),
            (np.array([[1.0, 0.3], [2.0, 0.5]]), "must sum to 1"),
        ],
    )
    def test_from_pdf_invalid(self, pdf, error_match):
        with pytest.raises(ValueError, match=error_match):
            Percentiles.from_pdf(pdf)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        instance, _dist_type, _values, _pdf, stats = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        for param in stats["percentiles"]:
            assert param in data_dict
            assert data_dict[param] == getattr(instance, param)

        recreated = Percentiles.model_validate(data_dict)
        assert isinstance(recreated, Percentiles)
        for param in stats["percentiles"]:
            assert getattr(recreated, param) == getattr(instance, param)


class TestDistributionSummary:
    @pytest.fixture
    def valid_instances(
        self,
        probability_distributions: tuple[
            str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
    ) -> tuple[
        DistributionSummary, str | None, np.ndarray, np.ndarray, dict[str, float]
    ]:
        dist_type, samples, pdf, stats = probability_distributions
        instance = DistributionSummary(
            mean=stats["mean"],
            median=stats["median"],
            mode=0.0,
            variance=stats["variance"],
            std_dev=stats["std_dev"],
            min=stats["min"],
            max=stats["max"],
            count=stats["count"],
            total_sum=stats["total_sum"],
            percentiles=Percentiles(**stats["percentiles"]),
            pdf=pdf,
        )

        return instance, dist_type, samples, pdf, stats

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert issubclass(DistributionSummary, BaseModel)
        assert "mean" in DistributionSummary.model_fields
        assert "median" in DistributionSummary.model_fields
        assert "mode" in DistributionSummary.model_fields
        assert "variance" in DistributionSummary.model_fields
        assert "std_dev" in DistributionSummary.model_fields
        assert "min" in DistributionSummary.model_fields
        assert "max" in DistributionSummary.model_fields
        assert "count" in DistributionSummary.model_fields
        assert "total_sum" in DistributionSummary.model_fields
        assert "percentiles" in DistributionSummary.model_fields
        assert "pdf" in DistributionSummary.model_fields
        assert hasattr(DistributionSummary, "from_pdf")
        assert hasattr(DistributionSummary, "from_values")
        assert hasattr(DistributionSummary, "rate_distribution_from_timings")
        assert hasattr(DistributionSummary, "concurrency_distribution_from_timings")

    @pytest.mark.smoke
    def test_initialization(
        self,
        valid_instances: tuple[
            DistributionSummary, str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
    ):
        instance, _dist_type, _samples, _pdf, stats = valid_instances
        assert instance.mean == stats["mean"]
        assert instance.median == stats["median"]
        assert instance.variance == stats["variance"]
        assert instance.std_dev == stats["std_dev"]
        assert instance.min == stats["min"]
        assert instance.max == stats["max"]
        assert instance.count == stats["count"]
        assert instance.total_sum == stats["total_sum"]
        assert isinstance(instance.percentiles, Percentiles)
        for param in stats["percentiles"]:
            assert getattr(instance.percentiles, param) == stats["percentiles"][param]
        assert instance.pdf is None or isinstance(instance.pdf, list)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "missing_field",
        [
            "mean",
            "median",
            "mode",
            "variance",
            "std_dev",
            "min",
            "max",
            "count",
            "total_sum",
            "percentiles",
        ],
    )
    def test_invalid_initialization(self, missing_field):
        test_kwargs = {
            "mean": 50.0,
            "median": 50.0,
            "mode": 50.0,
            "variance": 835.0,
            "std_dev": math.sqrt(835.0),
            "min": 0.0,
            "max": 100.0,
            "count": 1001,
            "total_sum": 50050.0,
            "percentiles": Percentiles(
                p001=0.1,
                p01=1.0,
                p05=5.0,
                p10=10.0,
                p25=25.0,
                p50=50.0,
                p75=75.0,
                p90=90.0,
                p95=95.0,
                p99=99.0,
                p999=99.9,
            ),
        }
        del test_kwargs[missing_field]

        with pytest.raises(ValidationError):
            DistributionSummary(**test_kwargs)

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_from_pdf(
        self,
        valid_instances: tuple[
            DistributionSummary, str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
        include_pdf: bool | int,
    ):
        _instance, _dist_type, values, pdf, stats = valid_instances

        tolerance = 0.1 * abs(stats["std_dev"])  # within 10% of standard deviation
        summary = DistributionSummary.from_pdf(pdf, include_pdf=include_pdf)
        assert summary.mean == pytest.approx(stats["mean"], abs=tolerance), (
            "mean mismatch"
        )
        assert summary.median == pytest.approx(stats["median"], abs=tolerance), (
            "median mismatch"
        )
        assert summary.variance == pytest.approx(stats["variance"], abs=tolerance), (
            "variance mismatch"
        )
        assert summary.std_dev == pytest.approx(stats["std_dev"], abs=tolerance), (
            "std_dev mismatch"
        )
        assert summary.min == pytest.approx(stats["min"], abs=tolerance), "min mismatch"
        assert summary.max == pytest.approx(stats["max"], abs=tolerance), "max mismatch"
        assert summary.count == stats["count"], "count mismatch"
        assert summary.total_sum == pytest.approx(stats["total_sum"], abs=tolerance), (
            "total_sum mismatch"
        )
        assert isinstance(summary.percentiles, Percentiles)
        for param in stats["percentiles"]:
            assert getattr(summary.percentiles, param) == pytest.approx(
                stats["percentiles"][param],
                abs=tolerance if param != "p999" else (tolerance * 2),
            ), f"{param} percentile mismatch"

        if include_pdf is False:
            assert summary.pdf is None
        elif include_pdf is True:
            assert summary.pdf is not None
            assert isinstance(summary.pdf, list)
            assert len(summary.pdf) == len(pdf)

    @pytest.mark.sanity
    @pytest.mark.parametrize("sample_size", [50, 100, 500])
    def test_from_pdf_with_sampling(self, sample_size: int):
        """Test DistributionSummary.from_pdf with integer sampling."""
        # Create a large PDF
        values = np.linspace(0, 100, 1000)
        probabilities = np.ones(1000) / 1000
        pdf = np.column_stack((values, probabilities))

        summary = DistributionSummary.from_pdf(pdf, include_pdf=sample_size)
        assert isinstance(summary, DistributionSummary)
        assert summary.pdf is not None
        assert len(summary.pdf) == sample_size

        # Verify sampled PDF still represents the distribution reasonably
        sampled_values = [val for val, _prob in summary.pdf]
        assert min(sampled_values) >= 0
        assert max(sampled_values) <= 100

        # Verify stats are accurate regardless of sampling
        assert summary.mean == pytest.approx(50.0, abs=1.0)
        assert summary.median == pytest.approx(50.0, abs=1.0)
        assert summary.min == pytest.approx(0.0, abs=1.0)
        assert summary.max == pytest.approx(100.0, abs=1.0)

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_from_values(
        self,
        valid_instances: tuple[
            DistributionSummary, str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
        include_pdf: bool | int,
    ):
        _instance, _dist_type, values, _pdf, stats = valid_instances

        tolerance = 0.1 * abs(stats["std_dev"])  # within 10% of standard deviation
        summary = DistributionSummary.from_values(values, include_pdf=include_pdf)
        assert summary.mean == pytest.approx(stats["mean"], abs=tolerance), (
            "mean mismatch"
        )
        assert summary.median == pytest.approx(stats["median"], abs=tolerance), (
            "median mismatch"
        )
        assert summary.variance == pytest.approx(stats["variance"], abs=tolerance), (
            "variance mismatch"
        )
        assert summary.std_dev == pytest.approx(stats["std_dev"], abs=tolerance), (
            "std_dev mismatch"
        )
        assert summary.min == pytest.approx(stats["min"], abs=tolerance), "min mismatch"
        assert summary.max == pytest.approx(stats["max"], abs=tolerance), "max mismatch"
        assert summary.count == stats["count"], "count mismatch"
        assert summary.total_sum == pytest.approx(stats["total_sum"], abs=tolerance), (
            "total_sum mismatch"
        )
        assert isinstance(summary.percentiles, Percentiles)
        for param in stats["percentiles"]:
            assert getattr(summary.percentiles, param) == pytest.approx(
                stats["percentiles"][param],
                abs=tolerance if param != "p999" else (tolerance * 2),
            ), f"{param} percentile mismatch"

        if include_pdf is False:
            assert summary.pdf is None
        elif include_pdf is True:
            assert summary.pdf is not None
            assert isinstance(summary.pdf, list)
            assert len(summary.pdf) > 0 if len(values) > 0 else len(summary.pdf) == 0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("limit_start_time", "limit_end_time", "include_pdf"),
        [
            (False, False, False),
            (True, False, True),
            (False, True, False),
            (True, True, True),
        ],
    )
    def test_rate_distribution_from_timings(
        self,
        valid_instances: tuple[
            DistributionSummary, str | None, np.ndarray, np.ndarray, dict[str, float]
        ],
        limit_start_time: bool,
        limit_end_time: bool,
        include_pdf: bool | int,
    ):
        _instance, dist_type, _values, pdf, stats = valid_instances

        if dist_type in ("exponential", "poisson"):
            pytest.skip(
                f"Skipping rate distribution test for {dist_type} distribution "
                "due to inherent variability and incompatibility with rate assumptions."
            )

        rng = np.random.default_rng(seed=42)

        if len(pdf) > 0:
            # The PDF gives the expected distribution for the rates
            # So, we can use it to sample individual, instantaneous rates
            # and convert those to timings by inverting and accumulating
            sampled_rates = rng.choice(pdf[:, 0], size=100000, p=pdf[:, 1])
            delta_times = 1.0 / np.clip(sampled_rates, a_min=1e-6, a_max=None)
            timings = np.cumsum(delta_times)
        else:
            timings = np.array([])

        # Now, compute the rate distribution from the timings and compare
        start_time = stats["mean"] if limit_start_time and len(timings) > 0 else None
        end_time = (
            np.max(timings) - stats["mean"]
            if limit_end_time and len(timings) > 0
            else None
        )
        distribution = DistributionSummary.rate_distribution_from_timings(
            timings, start_time=start_time, end_time=end_time, include_pdf=include_pdf
        )

        # Check expected nearly exact values (mean and count)
        expected_rate = (
            len(timings) / (timings[-1] - timings[0]) if len(timings) > 1 else 0.0
        )
        assert distribution.mean == pytest.approx(expected_rate, rel=10e-4), (
            "expected mean rate mismatch"
        )
        expected_count = len(timings)
        if start_time and len(timings) > 0:
            expected_count -= len(timings[timings < start_time])
        if end_time and len(timings) > 0:
            expected_count -= len(timings[timings > end_time])
        assert distribution.count == expected_count, "expected count mismatch"

        # Loosely validate against original stats (randomness in sampling)
        tolerance = 0.5 * abs(stats["std_dev"])  # within 10% of standard deviation
        assert distribution.mean == pytest.approx(stats["mean"], abs=tolerance), (
            "mean mismatch"
        )
        assert distribution.median == pytest.approx(stats["median"], abs=tolerance), (
            "median mismatch"
        )
        assert distribution.std_dev == pytest.approx(stats["std_dev"], abs=tolerance), (
            "std_dev mismatch"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("concurrency_type", "include_pdf"),
        [
            ("sequential", False),
            ("parallel", True),
            ("constant_rate", False),
            ("burst", True),
            ("triangular_ramp", False),
        ],
    )
    def test_concurrency_distribution_from_timings(self, concurrency_type, include_pdf):
        (
            _concurrency_type,
            requests,
            stats,
        ) = concurrency_distributions(concurrency_type, num_requests=1000)

        distribution = DistributionSummary.concurrency_distribution_from_timings(
            requests,
            start_time=stats["start_time"],
            end_time=stats["end_time"],
            include_pdf=include_pdf,
        )

        assert distribution.mean == pytest.approx(
            stats["mean_concurrency"], rel=1e-2
        ), "mean concurrency mismatch"
        assert distribution.median == pytest.approx(
            stats["median_concurrency"], rel=1e-2
        ), "median concurrency mismatch"
        assert distribution.std_dev == pytest.approx(
            stats["std_dev_concurrency"], rel=1e-2
        ), "std_dev concurrency mismatch"

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        instance, _dist_type, _values, _pdf, stats = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        for param in [
            "mean",
            "median",
            "mode",
            "variance",
            "std_dev",
            "min",
            "max",
            "count",
            "total_sum",
            "percentiles",
            "pdf",
        ]:
            assert param in data_dict
            if param == "percentiles":
                for p_param in stats["percentiles"]:
                    assert (
                        getattr(instance.percentiles, p_param)
                        == data_dict["percentiles"][p_param]
                    )
            else:
                assert data_dict[param] == getattr(instance, param)

        recreated = DistributionSummary.model_validate(data_dict)
        assert isinstance(recreated, DistributionSummary)
        for param in [
            "mean",
            "median",
            "mode",
            "variance",
            "std_dev",
            "min",
            "max",
            "count",
            "total_sum",
            "percentiles",
            "pdf",
        ]:
            if param == "percentiles":
                for p_param in stats["percentiles"]:
                    assert getattr(recreated.percentiles, p_param) == getattr(
                        instance.percentiles, p_param
                    )
            else:
                assert getattr(recreated, param) == getattr(instance, param)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("values", "error_type"),
        [
            ("not_a_list", ValueError),
            ({"invalid": "dict"}, ValueError),
            (None, TypeError),  # None raises TypeError instead of ValueError
        ],
    )
    def test_from_values_invalid_input(self, values, error_type):
        """Test DistributionSummary.from_values with invalid input types."""
        with pytest.raises(error_type):
            DistributionSummary.from_values(values)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("pdf", "error_match"),
        [
            (np.array([1, 2, 3]), "must be a 2D array"),
            (np.array([[1, 2, 3]]), "must be a 2D array"),
            (np.array([[1.0, -0.5], [2.0, 0.5]]), "must be non-negative"),
            (np.array([[1.0, 0.3], [2.0, 0.5]]), "must sum to 1"),
        ],
    )
    def test_from_pdf_invalid(self, pdf, error_match):
        """Test DistributionSummary.from_pdf with invalid PDFs."""
        with pytest.raises(ValueError, match=error_match):
            DistributionSummary.from_pdf(pdf)

    @pytest.mark.sanity
    def test_from_values_with_weights(self):
        """Test DistributionSummary.from_values with weighted values."""
        # Values with weights: (value, weight)
        values = [(1.0, 2.0), (2.0, 1.0), (3.0, 1.0)]
        summary = DistributionSummary.from_values(values)

        assert isinstance(summary, DistributionSummary)
        # Count is sum of weights: 2 + 1 + 1 = 4
        assert summary.count == 4
        # Mean should be weighted: (1*2 + 2*1 + 3*1) / (2+1+1) = 7/4 = 1.75
        assert summary.mean == pytest.approx(1.75, abs=0.01)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("values", "expected_count"),
        [
            ([], 0),
            ([5.0], 1),
            ([3.0, 7.0], 2),
        ],
    )
    def test_from_values_small_distributions(self, values, expected_count):
        """Test DistributionSummary.from_values with edge cases of 0, 1, 2 values."""
        summary = DistributionSummary.from_values(values)
        assert isinstance(summary, DistributionSummary)
        assert summary.count == expected_count

        if expected_count == 0:
            assert summary.mean == 0.0
            assert summary.median == 0.0
            assert summary.std_dev == 0.0
            assert summary.min == 0.0
            assert summary.max == 0.0
        elif expected_count == 1:
            assert summary.mean == values[0]
            assert summary.median == values[0]
            assert summary.std_dev == 0.0
            assert summary.min == values[0]
            assert summary.max == values[0]
        elif expected_count == 2:
            assert summary.mean == pytest.approx(np.mean(values), abs=1e-6)
            # For 2 values, median from PDF is the first value at CDF >= 0.5
            assert summary.median in values
            assert summary.min == min(values)
            assert summary.max == max(values)

    @pytest.mark.sanity
    def test_rate_distribution_empty_timings(self):
        """Test rate_distribution_from_timings with empty input."""
        summary = DistributionSummary.rate_distribution_from_timings([])
        assert summary.count == 0
        assert summary.mean == 0.0

    @pytest.mark.sanity
    def test_concurrency_distribution_empty_intervals(self):
        """Test concurrency_distribution_from_timings with empty input."""
        summary = DistributionSummary.concurrency_distribution_from_timings([])
        assert summary.count == 0
        assert summary.mean == 0.0

    @pytest.mark.sanity
    def test_rate_distribution_single_event(self):
        """Test rate_distribution_from_timings with single event."""
        summary = DistributionSummary.rate_distribution_from_timings([1.0])
        # Single event results in no rates (need at least 2 for intervals)
        assert summary.count == 0
        assert summary.mean == 0.0

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("timings", "expected_behavior"),
        [
            ([1.0, 1.0, 1.0], "duplicate_times"),
            ([1.0, 1.00001, 1.00002], "within_threshold"),
            ([5.0, 3.0, 1.0, 4.0, 2.0], "unsorted"),
        ],
    )
    def test_rate_distribution_edge_cases(self, timings, expected_behavior):
        """Test rate_distribution_from_timings with edge cases."""
        summary = DistributionSummary.rate_distribution_from_timings(
            timings, threshold=1e-4
        )
        assert isinstance(summary, DistributionSummary)
        assert summary.count >= 0

        if expected_behavior == "duplicate_times":
            # All duplicates merge to single time, no rate distribution
            assert summary.count in {0, 1}
        elif expected_behavior == "within_threshold":
            # Times within threshold should merge
            assert summary.count <= len(timings)
        elif expected_behavior == "unsorted":
            # Should handle unsorted input correctly
            assert summary.count > 0

    @pytest.mark.sanity
    def test_rate_distribution_with_weights(self):
        """Test rate_distribution_from_timings with weighted timestamps."""
        # Events with weights: (timestamp, weight)
        weighted_times = [(1.0, 2.0), (5.0, 1.0), (10.0, 3.0)]
        summary = DistributionSummary.rate_distribution_from_timings(weighted_times)
        assert isinstance(summary, DistributionSummary)
        # Total count should be sum of weights
        assert summary.count == 6

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("start_time", "end_time", "num_events"),
        [
            (5.0, None, 50),
            (None, 95.0, 50),
            (5.0, 95.0, 50),
        ],
    )
    def test_rate_distribution_time_windows(
        self, start_time: float | None, end_time: float | None, num_events: int
    ):
        """Test rate_distribution_from_timings with time window filtering."""
        rng = np.random.default_rng(seed=42)
        timings = sorted(rng.uniform(0, 100, num_events))

        summary = DistributionSummary.rate_distribution_from_timings(
            timings, start_time=start_time, end_time=end_time
        )
        assert isinstance(summary, DistributionSummary)

        # Count should be less than or equal to original when filtered
        if start_time or end_time:
            assert summary.count <= num_events
        else:
            assert summary.count == num_events

    @pytest.mark.sanity
    def test_concurrency_with_weighted_intervals(self):
        """Test concurrency_distribution_from_timings with weighted intervals."""
        # Intervals with weights: (start, end, weight)
        intervals = [(0.0, 10.0, 2.0), (5.0, 15.0, 1.0)]
        summary = DistributionSummary.concurrency_distribution_from_timings(intervals)

        assert isinstance(summary, DistributionSummary)
        assert summary.count == 2

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("intervals", "expected_behavior"),
        [
            ([(1.0, 1.0)], "zero_duration"),
            ([(1.0, 5.0), (2.0, 6.0), (3.0, 7.0)], "overlapping"),
            ([(1.0, 10.0), (2.0, 8.0), (3.0, 6.0)], "nested"),
            ([(1.0, 3.0), (5.0, 7.0), (9.0, 11.0)], "non_overlapping"),
        ],
    )
    def test_concurrency_distribution_edge_cases(self, intervals, expected_behavior):
        """Test concurrency_distribution_from_timings with various interval patterns."""
        summary = DistributionSummary.concurrency_distribution_from_timings(intervals)
        assert isinstance(summary, DistributionSummary)
        assert summary.count == len(intervals)

        if expected_behavior == "zero_duration":
            # Zero duration intervals should be handled
            assert summary.mean == 0.0 or summary.count == 1
        elif expected_behavior == "overlapping":
            # Overlapping intervals should show concurrency > 1 at some points
            assert summary.max >= 1.0
        elif expected_behavior == "nested":
            # Nested intervals should show max concurrency
            assert summary.max >= 2.0
        elif expected_behavior == "non_overlapping":
            # Non-overlapping intervals should have mean concurrency closer to 1
            assert summary.mean >= 0.0

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("start_time", "end_time"),
        [
            (5.0, None),
            (None, 75.0),
            (5.0, 75.0),
        ],
    )
    def test_concurrency_distribution_time_windows(
        self, start_time: float | None, end_time: float | None
    ):
        """Test concurrency_distribution_from_timings with time window filtering."""
        rng = np.random.default_rng(seed=42)
        num_intervals = 50
        starts = rng.uniform(0, 80, num_intervals)
        intervals = [(start, start + rng.uniform(5, 25)) for start in starts]

        summary = DistributionSummary.concurrency_distribution_from_timings(
            intervals, start_time=start_time, end_time=end_time
        )
        assert isinstance(summary, DistributionSummary)
        assert summary.count <= num_intervals

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("distribution", "dist_params", "sample_size"),
        [
            ("normal", {"loc": 50.0, "scale": 10.0}, 10000),
            ("uniform", {"low": 0.0, "high": 100.0}, 10000),
            ("exponential", {"scale": 5.0}, 10000),
        ],
    )
    def test_statistical_accuracy_against_numpy(
        self, distribution: str, dist_params: dict, sample_size: int
    ):
        """Validate statistical calculations against numpy's implementations."""
        rng = np.random.default_rng(seed=42)

        # Generate samples using numpy's distribution
        samples = getattr(rng, distribution)(**dist_params, size=sample_size)

        # Create distribution summary
        summary = DistributionSummary.from_values(samples)

        # Validate against numpy's statistics
        tolerance = 0.05 * abs(np.std(samples))  # 5% of std dev
        assert summary.mean == pytest.approx(np.mean(samples), abs=tolerance)
        assert summary.median == pytest.approx(np.median(samples), abs=tolerance)
        assert summary.variance == pytest.approx(np.var(samples), abs=tolerance * 2)
        assert summary.std_dev == pytest.approx(np.std(samples), abs=tolerance)
        assert summary.min == pytest.approx(np.min(samples), abs=tolerance)
        assert summary.max == pytest.approx(np.max(samples), abs=tolerance)
        assert summary.count == sample_size
        assert summary.total_sum == pytest.approx(
            np.sum(samples), abs=tolerance * sample_size
        )

        # Validate percentiles
        for percentile_name, percentile_value in {
            "p001": 0.1,
            "p01": 1.0,
            "p05": 5.0,
            "p10": 10.0,
            "p25": 25.0,
            "p50": 50.0,
            "p75": 75.0,
            "p90": 90.0,
            "p95": 95.0,
            "p99": 99.0,
            "p999": 99.9,
        }.items():
            expected = np.percentile(samples, percentile_value)
            actual = getattr(summary.percentiles, percentile_name)
            # Use larger tolerance for extreme percentiles (p001, p999)
            percentile_tolerance = (
                tolerance * 3 if percentile_name in {"p001", "p999"} else tolerance
            )
            assert actual == pytest.approx(expected, abs=percentile_tolerance)

    @pytest.mark.regression
    def test_pdf_normalization_accuracy(self):
        """Test that PDF creation maintains accurate probability normalization."""
        rng = np.random.default_rng(seed=42)
        samples = rng.normal(loc=100.0, scale=15.0, size=5000)

        summary = DistributionSummary.from_values(samples, include_pdf=True)

        assert summary.pdf is not None
        # Extract probabilities and verify they sum to approximately 1
        probabilities = np.array([prob for _val, prob in summary.pdf])
        assert np.sum(probabilities) == pytest.approx(1.0, abs=1e-6)

        # Verify all probabilities are non-negative
        assert np.all(probabilities >= 0.0)

        # Verify PDF values are sorted
        values = np.array([val for val, _prob in summary.pdf])
        assert np.all(values[:-1] <= values[1:])

    @pytest.mark.regression
    def test_weighted_statistics_accuracy(self):
        """Test that weighted statistics are calculated correctly."""
        # Known weighted distribution
        values_weights = [
            (10.0, 1.0),
            (20.0, 2.0),
            (30.0, 3.0),
            (40.0, 2.0),
            (50.0, 1.0),
        ]

        summary = DistributionSummary.from_values(values_weights)

        # Manual calculation for verification
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        expected_mean = np.sum(values * weights) / np.sum(weights)
        expected_count = int(np.sum(weights))
        expected_total = np.sum(values * weights)

        assert summary.count == expected_count
        assert summary.mean == pytest.approx(expected_mean, abs=1e-6)
        assert summary.total_sum == pytest.approx(expected_total, abs=1e-6)

        # Verify variance calculation
        expected_variance = np.sum(weights * (values - expected_mean) ** 2) / np.sum(
            weights
        )
        assert summary.variance == pytest.approx(expected_variance, abs=1e-6)


class TestStatusDistributionSummary:
    @pytest.fixture(
        params=[
            {
                "successful": [1.0, 2.0, 3.0],
                "incomplete": [4.0, 5.0],
                "errored": [6.0],
            },
            {
                "successful": np.array([10.0, 20.0, 30.0, 40.0]),
                "incomplete": np.array([50.0]),
                "errored": np.array([]),
            },
            {
                "successful": [],
                "incomplete": [],
                "errored": [],
            },
        ]
    )
    def valid_instances(
        self,
        request,
    ) -> tuple[StatusDistributionSummary, dict[str, list[float] | np.ndarray]]:
        """Fixture providing test data for StatusDistributionSummary."""
        test_data = request.param
        instance = StatusDistributionSummary.from_values(
            successful=test_data["successful"],
            incomplete=test_data["incomplete"],
            errored=test_data["errored"],
        )
        return instance, test_data

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test StatusDistributionSummary class structure and methods."""
        assert hasattr(StatusDistributionSummary, "from_values")
        assert hasattr(StatusDistributionSummary, "from_values_function")
        assert hasattr(StatusDistributionSummary, "rate_distribution_from_timings")
        assert hasattr(
            StatusDistributionSummary, "rate_distribution_from_timings_function"
        )
        assert hasattr(
            StatusDistributionSummary, "concurrency_distribution_from_timings"
        )
        assert hasattr(
            StatusDistributionSummary, "concurrency_distribution_from_timings_function"
        )
        assert "total" in StatusDistributionSummary.model_fields
        assert "successful" in StatusDistributionSummary.model_fields
        assert "incomplete" in StatusDistributionSummary.model_fields
        assert "errored" in StatusDistributionSummary.model_fields
        # Properties are checked via hasattr, actual property nature tested separately
        assert hasattr(StatusDistributionSummary, "count")
        assert hasattr(StatusDistributionSummary, "total_sum")

    @pytest.mark.smoke
    def test_initialization(
        self,
        valid_instances: tuple[
            StatusDistributionSummary, dict[str, list[float] | np.ndarray]
        ],
    ):
        """Test StatusDistributionSummary initialization."""
        instance, test_data = valid_instances
        assert isinstance(instance, StatusDistributionSummary)
        assert isinstance(instance.total, DistributionSummary)
        assert isinstance(instance.successful, DistributionSummary)
        assert isinstance(instance.incomplete, DistributionSummary)
        assert isinstance(instance.errored, DistributionSummary)

        # Verify counts match expected
        successful_count = (
            len(test_data["successful"])
            if isinstance(test_data["successful"], list)
            else test_data["successful"].shape[0]
        )
        incomplete_count = (
            len(test_data["incomplete"])
            if isinstance(test_data["incomplete"], list)
            else test_data["incomplete"].shape[0]
        )
        errored_count = (
            len(test_data["errored"])
            if isinstance(test_data["errored"], list)
            else test_data["errored"].shape[0]
        )

        assert instance.successful.count == successful_count
        assert instance.incomplete.count == incomplete_count
        assert instance.errored.count == errored_count
        assert (
            instance.total.count == successful_count + incomplete_count + errored_count
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("successful", "invalid_string"),
            ("incomplete", 123),
            ("errored", [1, 2, 3]),
            ("total", {"dict": "value"}),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test StatusDistributionSummary with invalid field types."""
        test_kwargs = {
            "successful": DistributionSummary.from_values([1.0, 2.0]),
            "incomplete": DistributionSummary.from_values([3.0]),
            "errored": DistributionSummary.from_values([]),
            "total": DistributionSummary.from_values([1.0, 2.0, 3.0]),
        }
        test_kwargs[field] = value

        with pytest.raises(ValidationError):
            StatusDistributionSummary(**test_kwargs)

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_from_values(
        self,
        valid_instances: tuple[
            StatusDistributionSummary, dict[str, list[float] | np.ndarray]
        ],
        include_pdf: bool | int,
    ):
        """Test creating StatusDistributionSummary from values."""
        _instance, test_data = valid_instances

        summary = StatusDistributionSummary.from_values(
            successful=test_data["successful"],
            incomplete=test_data["incomplete"],
            errored=test_data["errored"],
            include_pdf=include_pdf,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert isinstance(summary.total, DistributionSummary)
        assert isinstance(summary.successful, DistributionSummary)
        assert isinstance(summary.incomplete, DistributionSummary)
        assert isinstance(summary.errored, DistributionSummary)

        if include_pdf is False:
            assert summary.total.pdf is None
            assert summary.successful.pdf is None
            assert summary.incomplete.pdf is None
            assert summary.errored.pdf is None
        elif include_pdf is True:
            assert summary.total.pdf is not None or summary.total.count == 0
            assert summary.successful.pdf is not None or summary.successful.count == 0
            assert summary.incomplete.pdf is not None or summary.incomplete.count == 0
            assert summary.errored.pdf is not None or summary.errored.count == 0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("limit_start_time", "limit_end_time", "include_pdf"),
        [
            (False, False, False),
            (True, False, True),
            (False, True, False),
            (True, True, True),
        ],
    )
    def test_rate_distribution_from_timings(
        self,
        limit_start_time: bool,
        limit_end_time: bool,
        include_pdf: bool | int,
    ):
        """Test creating rate distribution from timings by status."""
        rng = np.random.default_rng(seed=42)
        successful_times = rng.uniform(0, 100, 50).tolist()
        incomplete_times = rng.uniform(0, 100, 20).tolist()
        errored_times = rng.uniform(0, 100, 10).tolist()

        start_time = 25.0 if limit_start_time else None
        end_time = 75.0 if limit_end_time else None

        summary = StatusDistributionSummary.rate_distribution_from_timings(
            successful=successful_times,
            incomplete=incomplete_times,
            errored=errored_times,
            start_time=start_time,
            end_time=end_time,
            include_pdf=include_pdf,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert isinstance(summary.total, DistributionSummary)
        assert isinstance(summary.successful, DistributionSummary)
        assert isinstance(summary.incomplete, DistributionSummary)
        assert isinstance(summary.errored, DistributionSummary)

        # Verify counts are reasonable
        assert summary.total.count >= 0
        assert summary.successful.count >= 0
        assert summary.incomplete.count >= 0
        assert summary.errored.count >= 0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "include_pdf",
        [
            False,
            True,
        ],
    )
    def test_concurrency_distribution_from_timings(self, include_pdf: bool | int):
        """Test creating concurrency distribution from intervals by status."""
        rng = np.random.default_rng(seed=42)
        num_successful = 30
        num_incomplete = 10
        num_errored = 5

        # Generate realistic intervals (start, end)
        successful_starts = rng.uniform(0, 80, num_successful)
        successful_intervals = [
            (start, start + rng.uniform(1, 20)) for start in successful_starts
        ]

        incomplete_starts = rng.uniform(0, 80, num_incomplete)
        incomplete_intervals = [
            (start, start + rng.uniform(1, 20)) for start in incomplete_starts
        ]

        errored_starts = rng.uniform(0, 80, num_errored)
        errored_intervals = [
            (start, start + rng.uniform(1, 20)) for start in errored_starts
        ]

        summary = StatusDistributionSummary.concurrency_distribution_from_timings(
            successful=successful_intervals,
            incomplete=incomplete_intervals,
            errored=errored_intervals,
            include_pdf=include_pdf,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert isinstance(summary.total, DistributionSummary)
        assert isinstance(summary.successful, DistributionSummary)
        assert isinstance(summary.incomplete, DistributionSummary)
        assert isinstance(summary.errored, DistributionSummary)

        # Verify counts match
        assert summary.successful.count == num_successful
        assert summary.incomplete.count == num_incomplete
        assert summary.errored.count == num_errored
        assert summary.total.count == num_successful + num_incomplete + num_errored

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test StatusDistributionSummary serialization and deserialization."""
        instance, _test_data = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert "total" in data_dict
        assert "successful" in data_dict
        assert "incomplete" in data_dict
        assert "errored" in data_dict

        # Verify each status has distribution summary data
        for status in ["total", "successful", "incomplete", "errored"]:
            assert isinstance(data_dict[status], dict)
            assert "mean" in data_dict[status]
            assert "median" in data_dict[status]
            assert "count" in data_dict[status]

        recreated = StatusDistributionSummary.model_validate(data_dict)
        assert isinstance(recreated, StatusDistributionSummary)
        assert recreated.total.count == instance.total.count
        assert recreated.successful.count == instance.successful.count
        assert recreated.incomplete.count == instance.incomplete.count
        assert recreated.errored.count == instance.errored.count

    @pytest.mark.smoke
    def test_count_property(self, valid_instances):
        """Test StatusDistributionSummary.count property."""
        instance, test_data = valid_instances
        assert isinstance(instance.count, int)
        assert instance.count == instance.total.count

        successful_count = (
            len(test_data["successful"])
            if isinstance(test_data["successful"], list)
            else test_data["successful"].shape[0]
        )
        incomplete_count = (
            len(test_data["incomplete"])
            if isinstance(test_data["incomplete"], list)
            else test_data["incomplete"].shape[0]
        )
        errored_count = (
            len(test_data["errored"])
            if isinstance(test_data["errored"], list)
            else test_data["errored"].shape[0]
        )

        expected_count = successful_count + incomplete_count + errored_count
        assert instance.count == expected_count

    @pytest.mark.smoke
    def test_total_sum_property(self, valid_instances):
        """Test StatusDistributionSummary.total_sum property."""
        instance, _test_data = valid_instances
        assert isinstance(instance.total_sum, float)
        assert instance.total_sum == instance.total.total_sum

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_from_values_function(self, include_pdf: bool | int):
        """Test creating StatusDistributionSummary from values via function."""

        class MockObject:
            def __init__(self, value: float):
                self.value = value

        def extract_value(obj: MockObject) -> float:
            return obj.value

        successful_objs = [MockObject(1.0), MockObject(2.0), MockObject(3.0)]
        incomplete_objs = [MockObject(4.0), MockObject(5.0)]
        errored_objs = [MockObject(6.0)]

        summary = StatusDistributionSummary.from_values_function(
            function=extract_value,
            successful=successful_objs,
            incomplete=incomplete_objs,
            errored=errored_objs,
            include_pdf=include_pdf,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert summary.successful.count == 3
        assert summary.incomplete.count == 2
        assert summary.errored.count == 1
        assert summary.total.count == 6

        # Verify means
        assert summary.successful.mean == pytest.approx(2.0, abs=0.01)
        assert summary.incomplete.mean == pytest.approx(4.5, abs=0.01)
        assert summary.errored.mean == pytest.approx(6.0, abs=0.01)

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_rate_distribution_from_timings_function(self, include_pdf: bool | int):
        """Test creating rate distribution from timings via function extraction."""

        class MockEvent:
            def __init__(self, timestamp: float):
                self.timestamp = timestamp

        def extract_timestamp(event: MockEvent) -> float:
            return event.timestamp

        rng = np.random.default_rng(seed=42)
        successful_events = [MockEvent(ts) for ts in rng.uniform(0, 100, 30)]
        incomplete_events = [MockEvent(ts) for ts in rng.uniform(0, 100, 15)]
        errored_events = [MockEvent(ts) for ts in rng.uniform(0, 100, 5)]

        summary = StatusDistributionSummary.rate_distribution_from_timings_function(
            function=extract_timestamp,
            successful=successful_events,
            incomplete=incomplete_events,
            errored=errored_events,
            include_pdf=include_pdf,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert isinstance(summary.total, DistributionSummary)
        assert isinstance(summary.successful, DistributionSummary)
        assert isinstance(summary.incomplete, DistributionSummary)
        assert isinstance(summary.errored, DistributionSummary)

        # Verify counts are reasonable
        assert summary.successful.count >= 0
        assert summary.incomplete.count >= 0
        assert summary.errored.count >= 0

    @pytest.mark.smoke
    @pytest.mark.parametrize("include_pdf", [False, True])
    def test_concurrency_distribution_from_timings_function(
        self, include_pdf: bool | int
    ):
        """Test creating concurrency distribution from intervals via function."""

        class MockRequest:
            def __init__(self, start: float, end: float):
                self.start = start
                self.end = end

        def extract_interval(request: MockRequest) -> tuple[float, float]:
            return (request.start, request.end)

        rng = np.random.default_rng(seed=42)
        num_successful = 20
        num_incomplete = 10
        num_errored = 5

        successful_starts = rng.uniform(0, 80, num_successful)
        successful_requests = [
            MockRequest(start, start + rng.uniform(1, 20))
            for start in successful_starts
        ]

        incomplete_starts = rng.uniform(0, 80, num_incomplete)
        incomplete_requests = [
            MockRequest(start, start + rng.uniform(1, 20))
            for start in incomplete_starts
        ]

        errored_starts = rng.uniform(0, 80, num_errored)
        errored_requests = [
            MockRequest(start, start + rng.uniform(1, 20)) for start in errored_starts
        ]

        summary = (
            StatusDistributionSummary.concurrency_distribution_from_timings_function(
                function=extract_interval,
                successful=successful_requests,
                incomplete=incomplete_requests,
                errored=errored_requests,
                include_pdf=include_pdf,
            )
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert isinstance(summary.total, DistributionSummary)
        assert isinstance(summary.successful, DistributionSummary)
        assert isinstance(summary.incomplete, DistributionSummary)
        assert isinstance(summary.errored, DistributionSummary)

        # Verify counts match
        assert summary.successful.count == num_successful
        assert summary.incomplete.count == num_incomplete
        assert summary.errored.count == num_errored
        assert summary.total.count == num_successful + num_incomplete + num_errored

    @pytest.mark.sanity
    def test_from_values_function_with_none_returns(self):
        """Test from_values_function when function returns None for some objects."""

        class MockObject:
            def __init__(self, value: float | None):
                self.value = value

        def extract_value(obj: MockObject) -> float | None:
            return obj.value

        successful_objs = [MockObject(1.0), MockObject(None), MockObject(3.0)]
        incomplete_objs = [MockObject(4.0), MockObject(None)]
        errored_objs = [MockObject(None), MockObject(6.0)]

        summary = StatusDistributionSummary.from_values_function(
            function=extract_value,
            successful=successful_objs,
            incomplete=incomplete_objs,
            errored=errored_objs,
        )

        assert isinstance(summary, StatusDistributionSummary)
        # Only non-None values should be counted
        assert summary.successful.count == 2
        assert summary.incomplete.count == 1
        assert summary.errored.count == 1
        assert summary.total.count == 4

    @pytest.mark.sanity
    def test_from_values_function_with_sequence_returns(self):
        """Test from_values_function when function returns sequences."""

        class MockObject:
            def __init__(self, values: list[float]):
                self.values = values

        def extract_values(obj: MockObject) -> list[float]:
            return obj.values

        successful_objs = [MockObject([1.0, 2.0]), MockObject([3.0])]
        incomplete_objs = [MockObject([4.0, 5.0, 6.0])]
        errored_objs = [MockObject([])]

        summary = StatusDistributionSummary.from_values_function(
            function=extract_values,
            successful=successful_objs,
            incomplete=incomplete_objs,
            errored=errored_objs,
        )

        assert isinstance(summary, StatusDistributionSummary)
        # Count should be total number of values extracted
        assert summary.successful.count == 3
        assert summary.incomplete.count == 3
        assert summary.errored.count == 0
        assert summary.total.count == 6

    @pytest.mark.sanity
    def test_rate_distribution_with_weighted_timestamps(self):
        """Test rate_distribution_from_timings with weighted timestamps by status."""
        weighted_successful = [(1.0, 2.0), (5.0, 1.0), (10.0, 3.0)]
        weighted_incomplete = [(2.0, 1.0), (8.0, 2.0)]
        weighted_errored = [(3.0, 1.0), (7.0, 2.0)]

        summary = StatusDistributionSummary.rate_distribution_from_timings(
            successful=weighted_successful,
            incomplete=weighted_incomplete,
            errored=weighted_errored,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert summary.successful.count == 6  # Sum of weights
        assert summary.incomplete.count == 3
        assert summary.errored.count == 3
        assert summary.total.count == 12

    @pytest.mark.sanity
    def test_concurrency_distribution_with_weighted_intervals(self):
        """Test concurrency_distribution_from_timings with weighted intervals."""
        weighted_successful = [(0.0, 10.0, 2.0), (5.0, 15.0, 1.0)]
        weighted_incomplete = [(8.0, 18.0, 3.0)]
        weighted_errored = [(12.0, 20.0, 1.0)]

        summary = StatusDistributionSummary.concurrency_distribution_from_timings(
            successful=weighted_successful,
            incomplete=weighted_incomplete,
            errored=weighted_errored,
        )

        assert isinstance(summary, StatusDistributionSummary)
        assert summary.successful.count == 2
        assert summary.incomplete.count == 1
        assert summary.errored.count == 1
        assert summary.total.count == 4
