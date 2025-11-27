from guidellm.benchmark.outputs.html import _filter_duplicate_percentiles
from guidellm.schemas import Percentiles


def test_filter_all_same_values():
    """Test filtering when all percentiles have the same value."""
    percentiles = {
        "p001": 15.288091352804853,
        "p01": 15.288091352804853,
        "p05": 15.288091352804853,
        "p10": 15.288091352804853,
        "p25": 15.288091352804853,
        "p50": 15.288091352804853,
        "p75": 15.288091352804853,
        "p90": 15.288091352804853,
        "p95": 15.288091352804853,
        "p99": 15.288091352804853,
        "p999": 15.288091352804853,
    }

    filtered = _filter_duplicate_percentiles(percentiles)

    # Should only keep the first one
    assert filtered == {"p001": 15.288091352804853}


def test_filter_consecutive_duplicates():
    """Test filtering when some consecutive percentiles have the same value."""
    percentiles = {
        "p001": 15.288091352804853,
        "p01": 15.288091352804853,
        "p05": 15.288091352804853,
        "p10": 15.288091352804853,
        "p25": 15.288091352804853,
        "p50": 16.41327511776994,  # Different value
        "p75": 16.41327511776994,
        "p90": 17.03541629998259,  # Different value
        "p95": 17.03541629998259,
        "p99": 17.03541629998259,
        "p999": 17.03541629998259,
    }

    filtered = _filter_duplicate_percentiles(percentiles)

    # Should keep first of each group
    assert filtered == {
        "p001": 15.288091352804853,
        "p50": 16.41327511776994,
        "p90": 17.03541629998259,
    }


def test_no_duplicates():
    """Test that unique values are all preserved."""
    percentiles = {
        "p001": 13.181080445834912,
        "p01": 13.181080445834912,  # Same as p001
        "p05": 13.530595573836457,  # Different
        "p10": 13.843972502554365,
        "p25": 14.086376978251748,
        "p50": 14.403258051191058,
        "p75": 14.738608817056042,
        "p90": 15.18136631856698,
        "p95": 15.7213110894772,
        "p99": 15.7213110894772,  # Same as p95
        "p999": 15.7213110894772,  # Same as p99
    }

    filtered = _filter_duplicate_percentiles(percentiles)

    assert filtered == {
        "p001": 13.181080445834912,
        "p05": 13.530595573836457,
        "p10": 13.843972502554365,
        "p25": 14.086376978251748,
        "p50": 14.403258051191058,
        "p75": 14.738608817056042,
        "p90": 15.18136631856698,
        "p95": 15.7213110894772,
    }


def test_empty_percentiles():
    """Test with empty percentiles dictionary."""
    filtered = _filter_duplicate_percentiles({})
    assert filtered == {}


def test_single_percentile():
    """Test with only one percentile."""
    percentiles = {"p50": 14.403258051191058}
    filtered = _filter_duplicate_percentiles(percentiles)
    assert filtered == {"p50": 14.403258051191058}


def test_two_different_values():
    """Test with two different values."""
    percentiles = {
        "p25": 14.086376978251748,
        "p50": 14.403258051191058,
    }
    filtered = _filter_duplicate_percentiles(percentiles)
    assert filtered == percentiles


def test_partial_percentiles():
    """Test that order is maintained even with partial percentiles."""
    percentiles = {
        "p50": 16.41327511776994,
        "p10": 15.288091352804853,
        "p90": 17.03541629998259,
    }

    filtered = _filter_duplicate_percentiles(percentiles)

    # Should maintain order from percentile_order list
    assert list(filtered.keys()) == ["p10", "p50", "p90"]


def test_model_dump_filters_duplicates():
    """Test that model_dump applies percentile filtering."""
    from guidellm.benchmark.outputs.html import _TabularDistributionSummary

    # Create a distribution with duplicate percentiles (typical of small datasets)
    dist = _TabularDistributionSummary(
        mean=15.5,
        median=15.288091352804853,
        mode=15.288091352804853,
        variance=0.1,
        std_dev=0.316,
        min=15.288091352804853,
        max=17.03541629998259,
        count=3,
        total_sum=46.5,
        percentiles=Percentiles(
            p001=15.288091352804853,
            p01=15.288091352804853,
            p05=15.288091352804853,
            p10=15.288091352804853,
            p25=15.288091352804853,
            p50=16.41327511776994,
            p75=16.41327511776994,
            p90=17.03541629998259,
            p95=17.03541629998259,
            p99=17.03541629998259,
            p999=17.03541629998259,
        ),
    )

    data = dist.model_dump()

    # Check that percentiles were filtered
    assert data["percentiles"] == {
        "p001": 15.288091352804853,
        "p50": 16.41327511776994,
        "p90": 17.03541629998259,
    }

    # Ensure other fields remain unchanged
    assert data["mean"] == 15.5
    assert data["median"] == 15.288091352804853
    assert data["count"] == 3
