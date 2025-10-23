import pytest

from guidellm.presentation.data_models import BenchmarkDatum, Bucket
from tests.unit.mock_benchmark import mock_generative_benchmark


@pytest.mark.smoke
def test_bucket_from_data():
    buckets, bucket_width = Bucket.from_data([8, 8, 8, 8, 8, 8], 1)
    assert len(buckets) == 1
    assert buckets[0].value == 8.0
    assert buckets[0].count == 6
    assert bucket_width == 1

    buckets, bucket_width = Bucket.from_data([8, 8, 8, 8, 8, 7], 1)
    assert len(buckets) == 2
    assert buckets[0].value == 7.0
    assert buckets[0].count == 1
    assert buckets[1].value == 8.0
    assert buckets[1].count == 5
    assert bucket_width == 1


@pytest.mark.smoke
def test_from_benchmark_includes_strategy_display_str():
    mock_bm = mock_generative_benchmark()
    bm = BenchmarkDatum.from_benchmark(mock_bm)
    assert bm.strategy_display_str == "synchronous"
