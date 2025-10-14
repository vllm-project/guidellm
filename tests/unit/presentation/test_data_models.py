import pytest

from guidellm.presentation.data_models import Bucket


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
