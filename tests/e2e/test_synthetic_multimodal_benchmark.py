"""E2E benchmarks for synthetic image/video data and report metrics."""

from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)


@pytest.mark.smoke
@pytest.mark.timeout(90)
def test_synthetic_image_benchmark(server: E2EServer, tmp_path: Path):
    """Run synthetic image data through the CLI and validate image metrics.

    ## WRITTEN BY AI ##
    """
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs="image.json"
    )
    client.start_benchmark(
        rate=2,
        max_seconds=3,
        data=[
            "kind=synthetic_text,prompt_tokens=20",
            (
                "kind=synthetic_image,width=128,height=128,format=jpeg,"
                "jpeg_quality=85,output_tokens=8,seed=11"
            ),
        ],
        additional_args="--data-loader kind=pytorch,samples=8",
    )
    client.wait_for_completion(timeout=60)

    assert client.process is not None
    assert client.process.returncode == 0, (client.stdout, client.stderr)
    assert_no_python_exceptions(client.stderr)
    benchmark = load_benchmark_report(tmp_path / "image.json")["benchmarks"][0]
    requests = benchmark["requests"]["successful"]
    assert requests, "expected at least one successful request"

    for request in requests:
        input_metrics = request["input_metrics"]
        assert input_metrics["image_pixels"] == 128 * 128
        assert input_metrics["image_bytes"] > 0

    image_stats = benchmark["metrics"]["image"]["pixels"]["input"]["successful"]
    assert image_stats["mean"] > 0
    assert image_stats["median"] > 0
    assert image_stats["total_sum"] == pytest.approx(
        image_stats["mean"] * image_stats["count"]
    )


@pytest.mark.smoke
@pytest.mark.timeout(90)
def test_synthetic_video_benchmark(server: E2EServer, tmp_path: Path):
    """Run synthetic video data through the CLI and validate video metrics.

    ## WRITTEN BY AI ##
    """
    client = GuidellmClient(
        target=server.get_url(), output_dir=tmp_path, outputs="video.json"
    )
    client.start_benchmark(
        rate=2,
        max_seconds=4,
        data=[
            "kind=synthetic_text,prompt_tokens=10",
            (
                "kind=synthetic_video,width=160,height=120,frames=4,fps=1,"
                "output_tokens=4,seed=23"
            ),
        ],
        additional_args="--data-loader kind=pytorch,samples=8",
    )
    client.wait_for_completion(timeout=60)

    assert client.process is not None
    assert client.process.returncode == 0, (client.stdout, client.stderr)
    assert_no_python_exceptions(client.stderr)
    benchmark = load_benchmark_report(tmp_path / "video.json")["benchmarks"][0]
    requests = benchmark["requests"]["successful"]
    assert requests, "expected at least one successful request"

    for request in requests:
        input_metrics = request["input_metrics"]
        assert input_metrics["video_frames"] == 4
        assert input_metrics["video_seconds"] == pytest.approx(4.0)
        assert input_metrics["video_bytes"] > 0

    video_stats = benchmark["metrics"]["video"]["frames"]["input"]["successful"]
    assert video_stats["mean"] > 0
    assert video_stats["median"] > 0
    assert video_stats["total_sum"] == pytest.approx(
        video_stats["mean"] * video_stats["count"]
    )
