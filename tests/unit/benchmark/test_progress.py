"""Unit tests for GenerativeConsoleBenchmarkerProgress refresh helpers."""

from unittest.mock import MagicMock, patch

import pytest

from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress


@pytest.fixture()
def progress():
    """Return a progress instance with Live internals stubbed out."""
    with patch("guidellm.benchmark.progress.Live.__init__", return_value=None):
        obj = GenerativeConsoleBenchmarkerProgress.__new__(
            GenerativeConsoleBenchmarkerProgress
        )
        # Replicate __init__ state without starting Live
        obj.display_scheduler_stats = False
        obj.run_progress = None
        obj.run_progress_task = None
        obj.tasks_progress = None
        obj._last_refresh = 0.0
        obj.refresh_per_second = 4
        obj.refresh = MagicMock()
        return obj


class TestForceRefresh:
    def test_calls_refresh_immediately(self, progress):
        progress._force_refresh()
        progress.refresh.assert_called_once()

    def test_resets_last_refresh_timestamp(self, progress):
        fake_now = 123.456
        with patch("guidellm.benchmark.progress.time.monotonic", return_value=fake_now):
            progress._force_refresh()
        assert progress._last_refresh == fake_now

    def test_always_refreshes_regardless_of_elapsed(self, progress):
        progress._last_refresh = 9999.0
        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=9999.001
        ):
            progress._force_refresh()
        progress.refresh.assert_called_once()


class TestThrottledRefresh:
    def test_refreshes_when_interval_elapsed(self, progress):
        progress._last_refresh = 0.0
        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=0.26
        ):
            progress._throttled_refresh()
        progress.refresh.assert_called_once()

    def test_skips_refresh_when_called_too_soon(self, progress):
        progress._last_refresh = 0.0
        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=0.10
        ):
            progress._throttled_refresh()
        progress.refresh.assert_not_called()

    def test_updates_timestamp_on_refresh(self, progress):
        fake_now = 5.0
        progress._last_refresh = 0.0
        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=fake_now
        ):
            progress._throttled_refresh()
        assert progress._last_refresh == fake_now

    def test_does_not_update_timestamp_when_skipped(self, progress):
        progress._last_refresh = 10.0
        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=10.05
        ):
            progress._throttled_refresh()
        assert progress._last_refresh == 10.0

    def test_respects_refresh_per_second_boundary(self, progress):
        progress._last_refresh = 0.0
        interval = 1.0 / progress.refresh_per_second  # 0.25s

        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=interval - 0.001
        ):
            progress._throttled_refresh()
        progress.refresh.assert_not_called()

        with patch(
            "guidellm.benchmark.progress.time.monotonic", return_value=interval
        ):
            progress._throttled_refresh()
        progress.refresh.assert_called_once()
