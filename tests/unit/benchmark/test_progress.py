"""Unit tests for GenerativeConsoleBenchmarkerProgress refresh helpers.

## WRITTEN BY AI ##
"""

from unittest.mock import MagicMock, patch

import pytest

from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress

_MONOTONIC = "guidellm.benchmark.progress.time.monotonic"


@pytest.fixture()
def progress():
    """Return a progress instance with Live internals stubbed out."""
    with patch("guidellm.benchmark.progress.Live.__init__", return_value=None):
        obj = GenerativeConsoleBenchmarkerProgress.__new__(
            GenerativeConsoleBenchmarkerProgress
        )
        obj.display_scheduler_stats = False
        obj.run_progress = None
        obj.run_progress_task = None
        obj.tasks_progress = None
        obj._last_refresh = 0.0
        obj.refresh_per_second = 4
        obj.refresh = MagicMock()
        return obj


class TestForceRefresh:
    @pytest.mark.smoke
    def test_calls_refresh_immediately(self, progress):
        """_force_refresh always calls refresh. ## WRITTEN BY AI ##"""
        progress._force_refresh()
        progress.refresh.assert_called_once()

    @pytest.mark.sanity
    def test_resets_last_refresh_timestamp(self, progress):
        """_force_refresh records current monotonic time. ## WRITTEN BY AI ##"""
        fake_now = 123.456
        with patch(_MONOTONIC, return_value=fake_now):
            progress._force_refresh()
        assert progress._last_refresh == fake_now

    @pytest.mark.sanity
    def test_always_refreshes_regardless_of_elapsed(self, progress):
        """_force_refresh bypasses the throttle interval. ## WRITTEN BY AI ##"""
        progress._last_refresh = 9999.0
        with patch(_MONOTONIC, return_value=9999.001):
            progress._force_refresh()
        progress.refresh.assert_called_once()


class TestThrottledRefresh:
    @pytest.mark.smoke
    def test_refreshes_when_interval_elapsed(self, progress):
        """_throttled_refresh renders after the interval passes. ## WRITTEN BY AI ##"""
        progress._last_refresh = 0.0
        with patch(_MONOTONIC, return_value=0.26):
            progress._throttled_refresh()
        progress.refresh.assert_called_once()

    @pytest.mark.smoke
    def test_skips_refresh_when_called_too_soon(self, progress):
        """_throttled_refresh is a no-op within the interval. ## WRITTEN BY AI ##"""
        progress._last_refresh = 0.0
        with patch(_MONOTONIC, return_value=0.10):
            progress._throttled_refresh()
        progress.refresh.assert_not_called()

    @pytest.mark.sanity
    def test_updates_timestamp_on_refresh(self, progress):
        """_throttled_refresh records the new time on render. ## WRITTEN BY AI ##"""
        fake_now = 5.0
        progress._last_refresh = 0.0
        with patch(_MONOTONIC, return_value=fake_now):
            progress._throttled_refresh()
        assert progress._last_refresh == fake_now

    @pytest.mark.sanity
    def test_does_not_update_timestamp_when_skipped(self, progress):
        """_throttled_refresh leaves _last_refresh unchanged when skipped. ## WRITTEN BY AI ##"""
        progress._last_refresh = 10.0
        with patch(_MONOTONIC, return_value=10.05):
            progress._throttled_refresh()
        assert progress._last_refresh == 10.0

    @pytest.mark.sanity
    def test_respects_refresh_per_second_boundary(self, progress):
        """_throttled_refresh fires exactly at the interval boundary. ## WRITTEN BY AI ##"""
        progress._last_refresh = 0.0
        interval = 1.0 / progress.refresh_per_second  # 0.25 s

        with patch(_MONOTONIC, return_value=interval - 0.001):
            progress._throttled_refresh()
        progress.refresh.assert_not_called()

        with patch(_MONOTONIC, return_value=interval):
            progress._throttled_refresh()
        progress.refresh.assert_called_once()
