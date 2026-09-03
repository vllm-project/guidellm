"""Shared pytest fixtures for unit tests."""

from __future__ import annotations

import pytest

from tests.unit.testing_utils import teardown_logger_state


@pytest.fixture(autouse=True)
def isolated_logger():  # noqa: PT004
    """Reset loguru handlers before and after every unit test."""
    teardown_logger_state()
    yield
    teardown_logger_state()
