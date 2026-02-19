from __future__ import annotations

from unittest.mock import Mock

import pytest

from guidellm.benchmark.profiles import (
    AsyncConstantStrategy,
    Profile,
    SweepProfile,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.scheduler import Constraint, ConstraintsInitializerFactory, SchedulingStrategy, MaxDurationConstraint, MaxNumberConstraint


def test_sweep_profile_strategies_generator_adaptive_rates():
    """
    Tests that the SweepProfile strategies_generator yields the correct sequence of
    strategies with adaptively calculated rates.
    """
    # 1. Initialize SweepProfile
    profile = SweepProfile(sweep_size=4, strategy_type="constant", max_concurrency=16)
    generator = profile.strategies_generator()

    # 2. First step should be SynchronousStrategy
    strategy, constraints = next(generator)
    assert isinstance(strategy, SynchronousStrategy)

    # 3. Send mock benchmark result for the synchronous run
    mock_sync_benchmark = Mock()
    mock_sync_benchmark.request_throughput.successful.mean = 50.0
    strategy, constraints = generator.send(mock_sync_benchmark)

    # 4. Second step should be ThroughputStrategy
    assert isinstance(strategy, ThroughputStrategy)
    assert strategy.max_concurrency == 16

    # 5. Send mock benchmark result for the throughput run
    mock_throughput_benchmark = Mock()
    mock_throughput_benchmark.request_throughput.successful.mean = 200.0
    strategy, constraints = generator.send(mock_throughput_benchmark)

    # The profile should now have calculated the rates for the async strategies.
    # np.linspace(50, 200, 3) -> [50., 125., 200.]. After slicing [1:], it's [125., 200.]
    assert profile.measured_rates == [125.0, 200.0]

    # 6. Third step should be AsyncConstantStrategy with the first calculated rate
    assert isinstance(strategy, AsyncConstantStrategy)
    assert strategy.rate == 125.0
    assert strategy.max_concurrency == 16

    # 7. Send a dummy benchmark result
    mock_async_benchmark_1 = Mock()
    strategy, constraints = generator.send(mock_async_benchmark_1)

    # 8. Fourth step should be AsyncConstantStrategy with the second calculated rate
    assert isinstance(strategy, AsyncConstantStrategy)
    assert strategy.rate == 200.0
    assert strategy.max_concurrency == 16

    # 9. Send the final dummy benchmark, expecting the generator to stop
    mock_async_benchmark_2 = Mock()
    with pytest.raises(StopIteration):
        generator.send(mock_async_benchmark_2)


def test_sweep_profile_strategy_constraints():
    """
    Tests that the SweepProfile applies both shared and per-strategy constraints
    correctly at each step of the strategy generation process.
    """
    # 1. Initialize SweepProfile with both shared and per-strategy constraints.
    # `max_duration` is shared across all steps.
    # `max_requests` has a specific value for each step.
    # `max_errors` is specified for some steps and disabled (None) for others.
    profile = SweepProfile(
        sweep_size=5,
        strategy_type="constant",
        rate=[1.0],  # Dummy rate, not directly used by constraints test
        max_duration=60,
        per_constraints={
            "max_requests": [10, 100, 200, 300, 400],
            "max_errors": [1, 5, 10, 15, 20],
        },
    )

    # 2. Verify that constraints were parsed and separated correctly.
    assert profile.per_constraints == {
        "max_requests": [10, 100, 200, 300, 400],
        "max_errors": [1, 5, 10, 15, 20],
    }

    generator = profile.strategies_generator()
    mock_benchmark = Mock()
    mock_benchmark.request_throughput.successful.mean = 50.0

    # 3. Test Step 1: Synchronous Strategy
    strategy, constraints_dict = next(generator)
    assert isinstance(strategy, SynchronousStrategy)
    assert constraints_dict["max_requests"].max_num == 10

    # 4. Test Step 2: Throughput Strategy
    strategy, constraints_dict = generator.send(mock_benchmark)
    assert isinstance(strategy, ThroughputStrategy)
    assert constraints_dict["max_requests"].max_num == 100
    assert constraints_dict["max_errors"].max_errors == 5

    # 5. Test Step 3, 4, 5: Async Strategies
    expected_async_max_requests = [200, 300, 400]
    expected_async_max_errors = [10, 15, 20]
    for i in range(3):
        strategy, constraints_dict  = generator.send(mock_benchmark)
        assert isinstance(strategy, AsyncConstantStrategy)
        assert constraints_dict is not None
        # Check shared and per-strategy constraints for this async step
        assert constraints_dict["max_requests"].max_num  == expected_async_max_requests[i]
        if expected_async_max_errors[i] is not None:
            assert "max_errors" in constraints_dict
            assert constraints_dict["max_errors"].max_errors == expected_async_max_errors[i]
        else:
            assert "max_errors" not in constraints_dict

    # 6. Expect StopIteration after the last step
    with pytest.raises(StopIteration):
        generator.send(mock_benchmark)
