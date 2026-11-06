"""Tests for users profile scheduling and strategy generation."""

from unittest.mock import MagicMock

import pytest

from guidellm.benchmark.profiles import ProfileFactory, UsersProfile
from guidellm.scheduler import SchedulerState, SchedulerUpdateAction, UsersStrategy
from guidellm.schemas.benchmark import ProfileArgs, UsersProfileArgs

USER_COUNTS = [4, 8, 16]


def _profile(**kwargs):
    """Build a users profile over the default user counts.

    ## WRITTEN BY AI ##
    """
    return UsersProfile(UsersProfileArgs(users=USER_COUNTS, **kwargs), 42, {})


def _run_profile(profile):
    """Exercise the result feedback protocol used by Benchmarker.run.

    ## WRITTEN BY AI ##
    """
    generator = profile.strategies_generator()
    strategy, _ = next(generator)
    strategies = []
    while True:
        strategies.append(strategy)
        try:
            strategy, _ = generator.send(None)
        except StopIteration:
            return strategies


@pytest.mark.smoke
def test_users_profile_is_registered():
    """Create users scheduling through the standard factory.

    ## WRITTEN BY AI ##
    """
    args = ProfileArgs.model_validate({"kind": "users", "users": USER_COUNTS})
    profile = ProfileFactory.create(args, 42, {})
    assert isinstance(profile, UsersProfile)
    assert profile.kind == "users"
    assert profile.strategy_types == ["users"] * len(USER_COUNTS)
    assert profile.conclusion is None


@pytest.mark.regression
def test_users_profile_runs_each_user_count_in_order():
    """Run one users strategy per configured user count, in order.

    ## WRITTEN BY AI ##
    """
    profile = _profile(rampup_duration=1.0)
    strategies = _run_profile(profile)
    assert [strategy.users for strategy in strategies] == USER_COUNTS
    assert all(isinstance(strategy, UsersStrategy) for strategy in strategies)
    assert all(strategy.rampup_duration == 1.0 for strategy in strategies)
    assert profile.next_strategy(None, None) is None


@pytest.mark.regression
@pytest.mark.parametrize("stop_index", [0, 1])
def test_users_profile_stops_escalating_on_global_constraint(stop_index):
    """A stopping_scope='all' constraint skips the remaining user counts.

    ## WRITTEN BY AI ##
    """
    profile = _profile()
    state = SchedulerState()
    state.end_queuing_constraints["limit"] = SchedulerUpdateAction(stopping_scope="all")
    halted = MagicMock(scheduler_state=state)

    generator = profile.strategies_generator()
    strategy, _ = next(generator)
    strategies = [strategy]
    for _ in range(stop_index):
        strategy, _ = generator.send(MagicMock(scheduler_state=SchedulerState()))
        strategies.append(strategy)

    with pytest.raises(StopIteration):
        generator.send(halted)

    assert [strategy.users for strategy in strategies] == USER_COUNTS[: stop_index + 1]
