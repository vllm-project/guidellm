import json
from pathlib import Path

import pytest

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import SweepProfile, SynchronousProfile
from guidellm.benchmark.schemas import BenchmarkGenerativeTextArgs
from guidellm.scheduler import (
    AsyncConstantStrategy,
    ConstraintsInitializerFactory,
    SynchronousStrategy,
    ThroughputStrategy,
)

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_resolve_profile_allows_per_constraints_for_sweep():
    profile = await resolve_profile(
        profile="sweep",
        rate=[5],
        random_seed=123,
        rampup=0.0,
        constraints={},
        max_seconds=None,
        max_requests=None,
        max_errors=None,
        max_error_rate=None,
        max_global_error_rate=None,
        console=None,
        per_constraints={"max_seconds": [1, 2, 3, 4, 5]},
    )

    assert isinstance(profile, SweepProfile)
    assert profile.per_constraints == {"max_seconds": [1, 2, 3, 4, 5]}

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_resolve_profile_rejects_per_constraints_for_non_sweep():
    with pytest.raises(ValueError, match="Per-strategy constraints are only supported with the 'sweep' profile."):
        await resolve_profile(
            profile="synchronous",
            rate=None,
            random_seed=123,
            rampup=0.0,
            constraints={},
            max_seconds=None,
            max_requests=None,
            max_errors=None,
            max_error_rate=None,
            max_global_error_rate=None,
            console=None,
            per_constraints={"max_seconds": [1]},
        )

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_resolve_profile_rejects_per_constraints_for_instances():
    synchronous_profile = SynchronousProfile()

    with pytest.raises(
        ValueError, match="Per-strategy constraints cannot be applied"
    ):
        await resolve_profile(
            profile=synchronous_profile,
            rate=None,
            random_seed=123,
            rampup=0.0,
            constraints={},
            max_seconds=None,
            max_requests=None,
            max_errors=None,
            max_error_rate=None,
            max_global_error_rate=None,
            console=None,
            per_constraints={"max_seconds": [1]},
        )

@pytest.mark.smoke
def test_sweep_profile_applies_per_constraints_sequence(monkeypatch):
    captured: list[dict[str, int]] = []

    def fake_resolve(value):
        captured.append(value)
        return value

    monkeypatch.setattr(
        ConstraintsInitializerFactory, "resolve", staticmethod(fake_resolve)
    )

    profile = SweepProfile(
        sweep_size=3,
        per_constraints={"max_seconds": [5, 10, 15]},
        constraints={"max_seconds": 30, "max_requests": 100},
    )

    sync = SynchronousStrategy()
    profile.next_strategy_constraints(sync, None, None)
    assert captured[-1]["max_seconds"] == 5
    assert captured[-1]["max_requests"] == 100

    profile.completed_strategies.append(sync)
    throughput = ThroughputStrategy(max_concurrency=1, rampup_duration=0.0)
    profile.next_strategy_constraints(throughput, sync, None)
    assert captured[-1]["max_seconds"] == 10
    assert captured[-1]["max_requests"] == 100

    profile.completed_strategies.append(throughput)
    async_strategy = AsyncConstantStrategy(rate=1.0, max_concurrency=None)
    profile.next_strategy_constraints(async_strategy, throughput, None)
    assert captured[-1]["max_seconds"] == 15
    assert captured[-1]["max_requests"] == 100

@pytest.mark.smoke
def test_benchmark_args_accept_per_constraints_from_scenario(tmp_path: Path):
    scenario_path = tmp_path / "scenario.json"
    scenario_content = {
        "target": "http://localhost:9000",
        "data": ["prompt_tokens=8,output_tokens=8"],
        "profile": "sweep",
        "rate": 5,
        "per_constraints": {"max_seconds": [5, 10, 15, 15, 20], "max_requests": [100, 200, 200, 400, 400]},
    }
    scenario_path.write_text(json.dumps(scenario_content))

    args = BenchmarkGenerativeTextArgs.create(scenario=scenario_path)

    assert args.per_constraints == {"max_seconds": [5, 10, 15, 15, 20], "max_requests": [100, 200, 200, 400, 400]}


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_resolve_profile_rejects_null_per_constraints():
    with pytest.raises(ValueError, match="Per-strategy constraints for 'max_seconds' contain null values, which are not allowed."):
        await resolve_profile(
            profile="sweep",
            rate=[5],
            random_seed=123,
            rampup=0.0,
            constraints={},
            max_seconds=None,
            max_requests=None,
            max_errors=None,
            max_error_rate=None,
            max_global_error_rate=None,
            console=None,
            per_constraints={"max_seconds": [5, None, 15, 20, 25, 30]},
        )


@pytest.mark.smoke
def test_sweep_profile_rejects_null_per_constraints():
    with pytest.raises(ValueError, match="Per-strategy constraints for 'max_requests' contain null values, which are not allowed."):
        SweepProfile(
            sweep_size=5,
            per_constraints={"max_requests": [100, None, 200, 300, 400]},
        )
