"""Configuration coverage for the knee profile."""

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import BenchmarkScenario, KneeProfileArgs, ProfileArgs


@pytest.mark.smoke
def test_knee_profile_configuration_round_trip():
    """Keep knee options inside the registered profile configuration.

    ## WRITTEN BY AI ##
    """
    scenario = BenchmarkScenario.create(
        scenario=None,
        spec={
            "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
            "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
            "profile": {
                "kind": "knee",
                "streams": [1, 5, 10, 20, 40],
                "adaptive": True,
                "points_each_side": 3,
                "max_step": 2,
                "rampup_duration": 1.0,
                "warmup": 2.0,
            },
        },
    )
    restored = BenchmarkScenario.model_validate_json(scenario.model_dump_json())
    assert isinstance(restored.spec.profile, KneeProfileArgs)
    assert restored.spec.profile == scenario.spec.profile
    assert restored.spec.profile.adaptive is True
    assert "knee_detection" not in restored.model_dump()


@pytest.mark.sanity
@pytest.mark.parametrize("streams", [4, [4], "4", "[4]"])
def test_knee_profile_defaults_to_analysis_only(streams):
    """Normalize CLI stream input without enabling adaptive execution by default.

    ## WRITTEN BY AI ##
    """
    args = ProfileArgs.model_validate({"kind": "knee", "streams": streams})
    assert isinstance(args, KneeProfileArgs)
    assert args.streams == [4]
    assert args.adaptive is False
    assert args.points_each_side == 5
    assert args.max_step == 5


@pytest.mark.sanity
@pytest.mark.parametrize("streams", [[], [1, 1], [0], [-1], [True], [1.5], "invalid"])
def test_knee_profile_rejects_invalid_streams(streams):
    """Reject ambiguous or invalid concurrency measurements before execution.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        KneeProfileArgs(streams=streams)


@pytest.mark.sanity
@pytest.mark.parametrize("field", ["points_each_side", "max_step"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_knee_profile_rejects_invalid_grid_options(field, value):
    """Require positive integer grid controls.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        KneeProfileArgs.model_validate({"streams": [1], field: value})
