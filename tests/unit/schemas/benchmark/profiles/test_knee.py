"""Configuration coverage for the knee profile."""

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import BenchmarkScenario, KneeProfileArgs, ProfileArgs


@pytest.mark.smoke
@pytest.mark.parametrize(
    "initial_options",
    [
        {"initial_streams": [1, 5, 10, 20, 40]},
        {"min_streams": 1, "max_streams": 9, "count": 5},
    ],
)
def test_knee_profile_configuration_round_trip(initial_options):
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
                **initial_options,
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
@pytest.mark.parametrize(
    ("initial_options", "expected"),
    [
        ({"initial_streams": [1, 2, 5, 25, 50]}, [1, 2, 5, 25, 50]),
        ({"initial_streams": ["1", "2", "5", "25", "50"]}, [1, 2, 5, 25, 50]),
        ({"min_streams": 1, "max_streams": 9, "count": 5}, [1, 3, 5, 7, 9]),
        ({"min_streams": "1", "max_streams": "9", "count": "5"}, [1, 3, 5, 7, 9]),
        ({"min_streams": 1, "max_streams": 10, "count": 7}, [1, 3, 4, 6, 7, 9, 10]),
    ],
)
def test_knee_profile_defaults_to_analysis_only(initial_options, expected):
    """Resolve initial concurrency points without enabling adaptive execution.

    ## WRITTEN BY AI ##
    """
    args = ProfileArgs.model_validate({"kind": "knee", **initial_options})
    assert isinstance(args, KneeProfileArgs)
    assert args.resolved_initial_streams() == expected
    assert args.adaptive is False
    assert args.points_each_side == 5
    assert args.max_step == 5


@pytest.mark.sanity
@pytest.mark.parametrize(
    "initial_options",
    [
        {},
        {"initial_streams": []},
        {"initial_streams": [1, 2, 5, 25]},
        {"initial_streams": [1, 2, 5, 5, 25]},
        {"initial_streams": [1, 5, 2, 25, 50]},
        {"initial_streams": [0, 1, 2, 5, 25]},
        {"initial_streams": [-1, 1, 2, 5, 25]},
        {"initial_streams": [1, "1", 2, 5, 25]},
        {"initial_streams": [1.5, 1, 2, 5, 25]},
        {"initial_streams": "[1,2,5,25,50]"},
        {"initial_streams": "invalid"},
        {"min_streams": 1, "max_streams": 9},
        {"min_streams": 9, "max_streams": 1, "count": 5},
        {"min_streams": 1, "max_streams": 4, "count": 5},
        {"min_streams": 1, "max_streams": 9, "count": 4},
        {
            "initial_streams": [1, 2, 5, 25, 50],
            "min_streams": 1,
            "max_streams": 9,
            "count": 5,
        },
    ],
)
def test_knee_profile_rejects_invalid_streams(initial_options):
    """Reject ambiguous or invalid concurrency measurements before execution.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        KneeProfileArgs.model_validate(initial_options)


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("points_each_side", "max_step"),
    [("3", "2"), (3.0, 2.0)],
)
def test_knee_profile_coerces_integer_grid_options(points_each_side, max_step):
    """Accept grid values that Pydantic can convert to positive integers.

    ## WRITTEN BY AI ##
    """
    args = KneeProfileArgs.model_validate(
        {
            "initial_streams": [1, 2, 5, 25, 50],
            "points_each_side": points_each_side,
            "max_step": max_step,
        }
    )
    assert args.points_each_side == 3
    assert args.max_step == 2


@pytest.mark.sanity
@pytest.mark.parametrize("field", ["points_each_side", "max_step"])
@pytest.mark.parametrize("value", [0, -1, 1.5, "invalid"])
def test_knee_profile_rejects_invalid_grid_options(field, value):
    """Reject grid values that cannot become positive integers.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        KneeProfileArgs.model_validate(
            {"initial_streams": [1, 2, 5, 25, 50], field: value}
        )
