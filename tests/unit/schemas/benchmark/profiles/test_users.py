"""Configuration coverage for the users profile."""

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import (
    BenchmarkScenario,
    ProfileArgs,
    UsersProfileArgs,
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "profile_args",
    [
        {"kind": "users", "users": 8},
        {
            "kind": "users",
            "users": [8, 16],
            "rampup_duration": 1.0,
            "warmup": 2.0,
        },
    ],
)
def test_users_profile_configuration_round_trip(profile_args):
    """Keep users options inside the registered profile configuration.

    ## WRITTEN BY AI ##
    """
    scenario = BenchmarkScenario.create(
        scenario=None,
        spec={
            "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
            "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
            "profile": profile_args,
        },
    )
    restored = BenchmarkScenario.model_validate_json(scenario.model_dump_json())
    assert isinstance(restored.spec.profile, UsersProfileArgs)
    assert restored.spec.profile == scenario.spec.profile


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("users_value", "expected"),
    [
        (8, [8]),
        (8.0, [8]),
        ([8, 16], [8, 16]),
        ((8, 16), [8, 16]),
        ("[8,16]", [8, 16]),
        ("8", [8]),
    ],
)
def test_users_profile_coerces_users_to_list(users_value, expected):
    """Accept a single count or a list of counts for the users field.

    ## WRITTEN BY AI ##
    """
    args = ProfileArgs.model_validate({"kind": "users", "users": users_value})
    assert isinstance(args, UsersProfileArgs)
    assert args.users == expected


@pytest.mark.sanity
@pytest.mark.parametrize(
    "profile_args",
    [
        {"kind": "users"},
        {"kind": "users", "users": []},
        {"kind": "users", "users": "invalid"},
        {"kind": "users", "users": [0]},
        {"kind": "users", "users": [-1]},
        {"kind": "users", "users": 8, "rampup_duration": -1.0},
    ],
)
def test_users_profile_rejects_invalid_arguments(profile_args):
    """Reject missing or contradictory users profile arguments.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        UsersProfileArgs.model_validate(profile_args)


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("profile_args", "expected_users"),
    [
        ({"users": 8}, [8]),
        ({"users": "[8,16]"}, [8, 16]),
    ],
)
def test_users_profile_accepts_string_values(profile_args, expected_users):
    """Coerce string values that JSON or numeric parsing can handle.

    ## WRITTEN BY AI ##
    """
    args = UsersProfileArgs.model_validate({"kind": "users", **profile_args})
    assert args.users == expected_users
