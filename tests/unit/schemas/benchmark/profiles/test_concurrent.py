"""Configuration coverage for the concurrent profile."""

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import ConcurrentProfileArgs, ProfileArgs


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("streams_value", "expected"),
    [
        (8, [8]),
        (8.0, [8]),
        ([8, 16], [8, 16]),
        ((8, 16), [8, 16]),
        ("[8,16]", [8, 16]),
        ("8", [8]),
    ],
)
def test_concurrent_profile_coerces_streams_to_list(streams_value, expected):
    """Accept a single stream count or a list of counts for the field.

    ## WRITTEN BY AI ##
    """
    args = ProfileArgs.model_validate({"kind": "concurrent", "streams": streams_value})
    assert isinstance(args, ConcurrentProfileArgs)
    assert args.streams == expected


@pytest.mark.sanity
@pytest.mark.parametrize(
    "profile_args",
    [
        {"kind": "concurrent"},
        {"kind": "concurrent", "streams": []},
        {"kind": "concurrent", "streams": "invalid"},
        {"kind": "concurrent", "streams": None},
        {"kind": "concurrent", "streams": [None]},
        {"kind": "concurrent", "streams": [0]},
        {"kind": "concurrent", "streams": [-1]},
        {"kind": "concurrent", "streams": 8.9},
        {"kind": "concurrent", "streams": [8.9]},
        {"kind": "concurrent", "streams": float("inf")},
        {"kind": "concurrent", "streams": float("nan")},
        {"kind": "concurrent", "streams": True},
        {"kind": "concurrent", "streams": [True]},
        {"kind": "concurrent", "streams": 8, "rampup_duration": -1.0},
    ],
)
def test_concurrent_profile_rejects_invalid_arguments(profile_args):
    """Reject missing, empty, or non-numeric stream counts.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        ConcurrentProfileArgs.model_validate(profile_args)
