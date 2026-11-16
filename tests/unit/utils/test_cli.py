"""
Tests for CLI override parsing utilities.
"""

import pytest

from guidellm.schemas.benchmark import BenchmarkScenario
from guidellm.utils.cli import overrides_to_benchmarks, parse_overrides
from guidellm.utils.typing import BLANK


class TestOverridesToBenchmarks:
    """Test cases for overrides_to_benchmarks and parse_overrides."""

    @pytest.mark.regression
    def test_blank_reuses_last_value(self):
        """
        Test that a blank entry reuses the last value given, not the first.
        """
        benchmarks = parse_overrides(None, None, [("profile.rate", "1,2,,4")])

        assert benchmarks == [
            {"profile.rate": 1},
            {"profile.rate": 2},
            {"profile.rate": 2},
            {"profile.rate": 4},
        ]

    @pytest.mark.regression
    def test_shorter_list_reuses_final_value(self):
        """
        Test that a list shorter than another reuses its final value.
        """
        benchmarks = overrides_to_benchmarks(
            ("profile.streams", [1, 2, 4, 8]),
            ("constraints[0].seconds", [10, 20]),
        )

        assert [b["constraints[0].seconds"] for b in benchmarks] == [10, 20, 20, 20]

    @pytest.mark.sanity
    def test_leading_blank_is_omitted(self):
        """
        Test that positions before an override's first value are left unset.
        """
        benchmarks = overrides_to_benchmarks(("profile.rate", [BLANK, 2, 3]))

        assert benchmarks == [{}, {"profile.rate": 2}, {"profile.rate": 3}]

    @pytest.mark.regression
    def test_shorter_list_reaches_benchmarks(self):
        """
        Test that the reused value reaches each benchmark built from a scenario.
        """
        scenario = BenchmarkScenario.model_validate(
            {
                "spec": {
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000/v1",
                    },
                    "data": [{"kind": "synthetic_text", "prompt_tokens": 32}],
                    "profile": {"kind": "concurrent", "streams": 1},
                    "constraints": [{"kind": "max_duration", "seconds": 5}],
                },
                "benchmarks": parse_overrides(
                    None,
                    None,
                    [
                        ("profile.streams", "1,2,4,8"),
                        ("constraints[0].seconds", "10,20"),
                    ],
                ),
            }
        )

        seconds = [b.constraints[0].seconds for b in scenario.get_benchmarks()]

        assert seconds == [10, 20, 20, 20]
