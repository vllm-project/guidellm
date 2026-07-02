"""
Integration tests for per-sub-benchmark constraint support.

Exercises the pipeline from BenchmarkScenario construction (with constraint
overrides) through resolve_to_single_benchmark merging and into
resolve_constraints, verifying that merged BenchmarkArgs and resulting
constraint initializers carry the correct per-benchmark values.
"""

from __future__ import annotations

from typing import Any

import pytest

from guidellm.benchmark.entrypoints import (
    resolve_constraints,
    resolve_to_single_benchmark,
)
from guidellm.benchmark.schemas import BenchmarkArgs
from guidellm.scheduler.constraints import (
    MaxDurationConstraintArgs,
    MaxRequestsConstraintArgs,
    OverSaturationConstraintArgs,
)
from guidellm.scheduler.constraints.error import (
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
)
from guidellm.scheduler.constraints.request import (
    MaxDurationConstraint,
    MaxNumberConstraint,
)
from guidellm.scheduler.schemas import SchedulerState
from guidellm.schemas import RequestInfo
from guidellm.utils.arg_string import ArgStringParser

_PIPELINE_DEFAULTS: dict[str, Any] = {
    "data": [{"kind": "synthetic_text", "prompt_tokens": 10, "output_tokens": 10}],
    "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
    "profile": {"kind": "sweep"},
    "tokenizer": {"kind": "huggingface_auto"},
    "data_column_mapper": {"kind": "generative_column_mapper"},
    "data_preprocessors": [{"kind": "encode_media"}],
    "data_finalizer": {"kind": "generative"},
    "data_loader": {"kind": "pytorch"},
}


def _make_benchmarks(
    constraints: list[dict[str, Any]],
    overrides_per_benchmark: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> list[BenchmarkArgs]:
    """
    Build a list of BenchmarkArgs with per-benchmark constraint overrides.

    Constructs each BenchmarkArgs by applying constraint field overrides on top
    of a shared base spec, mimicking what ``BenchmarkScenario.get_benchmarks()``
    produces when the user passes ``--override "constraints[i].field" v1,v2,...``.

    :param constraints: Base constraint dicts shared by all sub-benchmarks.
    :param overrides_per_benchmark: One dict per sub-benchmark mapping
        ``"constraints[i].field"`` keys to per-benchmark scalar values.
    :param profile: Optional profile dict override (default: ``{"kind": "sweep"}``).
    :return: List of fully-validated ``BenchmarkArgs``.

    ## WRITTEN BY AI ##
    """
    results = []
    base = {
        **_PIPELINE_DEFAULTS,
        "constraints": constraints,
    }
    if profile is not None:
        base["profile"] = profile

    parser = ArgStringParser(allow_overwrite=True)
    for override in overrides_per_benchmark:
        bench_dict: dict[str, Any] = {
            k: v if not isinstance(v, list) else list(v) for k, v in base.items()
        }
        # Deep-copy the constraints list so overrides don't mutate the base
        bench_dict["constraints"] = [dict(c) for c in constraints]
        if profile is not None:
            bench_dict["profile"] = dict(profile)

        for key, value in override.items():
            parser.set(bench_dict, key, value)

        results.append(BenchmarkArgs.model_validate(bench_dict))

    return results


class TestSingleConstraintPerSubBenchmark:
    """Merge of a single constraint kind across multiple sub-benchmarks."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("kind", "field_name", "override_values", "expected_merged"),
        [
            pytest.param(
                "max_duration",
                "seconds",
                [5, 10, 15],
                [5, 10, 15],
                id="max_duration-seconds",
            ),
            pytest.param(
                "max_requests",
                "count",
                [100, 200, 300],
                [100, 200, 300],
                id="max_requests-count",
            ),
            pytest.param(
                "max_errors",
                "count",
                [5, 10, 15],
                [5, 10, 15],
                id="max_errors-count",
            ),
            pytest.param(
                "max_error_rate",
                "rate",
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                id="max_error_rate-rate",
            ),
        ],
    )
    def test_single_constraint_merges_list_field(
        self,
        kind: str,
        field_name: str,
        override_values: list[Any],
        expected_merged: list[Any],
    ):
        """
        Per-sub-benchmark values for a single constraint are merged into one list.

        ## WRITTEN BY AI ##
        """
        first_value = override_values[0]
        base_constraint = {
            "kind": kind,
            field_name: first_value,
        }
        overrides = [{f"constraints[0].{field_name}": val} for val in override_values]

        benchmarks = _make_benchmarks([base_constraint], overrides)
        merged = resolve_to_single_benchmark(benchmarks)

        assert len(merged.constraints) == 1
        merged_val = getattr(merged.constraints[0], field_name)
        assert merged_val == expected_merged


class TestMultipleConstraintsOneVarying:
    """Two constraints present, only one varies across sub-benchmarks."""

    @pytest.mark.sanity
    def test_only_varying_constraint_becomes_list(self):
        """
        When only one of two constraints varies, the other stays scalar.

        ## WRITTEN BY AI ##
        """
        constraints = [
            {"kind": "max_requests", "count": 10},
            {"kind": "max_duration", "seconds": 60},
        ]
        overrides = [
            {"constraints[0].count": 10},
            {"constraints[0].count": 20},
            {"constraints[0].count": 30},
        ]

        benchmarks = _make_benchmarks(constraints, overrides)
        merged = resolve_to_single_benchmark(benchmarks)

        assert len(merged.constraints) == 2
        assert isinstance(merged.constraints[0], MaxRequestsConstraintArgs)
        assert merged.constraints[0].count == [10, 20, 30]
        assert isinstance(merged.constraints[1], MaxDurationConstraintArgs)
        # The merge always collects list-capable fields across sub-benchmarks,
        # so even a non-overridden field becomes a repeated-value list.
        assert merged.constraints[1].seconds == [60, 60, 60]


class TestMultipleConstraintsBothVarying:
    """Two constraints, both with per-sub-benchmark overrides."""

    @pytest.mark.sanity
    def test_both_constraints_merge_independently(self):
        """
        Both constraints merge their respective list-capable fields independently.

        ## WRITTEN BY AI ##
        """
        constraints = [
            {"kind": "max_requests", "count": 10},
            {"kind": "max_duration", "seconds": 30},
        ]
        overrides = [
            {"constraints[0].count": 10, "constraints[1].seconds": 30},
            {"constraints[0].count": 20, "constraints[1].seconds": 60},
        ]

        benchmarks = _make_benchmarks(constraints, overrides)
        merged = resolve_to_single_benchmark(benchmarks)

        assert len(merged.constraints) == 2
        assert merged.constraints[0].count == [10, 20]
        assert merged.constraints[1].seconds == [30, 60]


class TestSingleBenchmarkPassthrough:
    """A single sub-benchmark passes through resolve_to_single_benchmark unchanged."""

    @pytest.mark.smoke
    def test_single_benchmark_returns_identity(self):
        """
        A single-element list returns the original BenchmarkArgs unmodified.

        ## WRITTEN BY AI ##
        """
        benchmarks = _make_benchmarks(
            [{"kind": "max_duration", "seconds": 30}],
            [{"constraints[0].seconds": 30}],
        )
        assert len(benchmarks) == 1
        merged = resolve_to_single_benchmark(benchmarks)

        assert merged is benchmarks[0]
        assert isinstance(merged.constraints[0], MaxDurationConstraintArgs)
        assert merged.constraints[0].seconds == 30


class TestCombinedProfileAndConstraintOverrides:
    """Profile rate and constraint overrides are both merged correctly."""

    @pytest.mark.sanity
    def test_profile_rate_and_constraint_merged_together(self):
        """
        Overriding both profile.rate and constraints[0].seconds merges both.

        ## WRITTEN BY AI ##
        """
        constraints = [{"kind": "max_duration", "seconds": 5}]
        profile = {"kind": "constant", "rate": 1.0}
        overrides = [
            {"profile.rate": 1.0, "constraints[0].seconds": 5},
            {"profile.rate": 2.0, "constraints[0].seconds": 10},
            {"profile.rate": 4.0, "constraints[0].seconds": 15},
        ]

        benchmarks = _make_benchmarks(constraints, overrides, profile=profile)
        merged = resolve_to_single_benchmark(benchmarks)

        assert merged.constraints[0].seconds == [5, 10, 15]
        assert merged.profile.rate == [1.0, 2.0, 4.0]  # type: ignore[union-attr]


class TestNonListConstraintPassthrough:
    """Constraints with no list-capable field pass through unchanged."""

    @pytest.mark.sanity
    def test_over_saturation_preserved_alongside_varying_constraint(self):
        """
        An over_saturation constraint (no list-capable field) is preserved
        unchanged when another constraint varies across sub-benchmarks.

        ## WRITTEN BY AI ##
        """
        constraints = [
            {"kind": "max_requests", "count": 10},
            {"kind": "over_saturation"},
        ]
        overrides = [
            {"constraints[0].count": 10},
            {"constraints[0].count": 20},
        ]

        benchmarks = _make_benchmarks(constraints, overrides)
        merged = resolve_to_single_benchmark(benchmarks)

        assert len(merged.constraints) == 2
        assert merged.constraints[0].count == [10, 20]
        assert isinstance(merged.constraints[1], OverSaturationConstraintArgs)
        # over_saturation uses all defaults -- verify it wasn't mutated
        assert merged.constraints[1].mode == "enforce"
        assert merged.constraints[1].min_seconds == 30.0


class TestNonMergeableFieldRaisesError:
    """Differing non-mergeable constraint fields must raise NotImplementedError."""

    @pytest.mark.smoke
    def test_differing_window_raises(self):
        """
        Two sub-benchmarks with different max_error_rate window values raise.

        ## WRITTEN BY AI ##
        """
        bench_a = BenchmarkArgs.model_validate(
            {
                **_PIPELINE_DEFAULTS,
                "constraints": [{"kind": "max_error_rate", "rate": 0.5, "window": 10}],
            }
        )
        bench_b = BenchmarkArgs.model_validate(
            {
                **_PIPELINE_DEFAULTS,
                "constraints": [{"kind": "max_error_rate", "rate": 0.5, "window": 20}],
            }
        )

        with pytest.raises(NotImplementedError, match="window"):
            resolve_to_single_benchmark([bench_a, bench_b])


class TestCreateConstraintIndexTracking:
    """
    After merging, resolve_constraints produces initializers whose
    create_constraint() calls yield the correct per-benchmark value.
    """

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("kind", "field_name", "values", "initializer_cls", "metadata_key"),
        [
            pytest.param(
                "max_requests",
                "count",
                [10, 20, 30],
                MaxNumberConstraint,
                "max_requests",
                id="max_requests",
            ),
            pytest.param(
                "max_duration",
                "seconds",
                [5, 10, 15],
                MaxDurationConstraint,
                "max_duration",
                id="max_duration",
            ),
            pytest.param(
                "max_errors",
                "count",
                [5, 10, 15],
                MaxErrorsConstraint,
                "max_errors",
                id="max_errors",
            ),
            pytest.param(
                "max_error_rate",
                "rate",
                [0.1, 0.2, 0.3],
                MaxErrorRateConstraint,
                "max_error_rate",
                id="max_error_rate",
            ),
        ],
    )
    def test_create_constraint_returns_correct_per_index_value(
        self,
        kind: str,
        field_name: str,
        values: list[Any],
        initializer_cls: type,
        metadata_key: str,
    ):
        """
        Each call to create_constraint() advances the index so the returned
        constraint evaluates with the correct value from the merged list.

        ## WRITTEN BY AI ##
        """
        first_val = values[0]
        base_constraint = {"kind": kind, field_name: first_val}
        overrides = [{f"constraints[0].{field_name}": v} for v in values]

        benchmarks = _make_benchmarks([base_constraint], overrides)
        merged = resolve_to_single_benchmark(benchmarks)
        resolved = resolve_constraints(merged)

        assert kind in resolved
        initializer = resolved[kind]
        assert isinstance(initializer, initializer_cls)

        state = SchedulerState(
            created_requests=0,
            processed_requests=0,
            errored_requests=0,
        )
        request_info = RequestInfo()

        for idx, expected_val in enumerate(values):
            constraint = initializer.create_constraint()
            action = constraint(state, request_info)
            actual = action.metadata[metadata_key]
            assert actual == expected_val, (
                f"Index {idx}: expected {metadata_key}={expected_val}, got {actual}"
            )
