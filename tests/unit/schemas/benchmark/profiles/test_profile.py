"""Unit tests for the profile argument base class."""

from __future__ import annotations

from typing import Any, Literal

import pytest

from guidellm.schemas.benchmark import (
    BenchmarkArgs,
    BenchmarkScenario,
    GenerativeMetricsArgs,
)
from guidellm.schemas.benchmark.profiles import ProfileArgs

VALID_BASE = {
    "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
    "data": [{"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}],
}

# Every shipped profile that does not override validate_metrics, with the
# minimum payload each needs to validate.
DEFAULT_HOOK_PROFILES = (
    {"kind": "async", "rate": 10.0},
    {"kind": "concurrent", "streams": 4},
    {"kind": "constant", "rate": 10.0},
    {"kind": "poisson", "rate": 10.0},
    {"kind": "replay"},
    {"kind": "sweep"},
    {"kind": "synchronous"},
    {"kind": "throughput", "max_concurrency": 64},
)


@pytest.fixture
def probe_profile():
    """
    Register a throwaway profile whose hook records and optionally rejects.

    Rebuilds the argument schemas so the new kind joins the discriminated
    union, and removes it again afterwards so the global registry is unchanged.

    ## WRITTEN BY AI ##
    """
    seen: list[Any] = []
    reject: list[bool] = [False]

    @ProfileArgs.register("validate_metrics_probe")
    class _ProbeProfileArgs(ProfileArgs):
        """Profile args used only to observe the hook."""

        kind: Literal["validate_metrics_probe"] = "validate_metrics_probe"

        def validate_metrics(self, metrics: Any) -> None:
            """
            Record the metrics received and reject when asked to.

            :param metrics: Validated metrics arguments for the run
            :raises ValueError: When the fixture is set to reject
            """
            seen.append(metrics)
            if reject[0]:
                raise ValueError("probe profile rejects this metrics configuration")

    ProfileArgs.reload_schema()
    BenchmarkArgs.reload_schema()
    BenchmarkScenario.reload_schema()
    try:
        yield seen, reject
    finally:
        assert ProfileArgs.registry is not None
        ProfileArgs.registry.pop("validate_metrics_probe", None)
        ProfileArgs.reload_schema()
        BenchmarkArgs.reload_schema()
        BenchmarkScenario.reload_schema()


class TestValidateMetricsHook:
    """
    Verify the generic hook profiles use to reject a metrics configuration.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_base_accepts_any_configuration(self):
        """
        Accept any metrics configuration by default.

        ## WRITTEN BY AI ##
        """
        args = ProfileArgs.model_validate({"kind": "synchronous"})

        assert args.validate_metrics(GenerativeMetricsArgs()) is None

    @pytest.mark.sanity
    @pytest.mark.parametrize("payload", DEFAULT_HOOK_PROFILES)
    def test_shipped_profiles_accept_the_default_metrics(self, payload):
        """
        Leave every profile that does not override the hook unaffected.

        ## WRITTEN BY AI ##
        """
        args = ProfileArgs.model_validate(payload)

        assert args.validate_metrics(GenerativeMetricsArgs()) is None

    @pytest.mark.sanity
    @pytest.mark.parametrize("payload", DEFAULT_HOOK_PROFILES)
    def test_shipped_profiles_still_build_a_full_configuration(self, payload):
        """
        Build a whole configuration for each profile that does not override
        the hook, so dispatching through it cannot break them.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate({**VALID_BASE, "profile": payload})

        assert args.profile.kind == payload["kind"]

    @pytest.mark.regression
    def test_benchmark_args_surfaces_a_rejection(self, probe_profile):
        """
        Surface a profile's rejection when the whole configuration validates.

        Uses a profile registered only for this test, so the dispatch is proven
        without depending on any particular shipped profile.

        ## WRITTEN BY AI ##
        """
        _, reject = probe_profile
        reject[0] = True

        with pytest.raises(ValueError, match="probe profile rejects"):
            BenchmarkArgs.model_validate(
                {**VALID_BASE, "profile": {"kind": "validate_metrics_probe"}}
            )

    @pytest.mark.regression
    def test_hook_receives_the_configured_metrics(self, probe_profile):
        """
        Pass the run's own metrics arguments to the hook, not a fresh default.

        ## WRITTEN BY AI ##
        """
        seen, _ = probe_profile

        BenchmarkArgs.model_validate(
            {
                **VALID_BASE,
                "profile": {"kind": "validate_metrics_probe"},
                "metrics": {"kind": "generative", "sample_size": 7},
            }
        )

        assert [metrics.sample_size for metrics in seen] == [7]

    @pytest.mark.regression
    def test_accepting_hook_leaves_configuration_valid(self, probe_profile):
        """
        Build the configuration normally when the hook raises nothing.

        ## WRITTEN BY AI ##
        """
        seen, _ = probe_profile

        args = BenchmarkArgs.model_validate(
            {**VALID_BASE, "profile": {"kind": "validate_metrics_probe"}}
        )

        assert args.profile.kind == "validate_metrics_probe"
        assert len(seen) == 1
