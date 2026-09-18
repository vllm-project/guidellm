from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["GoodputProfileArgs"]


@ProfileArgs.register("goodput")
class GoodputProfileArgs(ProfileArgs):
    """Pydantic model for goodput search profile creation arguments."""

    kind: Literal["goodput"] = Field(
        default="goodput",
        description="Profile type discriminator for goodput search scheduling",
    )
    target_attainment: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of requests that must meet every configured latency "
            "objective for a concurrency level to pass. The default of 0.95 is "
            "equivalent to requiring the p95 of each objective's metric to sit "
            "within its threshold. Must be below 1.0: a finite sample cannot "
            "establish that no request in the population violates an objective"
        ),
        examples=[0.95, 0.99],
    )
    initial_streams: PositiveInt = Field(
        default=4,
        description=(
            "Concurrency level probed first. The search doubles from here until "
            "a level fails, then bisects between the last pass and first failure"
        ),
    )
    max_streams: PositiveInt = Field(
        default=1024,
        description=(
            "Upper limit on concurrency the search will probe. Reaching it "
            "without a failure ends the search and reports the objectives as "
            "met at every level tested"
        ),
    )
    tolerance: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Relative width of the bracket at which the search stops, as a "
            "fraction of the highest passing concurrency. Bisecting to an exact "
            "integer costs a probe per halving regardless of scale, so the "
            "default stops once the answer is known to within 10 percent"
        ),
        examples=[0.1, 0.05],
    )
    max_probes: PositiveInt = Field(
        default=15,
        description=(
            "Maximum number of benchmark runs the search may execute before "
            "reporting its best result so far. Doubling from initial_streams to "
            "max_streams costs log2(max_streams / initial_streams) probes and "
            "the bisection that follows costs about log2(1 / tolerance) more"
        ),
    )
    confidence: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description=(
            "Confidence level for the Wilson score interval reported around "
            "each probe's attainment, used to flag results the run was too "
            "short to resolve"
        ),
    )

    def validate_metrics(self, metrics: Any) -> None:
        """
        Require latency objectives for the search to search against.

        Without this the run fails only once the first probe has finished and
        its attainment turns out to be unmeasurable, wasting a full probe
        duration on a configuration error.

        :param metrics: Validated metrics arguments for the run
        :raises ValueError: If no latency objectives are configured
        """
        # Read through model_dump rather than importing GenerativeMetricsArgs,
        # which would import this module back through the args package.
        if metrics.model_dump().get("slo") is None:
            raise ValueError(
                "The goodput profile searches for the highest load meeting "
                "latency objectives, so it requires objectives to be set, for "
                'example --metrics \'{"kind":"generative","slo":'
                '{"ttft_ms":2000}}\''
            )

    @model_validator(mode="after")
    def _check_stream_bounds(self) -> GoodputProfileArgs:
        """
        Validate that the search range is non-empty.

        :return: The validated instance
        :raises ValueError: If initial_streams exceeds max_streams
        """
        if self.initial_streams > self.max_streams:
            raise ValueError(
                f"initial_streams ({self.initial_streams}) must not exceed "
                f"max_streams ({self.max_streams})"
            )

        return self
