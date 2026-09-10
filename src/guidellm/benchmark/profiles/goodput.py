"""Goodput search profile for locating peak load under latency objectives."""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.logger import logger
from guidellm.scheduler import (
    ConcurrentStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
)
from guidellm.scheduler.constraints.saturation import approx_t_ppf
from guidellm.schemas.base import StandardBaseModel
from guidellm.schemas.benchmark.profiles import GoodputProfileArgs

from .profile import Profile, ProfileFactory

__all__ = [
    "GoodputProbe",
    "GoodputProfile",
    "GoodputSearchState",
    "wilson_interval",
]

_NORMAL_LIMIT_DF = 1.0e9
"""Degrees of freedom at which approx_t_ppf matches the standard normal."""

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute a Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because attainment is measured near
    1.0, where the normal interval extends above 1.0 and understates uncertainty
    for the small request counts a short probe produces.

    :param successes: Number of conforming requests, clamped to [0, trials]
    :param trials: Number of requests with a determined verdict
    :param confidence: Two-sided confidence level, clamped to [0.5, 0.999]
    :return: Tuple of (lower bound, upper bound), both within [0.0, 1.0]
    """
    if trials <= 0:
        return 0.0, 1.0

    # This is public API, so keep a caller that passes an out-of-range count or
    # confidence from reaching a domain error inside the formula below.
    successes = min(max(successes, 0), trials)
    confidence = min(max(confidence, 0.5), 0.999)

    # The Wilson interval is defined against the standard normal quantile. The
    # t-distribution approximation already used for slope detection converges to
    # it, so a large degrees-of-freedom value reuses that helper rather than
    # duplicating a second quantile routine; the residual error is under 1e-3.
    z = approx_t_ppf((1.0 + confidence) / 2.0, _NORMAL_LIMIT_DF)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )

    return max(0.0, center - spread), min(1.0, center + spread)


class GoodputProbe(StandardBaseModel):
    """Outcome of one concurrency level tested during the search."""

    streams: int = Field(description="Concurrency level probed")
    attainment: float = Field(
        description="Fraction of evaluable requests meeting every objective"
    )
    attainment_lower: float = Field(
        description="Lower bound of the confidence interval on attainment"
    )
    attainment_upper: float = Field(
        description="Upper bound of the confidence interval on attainment"
    )
    determined_requests: int = Field(
        description="Requests the attainment fraction was computed over"
    )
    goodput: float | None = Field(
        description="Objective-conforming requests per second at this level"
    )
    passed: bool = Field(description="Whether attainment met the configured target")
    resolved: bool = Field(
        description=(
            "Whether the confidence interval sits wholly on one side of the "
            "target, meaning the probe collected enough requests to decide"
        )
    )
    aborted: bool = Field(
        default=False,
        description=(
            "Whether a constraint stopped the run during this probe, in which "
            "case it is recorded but not used as a search bound"
        ),
    )


class GoodputSearchState(StandardBaseModel):
    """Search progress, recorded on the profile for reporting and testing."""

    probes: list[GoodputProbe] = Field(
        default_factory=list, description="Probes executed so far, in order"
    )
    best_passing_streams: int | None = Field(
        default=None, description="Highest concurrency that met the target"
    )
    lowest_failing_streams: int | None = Field(
        default=None, description="Lowest concurrency that missed the target"
    )
    stop_reason: str | None = Field(
        default=None, description="Why the search stopped issuing probes"
    )


@ProfileFactory.register("goodput")
class GoodputProfile(Profile):
    """
    Locate the highest concurrency meeting configured latency objectives.

    Doubles concurrency until a level fails its objectives, then bisects between
    the highest passing and lowest failing level. Concurrency is the control
    variable rather than request rate because every concurrency level has a
    well-defined steady state, whereas a rate above the server's capacity
    produces a growing backlog whose measurements describe the backlog rather
    than the server.

    Each probe's pass or fail decision uses SLO attainment, the fraction of
    requests meeting every objective. Attainment is a ratio over the measured
    population, so unlike a rate it is unaffected by how much of the measurement
    window the server spent filling its pipeline.
    """

    args: GoodputProfileArgs

    def __init__(
        self,
        args: GoodputProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args
        self.search = GoodputSearchState()
        self._next_streams: int | None = args.initial_streams

    @property
    def strategy_types(self) -> list[str]:
        """
        Declare the probe budget rather than the probes run so far.

        The progress display sizes its task list from this before the first
        strategy is generated, so reporting completed probes would leave it
        empty and render every run as complete.

        :return: Concurrent strategy types, one per probe the search may run
        """
        return ["concurrent"] * self.args.max_probes

    @property
    def result(self) -> dict[str, Any] | None:
        """
        :return: The recorded search trace, bounds and stop reason
        """
        return self.search.model_dump()

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ConcurrentStrategy | None:
        """
        Generate the next concurrency level to probe.

        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from the previous probe
        :return: ConcurrentStrategy for the next level, or None when the search
            has converged, exhausted its probe budget, or hit its stream ceiling
        """
        if prev_strategy is not None and prev_benchmark is not None:
            aborted = self._should_stop_escalating(prev_benchmark)
            self._record_probe(prev_strategy, prev_benchmark, aborted=aborted)
            if aborted:
                self._next_streams = None
                self.search.stop_reason = "constraint_stopped_escalation"
            else:
                self._advance()

        if self._next_streams is None:
            self._log_result()
            return None

        if len(self.completed_strategies) >= self.args.max_probes:
            self.search.stop_reason = "max_probes_exhausted"
            self._log_result()
            return None

        return ConcurrentStrategy(
            streams=self._next_streams,
            rampup_duration=self.args.rampup_duration,
        )

    def _record_probe(
        self,
        prev_strategy: SchedulingStrategy,
        prev_benchmark: Benchmark,
        aborted: bool = False,
    ) -> None:
        """
        Score the completed probe against the target attainment.

        An aborted probe is recorded but never becomes a bound. A constraint
        that stops the run mid-probe, such as enforced over-saturation, cancels
        active requests; those are excluded from attainment, so the completed
        remainder can look conforming and would otherwise report an unsafe
        concurrency as the highest passing level.

        :param prev_strategy: Strategy that produced the benchmark
        :param prev_benchmark: Benchmark results to score
        :param aborted: Whether a constraint stopped the run during this probe
        :raises RuntimeError: If no latency objectives were configured
        """
        if not isinstance(prev_strategy, ConcurrentStrategy):
            raise RuntimeError(
                "The goodput profile only issues concurrent strategies but was "
                f"given a {type(prev_strategy).__name__} to score."
            )
        if not isinstance(prev_benchmark, GenerativeBenchmark):
            raise RuntimeError(
                "The goodput profile requires generative benchmark results to "
                f"read latency objectives from, got {type(prev_benchmark).__name__}."
            )

        streams = prev_strategy.streams
        attainment = prev_benchmark.metrics.slo_attainment
        determined = prev_benchmark.metrics.slo_determined_requests

        if attainment is None:
            raise RuntimeError(
                "The goodput profile requires latency objectives that can be "
                "measured on this workload. No request produced a determined "
                "verdict; check that --metrics defines an slo and that the "
                "objectives it names are measurable, for example that ttft_ms "
                "and tpot_ms are only used with a streaming backend."
            )

        lower, upper = wilson_interval(
            successes=round(attainment * determined),
            trials=determined,
            confidence=self.args.confidence,
        )
        target = self.args.target_attainment
        # The probe resolves the question only when the whole interval sits on
        # one side of the target. Otherwise the run was too short to tell.
        resolved = lower >= target or upper < target
        passed = attainment >= target

        probe = GoodputProbe(
            streams=streams,
            attainment=attainment,
            attainment_lower=lower,
            attainment_upper=upper,
            determined_requests=determined,
            goodput=(
                prev_benchmark.metrics.request_goodput.successful.mean
                if prev_benchmark.metrics.request_goodput is not None
                else None
            ),
            passed=passed,
            resolved=resolved,
            aborted=aborted,
        )
        self.search.probes.append(probe)

        if aborted:
            return

        if not resolved:
            logger.warning(
                "Goodput probe at concurrency {} is unresolved: attainment "
                "{:.3f} with {:.0f}% interval [{:.3f}, {:.3f}] straddles the "
                "target {:.3f}. Increase the per-probe duration or request "
                "count to resolve it.",
                streams,
                attainment,
                self.args.confidence * 100,
                lower,
                upper,
                target,
            )

        if passed:
            if (
                self.search.best_passing_streams is None
                or streams > self.search.best_passing_streams
            ):
                self.search.best_passing_streams = streams
        elif (
            self.search.lowest_failing_streams is None
            or streams < self.search.lowest_failing_streams
        ):
            self.search.lowest_failing_streams = streams

    def _log_result(self) -> None:
        """
        Report the search outcome once no further probe will run.

        The per-benchmark config captures profile state before each run, so the
        final probe and the stop reason never reach the serialized report. This
        is where a user learns whether the answer is the objective's knee or
        merely as far as the search got.

        A converged, fully resolved search logs at info level. Any other outcome
        logs at warning, because the console log level defaults to warning and a
        result that is only a lower bound is worse to miss than to over-report.
        """
        best = self.search.best_passing_streams
        reason = self.search.stop_reason
        unresolved = [
            probe.streams for probe in self.search.probes if not probe.resolved
        ]

        if best is None:
            if reason == "indeterminate_at_minimum":
                logger.warning(
                    "Goodput search found no concurrency meeting attainment "
                    ">= {:.3f}, but probes at {} streams collected too few "
                    "requests to separate their attainment from the target. "
                    "Raise the per-probe duration or request count before "
                    "concluding the objectives cannot be met.",
                    self.args.target_attainment,
                    unresolved,
                )
                return

            logger.warning(
                "Goodput search found no concurrency meeting attainment >= "
                "{:.3f}; the objectives are not met even at a single stream "
                "({}).",
                self.args.target_attainment,
                reason,
            )
            return

        if reason == "converged" and not unresolved:
            logger.info(
                "Goodput search converged after {} probes: the highest "
                "concurrency meeting attainment >= {:.3f} is {} streams.",
                len(self.search.probes),
                self.args.target_attainment,
                best,
            )
            return

        caveats = []
        if reason != "converged":
            caveats.append(
                f"the search stopped early ({reason}), so {best} is a lower "
                "bound rather than the highest passing level"
            )
        if unresolved:
            caveats.append(
                f"probes at {unresolved} streams collected too few requests to "
                "separate their attainment from the target; raise the per-probe "
                "duration or request count"
            )
        logger.warning(
            "Goodput search finished after {} probes with attainment >= {:.3f} "
            "at {} streams, but {}.",
            len(self.search.probes),
            self.args.target_attainment,
            best,
            " and ".join(caveats),
        )

    def _advance(self) -> None:
        """Choose the next concurrency level, or stop when the search is done."""
        best = self.search.best_passing_streams
        worst = self.search.lowest_failing_streams

        if worst is None:
            # No failure seen yet: double until one appears or the ceiling is hit.
            current = best if best is not None else self.args.initial_streams
            if current >= self.args.max_streams:
                self._next_streams = None
                self.search.stop_reason = "max_streams_reached"
                return
            self._next_streams = min(current * 2, self.args.max_streams)
            return

        if best is None:
            # The lowest level tested already failed; walk down toward 1.
            if worst <= 1:
                self._next_streams = None
                # Each failure was decided on a point estimate. If any of those
                # intervals straddled the target, the descent is not evidence
                # that no concurrency meets the objectives, only that the
                # probes were too short to tell them apart.
                self.search.stop_reason = (
                    "objectives_unmet_at_minimum"
                    if all(probe.resolved for probe in self.search.probes)
                    else "indeterminate_at_minimum"
                )
                return
            self._next_streams = worst // 2
            return

        if worst - best <= max(1, self.args.tolerance * best):
            self._next_streams = None
            self.search.stop_reason = "converged"
            return

        # worst - best >= 2 here, so the midpoint is strictly inside the bracket.
        self._next_streams = (best + worst) // 2
