"""Concurrent benchmark profile that estimates and refines a throughput knee."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from guidellm.benchmark.analysis import (
    AdaptiveConcurrencyPlan,
    KneeAnalysis,
    KneeDetectionConclusion,
    analyze_knee,
    benchmark_concurrency,
    generate_adaptive_concurrency_plan,
)
from guidellm.benchmark.schemas import Benchmark, GenerativeBenchmark
from guidellm.logger import logger
from guidellm.scheduler import (
    ConcurrentStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
)
from guidellm.schemas.benchmark.profiles import KneeProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["KneeProfile"]


@ProfileFactory.register("knee")
class KneeProfile(Profile):
    """Analyze an initial concurrency sweep and optionally refine its knee once."""

    args: KneeProfileArgs

    def __init__(
        self,
        args: KneeProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args
        self._streams = list(args.streams)
        self._benchmarks: list[GenerativeBenchmark] = []
        self._initial: KneeAnalysis | None = None
        self._adaptive_plan: AdaptiveConcurrencyPlan | None = None
        self._conclusion: KneeDetectionConclusion | None = None

    @property
    def strategy_types(self) -> list[str]:
        """
        Reserve progress slots for the initial sweep and possible adaptive points.

        :return: Concurrent strategy types, one per possible benchmark
        """
        extra = 2 * self.args.points_each_side + 1 if self.args.adaptive else 0
        return ["concurrent"] * (len(self.args.streams) + extra)

    @property
    def conclusion(self) -> dict[str, Any] | None:
        """
        :return: Initial analysis, adaptive plan and final analysis after completion
        """
        return self._conclusion.model_dump(mode="json") if self._conclusion else None

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ConcurrentStrategy | None:
        """
        Execute the initial points followed by at most one adaptive grid.

        :param prev_strategy: Previously completed strategy
        :param prev_benchmark: Benchmark results from the previous concurrency
        :return: Next concurrent strategy, or None when complete or stopped
        :raises RuntimeError: If the previous result is not a generative benchmark
        """
        _ = prev_strategy
        if self._conclusion is not None:
            return None

        stopped = False
        if prev_benchmark is not None:
            if not isinstance(prev_benchmark, GenerativeBenchmark):
                raise RuntimeError(
                    "The knee profile requires generative benchmark results, "
                    f"got {type(prev_benchmark).__name__}."
                )
            self._benchmarks.append(prev_benchmark)
            stopped = self._should_stop_escalating(prev_benchmark) or any(
                action.stopping_scope == "all"
                for action in (
                    prev_benchmark.scheduler_state.end_processing_constraints.values()
                )
            )

        if self._initial is None and (
            stopped or len(self._benchmarks) == len(self.args.streams)
        ):
            self._plan_refinement(stopped)

        index = len(self.completed_strategies)
        if stopped or index >= len(self._streams):
            self._finish()
            return None

        return ConcurrentStrategy(
            streams=self._streams[index],
            rampup_duration=self.args.rampup_duration,
        )

    def _plan_refinement(self, stopped: bool) -> None:
        """Analyze the initial sweep and append eligible adaptive points."""
        self._initial = analyze_knee(self._benchmarks)
        if stopped:
            self._adaptive_plan = AdaptiveConcurrencyPlan(
                status="skipped",
                reason=(
                    "An initial benchmark triggered a stopping_scope='all' constraint"
                ),
            )
        elif not self.args.adaptive:
            self._adaptive_plan = AdaptiveConcurrencyPlan(
                status="skipped", reason="Adaptive refinement is disabled"
            )
        else:
            self._adaptive_plan = generate_adaptive_concurrency_plan(
                self._initial,
                [benchmark_concurrency(benchmark) for benchmark in self._benchmarks],
                points_each_side=self.args.points_each_side,
                max_step=self.args.max_step,
            )
        self._streams.extend(self._adaptive_plan.concurrencies)
        logger.info(
            "Knee profile adaptive plan: {} ({})",
            self._adaptive_plan.concurrencies,
            self._adaptive_plan.reason,
        )

    def _finish(self) -> None:
        """Include the last measurement in the final profile conclusion."""
        if self._initial is None or self._adaptive_plan is None:
            raise RuntimeError("The knee profile finished before analyzing its sweep")
        self._conclusion = KneeDetectionConclusion(
            initial=self._initial,
            adaptive_plan=self._adaptive_plan,
            final=analyze_knee(self._benchmarks),
        )
        result = self._conclusion.final.throughput
        logger.info(
            "Knee profile complete after {} benchmarks: knee={} ({})",
            len(self._benchmarks),
            result.knee,
            result.reason,
        )
