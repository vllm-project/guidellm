"""
Benchmark progress tracking and console display abstractions.

Provides progress tracking interfaces and implementations for monitoring benchmark
execution, displaying real-time statistics, and managing UI updates during
generative benchmarking operations.

Classes:
    BenchmarkerProgress: Abstract base for benchmark progress tracking.
    BenchmarkerProgressGroup: Composite progress handler for multiple instances.
    GenerativeConsoleBenchmarkerProgress: Console-based progress display.

Type Variables:
    BenchmarkT: Generic benchmark object type.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Literal

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from guidellm.benchmark.aggregator import AggregatorState
from guidellm.benchmark.objects import BenchmarkT, GenerativeBenchmark
from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    SchedulerState,
    SchedulingStrategy,
    StrategyType,
)
from guidellm.utils import Colors, format_value_display

__all__ = [
    "BenchmarkerProgress",
    "BenchmarkerProgressGroup",
    "GenerativeConsoleBenchmarkerProgress",
]


class BenchmarkerProgress(Generic[BenchmarkT], ABC):
    """
    Abstract base class for tracking and displaying benchmark progress.

    Provides lifecycle hooks for monitoring benchmark execution stages including
    initialization, start, updates, completion, and finalization. Supports
    enable/disable functionality for conditional progress tracking.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize progress tracker.

        :param enabled: Whether to enable progress tracking and display.
        """
        self._enabled = enabled
        self.profile: Profile = None
        self.current_strategy: SchedulingStrategy = None

    @property
    def enabled(self) -> bool:
        """
        :return: Whether progress tracking is currently enabled.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """
        :param value: True to enable progress tracking, False to disable.
        :raises RuntimeError: If called after progress run has started.
        """
        if self.profile is not None:
            raise RuntimeError(
                "Cannot change enabled state after __call__ for progress run"
            )

        self._enabled = value

    def __call__(
        self,
        profile: Profile,
        agen: AsyncIterable[
            tuple[
                AggregatorState | None,
                BenchmarkT | None,
                SchedulingStrategy,
                SchedulerState | None,
            ]
        ],
    ) -> AsyncIterator[
        tuple[
            AggregatorState | None,
            BenchmarkT | None,
            SchedulingStrategy,
            SchedulerState | None,
        ]
    ]:
        """
        Track progress through benchmark execution pipeline.

        Wraps the provided async generator to monitor benchmark progress,
        calling appropriate lifecycle hooks based on execution state.

        :param profile: Benchmark profile configuration.
        :param agen: Async generator yielding benchmark execution updates.
        :return: Async iterator forwarding original updates with progress tracking.
        """

        async def aiterator() -> AsyncIterator[
            tuple[
                AggregatorState | None,
                BenchmarkT | None,
                SchedulingStrategy,
                SchedulerState | None,
            ]
        ]:
            self.profile = profile
            if self.enabled:
                await self.on_initialize(profile)

            async for aggregator_update, benchmark, strategy, scheduler_state in agen:
                if self.enabled:
                    await self.on_raw_update(
                        profile,
                        aggregator_update,
                        benchmark,
                        strategy,
                        scheduler_state,
                    )

                    if self.current_strategy != strategy:
                        self.current_strategy = strategy
                        await self.on_benchmark_start(strategy)
                    elif benchmark is not None:
                        await self.on_benchmark_complete(benchmark)
                        self.current_strategy = None
                    else:
                        await self.on_benchmark_update(
                            aggregator_update, scheduler_state
                        )

                yield aggregator_update, benchmark, strategy, scheduler_state

            if self.enabled:
                await self.on_finalize()

        return aiterator()

    @abstractmethod
    async def on_initialize(self, profile: Profile):
        """
        Initialize progress tracking for benchmark profile.

        :param profile: Benchmark profile configuration.
        """

    @abstractmethod
    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        """
        Handle start of new benchmark strategy execution.

        :param strategy: Scheduling strategy being executed.
        """

    @abstractmethod
    async def on_benchmark_update(
        self, aggregator_update: AggregatorState, scheduler_state: SchedulerState
    ):
        """
        Handle benchmark execution progress update.

        :param aggregator_update: Current benchmark metrics and statistics.
        :param scheduler_state: Current scheduler execution state.
        """

    @abstractmethod
    async def on_benchmark_complete(self, benchmark: BenchmarkT):
        """
        Handle completion of benchmark strategy execution.

        :param benchmark: Completed benchmark results.
        """

    @abstractmethod
    async def on_finalize(self):
        """Finalize progress tracking and cleanup resources."""

    async def on_raw_update(
        self,
        profile: Profile,
        aggregator_update: AggregatorState | None,
        benchmark: BenchmarkT | None,
        strategy: SchedulingStrategy,
        scheduler_state: SchedulerState | None,
    ):
        """
        Handle raw benchmark execution update.

        Optional hook for accessing all execution state updates. Default
        implementation does nothing.

        :param profile: Benchmark profile configuration.
        :param aggregator_update: Current benchmark metrics and statistics.
        :param benchmark: Completed benchmark if available.
        :param strategy: Current scheduling strategy.
        :param scheduler_state: Current scheduler execution state.
        """


class BenchmarkerProgressGroup(BenchmarkerProgress[BenchmarkT]):
    """
    Composite progress handler that manages multiple progress instances.

    Distributes progress events to all contained progress instances, enabling
    parallel progress tracking through multiple channels (e.g., console display
    and file logging).

    :param instances: Collection of progress handlers to manage.
    :param enabled: Whether the group is active.
    """

    def __init__(
        self,
        instances: (
            Iterable[BenchmarkerProgress[BenchmarkT]]
            | list[BenchmarkerProgress[BenchmarkT]]
        ),
        enabled: bool = True,
    ):
        """
        Initialize progress group with handler instances.

        :param instances: Progress handler instances to coordinate.
        :param enabled: Whether to enable the progress group.
        """
        self.instances: list[BenchmarkerProgress[BenchmarkT]] = list(instances)
        super().__init__(enabled=enabled)

    @property
    def enabled(self) -> bool:
        """Whether the progress group is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        Set enabled state for group and all contained instances.

        :param value: New enabled state.
        """
        self._enabled = value
        for instance in self.instances:
            instance.enabled = value

    async def on_initialize(self, profile: Profile):
        """
        Initialize all progress handler instances.

        :param profile: Benchmark profile configuration.
        """
        await asyncio.gather(
            *[child.on_initialize(profile) for child in self.instances]
        )

    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        """
        Notify all handlers of benchmark strategy start.

        :param strategy: Scheduling strategy being executed.
        """
        await asyncio.gather(
            *[child.on_benchmark_start(strategy) for child in self.instances]
        )

    async def on_benchmark_update(
        self, aggregator_update: AggregatorState, scheduler_state: SchedulerState
    ):
        """
        Distribute benchmark updates to all handlers.

        :param aggregator_update: Current benchmark metrics and statistics.
        :param scheduler_state: Current scheduler execution state.
        """
        await asyncio.gather(
            *[
                child.on_benchmark_update(aggregator_update, scheduler_state)
                for child in self.instances
            ]
        )

    async def on_benchmark_complete(self, benchmark: BenchmarkT):
        """
        Notify all handlers of benchmark completion.

        :param benchmark: Completed benchmark results.
        """
        await asyncio.gather(
            *[child.on_benchmark_complete(benchmark) for child in self.instances]
        )

    async def on_finalize(self):
        """Finalize all progress handler instances."""
        await asyncio.gather(*[child.on_finalize() for child in self.instances])

    async def on_raw_update(
        self,
        profile: Profile,
        aggregator_update: AggregatorState | None,
        benchmark: BenchmarkT | None,
        strategy: SchedulingStrategy,
        scheduler_state: SchedulerState | None,
    ):
        """
        Distribute raw updates to all handlers.

        :param profile: Benchmark profile configuration.
        :param aggregator_update: Current benchmark metrics and statistics.
        :param benchmark: Completed benchmark if available.
        :param strategy: Current scheduling strategy.
        :param scheduler_state: Current scheduler execution state.
        """
        await asyncio.gather(
            *[
                child.on_raw_update(
                    profile,
                    aggregator_update,
                    benchmark,
                    strategy,
                    scheduler_state,
                )
                for child in self.instances
            ]
        )


class GenerativeConsoleBenchmarkerProgress(
    BenchmarkerProgress[GenerativeBenchmark], Live
):
    """
    Console-based progress display for generative benchmarks.

    Provides real-time visual progress tracking using Rich library components,
    displaying benchmark execution statistics, timing information, and progress
    bars in a structured console interface.
    """

    def __init__(self, enabled: bool = True, display_scheduler_stats: bool = False):
        """
        Initialize console progress display.

        :param enabled: Whether to enable progress tracking and display.
        :param display_scheduler_stats: Whether to display scheduler statistics.
        """
        BenchmarkerProgress.__init__(self, enabled=enabled)
        Live.__init__(
            self,
            refresh_per_second=4,
            auto_refresh=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self.display_scheduler_stats: bool = display_scheduler_stats
        self.run_progress: Progress = None
        self.run_progress_task: TaskID = None
        self.tasks_progress: _GenerativeProgressTasks = None

    async def on_initialize(self, profile: Profile):
        """
        Initialize console display components and start rendering.

        :param profile: Benchmark profile configuration.
        """
        self.tasks_progress = _GenerativeProgressTasks(
            profile=profile, display_scheduler_stats=self.display_scheduler_stats
        )
        self.run_progress = Progress(
            TextColumn("Generating...", style=f"italic {Colors.progress}"),
            BarColumn(
                bar_width=None,
                complete_style=Colors.progress,
                finished_style=Colors.success,
            ),
            TextColumn(
                "({task.fields[completed_benchmarks]}/{task.fields[total_benchmarks]})",
                style=Colors.progress,
            ),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("]"),
        )
        self.run_progress_task = self.run_progress.add_task("")
        self._sync_run_progress()
        self.update(
            Group(
                Panel(
                    self.tasks_progress,
                    title="Benchmarks",
                    title_align="left",
                    expand=True,
                ),
                self.run_progress,
            )
        )
        self.start()

    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        """
        Update display for new benchmark strategy start.

        :param strategy: Scheduling strategy being executed.
        """
        self.tasks_progress.start_benchmark(strategy)
        self._sync_run_progress()

    async def on_benchmark_update(
        self, aggregator_update: AggregatorState | None, scheduler_state: SchedulerState
    ):
        """
        Update display with current benchmark progress.

        :param aggregator_update: Current benchmark metrics and statistics.
        :param scheduler_state: Current scheduler execution state.
        """
        self.tasks_progress.update_benchmark(aggregator_update, scheduler_state)
        self._sync_run_progress()

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark):
        """
        Update display for completed benchmark.

        :param benchmark: Completed benchmark results.
        """
        self.tasks_progress.complete_benchmark(benchmark)
        self._sync_run_progress()

    async def on_finalize(self):
        """Stop display rendering and cleanup resources."""
        self.tasks_progress.finalize()
        self._sync_run_progress()
        self.run_progress.stop_task(self.run_progress_task)
        self.stop()
        self.run_progress = None
        self.run_progress_task = None
        self.tasks_progress = None

    def _sync_run_progress(self):
        """Synchronize overall progress display with task progress."""
        self.run_progress.update(
            self.run_progress_task,
            total=self.tasks_progress.steps_total,
            completed=self.tasks_progress.steps_progress,
            completed_benchmarks=self.tasks_progress.tasks_progress,
            total_benchmarks=self.tasks_progress.tasks_total,
        )


# Scaling factor for progress calculations to provide granular progress updates
_PROGRESS_SCALE = 1000


class _GenerativeProgressTasks(Progress):
    def __init__(self, profile: Profile, display_scheduler_stats: bool):
        self.profile: Profile = profile
        self.display_scheduler_stats: bool = display_scheduler_stats
        self.benchmark_task_states: list[_GenerativeProgressTaskState] = []
        self.current_index: int = -1

        summary_text = "{task.fields[requests_summary]}\n{task.fields[tokens_summary]}"
        if self.display_scheduler_stats:
            summary_text += "\n{task.fields[scheduler_stats]}"
        super().__init__(
            TextColumn("[{task.fields[start_time]}]"),
            SpinnerColumn(style=Colors.progress),
            TaskProgressColumn(style=Colors.progress),
            TextColumn("{task.description}"),
            TextColumn("({task.fields[progress_status]})"),
            TextColumn(" "),
            TextColumn(summary_text),
        )

        for strategy_type in profile.strategy_types:
            task_state = _GenerativeProgressTaskState(
                strategy_type=strategy_type,
            )
            task_id = self.add_task(**task_state.current)
            task_state.task_id = task_id
            self.benchmark_task_states.append(task_state)

    @property
    def tasks_total(self) -> int:
        return len(self.benchmark_task_states)

    @property
    def tasks_progress(self) -> int:
        return self.current_index + 1

    @property
    def steps_total(self) -> int:
        return _PROGRESS_SCALE * len(self.benchmark_task_states)

    @property
    def steps_progress(self) -> int:
        progress_current_task = (
            self.benchmark_task_states[self.current_index].progress
            if self.current_index < len(self.benchmark_task_states)
            else 0
        )
        progress_total = self.current_index + (progress_current_task or 0)

        return progress_total * _PROGRESS_SCALE

    def start_benchmark(self, strategy: SchedulingStrategy):
        self.current_index += 1
        if self.current_index >= len(self.benchmark_task_states):
            # New task past initially estimated, append it to the end
            task_state = _GenerativeProgressTaskState(strategy_type=strategy.type_)
            task_id = self.add_task(**task_state.current)
            task_state.task_id = task_id
            self.benchmark_task_states.append(task_state)

        self.benchmark_task_states[self.current_index].start(strategy)
        self.update(
            self.benchmark_task_states[self.current_index].task_id,
            start=True,
            **self.benchmark_task_states[self.current_index].current,
        )

    def update_benchmark(
        self, aggregator_update: AggregatorState, scheduler_state: SchedulerState
    ):
        self.benchmark_task_states[self.current_index].update(
            aggregator_update, scheduler_state
        )
        self.update(
            self.benchmark_task_states[self.current_index].task_id,
            **self.benchmark_task_states[self.current_index].current,
        )

    def complete_benchmark(self, benchmark: GenerativeBenchmark):
        self.benchmark_task_states[self.current_index].complete(benchmark)
        self.update(
            self.benchmark_task_states[self.current_index].task_id,
            **self.benchmark_task_states[self.current_index].current,
        )

    def finalize(self):
        self.stop()


@dataclass
class _GenerativeProgressTaskState:
    strategy_type: StrategyType
    task_id: TaskID = None
    strategy: SchedulingStrategy | None = None
    benchmark_status: Literal[
        "pending", "in_warmup", "in_progress", "in_cooldown", "completed"
    ] = "pending"
    progress: float | None = None
    start_time: float = -1.0
    successful_requests: int = 0
    cancelled_requests: int = 0
    errored_requests: int = 0
    request_concurrency: int = 0
    requests_per_second: float = 0
    request_latency: float = 0
    output_tokens: int = 0
    output_tokens_rate: float = 0
    prompt_tokens: int = 0
    total_tokens_rate: float = 0
    time_to_first_token: float = 0
    inter_token_latency: float = 0
    queued_time: float = 0
    request_targeted_start_delay: float = 0
    scheduler_overheads_time: float = 0

    @property
    def current(self) -> dict[str, Any]:
        return {
            "start_time": self.formatted_start_time,
            "description": str(self.strategy or self.strategy_type),
            "progress_status": self.formatted_progress_status,
            "requests_summary": self.formatted_requests_summary,
            "tokens_summary": self.formatted_tokens_summary,
            "scheduler_stats": self.formatted_scheduler_stats,
            "completed": self.completed,
            "total": self.total,
        }

    @property
    def completed(self) -> float:
        if self.benchmark_status == "pending":
            return 0

        if self.benchmark_status == "completed":
            return _PROGRESS_SCALE

        return self.progress * _PROGRESS_SCALE if self.progress is not None else None

    @property
    def total(self) -> float:
        return _PROGRESS_SCALE

    @property
    def formatted_start_time(self) -> str:
        if self.start_time < 0.0:
            return "--:--:--"

        from guidellm.utils.general import safe_format_timestamp

        return safe_format_timestamp(self.start_time, "%H:%M:%S", "--:--:--")

    @property
    def formatted_progress_status(self) -> str:
        if self.benchmark_status == "in_warmup":
            status = "warmup"
            color = Colors.progress
        elif self.benchmark_status == "in_progress":
            status = "running"
            color = Colors.progress
        elif self.benchmark_status == "in_cooldown":
            status = "cooldown"
            color = Colors.progress
        elif self.benchmark_status == "completed":
            status = "complete"
            color = Colors.success
        else:
            status = "pending"
            color = Colors.info

        return f"[{color}]{status.ljust(8)}[/{color}]"

    @property
    def formatted_requests_summary(self) -> str:
        if self.benchmark_status == "pending":
            return " "

        return (
            f"[{Colors.info}]Req:[/{Colors.info}] "
            + format_value_display(
                value=self.requests_per_second,
                label="req/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.request_latency,
                label="Lat",
                units="s",
                total_characters=12,
                digits_places=4,
                decimal_places=2,
            )
            + ", "
            + format_value_display(
                value=self.request_concurrency,
                label="Conc",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.successful_requests,
                label="Comp",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + format_value_display(
                value=self.cancelled_requests,
                label="Inc",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + format_value_display(
                value=self.errored_requests,
                label="Err",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
        )

    @property
    def formatted_tokens_summary(self) -> str:
        if self.benchmark_status == "pending":
            return " "

        return (
            f"[{Colors.info}]Tok:[/{Colors.info}] "
            + format_value_display(
                value=self.output_tokens_rate,
                label="gen/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.total_tokens_rate,
                label="tot/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.time_to_first_token,
                label="TTFT",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.inter_token_latency,
                label="ITL",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.prompt_tokens,
                label="Prompt",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
            + ", "
            + format_value_display(
                value=self.output_tokens,
                label="Gen",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
        )

    @property
    def formatted_scheduler_stats(self) -> str:
        if self.benchmark_status == "pending":
            return " "

        return (
            f"[{Colors.info}]Sys:[/{Colors.info}] , "
            + format_value_display(
                value=self.request_targeted_start_delay,
                label="Start Del",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
            + format_value_display(
                value=self.scheduler_overheads_time,
                label="Sched OH",
                units="ms",
                total_characters=18,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + format_value_display(
                value=self.queued_time,
                label="Queued",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
        )

    def start(self, strategy: SchedulingStrategy):
        self.strategy = strategy
        self.strategy_type = strategy.type_

    def update(
        self, aggregator_update: AggregatorState, scheduler_state: SchedulerState
    ):
        self.progress = scheduler_state.remaining_fraction
        status: Literal["in_warmup", "in_progress", "in_cooldown"] | None = (
            "in_progress"  # Need to handle requests_in_* isn't in aggregator_update
        )
        if aggregator_update.get("requests_in_warmup"):
            status = "in_warmup"
        elif aggregator_update.get("requests_in_cooldown"):
            status = "in_cooldown"
        self._update_processing_states(
            benchmark_status=status,
            start_time=scheduler_state.start_time,
            successful_requests=scheduler_state.successful_requests,
            cancelled_requests=scheduler_state.cancelled_requests,
            errored_requests=scheduler_state.errored_requests,
        )
        self._update_request_stats(
            request_concurrency=aggregator_update.get_metric(
                key="requests", type_="avg", prefix="completed"
            ),
            requests_per_second=aggregator_update.get_metric(
                key="requests",
                type_="rate",
                prefix="completed",
            ),
            request_latency=aggregator_update.get_metric(
                key="request_latency", type_="avg", prefix="completed"
            ),
        )
        self._update_token_stats(
            output_tokens=aggregator_update.get_metric(
                key="output_tokens", type_="avg", prefix="completed"
            ),
            output_tokens_rate=aggregator_update.get_metric(
                key="output_tokens", type_="rate"
            ),
            prompt_tokens=aggregator_update.get_metric(
                key="prompt_tokens", type_="avg", prefix="completed"
            ),
            total_tokens_rate=aggregator_update.get_metric(
                key="total_tokens", type_="rate"
            ),
            time_to_first_token=(
                aggregator_update.get_metric(key="time_to_first_token", type_="avg")
            ),
            inter_token_latency=(
                aggregator_update.get_metric(key="inter_token_latency", type_="avg")
            ),
        )
        if aggregator_update.get("updated_scheduler_stats"):
            self._update_system_stats(
                request_targeted_start_delay=(
                    aggregator_update.get_metric(
                        key="request_targeted_start_delay", type_="avg", default=0.0
                    )
                ),
                queued_time=(
                    aggregator_update.get_metric(
                        key="queued_time", type_="avg", default=0.0
                    )
                ),
                scheduler_overheads_time=0.0,  # Need to add up metrics here
            )

    def complete(self, benchmark: GenerativeBenchmark):
        self._update_processing_states(
            benchmark_status="completed",
            start_time=benchmark.start_time,
            successful_requests=benchmark.request_totals.successful,
            cancelled_requests=benchmark.request_totals.incomplete,
            errored_requests=benchmark.request_totals.errored,
        )
        self._update_request_stats(
            request_concurrency=benchmark.metrics.request_concurrency.successful.mean,
            requests_per_second=benchmark.metrics.requests_per_second.successful.mean,
            request_latency=benchmark.metrics.request_latency.successful.mean,
        )
        self._update_token_stats(
            output_tokens=benchmark.metrics.output_token_count.successful.mean,
            output_tokens_rate=benchmark.metrics.output_tokens_per_second.successful.mean,
            prompt_tokens=benchmark.metrics.prompt_token_count.successful.mean,
            total_tokens_rate=benchmark.metrics.tokens_per_second.successful.mean,
            time_to_first_token=(
                benchmark.metrics.time_to_first_token_ms.successful.mean
            ),
            inter_token_latency=(
                benchmark.metrics.inter_token_latency_ms.successful.mean
            ),
            converted=True,
        )

    def _update_processing_states(
        self,
        benchmark_status: Literal[
            "pending", "in_warmup", "in_progress", "in_cooldown", "completed"
        ],
        start_time: float | None = None,
        successful_requests: int | None = None,
        cancelled_requests: int | None = None,
        errored_requests: int | None = None,
    ):
        if self.benchmark_status is not None:
            self.benchmark_status = benchmark_status
        if start_time is not None:
            self.start_time = start_time
        if successful_requests is not None:
            self.successful_requests = successful_requests
        if cancelled_requests is not None:
            self.cancelled_requests = cancelled_requests
        if errored_requests is not None:
            self.errored_requests = errored_requests

    def _update_request_stats(
        self,
        request_concurrency: int | None = None,
        requests_per_second: float | None = None,
        request_latency: float | None = None,
    ):
        if request_concurrency is not None:
            self.request_concurrency = request_concurrency
        if requests_per_second is not None:
            self.requests_per_second = requests_per_second
        if request_latency is not None:
            self.request_latency = request_latency

    def _update_token_stats(
        self,
        output_tokens: int | None = None,
        output_tokens_rate: float | None = None,
        prompt_tokens: int | None = None,
        total_tokens_rate: float | None = None,
        time_to_first_token: float | None = None,
        inter_token_latency: float | None = None,
        converted: bool = False,
    ):
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if output_tokens_rate is not None:
            self.output_tokens_rate = output_tokens_rate
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if total_tokens_rate is not None:
            self.total_tokens_rate = total_tokens_rate
        if time_to_first_token is not None:
            self.time_to_first_token = time_to_first_token * (
                1000 if not converted else 1
            )
        if inter_token_latency is not None:
            self.inter_token_latency = inter_token_latency * (
                1000 if not converted else 1
            )

    def _update_system_stats(
        self,
        request_targeted_start_delay: float | None = None,
        queued_time: float | None = None,
        scheduler_overheads_time: float | None = None,
        converted: bool = False,
    ):
        if request_targeted_start_delay is not None:
            self.request_targeted_start_delay = request_targeted_start_delay * (
                1000 if not converted else 1
            )
        if queued_time is not None:
            self.queued_time = queued_time * (1000 if not converted else 1)
        if scheduler_overheads_time is not None:
            self.scheduler_overheads_time = scheduler_overheads_time * (
                1000 if not converted else 1
            )
