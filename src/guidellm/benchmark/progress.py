"""
Progress tracking and console display for benchmark execution monitoring.

Provides abstract interfaces and concrete implementations for tracking benchmark
progress during execution. The module enables real-time display of benchmark
statistics, metrics, and execution state through console-based UI components.
Primary use cases include monitoring generative benchmark runs with detailed
request/token statistics and scheduler state updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

from guidellm.benchmark.profiles import Profile
from guidellm.benchmark.schemas import (
    BenchmarkAccumulatorT,
    BenchmarkT,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
)
from guidellm.scheduler import SchedulerState, SchedulingStrategy
from guidellm.utils import Colors, format_value_display, safe_format_timestamp

__all__ = ["BenchmarkerProgress", "GenerativeConsoleBenchmarkerProgress"]


class BenchmarkerProgress(Generic[BenchmarkAccumulatorT, BenchmarkT], ABC):
    """
    Abstract interface for tracking and displaying benchmark execution progress.

    Provides lifecycle hooks for monitoring benchmark stages including initialization,
    execution start, progress updates, completion, and finalization. Implementations
    handle display updates, progress tracking, and resource management for benchmark
    monitoring.
    """

    def __init__(self):
        """Initialize progress tracker with default state."""
        self.profile: Profile | None = None
        self.current_strategy: SchedulingStrategy | None = None

    @abstractmethod
    async def on_initialize(self, profile: Profile):
        """
        Initialize progress tracking for the given benchmark profile.

        :param profile: Benchmark profile configuration defining execution parameters
        """

    @abstractmethod
    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        """
        Handle benchmark strategy execution start event.

        :param strategy: Scheduling strategy configuration being executed
        """

    @abstractmethod
    async def on_benchmark_update(
        self, accumulator: BenchmarkAccumulatorT, scheduler_state: SchedulerState
    ):
        """
        Handle benchmark execution progress update with current metrics.

        :param accumulator: Current accumulated benchmark metrics and statistics
        :param scheduler_state: Current scheduler execution state and counters
        """

    @abstractmethod
    async def on_benchmark_complete(self, benchmark: BenchmarkT):
        """
        Handle benchmark strategy execution completion event.

        :param benchmark: Completed benchmark results with final metrics
        """

    @abstractmethod
    async def on_finalize(self):
        """Finalize progress tracking and release associated resources."""


class GenerativeConsoleBenchmarkerProgress(
    BenchmarkerProgress[GenerativeBenchmarkAccumulator, GenerativeBenchmark], Live
):
    """
    Console-based real-time progress display for generative benchmarks.

    Renders live benchmark execution statistics using Rich library components with
    structured progress bars, timing information, request/token metrics, and optional
    scheduler statistics. Updates refresh automatically during benchmark execution.

    :cvar display_scheduler_stats: Whether to include scheduler statistics in display
    """

    def __init__(self, display_scheduler_stats: bool = False):
        """
        Initialize console progress display with rendering configuration.

        :param display_scheduler_stats: Whether to display scheduler timing statistics
        """
        super().__init__()
        Live.__init__(
            self,
            refresh_per_second=4,
            auto_refresh=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self.display_scheduler_stats: bool = display_scheduler_stats
        self.run_progress: Progress | None = None
        self.run_progress_task: TaskID | None = None
        self.tasks_progress: _GenerativeProgressTasks | None = None

    async def on_initialize(self, profile: Profile):
        """
        Initialize console display components and begin live rendering.

        :param profile: Benchmark profile configuration defining execution parameters
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
        Update display for benchmark strategy execution start.

        :param strategy: Scheduling strategy configuration being executed
        """
        if self.tasks_progress is not None:
            self.tasks_progress.start_benchmark(strategy)
            self._sync_run_progress()

    async def on_benchmark_update(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        """
        Update display with current benchmark progress and metrics.

        :param accumulator: Current accumulated benchmark metrics and statistics
        :param scheduler_state: Current scheduler execution state and counters
        """
        if self.tasks_progress is not None:
            self.tasks_progress.update_benchmark(accumulator, scheduler_state)
            self._sync_run_progress()

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark):
        """
        Update display for completed benchmark strategy.

        :param benchmark: Completed benchmark results with final metrics
        """
        if self.tasks_progress is not None:
            self.tasks_progress.complete_benchmark(benchmark)
            self._sync_run_progress()

    async def on_finalize(self):
        """Stop display rendering and release resources."""
        if self.tasks_progress is not None:
            self.tasks_progress.finalize()
            self._sync_run_progress()
        if self.run_progress is not None and self.run_progress_task is not None:
            self.run_progress.stop_task(self.run_progress_task)
        self.stop()
        self.run_progress = None
        self.run_progress_task = None
        self.tasks_progress = None

    def _sync_run_progress(self):
        """Synchronize overall progress display with task progress."""
        if (
            self.run_progress is not None
            and self.run_progress_task is not None
            and self.tasks_progress is not None
        ):
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

        return int(progress_total * _PROGRESS_SCALE)

    def start_benchmark(self, strategy: SchedulingStrategy):
        self.current_index += 1
        if self.current_index >= len(self.benchmark_task_states):
            # New task past initially estimated, append it to the end
            task_state = _GenerativeProgressTaskState(strategy_type=strategy.type_)
            task_id = self.add_task(**task_state.current)
            task_state.task_id = task_id
            self.benchmark_task_states.append(task_state)

        current_state = self.benchmark_task_states[self.current_index]
        current_state.start(strategy)
        if current_state.task_id is not None:
            self.update(
                current_state.task_id,
                start=True,
                **current_state.current,
            )

    def update_benchmark(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        current_state = self.benchmark_task_states[self.current_index]
        current_state.update(accumulator, scheduler_state)
        if current_state.task_id is not None:
            self.update(
                current_state.task_id,
                **current_state.current,
            )

    def complete_benchmark(self, benchmark: GenerativeBenchmark):
        current_state = self.benchmark_task_states[self.current_index]
        current_state.complete(benchmark)
        if current_state.task_id is not None:
            self.update(
                current_state.task_id,
                **current_state.current,
            )

    def finalize(self):
        self.stop()


@dataclass
class _GenerativeProgressTaskState:
    strategy_type: str
    task_id: TaskID | None = None
    strategy: SchedulingStrategy | None = None
    benchmark_status: Literal[
        "pending", "warmup", "active", "cooldown", "completed"
    ] = "pending"
    progress: float | None = None
    start_time: float = -1.0
    successful_requests: int = 0
    cancelled_requests: int = 0
    errored_requests: int = 0
    request_concurrency: float = 0.0
    requests_per_second: float = 0.0
    request_latency: float = 0.0
    output_tokens: float = 0
    output_tokens_rate: float = 0.0
    prompt_tokens: float = 0
    total_tokens_rate: float = 0.0
    time_to_first_token: float = 0.0
    inter_token_latency: float = 0.0
    queued_time: float = 0.0
    request_targeted_start_delay: float = 0.0
    scheduler_overheads_time: float = 0.0

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
            return 0.0

        if self.benchmark_status == "completed":
            return float(_PROGRESS_SCALE)

        return self.progress * _PROGRESS_SCALE if self.progress is not None else 0.0

    @property
    def total(self) -> float:
        return _PROGRESS_SCALE

    @property
    def formatted_start_time(self) -> str:
        if self.start_time < 0.0:
            return "--:--:--"

        return safe_format_timestamp(self.start_time, format_="%H:%M:%S")

    @property
    def formatted_progress_status(self) -> str:
        if self.benchmark_status == "warmup":
            status = "warmup"
            color = Colors.progress
        elif self.benchmark_status == "active":
            status = "running"
            color = Colors.progress
        elif self.benchmark_status == "cooldown":
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
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        self.progress = (
            (1.0 - scheduler_state.progress.remaining_fraction)
            if scheduler_state.progress.remaining_fraction is not None
            else 0.0
        )
        self._update_processing_states(
            benchmark_status=self._map_status(accumulator.timings.status),
            start_time=accumulator.timings.measure_start,
            successful_requests=scheduler_state.successful_requests,
            cancelled_requests=scheduler_state.cancelled_requests,
            errored_requests=scheduler_state.errored_requests,
        )
        self._update_request_stats(
            request_concurrency=accumulator.concurrency_metric.time_weighted_mean,
            requests_per_second=accumulator.completed_metrics.requests.rate_per_second,
            request_latency=accumulator.completed_metrics.request_latency.mean,
        )
        self._update_token_stats(
            output_tokens=accumulator.completed_metrics.total_tokens.mean,
            output_tokens_rate=accumulator.completed_metrics.output_tokens.rate_per_second,
            prompt_tokens=accumulator.completed_metrics.prompt_tokens.mean,
            total_tokens_rate=accumulator.completed_metrics.total_tokens.rate_per_second,
            time_to_first_token=accumulator.completed_metrics.time_to_first_token_ms.mean,
            inter_token_latency=accumulator.completed_metrics.inter_token_latency_ms.mean,
            converted=True,
        )
        self._update_system_stats(
            request_targeted_start_delay=accumulator.scheduler_metrics.request_targeted_start_delay.mean,
            queued_time=accumulator.scheduler_metrics.queued_time.mean,
            scheduler_overheads_time=accumulator.scheduler_metrics.resolve_end_delay.mean,
            converted=False,
        )

    def complete(self, benchmark: GenerativeBenchmark):
        self._update_processing_states(
            benchmark_status="completed",
            start_time=benchmark.start_time,
            successful_requests=benchmark.metrics.request_totals.successful,
            cancelled_requests=benchmark.metrics.request_totals.incomplete,
            errored_requests=benchmark.metrics.request_totals.errored,
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

    @staticmethod
    def _map_status(
        status: Literal["pending", "warmup", "active", "cooldown", "completed"],
    ) -> Literal["pending", "warmup", "active", "cooldown", "completed"]:
        """Map accumulator status to internal progress status representation."""
        return status

    def _update_processing_states(
        self,
        benchmark_status: Literal[
            "pending", "warmup", "active", "cooldown", "completed"
        ]
        | None = None,
        start_time: float | None = None,
        successful_requests: int | None = None,
        cancelled_requests: int | None = None,
        errored_requests: int | None = None,
    ):
        if benchmark_status is not None:
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
        request_concurrency: float | None = None,
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
        output_tokens: float | None = None,
        output_tokens_rate: float | None = None,
        prompt_tokens: float | None = None,
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
