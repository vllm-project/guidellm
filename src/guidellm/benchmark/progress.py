import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Optional, TypeVar, Union

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from guidellm.benchmark.aggregator import (
    BenchmarkAggregator,
    GenerativeBenchmarkAggregator,
)
from guidellm.benchmark.benchmark import Benchmark, GenerativeBenchmark
from guidellm.benchmark.benchmarker import BenchmarkerResult
from guidellm.scheduler import (
    SchedulingStrategy,
    StrategyType,
    strategy_display_str,
)
from guidellm.utils import Colors

__all__ = [
    "BenchmarkerProgressDisplay",
    "BenchmarkerTaskProgressState",
    "GenerativeTextBenchmarkerProgressDisplay",
    "GenerativeTextBenchmarkerTaskProgressState",
]


@dataclass
class BenchmarkerTaskProgressState:
    display_scheduler_stats: bool

    task_id: TaskID
    strategy: Union[StrategyType, SchedulingStrategy]
    started: bool = False
    compiling: bool = False
    ended: bool = False

    start_time: Optional[float] = None
    max_number: Optional[float] = None
    max_duration: Optional[float] = None
    in_warmup: bool = False
    in_cooldown: bool = False

    requests_rate: float = 0
    request_latency: float = 0
    requests_processing: float = 0
    requests_successful: float = 0
    requests_incomplete: float = 0
    requests_errored: float = 0

    worker_overheads_time_ms: float = 0.0
    backend_overheads_time_ms: float = 0.0
    requests_sleep_time_ms: float = 0.0
    requests_targeted_start_time_delay_ms: float = 0.0

    @property
    def description(self) -> str:
        return strategy_display_str(self.strategy)

    @property
    def total(self) -> Optional[float]:
        if self.max_number is None and self.max_duration is None:
            return None

        return 1000

    @property
    def completed(self) -> int:
        if self.ended:
            return 1000

        if self.max_number is None and self.max_duration is None:
            return 0

        number = self.requests_successful + self.requests_errored
        number_percent = (
            number / float(self.max_number) * 1000 if self.max_number else -math.inf
        )
        duration_percent = (
            (time.time() - self.start_time) / self.max_duration * 1000
            if self.max_duration and self.start_time
            else -math.inf
        )

        return min(int(max(number_percent, duration_percent)), 1000)

    @property
    def fields(self) -> dict[str, str]:
        fields = {
            "start_time": self.formatted_start_time,
            "progress_status": self.formatted_progress_status,
            "requests_summary": self.formatted_requests_summary,
        }

        if self.display_scheduler_stats:
            fields["scheduler_stats"] = self.formatted_scheduler_stats

        return fields

    @property
    def formatted_start_time(self) -> str:
        if self.start_time is None:
            return "--:--:--"

        return datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")

    @property
    def formatted_progress_status(self) -> str:
        if self.ended:
            status = "complete"
            color = Colors.SUCCESS
        elif self.compiling:
            status = "compiling"
            color = Colors.PROGRESS
        elif self.started and self.in_warmup:
            status = "warmup"
            color = Colors.PROGRESS
        elif self.started and self.in_cooldown:
            status = "cooldown"
            color = Colors.PROGRESS
        elif self.started:
            status = "running"
            color = Colors.PROGRESS
        else:
            status = "pending"
            color = Colors.INFO

        return f"[{color}]{status.ljust(8)}[/{color}]"

    @property
    def formatted_requests_summary(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Req:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_rate,
                label="req/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.request_latency,
                label="Lat",
                units="s",
                total_characters=12,
                digits_places=4,
                decimal_places=2,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_processing,
                label="Conc",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_successful,
                label="Comp",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_incomplete,
                label="Inc",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_errored,
                label="Err",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
        )

    @property
    def formatted_scheduler_stats(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Sys:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.worker_overheads_time_ms,
                label="Work OH",
                units="ms",
                total_characters=18,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.backend_overheads_time_ms,
                label="Back OH",
                units="ms",
                total_characters=18,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_sleep_time_ms,
                label="Req Sleep",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_targeted_start_time_delay_ms,
                label="Start Del",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
        )

    @staticmethod
    def format_progress_display(
        value: float,
        label: str,
        units: str = "",
        total_characters: Optional[int] = None,
        digits_places: Optional[int] = None,
        decimal_places: Optional[int] = None,
    ) -> str:
        if decimal_places is None and digits_places is None:
            formatted_number = f"{value:.0f}"
        elif digits_places is None:
            formatted_number = f"{value:.{decimal_places}f}"
        elif decimal_places is None:
            formatted_number = f"{value:>{digits_places}f}"
        else:
            formatted_number = f"{value:>{digits_places}.{decimal_places}f}"

        result = f"{formatted_number}{units} [{Colors.INFO}]{label}[/{Colors.INFO}]"

        if total_characters is not None:
            total_characters += len(Colors.INFO) * 2 + 5

            if len(result) < total_characters:
                result = result.rjust(total_characters)

        return result


class GenerativeTextBenchmarkerTaskProgressState(BenchmarkerTaskProgressState):
    output_tokens: float = 0
    prompt_tokens: float = 0
    output_tokens_rate: float = 0
    total_tokens_rate: float = 0
    tokens_ttft: float = 0
    tokens_itl: float = 0

    @property
    def fields(self) -> dict[str, str]:
        fields = super().fields
        fields["tokens_summary"] = self.formatted_tokens_summary
        return fields

    @property
    def formatted_tokens_summary(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Tok:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.output_tokens_rate,
                label="gen/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.total_tokens_rate,
                label="tot/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.tokens_ttft,
                label="TTFT",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.tokens_itl,
                label="ITL",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.prompt_tokens,
                label="Prompt",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.output_tokens,
                label="Gen",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
        )


BTPS = TypeVar("BTPS", bound=BenchmarkerTaskProgressState)


class BenchmarkerProgressDisplay(Generic[BTPS]):
    def __init__(self, display_scheduler_stats: bool):
        self.display_scheduler_stats = display_scheduler_stats
        self.started = False
        self.benchmarker_tasks_progress = Progress(*self.create_task_progress_columns())
        self.benchmarker_tasks_panel = Panel(
            self.benchmarker_tasks_progress,
            title="Benchmarks",
            title_align="left",
            expand=True,
        )
        self.benchmarker_progress = Progress(
            TextColumn("Generating...", style=f"italic {Colors.PROGRESS}"),
            BarColumn(
                bar_width=None,
                complete_style=Colors.PROGRESS,
                finished_style=Colors.SUCCESS,
            ),
            TextColumn(
                "({task.fields[completed_benchmarks]}/{task.fields[total_benchmarks]})",
                style=Colors.PROGRESS,
            ),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("]"),
        )
        self.benchmarker_live = Live(
            Group(
                self.benchmarker_tasks_panel,
                self.benchmarker_progress,
            ),
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self.active_task: Optional[TaskID] = None
        self.benchmarker_tasks: list[BTPS] = []
        self.progress_task: Optional[TaskID] = None

    def update(self, result: BenchmarkerResult):
        if result.type_ == "run_start":
            if self.started:
                raise RuntimeError("Progress display already started.")

            self.handle_start(result)
            self.started = True
        elif result.type_ == "run_complete":
            if not self.started:
                raise RuntimeError("Progress display not started.")

            self.handle_end(result)
            self.started = False
        else:
            if not self.started:
                raise RuntimeError("Progress display not started.")

            self.handle_update(result)

    def handle_start(self, result: BenchmarkerResult):
        self.benchmarker_live.start()

        for index, strategy_type in enumerate(result.profile.strategy_types):
            task_id = self.benchmarker_tasks_progress.add_task(
                description=strategy_type,
                start=False,
                total=None,
                completed=0,
                visible=False,
            )
            task_progress_state = self.create_task_progress_state(
                task_id=task_id,
                index=index,
                strategy_type=strategy_type,
                result=result,
            )
            self.benchmarker_tasks.append(task_progress_state)
            self.benchmarker_tasks_progress.update(
                task_id,
                description=task_progress_state.description,
                visible=True,
                **task_progress_state.fields,  # type: ignore[arg-type]
            )

        self.progress_task = self.benchmarker_progress.add_task(
            "",
            total=len(self.benchmarker_tasks) * 1000,
            completed_benchmarks=0,
            total_benchmarks=len(self.benchmarker_tasks),
        )

    def handle_update(self, result: BenchmarkerResult):
        current_state: BTPS = self.benchmarker_tasks[result.current_index]

        if result.type_ == "scheduler_start":
            self.handle_update_scheduler_start(current_state, result)
            self.active_task = current_state.task_id
        elif result.type_ == "scheduler_update":
            self.handle_update_scheduler_update(current_state, result)
        elif result.type_ == "scheduler_complete":
            self.handle_update_scheduler_complete(current_state, result)
        elif result.type_ == "benchmark_compiled":
            self.handle_update_benchmark_compiled(current_state, result)
        else:
            raise ValueError(f"Unknown result type: {result.type_}")

        if self.progress_task is None:
            raise RuntimeError("Progress task not set.")

        self.benchmarker_tasks_progress.update(
            current_state.task_id,
            description=current_state.description,
            completed=current_state.completed,
            total=current_state.total,
            **current_state.fields,  # type: ignore[arg-type]
        )
        self.benchmarker_progress.update(
            self.progress_task,
            completed=(result.current_index * 1000) + current_state.completed,
            total=1000 * len(self.benchmarker_tasks),
            completed_benchmarks=(
                result.current_index + (1 if current_state.ended else 0)
            ),
            total_benchmarks=len(self.benchmarker_tasks),
        )

        if current_state.ended:
            self.benchmarker_tasks_progress.stop_task(current_state.task_id)
            self.active_task = None

    def handle_update_scheduler_start(
        self, progress_state: BTPS, result: BenchmarkerResult
    ):
        if self.active_task is not None:
            raise RuntimeError("Active task already set.")

        progress_state.strategy = result.current_strategy  # type: ignore[assignment]
        progress_state.started = True
        current_aggregator: BenchmarkAggregator = result.current_aggregator  # type: ignore[assignment]
        progress_state.start_time = (
            current_aggregator.requests_stats.totals.total.start_time
        )
        progress_state.max_number = current_aggregator.args.max_number
        progress_state.max_duration = current_aggregator.args.max_duration

    def handle_update_scheduler_update(
        self, progress_state: BTPS, result: BenchmarkerResult
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        current_aggregator: BenchmarkAggregator = result.current_aggregator  # type: ignore[assignment]
        progress_state.in_warmup = current_aggregator.in_warmup
        progress_state.in_cooldown = current_aggregator.in_cooldown
        progress_state.requests_rate = (
            current_aggregator.requests_stats.totals.successful.rate
        )
        progress_state.request_latency = (
            current_aggregator.requests_stats.request_time.mean
        )
        progress_state.requests_processing = (
            current_aggregator.scheduler_stats.processing_requests.last
        )
        progress_state.requests_successful = (
            current_aggregator.requests_stats.totals.successful.total
        )
        progress_state.requests_incomplete = (
            current_aggregator.requests_stats.totals.incomplete.total
        )
        progress_state.requests_errored = (
            current_aggregator.requests_stats.totals.errored.total
        )
        progress_state.worker_overheads_time_ms = (
            current_aggregator.requests_stats.scheduled_time_delay.mean_ms
            + current_aggregator.requests_stats.worker_start_delay.mean_ms
        )
        progress_state.backend_overheads_time_ms = (
            current_aggregator.requests_stats.request_time_delay.mean_ms
        )
        progress_state.requests_sleep_time_ms = (
            current_aggregator.requests_stats.scheduled_time_sleep.mean_ms
        )
        progress_state.requests_targeted_start_time_delay_ms = (
            current_aggregator.requests_stats.request_start_time_targeted_delay.mean_ms
        )

    def handle_update_scheduler_complete(
        self,
        progress_state: BTPS,
        result: BenchmarkerResult,  # noqa: ARG002
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        progress_state.in_warmup = False
        progress_state.in_cooldown = False
        progress_state.compiling = True

    def handle_update_benchmark_compiled(
        self, progress_state: BTPS, result: BenchmarkerResult
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        current_benchmark: Benchmark = result.current_benchmark  # type: ignore[assignment]
        progress_state.compiling = False
        progress_state.ended = True
        progress_state.requests_rate = (
            current_benchmark.metrics.requests_per_second.successful.mean
        )
        progress_state.requests_processing = (
            current_benchmark.metrics.request_concurrency.successful.mean
        )

    def handle_end(self, result: BenchmarkerResult):  # noqa: ARG002
        if self.progress_task is None:
            raise RuntimeError("Progress task not set.")

        self.benchmarker_progress.update(
            self.progress_task,
            completed=len(self.benchmarker_tasks) * 1000,
            total=len(self.benchmarker_tasks) * 1000,
            completed_benchmarks=len(self.benchmarker_tasks),
            total_benchmarks=len(self.benchmarker_tasks),
        )
        self.benchmarker_progress.stop_task(self.progress_task)
        self.benchmarker_live.stop()
        self.active_task = None
        self.benchmarker_tasks = []
        self.progress_task = None

    def create_task_progress_columns(self) -> list[ProgressColumn]:
        columns = [
            TextColumn("[{task.fields[start_time]}]"),
            SpinnerColumn(style=Colors.PROGRESS),
            TaskProgressColumn(style=Colors.PROGRESS),
            TextColumn("{task.description}"),
            TextColumn("({task.fields[progress_status]})"),
            TextColumn(" "),
        ]

        if not self.display_scheduler_stats:
            columns += [
                TextColumn("{task.fields[requests_summary]}\n"),
            ]
        else:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[scheduler_stats]}\n"
                ),
            ]

        return columns

    def create_task_progress_state(
        self,
        task_id: TaskID,
        index: int,  # noqa: ARG002
        strategy_type: StrategyType,
        result: BenchmarkerResult,  # noqa: ARG002
    ) -> BTPS:
        return BenchmarkerTaskProgressState(  # type: ignore[return-value]
            display_scheduler_stats=self.display_scheduler_stats,
            task_id=task_id,
            strategy=strategy_type,
        )


class GenerativeTextBenchmarkerProgressDisplay(
    BenchmarkerProgressDisplay[GenerativeTextBenchmarkerTaskProgressState]
):
    def handle_update_scheduler_update(
        self,
        progress_state: GenerativeTextBenchmarkerTaskProgressState,
        result: BenchmarkerResult,
    ):
        super().handle_update_scheduler_update(progress_state, result)
        current_aggregator: GenerativeBenchmarkAggregator = result.current_aggregator  # type: ignore[assignment]
        progress_state.output_tokens = (
            current_aggregator.requests_stats.output_tokens.mean
        )
        progress_state.prompt_tokens = (
            current_aggregator.requests_stats.prompt_tokens.mean
        )
        progress_state.output_tokens_rate = (
            current_aggregator.requests_stats.output_tokens.rate
        )
        progress_state.total_tokens_rate = (
            current_aggregator.requests_stats.total_tokens.rate
        )
        progress_state.tokens_ttft = (
            current_aggregator.requests_stats.time_to_first_token.mean_ms
        )
        progress_state.tokens_itl = (
            current_aggregator.requests_stats.inter_token_latency.mean_ms
        )

    def handle_update_benchmark_compiled(
        self,
        progress_state: GenerativeTextBenchmarkerTaskProgressState,
        result: BenchmarkerResult,
    ):
        super().handle_update_benchmark_compiled(progress_state, result)

        current_benchmark: GenerativeBenchmark = result.current_benchmark  # type: ignore[assignment]
        progress_state.request_latency = (
            current_benchmark.metrics.request_latency.successful.mean
        )
        progress_state.requests_successful = current_benchmark.request_totals.successful
        progress_state.requests_errored = current_benchmark.request_totals.errored
        progress_state.requests_incomplete = current_benchmark.request_totals.incomplete
        progress_state.output_tokens = (
            current_benchmark.metrics.output_token_count.successful.mean
        )
        progress_state.prompt_tokens = (
            current_benchmark.metrics.prompt_token_count.successful.mean
        )
        progress_state.output_tokens_rate = (
            current_benchmark.metrics.output_tokens_per_second.successful.mean
        )
        progress_state.total_tokens_rate = (
            current_benchmark.metrics.tokens_per_second.successful.mean
        )
        progress_state.tokens_ttft = (
            current_benchmark.metrics.time_to_first_token_ms.successful.mean
        )
        progress_state.tokens_itl = (
            current_benchmark.metrics.inter_token_latency_ms.successful.mean
        )

    def create_task_progress_state(
        self,
        task_id: TaskID,
        index: int,  # noqa: ARG002
        strategy_type: StrategyType,
        result: BenchmarkerResult,  # noqa: ARG002
    ) -> GenerativeTextBenchmarkerTaskProgressState:
        return GenerativeTextBenchmarkerTaskProgressState(
            display_scheduler_stats=self.display_scheduler_stats,
            task_id=task_id,
            strategy=strategy_type,
        )

    def create_task_progress_columns(self) -> list[ProgressColumn]:
        columns = super().create_task_progress_columns()
        columns = columns[:-1]  # remove the last display info column

        if not self.display_scheduler_stats:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[tokens_summary]}",
                ),
            ]
        else:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[tokens_summary]}\n{task.fields[scheduler_stats]}",
                ),
            ]

        return columns
