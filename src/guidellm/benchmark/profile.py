from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field, computed_field

from guidellm.config import settings
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    SchedulingStrategy,
    StrategyType,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.utils import StandardBaseModel

__all__ = [
    "AsyncProfile",
    "ConcurrentProfile",
    "Profile",
    "ProfileType",
    "SweepProfile",
    "SynchronousProfile",
    "ThroughputProfile",
    "create_profile",
]

ProfileType = Literal["synchronous", "concurrent", "throughput", "async", "sweep"]


class Profile(StandardBaseModel):
    type_: Literal["profile"] = Field(
        description="The type of benchmarking profile to use.",
    )
    completed_strategies: int = Field(
        default=0,
        description="The number of scheduling strategies generated so far.",
    )
    measured_rates: list[float] = Field(
        default_factory=list,
        description=("The average rates measured for the strategies that have run."),
    )
    measured_concurrencies: list[float] = Field(
        default_factory=list,
        description=(
            "The average concurrency measured for the strategies that have run."
        ),
    )

    def completed_strategy(self, average_rate: float, average_concurrency: float):
        self.measured_rates.append(average_rate)
        self.measured_concurrencies.append(average_concurrency)
        self.completed_strategies += 1

    @computed_field  # type: ignore[misc]
    @property
    def strategy_types(self) -> list[StrategyType]:
        return []

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        return None


class SynchronousProfile(Profile):
    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @property
    def strategy_types(self) -> list[StrategyType]:
        return [self.type_]

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        if self.completed_strategies >= 1:
            return None

        return SynchronousStrategy()

    @staticmethod
    def from_standard_args(
        rate_type: Union[StrategyType, ProfileType],
        rate: Optional[Union[float, Sequence[float]]],
        **kwargs,
    ) -> "SynchronousProfile":
        if rate_type != "synchronous":
            raise ValueError("Rate type must be 'synchronous' for synchronous profile.")

        if rate is not None:
            raise ValueError(
                "Rate does not apply to synchronous profile, it must be set to None."
            )

        if kwargs:
            raise ValueError(
                "No additional arguments are allowed for synchronous profile."
            )

        return SynchronousProfile()


class ConcurrentProfile(Profile):
    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: Union[int, Sequence[int]] = Field(
        description="The number of concurrent streams to use.",
    )

    @property
    def strategy_types(self) -> list[StrategyType]:
        num_strategies = len(self.streams) if isinstance(self.streams, Sequence) else 1

        return [self.type_] * num_strategies

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        streams = self.streams if isinstance(self.streams, Sequence) else [self.streams]

        if self.completed_strategies >= len(streams):
            return None

        return ConcurrentStrategy(
            streams=streams[self.completed_strategies],
        )

    @staticmethod
    def from_standard_args(
        rate_type: Union[StrategyType, ProfileType],
        rate: Optional[Union[float, Sequence[float]]],
        **kwargs,
    ) -> "ConcurrentProfile":
        if rate_type != "concurrent":
            raise ValueError("Rate type must be 'concurrent' for concurrent profile.")

        if not rate:
            raise ValueError("Rate (streams) must be provided for concurrent profile.")

        if not isinstance(rate, Sequence):
            rate = [rate]

        if not all(stream.is_integer() and stream > 0 for stream in rate):
            raise ValueError(
                f"All rate values (streams) must be positive integers, received {rate}"
            )

        if kwargs:
            raise ValueError(
                "No additional arguments are allowed for concurrent profile."
            )

        return ConcurrentProfile(streams=[int(rat) for rat in rate])


class ThroughputProfile(Profile):
    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: Optional[int] = Field(
        default=None,
        description="The maximum number of concurrent requests that can be scheduled.",
    )

    @property
    def strategy_types(self) -> list[StrategyType]:
        return [self.type_]

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        if self.completed_strategies >= 1:
            return None

        return ThroughputStrategy(
            max_concurrency=self.max_concurrency,
        )

    @staticmethod
    def from_standard_args(
        rate_type: Union[StrategyType, ProfileType],
        rate: Optional[Union[float, Sequence[float]]],
        **kwargs,
    ) -> "ThroughputProfile":
        if rate_type != "throughput":
            raise ValueError("Rate type must be 'throughput' for throughput profile.")

        if rate is not None:
            raise ValueError(
                "Rate does not apply to throughput profile, it must be set to None."
            )

        return ThroughputProfile(**kwargs)


class AsyncProfile(ThroughputProfile):
    type_: Literal["async"] = "async"  # type: ignore[assignment]
    strategy_type: Literal["constant", "poisson"] = Field(
        description="The type of asynchronous strategy to use.",
    )
    rate: Union[float, Sequence[float]] = Field(
        description="The rate of requests per second to use.",
    )
    initial_burst: bool = Field(
        default=True,
        description=(
            "True to send an initial burst of requests (math.floor(self.rate)) "
            "to reach target rate. False to not send an initial burst."
        ),
    )
    random_seed: int = Field(
        default=42,
        description=(
            "The random seed to use for the asynchronous strategy. "
            "This is used to generate random numbers for the Poisson strategy."
        ),
    )

    @property
    def strategy_types(self) -> list[StrategyType]:
        num_strategies = len(self.rate) if isinstance(self.rate, Sequence) else 1

        return [self.strategy_type] * num_strategies

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        rate = self.rate if isinstance(self.rate, Sequence) else [self.rate]

        if self.completed_strategies >= len(rate):
            return None

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=rate[self.completed_strategies],
                initial_burst=self.initial_burst,
                max_concurrency=self.max_concurrency,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=rate[self.completed_strategies],
                initial_burst=self.initial_burst,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")

    @staticmethod
    def from_standard_args(  # type: ignore[override]
        rate_type: Union[StrategyType, ProfileType],
        rate: Optional[Union[float, Sequence[float]]],
        random_seed: int,
        **kwargs,
    ) -> "AsyncProfile":
        if rate_type not in ("async", "constant", "poisson"):
            raise ValueError(
                "Rate type must be in ('async', 'constant', 'poisson') "
                f"for async profile. Received: {rate_type}"
            )

        if not rate:
            raise ValueError("Rate must be provided for async profile.")

        if not isinstance(rate, Sequence):
            rate = [rate]

        if not all(isinstance(r, (float, int)) and r > 0 for r in rate):
            raise ValueError(
                f"All rate values must be positive numbers, received {rate}"
            )

        if rate_type == "async":
            rate_type = "constant"  # default to constant if not specified

        return AsyncProfile(
            strategy_type=rate_type,  # type: ignore[arg-type]
            rate=rate,
            random_seed=random_seed,
            **kwargs,
        )


class SweepProfile(AsyncProfile):
    type_: Literal["sweep"] = "sweep"  # type: ignore[assignment]
    sweep_size: int = Field(
        description="The number of strategies to generate for the sweep.",
    )
    rate: float = -1
    rate_type: Literal["constant", "poisson"] = "constant"

    @property
    def strategy_types(self) -> list[StrategyType]:
        return (
            ["synchronous"] + ["throughput"] + [self.rate_type] * (self.sweep_size - 2)  # type: ignore[return-value]
        )

    def next_strategy(self) -> Optional[SchedulingStrategy]:
        if self.completed_strategies >= self.sweep_size:
            return None

        if self.completed_strategies == 0:
            return SynchronousStrategy()

        if self.completed_strategies == 1:
            return ThroughputStrategy(
                max_concurrency=self.max_concurrency,
            )

        min_rate = self.measured_rates[0]
        max_rate = self.measured_rates[1]
        rates = np.linspace(min_rate, max_rate, self.sweep_size - 1)[1:]

        if self.rate_type == "constant":
            return AsyncConstantStrategy(
                rate=rates[self.completed_strategies - 2],
                initial_burst=self.initial_burst,
                max_concurrency=self.max_concurrency,
            )
        elif self.rate_type == "poisson":
            return AsyncPoissonStrategy(
                rate=rates[self.completed_strategies - 2],
                initial_burst=self.initial_burst,
                max_concurrency=self.max_concurrency,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.rate_type}")

    @staticmethod
    def from_standard_args(  # type: ignore[override]
        rate_type: Union[StrategyType, ProfileType],
        rate: Optional[Union[float, Sequence[float]]],
        random_seed: int,
        **kwargs,
    ) -> "SweepProfile":
        if rate_type != "sweep":
            raise ValueError("Rate type must be 'sweep' for sweep profile.")

        if "sweep_size" in kwargs:
            raise ValueError("Sweep size must not be provided, use rate instead.")

        if isinstance(rate, Sequence):
            if len(rate) != 1:
                raise ValueError(
                    "Rate must be a single value for sweep profile, received "
                    f"{len(rate)} values."
                )
            rate = rate[0]

        if not rate:
            rate = settings.default_sweep_number

        if not rate:
            raise ValueError(
                "Rate (sweep_size) must be provided for concurrent profile."
            )

        if (
            not isinstance(rate, (int, float))
            or (isinstance(rate, float) and not rate.is_integer())
            or rate <= 1
        ):
            raise ValueError(
                f"Rate (sweep_size) must be a positive integer > 1, received {rate} "
                f"with type {type(rate)}"
            )

        if not kwargs:
            kwargs = {}

        if "strategy_type" not in kwargs:
            kwargs["strategy_type"] = "constant"

        return SweepProfile(sweep_size=int(rate), random_seed=random_seed, **kwargs)


def create_profile(
    rate_type: Union[StrategyType, ProfileType],
    rate: Optional[Union[float, Sequence[float]]],
    random_seed: int = 42,
    **kwargs,
) -> "Profile":
    if rate_type == "synchronous":
        return SynchronousProfile.from_standard_args(
            rate_type=rate_type,
            rate=rate,
            **kwargs,
        )

    if rate_type == "concurrent":
        return ConcurrentProfile.from_standard_args(
            rate_type=rate_type,
            rate=rate,
            **kwargs,
        )

    if rate_type == "throughput":
        return ThroughputProfile.from_standard_args(
            rate_type=rate_type,
            rate=rate,
            **kwargs,
        )

    if rate_type in ("async", "constant", "poisson"):
        return AsyncProfile.from_standard_args(
            rate_type=rate_type,
            rate=rate,
            random_seed=random_seed,
            **kwargs,
        )

    if rate_type == "sweep":
        return SweepProfile.from_standard_args(
            rate_type=rate_type,
            rate=rate,
            random_seed=random_seed,
            **kwargs,
        )

    raise ValueError(f"Invalid profile type: {rate_type}")
