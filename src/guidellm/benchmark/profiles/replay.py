"""Trace replay benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.data import DataArgs
from guidellm.data.deserializers import TraceSyntheticDataArgs
from guidellm.scheduler import (
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
    TraceReplayStrategy,
)
from guidellm.scheduler.constraints.request import MaxRequestsConstraintArgs
from guidellm.utils.trace_io import load_relative_timestamps

from .profile import Profile, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


def _normalize_data_args(data: list[Any]) -> list[DataArgs]:
    """
    Coerce serialized data entries back into validated ``DataArgs`` instances.

    :param data: Data sources as models or serialized mappings from ``model_dump``
    :return: Validated data argument instances
    :raises ValueError: If an entry is not a supported data configuration type
    """
    normalized: list[DataArgs] = []
    for item in data:
        if isinstance(item, DataArgs):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(DataArgs.model_validate(item))
        else:
            raise ValueError(
                "ReplayProfile data entries must be DataArgs instances or mappings, "
                f"got {type(item).__name__}"
            )
    return normalized


def _resolve_relative_timestamps(
    data: list[Any],
    data_samples: int,
) -> list[float]:
    """
    Load and optionally truncate relative timestamps from a trace data source.

    :param data: Data sources configured for the replay profile
    :param data_samples: Maximum number of timestamps to retain; non-positive keeps all
    :return: Sorted timestamps relative to the first event
    :raises ValueError: If data configuration is invalid or yields no timestamps
    """
    data_args = _normalize_data_args(data)
    if len(data_args) != 1:
        raise ValueError(
            f"ReplayProfile requires exactly one data source, received {len(data_args)}"
        )
    config = data_args[0]
    if not isinstance(config, TraceSyntheticDataArgs):
        raise ValueError("ReplayProfile requires a trace data source")
    if not (path := Path(config.path)).exists():
        raise ValueError(f"Replay trace file not found: {config.path}")

    relative_timestamps = load_relative_timestamps(path, config.timestamp_column)
    if data_samples > 0:
        relative_timestamps = relative_timestamps[:data_samples]

    if not relative_timestamps:
        raise ValueError(
            "No timestamps remain after applying data_samples. "
            "The trace is empty or all events were filtered out."
        )

    return relative_timestamps


@ProfileArgs.register("replay")
class ReplayProfileArgs(ProfileArgs):
    """Pydantic model for replay profile creation arguments."""

    kind: Literal["replay"] = Field(
        default="replay",
        description="Profile type discriminator for polymorphic serialization",
    )
    time_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor applied to relative timestamps",
    )


@ProfileFactory.register("replay")
class ReplayProfile(Profile):
    """
    Replay a trace file using per-row ``relative_timestamp`` from the dataset.

    Each request is scheduled at
    ``start_time + time_scale * relative_timestamp`` via ``RequestSettings`` on
    the dataset finalizer output. For this profile, ``rate`` is interpreted as
    ``time_scale`` (not requests per second).

    When ``data_samples`` is set, the default ``max_requests`` constraint matches
    the truncated dataset size.
    """

    args: ReplayProfileArgs

    def __init__(
        self,
        args: ReplayProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args

        data = kwargs.get("data", [])
        data_samples = kwargs.get("data_samples", -1)
        relative_timestamps = _resolve_relative_timestamps(data, data_samples)
        new_constraints = dict(constraints or {})
        if "max_requests" not in new_constraints:
            constraint_args = MaxRequestsConstraintArgs(count=len(relative_timestamps))
            new_constraints["max_requests"] = ConstraintsInitializerFactory.create(
                constraint_args
            )
            self.constraints = new_constraints

    @property
    def strategy_types(self) -> list[str]:
        return ["trace"]

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> TraceReplayStrategy | None:
        _ = prev_benchmark
        # Replay has a single strategy; return it once, then None
        if prev_strategy is not None:
            return None
        return TraceReplayStrategy(time_scale=self.args.time_scale)
