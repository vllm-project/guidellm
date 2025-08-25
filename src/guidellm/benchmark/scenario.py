from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypeVar, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import BeforeValidator, Field, NonNegativeInt, PositiveFloat, PositiveInt
from transformers.tokenization_utils_base import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend.backend import BackendType
from guidellm.benchmark.profile import ProfileType
from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler.strategy import StrategyType

__ALL__ = ["Scenario", "GenerativeTextScenario", "get_builtin_scenarios"]

SCENARIO_DIR = Path(__file__).parent / "scenarios/"


@cache
def get_builtin_scenarios() -> list[str]:
    """Returns list of builtin scenario names."""
    return [p.stem for p in SCENARIO_DIR.glob("*.json")]


def parse_float_list(value: Union[str, float, list[float]]) -> list[float]:
    """
    Parse a comma separated string to a list of float
    or convert single float list of one or pass float
    list through.
    """
    if isinstance(value, (int, float)):
        return [value]
    elif isinstance(value, list):
        return value

    values = value.split(",") if "," in value else [value]

    try:
        return [float(val) for val in values]
    except ValueError as err:
        raise ValueError(
            "must be a number or comma-separated list of numbers."
        ) from err


T = TypeVar("T", bound="Scenario")


class Scenario(StandardBaseModel):
    """
    Parent Scenario class with common options for all benchmarking types.
    """

    target: str

    @classmethod
    def from_builtin(cls: type[T], name: str, overrides: Optional[dict] = None) -> T:
        filename = SCENARIO_DIR / f"{name}.json"

        if not filename.is_file():
            raise ValueError(f"{name} is not a valid builtin scenario")

        return cls.from_file(filename, overrides)


class GenerativeTextScenario(Scenario):
    """
    Scenario class for generative text benchmarks.
    """

    class Config:
        # NOTE: This prevents errors due to unvalidatable
        # types like PreTrainedTokenizerBase
        arbitrary_types_allowed = True

    backend_type: BackendType = "openai_http"
    backend_args: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    processor: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None
    processor_args: Optional[dict[str, Any]] = None
    data: Union[
        str,
        Path,
        Iterable[Union[str, dict[str, Any]]],
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    ]
    data_args: Optional[dict[str, Any]] = None
    data_sampler: Optional[Literal["random"]] = None
    rate_type: Union[StrategyType, ProfileType]
    rate: Annotated[
        Optional[list[PositiveFloat]], BeforeValidator(parse_float_list)
    ] = None
    start_rate: Optional[PositiveFloat] = None
    increment_factor: Optional[PositiveFloat] = None
    rate_limit: Optional[PositiveInt] = None
    max_seconds: Optional[PositiveFloat] = None
    max_requests: Optional[PositiveInt] = None
    warmup_percent: Annotated[Optional[float], Field(gt=0, le=1)] = None
    cooldown_percent: Annotated[Optional[float], Field(gt=0, le=1)] = None
    output_sampling: Optional[NonNegativeInt] = None
    random_seed: int = 42
