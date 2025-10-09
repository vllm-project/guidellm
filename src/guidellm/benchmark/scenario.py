from __future__ import annotations

import json
from functools import cache, wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar

import yaml
from loguru import logger
from pydantic import BeforeValidator, Field, PositiveFloat, PositiveInt, SkipValidation

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.types import AggregatorInputT, DataInputT, ProcessorInputT
from guidellm.scheduler import StrategyType
from guidellm.utils import StandardBaseModel

__all__ = [
    "GenerativeTextScenario",
    "Scenario",
    "enable_scenarios",
    "get_builtin_scenarios",
]

SCENARIO_DIR = Path(__file__).parent / "scenarios/"


@cache
def get_builtin_scenarios() -> list[str]:
    """Returns list of builtin scenario names."""
    return [p.stem for p in SCENARIO_DIR.glob("*.json")]


def parse_float_list(value: str | float | list[float]) -> list[float]:
    """
    Parse a comma separated string to a list of float
    or convert single float list of one or pass float
    list through.
    """
    if isinstance(value, int | float):
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
    def get_default(cls: type[T], field: str) -> Any:
        """Get default values for model fields"""
        return cls.model_fields[field].default

    @classmethod
    def from_file(cls: type[T], filename: Path, overrides: dict | None = None) -> T:
        """
        Attempt to create a new instance of the model using
        data loaded from json or yaml file.
        """
        try:
            with filename.open() as f:
                if str(filename).endswith(".json"):
                    data = json.load(f)
                else:  # Assume everything else is yaml
                    data = yaml.safe_load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Failed to parse {filename} as type {cls.__name__}")
            raise ValueError(f"Error when parsing file: {filename}") from e

        data.update(overrides or {})
        return cls.model_validate(data)

    @classmethod
    def from_builtin(cls: type[T], name: str, overrides: dict | None = None) -> T:
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

    data: Annotated[
        DataInputT,
        # BUG: See https://github.com/pydantic/pydantic/issues/9541
        SkipValidation,
    ]
    profile: StrategyType | ProfileType | Profile
    rate: Annotated[list[PositiveFloat] | None, BeforeValidator(parse_float_list)] = (
        None
    )
    random_seed: int = 42
    # Backend configuration
    backend: BackendType | Backend = "openai_http"
    backend_kwargs: dict[str, Any] | None = None
    model: str | None = None
    # Data configuration
    processor: ProcessorInputT | None = None
    processor_args: dict[str, Any] | None = None
    data_args: dict[str, Any] | None = None
    data_sampler: Literal["random"] | None = None
    # Aggregators configuration
    add_aggregators: AggregatorInputT | None = None
    warmup: Annotated[float | None, Field(gt=0, le=1)] = None
    cooldown: Annotated[float | None, Field(gt=0, le=1)] = None
    request_samples: PositiveInt | None = 20
    # Constraints configuration
    max_seconds: PositiveFloat | PositiveInt | None = None
    max_requests: PositiveInt | None = None
    max_errors: PositiveInt | None = None
    max_error_rate: PositiveFloat | None = None
    max_global_error_rate: PositiveFloat | None = None


# Decorator function to apply scenario to a function
def enable_scenarios(func: Callable) -> Any:
    @wraps(func)
    async def decorator(*args, scenario: Scenario | None = None, **kwargs) -> Any:
        if scenario is not None:
            kwargs.update(scenario.model_dump())
        return await func(*args, **kwargs)

    # Modify the signature of the decorator to include the `scenario` argument
    sig = signature(func)
    params = list(sig.parameters.values())
    # Place `scenario` before `**kwargs` or any parameter with a default value
    loc = next(
        (
            i
            for i, p in enumerate(params)
            if p.kind is Parameter.VAR_KEYWORD or p.default is not Parameter.empty
        ),
        len(params),
    )
    params.insert(
        loc,
        Parameter(
            "scenario",
            Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=Scenario | None,
        ),
    )
    decorator.__signature__ = sig.replace(parameters=params)  # type: ignore [attr-defined]

    return decorator
