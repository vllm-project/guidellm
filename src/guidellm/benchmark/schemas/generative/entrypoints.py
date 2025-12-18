"""
Configuration entrypoints for generative text benchmark execution.

Defines parameter schemas and construction logic for creating benchmark runs from
scenario files or runtime arguments. Provides flexible configuration loading with
support for built-in scenarios, custom YAML/JSON files, and programmatic overrides.
Handles serialization of complex types including backends, processors, and profiles
for persistent storage and reproduction of benchmark configurations.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    AliasChoices,
    AliasGenerator,
    ConfigDict,
    Field,
    NonNegativeFloat,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
)
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.profiles import Profile, ProfileType
from guidellm.benchmark.scenarios import get_builtin_scenarios
from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.data import DatasetPreprocessor, RequestFormatter
from guidellm.scheduler import StrategyType
from guidellm.schemas import StandardBaseModel

__all__ = ["BenchmarkGenerativeTextArgs"]


class BenchmarkGenerativeTextArgs(StandardBaseModel):
    """
    Configuration arguments for generative text benchmark execution.

    Defines all parameters for benchmark setup including target endpoint, data
    sources, backend configuration, processing pipeline, output formatting, and
    execution constraints. Supports loading from scenario files and merging with
    runtime overrides for flexible benchmark construction from multiple sources.

    Example::

        # Load from built-in scenario with overrides
        args = BenchmarkGenerativeTextArgs.create(
            scenario="chat",
            target="http://localhost:8000/v1",
            max_requests=1000
        )

        # Create from keyword arguments only
        args = BenchmarkGenerativeTextArgs(
            target="http://localhost:8000/v1",
            data=["path/to/dataset.json"],
            profile="fixed",
            rate=10.0
        )
    """

    @classmethod
    def create(
        cls, scenario: Path | str | None, **kwargs: dict[str, Any]
    ) -> BenchmarkGenerativeTextArgs:
        """
        Create benchmark args from scenario file and keyword arguments.

        Loads base configuration from scenario file (built-in or custom) and merges
        with provided keyword arguments. Arguments explicitly set via kwargs override
        scenario values, while defaulted kwargs are ignored to preserve scenario
        settings.

        :param scenario: Path to scenario file, built-in scenario name, or None
        :param kwargs: Keyword arguments to override scenario values
        :return: Configured benchmark args instance
        :raises ValueError: If scenario is not found or file format is unsupported
        """
        constructor_kwargs = {}

        if scenario is not None:
            if isinstance(scenario, str) and scenario in (
                builtin_scenarios := get_builtin_scenarios()
            ):
                scenario_path = builtin_scenarios[scenario]
            elif Path(scenario).exists() and Path(scenario).is_file():
                scenario_path = Path(scenario)
            else:
                raise ValueError(f"Scenario '{scenario}' not found.")

            with scenario_path.open() as file:
                if scenario_path.suffix == ".json":
                    scenario_data = json.load(file)
                elif scenario_path.suffix in {".yaml", ".yml"}:
                    scenario_data = yaml.safe_load(file)
                else:
                    raise ValueError(
                        f"Unsupported scenario file format: {scenario_path.suffix}"
                    )
            if "args" in scenario_data:
                # loading from a report file
                scenario_data = scenario_data["args"]
            constructor_kwargs.update(scenario_data)

        # Apply overrides from kwargs
        constructor_kwargs.update(kwargs)

        return cls.model_validate(constructor_kwargs)

    @classmethod
    def get_default(cls: type[BenchmarkGenerativeTextArgs], field: str) -> Any:
        """
        Retrieve default value for a model field.

        Extracts the default value from field metadata, handling both static defaults
        and factory functions.

        :param field: Field name to retrieve default value for
        :return: Default value for the field
        :raises ValueError: If field does not exist
        """
        if field not in cls.model_fields:
            raise ValueError(f"Field '{field}' not found in {cls.__name__}")

        field_info = cls.model_fields[field]
        factory = field_info.default_factory

        if factory is None:
            return field_info.default

        if len(inspect.signature(factory).parameters) == 0:
            return factory()  # type: ignore[call-arg]
        else:
            return factory({})  # type: ignore[call-arg]

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        validate_by_alias=True,
        validate_by_name=True,
        alias_generator=AliasGenerator(
            # Support field names with hyphens
            validation_alias=lambda field_name: AliasChoices(
                field_name, field_name.replace("_", "-")
            ),
        ),
    )

    # Required
    target: str = Field(description="Target endpoint URL for benchmark execution")
    data: list[Any] = Field(
        description="List of dataset sources or data files",
        default_factory=list,
        min_length=1,
    )
    # Benchmark configuration
    profile: StrategyType | ProfileType | Profile = Field(
        default="sweep", description="Benchmark profile or scheduling strategy type"
    )
    rate: list[float] | None = Field(
        default=None, description="Request rate(s) for rate-based scheduling"
    )
    # Backend configuration
    backend: BackendType | Backend = Field(
        default="openai_http", description="Backend type or instance for execution"
    )
    backend_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional backend configuration arguments"
    )
    model: str | None = Field(default=None, description="Model identifier for backend")
    # Data configuration
    processor: str | Path | PreTrainedTokenizerBase | None = Field(
        default=None, description="Tokenizer path, name, or instance for processing"
    )
    processor_args: dict[str, Any] | None = Field(
        default=None, description="Additional tokenizer configuration arguments"
    )
    data_args: list[dict[str, Any]] | None = Field(
        default_factory=list,  # type: ignore[arg-type]
        description="Per-dataset configuration arguments",
    )
    data_samples: int = Field(
        default=-1, description="Number of samples to use from datasets (-1 for all)"
    )
    data_column_mapper: (
        DatasetPreprocessor
        | dict[str, str | list[str]]
        | Literal["generative_column_mapper"]
    ) = Field(
        default="generative_column_mapper",
        description="Column mapping preprocessor for dataset fields",
    )
    data_request_formatter: RequestFormatter | dict[str, Any] | str = Field(
        default="chat_completions",
        description="Request formatting preprocessor or template name",
        validation_alias=AliasChoices(
            "data_request_formatter",
            "data-request-formatter",
            "request_type",
            "request-type",
        ),
    )
    data_collator: Callable | Literal["generative"] | None = Field(
        default="generative", description="Data collator for batch processing"
    )
    data_sampler: Sampler[int] | Literal["shuffle"] | None = Field(
        default=None, description="Data sampler for request ordering"
    )
    data_num_workers: int | None = Field(
        default=1, description="Number of workers for data loading"
    )
    dataloader_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional dataloader configuration arguments"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    # Output configuration
    outputs: list[str] | tuple[str] = Field(
        default_factory=lambda: ["json", "csv", "html"],
        description=(
            "The aliases of the output types to create with their default filenames "
            "the file names and extensions of the output types to create"
        ),
    )
    output_dir: str | Path = Field(
        default_factory=Path.cwd,
        description="The directory path to save file output types in",
    )
    # Benchmarker configuration
    sample_requests: int | None = Field(
        default=10,
        description="Number of requests to sample for detailed metrics (None for all)",
    )
    warmup: int | float | dict | TransientPhaseConfig | None = Field(
        default=None,
        description=(
            "Warmup phase config: time or requests before measurement starts "
            "(overlapping requests count toward measurement)"
        ),
    )
    cooldown: int | float | dict | TransientPhaseConfig | None = Field(
        default=None,
        description=(
            "Cooldown phase config: time or requests after measurement ends "
            "(overlapping requests count toward measurement)"
        ),
    )
    rampup: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "The time, in seconds, to ramp up the request rate over. "
            "Only applicable for Throughput/Concurrent strategies"
        ),
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description="Whether to prefer backend response metrics over request metrics",
    )
    # Constraints configuration
    max_seconds: int | float | None = Field(
        default=None, description="Maximum benchmark execution time in seconds"
    )
    max_requests: int | None = Field(
        default=None, description="Maximum number of requests to execute"
    )
    max_errors: int | None = Field(
        default=None, description="Maximum number of errors before stopping"
    )
    max_error_rate: float | None = Field(
        default=None, description="Maximum error rate (0-1) before stopping"
    )
    max_global_error_rate: float | None = Field(
        default=None, description="Maximum global error rate (0-1) before stopping"
    )
    over_saturation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Over-saturation detection configuration. A dict with configuration "
            "parameters (enabled, min_seconds, max_window_seconds, "
            "moe_threshold, etc.)."
        ),
    )

    @field_validator("data", "data_args", "rate", mode="wrap")
    @classmethod
    def single_to_list(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> list[Any]:
        """
        Ensures field is always a list.

        :param value: Input value for the 'data' field
        :return: List of data sources
        """
        try:
            return handler(value)
        except ValidationError as err:
            # If validation fails, try wrapping the value in a list
            if err.errors()[0]["type"] == "list_type":
                return handler([value])
            else:
                raise

    @field_serializer("backend")
    def serialize_backend(self, backend: BackendType | Backend) -> str:
        """Serialize backend to type string."""
        return backend.type_ if isinstance(backend, Backend) else backend

    @field_serializer("data")
    def serialize_data(self, data: list[Any]) -> list[str | None]:
        """Serialize data items to strings."""
        return [
            item if isinstance(item, str | type(None)) else str(item) for item in data
        ]

    @field_serializer("data_collator")
    def serialize_data_collator(
        self, data_collator: Callable | Literal["generative"] | None
    ) -> str | None:
        """Serialize data_collator to string or None."""
        return data_collator if isinstance(data_collator, str) else None

    @field_serializer("data_column_mapper")
    def serialize_data_column_mapper(
        self,
        data_column_mapper: (
            DatasetPreprocessor
            | dict[str, str | list[str]]
            | Literal["generative_column_mapper"]
        ),
    ) -> dict | str:
        """Serialize data_column_mapper to dict or string."""
        return data_column_mapper if isinstance(data_column_mapper, dict | str) else {}

    @field_serializer("data_request_formatter")
    def serialize_data_request_formatter(
        self, data_request_formatter: RequestFormatter | dict[str, Any] | str
    ) -> dict | str:
        """Serialize data_request_formatter to dict or string."""
        return (
            data_request_formatter
            if isinstance(data_request_formatter, dict | str)
            else {}
        )

    @field_serializer("data_sampler")
    def serialize_data_sampler(
        self, data_sampler: Sampler[int] | Literal["shuffle"] | None
    ) -> str | None:
        """Serialize data_sampler to string or None."""
        return data_sampler if isinstance(data_sampler, str) else None

    @field_serializer("output_dir")
    def serialize_output_dir(self, output_dir: str | Path) -> str | None:
        """Serialize output_dir to string."""
        return str(output_dir) if output_dir is not None else None

    @field_serializer("processor")
    def serialize_processor(
        self, processor: str | Path | PreTrainedTokenizerBase | None
    ) -> str | None:
        """Serialize processor to string."""
        if processor is None:
            return None
        return processor if isinstance(processor, str) else str(processor)

    @field_serializer("profile")
    def serialize_profile(self, profile: StrategyType | ProfileType | Profile) -> str:
        """Serialize profile to type string."""
        return profile.type_ if isinstance(profile, Profile) else profile
