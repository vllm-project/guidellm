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
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
)
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.scenarios import get_builtin_scenarios
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
    data_request_formatter: RequestFormatter | dict[str, str] | str = Field(
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
        default=None, description="Number of workers for data loading"
    )
    dataloader_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional dataloader configuration arguments"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    # Output configuration
    output_path: str | Path | None = Field(
        default_factory=Path.cwd, description="Directory path for output files"
    )
    output_formats: list[str] | dict[str, str | dict[str, Any]] | None = Field(
        default_factory=lambda: ["console", "json", "csv"],
        description="Output format names or configuration mappings",
    )
    # Benchmarker configuration
    sample_requests: int | None = Field(
        default=10,
        description="Number of requests to sample for detailed metrics (None for all)",
    )
    warmup: float | None = Field(
        default=None,
        description="Warmup period in seconds, requests, or fraction (0-1)",
    )
    cooldown: float | None = Field(
        default=None,
        description="Cooldown period in seconds, requests, or fraction (0-1)",
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

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        """
        Convert model to serializable dictionary format.

        Transforms complex types (Backend, Profile, Path, etc.) to JSON-compatible
        primitives while preserving configuration semantics for storage and
        reproduction.

        :return: Dictionary representation for JSON/YAML serialization
        """
        return {
            # target - serialize as is
            "target": self.target,
            "data": [
                item if isinstance(item, str | type(None)) else str(item)
                for item in self.data
            ],  # data - for each item in the list, if not a str or None, save str(item)
            "profile": (
                self.profile.type_
                if isinstance(self.profile, Profile)
                else self.profile
            ),  # profile - if instance of Profile, then save as profile.type_
            "rate": self.rate,
            "backend": (
                self.backend.type_
                if isinstance(self.backend, Backend)
                else self.backend
            ),  # backend - if instance of Backend, then save as backend.type_
            "backend_kwargs": self.backend_kwargs,
            "model": self.model,
            "processor": (
                self.processor
                if isinstance(self.processor, str)
                else str(self.processor)
                if self.processor is not None
                else None
            ),  # processor - if not str, then save as str(processor)
            "processor_args": self.processor_args,
            "data_args": self.data_args,
            "data_samples": self.data_samples,
            "data_column_mapper": (
                self.data_column_mapper
                if isinstance(self.data_column_mapper, dict | str)
                else {}
            ),  # data_column_mapper - if not dict or str, then save as an empty dict
            "data_request_formatter": (
                self.data_request_formatter
                if isinstance(self.data_request_formatter, dict | str)
                else {}
            ),  # data_request_formatter - if not dict or str, then save as empty dict
            "data_collator": (
                self.data_collator if isinstance(self.data_collator, str) else None
            ),  # data_collator - if not str, then save as None
            "data_sampler": (
                self.data_sampler if isinstance(self.data_sampler, str) else None
            ),  # data_sampler - if not str, then save as None
            "data_num_workers": self.data_num_workers,
            "dataloader_kwargs": self.dataloader_kwargs,
            "random_seed": self.random_seed,
            "output_path": (
                str(self.output_path) if self.output_path is not None else None
            ),  # output_path - if not None, then ensure it's a str
            "output_formats": self.output_formats,
            "sample_requests": self.sample_requests,
            "warmup": self.warmup,
            "cooldown": self.cooldown,
            "prefer_response_metrics": self.prefer_response_metrics,
            "max_seconds": self.max_seconds,
            "max_requests": self.max_requests,
            "max_errors": self.max_errors,
            "max_error_rate": self.max_error_rate,
            "max_global_error_rate": self.max_global_error_rate,
        }
