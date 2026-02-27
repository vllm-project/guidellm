"""
Base configuration arguments shared across benchmark types.

Provides common configuration fields and methods used by both embeddings
and generative text benchmarks to reduce code duplication and ensure
consistent configuration patterns.
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
    field_serializer,
    field_validator,
)
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.profiles import Profile, ProfileType
from guidellm.benchmark.scenarios import get_builtin_scenarios
from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.data import DatasetFinalizer, DatasetPreprocessor
from guidellm.scheduler import StrategyType
from guidellm.schemas import StandardBaseModel

__all__ = ["BaseBenchmarkArgs"]


class BaseBenchmarkArgs(StandardBaseModel):
    """
    Base configuration arguments shared across benchmark types.

    Defines common parameters for benchmark setup including target endpoint,
    data sources, backend configuration, processing pipeline, output
    formatting, and execution constraints. Subclasses add domain-specific
    fields (embeddings vs generative text).
    """

    @classmethod
    def create(
        cls, scenario: Path | str | None, **kwargs: dict[str, Any]
    ) -> BaseBenchmarkArgs:
        """
        Create benchmark args from scenario file and keyword arguments.

        :param scenario: Path to scenario file, built-in scenario name,
            or None
        :param kwargs: Keyword arguments to override scenario values
        :return: Configured benchmark args instance
        :raises ValueError: If scenario is not found or file format is
            unsupported
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
                scenario_data = scenario_data["args"]
            constructor_kwargs.update(scenario_data)

        constructor_kwargs.update(kwargs)
        return cls.model_validate(constructor_kwargs)

    @classmethod
    def get_default(cls, field: str) -> Any:
        """
        Retrieve default value for a model field.

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
        default="sweep",
        description="Benchmark profile or scheduling strategy type",
    )
    rate: list[float] | None = Field(
        default=None,
        description="Request rate(s) for rate-based scheduling",
    )

    # Backend configuration
    backend: BackendType | Backend = Field(
        default="openai_http",
        description="Backend type or instance for execution",
    )
    backend_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional backend configuration arguments",
    )
    request_format: str | None = Field(
        default=None,
        description="Query format for backend operations",
    )
    model: str | None = Field(
        default=None,
        description="Model identifier for backend",
    )

    # Data configuration
    processor: str | Path | PreTrainedTokenizerBase | None = Field(
        default=None,
        description="Tokenizer path, name, or instance for processing",
    )
    processor_args: dict[str, Any] | None = Field(
        default=None,
        description="Additional tokenizer configuration arguments",
    )
    data_args: list[dict[str, Any]] | None = Field(
        default_factory=list,  # type: ignore[arg-type]
        description="Per-dataset configuration arguments",
    )
    data_samples: int = Field(
        default=-1,
        description="Number of samples to use from datasets (-1 for all)",
    )
    data_column_mapper: DatasetPreprocessor | dict[str, str | list[str]] | str = Field(
        default="generative_column_mapper",
        description="Column mapping preprocessor for dataset fields",
    )
    data_preprocessors: list[DatasetPreprocessor | dict[str, str | list[str]] | str] = (
        Field(
            default_factory=list,  # type: ignore[arg-type]
            description="List of dataset preprocessors to apply in order",
        )
    )
    data_preprocessors_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Global arguments for data preprocessors",
    )
    data_finalizer: DatasetFinalizer | str | dict[str, Any] = Field(
        default="generative",
        description="Finalizer for preparing data samples into requests",
    )
    data_collator: Callable | str | None = Field(
        default="generative",
        description="Data collator for batch processing",
    )
    data_sampler: Sampler[int] | Literal["shuffle"] | None = Field(
        default=None,
        description="Data sampler for request ordering",
    )
    data_num_workers: int | None = Field(
        default=1,
        description="Number of workers for data loading",
    )
    dataloader_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional dataloader configuration arguments",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # Output configuration
    outputs: list[str] | tuple[str] = Field(
        default_factory=lambda: ["json"],
        description="Output types to create (json, yaml, etc.)",
    )
    output_dir: str | Path = Field(
        default_factory=Path.cwd,
        description="Directory for saving output files",
    )

    # Constraint configuration
    max_requests: int | None = Field(
        default=None,
        description="Maximum number of requests to execute",
    )
    max_errors: int | None = Field(
        default=None,
        description="Maximum allowed errors before stopping",
    )
    warmup: TransientPhaseConfig | float | int | dict | None = Field(
        default=None,
        description="Warmup phase configuration",
    )
    cooldown: TransientPhaseConfig | float | int | dict | None = Field(
        default=None,
        description="Cooldown phase configuration",
    )

    @field_validator("data", "data_args", "rate", "data_preprocessors", mode="wrap")
    @classmethod
    def single_to_list(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> list[Any]:
        """
        Ensures field is always a list.

        :param value: Input value for the field
        :param handler: Validation handler
        :return: List value
        """
        try:
            return handler(value)
        except ValidationError as err:
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
            (item if isinstance(item, str | type(None)) else str(item)) for item in data
        ]

    @field_serializer("data_collator")
    def serialize_data_collator(
        self, data_collator: Callable | str | None
    ) -> str | None:
        """Serialize data_collator to string or None."""
        return data_collator if isinstance(data_collator, str) else None

    @field_serializer("data_column_mapper")
    def serialize_preprocessor(
        self,
        data_preprocessor: DatasetPreprocessor | dict[str, str | list[str]] | str,
    ) -> dict | str:
        """Serialize a preprocessor to dict or string."""
        return data_preprocessor if isinstance(data_preprocessor, dict | str) else {}

    @field_serializer("data_preprocessors")
    def serialize_preprocessors(
        self,
        data_preprocessors: list[
            DatasetPreprocessor | dict[str, str | list[str]] | str
        ],
    ) -> list[dict | str]:
        """Serialize each preprocessor to dict or string."""
        return [self.serialize_preprocessor(p) for p in data_preprocessors]

    @field_serializer("data_finalizer")
    def serialize_data_finalizer(
        self, data_finalizer: DatasetFinalizer | dict[str, Any] | str
    ) -> dict | str:
        """Serialize data_finalizer to dict or string."""
        return data_finalizer if isinstance(data_finalizer, dict | str) else {}

    @field_serializer("data_sampler")
    def serialize_data_sampler(
        self, data_sampler: Sampler[int] | Literal["shuffle"] | None
    ) -> str | None:
        """Serialize data_sampler to string or None."""
        return data_sampler if isinstance(data_sampler, str) else None

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        """Serialize Path to string for JSON/YAML."""
        return str(value)

    @field_serializer("processor")
    def serialize_processor(self, value: Any) -> str | None:
        """Serialize processor to string representation."""
        if value is None:
            return None
        if isinstance(value, str | Path):
            return str(value)
        # For PreTrainedTokenizer instances, return name_or_path
        return getattr(value, "name_or_path", str(value))

    @field_serializer("profile")
    def serialize_profile(self, profile: StrategyType | ProfileType | Profile) -> str:
        """Serialize profile to type string."""
        return profile.type_ if isinstance(profile, Profile) else profile
