"""
Configuration entrypoints for embeddings benchmark execution.

Defines parameter schemas for creating embeddings benchmark runs from scenario files
or runtime arguments. Extends standard benchmark configuration with embeddings-specific
options including quality validation settings (baseline model, cosine similarity
tolerance) and MTEB benchmark integration.
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
    field_serializer,
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

__all__ = ["BenchmarkEmbeddingsArgs"]


class BenchmarkEmbeddingsArgs(StandardBaseModel):
    """
    Configuration arguments for embeddings benchmark execution.

    Defines all parameters for embeddings benchmark setup including target endpoint,
    data sources, backend configuration, processing pipeline, output formatting,
    execution constraints, and embeddings-specific quality validation options.

    Example::

        # Basic embeddings benchmark
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000/v1",
            data=["path/to/texts.json"],
            profile="sweep"
        )

        # With quality validation
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000/v1",
            data=["path/to/texts.json"],
            enable_quality_validation=True,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            quality_tolerance=1e-2
        )

        # With MTEB benchmarking
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000/v1",
            data=["path/to/texts.json"],
            enable_mteb=True,
            mteb_tasks=["STS12", "STS13"]
        )
    """

    @classmethod
    def create(
        cls, scenario: Path | str | None, **kwargs: dict[str, Any]
    ) -> BenchmarkEmbeddingsArgs:
        """
        Create benchmark args from scenario file and keyword arguments.

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
                scenario_data = scenario_data["args"]
            constructor_kwargs.update(scenario_data)

        constructor_kwargs.update(kwargs)
        return cls.model_validate(constructor_kwargs)

    @classmethod
    def get_default(cls: type[BenchmarkEmbeddingsArgs], field: str) -> Any:
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
    request_format: str | None = Field(
        default=None,
        description="Query format for backend operations"
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
        | Literal["embeddings_column_mapper"]
    ) = Field(
        default="embeddings_column_mapper",
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
        default="embeddings",
        description="Finalizer for preparing data samples into requests",
    )
    data_collator: Callable | Literal["embeddings"] | None = Field(
        default="embeddings", description="Data collator for batch processing"
    )
    data_sampler: Sampler[int] | Literal["shuffle"] | None = Field(
        default=None, description="Data sampler for request ordering"
    )
    data_num_workers: int | None = Field(
        default=0, description="Number of workers for data loading"
    )
    dataloader_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional dataloader configuration arguments"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Output configuration
    outputs: list[str] | tuple[str] = Field(
        default_factory=lambda: ["json", "csv", "html"],
        description="Output types to create (json, csv, html)",
    )
    output_dir: str | Path = Field(
        default_factory=Path.cwd,
        description="Directory for saving output files",
    )
    output_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional output formatter arguments"
    )

    # Constraint configuration
    max_requests: int | None = Field(
        default=None, description="Maximum number of requests to execute"
    )
    max_errors: int | None = Field(
        default=None, description="Maximum allowed errors before stopping"
    )
    max_duration: float | None = Field(
        default=None, description="Maximum duration in seconds"
    )
    warmup: TransientPhaseConfig | float | int | dict | None = Field(
        default=None, description="Warmup phase configuration"
    )
    cooldown: TransientPhaseConfig | float | int | dict | None = Field(
        default=None, description="Cooldown phase configuration"
    )

    # EMBEDDINGS-SPECIFIC: Quality validation options
    enable_quality_validation: bool = Field(
        default=False,
        description="Enable quality validation against baseline model",
    )
    baseline_model: str | None = Field(
        default=None,
        description=(
            "HuggingFace model for baseline comparison "
            "(e.g., 'sentence-transformers/all-MiniLM-L6-v2')"
        ),
    )
    quality_tolerance: float = Field(
        default=1e-2,
        description=(
            "Cosine similarity tolerance threshold (1e-2 standard, 5e-4 MTEB-level)"
        ),
    )

    # EMBEDDINGS-SPECIFIC: MTEB benchmark options
    enable_mteb: bool = Field(
        default=False,
        description="Enable MTEB benchmark evaluation",
    )
    mteb_tasks: list[str] | None = Field(
        default=None,
        description=(
            "MTEB tasks to evaluate (default: ['STS12', 'STS13', 'STSBenchmark'])"
        ),
    )

    # EMBEDDINGS-SPECIFIC: Encoding format
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Embedding encoding format (float or base64)",
    )

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        """Serialize Path to string for JSON/YAML."""
        return str(value)

    @field_serializer("processor")
    def serialize_processor(self, value: Any) -> str | None:
        """Serialize processor to string representation."""
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            return str(value)
        # For PreTrainedTokenizer instances, return name_or_path
        return getattr(value, "name_or_path", str(value))
