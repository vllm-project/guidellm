"""
Configuration entrypoints for generative text benchmark execution.

Defines parameter schemas and construction logic for creating benchmark runs from
scenario files or runtime arguments. Provides flexible configuration loading with
support for built-in scenarios, custom YAML/JSON files, and programmatic overrides.
Handles serialization of complex types including backends, processors, and profiles
for persistent storage and reproduction of benchmark configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    AliasChoices,
    AliasGenerator,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.utils import deep_update
from pydantic_settings import BaseSettings, SettingsConfigDict

from guidellm.backends import BackendArgs
from guidellm.benchmark.scenarios import get_builtin_scenarios
from guidellm.benchmark.schemas.output import BenchmarkOutputArgs
from guidellm.benchmark.schemas.profiles import ProfileArgs
from guidellm.benchmark.schemas.random import RandomArgs
from guidellm.data import (
    DataArgs,
    DataFinalizerArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
    DataTokenizerArgs,
)
from guidellm.scheduler.constraints import ConstraintArgs
from guidellm.schemas import StandardBaseModel, standard_model_config
from guidellm.utils.arg_string import ArgStringParser

__all__ = [
    "BenchmarkArgs",
    "BenchmarkMetadata",
    "BenchmarkScenario",
]


def args_model_config() -> ConfigDict:
    return standard_model_config(
        extra="forbid",
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        alias_generator=AliasGenerator(
            # Support field names with hyphens
            validation_alias=lambda field_name: AliasChoices(
                field_name, field_name.replace("_", "-")
            ),
        ),
    )


def default_kind(kind: str) -> dict[str, Any]:
    """Default factory for argument models to set the 'kind' field."""
    return {"kind": kind}


def default_kind_list(*kinds: str) -> list[dict[str, Any]]:
    """Default factory for lists of argument models to set the 'kind' field."""
    return [default_kind(kind) for kind in kinds]


class BenchmarkArgs(StandardBaseModel):
    """Common benchmark configuration arguments."""

    model_config = args_model_config()

    backend: BackendArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("openai_http"),
        description="Backend configuration arguments",
        json_schema_extra={"argument_alias": "backend"},
    )
    profile: ProfileArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("sweep"),
        description="Benchmark profile configuration arguments",
        json_schema_extra={"argument_alias": "profile"},
    )
    constraints: list[ConstraintArgs] = Field(  # type: ignore[assignment]
        description="List of constraints to enforce during benchmark execution",
        default_factory=list,
        json_schema_extra={"argument_alias": "constraint"},
    )
    tokenizer: DataTokenizerArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("huggingface_auto"),
        description="Tokenizer configuration arguments",
        json_schema_extra={"argument_alias": "tokenizer"},
    )
    data: list[DataArgs] = Field(  # type: ignore[assignment]
        description="List of dataset sources or data files",
        min_length=1,
        json_schema_extra={"argument_alias": "data"},
    )
    data_column_mapper: DataPreprocessorArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative_column_mapper"),
        description="Column mapping preprocessor for dataset fields",
        json_schema_extra={"argument_alias": "data_column_mapper"},
    )
    data_preprocessors: list[DataPreprocessorArgs] = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind_list("encode_media"),  # type: ignore[arg-type]
        description="List of dataset preprocessors to apply in order",
        json_schema_extra={"argument_alias": "data_preprocessor"},
    )
    data_finalizer: DataFinalizerArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative"),
        description="Finalizer for preparing data samples into requests",
        json_schema_extra={"argument_alias": "data_finalizer"},
    )
    data_loader: DataLoaderArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("pytorch"),
        description="Dataloader configuration arguments",
        json_schema_extra={"argument_alias": "data_loader"},
    )
    seed: RandomArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("static"),
        description="Random configuration for reproducibility (e.g., seed value)",
        json_schema_extra={"argument_alias": "seed"},
    )
    outputs: list[BenchmarkOutputArgs] = Field(
        default_factory=list,
        description="Benchmark outputs",
        json_schema_extra={"argument_alias": "output"},
    )


class BenchmarkMetadata(StandardBaseModel):
    """
    Metadata about the benchmark scenario.

    Contains information such as name, description, and tags that describe the
    benchmark scenario. This metadata is used for reporting and organizational
    purposes but does not affect benchmark execution.
    """

    model_config = args_model_config()

    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs of metadata labels describing the benchmark.",
    )


class BenchmarkScenario(BaseSettings):
    """
    Configuration arguments for generative text benchmark execution.

    Defines all parameters for benchmark setup including target endpoint, data
    sources, backend configuration, processing pipeline, output formatting, and
    execution constraints. Supports loading from scenario files and merging with
    runtime overrides for flexible benchmark construction from multiple sources.

    Example::

        # Load from built-in scenario with overrides
        args = BenchmarkScenario.create(
            scenario="chat",
            spec={"backend": {"kind": "openai_http", "target": "http://localhost:8000/v1"}},
        )

        # Create from keyword arguments only
        args = BenchmarkScenario(
            spec=BenchmarkArgs(
                backend={"kind": "openai_http", "target": "http://localhost:8000/v1"},
                data=[{"kind": "synthetic_text"}],
            ),
        )
    """

    model_config = SettingsConfigDict(
        env_prefix="GUIDELLM_",
        env_nested_delimiter="_",
        validate_default=True,
    )

    @classmethod
    def create(
        cls, scenario: Path | str | None, **kwargs: dict[str, Any]
    ) -> BenchmarkScenario:
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

        # NOTE In the future replace deep_update with a more intelligent merging
        #      strategy that accounts for changes to `kind`.
        # Apply overrides from kwargs
        constructor_kwargs = deep_update(constructor_kwargs, kwargs)

        return cls.model_validate(constructor_kwargs)

    def get_benchmarks(self) -> list[BenchmarkArgs]:
        """
        Get list of benchmark argument instances for each individual benchmark.

        Combines global arguments with individual benchmark overrides to produce a
        list of fully configured benchmark argument instances for execution.

        :return: List of benchmark argument instances
        """
        parser = ArgStringParser(allow_overwrite=True)
        benchmarks = []
        for benchmark_override in self.benchmarks:
            if benchmark_override is None:
                benchmarks.append(self.spec.model_copy(deep=True))
            else:
                # Create a copy of the common args to apply overrides to
                benchmark_args = self.spec.model_dump(mode="python")
                for key, value in benchmark_override.items():
                    parser.set(benchmark_args, key, value)
                benchmarks.append(BenchmarkArgs.model_validate(benchmark_args))

        return benchmarks

    metadata: BenchmarkMetadata = Field(
        default_factory=BenchmarkMetadata,
        description="Metadata describing the benchmark scenario.",
    )
    spec: BenchmarkArgs = Field(
        default_factory=BenchmarkArgs,  # type: ignore[arg-type]
        description="Global configuration parameters for benchmark execution.",
    )
    benchmarks: list[dict[str, Any] | None] = Field(
        default_factory=lambda: [None],  # type: ignore[arg-type]
        description="Individual benchmark overrides",
        min_length=1,
    )

    @model_validator(mode="before")
    @classmethod
    def insert_first_benchmark(cls, data: Any) -> Any:
        """
        Inserts the first benchmark into the common args.

        This allows users to ommit fields from the common args if they have overrides
        in the first benchmark.
        """
        if not isinstance(data, dict):
            return data

        if "benchmarks" not in data or not data["benchmarks"]:
            # No benchmarks provided, insert a blank one
            data["benchmarks"] = [None]

        first_benchmark: dict[str, Any] | None = data["benchmarks"][0]
        if isinstance(first_benchmark, dict) and first_benchmark:
            # Ensure "spec" field exists for the parser to insert into
            data["spec"] = data.get("spec", {})
            parser = ArgStringParser(allow_overwrite=True)

            # Insert the first benchmark into the common args
            # Create fields recursively.
            for key, value in first_benchmark.items():
                parser.set(data["spec"], key, value)

        return data
