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
from abc import ABC
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import (
    AliasChoices,
    AliasGenerator,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from guidellm.schemas import (
    PydanticClassRegistryMixin,
    ReloadableBaseModel,
    StandardBaseModel,
    standard_model_config,
)
from guidellm.schemas.backends import BackendArgs
from guidellm.schemas.benchmark.collectors import CollectorArgs
from guidellm.schemas.benchmark.goodput import GoodputSLO
from guidellm.schemas.benchmark.outputs import BenchmarkOutputArgs
from guidellm.schemas.benchmark.profiles import ProfileArgs
from guidellm.schemas.benchmark.random import RandomArgs
from guidellm.schemas.benchmark.scenarios import get_builtin_scenarios
from guidellm.schemas.benchmark.warnings import (
    WarningRuleArgs,
    default_warning_rules,
)
from guidellm.schemas.data import (
    DataArgs,
    DataFinalizerArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
    DataTokenizerArgs,
)
from guidellm.schemas.scheduler import ConstraintArgs
from guidellm.utils.arg_string import ArgStringParser
from guidellm.utils.typing import BLANK

__all__ = [
    "BenchmarkArgs",
    "BenchmarkMetadata",
    "BenchmarkScenario",
    "GenerativeMetricsArgs",
    "MetricsArgs",
    "merge_scenario_files",
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


def _merge_config_values(base: Any, override: Any) -> Any:
    """Merge configuration values, replacing incompatible registry objects."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)

    # A registry object is identified by its ``kind`` discriminator. A change
    # in kind means that the old object's fields are no longer applicable. When
    # the kinds match, retain the normal recursive merge so a later file can
    # override only selected attributes.
    if "kind" in base and "kind" in override and base["kind"] != override["kind"]:
        return deepcopy(override)

    merged = deepcopy(base)
    for key, value in override.items():
        if value is BLANK:
            continue
        if key in merged:
            merged[key] = _merge_config_values(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_scenario_path(scenario: Path | str) -> Path:
    """Resolve a built-in scenario name or validate a scenario file path."""
    if isinstance(scenario, str) and scenario in (
        builtin_scenarios := get_builtin_scenarios()
    ):
        return builtin_scenarios[scenario]

    scenario_path = Path(scenario)
    if scenario_path.exists() and scenario_path.is_file():
        return scenario_path

    raise ValueError(f"Scenario '{scenario}' not found.")


def _load_scenario_file(scenario_path: Path) -> dict[str, Any]:
    """Load one scenario file and unwrap benchmark report configurations."""
    with scenario_path.open() as file:
        if scenario_path.suffix == ".json":
            scenario_data = json.load(file)
        elif scenario_path.suffix in {".yaml", ".yml"}:
            scenario_data = yaml.safe_load(file)
        else:
            raise ValueError(
                f"Unsupported scenario file format: {scenario_path.suffix}"
            )

    if not isinstance(scenario_data, dict):
        raise ValueError(f"Scenario '{scenario_path}' must contain a mapping.")

    # Reports contain the benchmark configuration under a top-level ``config``
    # key. Keep accepting them as scenario inputs for backwards compatibility.
    if "config" in scenario_data:
        scenario_data = scenario_data["config"]
        if not isinstance(scenario_data, dict):
            raise ValueError(
                f"Scenario report '{scenario_path}' has an invalid 'config'."
            )

    return scenario_data


def merge_scenario_files(
    scenario_files: Sequence[Path | str],
) -> dict[str, Any]:
    """Load and merge scenario files in the order supplied.

    Each file is loaded as YAML or JSON and merged into the result from the
    preceding files. Nested mappings are merged recursively. Mappings that
    both contain a registry ``kind`` are treated as one polymorphic object: a
    matching kind permits an attribute-level merge, while a different kind
    replaces the complete object so fields belonging to the previous variant
    cannot leak into the new configuration. Lists and scalar values from a
    later file replace earlier values.

    Scenario file metadata is handled specially, in order to collect an
    aggregated set of metadata identifying the stacked configurations:
    a "scenario" list is constructed identifying the name and metadata from
    each scenario file, in order.

    Both file paths and built-in scenario names are accepted, along with
    GuideLLM JSON report files, which contain an embedded "config" object.

    :param scenario_files: Ordered scenario names or YAML/JSON file paths
    :return: The merged, unvalidated configuration mapping
    :raises ValueError: If a scenario cannot be found, has an unsupported format,
        or does not contain a mapping
    """
    merged_config: dict[str, Any] = {}
    includes = []
    for scenario in scenario_files:
        scenario_path = _resolve_scenario_path(scenario)
        scenario_data = _load_scenario_file(scenario_path)
        scenario_metadata = scenario_data.get("metadata", {})
        metadata = {
            "scenario": scenario_path.name,
            "metadata": scenario_metadata,
        }
        if isinstance(scenario_metadata, dict) and isinstance(
            scenario_metadata.get("config_files"), list
        ):
            includes.extend(scenario_metadata["config_files"])
        else:
            includes.append(metadata)
        merged_config = _merge_config_values(merged_config, scenario_data)

    if isinstance(merged_config.get("metadata"), dict):
        merged_config["metadata"] = {"includes": includes}
    return merged_config


class MetricsArgs(PydanticClassRegistryMixin["MetricsArgs"], ABC):
    """Base class for metrics collection arguments.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[MetricsArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base MetricsArgs class for schema validation
        """
        if cls.__name__ == "MetricsArgs":
            return cls

        return MetricsArgs

    kind: str = Field(
        description="The kind of metrics configuration to use.",
    )
    warnings: list[WarningRuleArgs] = Field(
        default_factory=default_warning_rules,
        description=(
            "Post-benchmark warnings. Each rule names a metric and a threshold, "
            "or a metric divided by a second metric. Omit the field to use the "
            "built-in rules. Set the field to replace those rules."
        ),
    )


@MetricsArgs.register("generative")
class GenerativeMetricsArgs(MetricsArgs):
    """Metrics configuration for generative (autoregressive) benchmarks."""

    kind: Literal["generative"] = Field(
        default="generative",
        description="The kind of metrics configuration to use.",
    )
    sample_size: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests per status group (completed, errored, "
            "incomplete) to retain full data (prompt, output, tool calls) for in "
            "the final benchmark. Lightweight stats (latency, token counts) are "
            "always kept for every request. None keeps all request data, 0 strips "
            "all request data, N > 0 uses reservoir sampling to retain N per group."
        ),
        examples=[None, 0, 100],
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description=(
            "Prioritize server-reported metrics over client-calculated metrics "
            "when both are available."
        ),
    )
    confidence: float | None = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description=(
            "Two-sided confidence level for the intervals reported alongside "
            "request-level metrics. Set to null to report those metrics without "
            "intervals."
        ),
        examples=[0.95, 0.99, None],
    )
    slo: GoodputSLO | None = Field(
        default=None,
        description=(
            "Per-request latency objectives defining which requests count "
            "toward goodput. None disables goodput measurement."
        ),
        examples=[None, {"ttft_ms": 2000, "tpot_ms": 100}],
    )


class BenchmarkArgs(ReloadableBaseModel):
    """Common benchmark configuration arguments."""

    model_config = args_model_config()

    backend: BackendArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("openai_http"),
        description=(
            "Backend configuration to define how to send requests to the model."
        ),
        examples=[
            {
                "kind": "openai_http",
                "target": "http://localhost:8000/v1",
            }
        ],
        json_schema_extra={"argument_alias": "backend"},
    )
    profile: ProfileArgs = Field(  # type: ignore[assignment]
        description="Profile configuration to control benchmark execution.",
        examples=[{"kind": "sweep", "sweep_size": [10.0]}],
        json_schema_extra={"argument_alias": "profile"},
    )
    constraints: list[ConstraintArgs] = Field(  # type: ignore[assignment]
        description="Execution constraints to enforce during benchmark execution",
        examples=[{"kind": "max_requests", "value": 10}],
        default_factory=list,
        json_schema_extra={"argument_alias": "constraint"},
    )
    tokenizer: DataTokenizerArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("huggingface_auto"),
        description="Tokenizer configuration",
        examples=[{"kind": "huggingface_auto"}],
        json_schema_extra={"argument_alias": "tokenizer"},
    )
    data: list[DataArgs] = Field(  # type: ignore[assignment]
        description="List of dataset sources to use in the benchmarks",
        examples=[
            {"kind": "synthetic_text", "prompt_tokens": 100, "output_tokens": 100},
            {
                "kind": "huggingface",
                "source": "my/dataset",
                "load_kwargs": {"split": "test", "name": "my_dataset"},
            },
        ],
        min_length=1,
        json_schema_extra={"argument_alias": "data"},
    )
    data_column_mapper: DataPreprocessorArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative_column_mapper"),
        description="Specify how to map dataset columns into prompts and outputs.",
        examples=[{"kind": "generative_column_mapper"}],
        json_schema_extra={"argument_alias": "data_column_mapper"},
    )
    data_preprocessors: list[DataPreprocessorArgs] = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind_list("encode_media"),  # type: ignore[arg-type]
        description="List of dataset preprocessors to apply to the datasets.",
        examples=[{"kind": "encode_media"}],
        json_schema_extra={"argument_alias": "data_preprocessor"},
    )
    data_finalizer: DataFinalizerArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative"),
        description="Finalizer for preparing data samples into requests",
        examples=[{"kind": "generative"}],
        json_schema_extra={"argument_alias": "data_finalizer"},
    )
    data_loader: DataLoaderArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("pytorch"),
        description="Specify how to load the datasets into memory.",
        examples=[{"kind": "pytorch"}],
        json_schema_extra={"argument_alias": "data_loader"},
    )
    seed: RandomArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("static"),
        description="Random configuration for reproducibility (e.g., seed value)",
        examples=[{"kind": "static", "value": 42}],
        json_schema_extra={"argument_alias": "seed"},
    )
    outputs: list[BenchmarkOutputArgs] = Field(
        default_factory=lambda: default_kind_list("json", "csv"),  # type: ignore[arg-type]
        description="Benchmark output formats and paths.",
        examples=[
            {"kind": "json", "path": "benchmarks.json"},
        ],
        json_schema_extra={"argument_alias": "output"},
    )
    metrics: MetricsArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative"),
        description="Configuration for metrics collection and request sampling.",
        examples=[{"kind": "generative", "confidence": 0.99}],
        json_schema_extra={"argument_alias": "metrics"},
    )
    collectors: list[CollectorArgs] = Field(  # type: ignore[assignment]
        default_factory=list,
        description=(
            "Collectors that gather data from outside GuideLLM while each "
            "benchmark runs, such as a server's Prometheus metrics. Each one "
            "reports a result per benchmark."
        ),
        examples=[
            {"kind": "prometheus", "url": "http://localhost:8000/metrics"},
        ],
        json_schema_extra={"argument_alias": "collector"},
    )

    @model_validator(mode="after")
    def _check_profile_supports_metrics(self) -> BenchmarkArgs:
        """
        Let the profile reject a metrics configuration it cannot work with.

        :return: The validated instance
        :raises ValueError: If the profile rejects the metrics configuration
        """
        self.profile.validate_metrics(self.metrics)

        return self


class BenchmarkMetadata(StandardBaseModel):
    """
    Metadata about the benchmark scenario.

    Contains information such as name, description, and tags that describe the
    benchmark scenario. This metadata is used for reporting and organizational
    purposes but does not affect benchmark execution.
    """

    model_config = args_model_config()

    name: str | None = Field(
        default=None,
        description="The name of a benchmark scenario",
        examples=[{"name": "benchmark"}],
    )

    description: str | None = Field(
        default=None,
        description="The description of a benchmark scenario",
        examples=[{"description": "Benchmark description"}],
    )

    includes: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "When importing a series of config files, GuideLLM builds "
            "a list of the BenchmarkMetadata defined in each"
        ),
        examples=[
            {
                "includes": [
                    {"name": "a", "description": "a config file", "labels": {}},
                    {"name": "b", "description": "another", "labels": {"key": "value"}},
                ]
            }
        ],
    )

    labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key-value pairs of metadata labels which will be written to the "
            "output reports."
        ),
        examples=[{"labels": {"gpu": "gpu1", "owner": "me"}}],
    )


class BenchmarkScenario(ReloadableBaseModel, BaseSettings):
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
        env_prefix="GUIDELLM__",
        env_nested_delimiter="__",
        validate_default=True,
    )

    @classmethod
    def create(
        cls,
        scenario: Path | str | Sequence[Path | str] | None,
        **kwargs: Any,
    ) -> BenchmarkScenario:
        """
        Create benchmark args from scenario files and keyword arguments.

        Loads each scenario file in order, merging the files before applying
        keyword arguments. Arguments explicitly set via kwargs override config
        values, while defaulted kwargs are ignored to preserve config settings.

        :param scenario: Ordered scenario paths or built-in names, or None
        :param kwargs: Keyword arguments to override scenario values
        :return: Configured benchmark args instance
        :raises ValueError: If file is not found or file format is unsupported
        """
        constructor_kwargs = {}

        if scenario is not None:
            scenario_files = (
                (scenario,) if isinstance(scenario, (str, Path)) else scenario
            )
            constructor_kwargs.update(merge_scenario_files(scenario_files))

        # Apply overrides from kwargs
        constructor_kwargs = _merge_config_values(constructor_kwargs, kwargs)
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
        description=(
            "User metadata to describe the benchmark run. This data is written "
            "to the output file but not otherwise used by GuideLLM)."
        ),
        examples=[
            {"labels": {"name": "benchmark", "description": "Benchmark description"}}
        ],
    )
    spec: BenchmarkArgs = Field(
        default_factory=BenchmarkArgs,  # type: ignore[arg-type]
        description="Global configuration parameters for benchmark execution.",
        examples=[
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                },
                "data": [{"kind": "synthetic_text"}],
            }
        ],
    )
    benchmarks: list[dict[str, Any] | None] = Field(
        default_factory=lambda: [None],  # type: ignore[arg-type]
        description=(
            "Individual benchmark parameter overrides. This allows overriding "
            "parameters and constraints for each benchmark run by a profile."
        ),
        min_length=1,
        examples=[
            {"profile.rate": 10.0, "constraints[0].seconds": 10},
            {"profile.rate": 20.0, "constraints[0].seconds": 20},
        ],
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
