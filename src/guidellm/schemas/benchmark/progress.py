"""Configuration for benchmark console progress displays."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Literal

from pydantic import Field

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config


class BenchmarkProgressArgs(PydanticClassRegistryMixin["BenchmarkProgressArgs"], ABC):
    """Registry of console progress configurations."""

    model_config = standard_model_config()
    schema_discriminator: ClassVar[str] = "kind"
    kind: str

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BenchmarkProgressArgs]:
        """:return: Base type for polymorphic progress configuration validation"""
        if cls.__name__ == "BenchmarkProgressArgs":
            return cls
        return BenchmarkProgressArgs


@BenchmarkProgressArgs.register("rich")
class RichBenchmarkProgressArgs(BenchmarkProgressArgs):
    """Interactive Rich console progress configuration."""

    kind: Literal["rich"] = "rich"
    display_scheduler_stats: bool = False


@BenchmarkProgressArgs.register("simple")
class SimpleBenchmarkProgressArgs(BenchmarkProgressArgs):
    """Plain text progress configuration for containers and redirected output."""

    kind: Literal["simple"] = "simple"
    interval: float = Field(
        default=10.0,
        gt=0,
        allow_inf_nan=False,
        description="Minimum seconds between periodic progress updates.",
    )
