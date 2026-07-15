from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = [
    "BenchmarkOutputArgs",
    "PlotBenchmarkOutputArgs",
]
ALLOWED_PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


class BenchmarkOutputArgs(PydanticClassRegistryMixin["BenchmarkOutputArgs"], ABC):
    """Base class for output creation arguments.

    This class serves as a base for defining argument models used in the creation
    of output instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    output configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BenchmarkOutputArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base BenchmarkOutputArgs class for schema validation
        """
        if cls.__name__ == "BenchmarkOutputArgs":
            return cls

        return BenchmarkOutputArgs

    kind: str = Field(
        description="Type identifier for the output configuration.",
    )


@BenchmarkOutputArgs.register("plot")
class PlotBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for Plot benchmark output arguments.

    Defines parameters for generating static image visualizations, enforcing
    image output suffix.
    """

    kind: Literal["plot"] = Field(
        default="plot",
        description="Type identifier for the plot configuration.",
    )
    path: Path = Field(
        default_factory=lambda: Path("./benchmarks.png"),
        description="The file to save the output plot to.",
    )
    dpi: int = Field(
        default=100,
        description="Resolution of the output image in Dots Per Inch.",
    )

    @field_validator("path", mode="after")
    @classmethod
    def validate_plot_suffix(cls, v: Path) -> Path:
        """Ensures the output file path ends with a supported plotting format extension.
        
        If the suffix is missing or not supported, it defaults to .png.
        """
        if v.suffix.lower() not in ALLOWED_PLOT_SUFFIXES:
            return v.with_suffix(".png")
        return v


