from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = [
    "BenchmarkOutputArgs",
    "JSONBenchmarkOutputArgs",
]


class BenchmarkOutputArgs(PydanticClassRegistryMixin["BenchmarkOutputArgs"], ABC):
    """
    Base class for output creation arguments.

    This class serves as a base for defining argument models used in the creation
    of backend instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    backend configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BenchmarkOutputArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base BackendArgs class for schema validation
        """
        if cls.__name__ == "BenchmarkOutputArgs":
            return cls

        return BenchmarkOutputArgs

    kind: str = Field(
        description="Type identifier for the output configuration.",
    )


@BenchmarkOutputArgs.register("json")
class JSONBenchmarkOutputArgs(BenchmarkOutputArgs):
    kind: Literal["json"] = Field(
        default="json",
        description="The kind of output.",
    )
    filename: Path = Field(
        default=Path("./benchmarks.json"),
        description="The filename to save the output to.",
    )
