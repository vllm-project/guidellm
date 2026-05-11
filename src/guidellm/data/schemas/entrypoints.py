from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from guidellm.schemas import PydanticClassRegistryMixin

__all__ = [
    "DataArgs",
]


class DataArgs(
    PydanticClassRegistryMixin["DataArgs"],
    ABC,
):
    """
    Base class for data loading and processing argument models.

    This class serves as a base for defining argument models related to data loading
    and processing. It inherits from PydanticClassRegistryMixin to enable automatic
    registration of subclasses, allowing for flexible and extensible data handling
    configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = ConfigDict(
        extra="forbid",
        serialize_by_alias=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[DataArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base Profile class for schema validation
        """
        if cls.__name__ == "DataArgs":
            return cls

        return DataArgs

    kind: str = Field(
        description="Type identifier for the data arguments configuration.",
    )
    load_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional arguments for data loading. These arguements are passed to the "
            "datasets library when loading the dataset."
        ),
    )
