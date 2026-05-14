from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict, Field

from guidellm.schemas import PydanticClassRegistryMixin, StandardBaseModel

__all__ = [
    "DataArgs",
    "DataEntrypointArgs",
    "DataLoaderArgs",
]


class DataLoaderArgs(
    PydanticClassRegistryMixin["DataLoaderArgs"],
    ABC,
):
    """
    Base class for data loader argument models.

    This class serves as a base for defining argument models related to data loading
    configurations. It inherits from PydanticClassRegistryMixin to enable automatic
    registration of subclasses, allowing for flexible and extensible data loading
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
    def __pydantic_schema_base_type__(cls) -> type[DataLoaderArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base DataLoaderArgs class for schema validation
        """
        if cls.__name__ == "DataLoaderArgs":
            return cls

        return DataLoaderArgs

    kind: str = Field(
        description="Type identifier for the data loader configuration.",
    )
    samples: int = Field(
        default=-1,
        description=(
            "Number of data samples to generate. If -1, the data loader will "
            "generate indefinitely until the dataset is exhausted."
        ),
    )


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


DataLoaderArgsT = TypeVar("DataLoaderArgsT", bound=DataLoaderArgs)


class DataEntrypointArgs(StandardBaseModel, Generic[DataLoaderArgsT]):
    """
    Arguments for data entry points.

    This class encapsulates the arguments required for data entry points, including
    the data loader configuration and the data loading arguments. It is designed to
    be flexible and extensible, allowing for various data loading configurations to be
    specified.
    """

    loader: DataLoaderArgsT = Field(
        description="Configuration for the data loader.",
    )
    data: list[DataArgs] = Field(
        default_factory=list,
        description="List of data loading argument configurations.",
    )
