from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from guidellm.schemas import PydanticClassRegistryMixin

__all__ = [
    "DataArgs",
    "DataFinalizerArgs",
    "DataLoaderArgs",
    "DataPreprocessorArgs",
    "DataTokenizerArgs",
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


class DataPreprocessorArgs(
    PydanticClassRegistryMixin["DataPreprocessorArgs"],
    ABC,
):
    """
    Base class for data preprocessor argument models.

    This class serves as a base for defining arguments related to data preprocessing
    configurations. It inherits from PydanticClassRegistryMixin to enable automatic
    registration of subclasses, allowing for flexible and extensible data preprocessing
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
    def __pydantic_schema_base_type__(cls) -> type[DataPreprocessorArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base DataPreprocessorArgs class for schema validation
        """
        if cls.__name__ == "DataPreprocessorArgs":
            return cls

        return DataPreprocessorArgs

    kind: str = Field(
        description="Type identifier for the data preprocessor arguments.",
    )


class DataFinalizerArgs(
    PydanticClassRegistryMixin["DataFinalizerArgs"],
    ABC,
):
    """
    Base class for data finalizer argument models.

    This class serves as a base for defining arguments related to data finalization
    configurations. It inherits from PydanticClassRegistryMixin to enable automatic
    registration of subclasses, allowing for flexible and extensible data finalization
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
    def __pydantic_schema_base_type__(cls) -> type[DataFinalizerArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base DataFinalizerArgs class for schema validation
        """
        if cls.__name__ == "DataFinalizerArgs":
            return cls

        return DataFinalizerArgs

    kind: str = Field(
        description="Type identifier for the data finalizer arguments.",
    )


class DataTokenizerArgs(
    PydanticClassRegistryMixin["DataTokenizerArgs"],
    ABC,
):
    """
    Base class for data tokenizer argument models.

    This class serves as a base for defining arguments related to data tokenization
    configurations. It inherits from PydanticClassRegistryMixin to enable automatic
    registration of subclasses, allowing for flexible and extensible data tokenization
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
    def __pydantic_schema_base_type__(cls) -> type[DataTokenizerArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base DataTokenizerArgs class for schema validation
        """
        if cls.__name__ == "DataTokenizerArgs":
            return cls

        return DataTokenizerArgs

    kind: str = Field(
        description="Type identifier for the data tokenizer arguments.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional model name or path for the tokenizer. This field can be used by "
            "tokenizer implementations that require a model specification, such as "
            "HuggingFace tokenizers."
        ),
    )
