from __future__ import annotations

from abc import ABC
from typing import ClassVar, Literal

from pydantic import Field

from guidellm.schemas import _PydanticClassRegistryMixin, standard_model_config

__all__ = [
    "RandomArgs",
    "StaticRandomArgs",
]


class RandomArgs(_PydanticClassRegistryMixin["RandomArgs"], ABC):
    """Base class for random initialization arguments.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[RandomArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base RandomArgs class for schema validation
        """
        if cls.__name__ == "RandomArgs":
            return cls

        return RandomArgs

    kind: str = Field(
        description="The kind of random configuration to use.",
    )


@RandomArgs.register("static")
class StaticRandomArgs(RandomArgs):
    kind: Literal["static"] = Field(
        default="static",
        description="The kind of random configuration to use.",
    )
    value: int = Field(
        default=42,
        description="The value to use for static random configuration.",
    )
