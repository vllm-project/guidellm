"""
Kind-discriminated constraint argument schemas for benchmark configuration.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from annotated_types import Gt, Lt
from pydantic import BeforeValidator, ConfigDict, Field

from guidellm.schemas import PydanticClassRegistryMixin

__all__ = [
    "ConstraintArgs",
    "ErrorRate",
    "ErrorRateOrList",
    "PositiveNum",
    "PositiveNumOrList",
]


def _unwrap_single(
    v: int | float | list[int | float],
) -> int | float | list[int | float]:
    """Normalize single-element lists to scalars for cleaner serialization."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v


PositiveNum = Annotated[int | float, Gt(0)]
PositiveNumOrList = Annotated[
    PositiveNum | list[PositiveNum], BeforeValidator(_unwrap_single)
]

ErrorRate = Annotated[int | float, Gt(0), Lt(1)]
ErrorRateOrList = Annotated[
    ErrorRate | list[ErrorRate], BeforeValidator(_unwrap_single)
]


class ConstraintArgs(PydanticClassRegistryMixin["ConstraintArgs"]):
    """
    Base class for constraint configuration arguments.

    Uses ``PydanticClassRegistryMixin`` to enable polymorphic deserialization
    based on the ``kind`` field. Each registered subclass represents a specific
    constraint type with its own parameters.

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
    def __pydantic_schema_base_type__(cls) -> type[ConstraintArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base ConstraintArgs class for schema validation
        """
        if cls.__name__ == "ConstraintArgs":
            return cls

        return ConstraintArgs

    kind: str = Field(
        description="Constraint type discriminator for polymorphic serialization",
    )

    @property
    def constraint_key(self) -> str:
        """
        The key to use when inserting into the constraints dict.

        Defaults to ``kind``, but subclasses may override if the factory
        registry key differs from the args kind.

        :return: Registry key for this constraint type
        """
        return self.kind
