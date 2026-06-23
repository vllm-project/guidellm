"""
Profile argument schemas for multi-strategy benchmark execution.

Defines the base argument model for profile configuration, including warmup
and cooldown phase settings. Uses Pydantic class registry for polymorphic
deserialization of profile-specific argument types.
"""

from __future__ import annotations

import contextlib
from abc import ABC
from typing import Any, ClassVar

from pydantic import (
    Field,
    NonNegativeFloat,
    field_validator,
)

from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config
from guidellm.utils.imports import json

__all__ = ["ProfileArgs"]


class ProfileArgs(PydanticClassRegistryMixin["ProfileArgs"], ABC):
    """Base class for profile creation arguments.

    This class serves as a base for defining argument models used in the creation
    of profile instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    profile configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[ProfileArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base ProfileArgs class for schema validation
        """
        if cls.__name__ == "ProfileArgs":
            return cls

        return ProfileArgs

    kind: str = Field(
        description="Profile type discriminator",
        examples=["concurrent", "synchronous"],
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=("Duration in seconds to ramp up the targeted scheduling rate"),
    )
    warmup: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Warmup phase to exclude initial transient period",
        examples=[0.0, 1.0, {"mode": "percent", "percent": 2.0}],
    )
    cooldown: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Cooldown phase to exclude final transient period",
        examples=[0.0, 1.0, {"mode": "duration", "value": 2.0}],
    )

    @field_validator("warmup", "cooldown", mode="before")
    @classmethod
    def _coerce_transient_phase(cls, v: Any) -> Any:
        if isinstance(v, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                v = json.loads(v)
        if isinstance(v, int | float | None):
            return TransientPhaseConfig.create_from_value(v)
        return v
