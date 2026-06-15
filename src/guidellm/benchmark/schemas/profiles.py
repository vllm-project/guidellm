"""
Profile argument schemas for multi-strategy benchmark execution.

Defines the base argument model for profile configuration, including warmup
and cooldown phase settings. Uses Pydantic class registry for polymorphic
deserialization of profile-specific argument types.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import (
    Field,
    NonNegativeFloat,
    field_validator,
)

from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = ["ProfileArgs"]


class ProfileArgs(PydanticClassRegistryMixin["ProfileArgs"], ABC):
    """
    Base class for profile creation arguments.

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

    @classmethod
    def _fail_on_duplicate_rate(cls, data: Any, key: str) -> Any:
        """Fail if both "rate" and <key> are specified.

        Some profile alias "rate" and a more specific key; if the user enters
        both, either directly or via the global "--rate" option, we should fail.

        for example:

            "--profile kind=concurrent,streams=2.0 --rate 3"
            "--profile kind=concurrent,streams=2,rate=3"

        Pydantic won't resolve all cases consistently, so we need to fail explicitly.

        :param data: The data to validate
        :param key: The key to check for duplicate rate
        :return: The data
        """
        if isinstance(data, dict) and all(key in data for key in ("rate", key)):
            raise ValueError(f"Both 'rate' and '{key}' cannot be specified.")
        return data

    kind: str = Field(
        description="Profile type discriminator for polymorphic serialization",
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds to ramp up the targeted scheduling rate, if applicable"
        ),
    )
    warmup: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Warmup phase configuration excluding initial transient period",
    )
    cooldown: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Cooldown phase configuration excluding final transient period",
    )

    @field_validator("warmup", "cooldown", mode="before")
    @classmethod
    def _coerce_transient_phase(cls, v: Any) -> Any:
        if isinstance(v, int | float | None):
            return TransientPhaseConfig.create_from_value(v)
        return v
