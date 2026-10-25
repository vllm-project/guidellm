"""
Base backend Args schema for polymorphic backend configuration.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar

from pydantic import Field, SecretStr

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = ["BackendArgs", "resolve_api_key_file"]


def resolve_api_key_file(
    api_key: SecretStr | None, api_key_file: Path | None
) -> SecretStr | None:
    """
    Resolve the effective API key from an inline value or a key file.

    The file is read as UTF-8 and surrounding whitespace (such as the trailing
    newline common in mounted secrets) is stripped. Error messages include the
    file path but never the file content.

    :param api_key: API key provided inline, if any
    :param api_key_file: Path to a file containing the API key, if any
    :return: The API key loaded from ``api_key_file`` if set, else ``api_key``
    :raises ValueError: If both are set, or the file is unreadable or empty
    """
    if api_key_file is None:
        return api_key

    if api_key is not None:
        raise ValueError("Only one of api_key or api_key_file may be set, not both.")

    try:
        value = api_key_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as err:
        raise ValueError(
            f"Could not read api_key_file '{api_key_file}': {type(err).__name__}"
        ) from None

    if not value:
        raise ValueError(f"api_key_file '{api_key_file}' is empty.")

    return SecretStr(value)


class BackendArgs(PydanticClassRegistryMixin["BackendArgs"], ABC):
    """
    Base class for backend creation arguments.

    This class serves as a base for defining argument models used in the creation
    of backend instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    backend configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BackendArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base BackendArgs class for schema validation
        """
        if cls.__name__ == "BackendArgs":
            return cls

        return BackendArgs

    kind: str = Field(
        description="Identify the desired backend implementation.",
        examples=["openai_http"],
    )
