"""Registry-backed argument model for the preprocess dataset CLI."""

from __future__ import annotations

from typing import Any

from pydantic import (
    AliasChoices,
    AliasGenerator,
    ConfigDict,
    Field,
)

from guidellm.benchmark.schemas import RandomArgs
from guidellm.data.schemas import DataPreprocessorArgs, DataTokenizerArgs
from guidellm.schemas import ReloadableBaseModel, standard_model_config

__all__ = [
    "PreprocessDatasetArgs",
    "default_kind",
]


def args_model_config() -> ConfigDict:
    return standard_model_config(
        extra="forbid",
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: AliasChoices(
                field_name, field_name.replace("_", "-")
            ),
        ),
    )


def default_kind(kind: str) -> dict[str, Any]:
    """Default factory for argument models to set the ``kind`` field."""
    return {"kind": kind}


class PreprocessDatasetArgs(ReloadableBaseModel):
    """Registry-backed CLI arguments for ``guidellm preprocess dataset``."""

    model_config = args_model_config()

    tokenizer: DataTokenizerArgs = Field(  # type: ignore[assignment]
        description=(
            "Tokenizer configuration for calculating token counts during "
            "dataset preprocessing."
        ),
        examples=[{"kind": "huggingface_auto", "model": "gpt2"}],
        json_schema_extra={"argument_alias": "tokenizer"},
    )
    data_column_mapper: DataPreprocessorArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("generative_column_mapper"),
        description="Specify how to map dataset columns into prompts and outputs.",
        examples=[{"kind": "generative_column_mapper"}],
        json_schema_extra={"argument_alias": "data_column_mapper"},
    )
    seed: RandomArgs = Field(  # type: ignore[assignment]
        default_factory=lambda: default_kind("static"),
        description="Random configuration for reproducible token sampling.",
        examples=[{"kind": "static", "value": 42}],
        json_schema_extra={"argument_alias": "seed"},
    )
