from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataPreprocessorArgs


@DataPreprocessorArgs.register(
    [
        "generative_column_mapper",
        "pooling_column_mapper",
    ]
)
class GenerativeColumnMapperArgs(DataPreprocessorArgs):
    """Model for generative column mapper preprocessor arguments."""

    kind: Literal["generative_column_mapper", "pooling_column_mapper"] = Field(
        default="generative_column_mapper",
        description="Type identifier for the generative column mapper preprocessor.",
    )
    column_mappings: dict[str, str | list[str]] | None = Field(
        default=None,
        description="Mappings for the column names.",
        examples=[
            {
                "prompt_tokens_count_column": [
                    "prompt_tokens_count",
                    "input_tokens_count",
                ],
                "output_tokens_count_column": [
                    "output_tokens_count",
                    "completion_tokens_count",
                ],
            }
        ],
    )
