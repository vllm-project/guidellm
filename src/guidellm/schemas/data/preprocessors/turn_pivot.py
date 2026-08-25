from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataPreprocessorArgs


class TurnPivotArgs(DataPreprocessorArgs):
    """Model for turn pivot preprocessor arguments."""

    kind: Literal["turn_pivot"] = Field(
        default="turn_pivot",
        description="Type identifier for the turn pivot preprocessor.",
    )
