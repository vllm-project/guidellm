from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, ConfigDict, Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["HuggingFaceDataArgs"]


@DataArgs.register(["huggingface", "hf"])
class HuggingFaceDataArgs(DataArgs):
    """Model for Hugging Face dataset deserializer arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["huggingface", "hf"] = Field(
        default="huggingface",
        description="Type identifier for the Hugging Face dataset deserializer.",
    )
    source: Any = Field(
        validation_alias=AliasChoices("source", "src", "from", "path", "name"),
        description=(
            "Data input for the Hugging Face dataset deserializer. This can be a "
            "Dataset, IterableDataset, DatasetDict, IterableDatasetDict, a string or "
            "Path to a local dataset directory or a local .py dataset script, or a "
            "dataset identifier from the Hugging Face Hub."
        ),
        examples=["wikimedia/structured-wikipedia", "./dataset.json"],
    )
