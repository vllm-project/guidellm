from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataLoaderArgs


@DataLoaderArgs.register("pytorch")
class TorchDataLoaderArgs(DataLoaderArgs):
    """Model for PyTorch data loader arguments."""

    kind: Literal["pytorch"] = Field(  # type: ignore[assignment]
        default="pytorch",
        description="Type identifier for the generative data loader.",
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle data rows at every epoch.",
    )
    num_workers: int = Field(
        default=1,
        description=(
            "Number of worker processes for data loading. If 0, data loading "
            "will be performed in the main process."
        ),
    )
