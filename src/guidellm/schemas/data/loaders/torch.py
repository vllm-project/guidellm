from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from guidellm.logger import logger
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
    prefetch_factor: int = Field(
        default=4096,
        description=(
            "Number of samples loaded in advance by each worker. "
            "Increasing this generates more data ahead of demand."
        ),
    )

    @field_validator("num_workers", mode="after")
    @classmethod
    def warn_if_changed(cls, v: int) -> int:
        if v != 1:
            logger.warning(
                "The value of data_loader.num_workers has been changed from its "
                "default value. This is currently not supported and may lead to "
                "unexpected behavior."
            )
        return v
