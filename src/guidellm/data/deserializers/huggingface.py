from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)
from datasets.exceptions import (
    DataFilesNotFoundError,
    DatasetNotFoundError,
    FileNotFoundDatasetsError,
)
from pydantic import AliasChoices, Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs

__all__ = ["HuggingFaceDatasetDeserializer"]


@DataArgs.register(["huggingface", "hf"])
class HuggingFaceDataArgs(DataArgs):
    kind: Literal["huggingface", "hf"] = Field(
        default="huggingface",
        description="Type identifier for the Hugging Face dataset deserializer.",
    )
    source: str | Dataset | IterableDataset | DatasetDict | IterableDatasetDict = Field(
        validation_alias=AliasChoices("source", "src", "from", "path", "name"),
        description=(
            "Data input for the Hugging Face dataset deserializer. This can be a "
            "Dataset, IterableDataset, DatasetDict, IterableDatasetDict, a string or "
            "Path to a local dataset directory or a local .py dataset script, or a "
            "dataset identifier from the Hugging Face Hub."
        ),
    )


@DatasetDeserializerFactory.register(["huggingface", "hf"])
class HuggingFaceDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: HuggingFaceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
        _ = (processor_factory, random_seed)
        data = config.source

        if isinstance(
            data, Dataset | IterableDataset | DatasetDict | IterableDatasetDict
        ):
            return data

        load_error = None

        if (
            isinstance(data, str | Path)
            and (path := Path(data)).exists()
            and ((path.is_file() and path.suffix == ".py") or path.is_dir())
        ):
            # Handle python script or nested python script in a directory
            try:
                return load_dataset(str(data), **config.load_kwargs)
            except (
                FileNotFoundDatasetsError,
                DatasetNotFoundError,
                DataFilesNotFoundError,
            ) as err:
                load_error = err
            except Exception:  # noqa: BLE001
                # Try loading as a local dataset directory next
                try:
                    return load_from_disk(str(data), **config.load_kwargs)
                except (
                    FileNotFoundDatasetsError,
                    DatasetNotFoundError,
                    DataFilesNotFoundError,
                ) as err2:
                    load_error = err2

        try:
            # Handle dataset identifier from the Hugging Face Hub
            return load_dataset(str(data), **config.load_kwargs)
        except (
            FileNotFoundDatasetsError,
            DatasetNotFoundError,
            DataFilesNotFoundError,
        ) as err:
            load_error = err

        not_supported = DataNotSupportedError(
            "Unsupported data for HuggingFaceDatasetDeserializer, "
            "expected Dataset, IterableDataset, DatasetDict, IterableDatasetDict, "
            "str or Path to a local dataset directory or a local .py dataset script, "
            f"got {data} and HF load error: {load_error}"
        )

        if load_error is not None:
            raise not_supported from load_error
        else:
            raise not_supported
