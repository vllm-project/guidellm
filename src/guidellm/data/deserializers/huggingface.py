from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)

__all__ = ["HuggingFaceDatasetDeserializer"]


@DatasetDeserializerFactory.register("huggingface")
class HuggingFaceDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        data_kwargs: dict[str, Any],
        processor: Any,
        random_seed: int,
    ) -> dict[str, list]:
        _ = (processor, random_seed)

        if isinstance(
            data, (Dataset, IterableDataset, DatasetDict, IterableDatasetDict)
        ):
            return data

        load_error = None

        if (
            isinstance(data, (str, Path))
            and (path := Path(data)).exists()
            and ((path.is_file() and path.suffix == ".py") or path.is_dir())
        ):
            # Handle python script or nested python script in a directory
            try:
                return load_dataset(str(data), **data_kwargs)
            except Exception as err:  # noqa: BLE001
                load_error = err

        if (
            isinstance(data, (str, Path))
            and (path := Path(data)).exists()
            and path.is_dir()
        ):
            # Handle local dataset directory
            try:
                return load_from_disk(str(data), **data_kwargs)
            except Exception as err:  # noqa: BLE001
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
