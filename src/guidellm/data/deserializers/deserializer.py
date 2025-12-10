from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Union, runtime_checkable

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizerBase

from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.utils import resolve_dataset_split
from guidellm.utils import RegistryMixin

__all__ = [
    "DatasetDeserializer",
    "DatasetDeserializerFactory",
]


@runtime_checkable
class DatasetDeserializer(Protocol):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict: ...


class DatasetDeserializerFactory(
    RegistryMixin[Union["type[DatasetDeserializer]", DatasetDeserializer]],
):
    @classmethod
    def deserialize(
        cls,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        type_: str | None = None,
        resolve_split: bool = True,
        select_columns: list[str] | None = None,
        remove_columns: list[str] | None = None,
        **data_kwargs: dict[str, Any],
    ) -> Dataset | IterableDataset:
        dataset: Dataset

        if type_ is None:
            dataset = cls._deserialize_with_registered_deserializers(
                data, processor_factory, random_seed, **data_kwargs
            )

        else:
            dataset = cls._deserialize_with_specified_deserializer(
                data, type_, processor_factory, random_seed, **data_kwargs
            )

        if resolve_split:
            dataset = resolve_dataset_split(dataset)

        if select_columns is not None or remove_columns is not None:
            column_names = dataset.column_names or list(next(iter(dataset)).keys())
            if select_columns is not None:
                remove_columns = [
                    col for col in column_names if col not in select_columns
                ]

            dataset = dataset.remove_columns(remove_columns)

        return dataset

    @classmethod
    def _deserialize_with_registered_deserializers(
        cls,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        if cls.registry is None:
            raise RuntimeError("registry is None; cannot deserialize dataset")
        dataset: Dataset | None = None

        errors: dict[str, Exception] = {}
        # Note: There is no priority order for the deserializers, so all deserializers
        #  must be mutually exclusive to ensure deterministic behavior.
        for _name, deserializer in cls.registry.items():
            deserializer_fn: DatasetDeserializer = (
                deserializer() if isinstance(deserializer, type) else deserializer
            )

            try:
                dataset = deserializer_fn(
                    data=data,
                    processor_factory=processor_factory,
                    random_seed=random_seed,
                    **data_kwargs,
                )
            except Exception as e:  # noqa: BLE001 # The exceptions are saved.
                errors[_name] = e

            if dataset is not None:
                return dataset  # Success

        if len(errors) > 0:
            err_msgs = ""

            def sort_key(item):
                return (isinstance(item[1], DataNotSupportedError), item[0])

            for key, err in sorted(errors.items(), key=sort_key):
                err_msgs += f"\n  - Deserializer '{key}': ({type(err).__name__}) {err}"
            raise ValueError(
                "Data deserialization failed, likely because the input doesn't "
                f"match any of the input formats. See the {len(errors)} error(s) that "
                f"occurred while attempting to deserialize the data {data}:{err_msgs}"
            )
        return dataset

    @classmethod
    def _deserialize_with_specified_deserializer(
        cls,
        data: Any,
        type_: str,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        deserializer_from_type = cls.get_registered_object(type_)
        if deserializer_from_type is None:
            raise ValueError(f"Deserializer type '{type_}' is not registered.")
        if isinstance(deserializer_from_type, type):
            deserializer_fn = deserializer_from_type()
        else:
            deserializer_fn = deserializer_from_type

        return deserializer_fn(
            data=data,
            processor_factory=processor_factory,
            random_seed=random_seed,
            **data_kwargs,
        )
