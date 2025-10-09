from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

__all__ = ["ColumnInputTypes", "DatasetCreator"]

ColumnInputTypes = Literal[
    "prompt_column",
    "text_column",
    "prompt_tokens_count_column",
    "output_tokens_count_column",
]


class DatasetCreator(ABC):
    DEFAULT_SPLITS_TRAIN = [
        "train",
        "training",
        "train_set",
        "training_set",
        "train_dataset",
        "training_dataset",
        "train_data",
        "training_data",
        "pretrain",
        "pretrain_set",
        "pretrain_dataset",
        "pretrain_data",
        "pretraining",
    ]
    DEFAULT_SPLITS_CALIB = [
        "calibration",
        "calib",
        "cal",
        "calibration_set",
        "calib_set",
        "cal_set",
        "calibration_dataset",
        "calib_dataset",
        "cal_set",
        "calibration_data",
        "calib_data",
        "cal_data",
    ]
    DEFAULT_SPLITS_VAL = [
        "validation",
        "val",
        "valid",
        "validation_set",
        "val_set",
        "validation_dataset",
        "val_dataset",
        "validation_data",
        "val_data",
        "dev",
        "dev_set",
        "dev_dataset",
        "dev_data",
    ]
    DEFAULT_SPLITS_TEST = [
        "test",
        "testing",
        "test_set",
        "testing_set",
        "test_dataset",
        "testing_dataset",
        "test_data",
        "testing_data",
        "eval",
        "eval_set",
        "eval_dataset",
        "eval_data",
    ]
    DEFAULT_SPLITS_DATASET: dict[str, str] = {}

    @classmethod
    def create(
        cls,
        data: Any,
        data_args: Optional[dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[dict[str, Any]],
        random_seed: int = 42,
        split_pref_order: Optional[list[str]] = None,
    ) -> tuple[Union[Dataset, IterableDataset], dict[ColumnInputTypes, str]]:
        if not cls.is_supported(data, data_args):
            raise ValueError(f"Unsupported data type: {type(data)} given for {data}. ")

        split = cls.extract_args_split(data_args)
        column_mappings = cls.extract_args_column_mappings(data_args)
        dataset = cls.handle_create(
            data, data_args, processor, processor_args, random_seed
        )

        if isinstance(dataset, DatasetDict | IterableDatasetDict):
            dataset = cls.extract_dataset_split(dataset, split, split_pref_order)

        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(
                f"Unsupported data type: {type(dataset)} given for {dataset}."
            )

        return dataset, column_mappings

    @classmethod
    def extract_args_split(cls, data_args: Optional[dict[str, Any]]) -> str:
        split = "auto"

        if data_args and "split" in data_args:
            split = data_args["split"]
            del data_args["split"]

        return split

    @classmethod
    def extract_args_column_mappings(
        cls,
        data_args: Optional[dict[str, Any]],
    ) -> dict[ColumnInputTypes, str]:
        columns: dict[ColumnInputTypes, str] = {}

        if data_args:
            if "prompt_column" in data_args:
                columns["prompt_column"] = data_args["prompt_column"]
                del data_args["prompt_column"]

            if "prompt_tokens_count_column" in data_args:
                columns["prompt_tokens_count_column"] = data_args[
                    "prompt_tokens_count_column"
                ]
                del data_args["prompt_tokens_count_column"]

            if "output_tokens_count_column" in data_args:
                columns["output_tokens_count_column"] = data_args[
                    "output_tokens_count_column"
                ]
                del data_args["output_tokens_count_column"]

        return columns

    @classmethod
    def extract_dataset_name(
        cls, dataset: Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict]
    ) -> Optional[str]:
        if isinstance(dataset, DatasetDict | IterableDatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, Dataset | IterableDataset):
            if not hasattr(dataset, "info") or not hasattr(
                dataset.info, "dataset_name"
            ):
                return None

            return dataset.info.dataset_name

        raise ValueError(f"Unsupported data type: {type(dataset)} given for {dataset}.")

    @classmethod
    def extract_dataset_split(
        cls,
        dataset: Union[DatasetDict, IterableDatasetDict],
        specified_split: Union[Literal["auto"], str] = "auto",
        split_pref_order: Optional[Union[Literal["auto"], list[str]]] = "auto",
    ) -> Union[Dataset, IterableDataset]:
        if not isinstance(dataset, DatasetDict | IterableDatasetDict):
            raise ValueError(
                f"Unsupported data type: {type(dataset)} given for {dataset}."
            )

        if specified_split != "auto":
            if specified_split not in dataset:
                raise ValueError(
                    f"Split {specified_split} not found in dataset {dataset}."
                )

            return dataset[specified_split]

        dataset_name = cls.extract_dataset_name(dataset)

        if dataset_name and dataset_name in cls.DEFAULT_SPLITS_DATASET:
            return dataset[cls.DEFAULT_SPLITS_DATASET[dataset_name]]

        if split_pref_order == "auto":
            split_pref_order = [
                *cls.DEFAULT_SPLITS_TEST,
                *cls.DEFAULT_SPLITS_VAL,
                *cls.DEFAULT_SPLITS_CALIB,
                *cls.DEFAULT_SPLITS_TRAIN,
            ]

        for test_split in split_pref_order or []:
            if test_split in dataset:
                return dataset[test_split]

        return dataset[list(dataset.keys())[0]]

    @classmethod
    @abstractmethod
    def is_supported(cls, data: Any, data_args: Optional[dict[str, Any]]) -> bool: ...

    @classmethod
    @abstractmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[dict[str, Any]],
        random_seed: int,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]: ...
