from __future__ import annotations

from typing import Literal

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

__all__ = ["DEFAULT_SPLITS", "resolve_dataset_split"]


DEFAULT_SPLITS: dict[Literal["train", "calib", "val", "test"], list[str]] = {
    "train": [
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
    ],
    "calib": [
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
    ],
    "val": [
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
    ],
    "test": [
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
    ],
}


def resolve_dataset_split(
    dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict,
    split: str | None = None,
) -> Dataset | IterableDataset:
    if split is not None and isinstance(dataset, DatasetDict | IterableDatasetDict):
        if split in dataset:
            return dataset[split]

        raise ValueError(f"Requested split '{split}' not found in dataset: {dataset}.")
    elif split is not None:
        raise ValueError(
            f"Requested split '{split}' but dataset has no splits: {dataset}."
        )

    if isinstance(dataset, Dataset | IterableDataset):
        return dataset

    for _, default_splits in DEFAULT_SPLITS.items():
        for default_split in default_splits:
            if default_split in dataset:
                return dataset[default_split]

    return dataset[list(dataset.keys())[0]]
