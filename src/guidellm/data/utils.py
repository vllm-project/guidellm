from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any, Literal

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

__all__ = [
    "DEFAULT_COLUMN_NAMES",
    "DEFAULT_SPLITS",
    "datasets_item_iterator",
    "resolve_dataset_split",
]


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


DEFAULT_COLUMN_NAMES: dict[str, list[str]] = {
    "prompt_tokens_count": ["prompt_tokens_count", "input_tokens_count"],
    "output_tokens_count": ["output_tokens_count", "completion_tokens_count"],
    "text_column": [
        "prompt",
        "instruction",
        "question",
        "input",
        "context",
        "content",
        "conversation",
        "turn",
        "text",
    ],
    "image_column": [
        "image",
        "picture",
        "photo",
        "img",
    ],
    "video_column": [
        "video",
        "clip",
        "movie",
        "footage",
        "mp4",
        "mov",
        "avi",
    ],
    "audio_column": [
        "audio",
        "sound",
        "voice",
        "speech",
        "wav",
        "mp3",
    ],
}


def resolve_dataset_split(
    dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict,
    split: str | None,
) -> Dataset | IterableDataset:
    if split is not None and isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        if split in dataset:
            return dataset[split]

        raise ValueError(f"Requested split '{split}' not found in dataset: {dataset}.")
    elif split is not None:
        raise ValueError(
            f"Requested split '{split}' but dataset has no splits: {dataset}."
        )

    if isinstance(dataset, (Dataset, IterableDataset)):
        return dataset

    for _, default_splits in DEFAULT_SPLITS.items():
        for default_split in default_splits:
            if default_split in dataset:
                return dataset[default_split]

    return dataset[list(dataset.keys())[0]]


def datasets_item_iterator(
    datasets: list[Dataset | IterableDataset],
    data_samples: int,
) -> Iterator[dict[Literal["items"], tuple[dict[str, Any]]]]:
    dataset_iters = [iter(dataset) for dataset in datasets]
    gen_count = 0

    with contextlib.suppress(StopIteration):
        while gen_count < data_samples or data_samples <= 0 or data_samples == math.inf:
            yield {"items": tuple(next(dataset_iter) for dataset_iter in dataset_iters)}
            gen_count += 1

    if gen_count < data_samples and data_samples > 0 and data_samples != math.inf:
        raise ValueError(
            f"Requested {data_samples} samples, but only {gen_count} available "
            "from the provided datasets."
        )
