from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, cast

from datasets import Dataset, IterableDataset

from guidellm.data.preprocessors.preprocessor import (
    DataDependentPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import GenerativeDatasetColumnType

__all__ = ["GenerativeColumnMapper"]


@PreprocessorRegistry.register("generative_column_mapper")
class GenerativeColumnMapper(DataDependentPreprocessor):
    defaults: ClassVar[dict[str, list[str]]] = {
        "prompt_tokens_count_column": ["prompt_tokens_count", "input_tokens_count"],
        "output_tokens_count_column": [
            "output_tokens_count",
            "completion_tokens_count",
        ],
        "prefix_column": [
            "system_prompt",
            "system",
            "prefix",
        ],
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

    @classmethod
    def datasets_default_mappings(
        cls, datasets: list[Dataset | IterableDataset]
    ) -> dict[GenerativeDatasetColumnType, list[tuple[int, str]]]:
        mappings: dict[GenerativeDatasetColumnType, list[tuple[int, str]]] = (
            defaultdict(list)
        )

        for index, dataset in enumerate(datasets):
            dataset_columns = dataset.column_names or list(next(iter(dataset)).keys())

            for column_type in cls.defaults:
                if column_type in mappings:
                    continue

                type_names = [
                    variant
                    for name in cls.defaults.get(column_type, [])
                    for plural in [name, f"{name}s", f"{name}es"]
                    for variant in [
                        plural,
                        plural.lower(),
                        plural.upper(),
                        plural.capitalize(),
                    ]
                ]

                for name in type_names:
                    if name in dataset_columns:
                        key = cast("GenerativeDatasetColumnType", column_type)
                        mappings[key].append((index, name))
                        break

        return mappings

    @classmethod
    def datasets_mappings(
        cls,
        datasets: list[Dataset | IterableDataset],
        input_mappings: dict[GenerativeDatasetColumnType, str | list[str]],
    ) -> dict[GenerativeDatasetColumnType, list[tuple[int, str]]]:
        mappings: dict[GenerativeDatasetColumnType, list[tuple[int, str]]] = (
            defaultdict(list)
        )
        datasets_named_indices = {
            (
                dataset.info.dataset_name
                if dataset.info and dataset.info.dataset_name
                else index
            ): index
            for index, dataset in enumerate(datasets)
        }
        datasets_columns = {
            index: dataset.column_names or list(next(iter(dataset)).keys())
            for index, dataset in enumerate(datasets)
        }

        # Parse out user mappings that were passed in and validate them
        # Must be in the format of:
        # {<column_type>: [<column_names>]}
        # where <column_names> can be a single string or list of strings
        # and each string can be any of:
        # - a column name (assumes the first dataset was intended)
        # - <int>.<column_name> where <int> is the dataset index
        # - <str>.<column_name> where <str> is the dataset name
        for column_type, names in input_mappings.items():
            mappings[column_type] = []
            for name in names if isinstance(names, list) else [names]:
                if "." in name:
                    dataset, column_name = name.split(".", 1)
                    dataset_index = (
                        int(dataset)
                        if dataset.isdigit()
                        else datasets_named_indices.get(dataset)
                    )
                else:
                    dataset_index = 0
                    column_name = name

                if dataset_index is None or dataset_index >= len(datasets):
                    raise ValueError(
                        f"Dataset '{name}' not found in datasets: "
                        f"{datasets_named_indices}."
                    )
                if column_name not in datasets_columns[dataset_index]:
                    raise ValueError(
                        f"Column '{column_name}' not found in dataset "
                        f"'{datasets[dataset_index]}' "
                        f"columns: {datasets_columns[dataset_index]}."
                    )
                mappings[column_type].append((dataset_index, column_name))

        return mappings

    def __init__(
        self,
        column_mappings: dict[GenerativeDatasetColumnType, str | list[str]]
        | None = None,
    ):
        self.input_mappings = column_mappings
        self.datasets_column_mappings: (
            dict[GenerativeDatasetColumnType, list[tuple[int, str]]] | None
        )

    def __call__(self, row: dict[str, Any]) -> dict[str, list[Any]]:
        if self.datasets_column_mappings is None:
            raise ValueError("DefaultGenerativeColumnMapper not setup with data.")

        items = cast("dict[int, dict[str, Any]]", row.pop("items"))
        mapped: dict[str, Any] = defaultdict(list)

        for column_type, column_mappings in self.datasets_column_mappings.items():
            for (
                dataset_index,
                dataset_column,
            ) in column_mappings:
                mapped[column_type].append(items[dataset_index][dataset_column])

        return dict(mapped)

    def setup_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[dict[str, Any]],
    ):
        _ = data_args  # Unused for this mapper
        self.datasets_column_mappings = (
            self.datasets_default_mappings(datasets)
            if self.input_mappings is None
            else self.datasets_mappings(datasets, self.input_mappings)
        )
