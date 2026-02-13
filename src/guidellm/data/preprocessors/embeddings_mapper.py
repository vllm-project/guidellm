"""
Column mapper for embeddings datasets.

Maps common text column names to the standard 'text_column' field expected by
the embeddings finalizer. Much simpler than the generative mapper since embeddings
only need a single text input field.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, cast

from datasets import Dataset, IterableDataset

from guidellm.data.preprocessors.preprocessor import (
    DataDependentPreprocessor,
    PreprocessorRegistry,
)

__all__ = ["EmbeddingsColumnMapper"]


@PreprocessorRegistry.register("embeddings_column_mapper")
class EmbeddingsColumnMapper(DataDependentPreprocessor):
    """
    Maps dataset columns to embeddings text field.

    Searches for common text column names and maps them to 'text_column'
    for the embeddings finalizer to consume.

    Example:
    ::
        # Dataset with "text" column
        mapper = EmbeddingsColumnMapper()
        dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        result = mapper.map(dataset)
        # result["text_column"] will contain the text values
    """

    defaults: ClassVar[dict[str, list[str]]] = {
        "text_column": [
            "text",
            "input",
            "content",
            "prompt",
            "sentence",
            "document",
            "passage",
            "query",
            "body",
            "message",
        ],
    }

    def __init__(
        self,
        column_mappings: dict[str, str | list[str]] | None = None,
        **_: Any,  # Ignore global kwargs
    ):
        self.input_mappings = column_mappings
        self.datasets_column_mappings: dict[str, list[tuple[int, str]]] | None = None

    @classmethod
    def datasets_default_mappings(
        cls, datasets: list[Dataset | IterableDataset]
    ) -> dict[str, list[tuple[int, str]]]:
        """
        Auto-detect text columns from datasets.

        :param datasets: List of datasets to analyze
        :return: Mapping of column types to (dataset_index, column_name) tuples
        """
        mappings: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for index, dataset in enumerate(datasets):
            dataset_columns = dataset.column_names or list(next(iter(dataset)).keys())

            # Try to find text column
            if "text_column" not in mappings or not mappings["text_column"]:
                for name_base in cls.defaults.get("text_column", []):
                    # Try various case variations
                    for variant in [
                        name_base,
                        name_base.lower(),
                        name_base.upper(),
                        name_base.capitalize(),
                    ]:
                        if variant in dataset_columns:
                            mappings["text_column"].append((index, variant))
                            break
                    if mappings["text_column"]:
                        break

        return mappings

    @classmethod
    def datasets_mappings(
        cls,
        datasets: list[Dataset | IterableDataset],
        input_mappings: dict[str, str | list[str]],
    ) -> dict[str, list[tuple[int, str]]]:
        """
        Create mappings from user-specified column names.

        :param datasets: List of datasets to map
        :param input_mappings: User-specified mappings
        :return: Validated mappings of column types to (dataset_index, column_name) tuples
        """
        mappings: dict[str, list[tuple[int, str]]] = defaultdict(list)

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

        # Parse user mappings
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
                        f"Column '{column_name}' not found in dataset {dataset_index}. "
                        f"Available columns: {datasets_columns[dataset_index]}"
                    )

                mappings[column_type].append((dataset_index, column_name))

        return mappings

    def __call__(self, row: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Transform a row by extracting text columns based on established mappings.

        :param row: Dictionary containing 'items' key with dataset rows
        :return: Mapped dictionary with 'text_column' key
        """
        if self.datasets_column_mappings is None:
            raise ValueError("EmbeddingsColumnMapper not setup with data.")

        items = cast("dict[int, dict[str, Any]]", row.pop("items"))
        mapped: dict[str, Any] = defaultdict(list)

        for column_type, column_mappings in self.datasets_column_mappings.items():
            for dataset_index, dataset_column in column_mappings:
                mapped[column_type].append(items[dataset_index][dataset_column])

        return dict(mapped)

    def setup_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[dict[str, Any]],
    ):
        """
        Initialize column mappings from datasets.

        :param datasets: List of datasets to process
        :param data_args: Arguments for each dataset (unused for this mapper)
        """
        _ = data_args  # Unused for this mapper
        self.datasets_column_mappings = (
            self.datasets_default_mappings(datasets)
            if self.input_mappings is None
            else self.datasets_mappings(datasets, self.input_mappings)
        )
