from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from datasets import Dataset, IterableDataset

from guidellm.data.objects import (
    GenerativeDatasetArgs,
    GenerativeDatasetColumnType,
)
from guidellm.data.preprocessors.objects import DatasetPreprocessor
from guidellm.data.utils import DEFAULT_COLUMN_NAMES

__all__ = ["ColumnMapping", "GenerativeColumnMapper"]


@dataclass
class ColumnMapping:
    indices: list[int]
    names: list[str]


class GenerativeColumnMapper(DatasetPreprocessor):
    def __init__(self):
        self.datasets: list[Dataset | IterableDataset] | None = None
        self.data_args: list[GenerativeDatasetArgs] | None = None
        self.column_mappings: (
            dict[GenerativeDatasetColumnType, ColumnMapping | None] | None
        ) = None

    def __call__(
        self, row: dict[Literal["items"], tuple[dict[str, Any]]]
    ) -> dict[str, Any]:
        if (
            self.datasets is None
            or self.data_args is None
            or self.column_mapping is None
        ):
            raise ValueError("GenerativeColumnMapper not initialized with data.")

        mapped: dict[GenerativeDatasetColumnType, list[Any]] = {}
        items = row.pop("items")

        for column_type, column_mapping in self.column_mapping.items():
            mapped[column_type] = [
                items[index].get(name)
                for index, name in zip(column_mapping.indices, column_mapping.names)
            ]

        return mapped

    def init_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[GenerativeDatasetArgs],
    ):
        self.datasets = datasets
        self.data_args = data_args
        self.column_mapping = self.generate_column_mapping()

    def generate_column_mapping(
        self,
    ) -> dict[GenerativeDatasetColumnType, ColumnMapping]:
        mappings: dict[GenerativeDatasetColumnType, ColumnMapping] = {}
        # Map any columns specified in the GenerativeDatasetArgs first
        self._fill_mappings_from_data_args(mappings)
        # For standard column types not mapped, fill in first one found from defaults
        self._fill_mappings_from_defaults(mappings)

        return mappings

    def _fill_mappings_from_data_args(
        self, mappings: dict[GenerativeDatasetColumnType, ColumnMapping]
    ):
        for index, args in enumerate(self.data_args):
            args_column_mappings = args.get_mapped_columns()
            for column_type, column_name in args_column_mappings.items():
                if column_type not in mappings:
                    mappings[column_type] = ColumnMapping(indices=[], names=[])
                column_mapping = mappings[column_type]

                for name in (
                    column_name if isinstance(column_name, list) else [column_name]
                ):
                    if name not in self.datasets[index].column_names:
                        raise ValueError(
                            f"Column '{name}' not found in dataset columns: "
                            f"{self.datasets[index].column_names}"
                        )
                    column_mapping.indices.append(index)
                    column_mapping.names.append(name)

    def _fill_mappings_from_defaults(
        self, mappings: dict[GenerativeDatasetColumnType, ColumnMapping]
    ):
        for column_type, default_names in DEFAULT_COLUMN_NAMES.items():
            if column_type in mappings:
                continue

            for index, dataset in enumerate(self.datasets):
                for name in default_names:
                    if name in dataset.column_names:
                        mappings[column_type] = ColumnMapping(
                            indices=[index], names=[name]
                        )
                        break
                    # Check for plural form of the name
                    if f"{name}s" in dataset.column_names:
                        mappings[column_type] = ColumnMapping(
                            indices=[index], names=[f"{name}s"]
                        )
                        break
                if column_type in mappings:
                    break
