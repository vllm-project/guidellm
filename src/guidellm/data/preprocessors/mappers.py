from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, ClassVar, TypeAlias, cast

from datasets import Dataset, IterableDataset

from guidellm.data.preprocessors.preprocessor import (
    DataDependentPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import GenerativeDatasetColumnType

__all__ = ["GenerativeColumnMapper"]

# (dataset_column_type, turn)  # noqa: ERA001
DatasetColumnKey: TypeAlias = tuple[GenerativeDatasetColumnType, int]
# (dataset_index, column_name)  # noqa: ERA001
DatasetColumnValue: TypeAlias = tuple[int, str]


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
    column_name_pattern: str = (
        r"^(?P<full_name>(?P<match_name>({name})(es|s)?)([-_](?P<turn>\d+))?)$"
    )

    @staticmethod
    def _filter_for_dataset(names: list[str], *dataset_names: str) -> list[str]:
        filtered_names: list[str] = []
        for name in names:
            if "." in name:
                dataset_part, column_part = name.split(".", 1)
                if dataset_part in dataset_names:
                    filtered_names.append(column_part)
            else:
                filtered_names.append(name)

        return filtered_names

    @staticmethod
    def _extract_turn_columns(
        turn_pattern: str, columns_str: str
    ) -> list[tuple[int, str]]:
        # Now find all columns that match a variant of the base name
        turn_matches = re.finditer(turn_pattern, columns_str, re.M | re.I)

        turn_columns: list[tuple[int, str]] = []
        turn_count = 0
        for match in turn_matches:
            column_name = match.group("full_name")
            if not column_name:
                continue

            turn_str = match.group("turn")
            turn = int(turn_str) if turn_str is not None else turn_count
            turn_columns.append((turn, column_name))
            turn_count += 1

        return turn_columns

    @classmethod
    def datasets_mappings(
        cls,
        datasets: list[Dataset | IterableDataset],
        input_mappings: dict[str, str | list[str]] | None = None,
    ) -> dict[DatasetColumnKey, list[DatasetColumnValue]]:
        mappings: dict[DatasetColumnKey, list[DatasetColumnValue]] = defaultdict(list)
        input_map: dict[str, list[str]] = cls.defaults
        if input_mappings:
            input_map = {
                k: v if isinstance(v, list) else [v] for k, v in input_mappings.items()
            }

        for index, dataset in enumerate(datasets):
            dataset_name = (
                dataset.info.dataset_name
                if dataset.info and dataset.info.dataset_name
                else index
            )
            dataset_columns = dataset.column_names or list(next(iter(dataset)).keys())
            dataset_columns_str = "\n".join(dataset_columns)

            for column_type, names in input_map.items():
                filtered_names = cls._filter_for_dataset(
                    names, str(index), str(dataset_name)
                )
                if not filtered_names:
                    continue

                column_pattern = cls.column_name_pattern.format(
                    name="|".join(re.escape(n) for n in filtered_names)
                )
                # Find the first matching column name
                base_match = re.search(column_pattern, dataset_columns_str, re.M | re.I)
                if not base_match:
                    continue

                turn_pattern = cls.column_name_pattern.format(
                    name=base_match.group("match_name"),
                )
                turn_columns = cls._extract_turn_columns(
                    turn_pattern,
                    dataset_columns_str,
                )

                # Re-enumerate to ensure we don't have a gap in turns
                for turn, (_, column_name) in enumerate(sorted(turn_columns)):
                    column_type = cast("GenerativeDatasetColumnType", column_type)
                    mappings[(column_type, turn)].append((index, column_name))

        return mappings

    def __init__(
        self,
        column_mappings: dict[str, str | list[str]] | None = None,
        **_: Any,  # Ignore global kwargs
    ):
        self.input_mappings = column_mappings
        self.datasets_column_mappings: (
            dict[DatasetColumnKey, list[DatasetColumnValue]] | None
        )

    def __call__(self, items: list[dict[str, Any]]) -> list[dict[str, list[Any]]]:
        if self.datasets_column_mappings is None:
            raise ValueError("DefaultGenerativeColumnMapper not setup with data.")

        mapped: list[dict[str, Any]] = []

        for (column_type, turn), column_mappings in sorted(
            self.datasets_column_mappings.items()
        ):
            # Ensure the mapped list has enough turns for this turn
            # Should never need to happen
            while len(mapped) <= turn:
                mapped.append(defaultdict(list))

            for (
                dataset_index,
                dataset_column,
            ) in column_mappings:
                mapped[turn][column_type].append(
                    items[dataset_index]["dataset"][dataset_column]
                )

        return [dict(m) for m in mapped if len(m) > 0]

    def setup_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[dict[str, Any]],
    ):
        _ = data_args  # Unused for this mapper
        self.datasets_column_mappings = self.datasets_mappings(
            datasets, self.input_mappings
        )
