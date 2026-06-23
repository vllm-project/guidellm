from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, ClassVar, Literal, TypeAlias, cast

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import Field

from guidellm.data.preprocessors.preprocessor import (
    DataDependentPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import (
    DataPreprocessorArgs,
    DatasetType,
    GenerativeDatasetColumnType,
)

__all__ = [
    "GenerativeColumnMapper",
    "GenerativeColumnMapperArgs",
    "PoolingColumnMapper",
]

# dataset_column_type and turn index
DatasetColumnKey: TypeAlias = tuple[GenerativeDatasetColumnType, int]
# dataset index and column_name
DatasetColumnValue: TypeAlias = tuple[int, str]


def _unwrap_dataset_dict(
    dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict,
) -> Dataset | IterableDataset:
    """Unwrap a DatasetDict/IterableDatasetDict into a single split.

    Prefers the ``"train"`` split if available, otherwise picks the first split.
    Returns the input unchanged if it is already a single Dataset/IterableDataset.

    :param dataset: The dataset or dataset dict to unwrap.
    :return: A single Dataset or IterableDataset.
    """
    if isinstance(dataset, DatasetDict | IterableDatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset))]
    return dataset


def _detect_json_wrapper(
    dataset: Dataset | IterableDataset, dataset_columns: list[str]
) -> str | None:
    """Check if a dataset has a single string column containing JSON dicts.

    :param dataset: The dataset to inspect.
    :param dataset_columns: The column names present in the dataset.
    :return: The wrapper column name if detected, or None.
    """
    if len(dataset_columns) != 1:
        return None

    candidate = dataset_columns[0]
    sample = next(iter(dataset))
    value = sample[candidate]
    if not isinstance(value, str):
        return None

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(parsed, dict) and parsed:
        return candidate
    return None


def _resolve_virtual_columns(
    dataset: Dataset | IterableDataset, wrapper_column: str
) -> list[str]:
    """Parse the first row's JSON wrapper and return its inner keys.

    :param dataset: The dataset to peek at.
    :param wrapper_column: The name of the column containing the JSON string.
    :return: List of inner key names from the parsed JSON dict.
    """
    sample = next(iter(dataset))
    parsed = json.loads(sample[wrapper_column])
    return list(parsed.keys())


def _extract_json_field(
    row_data: dict[str, Any], wrapper_column: str, field: str
) -> Any:
    """Parse a JSON wrapper column and extract a specific inner field.

    :param row_data: The raw row dict from the dataset.
    :param wrapper_column: The column name containing the JSON string.
    :param field: The key to extract from the parsed JSON dict.
    :return: The value of the requested field, or None if not present.
    """
    raw = row_data.get(wrapper_column)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed.get(field)


@DataPreprocessorArgs.register(
    [
        "generative_column_mapper",
        "pooling_column_mapper",
    ]
)
class GenerativeColumnMapperArgs(DataPreprocessorArgs):
    """Model for generative column mapper preprocessor arguments."""

    kind: Literal["generative_column_mapper", "pooling_column_mapper"] = Field(
        default="generative_column_mapper",
        description="Type identifier for the generative column mapper preprocessor.",
    )
    column_mappings: dict[str, str | list[str]] | None = Field(
        default=None,
        description="Mappings for the column names.",
        examples=[
            {
                "prompt_tokens_count_column": [
                    "prompt_tokens_count",
                    "input_tokens_count",
                ],
                "output_tokens_count_column": [
                    "output_tokens_count",
                    "completion_tokens_count",
                ],
            }
        ],
    )


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
        "tools_column": [
            "tools",
            "functions",
            "tool_definitions",
        ],
        "tool_response_column": [
            "tool_response",
            "tool_result",
            "tool_output",
        ],
        "turn_type_column": [
            "turn_type",
        ],
        "relative_timestamp_column": ["relative_timestamp"],
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
    ) -> tuple[dict[DatasetColumnKey, list[DatasetColumnValue]], dict[int, str]]:
        """
        Resolve column mappings across one or more datasets.

        For each dataset, matches actual column names against the requested
        mapping names (or :attr:`defaults`) using regex patterns that account
        for pluralisation and turn suffixes (e.g. ``prompt-0``, ``prompt-1``).

        When a dataset has no direct column matches but contains a single
        JSON-string column, the inner keys of that JSON are used as virtual
        column names and matching is retried.

        :param datasets: The loaded datasets to inspect for column names.
        :param input_mappings: Optional explicit column mappings. When ``None``,
            :attr:`defaults` is used. Values may be a single name or a list of
            candidate names in priority order.
        :return: A tuple of (mappings, json_wrappers) where mappings is a dict
            keyed by ``(column_type, turn_index)`` whose values are lists of
            ``(dataset_index, column_name)`` pairs, and json_wrappers is a dict
            mapping dataset_index to the wrapper column name for datasets that
            required JSON unwrapping.
        """
        mappings: dict[DatasetColumnKey, list[DatasetColumnValue]] = defaultdict(list)
        json_wrappers: dict[int, str] = {}
        input_map: dict[str, list[str]] = cls.defaults
        if input_mappings:
            input_map = {
                k: v if isinstance(v, list) else [v] for k, v in input_mappings.items()
            }

        for index, raw_dataset in enumerate(datasets):
            dataset = _unwrap_dataset_dict(raw_dataset)
            dataset_name = (
                dataset.info.dataset_name
                if dataset.info and dataset.info.dataset_name
                else index
            )
            dataset_columns = dataset.column_names or list(next(iter(dataset)).keys())
            dataset_columns_str = "\n".join(dataset_columns)

            matched = cls._match_columns(
                index, input_map, dataset_name, dataset_columns_str
            )

            # Fallback: if no matches found, try JSON unwrapping
            if not matched:
                wrapper = _detect_json_wrapper(dataset, dataset_columns)
                if wrapper:
                    virtual_columns = _resolve_virtual_columns(dataset, wrapper)
                    virtual_columns_str = "\n".join(virtual_columns)
                    matched = cls._match_columns(
                        index, input_map, dataset_name, virtual_columns_str
                    )
                    if matched:
                        json_wrappers[index] = wrapper

            for key, values in matched.items():
                mappings[key].extend(values)

        return mappings, json_wrappers

    @classmethod
    def _match_columns(
        cls,
        index: int,
        input_map: dict[str, list[str]],
        dataset_name: str | int,
        dataset_columns_str: str,
    ) -> dict[DatasetColumnKey, list[DatasetColumnValue]]:
        """Match input_map names against dataset columns using regex patterns.

        :param index: The dataset index in the multi-dataset list.
        :param input_map: Mapping of column types to candidate column names.
        :param dataset_name: Name or index of the dataset for filtering.
        :param dataset_columns_str: Newline-joined string of column names.
        :return: Dict of matched (column_type, turn) -> [(index, column_name)].
        """
        matched: dict[DatasetColumnKey, list[DatasetColumnValue]] = defaultdict(list)

        for column_type, names in input_map.items():
            filtered_names = cls._filter_for_dataset(
                names, str(index), str(dataset_name)
            )
            if not filtered_names:
                continue

            column_pattern = cls.column_name_pattern.format(
                name="|".join(re.escape(n) for n in filtered_names)
            )
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

            for turn, column_name in sorted(turn_columns):
                column_type = cast("GenerativeDatasetColumnType", column_type)
                matched[(column_type, turn)].append((index, column_name))

        return matched

    def __init__(
        self,
        config: GenerativeColumnMapperArgs,
    ):
        self.input_mappings = config.column_mappings
        self.datasets_column_mappings: (
            dict[DatasetColumnKey, list[DatasetColumnValue]] | None
        )
        self._json_wrappers: dict[int, str] = {}

    def _get_column_value(
        self, items: list[dict[str, Any]], dataset_index: int, dataset_column: str
    ) -> Any:
        """Read a column value, unwrapping JSON if needed for this dataset.

        :param items: The per-dataset row items from the iterator.
        :param dataset_index: Index of the dataset to read from.
        :param dataset_column: Column name (possibly virtual) to extract.
        :return: The column value from the row.
        """
        row_data = items[dataset_index]["dataset"]
        wrapper = self._json_wrappers.get(dataset_index)
        if wrapper is not None:
            return _extract_json_field(row_data, wrapper, dataset_column)
        return row_data[dataset_column]

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
                    self._get_column_value(items, dataset_index, dataset_column)
                )

        return [dict(m) for m in mapped if len(m) > 0]

    def setup_data(
        self,
        datasets: list[DatasetType],
    ):
        self.datasets_column_mappings, self._json_wrappers = self.datasets_mappings(
            datasets, self.input_mappings
        )

        if not self.datasets_column_mappings:
            raise ValueError(
                "GenerativeColumnMapper found no matching columns. "
                f"Requested mappings: {self.input_mappings or 'default mappings'}. "
                "Every row will produce an empty result."
            )


@PreprocessorRegistry.register("pooling_column_mapper")
class PoolingColumnMapper(GenerativeColumnMapper):
    defaults: ClassVar[dict[str, list[str]]] = {"pooling_column": ["prompt"]}
