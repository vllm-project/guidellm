from abc import abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

from guidellm.config import settings
from guidellm.dataset import ColumnInputTypes, load_dataset
from guidellm.objects import StandardBaseModel
from guidellm.request.request import GenerationRequest

__all__ = [
    "GenerativeRequestLoader",
    "GenerativeRequestLoaderDescription",
    "RequestLoader",
    "RequestLoaderDescription",
]


class RequestLoaderDescription(StandardBaseModel):
    type_: Literal["request_loader"] = "request_loader"


class RequestLoader(Iterable):
    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def description(self) -> RequestLoaderDescription: ...


class GenerativeRequestLoaderDescription(RequestLoaderDescription):
    type_: Literal["generative_request_loader"] = "generative_request_loader"  # type: ignore[assignment]
    data: str
    data_args: Optional[dict[str, Any]]
    processor: str
    processor_args: Optional[dict[str, Any]]


class GenerativeRequestLoader(RequestLoader):
    DEFAULT_PROMPT_COLUMNS = [
        "prompt",
        "prompts",
        "instruction",
        "instructions",
        "question",
        "questions",
        "input",
        "inputs",
        "context",
        "content",
        "conversation",
        "conversations",
        "turn",
        "turns",
        "text",
    ]

    def __init__(
        self,
        data: Union[
            str,
            Path,
            Iterable[Union[str, dict[str, Any]]],
            Dataset,
            DatasetDict,
            IterableDataset,
            IterableDatasetDict,
        ],
        data_args: Optional[dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[dict[str, Any]],
        shuffle: bool = True,
        iter_type: Literal["finite", "infinite"] = "finite",
        random_seed: int = 42,
    ):
        self.data = data
        self.data_args = data_args
        dataset, args_column_mappings = load_dataset(
            data,
            data_args,
            processor,
            processor_args,
            random_seed,
        )
        self.dataset = dataset
        self.processor = processor
        self.processor_args = processor_args
        self.shuffle = shuffle
        self.iter_type = iter_type
        self.random_seed = random_seed

        self.column_mappings = self._create_column_mappings(args_column_mappings)
        self.preserve_iter_state = iter_type == "infinite"  # ensure no caching requests
        self._preserved_iter = None

    def __iter__(self) -> Iterator[GenerationRequest]:
        scope_create_count = 0

        while (dataset_iter := self._get_dataset_iter(scope_create_count)) is not None:
            scope_create_count += 1

            for item in dataset_iter:
                yield self._create_request(item)

            self._preserved_iter = None

    def __len__(self) -> int:
        if self.iter_type == "finite":
            return self.num_unique_items()

        raise ValueError(f"Unable to determine length of dataset: {self.data}")

    @property
    def description(self) -> GenerativeRequestLoaderDescription:
        return GenerativeRequestLoaderDescription(
            data=str(self.data),
            data_args=self.data_args,
            processor=str(self.processor),
            processor_args=self.processor_args,
        )

    def num_unique_items(self, raise_err: bool = True) -> int:
        try:
            return len(self.dataset)
        except Exception:  # noqa: BLE001, S110
            pass

        dataset_size = self.dataset.info.dataset_size
        if dataset_size is not None:
            return dataset_size

        if raise_err:
            raise ValueError("Unable to determine number of items in the dataset")

        return -1

    def _create_column_mappings(
        self,
        args_column_mappings: dict[ColumnInputTypes, str],
    ) -> dict[ColumnInputTypes, str]:
        column_mappings: dict[ColumnInputTypes, str] = {}

        if "text_column" in args_column_mappings:
            column_mappings["prompt_column"] = args_column_mappings["text_column"]
        else:
            column_mappings["prompt_column"] = self._extract_text_column()

        if "prompt_tokens_count_column" in args_column_mappings:
            column_mappings["prompt_tokens_count_column"] = args_column_mappings[
                "prompt_tokens_count_column"
            ]
        elif prompt_tokens_count_column := self._extract_prompt_tokens_count_column():
            column_mappings["prompt_tokens_count_column"] = prompt_tokens_count_column

        if "output_tokens_count_column" in args_column_mappings:
            column_mappings["output_tokens_count_column"] = args_column_mappings[
                "output_tokens_count_column"
            ]
        elif output_tokens_count_column := self._extract_output_tokens_count_column():
            column_mappings["output_tokens_count_column"] = output_tokens_count_column

        return column_mappings

    def _extract_text_column(self) -> str:
        column_names = self._dataset_columns(
            err_msg=(
                "Unable to determine text column from dataset and it is required. "
                "To specify the text column, set the 'text_column' key in the "
                "'data_args' dictionary."
            )
        )

        if not column_names:
            raise ValueError(
                "Unable to determine text column from dataset and it is required. "
                "To specify the text column, set the 'text_column' key in the "
                "'data_args' dictionary."
            )

        if len(column_names) == 1:
            return column_names[0]

        for def_column in self.DEFAULT_PROMPT_COLUMNS:
            if def_column in column_names:
                return def_column

        raise ValueError(
            f"Unable to determine text column from dataset columns: {column_names}. "
            "To specify the text column, set the 'text_column' key in the "
            "'data_args' dictionary."
        )

    def _extract_prompt_tokens_count_column(self) -> Optional[str]:
        column_names = self._dataset_columns()

        if column_names and "prompt_tokens_count" in column_names:
            return "prompt_tokens_count"

        if column_names and "prompt_tokens" in column_names:
            return "prompt_tokens"

        return None

    def _extract_output_tokens_count_column(self) -> Optional[str]:
        column_names = self._dataset_columns()

        if column_names and "output_tokens_count" in column_names:
            return "output_tokens_count"

        if column_names and "output_tokens" in column_names:
            return "output_tokens"

        return None

    def _dataset_columns(self, err_msg: Optional[str] = None) -> Optional[list[str]]:
        try:
            column_names = self.dataset.column_names

            if not column_names and err_msg:
                raise ValueError(f"No column names found in dataset: {self.data}")
        except Exception as err:
            if err_msg:
                raise ValueError(err_msg) from err

            column_names = None

        return column_names

    def _get_dataset_iter(
        self, scope_create_count: int
    ) -> Optional[Iterator[dict[str, Any]]]:
        if scope_create_count > 0 and self.iter_type != "infinite":
            return None

        if self.preserve_iter_state and self._preserved_iter is not None:
            return self._preserved_iter

        dataset = (
            self.dataset
            if not self.shuffle
            else self.dataset.shuffle(seed=self.random_seed)
        )

        dataset_iter = iter(dataset)

        if self.preserve_iter_state:
            self._preserved_iter = dataset_iter

        return dataset_iter

    def _create_request(self, item: dict[str, Any]) -> GenerationRequest:
        prompt_tokens = (
            item[self.column_mappings["prompt_tokens_count_column"]]
            if "prompt_tokens_count_column" in self.column_mappings
            else None
        )
        output_tokens = (
            item[self.column_mappings["output_tokens_count_column"]]
            if "output_tokens_count_column" in self.column_mappings
            else None
        )

        return GenerationRequest(
            request_type=settings.preferred_route,
            content=item[self.column_mappings["prompt_column"]],
            stats=(
                {"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {}
            ),
            constraints=(
                {"output_tokens": output_tokens} if output_tokens is not None else {}
            ),
        )
