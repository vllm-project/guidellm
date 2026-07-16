from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal, TypeVar

import torch
from faker import Faker
from pydantic import Field
from torch.utils.data.dataloader import DataLoader as PyTorchDataLoader
from torch.utils.data.dataset import IterableDataset as TorchIterableDataset

from guidellm.data.finalizers import DatasetFinalizer
from guidellm.data.loaders.loader import DataLoader, DataLoaderRegistry
from guidellm.data.preprocessors import (
    DataDependentPreprocessor,
    DatasetPreprocessor,
)
from guidellm.data.schemas import (
    DataLoaderArgs,
    DatasetType,
)
from guidellm.logger import logger
from guidellm.schemas.conversation_graph import GenerativeConversationGraph
from guidellm.schemas.info import RequestSettings
from guidellm.schemas.request import GenerationRequest
from guidellm.utils.mixins import InfoMixin

__all__ = ["DatasetsIterator", "TorchDataLoader", "TorchDataLoaderArgs"]


def _collate_first(batch: list) -> Any:
    return batch[0]


@DataLoaderArgs.register("pytorch")
class TorchDataLoaderArgs(DataLoaderArgs):
    """Model for PyTorch data loader arguments."""

    kind: Literal["pytorch"] = Field(  # type: ignore[assignment]
        default="pytorch",
        description="Type identifier for the generative data loader.",
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle data rows at every epoch.",
    )
    num_workers: int = Field(
        default=1,
        description=(
            "Number of worker processes for data loading. If 0, data loading "
            "will be performed in the main process."
        ),
    )


DataT = TypeVar("DataT")


class DatasetsIterator(TorchIterableDataset[DataT]):
    def __init__(
        self,
        datasets: list[DatasetType],
        data_samples: int,
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        finalizer: DatasetFinalizer[DataT],
        branch_specs: list[dict[str, Any]] | None = None,
    ):
        self.datasets = datasets
        self.preprocessors = preprocessors
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, DataDependentPreprocessor):
                preprocessor.setup_data(
                    datasets=self.datasets,
                )
        self.finalizer = finalizer
        self.branch_specs = branch_specs or []
        self._faker = Faker() if branch_specs else None
        self.precache: list[Any] | None = (
            list(self.generator(data_samples)) if data_samples else None
        )
        self.epoch = 0

    _MIN_TOKENS_FOR_SCALED_TEXT = 20
    _TOKENS_PER_PARAGRAPH = 65

    def _make_branch_request(
        self,
        branch_index: int,
        turn_index: int,  # noqa: ARG002
    ) -> tuple[GenerationRequest, RequestSettings]:
        """Generate a synthetic request/settings pair for a branch turn.

        Uses the branch's configured prompt/output token counts if
        available, falling back to generating a paragraph of text.
        """
        spec = (
            self.branch_specs[branch_index]
            if branch_index < len(self.branch_specs)
            else {}
        )
        prompt_tokens = spec.get("prompt_tokens")
        output_tokens = spec.get("output_tokens")

        # Generate text roughly matching the target token count
        if prompt_tokens and prompt_tokens > self._MIN_TOKENS_FOR_SCALED_TEXT:
            num_paragraphs = max(1, prompt_tokens // self._TOKENS_PER_PARAGRAPH)
            text = " ".join(
                self._faker.paragraph(nb_sentences=5) for _ in range(num_paragraphs)
            )
        else:
            text = self._faker.paragraph(nb_sentences=3)

        columns: dict[str, list[Any]] = {"text_column": [text]}
        if output_tokens:
            columns["output_tokens_count_column"] = [output_tokens]

        return GenerationRequest(columns=columns), RequestSettings()

    def __iter__(self) -> Iterator[DataT]:
        worker_info = torch.utils.data.get_worker_info()
        worker_modulus = worker_info.num_workers if worker_info is not None else 1
        worker_index = worker_info.id if worker_info is not None else 0

        if self.precache:
            for index, item in enumerate(self.precache):
                if (index + worker_index) % worker_modulus == 0:
                    yield item
        else:
            yield from self.generator(
                modulus=worker_modulus, offset=worker_index, epoch=self.epoch
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generator(  # noqa: C901, PLR0912
        self,
        max_items: int | None = None,
        modulus: int | None = None,
        offset: int | None = None,
        epoch: int = 0,
    ) -> Iterator[DataT]:
        gen_count = 0
        yield_count = 0
        error_count = 0

        with contextlib.suppress(StopIteration):
            dataset_iters = []
            for dataset in self.datasets:
                if hasattr(dataset, "set_epoch"):
                    with contextlib.suppress(Exception):
                        dataset.set_epoch(epoch)
                dataset_iters.append(iter(dataset))

            while max_items is None or gen_count < max_items:
                try:
                    row: list[dict[str, Any]] = [
                        {"dataset": next(dataset_iter)}
                        for dataset_iter in dataset_iters
                    ]
                    gen_count += 1

                    if (
                        modulus is not None
                        and offset is not None
                        and (gen_count % modulus) != offset
                    ):
                        continue

                    # Apply preprocessors in sequence
                    for preprocessor in self.preprocessors:
                        row = preprocessor(row)

                    result = self.finalizer(row)
                    # Filter empty results (e.g. column mapper matched
                    # no columns, so finalizer returned an empty list)
                    if not result:
                        continue

                    # Wrap linear chain as a ConversationGraph.
                    # Keep (request, settings) pairs so node.settings is
                    # populated from scheduling metadata, not RequestT.
                    if isinstance(result, list):
                        if self.branch_specs:
                            result = (
                                GenerativeConversationGraph
                                .from_linear_chain_with_branches(
                                    main_requests=result,
                                    branches=self.branch_specs,
                                    branch_request_factory=(
                                        self._make_branch_request
                                    ),
                                )
                            )
                        else:
                            result = (
                                GenerativeConversationGraph.from_linear_chain(
                                    result
                                )
                            )

                    yield result
                    yield_count += 1
                except StopIteration:
                    raise  # Stop iteration when any dataset is exhausted
                except Exception as err:  # noqa: BLE001 # Exception logged
                    error_count += 1
                    logger.error(
                        "Skipping data row due to error: {}. "
                        "Check data format and preprocessor configuration.",
                        err,
                    )
                    gen_count -= 1

        if gen_count > 0 and yield_count == 0:
            raise ValueError(
                f"Dataset iterator processed {gen_count} rows but yielded "
                f"zero results ({error_count} errors; {gen_count - error_count} "
                f"empty). Check your data and data arguments."
            )

        if max_items is not None and gen_count < max_items:
            raise ValueError(
                f"Requested {max_items} samples, but only {gen_count} "
                "available from the provided datasets."
            )


@DataLoaderRegistry.register("pytorch")
class TorchDataLoader(PyTorchDataLoader[DataT], InfoMixin, DataLoader[DataT]):
    def __init__(
        self,
        config: TorchDataLoaderArgs,
        datasets: list[DatasetType],
        preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor],
        finalizer: DatasetFinalizer[DataT],
        random_seed: int = 42,
        branch_specs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        iterator: DatasetsIterator[DataT] = DatasetsIterator(
            datasets=datasets,
            data_samples=config.samples,
            preprocessors=preprocessors,
            finalizer=finalizer,
            branch_specs=branch_specs,
        )
        self._info: dict[str, Any] = config.model_dump(mode="json")
        self.epoch = 0

        gen = torch.Generator()
        gen.manual_seed(random_seed)
        super().__init__(
            dataset=iterator,
            batch_size=1,
            shuffle=config.shuffle,
            collate_fn=_collate_first,
            num_workers=config.num_workers,
            generator=gen,
            **kwargs,
        )

    def __iter__(self):
        if isinstance(self.dataset, DatasetsIterator):
            self.dataset.set_epoch(self.epoch)
        self.epoch += 1

        return super().__iter__()

    @property
    def info(self) -> dict[str, Any]:
        return self._info
