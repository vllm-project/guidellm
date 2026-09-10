"""Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (consisting of at least the columns timestamp, input_length,
output_length) and yields one row per line with a synthetic prompt matching the
requested input_length for replay benchmarks."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np
from datasets import (
    Dataset,
    DatasetInfo,
    Features,
    IterableDataset,
    Value,
)
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_session_timing import TraceSessionTiming
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas.data.deserializers import TraceDataArgs
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "TraceDatasetDeserializer",
    "TraceFormatBase",
    "TraceFormatRegistry",
    "create_distinct_token_block",
    "create_prompt_from_hash_ids",
    "decode_prompt",
    "generate_token_ids",
    "get_missing_columns",
]


def decode_prompt(
    processor: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    """Decode token ids into a prompt string."""
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


def generate_token_ids(
    token_count: int,
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    margin_of_safety: int = 8,
) -> tuple[int, ...]:
    """Generate `token_count` synthetic token ids for trace prompt construction.

    Ideally, `margin_of_safety` should be set to slighty more than
    the average number of characters used by tokenizers to form one token."""
    attempt = 0
    while True:
        attempt += 1
        # The Faker.text() can only generate text of at least 5 characters.
        num_chars = max(token_count * margin_of_safety * attempt, 5)
        text = faker.text(num_chars)
        token_ids = processor.encode(text)
        if len(token_ids) >= token_count:
            return tuple(token_ids[:token_count])


def get_missing_columns(
    required_columns: list[str], actual_columns: list[str]
) -> list[str]:
    return [c for c in required_columns if c not in actual_columns]


def create_prompt_from_hash_ids(
    hash_ids: list[int],
    hash_id_table: dict[int, tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
) -> str:
    """Returns a synthetic prompt from `hash_ids` using pre-generated token blocks.

    Precondition: All ids in `hash_ids` appear in `hash_id_table`."""
    prompt_token_ids = [
        token for hash_id in hash_ids for token in hash_id_table[hash_id]
    ]
    return decode_prompt(processor, prompt_token_ids)


def create_distinct_token_block(
    block_size: int,
    sibling_token_blocks: set[tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    max_attempts: int = 20,
) -> tuple[int, ...]:
    """Constructs a new token block of `block_size` that does not appear in
    `sibling_token_blocks`."""
    attempt = 0
    while attempt < max_attempts:
        token_ids = generate_token_ids(block_size, processor, faker)
        if token_ids not in sibling_token_blocks:
            return token_ids
        attempt += 1
    raise ValueError(
        f"Failed to generate distinct synthetic token block after {attempt} attempts"
    )


class TraceFormatBase(Protocol):
    config: TraceDataArgs

    def __init__(self, config, dataset: Dataset) -> None: ...

    def __iter__(self) -> Iterable[Dataset]:
        """Returns the next conversation as a `Dataset`."""

    def reset(self) -> None:
        pass

    def required_columns(self) -> Features: ...

    def find_required_columns(self, columns: list[str]) -> list[str]:
        """Checks if all required columns needed by the format exist
        and are located in the expected place."""

    def validate_row(self, row: dict) -> None:
        """Called during iteration via ``_validate_api_row``."""

    def create_prompt(
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        """Called within `trace_common.TraceExamplesIterable` on each iteration.
        Returns a generated synthetic prompt."""

    def build_conversation_graph(
        self,
        conversation: Dataset,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> ConversationGraphData:
        """Build a conversation graph from one ``__iter__`` conversation.

        The default emits a linear ``main_*`` chain. Formats with branches
        or subagents should override this rather than branching in the
        shared iterable.
        """
        start_ts = conversation[0][self.config.timestamp_column]
        turns = []
        for turn_idx, turn in enumerate(conversation):
            parents = []
            if turn_idx > 0:
                parents.append(
                    ConversationParentRef(parent_node_id=f"main_{turn_idx - 1}")
                )

            _validate_api_row(turn, self.config, self.validate_row)
            prompt = self.create_prompt(turn, processor, faker)
            relative_timestamp = turn[self.config.timestamp_column] - start_ts
            columns = {
                "text_column": [prompt],
                "prompt_tokens_count_column": [turn[self.config.prompt_tokens_column]],
                "output_tokens_count_column": [turn[self.config.output_tokens_column]],
                "relative_timestamp_column": [relative_timestamp],
            }
            turns.append(
                ConversationTurnData(
                    node_id=f"main_{turn_idx}",
                    agent_id="default",
                    parents=parents,
                    columns=columns,
                )
            )
        return ConversationGraphData(turns=turns)


class TraceFormatRegistry(RegistryMixin[type[TraceFormatBase]]):
    @classmethod
    def dispatch(cls, config: TraceDataArgs, dataset: Dataset) -> TraceFormatBase:
        format_from_type = cls.get_registered_object(config.kind)
        if format_from_type is None:
            raise DataNotSupportedError(
                f"Format type '{config.kind}' is not registered."
            )
        return format_from_type(config, dataset)


class TraceExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: TraceDataArgs,
        trace_format: TraceFormatBase,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.format = trace_format
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        samples_count = 0
        # Fresh instance per iteration so packing state does not leak across epochs.
        timing = TraceSessionTiming(
            max_wait=self.config.max_wait,
            max_session_wait=self.config.max_session_wait,
            min_concurrent_sessions=self.config.min_concurrent_sessions,
            time_scale=self.config.time_scale,
        )
        for conv in self.format:  # type: ignore[attr-defined]
            graph_data = self.format.build_conversation_graph(
                conv, self.processor, self.faker
            )
            timing.apply(graph_data)
            samples_count += len(graph_data.turns)
            payload = json.dumps(graph_data.model_dump(mode="json"))
            yield (
                samples_count,
                {
                    "conversation_turns": (
                        payload.decode() if isinstance(payload, bytes) else payload
                    )
                },
            )
            self.format.reset()

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features({"conversation_turns": Value("large_string")})

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> TraceExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> TraceExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self):
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class TraceDataset(IterableDataset):
    def __init__(
        self,
        config: TraceDataArgs,
        trace_format: TraceFormatBase,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        ex_iterable = TraceExamplesIterable(
            config, trace_format, processor, random_seed
        )
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic trace dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if hasattr(self._ex_iterable, "iteration_count"):
            self._ex_iterable.iteration_count = epoch


def _validate_api_row(
    row: dict,
    config: TraceDataArgs,
    validate_row: Callable[[dict], None],
) -> None:
    """Validate one API request row during iteration."""
    _validate_row(row, config)
    validate_row(row)


def _validate_row(row: dict, config: TraceDataArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    if n_in < 0 or n_out < 0:
        raise InvalidRowError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )


def _handle_column_search(config: TraceDataArgs, trace_format: TraceFormatBase) -> None:
    features = Features(
        {
            config.timestamp_column: Value("float"),
            config.prompt_tokens_column: Value("int32"),
            config.output_tokens_column: Value("int32"),
            **dict(trace_format.required_columns()),
        }
    )
    missing = trace_format.find_required_columns(list(features.keys()))
    if missing:
        raise DataNotSupportedError(f"Trace missing required columns: {missing}")


@DatasetDeserializerFactory.register(["trace_synthetic"])
class TraceDatasetDeserializer(DatasetDeserializer):
    """Dataset deserializer for all trace formats."""

    def __call__(
        self,
        config: TraceDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int = 42,
    ) -> IterableDataset:
        try:
            dataset = DatasetDeserializerFactory.deserialize(
                config=config.source,
                processor_factory=processor_factory,
                random_seed=random_seed,
            )
        except ValueError as e:
            raise DataNotSupportedError(str(e)) from e
        if not dataset:
            raise DataNotSupportedError(
                f"Trace file has no valid rows: {config.source}"
            )
        trace_format = TraceFormatRegistry.dispatch(config, dataset)
        _handle_column_search(config, trace_format)
        return TraceDataset(config, trace_format, processor_factory(), random_seed)
