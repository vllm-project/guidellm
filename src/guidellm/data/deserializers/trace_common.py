"""Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (consisting of at least the columns timestamp, input_length,
output_length) and yields one row per line with a synthetic prompt matching the
requested input_length for replay benchmarks."""

from __future__ import annotations

import bisect
import importlib.resources
import math
import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple, Protocol

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
from guidellm.data.deserializers.trace_session_timing import (
    TraceSessionTiming,
    graph_max_timestamp,
    graph_min_timestamp,
    shift_graph_timestamps,
)
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.logger import logger
from guidellm.schemas.data.deserializers import TraceDataArgs
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "EnglishTokenBuffer",
    "HashTokenBlock",
    "TraceDatasetDeserializer",
    "TraceFormatBase",
    "TraceFormatRegistry",
    "create_distinct_token_block",
    "create_prompt_from_hash_ids",
    "decode_prompt",
    "decodes_hash_blocks_concatenatively",
    "duration_columns",
    "fill_hash_id_table",
    "generate_token_ids",
    "get_missing_columns",
    "sample_english_word",
]

# Offset from the dataset seed so the probe Faker does not alias a replay copy.
# Replay copies use ``random_seed + copy_index * 1_000_003``.
_PROBE_SEED_OFFSET = 1_000_003_007
_MIN_PROBE_BLOCKS = 2
_REPLACEMENT_CHAR = "\ufffd"


class HashTokenBlock(NamedTuple):
    """One synthetic hash-id block: token ids and the decode of those ids.

    ``text`` is computed once, when the block is inserted into the hash table.
    """

    token_ids: tuple[int, ...]
    text: str


def decode_prompt(
    processor: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    """Decode token ids into a prompt string."""
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


_MIN_ENGLISH_WORDS = 2


def _load_english_words() -> tuple[str, ...]:
    """Load the packaged frequency-ordered English word list.

    Lines are most common first. Comments and tokens that are not lowercase
    ASCII letters are skipped. Order of first occurrence is kept.

    :return: English words used for Zipf prompt sampling.
    """
    text = (
        importlib.resources.files("guidellm.data.deserializers")
        .joinpath("english_words.txt")
        .read_text(encoding="utf-8")
    )
    words: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        word = line.strip()
        if not word or word.startswith("#"):
            continue
        if not word.isascii() or not word.isalpha() or not word.islower():
            continue
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
    if len(words) < _MIN_ENGLISH_WORDS:
        raise RuntimeError("English word list must contain at least two words")
    return tuple(words)


# Most-common-first English words. Synthetic prompts sample this list with
# Zipf weights so frequent words stay frequent, and so blocks stay in one
# language instead of mixing scripts from a multilingual vocab. The file
# extends well past the most common words so rare ranks can be drawn.
_ENGLISH_WORDS: tuple[str, ...] = _load_english_words()

# Words per encode call. One chunk covers many hash blocks.
_ENGLISH_CHUNK_WORDS = 512


def _zipf_cumulative(word_count: int) -> tuple[float, ...]:
    """Cumulative Zipf masses for ranks ``1 .. word_count`` (weight ``1/rank``)."""
    weights = [1.0 / rank for rank in range(1, word_count + 1)]
    total = sum(weights)
    running = 0.0
    cumulative: list[float] = []
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    return tuple(cumulative)


_ZIPF_CUMULATIVE = _zipf_cumulative(len(_ENGLISH_WORDS))


def sample_english_word(rng: random.Random) -> str:
    """Draw one English word with probability proportional to ``1/rank``.

    Rank 1 is :data:`_ENGLISH_WORDS` index 0, the most common word.

    :param rng: Generator advanced by this draw.
    :return: A lowercase ASCII English word from the frequency list.
    """
    index = bisect.bisect_left(_ZIPF_CUMULATIVE, rng.random())
    if index >= len(_ENGLISH_WORDS):
        index = len(_ENGLISH_WORDS) - 1
    return _ENGLISH_WORDS[index]


class EnglishTokenBuffer:
    """Cursor over token ids from Zipf-sampled English text.

    Text is encoded in chunks. Callers take the next ``count`` ids, which
    keeps block generation off the per-id encode path and off the multilingual
    vocab.
    """

    def __init__(self) -> None:
        self._tokens: list[int] = []
        self._cursor = 0

    def reset(self) -> None:
        """Drop buffered ids so the next draw starts a new stretch of text."""
        self._tokens.clear()
        self._cursor = 0

    def take(
        self,
        count: int,
        processor: PreTrainedTokenizerBase,
        rng: random.Random,
    ) -> tuple[int, ...]:
        """Return the next ``count`` token ids from the English stream.

        :param count: Number of ids to return. Zero yields an empty tuple.
        :param processor: Tokenizer used when the buffer must be extended.
        :param rng: Generator for the Zipf word draws in an extension.
        :return: ``count`` token ids from encoded English text.
        """
        if count <= 0:
            return ()
        while self._cursor + count > len(self._tokens):
            self._extend(processor, rng)
        start = self._cursor
        self._cursor += count
        return tuple(self._tokens[start : self._cursor])

    def _extend(self, processor: PreTrainedTokenizerBase, rng: random.Random) -> None:
        words = [sample_english_word(rng) for _ in range(_ENGLISH_CHUNK_WORDS)]
        # Spaces stay in the encoded text so decoded blocks read as English words.
        encoded = processor.encode(" ".join(words))
        if not encoded:
            raise ValueError("Tokenizer encoded English text to zero tokens")
        self._tokens.extend(encoded)


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


def duration_columns(row: dict, config: TraceDataArgs) -> dict[str, list[float]]:
    """Map an optional duration column onto request scheduling columns.

    :param row: Trace row that may contain ``config.duration_column``.
    :param config: Trace format arguments naming that column.
    :return: ``{"request_duration_column": [seconds]}`` when the column is
        present and non-null, otherwise an empty dict.
    """
    column = config.duration_column
    if column not in row or row[column] is None:
        return {}
    return {"request_duration_column": [float(row[column])]}


def create_prompt_from_hash_ids(
    hash_ids: list[int],
    hash_id_table: dict[int, HashTokenBlock],
    processor: PreTrainedTokenizerBase,
    *,
    join_decoded_blocks: bool = False,
) -> str:
    """Return a synthetic prompt from ``hash_ids`` using pre-generated token blocks.

    When ``join_decoded_blocks`` is true, the prompt is the concatenation of
    each block's cached decode. That is equal to decoding the concatenated
    ids only for tokenizers that do not insert or merge text at piece
    boundaries. A block whose decode contains U+FFFD is not closed (a later
    token can complete a partial code unit), so that turn is decoded as one
    sequence even if the tokenizer-level gate is on.

    Precondition: every id in ``hash_ids`` is present in ``hash_id_table``.

    :param hash_ids: Ordered hash IDs for one prompt.
    :param hash_id_table: Mapping of hash ID to token block and cached decode.
    :param processor: Tokenizer used when the blocks must be decoded together.
    :param join_decoded_blocks: Use cached per-block strings. Set only after
        :func:`decodes_hash_blocks_concatenatively` has passed.
    :return: Synthetic prompt text for the hash-id prefix.
    """
    blocks = [hash_id_table[hash_id] for hash_id in hash_ids]
    # Joining is not ``decode(all_ids)`` when the tokenizer rewrites boundaries.
    blocks_are_closed = all(_REPLACEMENT_CHAR not in block.text for block in blocks)
    if join_decoded_blocks and blocks_are_closed:
        return "".join(block.text for block in blocks)
    prompt_token_ids = [token for block in blocks for token in block.token_ids]
    return decode_prompt(processor, prompt_token_ids)


def decodes_hash_blocks_concatenatively(
    processor: PreTrainedTokenizerBase,
    block_size: int,
    random_seed: int,
    sample_count: int = 4,
    *,
    use_english: bool = False,
) -> bool:
    """Return whether joining per-block decodes matches one full decode.

    When ``use_english`` is true, blocks are sliced from a private
    :class:`EnglishTokenBuffer`, the same pattern WEKA uses for hash blocks.
    Otherwise blocks come from :func:`generate_token_ids`. Joining those
    strings is not equal to ``decode`` of the concatenated ids for every
    tokenizer: some insert spaces between pieces, and some merge bytes
    across a block boundary (decoded as U+FFFD when the block is decoded
    alone). A failed or inconclusive probe keeps the full-sequence decode.

    The generator is private so this does not advance the replay copy's stream.

    :param processor: Tokenizer under test.
    :param block_size: Token count of one hash block.
    :param random_seed: Dataset seed. Combined with a fixed offset so the
        probe stream does not alias a replay copy.
    :param sample_count: Number of synthetic blocks to compare. At least two.
    :param use_english: Draw probe blocks from Zipf English text.
    :return: True when every adjacent pair and the full chain concatenate.
    """
    if block_size <= 0 or sample_count < _MIN_PROBE_BLOCKS:
        return False
    try:
        if use_english:
            rng = random.Random(random_seed + _PROBE_SEED_OFFSET)  # noqa: S311
            buffer = EnglishTokenBuffer()
            blocks = [
                buffer.take(block_size, processor, rng) for _ in range(sample_count)
            ]
        else:
            faker = Faker()
            faker.seed_instance(random_seed + _PROBE_SEED_OFFSET)
            blocks = [
                generate_token_ids(block_size, processor, faker)
                for _ in range(sample_count)
            ]
        texts = [decode_prompt(processor, list(block)) for block in blocks]
        if any(_REPLACEMENT_CHAR in text for text in texts):
            return False
        for index in range(len(blocks) - 1):
            combined = list(blocks[index]) + list(blocks[index + 1])
            if decode_prompt(processor, combined) != texts[index] + texts[index + 1]:
                return False
        chain = [token for block in blocks for token in block]
        return decode_prompt(processor, chain) == "".join(texts)
    except Exception:  # noqa: BLE001
        # A tokenizer that cannot decode the probe must not abort dataset load
        # or switch on a join that was never shown to match a full decode.
        logger.debug(
            "Hash-block decode probe failed; keeping full-sequence decode",
            exc_info=True,
        )
        return False


def create_distinct_token_block(
    block_size: int,
    sibling_token_blocks: set[tuple[int, ...]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    max_attempts: int = 20,
    english_buffer: EnglishTokenBuffer | None = None,
) -> tuple[int, ...]:
    """Constructs a new token block of `block_size` that does not appear in
    `sibling_token_blocks`.

    When ``english_buffer`` is set, the next ids are taken from that Zipf
    English stream. Otherwise text is generated and encoded.
    """
    attempt = 0
    while attempt < max_attempts:
        if english_buffer is not None:
            token_ids = english_buffer.take(block_size, processor, faker.random)
        else:
            token_ids = generate_token_ids(block_size, processor, faker)
        if token_ids not in sibling_token_blocks:
            return token_ids
        attempt += 1
    raise ValueError(
        f"Failed to generate distinct synthetic token block after {attempt} attempts"
    )


def fill_hash_id_table(
    ids: Sequence[int],
    hash_id_table: dict[int, HashTokenBlock],
    sibling_token_blocks: dict[Any, set[tuple[int, ...]]],
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    tokens_for_hash_id: Callable[[int, int], int],
    english_buffer: EnglishTokenBuffer | None = None,
) -> None:
    """Ensure each id has a distinct sibling-aware token block in ``hash_id_table``.

    Unseen hash IDs are allocated with :func:`create_distinct_token_block` so
    siblings under the same previous id receive different token blocks.
    Each new block is decoded once and stored next to its token ids.
    Existing entries are left unchanged. Sibling distinctness compares token
    ids only, not the decoded text.

    :param ids: Ordered hash IDs for one prompt.
    :param hash_id_table: Mapping of hash ID to token block. Mutated in place.
    :param sibling_token_blocks: Token blocks already used per previous hash ID.
        Mutated in place.
    :param processor: Tokenizer used to generate and decode synthetic blocks.
    :param faker: Random text source for synthetic tokens.
    :param tokens_for_hash_id: ``(idx, hash_id) -> block size`` for unseen IDs.
    :param english_buffer: When set, new blocks are sliced from this English
        stream instead of encoded Faker text.
    """
    for idx, hash_id in enumerate(ids):
        if hash_id not in hash_id_table:
            prev_id = None if idx == 0 else ids[idx - 1]
            sibling_token_blocks.setdefault(prev_id, set())
            token_ids = create_distinct_token_block(
                tokens_for_hash_id(idx, hash_id),
                sibling_token_blocks[prev_id],
                processor,
                faker,
                english_buffer=english_buffer,
            )
            hash_id_table[hash_id] = HashTokenBlock(
                token_ids=token_ids,
                text=decode_prompt(processor, list(token_ids)),
            )
            sibling_token_blocks[prev_id].add(token_ids)


def _seeded_faker(random_seed: int, copy_index: int) -> Faker:
    """Build a Faker instance for sequential dataset copy ``copy_index``."""
    faker = Faker()
    faker.seed_instance(random_seed + copy_index * 1_000_003)
    return faker


class TraceFormatBase(Protocol):
    config: TraceDataArgs
    dataset: Dataset

    def __init__(self, config, dataset: Dataset) -> None: ...

    def has_duration_column(self) -> bool:
        """
        Return whether this trace includes the configured duration column.

        The default checks top-level dataset columns. Nested formats override
        this to look at the row shape they actually read. Called once at load.

        :return: True when ``config.duration_column`` is present
        """
        return self.config.duration_column in self.dataset.column_names

    def __iter__(self) -> Iterable[Dataset]:
        """Returns the next conversation as a `Dataset`."""

    def reset(self) -> None:
        pass

    def reset_hash_tables(self) -> None: ...

    def prepare_processor(
        self,
        processor: PreTrainedTokenizerBase,  # noqa: ARG002
        random_seed: int,  # noqa: ARG002
    ) -> None:
        """Called once when the dataset tokenizer is loaded.

        Formats that cache tokenizer-specific prompt state override this.
        The default does nothing.

        :param processor: Tokenizer used to build synthetic prompts.
        :param random_seed: Dataset seed. Probe randomness must not consume
            the replay Faker derived from this seed.
        """

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
                **duration_columns(turn, self.config),
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


class SingleTurnTraceFormat(TraceFormatBase):
    """Replay each trace row as an independent conversation on a shared timeline."""

    dataset: Dataset
    _trace_start_timestamp: float

    def __iter__(self) -> Iterable[Dataset]:
        """Yield one timestamp-sorted row at a time for lazy prompt generation."""
        ordered = self.dataset.sort(self.config.timestamp_column)
        if not len(ordered):
            return
        self._trace_start_timestamp = ordered[0][self.config.timestamp_column]
        for index in range(len(ordered)):
            yield ordered.select([index])

    def build_conversation_graph(
        self,
        conversation: Dataset,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> ConversationGraphData:
        """Build a single root turn with its offset from the start of the trace.

        :param conversation: One trace row returned by iteration
        :param processor: Tokenizer for generating the synthetic prompt
        :param faker: Seeded synthetic text generator
        :return: An independent conversation preserving its trace arrival time
        """
        graph = super().build_conversation_graph(conversation, processor, faker)
        graph.turns[0].columns["relative_timestamp_column"] = [
            conversation[0][self.config.timestamp_column] - self._trace_start_timestamp
        ]
        return graph


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
        self._copy_fakers = [
            _seeded_faker(random_seed, copy_index)
            for copy_index in range(config.copies)
        ]
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        samples_count = 0
        pass_offset = 0.0
        # Shared across copies so packing sees the combined timeline.
        packer = TraceSessionTiming(
            min_concurrent_sessions=self.config.min_concurrent_sessions,
        )
        scaler = TraceSessionTiming(time_scale=self.config.time_scale)
        for copy_index in range(self.config.copies):
            self.format.reset_hash_tables()
            faker_copy = self._copy_fakers[copy_index]
            wait_timing = TraceSessionTiming(
                max_wait=self.config.max_wait,
                max_session_wait=self.config.max_session_wait,
            )
            copy_min = math.inf
            copy_max = -math.inf
            for conv in self.format:  # type: ignore[attr-defined]
                graph_data = self.format.build_conversation_graph(
                    conv, self.processor, faker_copy
                )
                if not graph_data.turns:
                    continue
                wait_timing.apply_wait_caps(graph_data)
                shift_graph_timestamps(graph_data, pass_offset)
                copy_min = min(copy_min, graph_min_timestamp(graph_data))
                copy_max = max(copy_max, graph_max_timestamp(graph_data))
                packer.apply_pack(graph_data)
                scaler.apply_scale(graph_data)
                samples_count += len(graph_data.turns)
                # The iterable is typed, so Hugging Face does not cast this
                # column to a string. The finalizer accepts the model directly.
                yield (
                    samples_count,
                    {"conversation_turns": graph_data},
                )
                self.format.reset()
            if math.isfinite(copy_min):
                pass_offset = copy_min + self.config.copy_offset * (copy_max - copy_min)

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
    if not trace_format.has_duration_column():
        logger.warning(
            "Trace duration column '{}' is missing; schedule_turn=idle_gap "
            "will treat each request as instantaneous.",
            trace_format.config.duration_column,
        )


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
        processor = processor_factory()
        trace_format.prepare_processor(processor, random_seed)
        return TraceDataset(config, trace_format, processor, random_seed)
