"""
The WEKA trace format and data arguments.

Reads a trace file and yields one row per line with a
synthetic prompt matching the requested input_length for replay
benchmarks. Checks for distinctness between hash IDs that share the
same previous hash ID.

Generates prompts starting from the first conversation.
When the conversation ends, the next conversation will be used.

Declared ``type: "subagent"`` groups are replayed as isolated child
chains that spawn from the preceding parent turn and join the following
parent turn. Tool call events (``stop: tool_use``) are still missing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, Features, List, Value
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_common import (
    TraceDatasetDeserializer,
    TraceFormatBase,
    TraceFormatRegistry,
    _raise_if_incorrect_types,
    _raise_if_nonetype_found,
    create_distinct_token_block,
    create_prompt_from_hash_ids,
    decode_prompt,
    generate_token_ids,
    get_missing_columns,
)
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.logger import logger
from guidellm.scheduler.schemas import HistoryContext
from guidellm.schemas.data.deserializers import WEKATraceFormatArgs

__all__ = ["WEKATraceFormat"]


def _find_requests_column(dataset: Dataset) -> str | None:
    for name, val in dataset.features.items():
        if (
            isinstance(val, List)
            and len(dataset[name][0]) > 0
            and isinstance(dataset[name][0][0], dict)
        ):
            return name
    return None


def _generate_remaining_prompt(
    num_tokens: int, processor: PreTrainedTokenizerBase, faker: Faker
) -> str:
    if num_tokens == 0:
        return ""
    token_ids = generate_token_ids(num_tokens, processor, faker)
    return decode_prompt(processor, list(token_ids))


def _is_subagent_entry(row: dict[str, Any]) -> bool:
    return row.get("type") == "subagent"


def _first_api_request(requests: list[Any]) -> dict[str, Any] | None:
    for row in requests:
        if not isinstance(row, dict):
            continue
        if _is_subagent_entry(row):
            found = _first_api_request(list(row.get("requests") or []))
            if found is not None:
                return found
            continue
        return row
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _inner_timestamp_transform(
    entry: dict[str, Any],
    inner_requests: list[dict[str, Any]],
    timestamp_column: str,
) -> Callable[[float], float]:
    """Map inner ``t`` to conversation-absolute time.

    The spec records inner timestamps relative to spawn. Published corpora
    often store absolute ``t`` on the same timeline as the parent. If the
    first inner timestamp is less than the subagent spawn time, treat
    inners as relative (``spawn_t + inner_t``); otherwise leave them as-is.
    """
    spawn_t = float(entry[timestamp_column])
    first_inner_t = float(inner_requests[0][timestamp_column])
    if first_inner_t < spawn_t:

        def _relative(inner_t: float) -> float:
            return spawn_t + inner_t

        return _relative

    def _absolute(inner_t: float) -> float:
        return inner_t

    return _absolute


def _copy_api_row(row: dict[str, Any], hash_ids_column: str) -> dict[str, Any]:
    copied = dict(row)
    hash_ids = copied.get(hash_ids_column)
    if hash_ids is not None:
        copied[hash_ids_column] = list(hash_ids)
    return copied


@dataclass
class _TurnSpec:
    node_id: str
    agent_id: str
    parents: list[ConversationParentRef]
    row: dict[str, Any]
    absolute_t: float


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "weka")


@TraceFormatRegistry.register("weka")
class WEKATraceFormat(TraceFormatBase):
    """WEKA trace format requires a column for timestamps, prompt token counts,
    ouput token counts and lists of hash IDs.

    Hash IDs are unique identifiers based on the current and previous token
    blocks in a prompt. The relationships of IDs forms a tree, where every first ID
    in a prompt has a parent node of `None`. Parent nodes can have an unbounded
    number of children. Two hash IDs can represent identical blocks of tokens so long
    as they do not share the same parent (previous ID).

    Declared ``type: "subagent"`` groups become isolated child chains that
    spawn from the preceding parent API turn (``history_context="new"``) and
    join the following parent API turn (``history_context="last"``). Adjacent
    subagents between the same parent turns run in parallel; the following
    parent waits for all of them.

    For more details, see [the WEKA trace format specification][trace-spec].

    [trace-spec]: https://github.com/callanjfox/agentic-coding-analysis/blob/master/docs/TRACE_FORMAT.md

    Generated prompts match the prompt token count of the row."""

    def __init__(self, config: WEKATraceFormatArgs, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset

        self.hash_id_table: dict[int, tuple[int, ...]] = {}
        self.sibling_token_blocks: dict[Any, set[tuple[int, ...]]] = {}
        # Filled by each ``__iter__`` pass so mixed subagent/API schemas are
        # not forced through a single HuggingFace Arrow table.
        self._conversation_queue: list[tuple[str, list[dict[str, Any]]]] = []
        self.requests_col = _find_requests_column(dataset)
        if self.requests_col is None:
            raise DataNotSupportedError(
                "WEKA format: Failed to find requests column or requests was empty"
            )

    def __iter__(self) -> Iterable[Dataset]:
        self._conversation_queue = []
        for row in self.dataset:
            conv_id = str(row[self.config.conversation_id_column])
            # Preserve file order: outer request order is the spawn/join
            # topology. Do not sort by timestamp.
            requests = [dict(item) for item in row[self.requests_col]]
            index = len(self._conversation_queue)
            self._conversation_queue.append((conv_id, requests))
            yield Dataset.from_dict({"_weka_index": [index]})

    def reset(self) -> None:
        self.hash_id_table = {}
        self.sibling_token_blocks = {}

    def required_columns(self) -> Features:
        return Features(
            {
                self.config.conversation_id_column: Value("string"),
                self.config.hash_ids_column: List(Value("int32")),
            }
        )

    def find_required_columns(self, columns: list[str]) -> list[str]:
        """TODO: Handle edge cases"""
        conv_col = self.config.conversation_id_column
        if conv_col not in self.dataset.column_names:
            return [self.config.conversation_id_column]
        required = [col for col in columns if col != conv_col]
        first_api = _first_api_request(self.dataset[self.requests_col][0])
        if first_api is None:
            return required
        return get_missing_columns(required, list(first_api.keys()))

    def validate_row(self, row: dict) -> None:
        n_in = row[self.config.prompt_tokens_column]
        n_blocks = len(row[self.config.hash_ids_column])
        block_size = self.config.hash_id_block_size
        for hash_id in row[self.config.hash_ids_column]:
            if hash_id < 0:
                raise DataNotSupportedError(
                    f"Hash ID must be non-negative, got {hash_id}"
                )
        expected = n_in / block_size
        if math.floor(expected) != n_blocks and math.ceil(expected) != n_blocks:
            raise DataNotSupportedError(
                f"Input token count of {n_in} split into blocks of size "
                f"{block_size} full blocks and "
                f"{block_size} full blocks + partially filled "
                f"trailing block does not match given {n_blocks} blocks"
            )

    def validate_conversation(self, conversation: Dataset) -> None:
        _, requests = self._unpack_conversation(conversation)
        if not requests:
            raise DataNotSupportedError("Trace conversation is empty")
        api_rows = self._collect_api_rows(requests)
        if not api_rows:
            raise DataNotSupportedError(
                "WEKA format: conversation has no API requests to replay"
            )
        features = Features(
            {
                self.config.timestamp_column: Value("float"),
                self.config.prompt_tokens_column: Value("int32"),
                self.config.output_tokens_column: Value("int32"),
                self.config.hash_ids_column: List(Value("int32")),
            }
        )
        api_dataset = Dataset.from_list(api_rows)
        _raise_if_nonetype_found(api_dataset, features)
        _raise_if_incorrect_types(api_dataset, features)
        for row in api_rows:
            n_in = row[self.config.prompt_tokens_column]
            n_out = row[self.config.output_tokens_column]
            if n_in < 0 or n_out < 0:
                raise DataNotSupportedError(
                    f"Trace token counts must be non-negative, got "
                    f"input_length={n_in}, output_length={n_out}"
                )
            self.validate_row(row)

    def create_prompt(
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        """Before generating the prompt, this first generates a block of tokens for
        each hash ID that has not already been seen.

        Hash IDs that are partially filled are discarded to match the specification.
        Remainder of the prompt is created after the creation via hash IDs token
        blocks."""
        ids = row[self.config.hash_ids_column]
        n_in = row[self.config.prompt_tokens_column]
        block_size = self.config.hash_id_block_size
        expected = n_in / block_size
        if math.floor(expected) != len(ids) and math.ceil(expected) == len(ids):
            ids.pop()
        for idx, hash_id in enumerate(ids):
            if hash_id not in self.hash_id_table:
                prev_id = None if idx == 0 else ids[idx - 1]
                self.sibling_token_blocks.setdefault(prev_id, set())
                self.hash_id_table[hash_id] = create_distinct_token_block(
                    block_size,
                    self.sibling_token_blocks[prev_id],
                    processor,
                    faker,
                )
                self.sibling_token_blocks[prev_id].add(self.hash_id_table[hash_id])
        prompt = create_prompt_from_hash_ids(ids, self.hash_id_table, processor)
        remainder = _generate_remaining_prompt(n_in % block_size, processor, faker)
        if not prompt:
            return remainder
        if not remainder:
            return prompt
        return f"{prompt} {remainder}"

    def build_conversation_graph(
        self,
        conversation: Dataset,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> ConversationGraphData:
        conv_id, requests = self._unpack_conversation(conversation)
        specs, _last = self._emit_chain(
            requests,
            agent_id="default",
            node_prefix="main",
            preceding_parent_id=None,
            first_turn_history="new",
            t_transform=lambda t: t,
            spawn_seq=[0],
            conversation_id=conv_id,
        )
        if not specs:
            raise DataNotSupportedError(
                "WEKA format: conversation has no API requests to replay"
            )
        min_t = min(spec.absolute_t for spec in specs)
        turns: list[ConversationTurnData] = []
        for spec in specs:
            prompt = self.create_prompt(spec.row, processor, faker)
            columns = {
                "text_column": [prompt],
                "prompt_tokens_count_column": [
                    spec.row[self.config.prompt_tokens_column]
                ],
                "output_tokens_count_column": [
                    spec.row[self.config.output_tokens_column]
                ],
                "relative_timestamp_column": [spec.absolute_t - min_t],
            }
            turns.append(
                ConversationTurnData(
                    node_id=spec.node_id,
                    agent_id=spec.agent_id,
                    parents=spec.parents,
                    columns=columns,
                )
            )
        return ConversationGraphData(turns=turns)

    def _unpack_conversation(
        self, conversation: Dataset
    ) -> tuple[str, list[dict[str, Any]]]:
        index = int(conversation[0]["_weka_index"])
        return self._conversation_queue[index]

    def _collect_api_rows(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        api_rows: list[dict[str, Any]] = []
        columns = (
            self.config.timestamp_column,
            self.config.prompt_tokens_column,
            self.config.output_tokens_column,
            self.config.hash_ids_column,
        )
        for row in requests:
            if _is_subagent_entry(row):
                api_rows.extend(self._collect_api_rows(list(row.get("requests") or [])))
                continue
            api_rows.append({col: row.get(col) for col in columns})
        return api_rows

    def _emit_chain(
        self,
        requests: list[dict[str, Any]],
        agent_id: str,
        node_prefix: str,
        preceding_parent_id: str | None,
        first_turn_history: HistoryContext,
        t_transform: Callable[[float], float],
        spawn_seq: list[int],
        conversation_id: str,
    ) -> tuple[list[_TurnSpec], str | None]:
        """Walk a request list into turn specs for one agent plus its children.

        API rows continue this agent's chain. ``type: "subagent"`` groups
        spawn a sibling chain from the latest API turn of this agent
        (``history_context="new"``). The next API turn of this agent joins
        every pending sibling (``history_context="last"``), so multiple
        subagents between the same parent turns run in parallel.
        """
        ts_col = self.config.timestamp_column
        specs: list[_TurnSpec] = []
        pending_join_ids: list[str] = []
        last_chain_id: str | None = None
        chain_idx = 0
        chain_events: list[tuple[float, float | None]] = []

        for item in requests:
            if _is_subagent_entry(item):
                inner = [dict(row) for row in (item.get("requests") or [])]
                if not inner:
                    continue
                inner.sort(key=lambda row: float(row[ts_col]))
                spawn_id = spawn_seq[0]
                spawn_seq[0] += 1
                child_agent = str(item.get("agent_id") or f"sa_{spawn_id}")
                if last_chain_id is None:
                    logger.warning(
                        "WEKA subagent '{}' in conversation '{}' has no "
                        "preceding parent turn; replaying as an independent root",
                        child_agent,
                        conversation_id,
                    )
                child_specs, child_last = self._emit_chain(
                    inner,
                    agent_id=child_agent,
                    node_prefix=f"sa_{spawn_id}",
                    preceding_parent_id=last_chain_id,
                    first_turn_history="new",
                    t_transform=_inner_timestamp_transform(item, inner, ts_col),
                    spawn_seq=spawn_seq,
                    conversation_id=conversation_id,
                )
                specs.extend(child_specs)
                if child_last is not None:
                    pending_join_ids.append(child_last)
                continue

            parents: list[ConversationParentRef] = []
            if last_chain_id is not None:
                parents.append(
                    ConversationParentRef(
                        parent_node_id=last_chain_id,
                        history_context="full",
                    )
                )
            elif preceding_parent_id is not None:
                parents.append(
                    ConversationParentRef(
                        parent_node_id=preceding_parent_id,
                        history_context=first_turn_history,
                    )
                )
            for join_id in pending_join_ids:
                parents.append(
                    ConversationParentRef(
                        parent_node_id=join_id,
                        history_context="last",
                    )
                )
            pending_join_ids = []

            abs_t = t_transform(float(item[ts_col]))
            node_id = f"{node_prefix}_{chain_idx}"
            specs.append(
                _TurnSpec(
                    node_id=node_id,
                    agent_id=agent_id,
                    parents=parents,
                    row=_copy_api_row(item, self.config.hash_ids_column),
                    absolute_t=abs_t,
                )
            )
            last_chain_id = node_id
            chain_idx += 1
            chain_events.append((abs_t, _optional_float(item.get("api_time"))))

        if pending_join_ids:
            logger.debug(
                "WEKA subagent(s) in conversation '{}' on agent '{}' have no "
                "following parent turn; replaying as background without a join",
                conversation_id,
                agent_id,
            )

        self._warn_chain_overlap(conversation_id, agent_id, chain_events)
        return specs, last_chain_id

    def _warn_chain_overlap(
        self,
        conversation_id: str,
        agent_id: str,
        events: list[tuple[float, float | None]],
    ) -> None:
        """Warn when consecutive turns of one agent overlap in time.

        Overlap uses recorded ``api_time`` when present
        (``t[i] + api_time[i] > t[i+1]``), otherwise non-increasing start
        times. Parallel subagents overlapping each other is intended and
        is not warned here; each agent chain is checked separately.
        """
        for idx in range(len(events) - 1):
            t_i, api_time = events[idx]
            t_next, _ = events[idx + 1]
            if api_time is not None:
                overlapped = t_i + api_time > t_next
            else:
                overlapped = t_next <= t_i
            if not overlapped:
                continue
            if api_time is not None:
                logger.debug(
                    "WEKA conversation '{}' agent '{}' has overlapping requests: "
                    "the request at t={} will run until t={}, which is after "
                    "the next request at t={}; they will be serialized on "
                    "this chain",
                    conversation_id,
                    agent_id,
                    t_i,
                    t_i + api_time,
                    t_next,
                )
            else:
                logger.debug(
                    "WEKA conversation '{}' agent '{}' has overlapping requests: "
                    "the request at t={} is followed by a request at t={} "
                    "that does not start later; they will be serialized on "
                    "this chain",
                    conversation_id,
                    agent_id,
                    t_i,
                    t_next,
                )
