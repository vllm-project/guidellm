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
parent turn. ``stop: tool_use`` and ``input_types: tool_result`` map
onto the existing client tool-call columns (``turn_type``, ``tools``,
``tool_response``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
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
    _validate_api_conversation,
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
from guidellm.schemas import TurnType
from guidellm.schemas.data.deserializers import (
    DEFAULT_SYNTHETIC_TOOLS,
    WEKATraceFormatArgs,
)
from guidellm.settings import settings
from guidellm.utils.imports import json
from guidellm.utils.random import IntegerRangeSampler

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


def _weka_stop_reason(row: dict[str, Any]) -> str:
    stop = row.get("stop")
    return stop if isinstance(stop, str) else ""


def _weka_input_types(row: dict[str, Any]) -> list[Any]:
    raw = row.get("input_types")
    return list(raw) if isinstance(raw, list) else []


def _serialized_tools(tools: list[dict[str, Any]] | None) -> str:
    raw = json.dumps(tools or DEFAULT_SYNTHETIC_TOOLS)
    return raw.decode() if isinstance(raw, bytes) else raw


def _classify_weka_tool_turn(
    row: dict[str, Any], prev_stop: str | None
) -> tuple[TurnType | None, bool, bool]:
    """Map a WEKA API row onto GuideLLM tool-call columns.

    Each row is one HTTP request. ``stop`` is the model output; ``input_types``
    is what the client added. A ``tool_result`` row is therefore an injection
    for the previous row's tool calls, even when that same row also has
    ``stop: tool_use`` (Claude Code multi-step loop). ``tools_column`` on an
    injection only changes the expected output (``tool_choice=required``).

    When ``input_types`` is absent, a row after ``stop: tool_use`` on the same
    agent chain is treated as ``tool_result``. Explicit ``input_types: ["text"]``
    is not.

    :return: ``(turn_type, include_tools, include_tool_response)``. ``turn_type``
        is ``None`` for ordinary text turns.
    """
    input_types = _weka_input_types(row)
    stop = _weka_stop_reason(row)
    is_tool_call = stop == "tool_use"
    is_tool_result = "tool_result" in input_types
    if not is_tool_result and not input_types and prev_stop == "tool_use":
        is_tool_result = True

    if is_tool_result:
        return "tool_response_injection", is_tool_call, True
    if is_tool_call:
        return "client_tool_call", True, False
    return None, False, False


@dataclass
class _TurnSpec:
    node_id: str
    agent_id: str
    parents: list[ConversationParentRef]
    row: dict[str, Any]
    absolute_t: float
    turn_type: TurnType | None = None
    include_tools: bool = False
    include_tool_response: bool = False


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

    ``stop: "tool_use"`` and ``input_types: ["tool_result"]`` map onto the
    existing client tool-call pipeline. Tool schemas and results are not in
    the trace (anonymized); pass ``tools`` / ``tool_response_tokens`` like
    synthetic data, or the default placeholder schema and response are used.
    A ``tool_result`` row that also has ``stop: "tool_use"`` is still an
    injection on the input side and carries ``tools_column`` so the model
    may emit further tool calls.

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
        self._tools_json = _serialized_tools(config.tools)
        self._tool_response_sampler: Iterator[int] | None = None
        self.requests_col = _find_requests_column(dataset)
        if self.requests_col is None:
            raise DataNotSupportedError(
                "WEKA format: Failed to find requests column or requests was empty"
            )

    def __iter__(self) -> Iterable[Dataset]:
        self._conversation_queue = []
        for row in self.dataset:
            conv_id = str(row[self.config.conversation_id_column])
            # File order is spawn/join topology for every request list,
            # including nested subagent groups. Do not sort by timestamp.
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
        # Only the first API row is searchable here. Missing fields on later
        # conversations are rejected in validate_conversation.
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
        _validate_api_conversation(
            Dataset.from_list(api_rows),
            self.config,
            self.required_columns(),
            self.validate_row,
        )

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
            columns: dict[str, Any] = {
                "text_column": [prompt],
                "prompt_tokens_count_column": [
                    spec.row[self.config.prompt_tokens_column]
                ],
                "output_tokens_count_column": [
                    spec.row[self.config.output_tokens_column]
                ],
                "relative_timestamp_column": [spec.absolute_t - min_t],
            }
            if spec.turn_type is not None:
                columns["turn_type_column"] = [spec.turn_type]
            if spec.include_tools:
                columns["tools_column"] = [self._tools_json]
            if spec.include_tool_response:
                columns["tool_response_column"] = [
                    self._tool_response_text(processor, faker)
                ]
            turns.append(
                ConversationTurnData(
                    node_id=spec.node_id,
                    agent_id=spec.agent_id,
                    parents=spec.parents,
                    columns=columns,
                )
            )
        return ConversationGraphData(turns=turns)

    def _tool_response_text(
        self, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        """Build the mocked tool result for an injection turn.

        Matches synthetic data: ``tool_response_tokens`` sizes a ``{"result": ...}``
        payload; otherwise the global placeholder is used.
        """
        if self.config.tool_response_tokens is None:
            return settings.default_synthetic_tool_response
        if self._tool_response_sampler is None:
            self._tool_response_sampler = iter(
                IntegerRangeSampler(
                    average=self.config.tool_response_tokens,
                    variance=self.config.tool_response_tokens_stdev,
                    min_value=self.config.tool_response_tokens_min,
                    max_value=self.config.tool_response_tokens_max,
                    random_seed=faker.random.getrandbits(32),
                )
            )
        body = _generate_remaining_prompt(
            next(self._tool_response_sampler), processor, faker
        )
        raw = json.dumps({"result": body})
        return raw.decode() if isinstance(raw, bytes) else raw

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

        File order is spawn/join topology. API rows continue this agent's
        chain. ``type: "subagent"`` groups spawn a sibling chain from the
        latest API turn of this agent (``history_context="new"``). The next
        API turn of this agent joins every pending sibling
        (``history_context="last"``), so multiple subagents between the same
        parent turns run in parallel. Nested subagent groups keep the same
        rule against the inner list.
        """
        ts_col = self.config.timestamp_column
        specs: list[_TurnSpec] = []
        pending_join_ids: list[str] = []
        last_chain_id: str | None = None
        chain_idx = 0
        chain_events: list[tuple[float, float | None]] = []
        # Previous API row's stop on this chain only. Subagent children are
        # interleaved in the flattened spec list, so classification cannot
        # use that list's adjacency.
        prev_stop: str | None = None

        for item in requests:
            if _is_subagent_entry(item):
                inner = [dict(row) for row in (item.get("requests") or [])]
                if not inner:
                    continue
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
            turn_type, include_tools, include_tool_response = _classify_weka_tool_turn(
                item, prev_stop
            )
            specs.append(
                _TurnSpec(
                    node_id=node_id,
                    agent_id=agent_id,
                    parents=parents,
                    row=_copy_api_row(item, self.config.hash_ids_column),
                    absolute_t=abs_t,
                    turn_type=turn_type,
                    include_tools=include_tools,
                    include_tool_response=include_tool_response,
                )
            )
            last_chain_id = node_id
            chain_idx += 1
            chain_events.append((abs_t, _optional_float(item.get("api_time"))))
            prev_stop = _weka_stop_reason(item)

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
