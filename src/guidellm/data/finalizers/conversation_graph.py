"""
Normalize mapped dataset rows into ConversationGraphData and expand tool turns.

Both indexed-column multiturn rows and pre-built ``conversation_turns`` payloads
converge here before the generative finalizer builds a runtime graph.
"""

from __future__ import annotations

from typing import Any, Literal

from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas import RequestSettings

__all__ = [
    "expand_client_tool_turns",
    "turns_from_mapped_items",
]

_SCHEDULING_COLUMNS = (
    "relative_timestamp_column",
    "requeue_delay_column",
)


def _optional_column_value(columns: dict[str, Any], column_name: str) -> Any | None:
    values = columns.get(column_name, [])
    return values[0] if values else None


def _lift_settings_from_columns(
    columns: dict[str, Any],
) -> tuple[dict[str, Any], RequestSettings | None]:
    """
    Copy columns without scheduling keys; return lifted RequestSettings when present.

    :param columns: Mapped turn columns (may include scheduling columns).
    :return: Content-only columns and optional settings lifted from scheduling keys.
    """
    content = dict(columns)
    had_scheduling = any(key in content for key in _SCHEDULING_COLUMNS)
    relative_timestamp = _optional_column_value(content, "relative_timestamp_column")
    requeue_delay = _optional_column_value(content, "requeue_delay_column")
    for key in _SCHEDULING_COLUMNS:
        content.pop(key, None)
    if not had_scheduling:
        return content, None
    return content, RequestSettings(
        relative_timestamp=relative_timestamp,
        requeue_delay=requeue_delay,
    )


def _parse_conversation_turns(raw: Any) -> ConversationGraphData:
    if isinstance(raw, str):
        return ConversationGraphData.model_validate_json(raw)
    return ConversationGraphData.model_validate(raw)


def _merge_sibling_columns_into_graph(
    items: list[dict[str, Any]],
    graph_item_index: int,
    graph_data: ConversationGraphData,
) -> ConversationGraphData:
    """
    Merge sibling dataset columns into the root turn(s) of the passed pre-built graph.

    :param items: Full mapper output (one dict of columns per outer turn index).
    :param graph_item_index: Index within items to the graph_data column.
    :param graph_data: The parsed conversation graph.

    :return: Graph with sibling columns merged to its root conversation_turn
             (or the original unchanged graph if there are no sibling columns)
    :raises ValueError: If sibling columns exist on an outer item inconsistent with the
            graph_item_index, or if the graph has more than one root conversation_turn.
    """

    sibling_columns = {
        key: vals
        for key, vals in items[graph_item_index].items()
        if key != "conversation_turns_column"
    }
    if not sibling_columns:
        return graph_data

    for index, item in enumerate(items):
        if index == graph_item_index or not item:
            continue
        raise ValueError(
            "Cannot combine a dataset that emits its own conversation_turns_column "
            "(e.g. kind=synthetic_text) with sibling dataset columns "
            f"({', '.join(sorted(item))}) mapped to a different turn index"
            f" ({index}) than the one carrying conversation_turns_column"
            f"({graph_item_index}). Provide sibling modality/content data at the "
            "same turn index as the packed conversation, or use a dataset without "
            "its own conversation_turns_column."
        )

    root_turns = [turn for turn in graph_data.turns if not turn.parents]
    if len(root_turns) != 1:
        raise ValueError(
            "Cannot attach sibling dataset columns "
            f"({', '.join(sorted(sibling_columns))}) to a conversation_turns_column"
            f" payload with {len(root_turns)} root turns; branched/subagent"
            "conversations are ambiguous for shared sibling data. Fold the "
            "modality/content data into the source dataset's own conversation_turns"
            " payload instead of a sibling --data source."
        )

    sibling_content, sibling_settings = _lift_settings_from_columns(sibling_columns)

    merged_turns = []
    for turn in graph_data.turns:
        if turn.node_id != root_turns[0].node_id:
            merged_turns.append(turn)
            continue
        merged_columns = dict(turn.columns)
        for key, values in sibling_content.items():
            merged_columns[key] = merged_columns.get(key, []) + list(values)
        merged_turns.append(
            ConversationTurnData(
                node_id=turn.node_id,
                agent_id=turn.agent_id,
                parents=turn.parents,
                columns=merged_columns,
                settings=(
                    turn.settings if turn.settings is not None else sibling_settings
                ),
            )
        )
    return ConversationGraphData(graph_id=graph_data.graph_id, turns=merged_turns)


def turns_from_mapped_items(items: list[dict[str, Any]]) -> ConversationGraphData:
    """
    Normalize mapper output into a :class:`ConversationGraphData` payload.

    If any item carries ``conversation_turns_column``, that payload is parsed and any
    sibling columns (like image_column/video_column/audio_column which passed via
    separate --data argument) - are added to the *same* conversation_turn item  —
    see :func:`_merge_sibling_columns_into_graph`. Otherwise, each item becomes a
    linear-chain turn (``turn_0``, ``turn_1``, …) with ``full`` history edges.

    Scheduling columns are lifted onto ``turn.settings`` so ``columns`` stay
    request-content only.

    :param items: Mapper output — either one graph payload item or one dict of
        columns per logical turn.
    :return: A conversation graph data object (possibly with an empty turns list).
    :raises ValueError: If ``conversation_turns_column`` is present but empty.
    """
    for index, item in enumerate(items):
        raw_values = item.get("conversation_turns_column")
        if not raw_values:
            continue
        graph_data = _parse_conversation_turns(raw_values[0])
        if not graph_data.turns:
            raise ValueError("ConversationGraphData.turns must not be empty")
        return _merge_sibling_columns_into_graph(items, index, graph_data)

    turns: list[ConversationTurnData] = []
    for item in items:
        columns, settings = _lift_settings_from_columns(item)
        if not columns and settings is None:
            continue
        parents: list[ConversationParentRef] = []
        if turns:
            parents.append(
                ConversationParentRef(
                    parent_node_id=turns[-1].node_id,
                    history_context="full",
                )
            )
        turns.append(
            ConversationTurnData(
                node_id=f"turn_{len(turns)}",
                columns=columns,
                settings=settings,
                parents=parents,
            )
        )

    return ConversationGraphData(turns=turns)


def _should_expand_client_tool_turn(
    turn: ConversationTurnData,
    tool_call_mode: Literal["client", "server"],
    existing_ids: set[str],
) -> bool:
    """Return True when this logical turn should become tool-call + injection.

    Explicit ``client_tool_call`` without ``tool_response_column`` is already
    split (the injection is a later node). Only expand when the mocked
    response is on this same turn, matching synthetic data.
    """
    injection_id = f"{turn.node_id}_injection"
    if injection_id in existing_ids:
        return False

    turn_type_values = turn.columns.get("turn_type_column", [])
    explicit_type = (
        turn_type_values[0] if turn_type_values and turn_type_values[0] else None
    )
    if explicit_type == "tool_response_injection":
        return False
    if explicit_type == "server_tool_call":
        return False
    # Pre-split graphs (WEKA) already emit a separate injection node and put
    # tool_response_column there. Expand explicit client_tool_call only for
    # the synthetic pattern where the mocked response lives on this turn.
    if explicit_type == "client_tool_call":
        return bool(turn.columns.get("tool_response_column"))
    if not turn.columns.get("tools_column"):
        return False
    return tool_call_mode == "client"


def expand_client_tool_turns(
    graph: ConversationGraphData,
    tool_call_mode: Literal["client", "server"] = "client",
) -> ConversationGraphData:
    """
    Expand logical client tool-call turns into tool-call + injection node pairs.

    Mirrors the historical linear finalizer split, but rewrites parent refs so
    fork/join graphs keep chain / spawn / merge semantics against the injection
    node (the end of the expanded pair).

    :param graph: Conversation graph with logical (possibly unsplit) turns.
    :param tool_call_mode: ``client`` expands tools turns; ``server`` leaves them
        for ``finalize_turn`` to mark as server-managed.
    :return: A new graph with injection nodes inserted where needed.
    """
    existing_ids = {turn.node_id for turn in graph.turns}
    # Precompute end ids so parent rewrites work regardless of turn list order.
    end_ids: dict[str, str] = {turn.node_id: turn.node_id for turn in graph.turns}
    expand_ids: set[str] = set()
    for turn in graph.turns:
        if _should_expand_client_tool_turn(turn, tool_call_mode, existing_ids):
            end_ids[turn.node_id] = f"{turn.node_id}_injection"
            expand_ids.add(turn.node_id)

    expanded: list[ConversationTurnData] = []
    for turn in graph.turns:
        rewritten_parents = [
            ConversationParentRef(
                parent_node_id=end_ids.get(
                    parent.parent_node_id, parent.parent_node_id
                ),
                history_context=parent.history_context,
            )
            for parent in turn.parents
        ]

        if turn.node_id not in expand_ids:
            expanded.append(
                ConversationTurnData(
                    node_id=turn.node_id,
                    agent_id=turn.agent_id,
                    parents=rewritten_parents,
                    columns=dict(turn.columns),
                    settings=turn.settings,
                )
            )
            continue

        tool_columns = dict(turn.columns)
        tool_response = tool_columns.pop("tool_response_column", None)
        output_tokens = tool_columns.pop("output_tokens_count_column", None)
        # Keep an explicit client type so finalize_turn does not reinterpret
        # tools_column under tool_call_mode="server".
        tool_columns["turn_type_column"] = ["client_tool_call"]

        expanded.append(
            ConversationTurnData(
                node_id=turn.node_id,
                agent_id=turn.agent_id,
                parents=rewritten_parents,
                columns=tool_columns,
                settings=None,
            )
        )

        injection_id = f"{turn.node_id}_injection"
        injection_columns: dict[str, Any] = {
            "turn_type_column": ["tool_response_injection"],
        }
        if tool_response:
            injection_columns["tool_response_column"] = tool_response
        if output_tokens:
            injection_columns["output_tokens_count_column"] = output_tokens

        expanded.append(
            ConversationTurnData(
                node_id=injection_id,
                agent_id=turn.agent_id,
                parents=[
                    ConversationParentRef(
                        parent_node_id=turn.node_id,
                        history_context="full",
                    )
                ],
                columns=injection_columns,
                settings=turn.settings,
            )
        )

    return ConversationGraphData(graph_id=graph.graph_id, turns=expanded)
