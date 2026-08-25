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


def turns_from_mapped_items(items: list[dict[str, Any]]) -> ConversationGraphData:
    """
    Normalize mapper output into a :class:`ConversationGraphData` payload.

    If any item carries ``conversation_turns_column``, that payload is parsed and
    returned. Otherwise each item becomes a linear-chain turn
    (``turn_0``, ``turn_1``, …) with ``full`` history edges.

    Scheduling columns are lifted onto ``turn.settings`` so ``columns`` stay
    request-content only.

    :param items: Mapper output — either one graph payload item or one dict of
        columns per logical turn.
    :return: A conversation graph data object (possibly with an empty turns list).
    :raises ValueError: If ``conversation_turns_column`` is present but empty.
    """
    for item in items:
        raw_values = item.get("conversation_turns_column")
        if not raw_values:
            continue
        graph_data = _parse_conversation_turns(raw_values[0])
        if not graph_data.turns:
            raise ValueError("ConversationGraphData.turns must not be empty")
        return graph_data

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
    """Return True when this logical turn should become tool-call + injection."""
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
    if explicit_type == "client_tool_call":
        return True
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
