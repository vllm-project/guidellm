"""
Data-layer conversation graph interchange with inline parent dependencies.

Turns declare their parents explicitly so branched / multi-agent datasets
(and future WEKA traces) can describe a DAG without a separate edges list.
The finalizer derives :class:`~guidellm.scheduler.schemas.ConversationEdge`
values from these parent refs when building a runtime graph.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from guidellm.scheduler.schemas import HistoryContext
from guidellm.schemas.base import StandardBaseModel

__all__ = [
    "ConversationGraphData",
    "ConversationParentRef",
    "ConversationTurnData",
]


class ConversationParentRef(StandardBaseModel):
    """
    A dependency from a turn to a parent turn.

    :param parent_node_id: Node id of the parent turn within the same graph.
    :param history_context: How much conversational context flows from the
        parent (``new``, ``last``, or ``full``).
    """

    parent_node_id: str = Field(
        description="Node ID of the parent turn within the same graph.",
    )
    history_context: HistoryContext = Field(
        default="full",
        description=(
            "How much conversational context flows from the parent: "
            "'new' (none), 'last' (parent pair only), or 'full' (ancestor walk-back)."
        ),
    )


class ConversationTurnData(StandardBaseModel):
    """
    One turn in a conversation graph data payload.

    :param node_id: Unique identifier for this turn within the graph.
    :param agent_id: Agent identity that owns this turn.
    :param parents: Inline parent dependencies (empty for roots).
    :param columns: Column dict in the shape expected by
        :meth:`~guidellm.data.finalizers.generative.GenerativeRequestFinalizer.finalize_turn`.
    :param relative_timestamp: Optional scheduling timestamp override.
    :param requeue_delay: Optional think-time / requeue delay override.
    """

    node_id: str = Field(
        description="Unique identifier for this turn within the graph.",
    )
    agent_id: str = Field(
        default="default",
        description="Agent identity that owns this turn.",
    )
    parents: list[ConversationParentRef] = Field(
        default_factory=list,
        description="Inline parent dependencies; empty for root turns.",
    )
    columns: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Request columns for finalization (e.g. text_column, "
            "prompt_tokens_count_column)."
        ),
    )
    relative_timestamp: float | None = Field(
        default=None,
        description="Optional relative timestamp for trace-style replay.",
    )
    requeue_delay: float | None = Field(
        default=None,
        description="Optional requeue / think-time delay for this turn.",
    )


class ConversationGraphData(StandardBaseModel):
    """
    One conversation example as a list of turns with inline parents.

    :param graph_id: Optional stable graph id; assigned at assemble time if omitted.
    :param turns: All turns (main and subagent) in this conversation.
    """

    graph_id: str | None = Field(
        default=None,
        description=(
            "Optional graph id; a UUID is assigned at assemble time if omitted."
        ),
    )
    turns: list[ConversationTurnData] = Field(
        description="Turns in this conversation, each with inline parent refs.",
    )

    @model_validator(mode="after")
    def _validate_unique_node_ids(self) -> ConversationGraphData:
        node_ids = [turn.node_id for turn in self.turns]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("ConversationGraphData turn node_id values must be unique")
        return self
