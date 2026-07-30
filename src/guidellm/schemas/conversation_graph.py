"""
Concrete conversation graph schemas for generative benchmarks.

Binds the generic :class:`~guidellm.scheduler.schemas.ConversationGraph` and
:class:`~guidellm.scheduler.schemas.ConversationNode` to
:class:`~guidellm.schemas.request.GenerationRequest`, providing the data pipeline
and benchmark layer with typed graph models and backward-compatible helpers for
converting linear conversation chains into degenerate single-path graphs.
"""

from __future__ import annotations

import uuid
from typing import cast

from typing_extensions import Self

from guidellm.scheduler.schemas import (
    ConversationEdge,
    ConversationGraph,
    ConversationNode,
    HistoryContext,
)
from guidellm.schemas.info import RequestSettings
from guidellm.schemas.request import GenerationRequest

__all__ = [
    "GenerativeConversationGraph",
    "GenerativeConversationNode",
]


class GenerativeConversationNode(ConversationNode[GenerationRequest]):
    """
    Concrete conversation node binding for generative benchmarks.

    Binds ``RequestT = GenerationRequest`` so the data pipeline and benchmark
    layer can work with fully typed graph nodes.
    """


class GenerativeConversationGraph(ConversationGraph[GenerationRequest]):
    """
    Concrete conversation graph binding for generative benchmarks.

    Binds ``RequestT = GenerationRequest`` and provides convenience methods
    for constructing graphs from existing data formats.
    """

    @classmethod
    def from_linear_chain(
        cls,
        requests: list[tuple[GenerationRequest, RequestSettings]],
        agent_id: str = "default",
    ) -> Self:
        """
        Wrap a linear list of request/settings pairs as a single-path graph.

        Each request becomes a node connected to the next via a ``full``
        edge, preserving the existing multi-turn conversation semantics.
        Scheduling metadata is taken from the pair's ``RequestSettings``,
        not from the request payload (keeps ``RequestT`` generic).

        :param requests: Ordered list of ``(request, settings)`` pairs
            forming a conversation chain.
        :param agent_id: Agent identifier to assign to all nodes.
            Defaults to ``"default"``.
        :return: A conversation graph with one path through all requests.
        :raises ValueError: If the requests list is empty.
        """
        if not requests:
            raise ValueError("Cannot create a graph from an empty request list")

        graph_id = str(uuid.uuid4())
        nodes: dict[str, GenerativeConversationNode] = {}
        edges: list[ConversationEdge] = []

        node_ids: list[str] = []
        for i, (request, settings) in enumerate(requests):
            node_id = f"turn_{i}"
            node_ids.append(node_id)
            nodes[node_id] = GenerativeConversationNode(
                node_id=node_id,
                agent_id=agent_id,
                request=request,
                settings=settings,
            )

        for i in range(len(node_ids) - 1):
            edges.append(
                ConversationEdge(
                    source_node_id=node_ids[i],
                    target_node_id=node_ids[i + 1],
                    history_context="full",
                )
            )

        return cls(
            graph_id=graph_id,
            nodes=cast("dict[str, ConversationNode[GenerationRequest]]", nodes),
            edges=edges,
        )

    @classmethod
    def from_nodes_with_parents(
        cls,
        nodes: dict[str, GenerativeConversationNode],
        parents_by_node: dict[str, list[tuple[str, HistoryContext]]],
        graph_id: str | None = None,
    ) -> Self:
        """
        Build a graph from nodes and inline parent dependency refs.

        :param nodes: Mapping of node_id to fully constructed nodes.
        :param parents_by_node: For each node_id, a list of
            ``(parent_node_id, history_context)`` pairs. Nodes omitted or
            mapped to an empty list are roots.
        :param graph_id: Optional graph id; a UUID is generated if omitted.
        :return: A conversation graph with edges derived from parent refs.
        :raises ValueError: If ``nodes`` is empty or a parent id is missing.
        """
        if not nodes:
            raise ValueError("Cannot create a graph from an empty node map")

        edges: list[ConversationEdge] = []
        for node_id, parents in parents_by_node.items():
            if node_id not in nodes:
                raise ValueError(f"Parent refs reference unknown node_id {node_id!r}")
            for parent_node_id, history_context in parents:
                if parent_node_id not in nodes:
                    raise ValueError(
                        f"Parent node_id {parent_node_id!r} for {node_id!r} "
                        "is not in the node map"
                    )
                edges.append(
                    ConversationEdge(
                        source_node_id=parent_node_id,
                        target_node_id=node_id,
                        history_context=history_context,
                    )
                )

        return cls(
            graph_id=graph_id or str(uuid.uuid4()),
            nodes=cast("dict[str, ConversationNode[GenerationRequest]]", nodes),
            edges=edges,
        )
