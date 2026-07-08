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
from typing import Self

from guidellm.scheduler.schemas import (
    ConversationEdge,
    ConversationGraph,
    ConversationNode,
    HistoryContext,
)
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
        requests: list[GenerationRequest],
        agent_id: str = "default",
    ) -> Self:
        """
        Wrap a linear list of requests as a degenerate single-path graph.

        Each request becomes a node connected to the next via a ``full``
        edge, preserving the existing multi-turn conversation semantics.
        Used for backward compatibility with linear datasets.

        :param requests: Ordered list of generation requests forming a
            conversation chain.
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
        for i, request in enumerate(requests):
            node_id = f"turn_{i}"
            node_ids.append(node_id)
            nodes[node_id] = GenerativeConversationNode(
                node_id=node_id,
                agent_id=agent_id,
                request=request,
                settings=request.settings,
            )

        for i in range(len(node_ids) - 1):
            edges.append(
                ConversationEdge(
                    source_node_id=node_ids[i],
                    target_node_id=node_ids[i + 1],
                    history_context=HistoryContext.FULL,
                )
            )

        return cls(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
        )
