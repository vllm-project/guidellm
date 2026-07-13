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
from collections.abc import Callable
from typing import Any, Self

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

    @classmethod
    def from_linear_chain_with_branches(
        cls,
        main_requests: list[GenerationRequest],
        branches: list[dict[str, Any]],
        branch_request_factory: Callable[[int, int], GenerationRequest],
        main_agent_id: str = "default",
    ) -> Self:
        """
        Build a graph from a main chain with sub-agent branches.

        Each branch spawns at ``at_turn`` via a ``new`` edge and merges
        back at ``at_turn + 1`` via a ``last`` edge. Multiple branches
        at the same turn are supported.

        :param main_requests: Ordered list of main chain requests.
        :param branches: List of branch specs, each with ``at_turn``,
            ``turns``, and optionally ``agent_id``.
        :param branch_request_factory: Callable that takes
            ``(branch_index, turn_index)`` and returns a
            ``GenerationRequest`` for that branch turn.
        :param main_agent_id: Agent ID for the main chain nodes.
        :return: A conversation graph with main chain and branches.
        :raises ValueError: If main_requests is empty or branch specs
            are invalid.
        """
        if not main_requests:
            raise ValueError("Cannot create a graph from an empty request list")

        graph_id = str(uuid.uuid4())
        nodes: dict[str, GenerativeConversationNode] = {}
        edges: list[ConversationEdge] = []

        # Build main chain nodes
        main_ids: list[str] = []
        for i, request in enumerate(main_requests):
            node_id = f"main_{i}"
            main_ids.append(node_id)
            nodes[node_id] = GenerativeConversationNode(
                node_id=node_id,
                agent_id=main_agent_id,
                request=request,
                settings=request.settings,
            )

        # Connect main chain with full edges
        for i in range(len(main_ids) - 1):
            edges.append(
                ConversationEdge(
                    source_node_id=main_ids[i],
                    target_node_id=main_ids[i + 1],
                    history_context=HistoryContext.FULL,
                )
            )

        # Build branch nodes and edges
        for b_idx, branch in enumerate(branches):
            at_turn: int = branch["at_turn"]
            num_turns: int = branch["turns"]
            agent_id: str = branch.get("agent_id", "worker")
            merge_turn = at_turn + 1

            branch_ids: list[str] = []
            for t in range(num_turns):
                node_id = f"branch_{b_idx}_{t}"
                branch_ids.append(node_id)
                request = branch_request_factory(b_idx, t)
                nodes[node_id] = GenerativeConversationNode(
                    node_id=node_id,
                    agent_id=agent_id,
                    request=request,
                    settings=request.settings,
                )

            # Connect branch turns with full edges
            for t in range(len(branch_ids) - 1):
                edges.append(
                    ConversationEdge(
                        source_node_id=branch_ids[t],
                        target_node_id=branch_ids[t + 1],
                        history_context=HistoryContext.FULL,
                    )
                )

            # Spawn edge: main chain → first branch node (new context)
            edges.append(
                ConversationEdge(
                    source_node_id=main_ids[at_turn],
                    target_node_id=branch_ids[0],
                    history_context=HistoryContext.NEW,
                )
            )

            # Merge edge: last branch node → merge turn (last context)
            edges.append(
                ConversationEdge(
                    source_node_id=branch_ids[-1],
                    target_node_id=main_ids[merge_turn],
                    history_context=HistoryContext.LAST,
                )
            )

        return cls(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
        )
