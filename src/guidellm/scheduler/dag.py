"""
DAG execution utilities for conversation graph processing.

Provides the core algorithms for executing conversation DAGs within a single
worker: topological ordering, node readiness tracking, walk-back history
assembly, and graph-level error handling. These utilities are independent of
the IPC/messaging layer and can be integrated into the worker process when
graph-native data sources are available.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from typing import Generic, NamedTuple, TypeVar

from guidellm.scheduler.schemas import (
    ConversationEdge,
    ConversationGraph,
)

__all__ = [
    "CompletedNodeData",
    "DAGExecutionState",
]


_RequestT = TypeVar("_RequestT")
_ResponseT = TypeVar("_ResponseT")


class CompletedNodeData(NamedTuple, Generic[_RequestT, _ResponseT]):
    """
    Stored result for a completed DAG node.

    :param request: The original request that was executed.
    :param response: The response returned by the backend.
    """

    request: _RequestT
    response: _ResponseT


class DAGExecutionState(Generic[_RequestT, _ResponseT]):
    """
    Tracks execution state for a single conversation graph on one worker.

    Manages node readiness, completion tracking, and history assembly
    via walk-back. Designed for the ``local`` branch distribution mode
    where the entire graph executes within a single worker process.

    :param graph: The conversation graph to execute.
    """

    def __init__(self, graph: ConversationGraph[_RequestT]):
        self._graph = graph

        # Pre-compute adjacency structures for efficient lookups
        self._incoming_edges: dict[str, list[ConversationEdge]] = {
            nid: [] for nid in graph.nodes
        }
        self._outgoing_edges: dict[str, list[ConversationEdge]] = {
            nid: [] for nid in graph.nodes
        }
        self._parent_count: dict[str, int] = dict.fromkeys(graph.nodes, 0)

        for edge in graph.edges:
            self._incoming_edges[edge.target_node_id].append(edge)
            self._outgoing_edges[edge.source_node_id].append(edge)
            self._parent_count[edge.target_node_id] += 1

        self._completed: dict[str, CompletedNodeData[_RequestT, _ResponseT]] = {}
        self._remaining_parents: dict[str, int] = dict(self._parent_count)
        # Think-time gate: node is schedulable only after this timestamp.
        # Set when the last parent completes (dependency-ready).
        self._available_after: dict[str, float] = dict.fromkeys(graph.nodes, 0.0)
        self._in_progress: set[str] = set()
        self._aborted: bool = False

    @property
    def graph(self) -> ConversationGraph[_RequestT]:
        """
        :return: The conversation graph being executed.
        """
        return self._graph

    @property
    def is_complete(self) -> bool:
        """
        :return: True if all nodes have completed (or the graph was aborted).
        """
        return self._aborted or len(self._completed) == len(self._graph.nodes)

    @property
    def is_aborted(self) -> bool:
        """
        :return: True if the graph was aborted due to a node error.
        """
        return self._aborted

    def get_ready_nodes(self) -> list[str]:
        """
        Find nodes that are dependency-ready, past their think-time gate,
        and not already claimed or completed.

        :return: Sorted list of node IDs ready for execution.
        """
        if self._aborted:
            return []

        now = time.time()
        ready = []
        for nid in self._graph.nodes:
            if (
                nid not in self._completed
                and nid not in self._in_progress
                and self._remaining_parents[nid] == 0
                and now >= self._available_after[nid]
            ):
                ready.append(nid)
        return sorted(ready)

    def next_delayed_ready_at(self) -> float | None:
        """
        Earliest time when a dependency-ready but delayed node becomes
        schedulable.

        :return: Absolute timestamp, or ``None`` if no delayed nodes are
            waiting.
        """
        if self._aborted:
            return None

        now = time.time()
        earliest: float | None = None
        for nid in self._graph.nodes:
            if (
                nid not in self._completed
                and nid not in self._in_progress
                and self._remaining_parents[nid] == 0
                and self._available_after[nid] > now
            ):
                unlock_at = self._available_after[nid]
                if earliest is None or unlock_at < earliest:
                    earliest = unlock_at
        return earliest

    def claim_node(self, node_id: str) -> None:
        """
        Mark a ready node as in-progress so concurrent slots cannot
        select it again.

        :param node_id: The node to claim.
        :raises ValueError: If the node does not exist or is not claimable.
        """
        if node_id not in self._graph.nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        if node_id in self._completed or node_id in self._in_progress:
            raise ValueError(f"Node '{node_id}' is not claimable")
        if self._remaining_parents[node_id] != 0:
            raise ValueError(f"Node '{node_id}' still has unmet parents")
        if time.time() < self._available_after[node_id]:
            raise ValueError(f"Node '{node_id}' is still time-gated")
        self._in_progress.add(node_id)

    def mark_completed(
        self,
        node_id: str,
        request: _RequestT,
        response: _ResponseT,
    ) -> list[str]:
        """
        Mark a node as completed and return newly dependency-satisfied children.

        When a child's last parent completes, think time starts: the child's
        ``available_after`` is set to ``now + unlocking_parent.requeue_delay``.
        Returned children may still be time-gated and not yet schedulable.

        :param node_id: The ID of the completed node.
        :param request: The request that was executed.
        :param response: The response from the backend.
        :return: Sorted list of child node IDs that became dependency-ready.
        :raises ValueError: If the node is already completed or doesn't exist.
        """
        if node_id not in self._graph.nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        if node_id in self._completed:
            raise ValueError(f"Node '{node_id}' already completed")

        self._completed[node_id] = CompletedNodeData(request, response)
        self._in_progress.discard(node_id)

        delay = self._graph.nodes[node_id].settings.requeue_delay or 0.0
        unlock_at = time.time() + delay

        newly_ready: list[str] = []
        for edge in self._outgoing_edges[node_id]:
            child_id = edge.target_node_id
            self._remaining_parents[child_id] -= 1
            if (
                self._remaining_parents[child_id] == 0
                and child_id not in self._completed
            ):
                # Think time starts only once all parents are done.
                self._available_after[child_id] = unlock_at
                newly_ready.append(child_id)

        return sorted(newly_ready)

    def abort(self) -> list[str]:
        """
        Abort the graph, cancelling all remaining nodes.

        :return: List of node IDs that were not yet completed.
        """
        self._aborted = True
        return sorted(nid for nid in self._graph.nodes if nid not in self._completed)

    def assemble_history(
        self, node_id: str
    ) -> list[tuple[_RequestT, _ResponseT]] | None:
        """
        Assemble the flat history list for a node via walk-back.

        Follows incoming edges to build the conversation history that
        should be passed to ``backend.resolve()``. The algorithm is
        determined by the edge ``history_context`` values:

        - ``full``: Walk backwards through the parent chain collecting
          all ancestor (request, response) pairs. Stop at nodes without
          a ``full`` incoming edge.
        - ``last``: Collect only the parent's final (request, response).
        - ``new``: Skip -- no history from this parent.

        :param node_id: The node to assemble history for.
        :return: Flat list of (request, response) pairs in chronological
            order, or None if the node has no history (only ``new`` edges
            or no incoming edges).
        """
        incoming = self._incoming_edges.get(node_id, [])
        if not incoming:
            return None

        full_chain: list[tuple[_RequestT, _ResponseT]] = []
        last_entries: list[tuple[str, tuple[_RequestT, _ResponseT]]] = []
        has_any_history = False

        # Walk-back through the single full parent (if any)
        full_edge = self._find_full_parent_edge(incoming)
        if full_edge is not None:
            has_any_history = True
            full_chain = self._walk_back_full(full_edge.source_node_id)

        # Collect last entries (sorted by source_node_id for determinism)
        for edge in sorted(incoming, key=lambda e: e.source_node_id):
            if edge.history_context == "last":
                has_any_history = True
                completed = self._completed.get(edge.source_node_id)
                if completed is not None:
                    last_entries.append(
                        (edge.source_node_id, (completed.request, completed.response))
                    )

        if not has_any_history:
            return None

        # Combine: full chain in chronological order, then last outputs
        result: list[tuple[_RequestT, _ResponseT]] = list(full_chain)
        for _, entry in last_entries:
            result.append(entry)

        return result if result else None

    def _find_full_parent_edge(
        self, incoming: Iterable[ConversationEdge]
    ) -> ConversationEdge | None:
        """
        Find the single ``full`` incoming edge, if any.

        Graph validation ensures at most one ``full`` incoming edge per node.

        :param incoming: The incoming edges for a node.
        :return: The full edge, or None.
        """
        for edge in incoming:
            if edge.history_context == "full":
                return edge
        return None

    def _walk_back_full(self, start_node_id: str) -> list[tuple[_RequestT, _ResponseT]]:
        """
        Walk backwards through ``full`` edges collecting ancestor history.

        Collects (request, response) pairs from the start node back through
        the chain of ``full`` parents, stopping when a node has no ``full``
        incoming edge (i.e., it was reached via ``new``, ``last``, or is a
        root). At intermediate nodes where the walk continues, also collects
        ``last`` parent outputs. ``last`` parents at the stopping node are
        NOT collected -- they belong to the stopping node's own context.

        :param start_node_id: The node to start walking back from.
        :return: List of (request, response) pairs in chronological order.
        """
        chain_reversed: list[tuple[_RequestT, _ResponseT]] = []
        interleaved_last: list[tuple[str, tuple[_RequestT, _ResponseT]]] = []
        current_id: str | None = start_node_id

        while current_id is not None:
            completed = self._completed.get(current_id)
            if completed is None:
                break

            chain_reversed.append((completed.request, completed.response))

            current_incoming = self._incoming_edges.get(current_id, [])
            full_edge = self._find_full_parent_edge(current_incoming)

            # Only collect last parents at nodes where the walk CONTINUES
            # (has a full parent). At the stopping node, last parents are
            # the node's own context, not part of downstream history.
            if full_edge is not None:
                for edge in sorted(current_incoming, key=lambda e: e.source_node_id):
                    if edge.history_context == "last":
                        last_completed = self._completed.get(edge.source_node_id)
                        if last_completed is not None:
                            interleaved_last.append(
                                (
                                    edge.source_node_id,
                                    (
                                        last_completed.request,
                                        last_completed.response,
                                    ),
                                )
                            )

            current_id = full_edge.source_node_id if full_edge is not None else None

        chain_reversed.reverse()

        result = list(chain_reversed)
        for _, entry in interleaved_last:
            result.append(entry)

        return result

    def get_remaining_node_ids(self) -> list[str]:
        """
        Get all node IDs that haven't been completed yet.

        :return: Sorted list of incomplete node IDs.
        """
        return sorted(nid for nid in self._graph.nodes if nid not in self._completed)

    def topological_order(self) -> list[str]:
        """
        Compute topological ordering of graph nodes via BFS (Kahn's algorithm).

        :return: List of node IDs in topological order.
        """
        in_degree: dict[str, int] = dict(self._parent_count)
        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for edge in self._outgoing_edges[nid]:
                child_id = edge.target_node_id
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        return order
