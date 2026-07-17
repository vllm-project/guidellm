from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Iterable
from typing import Generic, Literal

from pydantic import Field, model_validator
from typing_extensions import Self, TypeAliasType

from guidellm.schemas import RequestInfo, RequestSettings, StandardBaseModel

from .types import RequestT

__all__ = [
    "BranchDistribution",
    "ConversationEdge",
    "ConversationGraph",
    "ConversationNode",
    "ConversationT",
    "DatasetIterT",
    "HistoryContext",
]

HistoryContext = Literal["new", "last", "full"]
"""
Controls how much conversational context flows through an edge
from source node to target node.

Determines the walk-back behavior during history assembly:
the worker uses each incoming edge's ``history_context`` to decide
what history the target node receives from that parent.

- ``"new"``: No context from the source. The target starts with a fresh,
  independent context. Used for spawning sub-agents that operate in
  isolation (e.g., Cursor subagents, Anthropic managed agent threads).
  The target's prompt contains all needed context. Also used when
  sub-agent results or compaction summaries are pre-recorded in the
  dataset: the consumer node's prompt already embeds the expected
  output, so no runtime data flow is needed.
- ``"last"``: Only the source's final (request, response) pair flows
  through. The target does not receive the source's ancestor history.
  Creates a history boundary -- downstream walk-backs stop here. Used
  for collecting sub-agent results, fan-in aggregation, and compaction
  consumers.
- ``"full"``: The source's entire ancestor history flows through via
  walk-back. Not a boundary -- downstream walk-backs continue through
  this edge. Used for sequential conversation continuation, multi-turn
  chains, and agent-to-agent handoffs where the receiving agent sees
  the full prior transcript.
"""

BranchDistribution = Literal["local", "distributed"]
"""
Controls how independent DAG branches are distributed across workers.

- ``"local"``: All branches of a graph stay on one worker. Uses
  asyncio.Semaphore concurrency within the worker process. Best for
  benchmark reproducibility and avoiding IPC overhead on short branches.
- ``"distributed"``: Always dispatch independent branches to separate
  workers when available. Maximizes throughput for large fan-outs.
  Not yet implemented; raises ``NotImplementedError`` at runtime.
"""


class ConversationEdge(StandardBaseModel):
    """
    An edge in a conversation DAG connecting two nodes.

    Edges are non-generic: they reference nodes by string ID and carry
    the ``history_context`` that controls how much conversational context
    flows from source to target during history assembly.
    """

    source_node_id: str = Field(
        description="Node ID of the edge source (parent).",
    )
    target_node_id: str = Field(
        description="Node ID of the edge target (child).",
    )
    history_context: HistoryContext = Field(
        default="full",
        description=(
            "How much conversational context flows through this edge: "
            "'new' (none), 'last' (source pair only), or 'full' (ancestor walk-back)."
        ),
    )


class ConversationNode(StandardBaseModel, Generic[RequestT]):
    """
    A single node in a conversation DAG, generic over request type.

    Each node represents one LLM request with an agent identity and
    optional scheduling metadata. History behavior is determined by
    the incoming edges' ``history_context``, not by node properties.
    """

    node_id: str = Field(
        description="Unique identifier for this node within the graph.",
    )
    agent_id: str = Field(
        description="Identifier for the simulated agent that owns this request.",
    )
    request: RequestT = Field(
        description="The request payload for this node.",
    )
    settings: RequestSettings = Field(
        default_factory=RequestSettings,
        description=(
            "Per-request scheduling metadata from the dataset, "
            "for example trace replay relative timestamps."
        ),
    )


class ConversationGraph(StandardBaseModel, Generic[RequestT]):
    """
    A directed acyclic graph of conversation nodes, generic over request type.

    The scheduler operates on this structure without knowledge of concrete
    request types like ``GenerationRequest``. Validated on construction:

    - Cycle detection via topological sort (raises if cycles found)
    - All edge source/target node IDs reference existing nodes
    - ``root_node_ids`` derived as nodes with no incoming edges
    - At most one ``full`` incoming edge per node (initially)
    """

    graph_id: str = Field(
        description="Unique identifier for this conversation graph.",
    )
    nodes: dict[str, ConversationNode[RequestT]] = Field(
        description="Nodes in the graph, keyed by node_id.",
    )
    edges: list[ConversationEdge] = Field(
        default_factory=list,
        description="Directed edges connecting nodes in the graph.",
    )
    root_node_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Node IDs with no incoming edges. Computed during validation; "
            "user-provided values are overwritten."
        ),
    )
    request_infos: dict[str, RequestInfo] = Field(
        default_factory=dict,
        description=(
            "Per-node RequestInfo, populated by the scheduler coordinator "
            "before dispatch. Not part of the dataset -- created at runtime."
        ),
    )

    @classmethod
    def from_linear_chain(
        cls,
        requests: list[tuple[RequestT, RequestSettings]],
        agent_id: str = "default",
    ) -> Self:
        """
        Wrap a linear list of request/settings pairs as a single-path graph.

        Each request becomes a node connected to the next via a ``full``
        edge, preserving multi-turn conversation semantics.

        :param requests: Ordered list of ``(request, settings)`` pairs.
        :param agent_id: Agent identifier assigned to all nodes.
        :return: A conversation graph with one path through all requests.
        :raises ValueError: If the requests list is empty.
        """
        if not requests:
            raise ValueError("Cannot create a graph from an empty request list")

        nodes: dict[str, ConversationNode[RequestT]] = {}
        edges: list[ConversationEdge] = []
        node_ids: list[str] = []

        for i, (request, settings) in enumerate(requests):
            node_id = f"turn_{i}"
            node_ids.append(node_id)
            nodes[node_id] = ConversationNode(
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

        return cls(graph_id=str(uuid.uuid4()), nodes=nodes, edges=edges)

    @model_validator(mode="after")
    def _validate_graph(self) -> ConversationGraph[RequestT]:
        """
        Validate the DAG structure on construction.

        Checks that all edge endpoints reference existing nodes, enforces
        the single-``full``-parent constraint, verifies acyclicity via
        Kahn's algorithm, and derives ``root_node_ids``.

        :raises ValueError: If edges reference missing nodes, a node has
            multiple ``full`` incoming edges, or the graph contains a cycle.
        """
        node_ids = set(self.nodes.keys())
        self._validate_edge_endpoints(node_ids)
        in_degree, children = self._build_adjacency(node_ids)
        self._check_acyclicity(node_ids, in_degree, children)
        self._derive_root_node_ids(node_ids)
        return self

    def _validate_edge_endpoints(self, node_ids: set[str]) -> None:
        for edge in self.edges:
            if edge.source_node_id not in node_ids:
                raise ValueError(f"Edge source '{edge.source_node_id}' not in nodes")
            if edge.target_node_id not in node_ids:
                raise ValueError(f"Edge target '{edge.target_node_id}' not in nodes")

    def _build_adjacency(
        self, node_ids: set[str]
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        in_degree = dict.fromkeys(node_ids, 0)
        children: dict[str, list[str]] = {nid: [] for nid in node_ids}
        full_parent_count = dict.fromkeys(node_ids, 0)

        for edge in self.edges:
            in_degree[edge.target_node_id] += 1
            children[edge.source_node_id].append(edge.target_node_id)
            if edge.history_context == "full":
                full_parent_count[edge.target_node_id] += 1

        for nid, count in full_parent_count.items():
            if count > 1:
                raise ValueError(
                    f"Node '{nid}' has {count} 'full' incoming edges; "
                    f"at most one is supported"
                )

        return in_degree, children

    def _check_acyclicity(
        self,
        node_ids: set[str],
        in_degree: dict[str, int],
        children: dict[str, list[str]],
    ) -> None:
        # Kahn's algorithm: work on a copy so we don't mutate the original
        deg = dict(in_degree)
        queue: deque[str] = deque(nid for nid, d in deg.items() if d == 0)
        visited_count = 0

        while queue:
            nid = queue.popleft()
            visited_count += 1
            for child_id in children[nid]:
                deg[child_id] -= 1
                if deg[child_id] == 0:
                    queue.append(child_id)

        if visited_count != len(node_ids):
            raise ValueError(
                "ConversationGraph contains a cycle; "
                f"{len(node_ids) - visited_count} nodes are unreachable "
                f"via topological sort"
            )

    def _derive_root_node_ids(self, node_ids: set[str]) -> None:
        incoming: set[str] = {edge.target_node_id for edge in self.edges}
        self.root_node_ids = sorted(nid for nid in node_ids if nid not in incoming)

    def subgraph_for_nodes(self, node_ids: set[str]) -> Self:
        """
        Build a validated subgraph containing only the given nodes.

        Used when request generation stops mid-graph so workers receive only
        nodes that have been queued (and have ``RequestInfo`` entries). A Kahn
        topological prefix is parent-closed, so truncating to queued node IDs
        always yields a valid DAG.

        :param node_ids: Node IDs to retain. Must be non-empty and a subset of
            this graph's nodes.
        :return: A new graph with the same ``graph_id``, filtered nodes/edges,
            and matching ``request_infos``.
        :raises ValueError: If ``node_ids`` is empty or references unknown IDs.
        """
        if not node_ids:
            raise ValueError("Cannot create a subgraph from an empty node set")

        unknown = node_ids - set(self.nodes)
        if unknown:
            raise ValueError(f"Unknown node IDs for subgraph: {sorted(unknown)}")

        return type(self)(
            graph_id=self.graph_id,
            nodes={nid: self.nodes[nid] for nid in node_ids},
            edges=[
                edge
                for edge in self.edges
                if (edge.source_node_id in node_ids and edge.target_node_id in node_ids)
            ],
            request_infos={
                nid: info for nid, info in self.request_infos.items() if nid in node_ids
            },
        )


ConversationT = TypeAliasType(
    "ConversationT",
    ConversationGraph[RequestT],
    type_params=(RequestT,),
)
"""A conversation graph of requests for multi-turn / multi-agent workloads."""

# NOTE: This is the interface between data and scheduler.
DatasetIterT = TypeAliasType(
    "DatasetIterT",
    Iterable[ConversationGraph[RequestT]],
    type_params=(RequestT,),
)
"""
Output of the data loader: an iterable of conversation graphs.
"""
