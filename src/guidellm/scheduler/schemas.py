"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations. Central to
the scheduler architecture, enabling request lifecycle tracking, backend
coordination, and state management across distributed worker processes.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from enum import Enum
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import Field, model_validator
from typing_extensions import TypeAliasType

from guidellm.schemas import RequestInfo, RequestSettings, StandardBaseModel
from guidellm.utils.registry import RegistryMixin, RegistryObjT

__all__ = [
    "BackendInterface",
    "BackendT",
    "BranchDistribution",
    "ConversationEdge",
    "ConversationGraph",
    "ConversationNode",
    "ConversationT",
    "DatasetIterT",
    "HistoryContext",
    "HistoryT",
    "RequestDataT",
    "RequestT",
    "ResponseT",
    "SchedulerMessagingPydanticRegistry",
    "SchedulerProgress",
    "SchedulerState",
    "SchedulerUpdateAction",
]

RequestT = TypeVar("RequestT")
"Generic request object type for scheduler processing"

ResponseT = TypeVar("ResponseT")
"Generic response object type returned by backend processing"

RequestDataT = TypeAliasType(
    "RequestDataT",
    tuple[RequestT, RequestInfo],
    type_params=(RequestT,),
)
"""Request including external metadata and scheduling config."""

ConversationT = TypeAliasType(
    "ConversationT",
    list[RequestDataT[RequestT]],
    type_params=(RequestT,),
)

HistoryT = TypeAliasType(
    "HistoryT",
    list[tuple[RequestT, ResponseT | None]],
    type_params=(RequestT, ResponseT),
)
"""Record of requests + responses in conversation."""

# NOTE: This is the interface between data and scheduler.
DatasetIterT = TypeAliasType(
    "DatasetIterT",
    Iterable[Iterable[tuple[RequestT, RequestSettings]]],
    type_params=(RequestT,),
)
"""
Output of data loader, an iterable of batches,
where each batch is an iterable of (request, timestamp) tuples.
"""


class HistoryContext(str, Enum):
    """
    Controls how much conversational context flows through an edge
    from source node to target node.

    Determines the walk-back behavior during history assembly:
    the worker uses each incoming edge's ``history_context`` to decide
    what history the target node receives from that parent.
    """

    NEW = "new"
    """
    No context from the source. The target starts with a fresh,
    independent context. Used for spawning sub-agents that operate
    in isolation (e.g., Cursor subagents, Anthropic managed agent
    threads). The target's prompt contains all needed context.

    Also used when sub-agent results or compaction summaries are
    pre-recorded in the dataset: the consumer node's prompt already
    embeds the expected output, so no runtime data flow is needed.
    """

    LAST = "last"
    """
    Only the source's final (request, response) pair flows through.
    The target does not receive the source's ancestor history.
    Creates a history boundary -- downstream walk-backs stop here.
    Used for collecting sub-agent results, fan-in aggregation,
    and compaction consumers.
    """

    FULL = "full"
    """
    The source's entire ancestor history flows through via walk-back.
    Not a boundary -- downstream walk-backs continue through this edge.
    Used for sequential conversation continuation, multi-turn chains,
    and agent-to-agent handoffs where the receiving agent sees the
    full prior transcript.
    """


class BranchDistribution(str, Enum):
    """
    Controls how independent DAG branches are distributed across workers.
    """

    LOCAL = "local"
    """
    All branches of a graph stay on one worker. Uses asyncio.Semaphore
    concurrency within the worker process. Best for benchmark
    reproducibility and avoiding IPC overhead on short branches.
    """

    AUTO = "auto"
    """
    Coordinator decides based on worker availability and graph size.

    .. note:: Not yet implemented. Raises ``NotImplementedError`` at runtime.
    """

    DISTRIBUTED = "distributed"
    """
    Always dispatch independent branches to separate workers when
    available. Maximizes throughput for large fan-outs.

    .. note:: Not yet implemented. Raises ``NotImplementedError`` at runtime.
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
        default=HistoryContext.FULL,
        description="How much conversational context flows through this edge.",
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
                raise ValueError(
                    f"Edge source '{edge.source_node_id}' not in nodes"
                )
            if edge.target_node_id not in node_ids:
                raise ValueError(
                    f"Edge target '{edge.target_node_id}' not in nodes"
                )

    def _build_adjacency(
        self, node_ids: set[str]
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        in_degree = dict.fromkeys(node_ids, 0)
        children: dict[str, list[str]] = {nid: [] for nid in node_ids}
        full_parent_count = dict.fromkeys(node_ids, 0)

        for edge in self.edges:
            in_degree[edge.target_node_id] += 1
            children[edge.source_node_id].append(edge.target_node_id)
            if edge.history_context == HistoryContext.FULL:
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
        queue: deque[str] = deque(
            nid for nid, d in deg.items() if d == 0
        )
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
        self.root_node_ids = sorted(
            nid for nid in node_ids if nid not in incoming
        )


class SchedulerMessagingPydanticRegistry(RegistryMixin[RegistryObjT]):
    """
    Registry for Pydantic types used in scheduler inter-process messaging.

    Enables generic interface for defining Pydantic class types used for
    communication between distributed scheduler components and worker processes.
    """


class BackendInterface(Protocol, Generic[RequestT, ResponseT]):
    """
    Protocol defining the interface for request processing backends.

    Establishes the contract for backend implementations that process requests
    within the scheduler system. Backends manage initialization, validation,
    processing, and shutdown lifecycle. All properties must be pickleable before
    process_startup is called for multi-process environments.

    Example:
    ::
        class CustomBackend(BackendInterface):
            @property
            def processes_limit(self) -> int:
                return 4

            async def resolve(self, request, request_info, history=None):
                yield response, updated_request_info
    """

    @property
    def processes_limit(self) -> int | None:
        """
        :return: Maximum worker processes supported, or None if unlimited
        """

    @property
    def requests_limit(self) -> int | None:
        """
        :return: Maximum concurrent requests supported, or None if unlimited
        """

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: Backend metadata including model initialization and configuration
        """

    async def process_startup(self) -> None:
        """
        Perform backend initialization and startup procedures.

        :raises Exception: Implementation-specific exceptions for startup failures
        """

    async def validate(self) -> None:
        """
        Validate backend configuration and operational status.

        :raises Exception: Implementation-specific exceptions for validation failures
        """

    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        :raises Exception: Implementation-specific exceptions for shutdown failures
        """

    async def resolve(
        self,
        request: RequestT,
        request_info: RequestInfo,
        history: HistoryT[RequestT, ResponseT] | None = None,
    ) -> AsyncIterator[tuple[ResponseT | None, RequestInfo]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process
        :param request_info: Scheduling metadata and timing information
        :param history: Conversation history for multi-turn requests
        :yield: Tuples of (response, updated_request_info) for each response chunk.
            Response may be None for intermediate updates (e.g., first token arrival).
        :raises Exception: Implementation-specific exceptions for processing failures
        """


BackendT = TypeVar("BackendT", bound=BackendInterface)
"Generic backend interface type for request processing"


class SchedulerProgress(StandardBaseModel):
    """
    Progress tracking data for scheduler operations.

    Provides estimates for remaining work in scheduler operations, including
    fraction complete, request counts, and duration. Used by constraints and
    monitoring systems to track execution progress and make termination decisions.
    """

    remaining_requests: float | None = Field(
        description="Estimated number of remaining requests to process", default=None
    )
    total_requests: float | None = Field(
        description="Total number of requests to process", default=None
    )
    remaining_duration: float | None = Field(
        description="Estimated remaining duration in seconds", default=None
    )
    total_duration: float | None = Field(
        description="Total duration in seconds to process for", default=None
    )
    stop_time: float | None = Field(
        description="The timestamp the processing stopped at", default=None
    )

    @property
    def remaining_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining progress, if known
        """
        fraction: float | None = None

        if (requests_fraction := self.remaining_requests_fraction) is not None:
            fraction = requests_fraction

        if (duration_fraction := self.remaining_duration_fraction) is not None:
            fraction = (
                duration_fraction
                if fraction is None
                else min(fraction, duration_fraction)
            )

        return fraction

    @property
    def remaining_requests_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining requests, if known
        """
        return (
            self.remaining_requests / float(self.total_requests)
            if self.remaining_requests is not None
            and self.total_requests is not None
            and self.total_requests > 0
            else None
        )

    @property
    def remaining_duration_fraction(self) -> float | None:
        """
        :return: Estimated fraction of remaining duration, if known
        """
        return (
            self.remaining_duration / float(self.total_duration)
            if self.remaining_duration is not None
            and self.total_duration is not None
            and self.total_duration > 0
            else None
        )

    def combine(self, other: SchedulerProgress) -> SchedulerProgress:
        """
        Combine two progress instances, taking the minimum remaining estimates.

        :param other: Another progress instance to combine with
        :return: New progress instance with combined estimates
        """
        if (other_req_fraction := other.remaining_requests_fraction) is not None and (
            (cur_req_fraction := self.remaining_requests_fraction) is None
            or other_req_fraction < cur_req_fraction
        ):
            # Only update if the other is more advanced (lower fraction)
            self.remaining_requests = other.remaining_requests
            self.total_requests = other.total_requests

        if (other_dur_fraction := other.remaining_duration_fraction) is not None and (
            (cur_dur_fraction := self.remaining_duration_fraction) is None
            or other_dur_fraction < cur_dur_fraction
        ):
            # Only update if the other is more advanced (lower fraction)
            self.remaining_duration = other.remaining_duration
            self.total_duration = other.total_duration

        if other.stop_time is not None and (
            self.stop_time is None or other.stop_time < self.stop_time
        ):
            # Only update if the other has an earlier stop time
            self.stop_time = other.stop_time

        return self


class SchedulerUpdateAction(StandardBaseModel):
    """
    Control directives for scheduler behavior and operations.

    Encapsulates control signals for scheduler operations including request
    queuing and processing directives. Used by constraints to communicate
    termination conditions and progress to scheduler components.

    Example:
    ::
        action = SchedulerUpdateAction(
            request_queuing="stop",
            request_processing="continue",
            metadata={"reason": "max_requests_reached"}
        )
    """

    request_queuing: Literal["continue", "stop"] = Field(
        default="continue", description="Action to take for request queuing operations"
    )
    request_processing: Literal["continue", "stop_local", "stop_all"] = Field(
        default="continue",
        description="Action to take for request processing operations",
    )
    stopping_scope: Literal["current", "all"] = Field(
        default="current",
        description=(
            "Whether this constraint's stop signal should halt only the current "
            "benchmark ('current') or also prevent escalation to subsequent "
            "rates/streams ('all')."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context and data for the scheduler action",
    )
    progress: SchedulerProgress = Field(
        default_factory=SchedulerProgress,
        description="Progress information for the scheduler action",
    )


class SchedulerState(StandardBaseModel):
    """
    Comprehensive state tracking for scheduler execution.

    Tracks scheduler execution progress, request counts, timing information,
    and constraint enforcement. Central to scheduler coordination, providing
    real-time metrics for monitoring and decision-making across distributed
    worker processes.

    Example:
    ::
        state = SchedulerState(node_id=0, num_processes=4)
        state.created_requests += 1
        state.queued_requests += 1
        completion_rate = state.processed_requests / state.created_requests
    """

    node_id: int = Field(
        description="Unique identifier for this scheduler node", default=-1
    )
    num_processes: int = Field(
        description="Number of worker processes in this scheduler", default=-1
    )
    start_time: float = Field(
        description="Unix timestamp when the scheduler started",
        default_factory=time.time,
    )
    end_time: float | None = Field(
        default=None, description="Unix timestamp when the scheduler stopped"
    )
    start_requests_time: float | None = Field(
        default=None, description="Unix timestamp of the first sent request"
    )
    end_requests_time: float | None = Field(
        default=None, description="Unix timestamp of the last finalized request"
    )
    end_queuing_time: float | None = Field(
        default=None, description="Unix timestamp when request queuing stopped"
    )
    end_queuing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: float | None = Field(
        default=None, description="Unix timestamp when request processing stopped"
    )
    end_processing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
    )
    scheduler_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Latest state from all constraints applied during scheduler run",
    )

    progress: SchedulerProgress = Field(
        default_factory=SchedulerProgress,
        description="Overall progress information for the scheduler run",
    )

    created_requests: int = Field(
        default=0, description="Total number of requests created"
    )
    queued_requests: int = Field(
        default=0, description="Total number of requests queued for processing"
    )
    pending_requests: int = Field(
        default=0,
        description="Number of requests pending processing within a worker",
    )
    processing_requests: int = Field(
        default=0, description="Number of requests currently being processed"
    )
    processed_requests: int = Field(
        default=0, description="Number of requests that completed processing"
    )
    successful_requests: int = Field(
        default=0, description="Number of requests that completed successfully"
    )
    errored_requests: int = Field(
        default=0, description="Number of requests that failed with errors"
    )
    cancelled_requests: int = Field(
        default=0, description="Number of requests that were cancelled"
    )
