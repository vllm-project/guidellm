"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations. Central to
the scheduler architecture, enabling request lifecycle tracking, backend
coordination, and state management across distributed worker processes.
"""

from __future__ import annotations

from .backend import BackendInterface, BackendT, SchedulerMessagingPydanticRegistry
from .conversation import (
    BranchDistribution,
    ConversationEdge,
    ConversationGraph,
    ConversationNode,
    ConversationT,
    DatasetIterT,
    HistoryContext,
)
from .state import SchedulerProgress, SchedulerState, SchedulerUpdateAction
from .types import (
    HistoryT,
    RequestDataT,
    RequestT,
    ResponseT,
)

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
