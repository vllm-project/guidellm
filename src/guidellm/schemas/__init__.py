"""
Pydantic schema models for GuideLLM operations.

Provides standardized data models and type definitions for generation requests,
responses, timing measurements, and statistics aggregation. These schemas ensure
type safety and consistent data handling across the benchmarking pipeline,
from request submission through backend processing to results compilation.
"""

from __future__ import annotations

from .base import (
    BaseModelT,
    ErroredT,
    IncompleteT,
    PydanticClassRegistryMixin,
    RegisterClassT,
    ReloadableBaseModel,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
    SuccessfulT,
    TotalT,
    standard_model_config,
)
from .dag import DAGNode, ForkNode, JoinNode, RequestNode, SpawnNode, StartNode
from .info import RequestInfo, RequestSettings, RequestTimings
from .request import (
    GenerationRequest,
    GenerationRequestArguments,
    TurnType,
    UsageMetrics,
)
from .request_stats import GenerativeRequestStats
from .response import GenerationResponse
from .statistics import (
    DistributionSummary,
    FunctionObjT,
    Percentiles,
    StatusDistributionSummary,
)
from .tool_call import ToolCall, ToolCallFunction

__all__ = [
    "BaseModelT",
    "DAGNode",
    "DistributionSummary",
    "ErroredT",
    "ForkNode",
    "FunctionObjT",
    "GenerationRequest",
    "GenerationRequestArguments",
    "GenerationResponse",
    "GenerativeRequestStats",
    "IncompleteT",
    "JoinNode",
    "Percentiles",
    "PydanticClassRegistryMixin",
    "RegisterClassT",
    "ReloadableBaseModel",
    "RequestInfo",
    "RequestNode",
    "RequestSettings",
    "RequestTimings",
    "SpawnNode",
    "StandardBaseDict",
    "StandardBaseModel",
    "StartNode",
    "StatusBreakdown",
    "StatusDistributionSummary",
    "SuccessfulT",
    "ToolCall",
    "ToolCallFunction",
    "TotalT",
    "TurnType",
    "UsageMetrics",
    "standard_model_config",
]
