"""
Pydantic schema models for GuideLLM operations.

Provides standardized data models and type definitions for generation requests,
responses, timing measurements, and statistics aggregation. These schemas ensure
type safety and consistent data handling across the benchmarking pipeline,
from request submission through backend processing to results compilation.
"""

from __future__ import annotations

from guidellm.schemas.base import (
    BaseModelT,
    DistributionSummary,
    ErroredT,
    FunctionObjT,
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    GenerativeRequestStats,
    IncompleteT,
    Percentiles,
    PydanticClassRegistryMixin,
    RegisterClassT,
    ReloadableBaseModel,
    RequestInfo,
    RequestSettings,
    RequestTimings,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
    SuccessfulT,
    ToolCall,
    ToolCallFunction,
    TotalT,
    TurnType,
    UsageMetrics,
    standard_model_config,
)

__all__ = [
    "BaseModelT",
    "DistributionSummary",
    "ErroredT",
    "FunctionObjT",
    "GenerationRequest",
    "GenerationRequestArguments",
    "GenerationResponse",
    "GenerativeRequestStats",
    "IncompleteT",
    "Percentiles",
    "PydanticClassRegistryMixin",
    "RegisterClassT",
    "ReloadableBaseModel",
    "RequestInfo",
    "RequestSettings",
    "RequestTimings",
    "StandardBaseDict",
    "StandardBaseModel",
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
