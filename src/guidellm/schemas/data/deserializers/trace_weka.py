from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from guidellm.schemas.data.deserializers.synthetic import (
    _require_mean_if_distribution_knobs,
)
from guidellm.schemas.data.deserializers.trace_common import TraceDataArgs
from guidellm.schemas.data.entrypoints import DataArgs
from guidellm.utils.imports import json

__all__ = ["WEKATraceFormatArgs"]


@DataArgs.register("weka")
class WEKATraceFormatArgs(TraceDataArgs):
    kind: Literal["weka"] = Field(
        default="weka",
        description="Type identifier for the WEKA trace format.",
    )
    timestamp_column: str = Field(
        default="t",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="in",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="out",
        description="Column name for output token counts in the trace file.",
    )
    conversation_id_column: str = Field(
        default="id",
        description="Column name for conversation UUIDs in the trace file.",
    )
    hash_ids_column: str = Field(
        default="hash_ids",
        description="Column name for lists of hash IDs in the trace file.",
    )
    hash_id_block_size: int = Field(
        gt=0,
        # Recommended in original github repository callanjfox/agentic-coding-analysis
        default=64,
        description="Amount of tokens represented by one hash ID.",
    )
    tools: list[dict[str, Any]] | None = Field(
        description=(
            "Tool definitions in OpenAI format. Traces do not include schemas; "
            "when this is None, a static placeholder tool definition is used "
            "on tool-call turns."
        ),
        default=None,
        examples=[
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Retrieve data from the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query"}
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    )
    tool_response_tokens: int | None = Field(
        description=(
            "Average number of tokens for synthetic tool call responses. "
            "When None (default), a short placeholder response is used."
        ),
        gt=0,
        default=None,
        examples=[10],
    )
    tool_response_tokens_stdev: int | None = Field(
        description="Standard deviation for tool response token count.",
        gt=0,
        default=None,
        examples=[1],
    )
    tool_response_tokens_min: int | None = Field(
        description="Minimum number of tokens for tool response.",
        gt=0,
        default=None,
        examples=[5],
    )
    tool_response_tokens_max: int | None = Field(
        description="Maximum number of tokens for tool response.",
        gt=0,
        default=None,
        examples=[20],
    )

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_tools(cls, v: Any) -> list[dict[str, Any]] | None:
        """Accept a JSON string from CLI/env the same way synthetic ``tools`` is passed.

        :param v: Parsed list, JSON string, or None.
        :return: Tool definitions, or None for the placeholder schema.
        """
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError) as err:
                raise ValueError(
                    f"tools string must be a JSON list of tool definitions, got {v!r}"
                ) from err
        if not isinstance(v, list):
            raise ValueError(f"tools must be a list of dicts, got {type(v)}")
        if not all(isinstance(item, dict) for item in v):
            raise ValueError("tools must be a list of dicts")
        return v

    @model_validator(mode="after")
    def _validate_tool_response_token_means(self) -> WEKATraceFormatArgs:
        """Require tool_response_tokens when its distribution knobs are set."""
        _require_mean_if_distribution_knobs(
            self.tool_response_tokens,
            self.tool_response_tokens_stdev,
            self.tool_response_tokens_min,
            self.tool_response_tokens_max,
            "tool_response_tokens",
        )
        return self
