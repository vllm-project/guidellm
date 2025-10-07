from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import Field

from guidellm.scheduler import (
    MeasuredRequestTimings,
    SchedulerMessagingPydanticRegistry,
)
from guidellm.utils import StandardBaseDict, StandardBaseModel

__all__ = [
    "GenerationRequest",
    "GenerationRequestArguments",
    "GenerationRequestTimings",
    "GenerativeDatasetColumnType",
    "GenerativeRequestType",
]


GenerativeRequestType = Literal[
    "text_completions",
    "chat_completions",
    "audio_transcriptions",
    "audio_translations",
]

GenerativeDatasetColumnType = Literal[
    "prompt_tokens_count_column",
    "output_tokens_count_column",
    "prefix_column",
    "text_column",
    "image_column",
    "video_column",
    "audio_column",
]


class GenerationRequestArguments(StandardBaseDict):
    @classmethod
    def model_combine_dict(  # noqa: C901, PLR0912
        cls, *arguments: GenerationRequestArguments | dict[str, Any]
    ) -> dict[str, Any]:
        combined = {}

        for args in arguments:
            args_dict = args if isinstance(args, dict) else args.model_dump()
            combined["url"] = args_dict.get("url", combined.get("url"))
            combined["path"] = args_dict.get("path", combined.get("path"))
            combined["method"] = args_dict.get("method", combined.get("method"))
            combined["stream"] = args_dict.get("stream", combined.get("stream"))
            combined["content_body"] = args_dict.get(
                "content_body", combined.get("content_body")
            )

            if (json_body := args_dict.get("json_body")) is not None:
                combined["json_body"] = combined.get("json_body", {}) + json_body
            if (files := args_dict.get("files")) is not None:
                combined["files"] = combined.get("files", {}) + files
            if (params := args_dict.get("params")) is not None:
                combined["params"] = combined.get("params", {}) + params
            if (headers := args_dict.get("headers")) is not None:
                combined["headers"] = combined.get("headers", {}) + headers

        return combined

    url: str | None = Field(
        default=None,
        description="The URL endpoint to which the request will be sent.",
    )
    path: str | None = Field(
        default=None,
        description="The path to append to the base URL for the request.",
    )
    method: str | None = Field(
        default=None,
        description="The HTTP method to use for the request (e.g., 'POST', 'GET').",
    )
    stream: bool | None = Field(
        default=None,
        description="Whether to stream the response, if applicable.",
    )
    content_body: Any | None = Field(
        default=None,
        description="Raw content to send in the request body, if applicable.",
    )
    json_body: dict[str, Any] | None = Field(
        default=None,
        description="JSON content to include in the request body, if applicable.",
    )
    files: dict[str, Any] | None = Field(
        default=None,
        description="Files to include in the request, if applicable.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Query parameters to include in the request URL, if applicable.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers to include in the request, if applicable.",
    )


@SchedulerMessagingPydanticRegistry.register()
class GenerationRequest(StandardBaseModel):
    """Request model for backend generation operations."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request.",
    )
    request_type: GenerativeRequestType | str = Field(
        description=(
            "Type of request. If url is not provided in arguments, "
            "this will be used to determine the request url."
        ),
    )
    arguments: GenerationRequestArguments = Field(
        description=(
            "Payload for the request, structured as a dictionary of arguments to pass "
            "to the respective backend method. For example, can contain "
            "'json', 'headers', 'files', etc."
        )
    )
    stats: dict[Literal["prompt_tokens", "output_tokens"], int] = Field(
        default_factory=dict,
        description="Request statistics including prompt and output token counts.",
    )


@SchedulerMessagingPydanticRegistry.register()
@MeasuredRequestTimings.register("generation_request_timings")
class GenerationRequestTimings(MeasuredRequestTimings):
    """Timing model for tracking generation request lifecycle events."""

    timings_type: Literal["generation_request_timings"] = "generation_request_timings"
    first_iteration: float | None = Field(
        default=None,
        description="Unix timestamp when the first generation iteration began.",
    )
    last_iteration: float | None = Field(
        default=None,
        description="Unix timestamp when the last generation iteration completed.",
    )
