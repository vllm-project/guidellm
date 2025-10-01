from __future__ import annotations

import uuid
from typing import Any, Literal, get_args

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
    "GenerativeDatasetArgs",
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
            if (
                url := args.get("url") if isinstance(args, dict) else args.url
            ) is not None:
                combined["url"] = url

            if (
                path := args.get("path") if isinstance(args, dict) else args.path
            ) is not None:
                combined["path"] = path

            if (
                method := args.get("method") if isinstance(args, dict) else args.method
            ) is not None:
                combined["method"] = method

            if (
                stream := args.get("stream") if isinstance(args, dict) else args.stream
            ) is not None:
                combined["stream"] = stream

            if (
                content_body := (
                    args.get("content_body")
                    if isinstance(args, dict)
                    else args.content_body
                )
            ) is not None:
                combined["content_body"] = content_body

            if (
                json_body := (
                    args.get("json_body") if isinstance(args, dict) else args.json_body
                )
            ) is not None:
                if "json_body" not in combined:
                    combined["json_body"] = {}
                combined["json_body"].update(json_body)

            if (
                files := args.get("files") if isinstance(args, dict) else args.files
            ) is not None:
                if "files" not in combined:
                    combined["files"] = {}
                combined["files"].update(files)

            if (
                params := args.get("params") if isinstance(args, dict) else args.params
            ) is not None:
                if "params" not in combined:
                    combined["params"] = {}
                combined["params"].update(params)

            if (
                headers := (
                    args.get("headers") if isinstance(args, dict) else args.headers
                )
            ) is not None:
                if "headers" not in combined:
                    combined["headers"] = {}
                combined["headers"].update(headers)

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


class GenerativeDatasetArgs(StandardBaseDict):
    type_: str | None = None
    split: str | None = None
    prompt_tokens_count_column: str | None = None
    output_tokens_count_column: str | None = None
    text_column: str | list[str] | None = None
    image_column: str | list[str] | None = None
    video_column: str | list[str] | None = None
    audio_column: str | list[str] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_extra.items()
            if not key.endswith("_column")
        }

    def get_mapped_columns(
        self,
    ) -> dict[GenerativeDatasetColumnType | str, str | list[str]]:
        column_mapping: dict[GenerativeDatasetColumnType | str, list[str] | None] = {}

        # Add in any non None columns from the fields
        for column in get_args(GenerativeDatasetColumnType):
            value = getattr(self, column)
            if value is not None:
                column_mapping[column] = value

        # Enable flexibility for extra columns to be passed through and referenced later
        for extra in self.model_extra:
            if (
                extra.endswith("_column")
                and extra not in column_mapping
                and self.model_extra[extra] is not None
            ):
                column_mapping[extra] = self.model_extra[extra]

        return column_mapping
