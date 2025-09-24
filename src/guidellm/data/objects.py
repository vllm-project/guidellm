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
    content: Any | None = Field(
        default=None,
        description="Raw content to send in the request body, if applicable.",
    )
    files: dict[str, Any] | None = Field(
        default=None,
        description="Files to include in the request, if applicable.",
    )
    json: dict[str, Any] | None = Field(
        default=None,
        description="JSON content to include in the request body, if applicable.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Query parameters to include in the request URL, if applicable.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers to include in the request, if applicable.",
    )

    def update(self, *payloads: GenerationRequestArguments | dict[str, Any]) -> None:
        for payload in payloads:
            if not isinstance(payload, GenerationRequestArguments):
                payload = GenerationRequestArguments(**payload)  # noqa: PLW2901

            self.url = payload.url or self.url
            self.method = payload.method or self.method
            self.stream = payload.stream if payload.stream is not None else self.stream
            self.content = payload.content or self.content

            if self.files is None:
                self.files = {}
            self.files.update(payload.files or {})

            if self.json is None:
                self.json = {}
            self.json.update(payload.json or {})

            if self.params is None:
                self.params = {}
            self.params.update(payload.params or {})

            if self.headers is None:
                self.headers = {}
            self.headers.update(payload.headers or {})


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
