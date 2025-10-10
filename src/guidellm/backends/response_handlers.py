from __future__ import annotations

import json
from typing import Any, Protocol, cast

from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.utils import RegistryMixin

try:
    import orjson
except ImportError:
    orjson = None

__all__ = [
    "AudioResponseHandler",
    "ChatCompletionsResponseHandler",
    "GenerationResponseHandler",
    "GenerationResponseHandlerFactory",
    "TextCompletionsResponseHandler",
]


class GenerationResponseHandler(Protocol):
    def compile_non_streaming(
        self, request: GenerationRequest, response: Any
    ) -> GenerationResponse: ...

    def add_streaming_line(self, line: str) -> int | None: ...

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse: ...


class GenerationResponseHandlerFactory(RegistryMixin[type[GenerationResponseHandler]]):
    pass


@GenerationResponseHandlerFactory.register("text_completions")
class TextCompletionsResponseHandler(GenerationResponseHandler):
    def __init__(self):
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, int | dict[str, int]] | None = None

    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
    ) -> GenerationResponse:
        choices = cast("list[dict]", response.get("choices", []))
        usage = cast("dict[str, int | dict[str, int]]", response.get("usage", {}))
        input_metrics, output_metrics = self.extract_metrics(usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=choices[0].get("text", "") if choices else "",
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return 0

        line = line[len("data:") :].strip()
        data = cast(
            "dict[str, Any]",
            json.loads(line) if orjson is None else orjson.loads(line),
        )
        updated = False

        if (choices := cast("list[dict]", data.get("choices"))) and (
            text := choices[0].get("text")
        ):
            self.streaming_texts.append(text)
            updated = True

        if usage := cast("dict[str, int | dict[str, int]]", data.get("usage")):
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        if not usage:
            return UsageMetrics(), UsageMetrics()

        input_details = cast("dict[str, int]", usage.get("prompt_tokens_details", {}))
        output_details = cast(
            "dict[str, int]", usage.get("completion_tokens_details", {})
        )

        return UsageMetrics(
            text_tokens=input_details.get("prompt_tokens")
            or cast("int", usage.get("prompt_tokens")),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        ), UsageMetrics(
            text_tokens=output_details.get("completion_tokens")
            or cast("int", usage.get("completion_tokens")),
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )


@GenerationResponseHandlerFactory.register("chat_completions")
class ChatCompletionsResponseHandler(TextCompletionsResponseHandler):
    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
    ) -> GenerationResponse:
        choices = cast("list[dict]", response.get("choices", []))
        usage = cast("dict[str, int | dict[str, int]]", response.get("usage", {}))
        input_metrics, output_metrics = self.extract_metrics(usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=cast("dict", choices[0].get("message", {})).get("content", "")
            if choices
            else "",
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return 0

        line = line[len("data:") :].strip()
        data = cast(
            "dict[str, Any]",
            json.loads(line) if orjson is None else orjson.loads(line),
        )
        updated = False

        # Extract delta content for chat completion chunks
        if choices := cast("list[dict]", data.get("choices")):
            delta = choices[0].get("delta", {})
            if content := delta.get("content"):
                self.streaming_texts.append(content)
            updated = True

        if usage := cast("dict[str, int | dict[str, int]]", data.get("usage")):
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@GenerationResponseHandlerFactory.register(
    ["audio_transcriptions", "audio_translations"]
)
class AudioResponseHandler:
    def __init__(self):
        self.streaming_buffer: bytearray = bytearray()
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, int | dict[str, int]] | None = None

    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
    ) -> GenerationResponse:
        usage = cast("dict[str, int]", response.get("usage", {}))
        input_details = cast("dict[str, int]", usage.get("input_token_details", {}))
        output_details = cast("dict[str, int]", usage.get("output_token_details", {}))
        text = response.get("text", "")

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=text,
            input_metrics=UsageMetrics(
                text_tokens=input_details.get("text_tokens", usage.get("input_tokens")),
                audio_tokens=input_details.get(
                    "audio_tokens", usage.get("input_tokens")
                ),
                audio_seconds=input_details.get("seconds", usage.get("seconds")),
            ),
            output_metrics=UsageMetrics(
                text_tokens=output_details.get(
                    "text_tokens", usage.get("output_tokens")
                ),
            ),
        )

    def add_streaming_line(self, line: str) -> int | None:
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("{"):
            return 0

        data = cast(
            "dict[str, Any]",
            json.loads(line) if orjson is None else orjson.loads(line),
        )
        updated = False

        if text := data.get("text"):
            self.streaming_texts.append(text)
            updated = True

        if usage := cast("dict[str, int | dict[str, int]]", data.get("usage")):
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        if not usage:
            return UsageMetrics(), UsageMetrics()

        input_details = cast("dict[str, int]", usage.get("input_token_details", {}))
        output_details = cast("dict[str, int]", usage.get("output_token_details", {}))

        return UsageMetrics(
            text_tokens=(
                input_details.get("text_tokens")
                or cast("int", usage.get("input_tokens"))
            ),
            audio_tokens=(
                input_details.get("audio_tokens")
                or cast("int", usage.get("audio_tokens"))
            ),
            audio_seconds=(
                input_details.get("seconds") or cast("int", usage.get("seconds"))
            ),
        ), UsageMetrics(
            text_tokens=output_details.get("text_tokens")
            or cast("int", usage.get("output_tokens")),
        )
