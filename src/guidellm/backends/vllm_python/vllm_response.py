"""
VLLM-specific response handler for compiling OpenAI-style response dicts.

Compiles response dictionaries (choices+usage or text+usage) and streaming
SSE lines into GenerationResponse. Infers format from response shape; no
request type parameter required.
"""

from __future__ import annotations

from typing import Any, cast

from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.utils import json

__all__ = ["VLLMResponseHandler"]


class VLLMResponseHandler:
    """
    Response handler for the vLLM Python backend.

    Compiles OpenAI-style response dicts into GenerationResponse by inferring
    shape: choices+usage (text/chat) or text+usage (audio). Parses streaming
    deltas (choices[].text, choices[].delta.content, or text) without
    requiring a request type.
    """

    def __init__(self) -> None:
        """Initialize streaming accumulation state."""
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, Any] | None = None
        self.streaming_response_id: str | None = None

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        response_dict: dict[str, Any],
    ) -> GenerationResponse:
        """
        Compile a non-streaming response dict into GenerationResponse.

        Infers format from keys: "text" (audio-style) vs "choices" (text/chat).

        :param request: Original generation request
        :param response_dict: OpenAI-style response (choices+usage or text+usage)
        :return: GenerationResponse with text and metrics
        """
        text, usage = self._extract_text_and_usage(response_dict)
        input_metrics, output_metrics = self._extract_metrics(usage, text)
        request_args = self._get_request_args(request)

        return GenerationResponse(
            request_id=request.request_id,
            response_id=response_dict.get("id"),
            request_args=request_args,
            text=text or None,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single SSE line from a streaming response.

        Infers delta shape: "text", choices[].text, or choices[].delta.content.
        Returns None if line indicates completion (e.g. data: [DONE]).

        :param line: Raw SSE line (e.g. "data: {...}")
        :return: 1 if content was extracted, 0 if ignored, None if stream done
        """
        data = self._parse_line(line)
        if data is None:
            return None
        if not data:
            return 0

        if "id" in data and self.streaming_response_id is None:
            self.streaming_response_id = data.get("id")

        if "usage" in data:
            self.streaming_usage = data.get("usage") or self.streaming_usage

        result = 0
        if "text" in data and data["text"]:
            self.streaming_texts.append(data["text"])
            result = 1
        else:
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                if "text" in choice and choice["text"]:
                    self.streaming_texts.append(choice["text"])
                    result = 1
                else:
                    delta = choice.get("delta", {})
                    if isinstance(delta, dict) and delta.get("content"):
                        self.streaming_texts.append(delta["content"])
                        result = 1
        return result

    def compile_streaming(
        self,
        request: GenerationRequest,
        *,
        text_override: str | None = None,
    ) -> GenerationResponse:
        """
        Compile accumulated streaming chunks into a final GenerationResponse.

        :param request: Original generation request
        :param text_override: If provided, use as response text instead of
            joining streaming_texts (e.g. when backend supplies final text only).
        :return: GenerationResponse with text and metrics
        """
        text = (
            text_override
            if text_override is not None
            else "".join(self.streaming_texts)
        )
        input_metrics, output_metrics = self._extract_metrics(
            self.streaming_usage, text
        )
        request_args = self._get_request_args(request)

        return GenerationResponse(
            request_id=request.request_id,
            response_id=self.streaming_response_id,
            request_args=request_args,
            text=text or None,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def _extract_text_and_usage(
        self, response_dict: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Extract text and usage from response dict; infer shape from keys."""
        usage = response_dict.get("usage", {}) or {}

        if "text" in response_dict:
            text = response_dict["text"]
            return text if isinstance(text, str) else str(text), usage

        choices = response_dict.get("choices", [])
        if not choices:
            return "", usage

        choice = choices[0]
        if "text" in choice:
            text = choice["text"]
        elif "message" in choice:
            msg = choice["message"]
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        else:
            text = ""

        return text if isinstance(text, str) else str(text), usage

    def _extract_metrics(
        self, usage: dict[str, Any] | None, text: str
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """Build UsageMetrics from usage dict and text (word/char counts)."""
        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=len(text.split()) if text else 0,
                text_characters=len(text) if text else 0,
            )

        usage_metrics = cast("dict[str, int]", usage)
        input_details = cast(
            "dict[str, int]", usage.get("prompt_tokens_details", {}) or {}
        )
        output_details = cast(
            "dict[str, int]", usage.get("completion_tokens_details", {}) or {}
        )

        input_metrics = UsageMetrics(
            text_tokens=(
                input_details.get("prompt_tokens")
                or usage_metrics.get("prompt_tokens")
                or 0
            ),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        )
        output_metrics = UsageMetrics(
            text_tokens=(
                output_details.get("completion_tokens")
                or usage_metrics.get("completion_tokens")
                or 0
            ),
            text_words=len(text.split()) if text else 0,
            text_characters=len(text) if text else 0,
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )
        return input_metrics, output_metrics

    def _parse_line(self, line: str) -> dict[str, Any] | None:
        """Parse SSE line to JSON dict; return None if done or invalid."""
        if line == "data: [DONE]":
            return None
        line = (line or "").strip()
        if not line or not line.startswith("data:"):
            return {} if line else None
        line = line[len("data:") :].strip()
        try:
            return json.loads(line)
        except (ValueError, TypeError):
            return {}

    def _get_request_args(self, request: GenerationRequest) -> str | None:
        """Get request_args string from request.arguments if present."""
        arguments = getattr(request, "arguments", None)
        if arguments is None:
            return None
        if hasattr(arguments, "model_dump_json"):
            return arguments.model_dump_json()
        return None
