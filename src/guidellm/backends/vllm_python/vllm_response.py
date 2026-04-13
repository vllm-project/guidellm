"""
VLLM-specific response handler for building GenerationResponse from vLLM output.

Builds GenerationResponse from text and usage dicts extracted directly from
vLLM's RequestOutput — no OpenAI-format intermediary.
"""

from __future__ import annotations

from typing import Any, cast

from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics

__all__ = ["VLLMResponseHandler"]


class VLLMResponseHandler:
    """
    Stateless response builder for the vLLM Python backend.

    Converts (text, usage dict) pairs into GenerationResponse with proper
    UsageMetrics.  All methods are static; no per-request accumulation state.
    """

    @staticmethod
    def build_response(
        request: GenerationRequest,
        text: str,
        usage: dict[str, int] | None,
        response_id: str | None = None,
    ) -> GenerationResponse:
        """
        Build a GenerationResponse from text and usage extracted from vLLM output.

        :param request: Original generation request (provides request_id)
        :param text: Generated text
        :param usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
        :param response_id: Optional response/request ID from vLLM
        :return: GenerationResponse with text and metrics
        """
        input_metrics, output_metrics = VLLMResponseHandler._extract_metrics(
            usage, text
        )
        return GenerationResponse(
            request_id=request.request_id,
            response_id=response_id,
            request_args=None,
            text=text or None,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    @staticmethod
    def _extract_metrics(
        usage: dict[str, Any] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """Build UsageMetrics from usage dict and text (word/char counts).

        text=None means text is not applicable (metrics will be None);
        text="" means text was applicable but empty (metrics will be 0).
        """
        if text is None:
            # text not applicable (e.g. tool-call-only) — exclude from aggregation
            text_words = None
            text_chars = None
        else:
            text_words = len(text.split())
            text_chars = len(text)

        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=text_words,
                text_characters=text_chars,
            )

        usage_metrics = cast("dict[str, int]", usage)
        input_details = cast(
            "dict[str, int]", usage.get("prompt_tokens_details", {}) or {}
        )
        output_details = cast(
            "dict[str, int]", usage.get("completion_tokens_details", {}) or {}
        )

        input_metrics = UsageMetrics(
            text_tokens=usage_metrics.get("prompt_tokens", 0),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        )
        output_metrics = UsageMetrics(
            text_tokens=usage_metrics.get("completion_tokens", 0),
            text_words=text_words,
            text_characters=text_chars,
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )
        return input_metrics, output_metrics
