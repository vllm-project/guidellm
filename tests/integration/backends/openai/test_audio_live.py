"""Opt-in checks against a live OpenAI-compatible audio server."""

from __future__ import annotations

import json
import os
from typing import Literal

import pytest

from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)
from guidellm.schemas.backends import OpenAIHTTPBackendArgs


def _live_audio_backend(
    request_format: Literal["/v1/audio/speech", "/v1/chat/completions"],
    body_env: str,
    default_body: dict[str, object],
) -> OpenAIHTTPBackend:
    """Create a live backend from documented environment configuration.

    :param request_format: OpenAI-compatible audio endpoint path.
    :param body_env: Environment variable containing optional JSON body fields.
    :param default_body: Body fields used when no endpoint override is set.
    :return: Backend configured for a manual live audio check.
    """
    target = os.environ.get("GUIDELLM_LIVE_AUDIO_TARGET")
    model = os.environ.get("GUIDELLM_LIVE_AUDIO_MODEL")
    if not target or not model:
        pytest.fail(
            "Set GUIDELLM_LIVE_AUDIO_TARGET and GUIDELLM_LIVE_AUDIO_MODEL "
            "when enabling live audio tests."
        )

    body_json = os.environ.get(body_env)
    body = json.loads(body_json) if body_json else default_body
    args = OpenAIHTTPBackendArgs(
        target=target,
        model=model,
        request_format=request_format,
        validate_backend=False,
        api_key=os.environ.get("GUIDELLM_LIVE_AUDIO_API_KEY"),
        extras=GenerationRequestArguments(body=body),
    )
    return OpenAIHTTPBackend(args)


async def _resolve_audio_response(
    backend: OpenAIHTTPBackend,
) -> GenerationResponse:
    """Resolve one request and return its final response and request timings.

    :param backend: Configured live audio backend.
    :return: Final generated-audio response.
    :raises AssertionError: If the backend produces no final response.
    """
    request = GenerationRequest(columns={"text_column": ["Say hello clearly."]})
    info = RequestInfo(request_id=request.request_id, status="pending")
    await backend.process_startup()
    try:
        responses = [
            response
            async for response, _ in backend.resolve(request, info)
            if response is not None
        ]
    finally:
        await backend.process_shutdown()

    assert responses, "Live audio endpoint returned no final response."
    return responses[-1]


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("GUIDELLM_RUN_LIVE_AUDIO_SPEECH_TESTS") != "1",
    reason="Set GUIDELLM_RUN_LIVE_AUDIO_SPEECH_TESTS=1 for manual server checks.",
)
async def test_live_speech_endpoint_returns_audio():
    """Resolve generated audio through an OpenAI-compatible speech endpoint.

    :return: None.
    ## WRITTEN BY AI ##
    """
    backend = _live_audio_backend(
        "/v1/audio/speech",
        "GUIDELLM_LIVE_AUDIO_SPEECH_BODY",
        {"response_format": "wav", "stream_format": "audio"},
    )

    response = await _resolve_audio_response(backend)

    assert response.output_media is not None
    assert any(
        item.get("modality") == "audio" and item.get("data")
        for item in response.output_media
    )


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("GUIDELLM_RUN_LIVE_AUDIO_CHAT_TESTS") != "1",
    reason="Set GUIDELLM_RUN_LIVE_AUDIO_CHAT_TESTS=1 for manual server checks.",
)
async def test_live_chat_endpoint_returns_audio():
    """Resolve generated audio through an OpenAI-compatible chat endpoint.

    :return: None.
    ## WRITTEN BY AI ##
    """
    backend = _live_audio_backend(
        "/v1/chat/completions",
        "GUIDELLM_LIVE_AUDIO_CHAT_BODY",
        {
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )

    response = await _resolve_audio_response(backend)

    assert response.output_media is not None
    assert any(
        item.get("modality") == "audio" and item.get("data")
        for item in response.output_media
    )
