"""Example external GuideLLM backend plugin used in tests."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Literal

import httpx2
from pydantic import Field

from guidellm.backends import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    UsageMetrics,
)
from guidellm.schemas.backends import BackendArgs

__all__ = ["EchoPluginBackend", "EchoPluginBackendArgs"]


@BackendArgs.register("echo_plugin")
class EchoPluginBackendArgs(BackendArgs):
    """Configuration schema for the example ``echo_plugin`` backend."""

    kind: Literal["echo_plugin"] = Field(  # type: ignore[assignment]
        default="echo_plugin",
        description="Type identifier for the example plugin backend.",
    )
    target: str = Field(
        description="Base URL of an OpenAI-compatible HTTP server.",
    )
    model: str = Field(
        default="",
        description="Optional model name; defaults to first server model.",
    )
    timeout: float = Field(
        default=10.0,
        description="HTTP client timeout in seconds.",
    )


@Backend.register("echo_plugin")
class EchoPluginBackend(Backend):
    """Minimal HTTP backend plugin targeting GuideLLM's mock or OpenAI server."""

    _args: EchoPluginBackendArgs

    def __init__(self, arguments: EchoPluginBackendArgs) -> None:
        super().__init__(arguments)
        self._client: httpx2.AsyncClient | None = None

    async def process_startup(self) -> None:
        """Create the async HTTP client for this worker process."""
        self._client = httpx2.AsyncClient(timeout=self._args.timeout)

    async def process_shutdown(self) -> None:
        """Close the async HTTP client for this worker process."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def validate(self) -> None:
        """Check that the target server health endpoint responds."""
        if self._client is None:
            raise RuntimeError("EchoPluginBackend has not been started.")
        response = await self._client.get(f"{self._args.target.rstrip('/')}/health")
        response.raise_for_status()

    async def default_model(self) -> str:
        """Return configured model or query ``/v1/models`` on the target server."""
        if self._args.model:
            return self._args.model
        if self._client is None:
            raise RuntimeError("EchoPluginBackend has not been started.")
        response = await self._client.get(f"{self._args.target.rstrip('/')}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        self._args.model = models[0]["id"] if models else ""
        return self._args.model

    async def resolve(  # type: ignore[override, misc]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse | None]]
        | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """Send a non-streaming completion request to the target server."""
        if self._client is None:
            raise RuntimeError("EchoPluginBackend has not been started.")

        prompts: list[str] = []
        if history:
            for prev_req, prev_res in history:
                prompts.extend(prev_req.columns.get("prefix_column", []))
                prompts.extend(prev_req.columns.get("text_column", []))
                if prev_res and prev_res.text:
                    prompts.append(prev_res.text)
        prompts.extend(request.columns.get("prefix_column", []))
        prompts.extend(request.columns.get("text_column", []))

        body = {
            "model": await self.default_model(),
            "prompt": " ".join(prompts) or "hello",
            "max_tokens": request.output_metrics.text_tokens or 8,
            "stream": False,
        }
        request_info.timings.request_start = time.time()
        response = await self._client.post(
            f"{self._args.target.rstrip('/')}/v1/completions",
            json=body,
        )
        request_info.timings.request_end = time.time()
        response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        usage = payload.get("usage", {})
        yield (
            GenerationResponse(
                request_id=request.request_id,
                request_args=json.dumps(body),
                text=text,
                input_metrics=UsageMetrics(text_tokens=usage.get("prompt_tokens")),
                output_metrics=UsageMetrics(text_tokens=usage.get("completion_tokens")),
            ),
            request_info,
        )
