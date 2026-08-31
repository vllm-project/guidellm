"""
OrcaRouter HTTP backend implementation for GuideLLM.

Provides an HTTP backend for the OrcaRouter API, an OpenAI-compatible AI
gateway. Reuses the OpenAI-compatible request machinery while specializing
the endpoint validation and default model behavior for OrcaRouter.
"""

from __future__ import annotations

from typing import Any

from guidellm.backends.backend import Backend
from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.schemas.backends import OrcaRouterHTTPBackendArgs

__all__ = [
    "OrcaRouterHTTPBackend",
]

# Default model used when no model is explicitly configured. OrcaRouter routes
# each request to the best provider for the workload, so this is a safe default.
DEFAULT_ORCAROUTER_MODEL = "orcarouter/auto"


@Backend.register("orcarouter_http")
class OrcaRouterHTTPBackend(OpenAIHTTPBackend):
    """
    HTTP backend for the OrcaRouter API.

    OrcaRouter is an OpenAI-compatible AI gateway that exposes a provider/model
    namespace across many models, with adaptive routing and automatic failover.
    This backend mirrors the ``openai_http`` backend but specializes validation
    (OrcaRouter exposes no ``/health`` route) and defaults the model to
    ``orcarouter/auto`` when none is configured.

    Example:
    ::
        backend_args = OrcaRouterHTTPBackendArgs(
            api_key="sk-orca-...",
            model="orcarouter/auto",
        )
        backend = OrcaRouterHTTPBackend(backend_args)

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            process_response(response)
        await backend.process_shutdown()
    """

    _args: OrcaRouterHTTPBackendArgs

    def __init__(
        self,
        arguments: OrcaRouterHTTPBackendArgs,
    ):
        """
        Initialize OrcaRouter HTTP backend with server configuration.

        :param arguments: OrcaRouter backend arguments
        """
        super().__init__(arguments)

    async def validate(self):
        """
        Validate backend connectivity against the OrcaRouter API.

        Probes the ``/v1/models`` endpoint instead of ``/health`` because
        OrcaRouter does not expose a health route.

        :raises RuntimeError: If backend cannot connect or validate configuration
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if not self._args.validate_backend:
            return

        try:
            validate_kwargs: dict[str, Any] = {
                "method": "GET",
                "url": f"{self._args.target}/{self._args.api_routes['/v1/models']}",
            }
            existing_headers = validate_kwargs.get("headers")
            built_headers = self._build_headers(existing_headers)
            validate_kwargs["headers"] = built_headers
            response = await self._async_client.request(**validate_kwargs)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the "
                "OrcaRouter API or validate the backend configuration."
            ) from exc

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: The configured model, or ``orcarouter/auto`` when none is set
        """
        return self._args.model or DEFAULT_ORCAROUTER_MODEL
