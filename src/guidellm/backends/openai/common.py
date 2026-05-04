"""Shared helpers for OpenAI-compatible HTTP and WebSocket backends."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FALLBACK_TIMEOUT",
    "build_headers",
    "resolve_validate_kwargs",
]

# NOTE: This value is taken from httpx's default
FALLBACK_TIMEOUT = 5.0


def build_headers(
    api_key: str | None,
    existing_headers: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """
    Build headers with bearer authentication for OpenAI-compatible requests.

    Merges the Authorization bearer token (if ``api_key`` is set) with any
    existing headers. User-provided headers take precedence over the bearer token.

    :param api_key: Optional API key for Bearer authentication
    :param existing_headers: Optional headers to merge in
    :return: Headers dict, or ``None`` if there are no headers to send
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if existing_headers:
        headers = {**headers, **existing_headers}
    return headers or None


def resolve_validate_kwargs(
    validate_backend: bool | str | dict[str, Any],
    target: str,
    api_routes: dict[str, str],
) -> dict[str, Any] | None:
    """
    Normalize ``validate_backend`` into kwargs for ``httpx`` request().

    Accepts the same shapes as
    :class:`~guidellm.backends.openai.http.OpenAIHTTPBackend`.
    """
    raw = validate_backend
    if not raw:
        return None

    if raw is True:
        raw = "/health"

    if isinstance(raw, str):
        url = f"{target}/{api_routes[raw]}" if raw in api_routes else raw
        request_kwargs: dict[str, Any] = {"method": "GET", "url": url}
    elif isinstance(raw, dict):
        request_kwargs = raw
    else:
        request_kwargs = raw

    if not isinstance(request_kwargs, dict) or "url" not in request_kwargs:
        raise ValueError(
            "validate_backend must be a boolean, string, or dictionary and contain "
            f"a target URL. Got: {request_kwargs}"
        )

    if "method" not in request_kwargs:
        request_kwargs["method"] = "GET"

    return request_kwargs
