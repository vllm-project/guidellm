"""Capture structured vLLM configuration outside the request execution process."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx2

from guidellm.collectors.collector import Collector, OneshotCollector
from guidellm.logger import logger
from guidellm.schemas.benchmark import CollectorArgs, VLLMServerInfoCollectorArgs

__all__ = ["VLLMServerInfoCollector"]

_SERVER_CONFIG_TIMEOUT = 5.0
_SERVER_CONFIG_MAX_BYTES = 1024 * 1024
_CONFIG_ASSIGNMENT_RE = re.compile(r"(?<![\w-])(?P<key>[A-Za-z_][\w.-]*)\s*=(?!=)")
_CONFIG_FLAG_RE = re.compile(r"(?<!\S)--(?P<key>[A-Za-z_][\w-]*)(?=\s|=|$)")
_CONFIG_HEADER_RE = re.compile(r"(?<![\w-])(?:proxy-)?authorization\s*:", re.IGNORECASE)
_CONFIG_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+")
_LOCAL_CONFIG_HOSTS = ("localhost", "127.0.0.1", "::1", "", None)


def _sensitive_config_key(key: str) -> bool:
    """Recognize credential names without treating every key or token as secret."""
    normalized = key.lower().replace("-", "_").replace(".", "_")
    names = (
        "token",
        "pass",
        "passwd",
        "credential",
        "credentials",
        "cookie",
        "cookies",
        "bearer",
        "pat",
        "auth",
        "authorization",
        "passphrase",
        "pwd",
        "ssh_key",
        "encryption_key",
        "signing_key",
        "license_key",
        "client_key",
        "sig",
        "signature",
    )
    return (
        normalized in names
        or normalized.endswith(tuple(f"_{name}" for name in names))
        or any(
            part in normalized
            for part in (
                "api_key",
                "apikey",
                "password",
                "secret",
                "access_key",
                "private_key",
            )
        )
    )


def _redact_config_text(value: str, redact_hosts: bool = True) -> str:
    """Drop sensitive free-form assignments/flags and strip URL user information."""
    if _CONFIG_HEADER_RE.search(value):
        return "[REDACTED]"
    for pattern in (_CONFIG_ASSIGNMENT_RE, _CONFIG_FLAG_RE):
        if any(
            _sensitive_config_key(match["key"]) for match in pattern.finditer(value)
        ):
            # Whole-string omission avoids guessing where quoted, multiline or
            # escaped credentials end. Other fields in the section are retained.
            return "[REDACTED]"

    for match in _CONFIG_URL_RE.finditer(value):
        _, remainder = match[0].split("://", 1)
        authority = re.split(r"[/?#]", remainder, maxsplit=1)[0]
        if ":" in authority and "@" in remainder[len(authority) :]:
            # An unescaped delimiter may split a password from its userinfo.
            # Omit the string rather than guess where the credentials end.
            return "[REDACTED]"

    return _CONFIG_URL_RE.sub(lambda match: _redact_url(match, redact_hosts), value)


def _redact_url(match: re.Match[str], redact_hosts: bool) -> str:
    """Filter user information and optionally hostnames from a matched URL."""
    scheme, remainder = match[0].split("://", 1)
    authority = re.split(r"[/?#]", remainder, maxsplit=1)[0]
    suffix = remainder[len(authority) :]
    if authority == "[REDACTED]":
        return match[0]
    try:
        host = urlsplit(f"{scheme}://{authority.rsplit('@', 1)[-1]}").hostname
    except ValueError:
        return "[REDACTED]"
    if redact_hosts and host not in _LOCAL_CONFIG_HOSTS:
        authority = "[REDACTED]"
    elif "@" in authority:
        authority = f"[REDACTED]@{authority.rsplit('@', 1)[1]}"
    return f"{scheme}://{authority}{suffix}"


def _redact_server_config(value: Any, redact_hosts: bool = True) -> Any:
    """Filter recognizable credentials and host fields without mutating input."""
    if isinstance(value, dict):
        result = {}
        name = value.get("name")
        sensitive_record = isinstance(name, str) and _sensitive_config_key(name)
        for key, item in value.items():
            normalized = key.lower().replace("-", "_") if isinstance(key, str) else ""
            if _sensitive_config_key(normalized) or (
                key == "value" and sensitive_record
            ):
                # Preserve unset values and boolean switches: neither is a secret.
                result[key] = (
                    item if item is None or isinstance(item, bool) else "[REDACTED]"
                )
            elif (
                redact_hosts
                and (
                    normalized in ("host", "hostname", "node_ip", "master_addr")
                    or normalized.endswith("_host")
                )
                and item not in _LOCAL_CONFIG_HOSTS
            ):
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact_server_config(item, redact_hosts)
        return result
    if isinstance(value, list):
        items = []
        redact_next = False
        for item in value:
            if redact_next:
                items.append("[REDACTED]")
                redact_next = False
            elif (
                isinstance(item, str)
                and (flag := _CONFIG_FLAG_RE.fullmatch(item))
                and _sensitive_config_key(flag["key"])
            ):
                items.append(item)
                redact_next = True
            else:
                items.append(_redact_server_config(item, redact_hosts))
        return items
    if isinstance(value, str):
        return _redact_config_text(value, redact_hosts)
    return value


@Collector.register("vllm_server_info")
class VLLMServerInfoCollector(OneshotCollector):
    """Read a server's structured configuration once after each benchmark."""

    def __init__(
        self,
        url: str,
        redact_hosts: bool = True,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        """
        :param url: Explicit server-info endpoint URL
        :param redact_hosts: Whether collected host fields should be redacted
        :param transport: Optional transport for substituting the HTTP server
        """
        super().__init__("vllm_server_info")
        self.url = url
        self.redact_hosts = redact_hosts
        self._transport = transport

    @classmethod
    def from_args(cls, args: CollectorArgs) -> VLLMServerInfoCollector:
        """
        Create the collector from its registered arguments.

        :param args: Server-info collector arguments
        :return: Configured collector
        :raises TypeError: If passed a different collector's arguments
        """
        if not isinstance(args, VLLMServerInfoCollectorArgs):
            raise TypeError("Expected VLLMServerInfoCollectorArgs")
        return cls(url=args.url, redact_hosts=args.redact_hosts)

    async def collect(self) -> dict[str, Any]:
        """
        Capture configuration, or report a failure without interrupting the run.

        :return: Filtered source, capture timestamp and configuration or error
        """
        result: dict[str, Any] = {
            "source": _redact_config_text(self.url, self.redact_hosts),
            "collected_at": time.time(),
        }
        try:
            result["vllm_config"] = await asyncio.wait_for(
                self._fetch_config(), timeout=_SERVER_CONFIG_TIMEOUT
            )
        except (
            TimeoutError,
            httpx2.HTTPError,
            httpx2.InvalidURL,
            ValueError,
            RecursionError,
        ) as exc:
            # Exception messages may include credentials, URLs or response bodies.
            result["error"] = type(exc).__name__
            logger.warning(
                "Server configuration was not recorded ({})", type(exc).__name__
            )
        return result

    async def _fetch_config(self) -> dict[str, Any]:
        async with (
            httpx2.AsyncClient(
                transport=self._transport,
                timeout=_SERVER_CONFIG_TIMEOUT,
                follow_redirects=False,
            ) as client,
            client.stream(
                "GET", self.url, params={"config_format": "json"}
            ) as response,
        ):
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > _SERVER_CONFIG_MAX_BYTES:
                    raise ValueError("Server configuration exceeds the size limit")
                body.extend(chunk)
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("vllm_config"), dict
        ):
            raise ValueError("Missing or unsupported structured vllm_config")
        return _redact_server_config(payload["vllm_config"], self.redact_hosts)
