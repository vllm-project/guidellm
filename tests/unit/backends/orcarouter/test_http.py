"""
Unit tests for OrcaRouterHTTPBackend implementation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from guidellm.backends.backend import Backend
from guidellm.backends.orcarouter.http import (
    DEFAULT_ORCAROUTER_MODEL,
    OrcaRouterHTTPBackend,
)
from guidellm.schemas.backends import BackendArgs, OrcaRouterHTTPBackendArgs
from tests.unit.testing_utils import async_timeout


def _make_backend(**kwargs) -> OrcaRouterHTTPBackend:
    """Create an OrcaRouterHTTPBackend from keyword arguments via BackendArgs."""
    args = OrcaRouterHTTPBackendArgs(**kwargs)
    return OrcaRouterHTTPBackend(args)


class TestOrcaRouterHTTPBackend:
    """Test cases for OrcaRouterHTTPBackend."""

    @pytest.fixture
    def valid_instances(self):
        """Fixture providing valid OrcaRouterHTTPBackend instances."""
        constructor_args = {"api_key": "sk-test"}
        instance = _make_backend(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """OrcaRouterHTTPBackend inherits Backend and exposes HTTP methods.

        ### WRITTEN BY AI ###
        """
        assert issubclass(OrcaRouterHTTPBackend, Backend)
        assert hasattr(OrcaRouterHTTPBackend, "process_startup")
        assert hasattr(OrcaRouterHTTPBackend, "process_shutdown")
        assert hasattr(OrcaRouterHTTPBackend, "validate")
        assert hasattr(OrcaRouterHTTPBackend, "resolve")
        assert hasattr(OrcaRouterHTTPBackend, "default_model")
        assert hasattr(OrcaRouterHTTPBackend, "available_models")

    @pytest.mark.smoke
    def test_registered(self):
        """orcarouter_http backend is registered and constructible.

        ### WRITTEN BY AI ###
        """
        assert Backend.is_registered("orcarouter_http")
        args = OrcaRouterHTTPBackendArgs()
        backend = Backend.create(args)
        assert isinstance(backend, OrcaRouterHTTPBackend)
        assert backend.kind == "orcarouter_http"

    @pytest.mark.smoke
    def test_backend_args_registered(self):
        """orcarouter_http args are registered with default target.

        ### WRITTEN BY AI ###
        """
        assert BackendArgs.is_registered("orcarouter_http")
        args = OrcaRouterHTTPBackendArgs()
        assert args.kind == "orcarouter_http"
        assert args.target == "https://api.orcarouter.ai"

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """OrcaRouterHTTPBackend stores configuration on args.

        ### WRITTEN BY AI ###
        """
        instance, _ = valid_instances
        assert isinstance(instance, OrcaRouterHTTPBackend)
        assert instance.kind == "orcarouter_http"
        assert instance._args.target == "https://api.orcarouter.ai"

    @pytest.mark.smoke
    def test_target_stripped_of_v1(self):
        """target values ending in /v1 are normalized.

        ### WRITTEN BY AI ###
        """
        backend = _make_backend(target="https://api.orcarouter.ai/v1")
        assert backend._args.target == "https://api.orcarouter.ai"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_default_model(self):
        """default_model returns configured model or orcarouter/auto.

        ### WRITTEN BY AI ###
        """
        backend1 = _make_backend(model="anthropic/claude-sonnet-5")
        assert await backend1.default_model() == "anthropic/claude-sonnet-5"

        backend2 = _make_backend()
        assert await backend2.default_model() == DEFAULT_ORCAROUTER_MODEL

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_probes_models_endpoint(self, httpx_mock: HTTPXMock):
        """validate probes /v1/models (not /health) on OrcaRouter.

        ### WRITTEN BY AI ###
        """
        httpx_mock.add_response(
            url="https://api.orcarouter.ai/v1/models",
            method="GET",
            json={"data": [{"id": "orcarouter/auto"}]},
        )

        backend = _make_backend()
        await backend.process_startup()
        await backend.validate()  # Should not raise

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_disabled(self):
        """validate is a no-op when validate_backend is False.

        ### WRITTEN BY AI ###
        """
        backend = _make_backend(validate_backend=False)
        await backend.process_startup()
        await backend.validate()  # Should not raise

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_not_in_process(self):
        """validate raises when backend is not started.

        ### WRITTEN BY AI ###
        """
        backend = _make_backend()
        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_failure(self):
        """validate raises RuntimeError on failed validation request.

        ### WRITTEN BY AI ###
        """
        backend = _make_backend()
        await backend.process_startup()

        def mock_fail(*args, **kwargs):
            raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

        with (
            patch.object(backend._async_client, "request", side_effect=mock_fail),
            pytest.raises(RuntimeError, match="Backend validation request failed"),
        ):
            await backend.validate()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_available_models(self, httpx_mock: HTTPXMock):
        """available_models lists models from the OrcaRouter API.

        ### WRITTEN BY AI ###
        """
        httpx_mock.add_response(
            url="https://api.orcarouter.ai/v1/models",
            json={"data": [{"id": "orcarouter/auto"}, {"id": "orcarouter/fusion"}]},
        )

        backend = _make_backend()
        await backend.process_startup()
        models = await backend.available_models()
        assert models == ["orcarouter/auto", "orcarouter/fusion"]
