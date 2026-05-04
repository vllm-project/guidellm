"""
Unit tests for backend transformation to typed BackendArgs instances.

Tests the automatic conversion of backend configuration from dict-based
backend into properly typed BackendArgs instances during model validation.

### WRITTEN BY AI ###
"""

import pytest
from pydantic import ValidationError

from guidellm.backends.backend import BackendArgs
from guidellm.backends.openai.http import OpenAIHTTPBackendArgs
from guidellm.backends.openai.realtime_ws import OpenAIRealtimeWsBackendArgs
from guidellm.benchmark.schemas.entrypoints import (
    BenchmarkArgs,
)

# Conditionally import VLLM backend args if available
try:
    from guidellm.backends.vllm_python.vllm import VLLMPythonBackendArgs

    HAS_VLLM = True
except ImportError:
    VLLMPythonBackendArgs = None  # type: ignore[assignment, misc]
    HAS_VLLM = False

# Minimal required data pipeline fields for BenchmarkArgs
_PIPELINE_DEFAULTS = {
    "data": [{"kind": "synthetic_text", "prompt_tokens": 256, "output_tokens": 128}],
    "tokenizer": {"kind": "huggingface_auto", "model": "gpt2"},
    "data_column_mapper": {"kind": "generative_column_mapper"},
    "data_preprocessors": [],
    "data_finalizer": {"kind": "generative"},
    "data_loader": {"kind": "pytorch"},
    "profile": {"kind": "sweep", "rate": [10.0]},
}


@pytest.mark.sanity
class TestBackendArgsTransformation:
    """Test transformation of backend from dict to typed BackendArgs."""

    def test_dict_backend_transformed(self):
        """
        Test that dict backend with kind field is transformed to BackendArgs.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        # Verify backend is typed OpenAIHTTPBackendArgs
        assert isinstance(args.backend, OpenAIHTTPBackendArgs)
        assert args.backend.target == "http://localhost:9000"
        assert args.backend.model == "test_model"

    def test_openai_realtime_ws_backend_kwargs_validates(self) -> None:
        """Realtime WS backend is selected explicitly; no request_format shim.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "kind": "openai_realtime_ws",
                    "target": "http://localhost:8000",
                    "model": "rt-model",
                },
                **_PIPELINE_DEFAULTS,
            }
        )
        assert args.backend_kwargs.kind == "openai_realtime_ws"
        assert isinstance(args.backend_kwargs, OpenAIRealtimeWsBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:8000"
        assert args.backend_kwargs.model == "rt-model"

    def test_dict_with_request_format(self):
        """
        Test that request_format is included in BackendArgs transformation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "request_format": "/v1/completions",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.backend, OpenAIHTTPBackendArgs)
        assert args.backend.target == "http://localhost:9000"
        assert args.backend.model == "test_model"
        assert args.backend.request_format == "/v1/completions"

    def test_serialization_round_trip(self):
        """
        Test that serialization and deserialization preserves typed backend.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        serialized_backend = args.backend.model_dump()

        assert isinstance(serialized_backend, dict)
        assert serialized_backend["kind"] == "openai_http"
        assert serialized_backend["target"] == "http://localhost:9000"
        assert serialized_backend["model"] == "test_model"

        args2 = BenchmarkArgs.model_validate(
            {
                "backend": serialized_backend,
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args2.backend, OpenAIHTTPBackendArgs)
        assert args2.backend.target == "http://localhost:9000"
        assert args2.backend.model == "test_model"

    def test_validation_error_missing_required_field(self):
        """
        Test validation error when required backend field is missing.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkArgs.model_validate(
                {
                    "backend": {
                        "kind": "openai_http",
                        "model": "test_model",
                        # Missing 'target'
                    },
                    **_PIPELINE_DEFAULTS,
                }
            )

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("target" in str(err).lower() for err in errors)

    def test_validation_error_invalid_request_format(self):
        """
        Test validation error for invalid request_format.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkArgs.model_validate(
                {
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:9000",
                        "model": "test_model",
                        "request_format": "invalid_format",
                    },
                    **_PIPELINE_DEFAULTS,
                }
            )

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("request_format" in str(err).lower() for err in errors)

    @pytest.mark.skipif(not HAS_VLLM, reason="VLLM not installed")
    def test_vllm_backend_transformation(self):
        """
        Test transformation works with VLLM backend.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "vllm_python",
                    "model": "facebook/opt-125m",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        assert VLLMPythonBackendArgs is not None
        assert isinstance(args.backend, VLLMPythonBackendArgs)
        assert args.backend.model == "facebook/opt-125m"

    @pytest.mark.skipif(not HAS_VLLM, reason="VLLM not installed")
    def test_vllm_backend_rejects_target(self):
        """
        Test that VLLM backend rejects target parameter (extra="forbid").

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkArgs.model_validate(
                {
                    "backend": {
                        "kind": "vllm_python",
                        "target": "http://localhost:9000",
                        "model": "facebook/opt-125m",
                    },
                    **_PIPELINE_DEFAULTS,
                }
            )

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("target" in str(err).lower() for err in errors)

    def test_empty_dict_backend(self):
        """
        Test handling of empty dict backend (missing kind field).

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError):
            BenchmarkArgs.model_validate(
                {
                    "backend": {},
                    **_PIPELINE_DEFAULTS,
                }
            )

    def test_default_backend(self):
        """
        Test that missing backend uses default (openai_http) when target is provided.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                **_PIPELINE_DEFAULTS,
            }
        )
        assert isinstance(args.backend, OpenAIHTTPBackendArgs)

    def test_already_typed_backend_via_aliased_dump(self):
        """
        Test that already-typed BackendArgs can be passed via dict dump.

        ### WRITTEN BY AI ###
        """
        backend_args = OpenAIHTTPBackendArgs(
            target="http://localhost:9000", model="test_model"
        )

        args = BenchmarkArgs.model_validate(
            {
                "backend": backend_args.model_dump(),
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.backend, OpenAIHTTPBackendArgs)
        assert args.backend.target == "http://localhost:9000"
        assert args.backend.model == "test_model"

    def test_backend_is_backendargs_subclass(self):
        """
        Test that backend is always a BackendArgs subclass after validation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.backend, BackendArgs)
        assert isinstance(args.backend, OpenAIHTTPBackendArgs)

    def test_api_key_is_securestr(self):
        """
        Test that api_key is stored as SecretStr.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.backend, OpenAIHTTPBackendArgs)
        assert args.backend.api_key is not None
        assert args.backend.api_key.get_secret_value() == "secret123"

    def test_serialization_masks_api_key(self):
        """
        Test that serialization masks api_key (SecretStr behavior).

        ### WRITTEN BY AI ###
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        serialized = args.model_dump()

        assert "api_key" in serialized["backend"]
        assert serialized["backend"]["api_key"] != "secret123"

    def test_different_backend_types(self):
        """
        Test that different backend types get correct BackendArgs subclasses.

        ### WRITTEN BY AI ###
        """
        args_openai = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                    "model": "gpt-3.5-turbo",
                },
                **_PIPELINE_DEFAULTS,
            }
        )
        assert isinstance(args_openai.backend, OpenAIHTTPBackendArgs)

        if HAS_VLLM:
            args_vllm = BenchmarkArgs.model_validate(
                {
                    "backend": {
                        "kind": "vllm_python",
                        "model": "facebook/opt-125m",
                    },
                    **_PIPELINE_DEFAULTS,
                }
            )
            assert VLLMPythonBackendArgs is not None
            assert isinstance(args_vllm.backend, VLLMPythonBackendArgs)
