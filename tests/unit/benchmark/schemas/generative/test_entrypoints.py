"""
Unit tests for backend_kwargs transformation to typed BackendArgs instances.

Tests the automatic conversion of backend configuration from dict-based
backend_kwargs into properly typed BackendArgs instances during model validation.

### WRITTEN BY AI ###
"""

import pytest
from pydantic import ValidationError

from guidellm.backends.backend import BackendArgs
from guidellm.backends.openai.http import OpenAIHTTPBackendArgs
from guidellm.benchmark.schemas.generative.entrypoints import (
    BenchmarkGenerativeTextArgs,
)

# Conditionally import VLLM backend args if available
try:
    from guidellm.backends.vllm_python.vllm import VLLMPythonBackendArgs

    HAS_VLLM = True
except ImportError:
    VLLMPythonBackendArgs = None  # type: ignore[assignment, misc]
    HAS_VLLM = False


@pytest.mark.sanity
class TestBackendArgsTransformation:
    """Test transformation of backend_kwargs from dict to typed BackendArgs."""

    def test_dict_backend_kwargs_transformed(self):
        """
        Test that dict backend_kwargs with type field is transformed to BackendArgs.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Verify backend_kwargs is typed OpenAIHTTPBackendArgs
        assert isinstance(args.backend_kwargs, OpenAIHTTPBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"

    def test_dict_with_request_format(self):
        """
        Test that request_format is included in BackendArgs transformation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "request_format": "/v1/completions",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        assert isinstance(args.backend_kwargs, OpenAIHTTPBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"
        assert args.backend_kwargs.request_format == "/v1/completions"

    def test_serialization_round_trip(self):
        """
        Test that serialization and deserialization preserves typed backend_kwargs.

        The round-trip requires by_alias=True so the 'type' discriminator field
        is serialized with its alias name rather than the Python field name 'type_'.

        ### WRITTEN BY AI ###
        """
        # Create instance with dict backend_kwargs
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Serialize backend_kwargs with by_alias=True so type discriminator is preserved
        serialized_kwargs = args.backend_kwargs.model_dump(by_alias=True)

        # Should serialize backend_kwargs as dict with type key
        assert isinstance(serialized_kwargs, dict)
        assert serialized_kwargs["type"] == "openai_http"
        assert serialized_kwargs["target"] == "http://localhost:9000"
        assert serialized_kwargs["model"] == "test_model"

        # Deserialize back using the aliased dict
        args2 = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": serialized_kwargs,
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Should reconstruct typed instance
        assert isinstance(args2.backend_kwargs, OpenAIHTTPBackendArgs)
        assert args2.backend_kwargs.target == "http://localhost:9000"
        assert args2.backend_kwargs.model == "test_model"

    def test_validation_error_missing_required_field(self):
        """
        Test validation error when required backend field is missing.

        ### WRITTEN BY AI ###
        """
        # OpenAI HTTP backend requires 'target'
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend_kwargs": {
                        "type": "openai_http",
                        "model": "test_model",
                        # Missing 'target'
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

        # Should have validation error for target
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("target" in str(err).lower() for err in errors)

    def test_validation_error_invalid_request_format(self):
        """
        Test validation error for invalid request_format.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend_kwargs": {
                        "type": "openai_http",
                        "target": "http://localhost:9000",
                        "model": "test_model",
                        "request_format": "invalid_format",
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

        # Should have validation error for request_format
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("request_format" in str(err).lower() for err in errors)

    @pytest.mark.skipif(not HAS_VLLM, reason="VLLM not installed")
    def test_vllm_backend_transformation(self):
        """
        Test transformation works with VLLM backend.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "vllm_python",
                    "model": "facebook/opt-125m",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Verify backend_kwargs is typed VLLMPythonBackendArgs
        assert VLLMPythonBackendArgs is not None
        assert isinstance(args.backend_kwargs, VLLMPythonBackendArgs)
        assert args.backend_kwargs.model == "facebook/opt-125m"

    @pytest.mark.skipif(not HAS_VLLM, reason="VLLM not installed")
    def test_vllm_backend_rejects_target(self):
        """
        Test that VLLM backend rejects target parameter (extra="forbid").

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend_kwargs": {
                        "type": "vllm_python",
                        "target": "http://localhost:9000",  # Not a field in VLLM args
                        "model": "facebook/opt-125m",
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

        # Should have validation error about target not being a valid field
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("target" in str(err).lower() for err in errors)

    def test_empty_dict_backend_kwargs(self):
        """
        Test handling of empty dict backend_kwargs (missing type field).

        ### WRITTEN BY AI ###
        """
        # Empty dict without 'type' should fail validation
        with pytest.raises(ValidationError):
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend_kwargs": {},
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

    def test_default_backend_kwargs(self):
        """
        Test that missing backend_kwargs fails validation (required field).

        ### WRITTEN BY AI ###
        """
        # backend_kwargs is required with no default
        with pytest.raises(ValidationError):
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    # No backend_kwargs provided
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

    def test_already_typed_backend_kwargs_via_aliased_dump(self):
        """
        Test that already-typed BackendArgs can be passed via aliased dict dump.

        Direct instance passing fails because Pydantic's discriminator looks for
        a 'type' attribute but the field is named 'type_'. Use model_dump(by_alias=True)
        to produce the correctly keyed dict for round-trip validation.

        ### WRITTEN BY AI ###
        """
        # Create a typed BackendArgs instance and dump with alias
        backend_args = OpenAIHTTPBackendArgs(
            target="http://localhost:9000", model="test_model"
        )

        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": backend_args.model_dump(by_alias=True),
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Should produce a typed instance
        assert isinstance(args.backend_kwargs, OpenAIHTTPBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"

    def test_backend_kwargs_is_backendargs_subclass(self):
        """
        Test that backend_kwargs is always a BackendArgs subclass after validation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Should be a BackendArgs subclass
        assert isinstance(args.backend_kwargs, BackendArgs)
        assert isinstance(args.backend_kwargs, OpenAIHTTPBackendArgs)

    def test_api_key_is_securestr(self):
        """
        Test that api_key is stored as SecretStr.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        assert isinstance(args.backend_kwargs, OpenAIHTTPBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        # api_key is SecretStr — access via get_secret_value()
        assert args.backend_kwargs.api_key is not None
        assert args.backend_kwargs.api_key.get_secret_value() == "secret123"

    def test_serialization_masks_api_key(self):
        """
        Test that serialization masks api_key (SecretStr behavior).

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        serialized = args.model_dump()

        # api_key key should be present in serialized output
        assert "api_key" in serialized["backend_kwargs"]
        # SecretStr serializes as "**********" by default
        assert serialized["backend_kwargs"]["api_key"] != "secret123"

    def test_different_backend_types(self):
        """
        Test that different backend types get correct BackendArgs subclasses.

        ### WRITTEN BY AI ###
        """
        # OpenAI HTTP backend
        args_openai = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend_kwargs": {
                    "type": "openai_http",
                    "target": "http://localhost:8000",
                    "model": "gpt-3.5-turbo",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )
        assert isinstance(args_openai.backend_kwargs, OpenAIHTTPBackendArgs)

        # VLLM Python backend (if available)
        if HAS_VLLM:
            args_vllm = BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend_kwargs": {
                        "type": "vllm_python",
                        "model": "facebook/opt-125m",
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )
            assert VLLMPythonBackendArgs is not None
            assert isinstance(args_vllm.backend_kwargs, VLLMPythonBackendArgs)
