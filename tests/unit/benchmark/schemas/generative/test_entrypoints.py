"""
Unit tests for backend_kwargs transformation to typed BackendArgs instances.

Tests the automatic conversion of backend configuration from dict-based
backend_kwargs into properly typed BackendArgs instances during model validation.

### WRITTEN BY AI ###
"""

import pytest
from pydantic import ValidationError

from guidellm.backends.backend import BackendArgs
from guidellm.backends.openai.http import OpenAIHttpBackendArgs
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
        Test that dict backend_kwargs is transformed to BackendArgs.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Verify backend_kwargs is typed OpenAIHttpBackendArgs
        assert isinstance(args.backend_kwargs, OpenAIHttpBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"

    def test_dict_with_request_format(self):
        """
        Test that request_format is included in BackendArgs transformation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "request_format": "/v1/completions",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        assert isinstance(args.backend_kwargs, OpenAIHttpBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"
        assert args.backend_kwargs.request_format == "/v1/completions"

    def test_serialization_round_trip(self):
        """
        Test that serialization and deserialization preserves typed backend_kwargs.

        ### WRITTEN BY AI ###
        """
        # Create instance with dict backend_kwargs
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Serialize to dict
        serialized = args.model_dump()

        # Should serialize backend_kwargs as dict
        assert isinstance(serialized["backend_kwargs"], dict)
        assert serialized["backend_kwargs"]["target"] == "http://localhost:9000"
        assert serialized["backend_kwargs"]["model"] == "test_model"

        # Deserialize back
        args2 = BenchmarkGenerativeTextArgs.model_validate(serialized)

        # Should reconstruct typed instance
        assert isinstance(args2.backend_kwargs, OpenAIHttpBackendArgs)
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
                    "backend": "openai_http",
                    "backend_kwargs": {
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
                    "backend": "openai_http",
                    "backend_kwargs": {
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
                "backend": "vllm_python",
                "backend_kwargs": {
                    "model": "facebook/opt-125m",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Verify backend_kwargs is typed VLLMPythonBackendArgs
        assert VLLMPythonBackendArgs is not None
        assert isinstance(args.backend_kwargs, VLLMPythonBackendArgs)
        assert args.backend_kwargs.model == "facebook/opt-125m"
        # VLLM backend doesn't use target
        assert args.backend_kwargs.target is None

    @pytest.mark.skipif(not HAS_VLLM, reason="VLLM not installed")
    def test_vllm_backend_rejects_target(self):
        """
        Test that VLLM backend rejects target parameter.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend": "vllm_python",
                    "backend_kwargs": {
                        "target": "http://localhost:9000",  # Not allowed for VLLM
                        "model": "facebook/opt-125m",
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

        # Should have validation error about target not being supported
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any("target" in str(err).lower() for err in errors)

    def test_empty_dict_backend_kwargs(self):
        """
        Test handling of empty dict backend_kwargs.

        ### WRITTEN BY AI ###
        """
        # Empty dict should fail validation if required fields are missing
        with pytest.raises(ValidationError):
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend": "openai_http",
                    "backend_kwargs": {},
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

    def test_default_backend_kwargs(self):
        """
        Test that default backend_kwargs (empty dict) fails validation.

        ### WRITTEN BY AI ###
        """
        # Default backend_kwargs should fail validation if required fields missing
        with pytest.raises(ValidationError):
            BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend": "openai_http",
                    # No backend_kwargs provided, uses default
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )

    def test_already_typed_backend_kwargs_preserved(self):
        """
        Test that already-typed BackendArgs instances are preserved.

        ### WRITTEN BY AI ###
        """
        # Create a typed BackendArgs instance
        backend_args = OpenAIHttpBackendArgs(
            target="http://localhost:9000", model="test_model"
        )

        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": backend_args,
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Should preserve the typed instance
        assert args.backend_kwargs is backend_args
        assert isinstance(args.backend_kwargs, OpenAIHttpBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        assert args.backend_kwargs.model == "test_model"

    def test_backend_kwargs_is_backendargs_subclass(self):
        """
        Test that backend_kwargs is always a BackendArgs subclass after validation.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        # Should be a BackendArgs subclass
        assert isinstance(args.backend_kwargs, BackendArgs)
        assert isinstance(args.backend_kwargs, OpenAIHttpBackendArgs)

    def test_extra_fields_allowed(self):
        """
        Test that extra fields in backend_kwargs are allowed.

        ### WRITTEN BY AI ###
        """
        # Extra fields should be allowed due to ConfigDict(extra="allow")
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",  # Extra field
                    "timeout": 30,  # Extra field
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        assert isinstance(args.backend_kwargs, OpenAIHttpBackendArgs)
        assert args.backend_kwargs.target == "http://localhost:9000"
        # Extra fields should be accessible
        assert hasattr(args.backend_kwargs, "api_key")
        assert args.backend_kwargs.api_key == "secret123"

    def test_serialization_preserves_extra_fields(self):
        """
        Test that serialization preserves extra fields.

        ### WRITTEN BY AI ###
        """
        args = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:9000",
                    "model": "test_model",
                    "api_key": "secret123",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )

        serialized = args.model_dump()

        # Extra fields should be in serialized output
        assert serialized["backend_kwargs"]["api_key"] == "secret123"

    def test_different_backend_types(self):
        """
        Test that different backend types get correct BackendArgs subclasses.

        ### WRITTEN BY AI ###
        """
        # OpenAI HTTP backend
        args_openai = BenchmarkGenerativeTextArgs.model_validate(
            {
                "backend": "openai_http",
                "backend_kwargs": {
                    "target": "http://localhost:8000",
                    "model": "gpt-3.5-turbo",
                },
                "data": ["prompt_tokens=256,output_tokens=128"],
            }
        )
        assert isinstance(args_openai.backend_kwargs, OpenAIHttpBackendArgs)

        # VLLM Python backend (if available)
        if HAS_VLLM:
            args_vllm = BenchmarkGenerativeTextArgs.model_validate(
                {
                    "backend": "vllm_python",
                    "backend_kwargs": {
                        "model": "facebook/opt-125m",
                    },
                    "data": ["prompt_tokens=256,output_tokens=128"],
                }
            )
            assert VLLMPythonBackendArgs is not None
            assert isinstance(args_vllm.backend_kwargs, VLLMPythonBackendArgs)
