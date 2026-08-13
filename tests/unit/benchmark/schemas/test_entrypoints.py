"""
Unit tests for backend transformation to typed BackendArgs instances,
MetricsArgs registry validation, and BenchmarkScenario environment variable support.

Tests the automatic conversion of backend configuration from dict-based
backend into properly typed BackendArgs instances during model validation,
MetricsArgs polymorphic dispatch to GenerativeMetricsArgs, and env var
loading for BenchmarkScenario.

### WRITTEN BY AI ###
"""

import pytest
import yaml
from pydantic import ValidationError

from guidellm.backends.backend import BackendArgs
from guidellm.backends.openai.http import OpenAIHTTPBackendArgs
from guidellm.backends.openai.websocket import OpenAIWebSocketBackendArgs
from guidellm.benchmark.schemas.entrypoints import (
    BenchmarkArgs,
    BenchmarkScenario,
    GenerativeMetricsArgs,
    MetricsArgs,
)
from guidellm.utils.typing import BLANK

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
    "profile": {"kind": "sweep", "sweep_size": 10},
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

    def test_openai_websocket_backend_validates(self) -> None:
        """WebSocket backend accepts ``request_format`` (CLI --request-format)."""
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_websocket",
                    "target": "http://localhost:8000",
                    "model": "rt-model",
                },
                **_PIPELINE_DEFAULTS,
            }
        )
        assert isinstance(args.backend, OpenAIWebSocketBackendArgs)
        assert args.backend.target == "http://localhost:8000"
        assert args.backend.model == "rt-model"
        assert args.backend.request_format == "/v1/realtime"

        with_format = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_websocket",
                    "target": "http://localhost:8000",
                    "model": "rt-model",
                    "request_format": "/v1/realtime",
                },
                **_PIPELINE_DEFAULTS,
            }
        )
        assert with_format.backend.request_format == "/v1/realtime"

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


class TestBenchmarkScenario:
    """Test BenchmarkScenario validation and defaults."""

    @pytest.mark.sanity
    def test_create_scenario_with_minimal_spec(self):
        """
        Test creating a BenchmarkScenario with minimal spec.

        ## WRITTEN BY AI ##
        """
        scenario = BenchmarkScenario.create(
            spec={
                **_PIPELINE_DEFAULTS,
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
            },
            scenario=None,
        )

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://localhost:8000"
        assert scenario.spec.profile.kind == "sweep"
        assert scenario.spec.profile.sweep_size == 10  # type: ignore[union-attr]

    @pytest.mark.sanity
    def test_create_from_file(self, tmp_path):
        """
        Test creating a BenchmarkScenario from a YAML file.

        ## WRITTEN BY AI ##
        """
        scenario = yaml.dump(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                },
                "benchmarks": [],
            }
        )

        yaml_file = tmp_path / "scenario.yaml"
        yaml_file.write_text(scenario)

        scenario = BenchmarkScenario.create(scenario=yaml_file)

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://localhost:8000"
        assert scenario.spec.profile.kind == "sweep"

    @pytest.mark.regression
    def test_create_from_file_without_overrides(self, tmp_path):
        """
        Test creating a BenchmarkScenario from a YAML file with overrides.

        ## WRITTEN BY AI ##
        """
        scenario = yaml.dump(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "profile": {"kind": "concurrent"},
                },
                "benchmarks": [
                    {"profile.streams": 1},
                    {"profile.streams": 2},
                    {"profile.streams": 4},
                ],
            }
        )

        yaml_file = tmp_path / "scenario.yaml"
        yaml_file.write_text(scenario)

        scenario = BenchmarkScenario.create(
            scenario=yaml_file,
            spec={
                "backend": {
                    "kind": "openai_http",
                    "target": "http://override-server:9000",
                },
            },
            benchmarks=BLANK,
        )

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://override-server:9000"
        assert len(scenario.benchmarks) == 3, "Expected 3 benchmarks from file"

    @pytest.mark.regression
    def test_create_from_file_with_overrides(self, tmp_path):
        """
        Test creating a BenchmarkScenario from a YAML file with overrides.

        ## WRITTEN BY AI ##
        """
        scenario = yaml.dump(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "profile": {"kind": "concurrent"},
                },
                "benchmarks": [
                    {"profile.streams": 1},
                    {"profile.streams": 2},
                    {"profile.streams": 4},
                ],
            }
        )

        yaml_file = tmp_path / "scenario.yaml"
        yaml_file.write_text(scenario)

        scenario = BenchmarkScenario.create(
            scenario=yaml_file,
            spec={
                "backend": {
                    "kind": "openai_http",
                    "target": "http://override-server:9000",
                },
            },
            benchmarks=[
                {"profile.streams": 8},
                {"profile.streams": 16},
            ],
        )

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://override-server:9000"
        assert len(scenario.benchmarks) == 2, "Expected 2 benchmarks from overrides"


@pytest.mark.sanity
class TestBenchmarkScenarioEnvVars:
    """Test BenchmarkScenario environment variable loading."""

    def test_backend_target_from_env(self, monkeypatch):
        """
        GUIDELLM__SPEC__BACKEND__TARGET sets the backend target.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__TARGET", "http://env-server:9000")

        scenario = BenchmarkScenario.model_validate(
            {"spec": {**_PIPELINE_DEFAULTS, "backend": {"kind": "openai_http"}}}
        )

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://env-server:9000"

    def test_discriminated_union_dispatch_from_env(self, monkeypatch):
        """
        Backend kind and variant-specific fields can be set via env vars.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__KIND", "openai_http")
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__TARGET", "http://env-server:9000")
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__MODEL", "env-model")

        scenario = BenchmarkScenario.model_validate({"spec": _PIPELINE_DEFAULTS})

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://env-server:9000"
        assert scenario.spec.backend.model == "env-model"

    def test_sweep_size_string_coercion_from_env(self, monkeypatch):
        """
        GUIDELLM__SPEC__PROFILE__SWEEP_SIZE with string value is coerced to int.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__PROFILE__SWEEP_SIZE", "5")

        scenario = BenchmarkScenario.model_validate(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "profile": {"kind": "sweep"},
                }
            }
        )

        assert scenario.spec.profile.sweep_size == 5  # type: ignore[union-attr]

    def test_kwargs_override_env_vars(self, monkeypatch):
        """
        Explicit kwargs take precedence over env var values.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__TARGET", "http://from-env:9000")

        scenario = BenchmarkScenario.model_validate(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://from-kwarg:8000",
                    },
                }
            }
        )

        assert isinstance(scenario.spec.backend, OpenAIHTTPBackendArgs)
        assert scenario.spec.backend.target == "http://from-kwarg:8000"


@pytest.mark.sanity
class TestMetricsArgsValidation:
    """Test MetricsArgs registry dispatch and GenerativeMetricsArgs validation."""

    def test_default_metrics_is_generative(self):
        """
        Default BenchmarkArgs creates GenerativeMetricsArgs with correct defaults.

        ## WRITTEN BY AI ##
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

        assert isinstance(args.metrics, GenerativeMetricsArgs)
        assert isinstance(args.metrics, MetricsArgs)
        assert args.metrics.kind == "generative"
        assert args.metrics.sample_size is None
        assert args.metrics.prefer_response_metrics is True

    def test_explicit_metrics_dict_validates(self):
        """
        Explicit metrics dict with kind=generative validates to GenerativeMetricsArgs
        with specified field values.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "metrics": {
                    "kind": "generative",
                    "sample_size": 100,
                    "prefer_response_metrics": False,
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.metrics, GenerativeMetricsArgs)
        assert args.metrics.sample_size == 100
        assert args.metrics.prefer_response_metrics is False

    def test_metrics_serialization_round_trip(self):
        """
        Serialization and deserialization preserves GenerativeMetricsArgs fields.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "metrics": {
                    "kind": "generative",
                    "sample_size": 50,
                },
                **_PIPELINE_DEFAULTS,
            }
        )

        serialized = args.metrics.model_dump()
        assert serialized["kind"] == "generative"
        assert serialized["sample_size"] == 50
        assert serialized["prefer_response_metrics"] is True

        args2 = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "metrics": serialized,
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args2.metrics, GenerativeMetricsArgs)
        assert args2.metrics.sample_size == 50

    def test_invalid_metrics_kind_raises(self):
        """
        Unregistered metrics kind raises ValidationError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            BenchmarkArgs.model_validate(
                {
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "metrics": {"kind": "nonexistent"},
                    **_PIPELINE_DEFAULTS,
                }
            )

    def test_metrics_sample_size_zero(self):
        """
        sample_size=0 is valid and means keep no request data.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "metrics": {"kind": "generative", "sample_size": 0},
                **_PIPELINE_DEFAULTS,
            }
        )

        assert isinstance(args.metrics, GenerativeMetricsArgs)
        assert args.metrics.sample_size == 0

    def test_metrics_from_scenario_spec(self):
        """
        Metrics args set in the scenario spec are preserved after validation.

        ## WRITTEN BY AI ##
        """
        scenario = BenchmarkScenario.model_validate(
            {
                "spec": {
                    **_PIPELINE_DEFAULTS,
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "metrics": {
                        "kind": "generative",
                        "sample_size": 200,
                    },
                }
            }
        )

        assert isinstance(scenario.spec.metrics, GenerativeMetricsArgs)
        assert scenario.spec.metrics.sample_size == 200
