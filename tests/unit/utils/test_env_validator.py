"""
Unit tests for environment variable validation utilities.

## WRITTEN BY AI ##
"""

import pytest
from pydantic import BaseModel

from guidellm.benchmark.schemas.entrypoints import BenchmarkScenario
from guidellm.settings import Settings
from guidellm.utils.env_validator import (
    _resolve_model_type,
    _walk_model_fields,
    get_valid_env_vars,
    validate_env_vars,
)


@pytest.mark.smoke
class TestResolveModelType:
    """Test _resolve_model_type annotation unwrapping."""

    def test_returns_none_for_none(self):
        """
        ## WRITTEN BY AI ##
        """
        assert _resolve_model_type(None) is None

    def test_returns_none_for_primitive(self):
        """
        ## WRITTEN BY AI ##
        """
        assert _resolve_model_type(int) is None
        assert _resolve_model_type(str) is None
        assert _resolve_model_type(float) is None

    def test_returns_base_model_subclass(self):
        """
        ## WRITTEN BY AI ##
        """

        class MyModel(BaseModel):
            x: int = 1

        assert _resolve_model_type(MyModel) is MyModel

    def test_unwraps_optional(self):
        """
        ## WRITTEN BY AI ##
        """

        class MyModel(BaseModel):
            x: int = 1

        assert _resolve_model_type(MyModel | None) is MyModel

    def test_unwraps_list(self):
        """
        ## WRITTEN BY AI ##
        """

        class MyModel(BaseModel):
            x: int = 1

        assert _resolve_model_type(list[MyModel]) is MyModel


@pytest.mark.smoke
class TestWalkModelFields:
    """Test _walk_model_fields field enumeration and prefix generation."""

    def test_returns_prefixes_for_nested_models(self):
        """
        Nested BaseModel fields produce prefix entries.

        ## WRITTEN BY AI ##
        """
        env_vars, prefixes = _walk_model_fields(BenchmarkScenario, "GUIDELLM__", "__")

        assert "GUIDELLM__SPEC__" in prefixes
        assert "GUIDELLM__SPEC__BACKEND__" in prefixes
        assert "GUIDELLM__SPEC__PROFILE__" in prefixes

    def test_returns_exact_names_for_leaf_fields(self):
        """
        Leaf fields (non-BaseModel) produce exact env var names.

        ## WRITTEN BY AI ##
        """
        env_vars, prefixes = _walk_model_fields(Settings, "GUIDELLM__", "__")

        assert "GUIDELLM__MAX_CONCURRENCY" in env_vars
        assert "GUIDELLM__DEFAULT_SWEEP_NUMBER" in env_vars

    def test_settings_no_false_positives(self):
        """
        Exact env var set does not contain prefix-style entries.

        ## WRITTEN BY AI ##
        """
        env_vars, _prefixes = _walk_model_fields(Settings, "GUIDELLM__", "__")

        for var in env_vars:
            assert not var.endswith("__"), f"Leaf var {var} looks like a prefix"


@pytest.mark.sanity
class TestValidateEnvVars:
    """Test validate_env_vars accepts/rejects env vars correctly."""

    def test_accepts_exact_settings_var(self, monkeypatch):
        """
        Known Settings leaf env var is accepted.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__MAX_CONCURRENCY", "64")
        invalid, valid = validate_env_vars(Settings)

        assert "GUIDELLM__MAX_CONCURRENCY" in valid
        assert "GUIDELLM__MAX_CONCURRENCY" not in invalid

    def test_accepts_nested_model_prefix_var(self, monkeypatch):
        """
        Env var under a nested model prefix is accepted via prefix matching.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__TARGET", "http://localhost:8000")
        invalid, valid = validate_env_vars(BenchmarkScenario)

        assert "GUIDELLM__SPEC__BACKEND__TARGET" in valid
        assert "GUIDELLM__SPEC__BACKEND__TARGET" not in invalid

    def test_rejects_unknown_var(self, monkeypatch):
        """
        Env var that doesn't match any valid name or prefix is rejected.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__TOTALLY_BOGUS", "value")
        invalid, valid = validate_env_vars(Settings, BenchmarkScenario)

        assert "GUIDELLM__TOTALLY_BOGUS" in invalid
        assert "GUIDELLM__TOTALLY_BOGUS" not in valid

    def test_accepts_discriminated_union_variant_fields(self, monkeypatch):
        """
        Variant-specific fields under a discriminated union prefix are accepted.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__KIND", "openai_http")
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__TARGET", "http://localhost:8000")
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__MODEL", "gpt2")
        monkeypatch.setenv("GUIDELLM__SPEC__BACKEND__API_KEY", "sk-test")

        invalid, valid = validate_env_vars(BenchmarkScenario)

        assert "GUIDELLM__SPEC__BACKEND__KIND" in valid
        assert "GUIDELLM__SPEC__BACKEND__TARGET" in valid
        assert "GUIDELLM__SPEC__BACKEND__MODEL" in valid
        assert "GUIDELLM__SPEC__BACKEND__API_KEY" in valid
        assert not invalid


@pytest.mark.sanity
class TestGetValidEnvVars:
    """Test get_valid_env_vars returns both exact and prefix sets."""

    def test_returns_tuple(self):
        """
        ## WRITTEN BY AI ##
        """
        result = get_valid_env_vars(Settings)
        assert isinstance(result, tuple)
        assert len(result) == 2
        env_vars, prefixes = result
        assert isinstance(env_vars, set)
        assert isinstance(prefixes, set)

    def test_merges_multiple_models(self):
        """
        ## WRITTEN BY AI ##
        """
        env_vars, prefixes = get_valid_env_vars(Settings, BenchmarkScenario)

        assert "GUIDELLM__MAX_CONCURRENCY" in env_vars
        assert "GUIDELLM__SPEC__BACKEND__" in prefixes
