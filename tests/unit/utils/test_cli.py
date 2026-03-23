from __future__ import annotations

import os

import click
import pytest
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from guidellm.utils.cli import EnvVarValidator


# Test fixtures for Pydantic models
class NestedModel(BaseModel):
    """Test nested model."""

    field1: str = "default1"
    field2: int = 42


class DeepNestedModel(BaseModel):
    """Test deeply nested model."""

    deep_field: str = "deep"


class MiddleModel(BaseModel):
    """Test middle level nested model."""

    middle_field: str = "middle"
    deep: DeepNestedModel = DeepNestedModel()


class MockSettings(BaseSettings):
    """Mock settings class for testing."""

    model_config = SettingsConfigDict(
        env_prefix="TEST__",
        env_nested_delimiter="__",
    )

    simple_field: str = "default"
    number_field: int = 10
    nested: NestedModel = NestedModel()
    optional_field: str | None = None


class MockSettingsWithDeepNesting(BaseSettings):
    """Mock settings with deep nesting for testing."""

    model_config = SettingsConfigDict(
        env_prefix="DEEP__",
        env_nested_delimiter="__",
    )

    top_field: str = "top"
    middle: MiddleModel = MiddleModel()


class TestEnvVarValidator:
    """Test EnvVarValidator static class.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_get_valid_env_vars_returns_set(self):
        """Test that get_valid_env_vars returns a set.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator.get_valid_env_vars()
        assert isinstance(result, set)

    @pytest.mark.smoke
    def test_get_valid_env_vars_includes_settings(self):
        """Test that result includes Settings environment variables.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator.get_valid_env_vars()
        # Should include some standard GuideLLM settings
        assert "GUIDELLM__LOGGING__DISABLED" in result
        assert "GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL" in result

    @pytest.mark.sanity
    def test_get_valid_env_vars_with_click_context(self):
        """Test extraction from Click context.

        ### WRITTEN BY AI ###
        """

        @click.command()
        @click.option("--name", default="test")
        @click.option("--value", default=42)
        @click.pass_context
        def test_command(ctx, name, value):
            # Get valid env vars with context
            return EnvVarValidator.get_valid_env_vars(ctx=ctx)

        # Create context with auto_envvar_prefix
        ctx = click.Context(
            test_command,
            auto_envvar_prefix="MYAPP",
        )

        with ctx:
            result = EnvVarValidator.get_valid_env_vars(ctx=ctx)
            assert "MYAPP_NAME" in result
            assert "MYAPP_VALUE" in result

    @pytest.mark.sanity
    def test_get_valid_env_vars_without_context(self):
        """Test that it works without Click context.

        ### WRITTEN BY AI ###
        """
        # Should work even without a context
        result = EnvVarValidator.get_valid_env_vars(ctx=None)
        assert isinstance(result, set)
        # Should still have settings env vars
        assert "GUIDELLM__LOGGING__DISABLED" in result


class TestExtractClickEnvVars:
    """Test Click environment variable extraction.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_simple_command_with_prefix(self):
        """Test simple command with auto_envvar_prefix.

        ### WRITTEN BY AI ###
        """

        @click.command()
        @click.option("--name", default="test")
        @click.option("--count", default=1)
        def test_command(name, count):
            pass

        ctx = click.Context(
            test_command,
            auto_envvar_prefix="TESTAPP",
        )

        result = EnvVarValidator._extract_click_env_vars(ctx)
        assert "TESTAPP_NAME" in result
        assert "TESTAPP_COUNT" in result

    @pytest.mark.sanity
    def test_nested_commands_with_parent_context(self):
        """Test nested commands with parent context.

        ### WRITTEN BY AI ###
        """

        @click.group()
        @click.option("--config", default="config.yml")
        def main(config):
            pass

        @main.command()
        @click.option("--input", default="input.txt")
        def process(input_file):
            pass

        # Create parent context
        parent_ctx = click.Context(
            main,
            auto_envvar_prefix="APP",
        )

        # Create child context with parent
        child_ctx = click.Context(
            process,
            parent=parent_ctx,
            auto_envvar_prefix="APP",
        )

        result = EnvVarValidator._extract_click_env_vars(child_ctx)
        # Should include both parent and child env vars
        assert "APP_CONFIG" in result
        assert "APP_INPUT" in result

    @pytest.mark.sanity
    def test_explicit_envvar_string(self):
        """Test explicit envvar as string.

        ### WRITTEN BY AI ###
        """

        @click.command()
        @click.option("--name", envvar="CUSTOM_NAME")
        def test_command(name):
            pass

        ctx = click.Context(
            test_command,
            auto_envvar_prefix="APP",
        )

        result = EnvVarValidator._extract_click_env_vars(ctx)
        assert "CUSTOM_NAME" in result

    @pytest.mark.sanity
    def test_explicit_envvar_list(self):
        """Test explicit envvar as list.

        ### WRITTEN BY AI ###
        """

        @click.command()
        @click.option("--name", envvar=["CUSTOM_NAME", "ALT_NAME"])
        def test_command(name):
            pass

        ctx = click.Context(
            test_command,
            auto_envvar_prefix="APP",
        )

        result = EnvVarValidator._extract_click_env_vars(ctx)
        assert "CUSTOM_NAME" in result
        assert "ALT_NAME" in result

    @pytest.mark.smoke
    def test_no_prefix_returns_empty(self):
        """Test that no prefix returns empty set.

        ### WRITTEN BY AI ###
        """

        @click.command()
        @click.option("--name", default="test")
        def test_command(name):
            pass

        # Context without auto_envvar_prefix
        ctx = click.Context(test_command)

        result = EnvVarValidator._extract_click_env_vars(ctx)
        # Should be empty since no auto prefix
        assert len(result) == 0


class TestExtractSettingsEnvVars:
    """Test Settings environment variable extraction.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_extract_guidellm_settings(self):
        """Test extraction from real GuideLLM settings.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._extract_settings_env_vars()

        # Should include GuideLLM settings
        assert "GUIDELLM__LOGGING__DISABLED" in result
        assert "GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL" in result
        assert "GUIDELLM__DEFAULT_ASYNC_LOOP_SLEEP" in result

    @pytest.mark.sanity
    def test_extract_nested_settings(self):
        """Test that nested settings are properly extracted.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._extract_settings_env_vars()

        # Should have nested fields with delimiter
        assert "GUIDELLM__LOGGING__DISABLED" in result
        assert "GUIDELLM__DATASET__PREFERRED_DATA_COLUMNS" in result

    @pytest.mark.sanity
    def test_extract_deeply_nested_settings(self):
        """Test that deeply nested settings are properly extracted.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._extract_settings_env_vars()

        # Should include deeply nested report generation settings
        assert "GUIDELLM__REPORT_GENERATION__SOURCE" in result


class TestWalkSettingsFields:
    """Test recursive field walking.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_flat_model_fields(self):
        """Test flat model fields.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._walk_settings_fields(
            NestedModel,
            "PREFIX__",
            "__",
        )

        assert "PREFIX__FIELD1" in result
        assert "PREFIX__FIELD2" in result

    @pytest.mark.sanity
    def test_nested_model_fields(self):
        """Test nested model fields.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._walk_settings_fields(
            MockSettings,
            "TEST__",
            "__",
        )

        assert "TEST__SIMPLE_FIELD" in result
        assert "TEST__NUMBER_FIELD" in result
        assert "TEST__NESTED__FIELD1" in result
        assert "TEST__NESTED__FIELD2" in result

    @pytest.mark.sanity
    def test_optional_type_handling(self):
        """Test Optional/Union type handling.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._walk_settings_fields(
            MockSettings,
            "TEST__",
            "__",
        )

        # Should handle Optional[str] field
        assert "TEST__OPTIONAL_FIELD" in result


class TestIntegrationWithRealSettings:
    """Integration tests with real GuideLLM settings.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_real_guidellm_settings(self):
        """Test with real GuideLLM settings.

        ### WRITTEN BY AI ###
        """
        result = EnvVarValidator._extract_settings_env_vars()

        # Check for some known settings
        assert "GUIDELLM__LOGGING__DISABLED" in result
        assert "GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL" in result
        assert "GUIDELLM__DEFAULT_ASYNC_LOOP_SLEEP" in result
        assert "GUIDELLM__DATASET__PREFERRED_DATA_COLUMNS" in result

    @pytest.mark.sanity
    def test_valid_env_detection(self):
        """Test that valid environment variables are correctly detected.

        ### WRITTEN BY AI ###
        """
        # Set a valid Settings env var
        os.environ["GUIDELLM__LOGGING__DISABLED"] = "true"

        try:
            valid_envs = EnvVarValidator.get_valid_env_vars()

            # Settings env vars should be recognized even without Click context
            assert "GUIDELLM__LOGGING__DISABLED" in valid_envs
            assert "GUIDELLM__DEFAULT_ASYNC_LOOP_SLEEP" in valid_envs

        finally:
            # Cleanup
            os.environ.pop("GUIDELLM__LOGGING__DISABLED", None)

    @pytest.mark.sanity
    def test_invalid_env_detection(self):
        """Test that invalid environment variables are correctly filtered.

        ### WRITTEN BY AI ###
        """
        # Set a mix of valid and invalid env vars
        os.environ["GUIDELLM__LOGGING__DISABLED"] = "true"
        os.environ["GUIDELLM_INVALID_VAR"] = "test"
        os.environ["GUIDELLM_TYPO"] = "value"

        try:
            valid_envs = EnvVarValidator.get_valid_env_vars()

            # Check which are valid
            assert "GUIDELLM__LOGGING__DISABLED" in valid_envs
            assert "GUIDELLM_INVALID_VAR" not in valid_envs
            assert "GUIDELLM_TYPO" not in valid_envs

        finally:
            # Cleanup
            os.environ.pop("GUIDELLM__LOGGING__DISABLED", None)
            os.environ.pop("GUIDELLM_INVALID_VAR", None)
            os.environ.pop("GUIDELLM_TYPO", None)
