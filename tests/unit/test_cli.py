"""
Unit tests for CLI functionality, specifically the version flag.
"""

import importlib.metadata
import re

import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.smoke
def test_version_flag_long():
    """Test that --version flag works correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "guidellm version:" in result.output
    assert result.output.strip().startswith("guidellm version:")


@pytest.mark.smoke
def test_version_flag_displays_actual_version():
    """Test that --version displays the actual version from version.py."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0

    version_pattern = r"guidellm version: \d+\.\d+"
    assert re.search(version_pattern, result.output)


@pytest.mark.smoke
def test_version_flag_exits_cleanly():
    """Test that --version exits without processing other commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version", "benchmark"])

    assert result.exit_code == 0
    assert "guidellm version:" in result.output
    assert "Commands to run a new benchmark" not in result.output


@pytest.mark.smoke
def test_help_shows_version_option():
    """Test that --help shows the --version option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.output
    assert "Show the version and exit" in result.output


@pytest.mark.smoke
def test_other_commands_still_work():
    """Test that other CLI commands still work after adding version flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "benchmark" in result.output
    assert "config" in result.output
    assert "preprocess" in result.output


@pytest.mark.smoke
def test_version_flag_case_sensitivity():
    """Test that --version flag is case sensitive."""
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "guidellm version:" in result.output

    # --VERSION should not work
    result = runner.invoke(cli, ["--VERSION"])
    assert result.exit_code != 0
    assert "No such option" in result.output


@pytest.mark.integration
def test_version_integration_with_actual_version():
    """Integration test to verify version matches importlib.metadata."""
    try:
        actual_version = importlib.metadata.version("guidellm")

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        expected_output = f"guidellm version: {actual_version}"
        assert expected_output in result.output
    except importlib.metadata.PackageNotFoundError:
        # If package is not installed, the CLI should show an error
        # This is expected behavior when the package isn't properly installed
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        # Click will handle the error when package is not found
        assert result.exit_code != 0
