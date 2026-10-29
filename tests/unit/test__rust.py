"""Tests for the connector's native Rust extension module."""

import pytest

from guidellm._rust import hello_world, version_info


class TestNativeRustExtension:
    """Tests for the native Rust extension module.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_hello_world_returns_string(self):
        """Verify the connector exposes the Rust core's greeting as a Python string.

        ## WRITTEN BY AI ##
        """
        result = hello_world()
        assert isinstance(result, str)
        assert result == "Hello from GuideLLM Rust!"

    @pytest.mark.smoke
    def test_version_info_returns_dict(self):
        """Verify version_info returns dict with git metadata.

        ## WRITTEN BY AI ##
        """
        info = version_info()
        assert isinstance(info, dict)
        assert "git_sha" in info
        assert "git_branch" in info
        assert "git_describe" in info
