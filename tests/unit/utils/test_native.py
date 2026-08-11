"""Tests for the native Rust extension modules."""

import pytest

from guidellm._rust import version_info
from guidellm.utils._rust import hello_world as rust_hello_world
from guidellm.utils.native import hello_world


class TestNativeRustExtension:
    """Tests for the native Rust extension module.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_hello_world_returns_string(self):
        """Verify hello_world returns expected greeting.

        ## WRITTEN BY AI ##
        """
        result = hello_world()
        assert isinstance(result, str)
        assert result == "Hello from GuideLLM Rust!"

    @pytest.mark.smoke
    def test_rust_utils_importable(self):
        """Verify utils._rust module is directly importable.

        ## WRITTEN BY AI ##
        """
        assert rust_hello_world() == "Hello from GuideLLM Rust!"

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
