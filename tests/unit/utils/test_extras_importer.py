"""
Tests for ExtrasImporter lazy-loading utilities.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import pytest

from guidellm.utils.extras_importer import ExtrasImporter, _ImportStub


class TestImportStub:
    """Test the _ImportStub class."""

    def test_stub_creation(self):
        """
        Test that stubs can be created with proper attributes.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", "test-extra")
        assert stub._stub_name == "TestClass"
        assert stub._stub_import_path == "test.module.TestClass"
        assert stub._stub_extras_group == "test-extra"

    def test_stub_call_raises_error(self):
        """
        Test that calling a stub raises ImportError with helpful message.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", "test-extra")
        with pytest.raises(
            ImportError,
            match=r"'TestClass' from 'test\.module\.TestClass' requires optional "
            r"dependencies\. Install with: pip install guidellm\[test-extra\]",
        ):
            stub()

    def test_stub_attribute_access_raises_error(self):
        """
        Test that accessing stub attributes raises AttributeError.

        This allows hasattr() checks to work properly.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", "test-extra")
        with pytest.raises(
            AttributeError,
            match=r"'TestClass\.some_attribute' is not available",
        ):
            _ = stub.some_attribute

    def test_stub_bool_returns_false(self):
        """
        Test that stubs are falsy for truthiness checks.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", "test-extra")
        assert not stub
        assert bool(stub) is False

    def test_stub_multiple_extras_groups(self):
        """
        Test error message with multiple extras groups.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", ["extra1", "extra2"])
        with pytest.raises(
            ImportError,
            match=r"Install with: pip install guidellm\[extra1\] or guidellm\[extra2\]",
        ):
            stub()

    def test_stub_repr(self):
        """
        Test that stub has a helpful representation.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub("TestClass", "test.module.TestClass", "test-extra")
        assert repr(stub) == "<ImportStub for 'TestClass' from 'test.module.TestClass'>"


class TestExtrasImporter:
    """Test the ExtrasImporter class."""

    def test_eager_loading_success(self):
        """
        Test successful eager import of available modules.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os", "path": "os.path"},
            extras_group="test",
            eager=True,
        )

        # Should be imported immediately
        assert importer.os is sys.modules["os"]
        assert importer.path is sys.modules["os"].path

    def test_eager_loading_missing_creates_stub(self):
        """
        Test that eager loading creates stubs for missing imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistent": "non_existent_module_xyz.NonExistent"},
            extras_group="test",
            eager=True,
        )

        # Should create stub, not raise
        stub = importer.NonExistent
        assert isinstance(stub, _ImportStub)

    def test_lazy_loading_success(self):
        """
        Test successful lazy import on first access.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
            eager=False,
        )

        # Should not be imported yet
        assert "os" not in importer._import_cache

        # Access should trigger import
        result = importer.os
        assert result is sys.modules["os"]
        assert "os" in importer._import_cache

    def test_lazy_loading_missing_creates_stub(self):
        """
        Test that lazy loading creates stubs for missing imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistent": "non_existent_module_xyz.NonExistent"},
            extras_group="test",
            eager=False,
        )

        # Access should create stub
        stub = importer.NonExistent
        assert isinstance(stub, _ImportStub)

    def test_import_caching(self):
        """
        Test that successful imports are cached.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
            eager=False,
        )

        # First access
        first_result = importer.os
        # Second access should return cached value
        second_result = importer.os

        assert first_result is second_result
        assert first_result is sys.modules["os"]

    def test_stub_caching(self):
        """
        Test that stubs are cached.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistent": "non_existent_module_xyz.NonExistent"},
            extras_group="test",
            eager=False,
        )

        # First access
        first_stub = importer.NonExistent
        # Second access should return same stub
        second_stub = importer.NonExistent

        assert first_stub is second_stub

    def test_is_available_all_present(self):
        """
        Test is_available returns True when all imports available.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os", "path": "os.path"},
            extras_group="test",
        )

        assert importer.is_available is True

    def test_is_available_some_missing(self):
        """
        Test is_available returns False when some imports missing.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os", "NonExistent": "non_existent_module_xyz.NonExistent"},
            extras_group="test",
        )

        assert importer.is_available is False

    def test_is_available_all_missing(self):
        """
        Test is_available returns False when all imports missing.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistent": "non_existent_module_xyz.NonExistent"},
            extras_group="test",
        )

        assert importer.is_available is False

    def test_module_import(self):
        """
        Test importing entire module (no attribute).

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
        )

        result = importer.os
        assert result is sys.modules["os"]

    def test_object_import(self):
        """
        Test importing object from module.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"path": "os.path"},
            extras_group="test",
        )

        result = importer.path
        assert result is sys.modules["os"].path

    def test_object_import_missing_module(self):
        """
        Test importing object from missing module creates stub.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"TestClass": "non_existent_module.TestClass"},
            extras_group="test",
        )

        result = importer.TestClass
        assert isinstance(result, _ImportStub)

    def test_object_import_missing_attribute(self):
        """
        Test importing missing attribute from existing module creates stub.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistentAttr": "os.non_existent_attribute"},
            extras_group="test",
        )

        result = importer.NonExistentAttr
        assert isinstance(result, _ImportStub)

    def test_unregistered_attribute_raises_error(self):
        """
        Test that accessing unregistered attribute raises AttributeError.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
        )

        with pytest.raises(
            AttributeError, match="No import registered for 'unregistered'"
        ):
            _ = importer.unregistered

    def test_private_attribute_raises_error(self):
        """
        Test that accessing private attributes raises AttributeError.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
        )

        with pytest.raises(
            AttributeError, match="'ExtrasImporter' has no attribute '_private'"
        ):
            _ = importer._private

    def test_thread_safety(self):
        """
        Test that concurrent access doesn't cause race conditions.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os", "path": "os.path"},
            extras_group="test",
            eager=False,  # Start with nothing cached
        )

        results: list[Any] = []
        errors: list[Exception] = []

        def access_imports():
            try:
                results.append(importer.os)
                results.append(importer.path)
            except (ImportError, AttributeError) as e:
                errors.append(e)

        threads = [threading.Thread(target=access_imports) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # All results should be the same cached objects
        os_results = results[::2]  # Every other starting from 0
        path_results = results[1::2]  # Every other starting from 1

        assert all(r is sys.modules["os"] for r in os_results)
        assert all(r is sys.modules["os"].path for r in path_results)

    def test_parse_import_path_module(self):
        """
        Test _parse_import_path for module imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
        )

        module_path, attr_name = importer._parse_import_path("os")
        assert module_path == "os"
        assert attr_name is None

    def test_parse_import_path_object(self):
        """
        Test _parse_import_path for object imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"path": "os.path"},
            extras_group="test",
        )

        module_path, attr_name = importer._parse_import_path("os.path")
        assert module_path == "os"
        assert attr_name == "path"

    def test_parse_import_path_nested_object(self):
        """
        Test _parse_import_path for nested object imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"join": "os.path.join"},
            extras_group="test",
        )

        module_path, attr_name = importer._parse_import_path("os.path.join")
        assert module_path == "os.path"
        assert attr_name == "join"


@pytest.mark.integration
class TestExtrasImporterIntegration:
    """Integration tests for ExtrasImporter."""

    def test_vllm_pattern(self):
        """
        Test the pattern used for vllm extras.

        ### WRITTEN BY AI ###
        """
        # This mimics the vllm.py usage pattern
        importer = ExtrasImporter(
            {
                "SamplingParams": "vllm.SamplingParams",
                "AsyncEngineArgs": "vllm.engine.arg_utils.AsyncEngineArgs",
                "AsyncLLMEngine": "vllm.engine.async_llm_engine.AsyncLLMEngine",
                "RequestOutput": "vllm.outputs.RequestOutput",
            },
            extras_group="vllm",
        )

        # Should have is_available property
        has_vllm = importer.is_available
        assert isinstance(has_vllm, bool)

        # Access should not raise (creates stub if not available)
        sampling_params = importer.SamplingParams
        if has_vllm:
            # If vllm is installed, should be the real class
            assert not isinstance(sampling_params, _ImportStub)
        else:
            # If vllm is not installed, should be a stub
            assert isinstance(sampling_params, _ImportStub)
            # Calling stub should raise with helpful message
            with pytest.raises(ImportError, match="pip install guidellm\\[vllm\\]"):
                sampling_params()

    def test_audio_pattern(self):
        """
        Test the pattern used for audio extras.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {
                "AudioSamples": "torchcodec.AudioSamples",
                "AudioDecoder": "torchcodec.decoders.AudioDecoder",
                "AudioEncoder": "torchcodec.encoders.AudioEncoder",
            },
            extras_group="audio",
        )

        has_audio = importer.is_available
        assert isinstance(has_audio, bool)

        audio_decoder = importer.AudioDecoder
        if has_audio:
            assert not isinstance(audio_decoder, _ImportStub)
        else:
            assert isinstance(audio_decoder, _ImportStub)
            with pytest.raises(ImportError, match="pip install guidellm\\[audio\\]"):
                audio_decoder()

    def test_vision_pattern(self):
        """
        Test the pattern used for vision extras.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {
                "PILImage": "PIL.Image",
            },
            extras_group="vision",
        )

        has_vision = importer.is_available
        assert isinstance(has_vision, bool)

        pil_image = importer.PILImage
        if has_vision:
            assert not isinstance(pil_image, _ImportStub)
        else:
            assert isinstance(pil_image, _ImportStub)
            with pytest.raises(ImportError, match="pip install guidellm\\[vision\\]"):
                pil_image.open("test.jpg")
