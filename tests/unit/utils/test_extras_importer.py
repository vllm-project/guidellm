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
        Test that accessing stub attributes raises ImportError when import fails.

        ### WRITTEN BY AI ###
        """
        stub = _ImportStub(
            "TestClass", "test.module.TestClass", "test-extra", eager=True
        )
        with pytest.raises(
            ImportError,
            match=r"'TestClass' from 'test\.module\.TestClass' requires optional",
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
        stub = _ImportStub(
            "TestClass", "test.module.TestClass", "test-extra", eager=False
        )
        assert repr(stub) == "<ImportStub (not imported) for 'TestClass'>"

        # After failed import attempt
        stub_eager = _ImportStub(
            "TestClass", "test.module.TestClass", "test-extra", eager=True
        )
        assert repr(stub_eager) == "<ImportStub (failed) for 'TestClass'>"


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

        # Stubs are created immediately and import eagerly
        # Accessing the attribute returns the stub, which forwards to the real object
        assert importer._stub_cache["os"]._stub_target is sys.modules["os"]
        assert importer._stub_cache["path"]._stub_target is sys.modules["os"].path

        # Direct access should return the real objects through stub forwarding
        os_module = importer.os
        path_module = importer.path
        assert os_module.__name__ == "os"
        assert path_module.__name__ in {"posixpath", "ntpath"}

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

        # Stub is created and import is attempted immediately (and fails)
        assert "NonExistent" in importer._stub_cache
        assert importer._stub_cache["NonExistent"]._stub_failed is True

        # Accessing it should raise ImportError
        with pytest.raises(ImportError, match="requires optional dependencies"):
            importer.NonExistent()

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

        # Stub should not exist yet
        assert "os" not in importer._stub_cache

        # Access should create stub and trigger import on first use
        stub = importer.os
        assert isinstance(stub, _ImportStub)
        assert "os" in importer._stub_cache

        # Stub should not have imported yet (lazy)
        assert stub._stub_target is None

        # Using the stub should trigger import
        result = stub.__name__  # Access an attribute
        assert result == "os"
        assert stub._stub_target is sys.modules["os"]

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

        # Access should create stub (not imported yet)
        stub = importer.NonExistent
        assert isinstance(stub, _ImportStub)
        assert stub._stub_target is None
        assert not stub._stub_failed

        # Using the stub should trigger import attempt and fail
        with pytest.raises(ImportError, match="requires optional dependencies"):
            stub()

    def test_import_caching(self):
        """
        Test that stubs are cached and imports are cached within stubs.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
            eager=False,
        )

        # First access creates stub
        first_stub = importer.os
        # Second access should return same stub
        second_stub = importer.os

        assert first_stub is second_stub
        assert isinstance(first_stub, _ImportStub)

        # Force import by accessing the stub
        _ = first_stub.__name__

        # Import should be cached in the stub
        assert first_stub._stub_target is sys.modules["os"]

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

        # First access creates stub
        first_stub = importer.NonExistent
        # Second access should return same cached stub
        second_stub = importer.NonExistent

        assert first_stub is second_stub
        assert isinstance(first_stub, _ImportStub)

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
            eager=True,
        )

        stub = importer.os
        assert isinstance(stub, _ImportStub)
        assert stub._stub_target is sys.modules["os"]

        # Stub should forward attribute access to the module
        assert stub.__name__ == "os"

    def test_object_import(self):
        """
        Test importing object from module.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"path": "os.path"},
            extras_group="test",
            eager=True,
        )

        stub = importer.path
        assert isinstance(stub, _ImportStub)
        assert stub._stub_target is sys.modules["os"].path

        # Stub should forward attribute access
        assert hasattr(stub, "join")

    def test_object_import_missing_module(self):
        """
        Test importing object from missing module creates failed stub.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"TestClass": "non_existent_module.TestClass"},
            extras_group="test",
            eager=True,
        )

        stub = importer.TestClass
        assert isinstance(stub, _ImportStub)
        assert stub._stub_failed is True

        # Using it should raise ImportError
        with pytest.raises(ImportError, match="requires optional dependencies"):
            stub()

    def test_object_import_missing_attribute(self):
        """
        Test importing missing attribute from existing module creates failed stub.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"NonExistentAttr": "os.non_existent_attribute"},
            extras_group="test",
            eager=True,
        )

        stub = importer.NonExistentAttr
        assert isinstance(stub, _ImportStub)
        assert stub._stub_failed is True

        # Using it should raise ImportError
        with pytest.raises(ImportError, match="requires optional dependencies"):
            stub()

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
                # Get stubs
                os_stub = importer.os
                path_stub = importer.path
                # Force import
                os_name = os_stub.__name__
                path_name = path_stub.join
                results.append((os_stub, path_stub, os_name, path_name))
            except (ImportError, AttributeError) as e:
                errors.append(e)

        threads = [threading.Thread(target=access_imports) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # All results should use the same stubs
        assert len(results) == 10
        first_os_stub, first_path_stub, _, _ = results[0]
        for os_stub, path_stub, os_name, path_name in results:
            assert os_stub is first_os_stub
            assert path_stub is first_path_stub
            assert os_name == "os"
            assert callable(path_name)

    def test_parse_module_path_simple(self):
        """
        Test _parse_module_path for simple module imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"os": "os"},
            extras_group="test",
        )

        module_path = importer._parse_module_path("os")
        assert module_path == "os"

    def test_parse_module_path_object(self):
        """
        Test _parse_module_path for object imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"path": "os.path"},
            extras_group="test",
        )

        module_path = importer._parse_module_path("os.path")
        assert module_path == "os"

    def test_parse_module_path_nested_object(self):
        """
        Test _parse_module_path for nested object imports.

        ### WRITTEN BY AI ###
        """
        importer = ExtrasImporter(
            {"join": "os.path.join"},
            extras_group="test",
        )

        module_path = importer._parse_module_path("os.path.join")
        assert module_path == "os.path"


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
            eager=False,  # vllm uses lazy loading
        )

        # Should have is_available property
        has_vllm = importer.is_available
        assert isinstance(has_vllm, bool)

        # Access should return stub (not raise)
        sampling_params = importer.SamplingParams
        assert isinstance(sampling_params, _ImportStub)

        if has_vllm:
            # If vllm is installed, using the stub should work
            # Access an attribute to trigger import
            assert hasattr(sampling_params, "__name__")
            assert sampling_params._stub_target is not None
        else:
            # If vllm is not installed, using stub should raise with helpful message
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
            eager=True,  # audio uses eager loading
        )

        has_audio = importer.is_available
        assert isinstance(has_audio, bool)

        audio_decoder = importer.AudioDecoder
        assert isinstance(audio_decoder, _ImportStub)

        if has_audio:
            # Import should have succeeded
            assert audio_decoder._stub_target is not None
            assert not audio_decoder._stub_failed
        else:
            # Import should have failed
            assert audio_decoder._stub_failed
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
            eager=True,  # vision uses eager loading
        )

        has_vision = importer.is_available
        assert isinstance(has_vision, bool)

        pil_image = importer.PILImage
        assert isinstance(pil_image, _ImportStub)

        if has_vision:
            # Import should have succeeded
            assert pil_image._stub_target is not None
            assert not pil_image._stub_failed
        else:
            # Import should have failed
            assert pil_image._stub_failed
            with pytest.raises(ImportError, match="pip install guidellm\\[vision\\]"):
                pil_image.open("test.jpg")
