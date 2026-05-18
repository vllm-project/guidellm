"""
Unit tests for guidellm.data.loaders.loader module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

import guidellm.data.loaders  # noqa: F401 — ensures TorchDataLoader is registered

# Import to ensure deserializer registry is populated
from guidellm.data.deserializers.huggingface import HuggingFaceDataArgs
from guidellm.data.finalizers.generative import GenerativeRequestFinalizerConfig
from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import TorchDataLoaderArgs
from guidellm.data.schemas.entrypoints import DataEntrypointArgs


class TestDataLoaderRegistry:
    """Tests for DataLoaderRegistry factory.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_pytorch_registered(self):
        """'pytorch' kind is registered in DataLoaderRegistry.

        ### WRITTEN BY AI ###
        """
        loader_cls = DataLoaderRegistry.get_registered_object("pytorch")
        assert loader_cls is not None

    @pytest.mark.sanity
    def test_unknown_kind_raises(self):
        """DataLoaderRegistry.create raises ValueError for unknown kind.

        ### WRITTEN BY AI ###
        """
        config = DataEntrypointArgs(
            loader=TorchDataLoaderArgs(),
            data=[HuggingFaceDataArgs(source="x")],
            preprocessors=[],
            finalizer=GenerativeRequestFinalizerConfig(),
        )
        # Monkey-patch kind to something not registered
        object.__setattr__(config.loader, "kind", "nonexistent_loader_kind")

        with pytest.raises(ValueError, match="not registered"):
            DataLoaderRegistry.create(config)

    @pytest.mark.smoke
    def test_registry_returns_class(self):
        """DataLoaderRegistry.get_registered_object returns a class, not an instance.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.loaders.torch import TorchDataLoader

        loader_cls = DataLoaderRegistry.get_registered_object("pytorch")
        assert loader_cls is TorchDataLoader
