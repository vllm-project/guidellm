"""
Unit tests for guidellm.data.loaders.loader module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import Dataset

import guidellm.data.loaders  # noqa: F401 — ensures TorchDataLoader is registered

# Import to ensure deserializer registry is populated
from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import TorchDataLoaderArgs


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
        config = TorchDataLoaderArgs()
        # Monkey-patch kind to something not registered
        object.__setattr__(config, "kind", "nonexistent_loader_kind")

        # Create minimal mocks for required parameters
        mock_dataset = MagicMock(spec=Dataset)
        mock_preprocessor = MagicMock()
        mock_finalizer = MagicMock()

        with pytest.raises(ValueError, match="not registered"):
            DataLoaderRegistry.create(
                config=config,
                datasets=[mock_dataset],
                preprocessors=[mock_preprocessor],
                finalizer=mock_finalizer,
                random_seed=42,
            )

    @pytest.mark.smoke
    def test_registry_returns_class(self):
        """DataLoaderRegistry.get_registered_object returns a class, not an instance.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.loaders.torch import TorchDataLoader

        loader_cls = DataLoaderRegistry.get_registered_object("pytorch")
        assert loader_cls is TorchDataLoader
