"""
Unit tests for guidellm.data.loaders.torch module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import TorchDataLoader, TorchDataLoaderArgs


class TestTorchDataLoaderArgs:
    """Tests for TorchDataLoaderArgs schema.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_default_kind(self):
        """TorchDataLoaderArgs defaults kind to 'pytorch'.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.kind == "pytorch"

    @pytest.mark.smoke
    def test_default_samples(self):
        """TorchDataLoaderArgs defaults samples to -1 (unlimited).

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.samples == -1

    @pytest.mark.smoke
    def test_default_num_workers(self):
        """TorchDataLoaderArgs defaults num_workers to 1.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.num_workers == 1

    @pytest.mark.smoke
    def test_default_sampler_is_none(self):
        """TorchDataLoaderArgs defaults sampler to None.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.sampler is None

    @pytest.mark.sanity
    def test_custom_values(self):
        """TorchDataLoaderArgs accepts custom field values.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs(samples=500, num_workers=4, sampler="shuffle")
        assert args.samples == 500
        assert args.num_workers == 4
        assert args.sampler == "shuffle"

    @pytest.mark.sanity
    def test_registered_in_registry(self):
        """TorchDataLoaderArgs is registered in DataLoaderRegistry as 'pytorch'.

        ### WRITTEN BY AI ###
        """
        loader_cls = DataLoaderRegistry.get_registered_object("pytorch")
        assert loader_cls is TorchDataLoader

    @pytest.mark.regression
    def test_serialization(self):
        """TorchDataLoaderArgs serializes correctly.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs(samples=100)
        dumped = args.model_dump()
        assert dumped["kind"] == "pytorch"
        assert dumped["samples"] == 100
        assert dumped["num_workers"] == 1
