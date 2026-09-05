"""
Unit tests for guidellm.data.loaders.torch module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import TorchDataLoader
from guidellm.schemas.data import TorchDataLoaderArgs


@pytest.mark.regression
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("samples", [-1, 0, 1])
def test_loader_allows_workers_without_assigned_rows(num_workers, samples):
    """A small dataset must load even when some workers receive no rows.

    ## WRITTEN BY AI ##
    """
    loader = TorchDataLoader(
        config=TorchDataLoaderArgs(samples=samples, num_workers=num_workers),
        datasets=[Dataset.from_dict({"text": ["hello"]})],
        preprocessors=[],
        finalizer=list,
    )

    assert list(loader) == [[{"dataset": {"text": "hello"}}]]


@pytest.mark.regression
def test_loader_rejects_rows_with_only_empty_results():
    """Rows assigned to a worker must still produce usable results.

    ## WRITTEN BY AI ##
    """
    loader = TorchDataLoader(
        config=TorchDataLoaderArgs(samples=0, num_workers=0),
        datasets=[Dataset.from_dict({"text": ["hello"]})],
        preprocessors=[],
        finalizer=lambda _: [],
    )

    with pytest.raises(ValueError, match="processed 1 rows but yielded zero results"):
        list(loader)


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
    def test_default_shuffle_is_false(self):
        """TorchDataLoaderArgs defaults shuffle to False.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.shuffle is False

    @pytest.mark.sanity
    def test_custom_values(self):
        """TorchDataLoaderArgs accepts custom field values.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs(samples=500, num_workers=4, shuffle=True)
        assert args.samples == 500
        assert args.num_workers == 4
        assert args.shuffle is True

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
