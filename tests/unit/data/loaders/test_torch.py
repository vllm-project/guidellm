"""
Unit tests for guidellm.data.loaders.torch module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import DatasetsIterator, TorchDataLoader
from guidellm.schemas.data import TorchDataLoaderArgs


@pytest.mark.regression
@pytest.mark.parametrize("num_workers", [1, 2])
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
    iterator = DatasetsIterator(
        data_samples=0,
        datasets=[Dataset.from_dict({"text": ["hello"]})],
        preprocessors=[],
        finalizer=lambda _: [],
    )

    with pytest.raises(ValueError, match="processed 1 rows but yielded zero results"):
        list(iterator)


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


class _EpochDataset:
    """Iterable dataset whose rows depend on the epoch, like the synthetic dataset.

    ## WRITTEN BY AI ##
    """

    def __init__(self, fail_from_epoch: int | None = None):
        self.epoch = 0
        self.fail_from_epoch = fail_from_epoch

    def set_epoch(self, epoch: int):
        if self.fail_from_epoch is not None and epoch >= self.fail_from_epoch:
            raise RuntimeError(f"set_epoch failed at epoch {epoch}")
        self.epoch = epoch

    def __iter__(self):
        row = 0
        while True:
            yield {"text": f"epoch{self.epoch}-row{row}"}
            row += 1


def _text_finalizer(row):
    return row[0]["dataset"]["text"]


def _epoch_iterator(samples: int, **dataset_kwargs) -> DatasetsIterator:
    return DatasetsIterator(
        data_samples=samples,
        datasets=[_EpochDataset(**dataset_kwargs)],
        preprocessors=[],
        finalizer=_text_finalizer,
    )


def _take(iterator, count: int) -> list[str]:
    return [row for _, row in zip(range(count), iterator, strict=False)]


def _advance_epoch_and_take(iterator, epoch: int) -> list[str]:
    iterator.set_epoch(epoch)
    return _take(iterator, 1)


@pytest.mark.regression
@pytest.mark.parametrize("samples", [1, 3])
def test_samples_cap_yields_fresh_rows_every_epoch(samples):
    """A positive samples cap must still draw new rows for each epoch.

    ## WRITTEN BY AI ##
    """
    iterator = _epoch_iterator(samples)
    passes = []
    for epoch in range(3):
        iterator.set_epoch(epoch)
        passes.append(list(iterator))

    assert all(len(rows) == samples for rows in passes)
    assert passes[0] == [f"epoch0-row{i}" for i in range(samples)]
    assert len({tuple(rows) for rows in passes}) == 3


@pytest.mark.regression
def test_samples_cap_epochs_are_reproducible():
    """The same epoch must give the same rows across runs and repeated passes.

    ## WRITTEN BY AI ##
    """
    first, second = _epoch_iterator(3), _epoch_iterator(3)
    for epoch in (0, 1, 2, 1):
        first.set_epoch(epoch)
        second.set_epoch(epoch)
        assert list(first) == list(second)
        assert list(first) == [f"epoch{epoch}-row{i}" for i in range(3)]


@pytest.mark.regression
def test_loader_samples_cap_yields_fresh_rows_for_each_pass():
    """Each pass over the loader (one per strategy) must see different rows.

    ## WRITTEN BY AI ##
    """
    loader = TorchDataLoader(
        config=TorchDataLoaderArgs(samples=3, num_workers=1),
        datasets=[_EpochDataset()],
        preprocessors=[],
        finalizer=_text_finalizer,
    )

    passes = [list(loader) for _ in range(3)]

    assert [rows[0] for rows in passes] == [
        "epoch0-row0",
        "epoch1-row0",
        "epoch2-row0",
    ]


@pytest.mark.regression
@pytest.mark.parametrize("samples", [-1, 3])
def test_set_epoch_failure_is_not_swallowed(samples):
    """A dataset whose set_epoch fails must raise instead of replaying epoch 0.

    ## WRITTEN BY AI ##
    """
    iterator = _epoch_iterator(samples, fail_from_epoch=1)
    iterator.set_epoch(0)
    assert _take(iterator, 1) == ["epoch0-row0"]

    with pytest.raises(RuntimeError, match="set_epoch failed at epoch 1"):
        _advance_epoch_and_take(iterator, 1)
