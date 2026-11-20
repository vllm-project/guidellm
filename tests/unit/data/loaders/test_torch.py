"""
Unit tests for guidellm.data.loaders.torch module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.loaders.loader import DataLoaderRegistry
from guidellm.data.loaders.torch import DatasetsIterator, TorchDataLoader
from guidellm.logger import logger
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


ROWS = [f"row{index:02d}" for index in range(20)]


def _shuffle_iterator(
    datasets: list, random_seed: int = 7, samples: int = 0, shuffle: bool = True
) -> DatasetsIterator:
    return DatasetsIterator(
        datasets=datasets,
        data_samples=samples,
        preprocessors=[],
        finalizer=list,
        shuffle=shuffle,
        random_seed=random_seed,
    )


def _column(rows: list, index: int = 0, name: str = "text") -> list:
    return [row[index]["dataset"][name] for row in rows]


def _epoch_rows(iterator: DatasetsIterator, epoch: int) -> list:
    iterator.set_epoch(epoch)
    return list(iterator)


@pytest.mark.regression
def test_loader_shuffle_gives_every_row_in_a_new_order_each_pass():
    """``shuffle=True`` must load, keep every row once and reorder each pass.

    ## WRITTEN BY AI ##
    """
    loader = TorchDataLoader(
        config=TorchDataLoaderArgs(shuffle=True),
        datasets=[Dataset.from_dict({"text": ROWS})],
        preprocessors=[],
        finalizer=list,
        random_seed=7,
    )

    first, second = _column(list(loader)), _column(list(loader))

    assert sorted(first) == ROWS
    assert sorted(second) == ROWS
    assert first != ROWS
    assert second != first


@pytest.mark.sanity
def test_loader_shuffle_order_follows_the_loader_seed():
    """Loaders with the same ``random_seed`` must agree; another seed must not.

    ## WRITTEN BY AI ##
    """

    def first_pass(random_seed: int) -> list:
        loader = TorchDataLoader(
            config=TorchDataLoaderArgs(shuffle=True),
            datasets=[Dataset.from_dict({"text": ROWS})],
            preprocessors=[],
            finalizer=list,
            random_seed=random_seed,
        )
        return _column(list(loader))

    seven = first_pass(7)

    assert first_pass(7) == seven
    assert first_pass(8) != seven


@pytest.mark.sanity
def test_shuffle_order_follows_the_seed():
    """The same seed must give the same order; another seed another order.

    ## WRITTEN BY AI ##
    """
    first = _shuffle_iterator([Dataset.from_dict({"text": ROWS})], random_seed=7)
    again = _shuffle_iterator([Dataset.from_dict({"text": ROWS})], random_seed=7)
    other = _shuffle_iterator([Dataset.from_dict({"text": ROWS})], random_seed=8)

    for epoch in (0, 1):
        assert _epoch_rows(first, epoch) == _epoch_rows(again, epoch)
    assert _column(_epoch_rows(first, 0)) != _column(_epoch_rows(other, 0))


@pytest.mark.sanity
def test_shuffle_changes_the_order_every_epoch():
    """Each epoch must get its own order, and repeating an epoch repeats it.

    ## WRITTEN BY AI ##
    """
    iterator = _shuffle_iterator([Dataset.from_dict({"text": ROWS})])

    epoch0 = _column(_epoch_rows(iterator, 0))
    epoch1 = _column(_epoch_rows(iterator, 1))

    assert sorted(epoch0) == ROWS
    assert sorted(epoch1) == ROWS
    assert epoch0 != epoch1
    assert _column(_epoch_rows(iterator, 0)) == epoch0


@pytest.mark.sanity
@pytest.mark.parametrize("label_count", [20, 12])
def test_shuffle_keeps_combined_rows_paired(label_count):
    """Shuffling must reorder whole rows, never pair items that were not paired.

    ## WRITTEN BY AI ##
    """
    labels = [f"label{index:02d}" for index in range(label_count)]
    iterator = _shuffle_iterator(
        [Dataset.from_dict({"text": ROWS}), Dataset.from_dict({"label": labels})]
    )

    rows = _epoch_rows(iterator, 0)
    texts, paired = _column(rows), _column(rows, index=1, name="label")

    assert sorted(texts) == ROWS[:label_count]
    assert texts != ROWS[:label_count]
    assert [label.replace("label", "row") for label in paired] == texts


@pytest.mark.sanity
@pytest.mark.parametrize("with_map_dataset", [True, False])
def test_shuffle_is_skipped_unless_every_dataset_is_map_style(with_map_dataset):
    """An iterable source can't be reordered, so nothing is shuffled and a
    warning says so.

    ## WRITTEN BY AI ##
    """
    labels = [f"label{index:02d}" for index in range(20)]
    datasets = [Dataset.from_dict({"label": labels}).to_iterable_dataset()]
    if with_map_dataset:
        datasets.insert(0, Dataset.from_dict({"text": ROWS}))
    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        iterator = _shuffle_iterator(datasets)
    finally:
        logger.remove(sink)

    rows = _epoch_rows(iterator, 0)

    assert _column(rows, index=len(datasets) - 1, name="label") == labels
    if with_map_dataset:
        assert _column(rows) == ROWS
    assert len(warnings) == 1
    assert "shuffle" in warnings[0]


@pytest.mark.regression
def test_shuffle_reads_no_rows_before_the_first_epoch(monkeypatch):
    """Without a sample cap, building the iterator must not shuffle anything.

    ## WRITTEN BY AI ##
    """
    selected: list[int] = []
    real_select = Dataset.select

    def counting_select(self, *args, **kwargs):
        selected.append(len(self))
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "select", counting_select)
    iterator = _shuffle_iterator([Dataset.from_dict({"text": ROWS})], samples=-1)

    assert selected == []
    assert sorted(_column(_epoch_rows(iterator, 0))) == ROWS


@pytest.mark.sanity
def test_shuffle_with_samples_draws_the_rows_once():
    """With ``samples``, a seeded sample is drawn once and reused every epoch.

    ## WRITTEN BY AI ##
    """
    iterator = _shuffle_iterator([Dataset.from_dict({"text": ROWS})], samples=5)

    epoch0 = _column(_epoch_rows(iterator, 0))

    assert epoch0 != ROWS[:5]
    assert len(set(epoch0)) == 5
    assert set(epoch0) <= set(ROWS)
    assert _column(_epoch_rows(iterator, 1)) == epoch0


@pytest.mark.regression
@pytest.mark.parametrize("texts", [[], ["only"], ROWS])
def test_shuffle_off_keeps_the_dataset_order(texts):
    """Without shuffle, every epoch must read the rows in dataset order.

    ## WRITTEN BY AI ##
    """
    iterator = _shuffle_iterator([Dataset.from_dict({"text": texts})], shuffle=False)

    for epoch in (0, 1):
        assert _column(_epoch_rows(iterator, epoch)) == texts


@pytest.mark.regression
@pytest.mark.parametrize("texts", [[], ["only"]])
def test_shuffle_of_empty_or_single_row_dataset(texts):
    """Shuffling zero or one row must give the same rows as not shuffling.

    ## WRITTEN BY AI ##
    """
    iterator = _shuffle_iterator([Dataset.from_dict({"text": texts})])

    assert _column(_epoch_rows(iterator, 0)) == texts


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
