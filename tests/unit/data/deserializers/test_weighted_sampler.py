from collections import Counter
from itertools import islice
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset, DatasetInfo, Features, IterableDataset, Value

from guidellm.data.deserializers import (
    DataNotSupportedError,
    DatasetDeserializerFactory,
    WeightedSamplerDataset,
    WeightedSamplerDatasetDeserializer,
)
from guidellm.data.finalizers.generative import GenerativeRequestFinalizer
from guidellm.data.loaders.torch import DatasetsIterator, TorchDataLoader
from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.schemas.data import (
    DataArgs,
    GenerativeColumnMapperArgs,
    GenerativeRequestFinalizerArgs,
    InMemoryDictDataArgs,
    SyntheticTextDataArgs,
    TorchDataLoaderArgs,
    WeightedSamplerDataArgs,
)
from guidellm.utils import arg_string


def sampler(weights=(10, 10, 20, 0), replacement=True, random_seed=42):
    config = WeightedSamplerDataArgs(
        source=InMemoryDictDataArgs(
            data={"weights": list(weights), "model": list("abcd")[: len(weights)]}
        ),
        replacement=replacement,
    )
    return DatasetDeserializerFactory.deserialize(config, Mock(), random_seed)


@pytest.mark.smoke
@pytest.mark.parametrize("options", ["", "replacement=false,", "weight_column=weight,"])
def test_csv_cli_configuration(tmp_path, options):
    """Load documented dotted CLI arguments through both registries without a tokenizer.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "models.csv"
    column = "weight" if "weight_column" in options else "weights"
    path.write_text(f"{column},model\n10,model-a\n10,model-b\n20,model-c\n")
    config = DataArgs.model_validate(
        arg_string.loads(
            f"kind=weighted_sampler,{options}source.kind=csv_file,source.path={path}"
        )
    )
    processor_factory = Mock(side_effect=AssertionError("Tokenizer not needed"))
    dataset = DatasetDeserializerFactory.deserialize(config, processor_factory, 17)
    assert isinstance(dataset, WeightedSamplerDataset)
    assert len(list(islice(dataset, 40))) == 40
    processor_factory.assert_not_called()


@pytest.mark.sanity
def test_replacement_frequencies():
    """Independent draws approximate the requested ratio and exclude zero weights.

    ## WRITTEN BY AI ##
    """
    counts = Counter(row["model"] for row in islice(sampler(), 10000))
    assert set(counts) == {"a", "b", "c"}
    for model, probability in (("a", 0.25), ("b", 0.25), ("c", 0.5)):
        assert counts[model] / 10000 == pytest.approx(probability, abs=0.03)


@pytest.mark.sanity
@pytest.mark.parametrize("weights", [(10, 10, 20, 0), (10.0, 10.0, 20.0, 0.0)])
def test_without_replacement_cycles(weights):
    """Each reduced four-entry cycle preserves exact counts, including repeated rows.

    ## WRITTEN BY AI ##
    """
    dataset = sampler(weights=weights, replacement=False)
    rows = list(islice(dataset, 80))
    cycles = [tuple(row["model"] for row in rows[i : i + 4]) for i in range(0, 80, 4)]
    assert all(Counter(cycle) == {"a": 1, "b": 1, "c": 2} for cycle in cycles)
    assert len(set(cycles)) > 1


@pytest.mark.regression
@pytest.mark.parametrize("replacement", [True, False])
def test_seed_epoch_and_iterator_isolation(replacement):
    """Equivalent seed/epoch inputs reproduce sequences even with interleaved iterators.

    ## WRITTEN BY AI ##
    """
    dataset = sampler(replacement=replacement)
    reference = sampler(replacement=replacement)
    first_epoch = list(islice(dataset, 80))
    assert first_epoch == list(islice(reference, 80))
    dataset.set_epoch(3)
    reference.set_epoch(3)
    active = iter(dataset)
    prefix = list(islice(active, 7))
    list(islice(dataset, 17))
    result = prefix + list(islice(active, 73))
    assert result == list(islice(reference, 80))
    assert result != first_epoch
    dataset.set_epoch(0)
    assert list(islice(dataset, 80)) == first_epoch


@pytest.mark.sanity
@pytest.mark.parametrize("replacement", [True, False])
def test_single_positive_row(replacement):
    """A single positive row remains available indefinitely in either mode.

    ## WRITTEN BY AI ##
    """
    rows = list(islice(sampler((0, 50, 0), replacement), 100))
    assert rows == [{"weights": 50, "model": "b"}] * 100


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"model": ["a"]}, "Missing weight column 'frequency'"),
        ({"frequency": []}, "'frequency'.*empty"),
        ({"frequency": [0, 0]}, "'frequency'.*positive"),
        *[
            ({"frequency": [value]}, "'frequency' at row 0")
            for value in (
                -1,
                0.5,
                None,
                True,
                False,
                "10",
                "bad",
                float("nan"),
                float("inf"),
                float("-inf"),
            )
        ],
    ],
)
def test_invalid_weights(data, message):
    """Invalid weights fail with the configured column and offending row when relevant.

    ## WRITTEN BY AI ##
    """
    config = WeightedSamplerDataArgs(
        source=InMemoryDictDataArgs(data=data), weight_column="frequency"
    )
    with pytest.raises(DataNotSupportedError, match=message):
        DatasetDeserializerFactory.deserialize(config, Mock(), 42)


@pytest.mark.regression
def test_streaming_source_rejected_without_consumption():
    """Reject iterable sources eagerly without reading rows or invoking the processor.

    ## WRITTEN BY AI ##
    """

    def fail_on_iteration():
        raise AssertionError("Streaming source must not be consumed")
        yield {}  # pragma: no cover

    source = IterableDataset.from_generator(fail_on_iteration)
    config = WeightedSamplerDataArgs(source=InMemoryDictDataArgs(data={}))
    processor_factory = Mock()
    with (
        patch.object(DatasetDeserializerFactory, "deserialize", return_value=source),
        pytest.raises(DataNotSupportedError, match="'weights'.*finite, indexable"),
    ):
        WeightedSamplerDatasetDeserializer()(config, processor_factory, 42)
    processor_factory.assert_not_called()


@pytest.mark.regression
def test_metadata_and_source_forwarding():
    """Retain rows, features, split, and dataset name while forwarding source arguments.

    ## WRITTEN BY AI ##
    """
    source = Dataset.from_dict(
        {"weights": [10, 20], "model": ["a", "b"], "extra": [[1], [2, 3]]},
        info=DatasetInfo(
            dataset_name="models",
            description="Model choices",
            features=Features(
                {
                    "weights": Value("int32"),
                    "model": Value("string"),
                    "extra": [Value("int16")],
                }
            ),
        ),
        split="train",
    )
    config = WeightedSamplerDataArgs(source=InMemoryDictDataArgs(data={}))
    processor_factory = Mock()
    with patch.object(
        DatasetDeserializerFactory, "deserialize", return_value=source
    ) as deserialize:
        dataset = WeightedSamplerDatasetDeserializer()(config, processor_factory, 17)
    deserialize.assert_called_once_with(
        config=config.source, processor_factory=processor_factory, random_seed=17
    )
    processor_factory.assert_not_called()
    assert dataset.features == source.features
    assert dataset.info.dataset_name == "models"
    assert dataset.info.description == source.info.description
    assert dataset.split == source.split
    assert all(row in list(source) for row in islice(dataset, 20))
    assert GenerativeColumnMapper.datasets_mappings(
        [dataset], {"model_column": "models.model"}
    ) == {("model_column", 0): [(0, "model")]}


@pytest.mark.regression
@pytest.mark.parametrize("replacement", [True, False])
def test_prompt_composition_and_worker_partitions(replacement):
    """Combine independent prompts and models through the mapper and loader pipeline.

    ## WRITTEN BY AI ##
    """
    iterator = DatasetsIterator(
        datasets=[
            Dataset.from_dict({"prompt": [f"prompt-{i}" for i in range(40)]}),
            sampler(replacement=replacement),
        ],
        data_samples=0,
        preprocessors=[GenerativeColumnMapper(GenerativeColumnMapperArgs())],
        finalizer=GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs()),
    )
    graphs = list(iterator.generator(epoch=2))
    requests = [next(iter(graph.nodes.values())).request for graph in graphs]
    assert len(requests) == 40
    assert all(
        request.columns["text_column"] == [f"prompt-{i}"]
        for i, request in enumerate(requests)
    )
    assert all(request.columns["model_column"][0] in "abc" for request in requests)
    for offset in range(3):
        partition = list(iterator.generator(modulus=3, offset=offset, epoch=2))
        assert [
            next(iter(graph.nodes.values())).request.columns for graph in partition
        ] == [
            request.columns
            for i, request in enumerate(requests)
            if (i + 1) % 3 == offset
        ]


@pytest.mark.regression
@pytest.mark.parametrize("num_workers", [1, 2])
def test_torch_workers_preserve_sampling(num_workers):
    """Worker processes retain all prompt/model pairs from the shared sampling sequence.

    ## WRITTEN BY AI ##
    """
    loader = TorchDataLoader(
        config=TorchDataLoaderArgs(samples=0, num_workers=num_workers),
        datasets=[
            Dataset.from_dict({"prompt": list(range(40))}),
            sampler(replacement=False),
        ],
        preprocessors=[],
        finalizer=list,
    )
    rows = sorted(loader, key=lambda row: row[0]["dataset"]["prompt"])
    expected = list(islice(sampler(replacement=False), 40))
    assert [row[0]["dataset"]["prompt"] for row in rows] == list(range(40))
    assert [row[1]["dataset"] for row in rows] == expected


@pytest.mark.regression
def test_synthetic_prompt_composition():
    """The documented synthetic prompt source combines with sampled model fields.

    ## WRITTEN BY AI ##
    """
    processor = Mock()
    processor.encode.side_effect = lambda text: list(range(len(text.split())))
    processor.decode.side_effect = lambda tokens, **kwargs: " ".join(
        f"token-{token}" for token in tokens
    )
    prompts = DatasetDeserializerFactory.deserialize(
        SyntheticTextDataArgs(prompt_tokens=8, output_tokens=4), lambda: processor, 42
    )
    iterator = DatasetsIterator(
        datasets=[prompts, sampler(replacement=False)],
        data_samples=8,
        preprocessors=[GenerativeColumnMapper(GenerativeColumnMapperArgs())],
        finalizer=GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs()),
    )
    requests = [next(iter(graph.nodes.values())).request for graph in iterator]
    assert len(requests) == 8
    assert all(request.columns["text_column"][0] for request in requests)
    assert all(
        request.columns["output_tokens_count_column"] == [4] for request in requests
    )
    assert Counter(request.columns["model_column"][0] for request in requests) == {
        "a": 2,
        "b": 2,
        "c": 4,
    }
