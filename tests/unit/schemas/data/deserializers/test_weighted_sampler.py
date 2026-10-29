import pytest
from pydantic import ValidationError

from guidellm.schemas.data import (
    DataArgs,
    FileDataArgs,
    InMemoryDictDataArgs,
    WeightedSamplerDataArgs,
)


@pytest.mark.smoke
def test_registry_and_defaults():
    """Resolve both wrapper and nested source through the registry.

    ## WRITTEN BY AI ##
    """
    config = DataArgs.model_validate(
        {
            "kind": "weighted_sampler",
            "source": {"kind": "in_memory_dict", "data": {"weights": [1]}},
        }
    )
    assert isinstance(config, WeightedSamplerDataArgs)
    assert isinstance(config.source, InMemoryDictDataArgs)
    assert config.weight_column == "weights"
    assert config.replacement is True
    assert DataArgs.model_validate(config.model_dump()) == config


@pytest.mark.sanity
def test_overrides_and_loading_options():
    """Outer loading options deep-merge with precedence over source options.

    ## WRITTEN BY AI ##
    """
    config = WeightedSamplerDataArgs.model_validate(
        {
            "weight_column": "frequency",
            "replacement": False,
            "load_kwargs": {"split": "test", "nested": {"override": 2}},
            "source": {
                "kind": "csv_file",
                "path": "models.csv",
                "load_kwargs": {
                    "split": "train",
                    "delimiter": ";",
                    "nested": {"keep": 1, "override": 1},
                },
            },
        }
    )
    assert isinstance(config.source, FileDataArgs)
    assert config.weight_column == "frequency"
    assert config.replacement is False
    assert config.source.load_kwargs == {
        "split": "test",
        "delimiter": ";",
        "nested": {"keep": 1, "override": 2},
    }


@pytest.mark.sanity
@pytest.mark.parametrize(
    "values",
    [
        {},
        {"source": None},
        {"source": {}},
        {"source": {"kind": "csv_file"}},
        {"source": {"kind": "unknown_source"}},
        {
            "source": {"kind": "in_memory_dict", "data": {"weights": [1]}},
            "weight_column": "",
        },
    ],
)
def test_invalid_configuration(values):
    """Require a valid nested source and a non-empty weight column.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        WeightedSamplerDataArgs.model_validate(values)
