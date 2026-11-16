from copy import deepcopy

import pytest
from datasets import Dataset

from guidellm.data.finalizers import FinalizerRegistry
from guidellm.data.finalizers.conversation_graph import turns_from_mapped_items
from guidellm.data.finalizers.generative import GenerativeRequestFinalizer
from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.schemas.data import DataFinalizerArgs
from guidellm.schemas.data.finalizers import GenerativeRequestFinalizerArgs
from guidellm.schemas.data.preprocessors import GenerativeColumnMapperArgs
from guidellm.utils.arg_string import ArgStringParser


def text_turn(text, count):
    return {"text_column": [text], "output_tokens_count_column": [count]}


@pytest.mark.regression
@pytest.mark.parametrize("length", [1, 14, 399, 400])
@pytest.mark.parametrize("padding", [("", 0), (None, None)])
def test_indexed_400_slot_pipeline(length, padding):
    """Preserve every real turn and full-history edge. ## WRITTEN BY AI ##"""
    row = {}
    for index in range(400):
        row[f"prompt_{index}"] = f"prompt {index}" if index < length else padding[0]
        row[f"output_tokens_count_{index}"] = 64000 if index < length else padding[1]
    source = Dataset.from_list([row])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([source])
    mapped = mapper([{"dataset": source[0]}])
    original = deepcopy(mapped)
    default = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())(mapped)
    fixed = GenerativeRequestFinalizer(
        GenerativeRequestFinalizerArgs(trim_text_padding=True)
    )(mapped)
    assert len(default.nodes) == 400
    assert len(fixed.nodes) == length
    assert len(fixed.edges) == length - 1
    assert mapped == original
    for index in range(length):
        node = fixed.nodes[f"turn_{index}"]
        assert node.request.columns["text_column"] == [f"prompt {index}"]
        assert node.request.output_metrics.text_tokens == 64000
    for index, edge in enumerate(fixed.edges):
        assert edge.source_node_id == f"turn_{index}"
        assert edge.target_node_id == f"turn_{index + 1}"
        assert edge.history_context == "full"


@pytest.mark.regression
@pytest.mark.parametrize("text", [None, ""])
@pytest.mark.parametrize("count", [None, 0])
def test_null_and_zero_padding(text, count):
    """Accept only the declared padding contract. ## WRITTEN BY AI ##"""
    graph = turns_from_mapped_items(
        [text_turn("hello", 1), text_turn(text, count)], trim_text_padding=True
    )
    assert len(graph.turns) == 1


@pytest.mark.regression
@pytest.mark.parametrize(
    "item",
    [
        text_turn("", 1),
        text_turn(" ", 0),
        {"text_column": [""]},
        {**text_turn("", 0), "image_column": [{"image": "x"}]},
        {**text_turn("", 0), "tools_column": [{"type": "function"}]},
        {**text_turn("", 0), "raw_messages_column": [[{"role": "user"}]]},
        {**text_turn("", 0), "relative_timestamp_column": [1.0]},
        {"text_column": ["", ""], "output_tokens_count_column": [0]},
        text_turn("", False),
        text_turn("", 0.0),
        text_turn("", "0"),
        {"text_column": "", "output_tokens_count_column": [0]},
        {"text_column": [""], "output_tokens_count_column": 0},
        {"text_column": [], "output_tokens_count_column": []},
    ],
)
def test_other_column_shapes_are_not_trimmed(item):
    """Keep ambiguous or semantically meaningful turns. ## WRITTEN BY AI ##"""
    graph = turns_from_mapped_items(
        [text_turn("hello", 1), item], trim_text_padding=True
    )
    assert len(graph.turns) == 2


@pytest.mark.regression
@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([text_turn("", 0)], "All-padding"),
        ([text_turn("a", 1), text_turn("", 0), text_turn("b", 1)], "before"),
    ],
)
def test_bad_padding_layout_rejected(items, message):
    """Avoid silently changing malformed conversation order. ## WRITTEN BY AI ##"""
    assert len(turns_from_mapped_items(items).turns) == len(items)
    with pytest.raises(ValueError, match=message):
        turns_from_mapped_items(items, trim_text_padding=True)


@pytest.mark.regression
def test_explicit_graph_is_unchanged():
    """An explicit graph may contain meaningful empty turns. ## WRITTEN BY AI ##"""
    payload = {
        "graph_id": "explicit",
        "turns": [{"node_id": "empty", "columns": text_turn("", 0)}],
    }
    items = [{"conversation_turns_column": [payload]}]
    assert turns_from_mapped_items(items, trim_text_padding=True) == (
        turns_from_mapped_items(items)
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    ("argument", "expected_turns"),
    [
        ("kind=generative", 2),
        ("kind=generative,trim_text_padding=false", 2),
        ("kind=generative,trim_text_padding=true", 1),
    ],
)
def test_finalizer_arg_string_controls_padding(argument, expected_turns):
    """Wire the option through the registered finalizer. ## WRITTEN BY AI ##"""
    config = DataFinalizerArgs.model_validate(ArgStringParser().decode(argument))
    finalizer = FinalizerRegistry.create(config)
    graph = finalizer([text_turn("hello", 32), text_turn("", 0)])
    assert len(graph.nodes) == expected_turns
