"""Tests for generative dataset column mapping."""

from typing import Any

import pytest
from datasets import Dataset

from guidellm.data.preprocessors.mappers import (
    GenerativeColumnMapper,
    GenerativeColumnMapperArgs,
)


def _structured_prompt(
    text: str = "A structured prompt",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured text prompt for mapper tests."""
    return {
        "type": "text",
        "text": text,
        **(payload or {}),
    }


@pytest.mark.regression
def test_generative_mapper_preserves_structured_prompt_objects():
    """Custom dataset prompt dictionaries pass through the mapper unchanged.

    ## WRITTEN BY AI ##
    """
    prompt = _structured_prompt()
    dataset = Dataset.from_list([{"prompt": prompt}])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    result = mapper([{"dataset": dataset[0]}])

    assert result == [{"text_column": [prompt]}]


@pytest.mark.smoke
def test_generative_mapper_keeps_plain_text_behavior():
    """Plain prompts and token-count columns retain their existing mapping.

    ## WRITTEN BY AI ##
    """
    dataset = Dataset.from_list([{"prompt": "A plain prompt", "input_tokens_count": 3}])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    result = mapper([{"dataset": dataset[0]}])

    assert result == [
        {
            "prompt_tokens_count_column": [3],
            "text_column": ["A plain prompt"],
        }
    ]


@pytest.mark.sanity
def test_generative_mapper_preserves_structured_prompt_with_explicit_mapping():
    """Structured prompts work when the source column has a custom name.

    ## WRITTEN BY AI ##
    """
    prompt = _structured_prompt(payload={"request_context": "custom-column"})
    dataset = Dataset.from_list([{"content_payload": prompt}])
    mapper = GenerativeColumnMapper(
        GenerativeColumnMapperArgs(column_mappings={"text_column": "content_payload"})
    )
    mapper.setup_data([dataset])

    result = mapper([{"dataset": dataset[0]}])

    assert result == [{"text_column": [prompt]}]


@pytest.mark.regression
def test_generative_mapper_preserves_nested_and_optional_metadata():
    """Nested, list, boolean, and null metadata survive column mapping.

    ## WRITTEN BY AI ##
    """
    prompt = _structured_prompt(
        payload={
            "metadata": {
                "category": "support",
                "labels": ["billing", "priority"],
            },
            "enabled": True,
            "optional_value": None,
        }
    )
    dataset = Dataset.from_list([{"prompt": prompt}])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    result = mapper([{"dataset": dataset[0]}])

    mapped_prompt = result[0]["text_column"][0]
    assert mapped_prompt["metadata"] == {
        "category": "support",
        "labels": ["billing", "priority"],
    }
    assert mapped_prompt["enabled"] is True
    assert mapped_prompt["optional_value"] is None


@pytest.mark.regression
def test_generative_mapper_keeps_per_row_payloads_independent():
    """Metadata from one dataset row does not leak into another.

    ## WRITTEN BY AI ##
    """
    prompts = [
        _structured_prompt(text="First item", payload={"request_id": "one"}),
        _structured_prompt(text="Second item", payload={"request_id": "two"}),
    ]
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    results = [mapper([{"dataset": dataset[index]}]) for index in range(2)]

    assert results[0][0]["text_column"][0] == prompts[0]
    assert results[1][0]["text_column"][0] == prompts[1]
    assert results[0][0]["text_column"][0]["request_id"] == "one"
    assert results[1][0]["text_column"][0]["request_id"] == "two"


@pytest.mark.regression
def test_generative_mapper_preserves_structured_multiturn_prompts():
    """Turn-suffixed structured prompt columns map to separate turns.

    ## WRITTEN BY AI ##
    """
    first_turn = _structured_prompt(text="First turn", payload={"turn_id": 0})
    second_turn = _structured_prompt(text="Second turn", payload={"turn_id": 1})
    dataset = Dataset.from_list([{"prompt_0": first_turn, "prompt_1": second_turn}])
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    result = mapper([{"dataset": dataset[0]}])

    assert result == [
        {"text_column": [first_turn]},
        {"text_column": [second_turn]},
    ]
