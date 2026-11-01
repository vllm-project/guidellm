"""Tests for multimodal generative dataset column mapping."""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.schemas.data.preprocessors import GenerativeColumnMapperArgs


@pytest.mark.regression
@pytest.mark.parametrize(
    ("source_column", "expected_text"),
    [("ref_text", "short transcript"), ("reference_text", "long transcript")],
)
def test_default_mapper_maps_reference_text_aliases(source_column, expected_text):
    """Map either documented transcript column name to ref_text_column.

    :param source_column: Dataset column name to resolve.
    :param expected_text: Transcript value expected in the mapped turn.
    :return: None.
    ## WRITTEN BY AI ##
    """
    dataset = Dataset.from_dict({source_column: [expected_text]})
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    mapped = mapper([{"dataset": dataset[0]}])

    assert mapped == [{"ref_text_column": [expected_text]}]


@pytest.mark.regression
def test_mapper_accepts_explicit_reference_transcript_column():
    """Honor an explicit source name for the reference transcript.

    :return: None.
    ## WRITTEN BY AI ##
    """
    dataset = Dataset.from_dict({"spoken_reference": ["read this aloud"]})
    mapper = GenerativeColumnMapper(
        GenerativeColumnMapperArgs(
            column_mappings={"ref_text_column": "spoken_reference"}
        )
    )
    mapper.setup_data([dataset])

    assert mapper([{"dataset": dataset[0]}]) == [
        {"ref_text_column": ["read this aloud"]}
    ]
