"""
Integration tests: JSONL with turn-indexed tool columns through the
full GenerativeColumnMapper-to-GenerativeRequestFinalizer pipeline.

## WRITTEN BY AI ##
"""

from __future__ import annotations

from typing import Any

import pytest

from guidellm.data.finalizers import GenerativeRequestFinalizer
from guidellm.schemas import GenerationRequest, RequestSettings


def _run_row_through_pipeline(
    row: dict[str, Any],
) -> list[tuple[GenerationRequest, RequestSettings]]:
    """Push a single dataset row through the column mapper and finalizer.

    ## WRITTEN BY AI ##
    """
    from datasets import Dataset

    from guidellm.data.finalizers.generative import GenerativeRequestFinalizerArgs
    from guidellm.data.preprocessors.mappers import (
        GenerativeColumnMapper,
        GenerativeColumnMapperArgs,
    )

    dataset = Dataset.from_dict({k: [v] for k, v in row.items()})

    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])

    finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())
    mapped_turns = mapper([{"dataset": row}])
    return finalizer(mapped_turns)


class TestJsonlMultiTurnToolCallPipeline:
    """Integration tests: JSONL with turn-indexed tool columns through the
    full mapper-to-finalizer pipeline.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_consecutive_tool_turns(self):
        """3-turn JSONL: turns 0-1 are tool calls, turn 2 is plain text.

        With injection turn splitting, this produces 5 requests:
        tool_call, injection, tool_call, injection, standard.

        ## WRITTEN BY AI ##
        """
        row = {
            "prompt_0": "Call the weather tool",
            "output_tokens_count_0": 50,
            "tools_0": '[{"type": "function", "function": {"name": "get_weather"}}]',
            "tool_response_0": '{"temp": 72}',
            "prompt_1": "Now call the stock tool",
            "output_tokens_count_1": 50,
            "tools_1": '[{"type": "function", "function": {"name": "get_stock"}}]',
            "tool_response_1": '{"price": 150}',
            "prompt_2": "Summarize everything",
            "output_tokens_count_2": 100,
        }

        rows = _run_row_through_pipeline(row)

        assert len(rows) == 5
        requests = [r[0] for r in rows]  # Extract GenerationRequest from each tuple

        assert requests[0].turn_type == "client_tool_call"
        assert "tools_column" in requests[0].columns
        assert "tool_response_column" not in requests[0].columns

        assert requests[1].turn_type == "tool_response_injection"
        assert requests[1].columns["tool_response_column"] == ['{"temp": 72}']

        assert requests[2].turn_type == "client_tool_call"
        assert "tools_column" in requests[2].columns
        assert "tool_response_column" not in requests[2].columns

        assert requests[3].turn_type == "tool_response_injection"
        assert requests[3].columns["tool_response_column"] == ['{"price": 150}']

        assert requests[4].turn_type == "standard"
        assert "tools_column" not in requests[4].columns
        assert "tool_response_column" not in requests[4].columns

    @pytest.mark.smoke
    def test_interleaved_tool_turns(self):
        """4-turn JSONL: tool calls on turns 0 and 3, plain text on 1 and 2.

        With injection turn splitting, this produces 6 requests:
        tool_call, injection, standard, standard, tool_call, injection.

        ## WRITTEN BY AI ##
        """
        row = {
            "prompt_0": "Look up the weather",
            "output_tokens_count_0": 50,
            "tools_0": '[{"type": "function", "function": {"name": "get_weather"}}]',
            "tool_response_0": '{"temp": 72}',
            "prompt_1": "Tell me about it",
            "output_tokens_count_1": 60,
            "prompt_2": "Any other thoughts?",
            "output_tokens_count_2": 60,
            "prompt_3": "Now check stocks",
            "output_tokens_count_3": 50,
            "tools_3": '[{"type": "function", "function": {"name": "get_stock"}}]',
            "tool_response_3": '{"price": 150}',
        }

        rows = _run_row_through_pipeline(row)

        assert len(rows) == 6
        requests = [r[0] for r in rows]  # Extract GenerationRequest from each tuple

        assert requests[0].turn_type == "client_tool_call"
        assert "tools_column" in requests[0].columns
        assert "tool_response_column" not in requests[0].columns

        assert requests[1].turn_type == "tool_response_injection"
        assert requests[1].columns["tool_response_column"] == ['{"temp": 72}']

        assert requests[2].turn_type == "standard"
        assert "tools_column" not in requests[2].columns
        assert "tool_response_column" not in requests[2].columns

        assert requests[3].turn_type == "standard"
        assert "tools_column" not in requests[3].columns
        assert "tool_response_column" not in requests[3].columns

        assert requests[4].turn_type == "client_tool_call"
        assert "tools_column" in requests[4].columns
        assert "tool_response_column" not in requests[4].columns

        assert requests[5].turn_type == "tool_response_injection"
        assert requests[5].columns["tool_response_column"] == ['{"price": 150}']
