"""
Unit tests for guidellm.data.preprocessors.tool_calling module.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import pytest

from guidellm.data.preprocessors.tool_calling import ToolCallingMessageExtractor


class TestToolCallingMessageExtractorToolResponses:
    """Verify the extractor populates tool_response_column from messages.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_extracts_tool_role_content(self):
        """Messages with role=tool have their content extracted.

        ## WRITTEN BY AI ##
        """
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Call the tool."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "fn", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "content": '{"status": "success", "data": [1, 2]}',
                "tool_call_id": "call_1",
            },
            {"role": "user", "content": "Thanks!"},
        ]

        items = [{"text_column": [messages]}]
        extractor = ToolCallingMessageExtractor()
        result = extractor(items)

        assert "tool_response_column" in result[0]
        assert result[0]["tool_response_column"] == [
            '{"status": "success", "data": [1, 2]}'
        ]

    @pytest.mark.sanity
    def test_no_tool_responses_when_absent(self):
        """When no role=tool messages exist, tool_response_column is not set.

        ## WRITTEN BY AI ##
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        items = [{"text_column": [messages]}]
        extractor = ToolCallingMessageExtractor()
        result = extractor(items)

        assert "tool_response_column" not in result[0]

    @pytest.mark.sanity
    def test_multiple_tool_responses_extracted(self):
        """Multiple role=tool messages are all extracted in order.

        ## WRITTEN BY AI ##
        """
        messages = [
            {"role": "user", "content": "Do two things."},
            {"role": "tool", "content": '{"first": true}', "tool_call_id": "c1"},
            {"role": "tool", "content": '{"second": true}', "tool_call_id": "c2"},
        ]

        items = [{"text_column": [messages]}]
        extractor = ToolCallingMessageExtractor()
        result = extractor(items)

        assert result[0]["tool_response_column"] == [
            '{"first": true}',
            '{"second": true}',
        ]
