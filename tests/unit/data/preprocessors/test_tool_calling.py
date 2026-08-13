"""
Unit tests for guidellm.data.preprocessors.tool_calling module.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import pytest

from guidellm.data.preprocessors.tool_calling import (
    ToolCallingMessageExtractor,
    ToolCallingMessageExtractorArgs,
    _normalize_message,
)


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
        extractor = ToolCallingMessageExtractor(ToolCallingMessageExtractorArgs())
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
        extractor = ToolCallingMessageExtractor(ToolCallingMessageExtractorArgs())
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
        extractor = ToolCallingMessageExtractor(ToolCallingMessageExtractorArgs())
        result = extractor(items)

        assert result[0]["tool_response_column"] == [
            '{"first": true}',
            '{"second": true}',
        ]


class TestNormalizeMessage:
    """Tests for _normalize_message helper handling alternate formats.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_openai_format(self):
        """Standard role/content keys are extracted as-is.

        ## WRITTEN BY AI ##
        """
        msg = {"role": "user", "content": "hello"}
        role, content = _normalize_message(msg)
        assert role == "user"
        assert content == "hello"

    @pytest.mark.smoke
    def test_sharegpt_format(self):
        """ShareGPT from/value keys are recognized.

        ## WRITTEN BY AI ##
        """
        msg = {"from": "system", "value": "You are helpful."}
        role, content = _normalize_message(msg)
        assert role == "system"
        assert content == "You are helpful."

    @pytest.mark.sanity
    def test_human_role_normalized_to_user(self):
        """The 'human' role alias is normalized to 'user'.

        ## WRITTEN BY AI ##
        """
        msg = {"from": "human", "value": "What's the weather?"}
        role, content = _normalize_message(msg)
        assert role == "user"
        assert content == "What's the weather?"

    @pytest.mark.sanity
    def test_empty_message(self):
        """A message with no recognized keys returns empty strings.

        ## WRITTEN BY AI ##
        """
        msg = {"unknown_key": "data"}
        role, content = _normalize_message(msg)
        assert role == ""
        assert content == ""


class TestToolCallingMessageExtractorShareGPT:
    """Verify the extractor works with ShareGPT-format messages.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_extracts_from_sharegpt_messages(self):
        """Messages using from/value keys are correctly parsed.

        ## WRITTEN BY AI ##
        """
        messages = [
            {"from": "system", "value": "You are a function calling AI."},
            {"from": "human", "value": "What's the weather in Paris?"},
            {"from": "tool", "value": '{"temp": 22}'},
        ]

        items = [{"text_column": [messages]}]
        extractor = ToolCallingMessageExtractor(ToolCallingMessageExtractorArgs())
        result = extractor(items)

        assert result[0]["text_column"] == ["What's the weather in Paris?"]
        assert result[0]["prefix_column"] == ["You are a function calling AI."]
        assert result[0]["tool_response_column"] == ['{"temp": 22}']

    @pytest.mark.sanity
    def test_mixed_format_messages(self):
        """Messages mixing OpenAI and ShareGPT keys are handled.

        ## WRITTEN BY AI ##
        """
        messages = [
            {"role": "system", "content": "System prompt."},
            {"from": "human", "value": "User question via ShareGPT."},
            {"role": "tool", "content": '{"result": true}'},
        ]

        items = [{"text_column": [messages]}]
        extractor = ToolCallingMessageExtractor(ToolCallingMessageExtractorArgs())
        result = extractor(items)

        assert result[0]["text_column"] == ["User question via ShareGPT."]
        assert result[0]["prefix_column"] == ["System prompt."]
        assert result[0]["tool_response_column"] == ['{"result": true}']
