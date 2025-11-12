import pytest

from guidellm.backends.response_handlers import (
    AudioResponseHandler,
    ChatCompletionsResponseHandler,
    TextCompletionsResponseHandler,
)
from guidellm.schemas import UsageMetrics


@pytest.fixture
def text_handler():
    return TextCompletionsResponseHandler()


@pytest.fixture
def chat_handler():
    return ChatCompletionsResponseHandler()


@pytest.fixture
def audio_handler():
    return AudioResponseHandler()


def assert_usage_metrics_equal(actual: UsageMetrics, expected: UsageMetrics):
    assert getattr(actual, "text_tokens", None) == getattr(
        expected, "text_tokens", None
    )
    assert getattr(actual, "image_tokens", None) == getattr(
        expected, "image_tokens", None
    )
    assert getattr(actual, "video_tokens", None) == getattr(
        expected, "video_tokens", None
    )
    assert getattr(actual, "audio_tokens", None) == getattr(
        expected, "audio_tokens", None
    )
    assert getattr(actual, "audio_seconds", None) == getattr(
        expected, "audio_seconds", None
    )


# =====================
# TextCompletionsResponseHandler::extract_metrics tests
# =====================


@pytest.mark.smoke
def test_extract_metrics_empty_usage(text_handler):
    in_metrics, out_metrics = text_handler.extract_metrics(None)
    assert_usage_metrics_equal(in_metrics, UsageMetrics())
    assert_usage_metrics_equal(out_metrics, UsageMetrics())

    in_metrics, out_metrics = text_handler.extract_metrics({})
    assert_usage_metrics_equal(in_metrics, UsageMetrics())
    assert_usage_metrics_equal(out_metrics, UsageMetrics())


@pytest.mark.smoke
def test_extract_metrics_top_level_only(text_handler):
    usage = {"prompt_tokens": 10, "completion_tokens": 20}
    in_metrics, out_metrics = text_handler.extract_metrics(usage)

    expected_in = UsageMetrics(text_tokens=10)
    expected_out = UsageMetrics(text_tokens=20)

    assert_usage_metrics_equal(in_metrics, expected_in)
    assert_usage_metrics_equal(out_metrics, expected_out)


@pytest.mark.smoke
def test_extract_metrics_with_details(text_handler):
    usage = {
        "prompt_tokens_details": {
            "prompt_tokens": 3,
            "image_tokens": 4,
            "video_tokens": 5,
            "audio_tokens": 6,
            "seconds": 1,
        },
        "completion_tokens_details": {
            "completion_tokens": 7,
            "image_tokens": 8,
            "video_tokens": 9,
            "audio_tokens": 10,
            "seconds": 2,
        },
    }

    in_metrics, out_metrics = text_handler.extract_metrics(usage)

    expected_in = UsageMetrics(
        text_tokens=3, image_tokens=4, video_tokens=5, audio_tokens=6, audio_seconds=1
    )
    expected_out = UsageMetrics(
        text_tokens=7, image_tokens=8, video_tokens=9, audio_tokens=10, audio_seconds=2
    )

    assert_usage_metrics_equal(in_metrics, expected_in)
    assert_usage_metrics_equal(out_metrics, expected_out)


@pytest.mark.smoke
def test_extract_metrics_details_none_fallback_to_top_level(text_handler):
    usage = {
        "prompt_tokens_details": None,
        "prompt_tokens": 11,
        "completion_tokens_details": None,
        "completion_tokens": 12,
    }

    in_metrics, out_metrics = text_handler.extract_metrics(usage)

    expected_in = UsageMetrics(text_tokens=11)
    expected_out = UsageMetrics(text_tokens=12)

    assert_usage_metrics_equal(in_metrics, expected_in)
    assert_usage_metrics_equal(out_metrics, expected_out)


@pytest.mark.smoke
def test_extract_metrics_zero_in_details_prefers_top_level(text_handler):
    usage = {
        "prompt_tokens_details": {"prompt_tokens": 0},
        "prompt_tokens": 7,
        "completion_tokens_details": {"completion_tokens": 0},
        "completion_tokens": 15,
    }

    in_metrics, out_metrics = text_handler.extract_metrics(usage)

    expected_in = UsageMetrics(text_tokens=7)
    expected_out = UsageMetrics(text_tokens=15)

    assert_usage_metrics_equal(in_metrics, expected_in)
    assert_usage_metrics_equal(out_metrics, expected_out)


# ============================
# TextCompletionsResponseHandler::add_streaming_line tests
# ============================


@pytest.mark.smoke
def test_textcomplation_done_line(text_handler):
    assert text_handler.add_streaming_line("data: [DONE]") is None
    assert text_handler.streaming_texts == []
    # streaming_usage should be None
    assert text_handler.streaming_usage is None


@pytest.mark.smoke
def test_textcomplation_blank_and_invalid_line(text_handler):
    assert text_handler.add_streaming_line("") == 0
    assert text_handler.add_streaming_line("random gibberish") == 0
    assert (
        text_handler.add_streaming_line("data:") == 0
    )  # empty data field -> extract_line_data returns {}
    assert text_handler.add_streaming_line('notdata: {"foo": 1}') == 0


@pytest.mark.smoke
def test_textcomplation_valid_line_with_text(text_handler):
    line = 'data: {"choices":[{"text":"Hello World!"}], \
            "usage": {"prompt_tokens":1}}'
    result = text_handler.add_streaming_line(line)
    assert result == 1
    assert text_handler.streaming_texts == ["Hello World!"]
    assert text_handler.streaming_usage == {"prompt_tokens": 1}


@pytest.mark.smoke
def test_textcomplation_valid_line_with_empty_choices(text_handler):
    line = 'data: {"choices":[]}'
    result = text_handler.add_streaming_line(line)
    assert result == 0
    assert text_handler.streaming_texts == []


@pytest.mark.smoke
def test_textcomplation_valid_line_with_choices_but_no_text(text_handler):
    line = 'data: {"choices":[{"notextfield":"foo"}]}'
    result = text_handler.add_streaming_line(line)
    assert result == 0
    assert text_handler.streaming_texts == []


@pytest.mark.smoke
def test_textcomplation_accumulate_multiple_lines(text_handler):
    # test append multiple lines
    line1 = 'data: {"choices":[{"text":"A"}]}'
    line2 = 'data: {"choices":[{"text":"B"}]}'
    assert text_handler.add_streaming_line(line1) == 1
    assert text_handler.add_streaming_line(line2) == 1
    assert text_handler.streaming_texts == ["A", "B"]


@pytest.mark.smoke
def test_textcomplation_line_with_only_usage(text_handler):
    # when no choices, usage: 0，update streaming_usage
    line = 'data: {"usage": {"prompt_tokens":2}}'
    assert text_handler.add_streaming_line(line) == 0
    assert text_handler.streaming_usage == {"prompt_tokens": 2}


# ============================
# ChatCompletionsResponseHandler::add_streaming_line tests
# ============================


@pytest.mark.smoke
def test_chatcomplation_done_line(chat_handler):
    assert chat_handler.add_streaming_line("data: [DONE]") is None
    assert chat_handler.streaming_texts == []
    assert chat_handler.streaming_usage is None


@pytest.mark.smoke
def test_chatcomplation_blank_and_invalid_line(chat_handler):
    assert chat_handler.add_streaming_line("") == 0
    assert chat_handler.add_streaming_line("random gibberish") == 0
    assert chat_handler.add_streaming_line("data:") == 0
    assert chat_handler.add_streaming_line('notdata: {"foo": 1}') == 0


@pytest.mark.smoke
def test_chatcomplation_valid_line_with_text(chat_handler):
    line = 'data: {"choices":[{"delta":{"content": "Hello World!"}}], \
            "usage": {"prompt_tokens":1}}'
    result = chat_handler.add_streaming_line(line)
    assert result == 1
    assert chat_handler.streaming_texts == ["Hello World!"]
    assert chat_handler.streaming_usage == {"prompt_tokens": 1}


@pytest.mark.smoke
def test_chatcomplation_valid_line_with_empty_choices(chat_handler):
    line = 'data: {"choices":[]}'
    result = chat_handler.add_streaming_line(line)
    assert result == 0
    assert chat_handler.streaming_texts == []


@pytest.mark.smoke
def test_chatcomplation_valid_line_with_choices_but_no_text(chat_handler):
    line = 'data: {"choices":[{"notextfield":"foo"}]}'
    result = chat_handler.add_streaming_line(line)
    assert result == 0
    assert chat_handler.streaming_texts == []


@pytest.mark.smoke
def test_chatcomplation_accumulate_multiple_lines(chat_handler):
    line1 = 'data: {"choices":[{"delta":{"content":"A"}}]}'
    line2 = 'data: {"choices":[{"delta":{"content":"B"}}]}'
    assert chat_handler.add_streaming_line(line1) == 1
    assert chat_handler.add_streaming_line(line2) == 1
    assert chat_handler.streaming_texts == ["A", "B"]


@pytest.mark.smoke
def test_chatcomplation_line_with_only_usage(chat_handler):
    line = 'data: {"usage": {"prompt_tokens":2}}'
    assert chat_handler.add_streaming_line(line) == 0
    assert chat_handler.streaming_usage == {"prompt_tokens": 2}


# ============================
# AudioResponseHandler::add_streaming_line tests
# ============================


@pytest.mark.smoke
def test_audio_done_line(audio_handler):
    assert audio_handler.add_streaming_line("data: [DONE]") is None
    assert audio_handler.streaming_texts == []
    assert audio_handler.streaming_usage is None


@pytest.mark.smoke
def test_audio_blank_and_invalid_line(audio_handler):
    assert audio_handler.add_streaming_line("") == 0
    assert audio_handler.add_streaming_line("notjson") == 0
    assert audio_handler.add_streaming_line("foo: bar") == 0


@pytest.mark.smoke
def test_audio_valid_line_with_text(audio_handler):
    line = '{"text":"hello audio"}'
    result = audio_handler.add_streaming_line(line)
    assert result == 1
    assert audio_handler.streaming_texts == ["hello audio"]


@pytest.mark.smoke
def test_audio_valid_line_with_usage(audio_handler):
    line = '{"usage": {"input_tokens": 5}}'
    result = audio_handler.add_streaming_line(line)
    assert result == 0
    assert audio_handler.streaming_usage == {"input_tokens": 5}


@pytest.mark.smoke
def test_audio_valid_line_with_text_and_usage(audio_handler):
    line = '{"text":"foo", "usage":{"input_tokens":9}}'
    result = audio_handler.add_streaming_line(line)
    assert result == 1
    assert audio_handler.streaming_texts == ["foo"]
    assert audio_handler.streaming_usage == {"input_tokens": 9}


@pytest.mark.smoke
def test_audio_accumulate_multiple_lines(audio_handler):
    assert audio_handler.add_streaming_line('{"text":"A"}') == 1
    assert audio_handler.add_streaming_line('{"text":"B"}') == 1
    assert audio_handler.streaming_texts == ["A", "B"]


# ============================
# AudioResponseHandler::compile_streaming tests
# ============================


class DummyRequest:
    def __init__(self, request_id: str, arguments=None):
        self.request_id = request_id
        self.arguments = arguments


@pytest.mark.smoke
def test_audio_compile_streaming_with_no_usage(audio_handler):
    # when streaming_usage is None，return two UsageMetrics() with all zero
    # meanwhile combine streaming_texts as text
    audio_handler.streaming_texts = ["part1", "part2"]
    audio_handler.streaming_usage = None

    req = DummyRequest(request_id="req-123", arguments=None)
    resp = audio_handler.compile_streaming(req)

    assert resp.request_id == "req-123"
    # test request_args usage: str(request.arguments.model_dump()
    # if request.arguments else None
    assert resp.request_args == "None"
    assert resp.text == "part1part2"

    assert_usage_metrics_equal(resp.input_metrics, UsageMetrics())
    assert_usage_metrics_equal(resp.output_metrics, UsageMetrics())


@pytest.mark.smoke
def test_audio_compile_streaming_with_input_output_details(audio_handler):
    # when streaming_usage include: input_token_details / output_token_details
    # details is high priority
    audio_handler.streaming_texts = ["hello", " ", "audio"]
    audio_handler.streaming_usage = {
        "input_token_details": {
            "text_tokens": 2,
            "audio_tokens": 4,
            "seconds": 6,
        },
        "output_token_details": {
            "text_tokens": 9,
        },
        # details is preferred
        "input_tokens": 99,
        "audio_tokens": 100,
        "seconds": 101,
        "output_tokens": 200,
    }

    req = DummyRequest(request_id="req-xyz", arguments=None)
    resp = audio_handler.compile_streaming(req)

    assert resp.request_id == "req-xyz"
    assert resp.request_args == "None"
    assert resp.text == "hello audio"

    expected_in = UsageMetrics(text_tokens=2, audio_tokens=4, audio_seconds=6)
    expected_out = UsageMetrics(text_tokens=9)

    assert_usage_metrics_equal(resp.input_metrics, expected_in)
    assert_usage_metrics_equal(resp.output_metrics, expected_out)
