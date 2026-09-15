from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.deserializers.trace_otel import (
    parse_gen_ai_messages,
    parse_span_timestamp,
)
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)
from guidellm.schemas.data import DEFAULT_SYNTHETIC_TOOLS, OTELTraceFormatArgs
from guidellm.settings import settings
from tests.unit.data.deserializers.trace_test_utils import trace_file_source


def mock_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is one token."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{i}" for i, _ in enumerate(tokens)
    )
    return proc


def write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def ibm_chat_span(
    *,
    span_id: str,
    trace_id: str,
    start_time: str,
    prompt_tokens: int,
    completion_tokens: int,
    operation: str = "chat",
    status_code: int | None = None,
    messages: list[dict] | str | None = None,
    output_messages: list[dict] | str | None = None,
    output_text: str | None = None,
    tools: list[dict] | str | None = None,
    finish_reasons: list[str] | str | None = None,
) -> dict:
    span = {
        "span_id": span_id,
        "trace_id": trace_id,
        "start_time": start_time,
        "end_time": start_time,
        "attributes": {
            "gen_ai.operation.name": operation,
            "gen_ai.usage.prompt_tokens": prompt_tokens,
            "gen_ai.usage.completion_tokens": completion_tokens,
        },
    }
    if status_code is not None:
        span["status"] = {"code": status_code, "message": ""}
    if messages is not None:
        span["attributes"]["gen_ai.input.messages"] = messages
    if output_messages is not None:
        span["attributes"]["gen_ai.output.messages"] = output_messages
    if output_text is not None:
        span["attributes"]["gen_ai.output.text"] = output_text
    if tools is not None:
        span["attributes"]["gen_ai.tool.definitions"] = tools
    if finish_reasons is not None:
        span["attributes"]["gen_ai.response.finish_reasons"] = finish_reasons
    return span


def ibm_session_line(trace_id: str, spans: list[dict]) -> dict:
    return {"trace_id": trace_id, "spans": spans}


def exgentic_chat_span(
    *,
    span_id: str,
    trace_id: str,
    start_time: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "chat",
) -> dict:
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "start_time": start_time,
        "end_time": start_time,
        "name": "chat model",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {
            "gen_ai.operation.name": operation,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        },
        "status": {"code": 1, "message": ""},
    }


def flat_chat_span(
    *,
    span_id: str,
    trace_id: str,
    start_time: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "chat",
) -> dict:
    return exgentic_chat_span(
        span_id=span_id,
        trace_id=trace_id,
        start_time=start_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        operation=operation,
    )


def invoke_agent_span(*, span_id: str, trace_id: str, start_time: str) -> dict:
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "start_time": start_time,
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    }


def execute_tool_span(
    *,
    span_id: str,
    trace_id: str,
    start_time: str,
    result: Any = None,
    operation: bool = True,
    name: str = "execute_tool get_weather",
) -> dict:
    attributes: dict[str, Any] = {}
    if operation:
        attributes["gen_ai.operation.name"] = "execute_tool"
    if result is not None:
        attributes["gen_ai.tool.call.result"] = result
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "start_time": start_time,
        "name": name,
        "attributes": attributes,
    }


USER_HELLO = {"role": "user", "content": "hello"}
ASSISTANT_HI = {"role": "assistant", "content": "hi"}
USER_AGAIN = {"role": "user", "content": "again"}
ASSISTANT_OK = {"role": "assistant", "content": "ok"}

WEATHER_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
]
USER_WEATHER = {
    "role": "user",
    "parts": [{"type": "text", "content": "Weather in Paris?"}],
}
ASSISTANT_WEATHER_CALL = {
    "role": "assistant",
    "parts": [
        {
            "type": "tool_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": {"location": "Paris"},
        }
    ],
}
TOOL_WEATHER_RESULT = {
    "role": "tool",
    "parts": [
        {"type": "tool_call_response", "id": "call_1", "result": "rainy, 57F"},
    ],
}
ASSISTANT_WEATHER_CALL_2 = {
    "role": "assistant",
    "parts": [
        {
            "type": "tool_call",
            "id": "call_2",
            "name": "get_weather",
            "arguments": {"location": "Lyon"},
        }
    ],
}
TOOL_WEATHER_RESULT_2 = {
    "role": "tool",
    "parts": [
        {"type": "tool_call_response", "id": "call_2", "result": "sunny, 70F"},
    ],
}
ASSISTANT_WEATHER_TEXT = {"role": "assistant", "content": "Rainy in Paris."}


def two_span_tool_session() -> dict:
    return ibm_session_line(
        "t0",
        [
            ibm_chat_span(
                span_id="s0",
                trace_id="t0",
                start_time="2024-01-01T12:00:00+00:00",
                prompt_tokens=10,
                completion_tokens=8,
                messages=[USER_WEATHER],
                output_messages=[ASSISTANT_WEATHER_CALL],
                tools=WEATHER_TOOLS,
            ),
            ibm_chat_span(
                span_id="s1",
                trace_id="t0",
                start_time="2024-01-01T12:00:02+00:00",
                prompt_tokens=20,
                completion_tokens=12,
                messages=[USER_WEATHER, ASSISTANT_WEATHER_CALL, TOOL_WEATHER_RESULT],
                output_messages=[ASSISTANT_WEATHER_TEXT],
            ),
        ],
    )


def multi_step_tool_session() -> dict:
    second_input = [USER_WEATHER, ASSISTANT_WEATHER_CALL, TOOL_WEATHER_RESULT]
    third_input = [
        *second_input,
        ASSISTANT_WEATHER_CALL_2,
        TOOL_WEATHER_RESULT_2,
    ]
    return ibm_session_line(
        "t0",
        [
            ibm_chat_span(
                span_id="s0",
                trace_id="t0",
                start_time="2024-01-01T12:00:00+00:00",
                prompt_tokens=10,
                completion_tokens=8,
                messages=[USER_WEATHER],
                output_messages=[ASSISTANT_WEATHER_CALL],
                tools=WEATHER_TOOLS,
            ),
            ibm_chat_span(
                span_id="s1",
                trace_id="t0",
                start_time="2024-01-01T12:00:02+00:00",
                prompt_tokens=20,
                completion_tokens=8,
                messages=second_input,
                output_messages=[ASSISTANT_WEATHER_CALL_2],
                tools=WEATHER_TOOLS,
            ),
            ibm_chat_span(
                span_id="s2",
                trace_id="t0",
                start_time="2024-01-01T12:00:04+00:00",
                prompt_tokens=30,
                completion_tokens=12,
                messages=third_input,
                output_messages=[ASSISTANT_WEATHER_TEXT],
            ),
        ],
    )


def accumulating_session(
    *,
    messages_as_json: bool = False,
    second_input: list[dict] | None = None,
) -> dict:
    first_in: list[dict] | str = [USER_HELLO]
    second_in: list[dict] | str = (
        list(second_input)
        if second_input is not None
        else [USER_HELLO, ASSISTANT_HI, USER_AGAIN]
    )
    first_out: list[dict] | str = [ASSISTANT_HI]
    second_out: list[dict] | str = [ASSISTANT_OK]
    if messages_as_json:
        first_in = json.dumps(first_in)
        second_in = json.dumps(second_in)
        first_out = json.dumps(first_out)
        second_out = json.dumps(second_out)
    return ibm_session_line(
        "t0",
        [
            ibm_chat_span(
                span_id="s0",
                trace_id="t0",
                start_time="2024-01-01T12:00:00+00:00",
                prompt_tokens=2,
                completion_tokens=1,
                messages=first_in,
                output_messages=first_out,
            ),
            ibm_chat_span(
                span_id="s1",
                trace_id="t0",
                start_time="2024-01-01T12:00:01+00:00",
                prompt_tokens=5,
                completion_tokens=1,
                messages=second_in,
                output_messages=second_out,
            ),
        ],
    )


def load_graph_turns(row: dict) -> list[ConversationTurnData]:
    graph = ConversationGraphData.model_validate(json.loads(row["conversation_turns"]))
    return graph.turns


def deserialize(path: Path, **kwargs):
    config = OTELTraceFormatArgs(source=trace_file_source(path), **kwargs)
    return TraceDatasetDeserializer()(
        config=config,
        processor_factory=mock_processor,
        random_seed=42,
    )


class TestOTELTraceFormat:
    @pytest.mark.regression
    def test_format_registered_with_deserializer(self, tmp_path: Path):
        """
        kind=otel is registered with the shared trace deserializer.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=5,
                        )
                    ],
                )
            ],
        )
        DatasetDeserializerFactory.deserialize(
            config=OTELTraceFormatArgs(source=trace_file_source(trace)),
            processor_factory=mock_processor,
            random_seed=42,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("kind", ["otel", "opentelemetry", "otel_trace"])
    def test_kind_aliases_load(self, tmp_path: Path, kind: str):
        """
        otel, opentelemetry, and otel_trace all dispatch to the OTEL format.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                        )
                    ],
                )
            ],
        )
        ds = DatasetDeserializerFactory.deserialize(
            config=OTELTraceFormatArgs(
                kind=kind, source=trace_file_source(trace), content="synthetic"
            ),
            processor_factory=mock_processor,
            random_seed=42,
        )
        turns = load_graph_turns(next(iter(ds)))
        assert len(turns) == 1
        assert turns[0].columns["prompt_tokens_count_column"][0] == 4

    @pytest.mark.smoke
    def test_nested_ibm_prompt_and_completion_tokens(self, tmp_path: Path):
        """
        Session-per-line files using deprecated prompt_tokens keys replay in order.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "career_advice_080",
                    [
                        ibm_chat_span(
                            span_id="turn_1",
                            trace_id="career_advice_080",
                            start_time="2024-01-01T12:00:01+00:00",
                            prompt_tokens=8,
                            completion_tokens=3,
                        ),
                        ibm_chat_span(
                            span_id="turn_0",
                            trace_id="career_advice_080",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert [turn.columns["prompt_tokens_count_column"][0] for turn in turns] == [
            4,
            8,
        ]
        assert [turn.columns["output_tokens_count_column"][0] for turn in turns] == [
            2,
            3,
        ]

    @pytest.mark.smoke
    def test_nested_exgentic_input_and_output_tokens(self, tmp_path: Path):
        """
        Session-per-line files using current input_tokens keys replay token counts.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                {
                    "trace_id": "session_0",
                    "spans": [
                        exgentic_chat_span(
                            span_id="s0",
                            trace_id="session_0",
                            start_time="2026-05-11T04:56:43.139567",
                            input_tokens=6,
                            output_tokens=9,
                        )
                    ],
                }
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert turns[0].columns["prompt_tokens_count_column"][0] == 6
        assert turns[0].columns["output_tokens_count_column"][0] == 9

    @pytest.mark.sanity
    def test_flat_span_per_line_groups_by_trace_id(self, tmp_path: Path):
        """
        Adjacent span-per-line rows with the same trace_id become one conversation.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                flat_chat_span(
                    span_id="a0",
                    trace_id="trace_a",
                    start_time="2024-01-01T12:00:00+00:00",
                    input_tokens=2,
                    output_tokens=1,
                ),
                flat_chat_span(
                    span_id="a1",
                    trace_id="trace_a",
                    start_time="2024-01-01T12:00:01+00:00",
                    input_tokens=4,
                    output_tokens=2,
                ),
                flat_chat_span(
                    span_id="b0",
                    trace_id="trace_b",
                    start_time="2024-01-01T12:00:00+00:00",
                    input_tokens=3,
                    output_tokens=1,
                ),
            ],
        )
        conversations = [
            load_graph_turns(row) for row in deserialize(trace, content="synthetic")
        ]
        assert len(conversations) == 2
        assert [
            turn.columns["prompt_tokens_count_column"][0] for turn in conversations[0]
        ] == [2, 4]
        assert [
            turn.columns["prompt_tokens_count_column"][0] for turn in conversations[1]
        ] == [3]

    @pytest.mark.regression
    def test_interleaved_span_per_line_is_three_conversations(self, tmp_path: Path):
        """
        Interleaved trace_ids are consecutive groups, not merged across gaps.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                flat_chat_span(
                    span_id="a0",
                    trace_id="trace_a",
                    start_time="2024-01-01T12:00:00+00:00",
                    input_tokens=2,
                    output_tokens=1,
                ),
                flat_chat_span(
                    span_id="b0",
                    trace_id="trace_b",
                    start_time="2024-01-01T12:00:00+00:00",
                    input_tokens=3,
                    output_tokens=1,
                ),
                flat_chat_span(
                    span_id="a1",
                    trace_id="trace_a",
                    start_time="2024-01-01T12:00:01+00:00",
                    input_tokens=4,
                    output_tokens=2,
                ),
            ],
        )
        conversations = [
            load_graph_turns(row) for row in deserialize(trace, content="synthetic")
        ]
        assert len(conversations) == 3
        assert [
            turn.columns["prompt_tokens_count_column"][0] for turn in conversations[0]
        ] == [2]
        assert [
            turn.columns["prompt_tokens_count_column"][0] for turn in conversations[1]
        ] == [3]
        assert [
            turn.columns["prompt_tokens_count_column"][0] for turn in conversations[2]
        ] == [4]

    @pytest.mark.sanity
    def test_iso_start_time_becomes_relative_seconds(self, tmp_path: Path):
        """
        ISO-8601 start_time values become conversation-relative offsets in seconds.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00Z",
                            prompt_tokens=2,
                            completion_tokens=1,
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00.500000Z",
                            prompt_tokens=2,
                            completion_tokens=1,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        offsets = [turn.columns["relative_timestamp_column"][0] for turn in turns]
        assert offsets[0] == pytest.approx(0.0)
        assert offsets[1] == pytest.approx(0.5)

    @pytest.mark.sanity
    def test_non_chat_spans_are_dropped(self, tmp_path: Path):
        """
        invoke_agent and similar non-LLM spans are not replayed.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        invoke_agent_span(
                            span_id="agent",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                        ),
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            prompt_tokens=5,
                            completion_tokens=2,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert len(turns) == 1
        assert turns[0].columns["prompt_tokens_count_column"][0] == 5

    @pytest.mark.sanity
    def test_failed_spans_are_dropped(self, tmp_path: Path):
        """
        Spans with error status are skipped even when they include token counts.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="fail",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=20,
                            completion_tokens=1,
                            status_code=2,
                        ),
                        ibm_chat_span(
                            span_id="ok",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                            status_code=1,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert len(turns) == 1
        assert turns[0].columns["prompt_tokens_count_column"][0] == 4

    @pytest.mark.regression
    def test_missing_usage_attributes_raise(self, tmp_path: Path):
        """
        Conversations with no LLM spans that carry usage tokens are skipped.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        invoke_agent_span(
                            span_id="agent",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                        ),
                        {
                            "span_id": "s0",
                            "trace_id": "t0",
                            "start_time": "2024-01-01T12:00:01+00:00",
                            "attributes": {"gen_ai.operation.name": "chat"},
                        },
                    ],
                )
            ],
        )
        ds = deserialize(trace, content="synthetic")
        with pytest.raises(InvalidRowError, match="no LLM spans"):
            next(iter(ds))

    @pytest.mark.regression
    def test_empty_conversation_after_first_row(self, tmp_path: Path):
        """
        Skip an empty nested conversation when iterating after the first row.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                        )
                    ],
                ),
                ibm_session_line(
                    "t1",
                    [
                        invoke_agent_span(
                            span_id="agent",
                            trace_id="t1",
                            start_time="2024-01-01T12:00:00+00:00",
                        )
                    ],
                ),
            ],
        )
        ds = deserialize(trace, content="synthetic")
        row_iter = iter(ds)
        turns = load_graph_turns(next(row_iter))
        assert len(turns) == 1
        assert turns[0].columns["prompt_tokens_count_column"][0] == 4
        with pytest.raises(InvalidRowError, match="no LLM spans"):
            next(row_iter)

    @pytest.mark.sanity
    def test_prefix_reuse_within_conversation(self, tmp_path: Path):
        """
        Synthetic+trace mode grows a full prompt of n_in and uses history_context=new.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=1,
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            prompt_tokens=8,
                            completion_tokens=1,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(
            next(iter(deserialize(trace, content="synthetic", history="trace")))
        )
        first = turns[0].columns["text_column"][0]
        second = turns[1].columns["text_column"][0]
        assert first.split() == ["tok0", "tok1", "tok2", "tok3"]
        assert second.split()[:4] == first.split()
        assert len(second.split()) == 8
        assert turns[1].parents[0].history_context == "new"

    @pytest.mark.smoke
    def test_raw_trace_sends_full_messages_and_new_history(self, tmp_path: Path):
        """
        Default raw+trace mode sends each span's full messages with history_context=new.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [accumulating_session()])
        turns = load_graph_turns(next(iter(deserialize(trace))))
        assert turns[0].columns["raw_messages_column"][0] == [USER_HELLO]
        assert turns[1].columns["raw_messages_column"][0] == [
            USER_HELLO,
            ASSISTANT_HI,
            USER_AGAIN,
        ]
        assert "text_column" not in turns[0].columns
        assert turns[1].parents[0].history_context == "new"
        assert turns[0].columns["output_tokens_count_column"][0] == 1

    @pytest.mark.smoke
    def test_raw_runtime_sends_delta_and_full_history(self, tmp_path: Path):
        """
        Raw+runtime mode verifies the accumulating prefix and sends only new messages.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [accumulating_session()])
        turns = load_graph_turns(
            next(iter(deserialize(trace, content="raw", history="runtime")))
        )
        assert turns[0].columns["raw_messages_column"][0] == [USER_HELLO]
        assert turns[1].columns["raw_messages_column"][0] == [USER_AGAIN]
        assert turns[1].parents[0].history_context == "full"

    @pytest.mark.regression
    def test_raw_runtime_mismatch_raises(self, tmp_path: Path):
        """
        Runtime history refuses to mix recorded assistant text that does not match.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                accumulating_session(
                    second_input=[USER_HELLO, USER_AGAIN],
                )
            ],
        )
        ds = deserialize(trace, content="raw", history="runtime")
        with pytest.raises(InvalidRowError, match="do not continue the previous span"):
            next(iter(ds))

    @pytest.mark.sanity
    def test_synthetic_runtime_sends_token_delta(self, tmp_path: Path):
        """
        Synthetic+runtime mode synthesizes only n_in - prev n_in - prev n_out tokens.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=1,
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            prompt_tokens=8,
                            completion_tokens=1,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(
            next(iter(deserialize(trace, content="synthetic", history="runtime")))
        )
        assert len(turns[0].columns["text_column"][0].split()) == 4
        # delta = 8 - 4 - 1 = 3
        assert len(turns[1].columns["text_column"][0].split()) == 3
        assert turns[1].parents[0].history_context == "full"

    @pytest.mark.sanity
    def test_json_string_messages_round_trip(self, tmp_path: Path):
        """
        gen_ai.input.messages stored as a JSON string parse like a native list.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [accumulating_session(messages_as_json=True)])
        turns = load_graph_turns(next(iter(deserialize(trace, content="raw"))))
        assert turns[0].columns["raw_messages_column"][0] == [USER_HELLO]
        assert turns[1].columns["raw_messages_column"][0] == [
            USER_HELLO,
            ASSISTANT_HI,
            USER_AGAIN,
        ]

    @pytest.mark.sanity
    def test_otel_parts_normalize_to_openai(self, tmp_path: Path):
        """
        OTel parts (text, tool_call, tool_call_response) become OpenAI chat dicts.

        ## WRITTEN BY AI ##
        """
        tools = [
            {
                "type": "function",
                "function": {"name": "search", "parameters": {"type": "object"}},
            }
        ]
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                            messages=[
                                {
                                    "role": "user",
                                    "parts": [{"type": "text", "content": "look up x"}],
                                }
                            ],
                            output_messages=[
                                {
                                    "role": "assistant",
                                    "parts": [
                                        {
                                            "type": "tool_call",
                                            "id": "c1",
                                            "name": "search",
                                            "arguments": {"q": "x"},
                                        }
                                    ],
                                }
                            ],
                            tools=tools,
                        )
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace))))
        assert turns[0].columns["raw_messages_column"][0] == [
            {"role": "user", "content": "look up x"}
        ]
        assert turns[0].columns["tools_column"][0] == tools
        assert turns[0].columns["turn_type_column"] == ["client_tool_call"]
        assert "output_tokens_count_column" not in turns[0].columns
        assert turns[1].columns["turn_type_column"] == ["tool_response_injection"]
        assert turns[1].columns["tool_response_column"] == [
            settings.default_synthetic_tool_response
        ]

    @pytest.mark.regression
    def test_raw_missing_messages_raises(self, tmp_path: Path):
        """
        content=raw errors on metrics-only spans and tells the user to use synthetic.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                        )
                    ],
                )
            ],
        )
        with pytest.raises(InvalidRowError, match="Pass content=synthetic"):
            next(iter(deserialize(trace)))

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("content", "history"),
        [
            ("raw", "trace"),
            ("raw", "runtime"),
            ("synthetic", "trace"),
            ("synthetic", "runtime"),
        ],
    )
    def test_tool_loop_all_modes(self, tmp_path: Path, content: str, history: str):
        """
        Recorded tool_call then tool-result delta becomes call + injection in all modes.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [two_span_tool_session()])
        turns = load_graph_turns(
            next(iter(deserialize(trace, content=content, history=history)))
        )
        assert len(turns) == 2
        assert turns[0].columns["turn_type_column"] == ["client_tool_call"]
        assert turns[0].columns["tools_column"][0] == WEATHER_TOOLS
        assert "output_tokens_count_column" not in turns[0].columns
        assert turns[1].columns["turn_type_column"] == ["tool_response_injection"]
        assert turns[1].columns["tool_response_column"] == ["rainy, 57F"]
        assert turns[1].columns["output_tokens_count_column"] == [12]
        assert "raw_messages_column" not in turns[1].columns
        assert "text_column" not in turns[1].columns
        assert turns[1].parents[0].history_context == "full"
        if content == "raw":
            assert turns[0].columns["raw_messages_column"][0] == [
                {"role": "user", "content": "Weather in Paris?"}
            ]
        else:
            assert "text_column" in turns[0].columns
            assert "raw_messages_column" not in turns[0].columns

    @pytest.mark.sanity
    def test_multi_step_injection_keeps_tools(self, tmp_path: Path):
        """
        An injection whose span also called tools keeps tools_column for the next loop.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [multi_step_tool_session()])
        turns = load_graph_turns(next(iter(deserialize(trace))))
        assert [turn.columns["turn_type_column"][0] for turn in turns] == [
            "client_tool_call",
            "tool_response_injection",
            "tool_response_injection",
        ]
        assert turns[1].columns["tools_column"][0] == WEATHER_TOOLS
        assert turns[1].columns["tool_response_column"] == ["rainy, 57F"]
        assert "tools_column" not in turns[2].columns
        assert turns[2].columns["tool_response_column"] == ["sunny, 70F"]

    @pytest.mark.sanity
    def test_tool_choice_auto_column(self, tmp_path: Path):
        """
        tool_choice=auto lands on tool_choice_column of the call node.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(tmp_path, [two_span_tool_session()])
        turns = load_graph_turns(next(iter(deserialize(trace, tool_choice="auto"))))
        assert turns[0].columns["tool_choice_column"] == ["auto"]

    @pytest.mark.sanity
    def test_missing_next_span_results_use_placeholder(self, tmp_path: Path):
        """
        A last-span tool call with no follower still emits a placeholder injection.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                            messages=[USER_WEATHER],
                            output_messages=[ASSISTANT_WEATHER_CALL],
                            tools=WEATHER_TOOLS,
                        )
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace))))
        assert len(turns) == 2
        assert turns[1].columns["tool_response_column"] == [
            settings.default_synthetic_tool_response
        ]
        assert turns[1].parents[0].history_context == "full"

    @pytest.mark.sanity
    def test_prefix_mismatch_keeps_next_span(self, tmp_path: Path):
        """
        When the next span is not a tool-result delta, inject a placeholder
        and replay it.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                            messages=[USER_WEATHER],
                            output_messages=[ASSISTANT_WEATHER_CALL],
                            tools=WEATHER_TOOLS,
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:02+00:00",
                            prompt_tokens=4,
                            completion_tokens=2,
                            messages=[USER_HELLO],
                            output_messages=[ASSISTANT_OK],
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace))))
        assert [turn.columns["turn_type_column"][0] for turn in turns[:2]] == [
            "client_tool_call",
            "tool_response_injection",
        ]
        assert "turn_type_column" not in turns[2].columns
        assert turns[2].columns["raw_messages_column"][0] == [USER_HELLO]

    @pytest.mark.smoke
    def test_metrics_only_finish_reasons_tool_loop(self, tmp_path: Path):
        """
        Metrics-only finish_reasons=tool_calls becomes a synthetic call plus injection.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                            finish_reasons=["tool_calls"],
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:02+00:00",
                            prompt_tokens=20,
                            completion_tokens=12,
                            finish_reasons=["stop"],
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert turns[0].columns["turn_type_column"] == ["client_tool_call"]
        assert turns[0].columns["tools_column"][0] == DEFAULT_SYNTHETIC_TOOLS
        assert turns[1].columns["turn_type_column"] == ["tool_response_injection"]
        assert turns[1].columns["tool_response_column"] == [
            settings.default_synthetic_tool_response
        ]
        assert "turn_type_column" not in turns[2].columns
        assert "text_column" in turns[2].columns

    @pytest.mark.sanity
    def test_metrics_only_execute_tool_span(self, tmp_path: Path):
        """
        A following execute_tool span marks the preceding LLM span as a tool call.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                        ),
                        execute_tool_span(
                            span_id="tool",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            result="rainy, 57F",
                        ),
                        ibm_chat_span(
                            span_id="s1",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:02+00:00",
                            prompt_tokens=20,
                            completion_tokens=12,
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert turns[0].columns["turn_type_column"] == ["client_tool_call"]
        assert turns[1].columns["tool_response_column"] == ["rainy, 57F"]
        assert "turn_type_column" not in turns[2].columns

    @pytest.mark.sanity
    def test_metrics_only_execute_tool_name_without_operation(self, tmp_path: Path):
        """
        Span name execute_tool is enough when gen_ai.operation.name is missing.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                        ),
                        execute_tool_span(
                            span_id="tool",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:01+00:00",
                            operation=False,
                            name="execute_tool get_weather",
                            result={"temp": 57},
                        ),
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert turns[0].columns["turn_type_column"] == ["client_tool_call"]
        assert json.loads(turns[1].columns["tool_response_column"][0]) == {"temp": 57}

    @pytest.mark.regression
    def test_tool_definitions_alone_are_not_a_tool_loop(self, tmp_path: Path):
        """
        gen_ai.tool.definitions without finish_reasons or execute_tool is not a call.

        ## WRITTEN BY AI ##
        """
        trace = write_jsonl(
            tmp_path,
            [
                ibm_session_line(
                    "t0",
                    [
                        ibm_chat_span(
                            span_id="s0",
                            trace_id="t0",
                            start_time="2024-01-01T12:00:00+00:00",
                            prompt_tokens=10,
                            completion_tokens=8,
                            tools=WEATHER_TOOLS,
                        )
                    ],
                )
            ],
        )
        turns = load_graph_turns(next(iter(deserialize(trace, content="synthetic"))))
        assert len(turns) == 1
        assert "turn_type_column" not in turns[0].columns
        assert "tools_column" not in turns[0].columns


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}],
            [{"role": "user", "content": "hello"}],
        ),
        (
            json.dumps([{"role": "user", "content": "hello"}]),
            [{"role": "user", "content": "hello"}],
        ),
        (
            [
                {
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "c1",
                            "result": "found",
                        }
                    ]
                },
            ],
            [{"role": "tool", "tool_call_id": "c1", "content": "found"}],
        ),
    ],
)
def test_parse_gen_ai_messages(value, expected):
    """
    JSON strings, OpenAI dicts, and OTel parts normalize to OpenAI chat messages.

    ## WRITTEN BY AI ##
    """
    assert parse_gen_ai_messages(value) == expected


def test_parse_gen_ai_messages_tool_call_parts():
    """
    OTel tool_call parts become OpenAI tool_calls with JSON-string arguments.

    ## WRITTEN BY AI ##
    """
    messages = parse_gen_ai_messages(
        [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "c1",
                        "name": "search",
                        "arguments": {"q": "x"},
                    }
                ],
            }
        ]
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] is None
    tool_call = messages[0]["tool_calls"][0]
    assert tool_call["id"] == "c1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "search"
    assert json.loads(tool_call["function"]["arguments"]) == {"q": "x"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-01T00:00:00+00:00", 1704067200.0),
        (1704067200, 1704067200.0),
        (1704067200_000, 1704067200.0),
        (1704067200_000_000_000, 1704067200.0),
    ],
)
def test_parse_span_timestamp(value, expected):
    """
    ISO strings, unix seconds, milliseconds, and nanoseconds convert to seconds.

    ## WRITTEN BY AI ##
    """
    assert parse_span_timestamp(value) == pytest.approx(expected)
