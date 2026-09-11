from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.deserializers.trace_otel import parse_span_timestamp
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)
from guidellm.schemas.data import OTELTraceFormatArgs
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
            config=OTELTraceFormatArgs(kind=kind, source=trace_file_source(trace)),
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
        turns = load_graph_turns(next(iter(deserialize(trace))))
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
        turns = load_graph_turns(next(iter(deserialize(trace))))
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
        conversations = [load_graph_turns(row) for row in deserialize(trace)]
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
        conversations = [load_graph_turns(row) for row in deserialize(trace)]
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
        turns = load_graph_turns(next(iter(deserialize(trace))))
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
        turns = load_graph_turns(next(iter(deserialize(trace))))
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
        turns = load_graph_turns(next(iter(deserialize(trace))))
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
        ds = deserialize(trace)
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
        ds = deserialize(trace)
        row_iter = iter(ds)
        turns = load_graph_turns(next(row_iter))
        assert len(turns) == 1
        assert turns[0].columns["prompt_tokens_count_column"][0] == 4
        with pytest.raises(InvalidRowError, match="no LLM spans"):
            next(row_iter)

    @pytest.mark.sanity
    def test_prefix_reuse_within_conversation(self, tmp_path: Path):
        """
        Later turns in a trace reuse the earlier turn's synthetic token prefix.

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
        turns = load_graph_turns(next(iter(deserialize(trace))))
        first = turns[0].columns["text_column"][0]
        second = turns[1].columns["text_column"][0]
        assert first.split() == ["tok0", "tok1", "tok2", "tok3"]
        assert second.split()[:4] == first.split()
        assert len(second.split()) == 8


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
