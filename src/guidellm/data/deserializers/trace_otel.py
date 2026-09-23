"""
OpenTelemetry GenAI trace format.

Normalizes session-per-line and span-per-line OTEL files into timed
conversation graphs. Replay sends recorded ``gen_ai.input.messages``,
either as each span's full input or as new-message deltas with DAG
runtime history of live completions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset, Features
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import (
    TraceDatasetDeserializer,
    TraceFormatBase,
    TraceFormatRegistry,
    _validate_api_row,
)
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.scheduler.schemas import HistoryContext
from guidellm.schemas.data.deserializers import OTELTraceFormatArgs
from guidellm.schemas.data.deserializers.synthetic import DEFAULT_SYNTHETIC_TOOLS
from guidellm.settings import settings
from guidellm.utils.imports import json

__all__ = ["OTELTraceFormat", "parse_gen_ai_messages"]

_LLM_OPERATIONS = frozenset({"chat", "generate", "text_completion"})
_NON_LLM_OPERATIONS = frozenset({"invoke_agent", "execute_tool", "embeddings"})
_FAILED_STATUS_CODE = 2
_TOOL_FINISH_REASONS = frozenset(
    {"tool_calls", "tool_call", "tool_use", "function_call"}
)

_OTEL_KINDS = ["otel", "opentelemetry"]

DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, _OTEL_KINDS)


def span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    """Return the span ``attributes`` mapping.

    Nested JSON columns are already decoded by ``datasets``. Non-dict values
    are treated as missing so the span is skipped rather than parsed here.

    :param span: One OTEL span dict.
    :return: Attribute mapping, or an empty dict when absent or not a dict.
    """
    attrs = span.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def first_attribute(attributes: dict[str, Any], keys: list[str]) -> Any:
    """Return the first present, non-null attribute value from ``keys``.

    Keys are tried in list order so current GenAI names can fall back to
    deprecated aliases.

    :param attributes: Span attribute mapping.
    :param keys: Keys to try in order.
    :return: The first matching value, or ``None``.
    """
    for key in keys:
        value = attributes.get(key)
        if value is not None:
            return value
    return None


def usage_tokens(
    attributes: dict[str, Any], config: OTELTraceFormatArgs
) -> tuple[int, int] | None:
    """Read prompt and output token counts from GenAI usage attributes.

    :param attributes: Span attribute mapping.
    :param config: Format args with attribute key fallbacks.
    :return: ``(prompt_tokens, output_tokens)``, or ``None`` if either is missing.
    """
    prompt = first_attribute(attributes, config.input_tokens_attributes)
    output = first_attribute(attributes, config.output_tokens_attributes)
    if prompt is None or output is None:
        return None
    return int(prompt), int(output)


def _json_dumps(value: Any) -> str:
    dumped = json.dumps(value)
    if isinstance(dumped, bytes):
        return dumped.decode()
    return dumped


def parse_gen_ai_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize ``gen_ai.input/output.messages`` to OpenAI chat dicts.

    Accepts a JSON string or a list. OTel ``parts`` (``text``, ``tool_call``,
    ``tool_call_response``) become ``{role, content}`` / ``tool_calls`` /
    ``role=tool`` messages. Values that are already OpenAI-shaped are passed
    through with tool-call arguments normalized to JSON strings.

    :param value: Raw attribute value (string, list, or ``None``).
    :return: OpenAI chat message dicts. Empty when ``value`` is missing.
    :raises InvalidRowError: If the value cannot be parsed as a message list.
    """
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str | bytes):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRowError(
                f"OTEL format: gen_ai messages are not valid JSON: {value!r}"
            ) from exc
    if not isinstance(parsed, list):
        raise InvalidRowError(
            f"OTEL format: gen_ai messages must be a list, got {type(parsed).__name__}"
        )
    messages: list[dict[str, Any]] = []
    for item in parsed:
        messages.extend(_normalize_message(item))
    return messages


def span_output_messages(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant (or tool) messages recorded on a completed span.

    Prefers ``gen_ai.output.messages``. Falls back to ``gen_ai.output.text``
    as a single assistant message. Used for runtime-history prefix checks,
    never as the live completion.

    :param attributes: Span attribute mapping.
    :return: Normalized OpenAI chat dicts, possibly empty.
    """
    raw = attributes.get("gen_ai.output.messages")
    messages = parse_gen_ai_messages(raw)
    if messages:
        return messages
    text = attributes.get("gen_ai.output.text")
    if text is None or text == "":
        return []
    return [{"role": "assistant", "content": str(text)}]


def span_tool_definitions(attributes: dict[str, Any]) -> Any | None:
    """Return ``gen_ai.tool.definitions`` decoded from JSON if needed.

    :param attributes: Span attribute mapping.
    :return: Tool definitions as stored on the span, or ``None``.
    """
    defs = attributes.get("gen_ai.tool.definitions")
    if defs is None or defs == "":
        return None
    if isinstance(defs, str | bytes):
        try:
            return json.loads(defs)
        except (TypeError, ValueError) as exc:
            raise InvalidRowError(
                "OTEL format: gen_ai.tool.definitions is not valid JSON"
            ) from exc
    return defs


def _normalize_message(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        raise InvalidRowError(
            f"OTEL format: message must be an object, got {type(item).__name__}"
        )
    parts = item.get("parts")
    if isinstance(parts, list):
        return _messages_from_parts(item.get("role"), parts)
    return [_normalize_openai_message(item)]


def _normalize_openai_message(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role") or "user"
    message: dict[str, Any] = {"role": role}
    if "content" in item:
        message["content"] = item["content"]
    if item.get("tool_calls"):
        message["tool_calls"] = [
            _normalize_openai_tool_call(call) for call in item["tool_calls"]
        ]
        message.setdefault("content", None)
    if role == "tool":
        tool_call_id = item.get("tool_call_id") or item.get("id")
        if tool_call_id is not None:
            message["tool_call_id"] = str(tool_call_id)
        if "name" in item:
            message["name"] = item["name"]
        if "content" not in message:
            result = item.get("result")
            message["content"] = "" if result is None else result
    return message


def _normalize_openai_tool_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise InvalidRowError("OTEL format: tool_calls entries must be objects")
    function = call.get("function")
    if not isinstance(function, dict):
        function = {}
    arguments = function.get("arguments", call.get("arguments", {}))
    if not isinstance(arguments, str):
        arguments = _json_dumps(arguments if arguments is not None else {})
    name = function.get("name") or call.get("name") or ""
    return {
        "id": str(call.get("id") or call.get("tool_call_id") or ""),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _messages_from_parts(role: Any, parts: list[Any]) -> list[dict[str, Any]]:
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_responses: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            content = part.get("content", part.get("text", ""))
            texts.append("" if content is None else str(content))
        elif part_type == "tool_call":
            tool_calls.append(_tool_call_from_part(part))
        elif part_type == "tool_call_response":
            tool_responses.append(_tool_response_from_part(part))
    messages: list[dict[str, Any]] = []
    if texts or tool_calls:
        message: dict[str, Any] = {
            "role": role or ("assistant" if tool_calls else "user"),
        }
        if texts:
            message["content"] = "".join(texts)
        elif tool_calls:
            message["content"] = None
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
    messages.extend(tool_responses)
    return messages


def _tool_call_from_part(part: dict[str, Any]) -> dict[str, Any]:
    function = part.get("function")
    arguments = part.get("arguments")
    name = part.get("name")
    if isinstance(function, dict):
        if arguments is None:
            arguments = function.get("arguments")
        if not name:
            name = function.get("name")
    if not isinstance(arguments, str):
        arguments = _json_dumps(arguments if arguments is not None else {})
    return {
        "id": str(part.get("id") or part.get("tool_call_id") or ""),
        "type": "function",
        "function": {"name": name or "", "arguments": arguments},
    }


def _tool_response_from_part(part: dict[str, Any]) -> dict[str, Any]:
    content = part.get("result", part.get("response", part.get("content", "")))
    if not isinstance(content, str):
        content = _json_dumps(content if content is not None else "")
    return {
        "role": "tool",
        "tool_call_id": str(part.get("id") or part.get("tool_call_id") or ""),
        "content": content,
    }


def parse_span_timestamp(value: Any) -> float:
    """Convert an OTEL span ``start_time`` to epoch seconds.

    Known dumps use ISO-8601 strings: naive on IBM synthetic-conversations
    and Exgentic-family traces, offset on IBM lmcache. A ``Z`` suffix is the
    same ISO family. HuggingFace may decode nested ``start_time`` to
    ``datetime``.

    :param value: Raw ``start_time`` (or equivalent) field.
    :return: Timestamp in seconds.
    :raises InvalidRowError: If the value cannot be parsed.
    """
    if isinstance(value, datetime):
        return _datetime_to_epoch_seconds(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidRowError(
                f"OTEL format: unsupported timestamp string {value!r}"
            ) from exc
        return _datetime_to_epoch_seconds(parsed)
    raise InvalidRowError(
        f"OTEL format: unsupported timestamp type {type(value).__name__}: {value!r}"
    )


def _datetime_to_epoch_seconds(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def is_failed_span(span: dict[str, Any]) -> bool:
    """Return whether the span ended in an error status.

    Failed calls are dropped so retries and 4xx/5xx responses are not replayed
    as successful load.

    :param span: One OTEL span dict.
    :return: ``True`` when ``status.code`` is an error.
    """
    status = span.get("status")
    if not isinstance(status, dict):
        return False
    return status.get("code") == _FAILED_STATUS_CODE


def is_llm_span(span: dict[str, Any], config: OTELTraceFormatArgs) -> bool:
    """Return whether ``span`` is a successful LLM request worth replaying.

    :param span: One OTEL span dict.
    :param config: Format args used to detect usage token attributes.
    :return: ``True`` for chat/generate/text_completion spans, or any span
        that already carries usage token counts.
    """
    if is_failed_span(span):
        return False
    attributes = span_attributes(span)
    operation = attributes.get("gen_ai.operation.name")
    if operation in _NON_LLM_OPERATIONS:
        return False
    if operation in _LLM_OPERATIONS:
        return True
    return usage_tokens(attributes, config) is not None


def finish_reasons_indicate_tool_call(attributes: dict[str, Any]) -> bool:
    """Return whether ``gen_ai.response.finish_reasons`` records a tool call.

    Accepts a list or a JSON string of a list (nested JSON encoding used by
    Exgentic-family dumps). Recognized values are ``tool_calls``,
    ``tool_call``, ``tool_use``, and ``function_call``.

    :param attributes: Span attribute mapping.
    :return: ``True`` when any recorded finish reason is a tool-call alias.
    """
    raw = attributes.get("gen_ai.response.finish_reasons")
    if raw is None or raw == "":
        return False
    parsed: Any = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return False
    if not isinstance(parsed, list):
        return False
    return any(str(item) in _TOOL_FINISH_REASONS for item in parsed)


def messages_have_tool_calls(messages: list[dict[str, Any]]) -> bool:
    """Return whether any OpenAI-shaped message carries ``tool_calls``.

    :param messages: Normalized chat messages.
    :return: ``True`` when at least one assistant message requested tools.
    """
    return any(bool(message.get("tool_calls")) for message in messages)


def stringify_tool_result(value: Any) -> str:
    """Coerce a recorded tool result to a string for ``tool_response_column``.

    :param value: Result content (string, bytes, or JSON-serializable).
    :return: String payload to inject.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _json_dumps(value)


def execute_tool_result_text(span: dict[str, Any]) -> str | None:
    """Return ``gen_ai.tool.call.result`` from an execute-tool span, if set.

    :param span: One OTEL span dict.
    :return: Stringified result, or ``None`` when the attribute is absent.
    """
    result = span_attributes(span).get("gen_ai.tool.call.result")
    if result is None or result == "":
        return None
    return stringify_tool_result(result)


def execute_tool_spans_between(
    spans: list[dict[str, Any]],
    start_ts: float,
    end_ts: float,
    config: OTELTraceFormatArgs,
) -> list[dict[str, Any]]:
    """Return execute-tool spans with start times in ``(start_ts, end_ts)``.

    Pairing is by timestamp order in the conversation group, not
    ``parent_span_id``. Spans whose timestamps cannot be parsed are skipped.

    :param spans: Raw spans in one ``trace_id`` group.
    :param start_ts: Exclusive lower bound (typically the LLM span start).
    :param end_ts: Exclusive upper bound (next LLM span start, or infinity).
    :param config: Format args for the timestamp field name.
    :return: Matching execute-tool spans, sorted by start time.
    """
    matched: list[tuple[float, dict[str, Any]]] = []
    for span in spans:
        if not span_attributes(span).get("gen_ai.operation.name") == "execute_tool":
            continue
        try:
            timestamp = parse_span_timestamp(span.get(config.span_timestamp_field))
        except InvalidRowError:
            continue
        if start_ts < timestamp < end_ts:
            matched.append((timestamp, span))
    matched.sort(key=lambda item: item[0])
    return [span for _, span in matched]


def tool_result_delta(prev: dict[str, Any], curr: dict[str, Any]) -> list[str] | None:
    """Return tool-result strings when ``curr`` is a pure tool-result continuation.

    Requires ``input[curr] == input[prev] + output[prev] + delta`` and that
    every delta message is a tool result. Prefix mismatch, missing messages,
    or a mixed (non-tool) delta return ``None`` so the caller can synthesize
    a placeholder without consuming ``curr`` as an injection.

    :param prev: LLM replay row for the tool-call span.
    :param curr: LLM replay row for the candidate follow-up span.
    :return: Tool result content strings, or ``None`` when this is not an injection.
    """
    prev_attrs = span_attributes(prev["span"])
    curr_attrs = span_attributes(curr["span"])
    prev_input = parse_gen_ai_messages(prev_attrs.get("gen_ai.input.messages"))
    prev_output = span_output_messages(prev_attrs)
    curr_input = parse_gen_ai_messages(curr_attrs.get("gen_ai.input.messages"))
    if not prev_input or not prev_output or not curr_input:
        return None
    prefix = prev_input + prev_output
    if curr_input[: len(prefix)] != prefix:
        return None
    delta = curr_input[len(prefix) :]
    if not delta or any(message.get("role") != "tool" for message in delta):
        return None
    return [stringify_tool_result(message.get("content")) for message in delta]


def span_to_replay_row(
    span: dict[str, Any], config: OTELTraceFormatArgs
) -> dict[str, Any] | None:
    """Flatten one LLM span into the shared replay column shape.

    :param span: One OTEL span dict.
    :param config: Format args for timestamp and token field names.
    :return: Replay row, or ``None`` when usage tokens are missing.
    :raises InvalidRowError: If the span timestamp cannot be parsed.
    """
    tokens = usage_tokens(span_attributes(span), config)
    if tokens is None:
        return None
    prompt_tokens, output_tokens = tokens
    timestamp = parse_span_timestamp(span.get(config.span_timestamp_field))
    return {
        config.timestamp_column: timestamp,
        config.prompt_tokens_column: prompt_tokens,
        config.output_tokens_column: output_tokens,
    }


def nested_spans_column(dataset: Dataset, config: OTELTraceFormatArgs) -> str | None:
    """Return the nested spans column name when the file is session-per-line.

    :param dataset: Loaded trace dataset.
    :param config: Format args with ``spans_column``.
    :return: Column name, or ``None`` for span-per-line files.
    """
    name = config.spans_column
    if name not in dataset.column_names:
        return None
    sample = dataset[name][0]
    if isinstance(sample, list):
        return name
    return None


def iter_raw_span_groups(
    dataset: Dataset, config: OTELTraceFormatArgs
) -> Iterator[list[dict[str, Any]]]:
    """Yield raw span lists, one conversation at a time.

    Session-per-line files already store spans per row. Span-per-line files are
    grouped by consecutive equal ``trace_id`` values; interleaved ids become
    separate conversations.

    :param dataset: Loaded trace dataset.
    :param config: Format args for column names.
    :return: Iterator of span lists.
    """
    spans_col = nested_spans_column(dataset, config)
    if spans_col is not None:
        for row in dataset:
            spans = row[spans_col] or []
            yield [span for span in spans if isinstance(span, dict)]
        return

    current_id: str | None = None
    current: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        span = dict(row)
        group_id = _span_group_id(span, config, index)
        if current and group_id != current_id:
            yield current
            current = []
        current_id = group_id
        current.append(span)
    if current:
        yield current


def _span_group_id(
    span: dict[str, Any], config: OTELTraceFormatArgs, index: int
) -> str:
    trace_id = span.get(config.trace_id_column)
    if trace_id:
        return str(trace_id)
    span_id = span.get("span_id")
    if span_id:
        return str(span_id)
    return f"span_{index}"


def _llm_replay_spans(
    spans: list[dict[str, Any]], config: OTELTraceFormatArgs
) -> list[dict[str, Any]]:
    """Return timestamp-sorted LLM spans that carry usage token counts."""
    rows: list[dict[str, Any]] = []
    for span in spans:
        if not is_llm_span(span, config):
            continue
        row = span_to_replay_row(span, config)
        if row is None:
            continue
        rows.append(
            {
                "span": span,
                "timestamp": row[config.timestamp_column],
                "prompt_tokens": row[config.prompt_tokens_column],
                "output_tokens": row[config.output_tokens_column],
            }
        )
    rows.sort(key=lambda item: item["timestamp"])
    return rows


@TraceFormatRegistry.register(_OTEL_KINDS)
class OTELTraceFormat(TraceFormatBase):
    """OpenTelemetry GenAI traces replayed as timed conversation graphs.

    Each ``trace_id`` is one conversation. Replay sends recorded
    ``gen_ai.input.messages``. ``history`` selects whether each call resends
    the full trace input or only the new messages with DAG runtime history.
    """

    def __init__(self, config: OTELTraceFormatArgs, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset
        # Filled by each ``__iter__`` pass so nested span lists are not forced
        # through a HuggingFace Arrow table.
        self._conversations: list[list[dict[str, Any]]] = []

    def __iter__(self) -> Iterable[Dataset]:
        self._conversations = []
        for spans in iter_raw_span_groups(self.dataset, self.config):
            index = len(self._conversations)
            self._conversations.append([dict(span) for span in spans])
            yield Dataset.from_dict({"_otel_index": [index]})

    def reset(self) -> None:
        return

    def required_columns(self) -> Features:
        return Features({})

    def find_required_columns(self, columns: list[str]) -> list[str]:
        # Only the first row is searchable here. Missing fields on later
        # conversations are skipped at iteration via InvalidRowError.
        _ = columns
        if nested_spans_column(self.dataset, self.config) is not None:
            return []
        sample = dict(self.dataset[0])
        if self.config.span_timestamp_field in sample or "attributes" in sample:
            return []
        return [self.config.span_timestamp_field]

    def validate_row(
        self,
        row: dict,  # noqa: ARG002
    ) -> None:
        return

    def create_prompt(
        self,
        row: dict,  # noqa: ARG002
        processor: PreTrainedTokenizerBase,  # noqa: ARG002
        faker: Faker,  # noqa: ARG002
    ) -> str:
        # OTEL overrides ``build_conversation_graph`` and sends recorded messages.
        return ""

    def build_conversation_graph(
        self,
        conversation: Dataset,
        processor: PreTrainedTokenizerBase,  # noqa: ARG002
        faker: Faker,  # noqa: ARG002
    ) -> ConversationGraphData:
        """Build a linear conversation from one ``__iter__`` stub.

        :param conversation: One-row Dataset yielded by ``__iter__``.
        :param processor: Unused; recorded messages do not need synthesis.
        :param faker: Unused; recorded messages do not need synthesis.
        :return: Linear ``main_*`` graph for the unpacked span list.
        :raises InvalidRowError: If the conversation has no LLM spans to replay
            or a replay row fails validation.
        """
        spans = self._unpack_conversation(conversation)
        rows = _llm_replay_spans(spans, self.config)
        if not rows:
            raise InvalidRowError(
                "OTEL format: conversation has no LLM spans with token counts to replay"
            )
        return self._build_linear_chain(rows, spans)

    def _unpack_conversation(self, conversation: Dataset) -> list[dict[str, Any]]:
        """Look up the raw span list for a conversation stub from ``__iter__``.

        Nested span lists are kept in ``self._conversations`` rather than the
        yielded Dataset. ``conversation[0]["_otel_index"]`` is the matching
        index from that pass.

        :param conversation: One-row Dataset yielded by ``__iter__``.
        :return: Raw span dicts for that conversation.
        """
        index = int(conversation[0]["_otel_index"])
        return self._conversations[index]

    def _build_linear_chain(
        self,
        rows: list[dict[str, Any]],
        all_spans: list[dict[str, Any]],
    ) -> ConversationGraphData:
        """Emit the linear ``main_*`` chain, pre-splitting recorded tool loops.

        Walk LLM spans with lookahead. A tool-call span becomes
        ``client_tool_call`` plus a ``tool_response_injection``. When the next
        span's new messages are only tool results, that span is the injection
        (not a second chat turn). Otherwise a placeholder injection is
        synthesized and the next span is still replayed.
        """
        start_ts = rows[0]["timestamp"]
        history_context: HistoryContext = (
            "full" if self.config.history == "runtime" else "new"
        )
        turns: list[ConversationTurnData] = []
        index = 0
        while index < len(rows):
            turn = rows[index]
            replay_row = {
                self.config.timestamp_column: turn["timestamp"],
                self.config.prompt_tokens_column: turn["prompt_tokens"],
                self.config.output_tokens_column: turn["output_tokens"],
            }
            _validate_api_row(replay_row, self.config, self.validate_row)
            if self._is_tool_call_turn(index, rows):
                index = self._append_tool_loop(
                    turns,
                    rows,
                    all_spans,
                    index,
                    start_ts,
                    history_context,
                )
                continue
            self._append_turn(
                turns,
                self._content_columns(index, turn, rows, turn["timestamp"] - start_ts),
                history_context,
            )
            index += 1
        return ConversationGraphData(turns=turns)

    def _append_turn(
        self,
        turns: list[ConversationTurnData],
        columns: dict[str, Any],
        history_context: HistoryContext,
        parent_history_context: HistoryContext | None = None,
    ) -> str:
        parents: list[ConversationParentRef] = []
        if turns:
            parents.append(
                ConversationParentRef(
                    parent_node_id=turns[-1].node_id,
                    history_context=parent_history_context or history_context,
                )
            )
        node_id = f"main_{len(turns)}"
        turns.append(
            ConversationTurnData(
                node_id=node_id,
                agent_id="default",
                parents=parents,
                columns=columns,
            )
        )
        return node_id

    def _is_tool_call_turn(
        self,
        index: int,
        rows: list[dict[str, Any]],
    ) -> bool:
        """Classify an LLM span as a client tool-call turn.

        Uses recorded output ``tool_calls``, else ``finish_reasons``. Tool
        definitions alone are not a trigger. ``execute_tool`` spans are not
        used to classify; they only supply placeholder injection text.
        """
        attributes = span_attributes(rows[index]["span"])
        if messages_have_tool_calls(span_output_messages(attributes)):
            return True
        return finish_reasons_indicate_tool_call(attributes)

    def _append_tool_loop(
        self,
        turns: list[ConversationTurnData],
        rows: list[dict[str, Any]],
        all_spans: list[dict[str, Any]],
        index: int,
        start_ts: float,
        history_context: HistoryContext,
    ) -> int:
        """Emit a call node plus one or more injections; return the next row index.

        When the following span is a pure tool-result continuation, consume it
        as the injection (and keep consuming while that injection also called
        tools). Otherwise synthesize a placeholder and leave the next span for
        the outer walk.
        """
        call = rows[index]
        self._append_turn(
            turns,
            self._tool_call_columns(index, call, rows, call["timestamp"] - start_ts),
            history_context,
        )
        current = index
        while True:
            next_index = current + 1
            delta = (
                tool_result_delta(rows[current], rows[next_index])
                if next_index < len(rows)
                else None
            )
            if delta is not None:
                next_row = rows[next_index]
                include_tools = self._is_tool_call_turn(next_index, rows)
                self._append_turn(
                    turns,
                    self._injection_columns(
                        responses=delta,
                        relative_timestamp=next_row["timestamp"] - start_ts,
                        output_tokens=next_row["output_tokens"],
                        tools_span=next_row["span"] if include_tools else None,
                    ),
                    history_context,
                    parent_history_context="full",
                )
                current = next_index
                if include_tools:
                    continue
                return current + 1
            responses = self._placeholder_or_execute_results(all_spans, rows, current)
            self._append_turn(
                turns,
                self._injection_columns(
                    responses=responses,
                    relative_timestamp=rows[current]["timestamp"] - start_ts,
                    output_tokens=None,
                    tools_span=None,
                ),
                history_context,
                parent_history_context="full",
            )
            return current + 1

    def _placeholder_or_execute_results(
        self,
        all_spans: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        index: int,
    ) -> list[str]:
        next_ts = (
            rows[index + 1]["timestamp"] if index + 1 < len(rows) else float("inf")
        )
        results: list[str] = []
        for span in execute_tool_spans_between(
            all_spans, rows[index]["timestamp"], next_ts, self.config
        ):
            text = execute_tool_result_text(span)
            if text is not None:
                results.append(text)
        if results:
            return results
        return [settings.default_synthetic_tool_response]

    def _tools_for_call(self, span: dict[str, Any]) -> Any:
        tools = span_tool_definitions(span_attributes(span))
        if tools is not None:
            return tools
        return DEFAULT_SYNTHETIC_TOOLS

    def _tool_call_columns(
        self,
        turn_idx: int,
        turn: dict[str, Any],
        rows: list[dict[str, Any]],
        relative_timestamp: float,
    ) -> dict[str, Any]:
        columns = self._content_columns(turn_idx, turn, rows, relative_timestamp)
        columns.pop("output_tokens_count_column", None)
        columns["turn_type_column"] = ["client_tool_call"]
        columns["tools_column"] = [self._tools_for_call(turn["span"])]
        columns["tool_choice_column"] = [self.config.tool_choice]
        return columns

    def _injection_columns(
        self,
        *,
        responses: list[str],
        relative_timestamp: float,
        output_tokens: int | None,
        tools_span: dict[str, Any] | None,
    ) -> dict[str, Any]:
        columns: dict[str, Any] = {
            "turn_type_column": ["tool_response_injection"],
            "tool_response_column": responses,
            "relative_timestamp_column": [relative_timestamp],
        }
        if output_tokens is not None:
            columns["output_tokens_count_column"] = [output_tokens]
        if tools_span is not None:
            columns["tools_column"] = [self._tools_for_call(tools_span)]
            columns["tool_choice_column"] = [self.config.tool_choice]
        return columns

    def _content_columns(
        self,
        turn_idx: int,
        turn: dict[str, Any],
        rows: list[dict[str, Any]],
        relative_timestamp: float,
    ) -> dict[str, Any]:
        return {
            "prompt_tokens_count_column": [turn["prompt_tokens"]],
            "output_tokens_count_column": [turn["output_tokens"]],
            "relative_timestamp_column": [relative_timestamp],
            "raw_messages_column": [self._raw_turn_messages(turn_idx, turn, rows)],
        }

    def _raw_turn_messages(
        self,
        turn_idx: int,
        turn: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        attributes = span_attributes(turn["span"])
        input_messages = parse_gen_ai_messages(attributes.get("gen_ai.input.messages"))
        if not input_messages:
            raise InvalidRowError(
                "OTEL format: span missing gen_ai.input.messages. "
                "Use kind=trace_synthetic for token-count-only traces."
            )
        if self.config.history != "runtime" or turn_idx == 0:
            return input_messages
        return self._runtime_raw_delta(input_messages, rows[turn_idx - 1])

    def _runtime_raw_delta(
        self,
        input_messages: list[dict[str, Any]],
        prev: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prev_attrs = span_attributes(prev["span"])
        prev_input = parse_gen_ai_messages(prev_attrs.get("gen_ai.input.messages"))
        prev_output = span_output_messages(prev_attrs)
        if not prev_output:
            raise InvalidRowError(
                "OTEL format: previous span has no gen_ai.output.messages or "
                "gen_ai.output.text for history=runtime prefix check"
            )
        prefix = prev_input + prev_output
        if input_messages[: len(prefix)] != prefix:
            raise InvalidRowError(
                "OTEL format: input messages do not continue the previous span "
                "(history=runtime requires input[i] == input[i-1] + "
                "output[i-1] + delta)"
            )
        return input_messages[len(prefix) :]
