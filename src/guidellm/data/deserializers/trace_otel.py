"""
OpenTelemetry GenAI trace format.

Normalizes session-per-line and span-per-line OTEL files into the shared
replay row shape (timestamp, input_length, output_length), then generates
synthetic prompts like the other trace formats.
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
    decode_prompt,
    generate_token_ids,
)
from guidellm.data.schemas import InvalidRowError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas.data.deserializers import OTELTraceFormatArgs

__all__ = ["OTELTraceFormat"]

_LLM_OPERATIONS = frozenset({"chat", "generate", "text_completion"})
_NON_LLM_OPERATIONS = frozenset({"invoke_agent", "execute_tool", "embeddings"})
_FAILED_STATUS_CODES = frozenset({2, "2", "ERROR", "STATUS_CODE_ERROR"})
_NANOSECONDS = 1e16
_MILLISECONDS = 1e11

_OTEL_KINDS = ["otel", "opentelemetry", "otel_trace"]

DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, _OTEL_KINDS)


def span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    """Return the span ``attributes`` mapping.

    Nested JSON columns are already decoded by ``datasets``. Non-dict values
    are treated as missing so the span is skipped rather than parsed here.

    :param span: One OTEL span dict.
    :return: Attribute mapping, or an empty dict when absent or not a dict.
    """
    attrs = span.get("attributes") or {}
    return attrs if isinstance(attrs, dict) else {}


def first_attribute(attributes: dict[str, Any], keys: list[str]) -> Any:
    """Return the first present, non-null attribute value from ``keys``.

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


def parse_span_timestamp(value: Any) -> float:
    """Convert an OTEL span timestamp to epoch seconds.

    Accepts ISO-8601 strings, ``datetime`` objects, unix seconds, milliseconds,
    and nanoseconds.

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
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) >= _NANOSECONDS:
            return number / 1e9
        if abs(number) >= _MILLISECONDS:
            return number / 1e3
        return number
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
    return status.get("code") in _FAILED_STATUS_CODES


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


def _replay_rows_from_spans(
    spans: list[dict[str, Any]], config: OTELTraceFormatArgs
) -> list[dict[str, Any]]:
    """Filter LLM spans and flatten them into timestamp-sorted replay rows."""
    rows: list[dict[str, Any]] = []
    for span in spans:
        if not is_llm_span(span, config):
            continue
        row = span_to_replay_row(span, config)
        if row is None:
            continue
        rows.append(row)
    rows.sort(key=lambda row: row[config.timestamp_column])
    return rows


@TraceFormatRegistry.register(_OTEL_KINDS)
class OTELTraceFormat(TraceFormatBase):
    """OpenTelemetry GenAI traces flattened into timed synthetic-prompt turns.

    Each ``trace_id`` is one conversation. Relative timestamps and the growing
    synthetic prefix reset between conversations, matching WEKA session scope.
    """

    def __init__(self, config: OTELTraceFormatArgs, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset
        self._prefix_token_ids: tuple[int, ...] = ()
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
        self._prefix_token_ids = ()

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
        self, row: dict, processor: PreTrainedTokenizerBase, faker: Faker
    ) -> str:
        n_in = int(row[self.config.prompt_tokens_column])
        if n_in <= 0:
            return ""
        # Later turns in a trace usually grow the prompt; reuse earlier tokens
        # as a shared prefix so multi-turn KV-cache behavior is not lost.
        if len(self._prefix_token_ids) < n_in:
            extra = generate_token_ids(
                n_in - len(self._prefix_token_ids), processor, faker
            )
            self._prefix_token_ids = self._prefix_token_ids + extra
        return decode_prompt(processor, list(self._prefix_token_ids[:n_in]))

    def build_conversation_graph(
        self,
        conversation: Dataset,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> ConversationGraphData:
        """Build a linear conversation from one ``__iter__`` stub.

        :param conversation: One-row Dataset yielded by ``__iter__``.
        :param processor: Tokenizer used to synthesize prompts.
        :param faker: Seeded faker for token generation.
        :return: Linear ``main_*`` graph for the unpacked span list.
        :raises InvalidRowError: If the conversation has no LLM spans to replay
            or a replay row fails validation.
        """
        spans = self._unpack_conversation(conversation)
        rows = _replay_rows_from_spans(spans, self.config)
        if not rows:
            raise InvalidRowError(
                "OTEL format: conversation has no LLM spans with token counts to replay"
            )
        return self._build_linear_chain(rows, processor, faker)

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
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> ConversationGraphData:
        """Emit the shared linear ``main_*`` chain from flattened replay rows."""
        start_ts = rows[0][self.config.timestamp_column]
        turns: list[ConversationTurnData] = []
        for turn_idx, turn in enumerate(rows):
            parents = []
            if turn_idx > 0:
                parents.append(
                    ConversationParentRef(parent_node_id=f"main_{turn_idx - 1}")
                )
            _validate_api_row(turn, self.config, self.validate_row)
            prompt = self.create_prompt(turn, processor, faker)
            relative_timestamp = turn[self.config.timestamp_column] - start_ts
            columns = {
                "text_column": [prompt],
                "prompt_tokens_count_column": [turn[self.config.prompt_tokens_column]],
                "output_tokens_count_column": [turn[self.config.output_tokens_column]],
                "relative_timestamp_column": [relative_timestamp],
            }
            turns.append(
                ConversationTurnData(
                    node_id=f"main_{turn_idx}",
                    agent_id="default",
                    parents=parents,
                    columns=columns,
                )
            )
        return ConversationGraphData(turns=turns)
