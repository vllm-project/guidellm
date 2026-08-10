from __future__ import annotations

from typing import Any

from guidellm.data.finalizers.conversation_graph import (
    expand_client_tool_turns,
    turns_from_mapped_items,
)
from guidellm.data.finalizers.finalizer import DatasetFinalizer, FinalizerRegistry
from guidellm.scheduler.schemas import HistoryContext
from guidellm.scheduler.schemas.conversation_graph import (
    GenerativeConversationGraph,
    GenerativeConversationNode,
)
from guidellm.schemas import (
    GenerationRequest,
    RequestSettings,
    TurnType,
    UsageMetrics,
)
from guidellm.schemas.data.finalizers import GenerativeRequestFinalizerArgs

__all__ = [
    "GenerativeRequestFinalizer",
]


@FinalizerRegistry.register("generative")
class GenerativeRequestFinalizer(DatasetFinalizer[GenerativeConversationGraph | list]):
    """
    Finalizer that converts dataset rows into a GenerativeConversationGraph,
    aggregating usage metrics from the provided columns.
    """

    def __init__(self, config: GenerativeRequestFinalizerArgs) -> None:
        self.config = config

    def __call__(
        self, items: list[dict[str, Any]]
    ) -> GenerativeConversationGraph | list:
        graph_data = turns_from_mapped_items(items)
        if not graph_data.turns:
            return []

        graph_data = expand_client_tool_turns(
            graph_data, tool_call_mode=self.config.tool_call_mode
        )

        nodes: dict[str, GenerativeConversationNode] = {}
        parents_by_node: dict[str, list[tuple[str, HistoryContext]]] = {}
        for turn in graph_data.turns:
            request, column_settings = self.finalize_turn(dict(turn.columns))
            settings = turn.settings if turn.settings is not None else column_settings
            nodes[turn.node_id] = GenerativeConversationNode(
                node_id=turn.node_id,
                agent_id=turn.agent_id,
                request=request,
                settings=settings,
            )
            parents_by_node[turn.node_id] = [
                (parent.parent_node_id, parent.history_context)
                for parent in turn.parents
            ]

        return GenerativeConversationGraph.from_nodes_with_parents(
            nodes=nodes,
            parents_by_node=parents_by_node,
            graph_id=graph_data.graph_id,
        )

    def finalize_turn(  # noqa: C901 PLR0912 PLR0915
        self, columns: dict[str, Any]
    ) -> tuple[GenerationRequest, RequestSettings]:
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Sum prompt token column
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Sum output token column
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens

        # Count words in prefixes
        for prefix in columns.get("prefix_column", []):
            if not prefix:
                continue

            input_metrics.add_text_metrics(prefix)

        # Count words in text prompts
        for text in columns.get("text_column", []):
            if not text:
                continue

            input_metrics.add_text_metrics(text)

        # Count pixels and bytes in images
        for image in columns.get("image_column", []):
            if not image:
                continue

            if (image_pixels := image.get("image_pixels")) is not None:
                input_metrics.image_pixels = (
                    input_metrics.image_pixels or 0
                ) + image_pixels
            if (image_bytes := image.get("image_bytes")) is not None:
                input_metrics.image_bytes = (
                    input_metrics.image_bytes or 0
                ) + image_bytes

        # Count frames, seconds, and bytes in videos
        for video in columns.get("video_column", []):
            if not video:
                continue

            if (video_frames := video.get("video_frames")) is not None:
                input_metrics.video_frames = (
                    input_metrics.video_frames or 0
                ) + video_frames
            if (video_seconds := video.get("video_seconds")) is not None:
                input_metrics.video_seconds = (
                    input_metrics.video_seconds or 0.0
                ) + video_seconds
            if (video_bytes := video.get("video_bytes")) is not None:
                input_metrics.video_bytes = (
                    input_metrics.video_bytes or 0
                ) + video_bytes

        # Count samples, seconds, and bytes in audio
        for audio in columns.get("audio_column", []):
            if not audio:
                continue

            if (audio_samples := audio.get("audio_samples")) is not None:
                input_metrics.audio_samples = (
                    input_metrics.audio_samples or 0
                ) + audio_samples
            if (audio_seconds := audio.get("audio_seconds")) is not None:
                input_metrics.audio_seconds = (
                    input_metrics.audio_seconds or 0.0
                ) + audio_seconds
            if (audio_bytes := audio.get("audio_bytes")) is not None:
                input_metrics.audio_bytes = (
                    input_metrics.audio_bytes or 0
                ) + audio_bytes

        # Resolve turn type with priority:
        # 1. Explicit turn_type_column (from synthetic server_tool_call_turns
        #    or hand-crafted datasets)
        # 2. tools_column presence + tool_call_mode config
        # 3. Default to "standard"
        turn_type_values = columns.get("turn_type_column", [])
        turn_type: TurnType
        if turn_type_values and turn_type_values[0]:
            turn_type = turn_type_values[0]
        elif columns.get("tools_column"):
            if self.config.tool_call_mode == "server":
                turn_type = "server_tool_call"
                # Tools are server-managed; strip data-provided definitions
                # so the request handler doesn't inject them.
                columns.pop("tools_column", None)
                columns.pop("tool_response_column", None)
            else:
                turn_type = "client_tool_call"
        else:
            turn_type = "standard"

        # Scheduling metadata belongs on RequestSettings, not request columns.
        relative_timestamp = self._get_optional_column_value(
            columns, "relative_timestamp_column"
        )
        requeue_delay = self._get_optional_column_value(columns, "requeue_delay_column")
        columns.pop("relative_timestamp_column", None)
        columns.pop("requeue_delay_column", None)

        return GenerationRequest(
            columns=columns,
            turn_type=turn_type,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        ), RequestSettings(
            relative_timestamp=relative_timestamp,
            requeue_delay=requeue_delay,
        )

    @staticmethod
    def _get_optional_column_value(
        columns: dict[str, Any],
        column_name: str,
    ) -> Any | None:
        """
        Retrieve the first value from a specified column in the columns dictionary.

        :param columns: A dictionary containing columns.
        :param column_name: The name of the column to retrieve the value from.
        :return: The first value from the specified column
        """
        values = columns.get(column_name, [])
        return values[0] if values else None
