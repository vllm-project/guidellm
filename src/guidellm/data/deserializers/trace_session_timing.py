"""Rewrite trace conversation timestamps to clamp waits, pack sessions, and scale time.

Applied in the dataset path after each conversation graph is built. Wait and
pack caps run in original trace seconds, then remaining
``relative_timestamp`` values are multiplied by ``time_scale``.
"""

from __future__ import annotations

from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)

__all__ = ["TraceSessionTiming"]


class TraceSessionTiming:
    """Compress intra-session gaps, inter-session idle, pack overlap, then scale.

    ``max_wait`` is applied independently inside each conversation.
    ``max_session_wait`` then clamps idle time from the previous session's
    last request to this session's first request. ``min_concurrent_sessions``
    then shifts this session earlier if needed so at least that many
    sessions overlap. ``time_scale`` multiplies the resulting timestamps.

    Caps are in unscaled trace seconds. Callers should construct a new
    instance per dataset iteration so packing state does not leak across epochs.
    """

    def __init__(
        self,
        max_wait: float | None = None,
        max_session_wait: float | None = None,
        min_concurrent_sessions: int | None = None,
        time_scale: float = 1.0,
    ) -> None:
        self.max_wait = max_wait
        self.max_session_wait = max_session_wait
        self.min_concurrent_sessions = min_concurrent_sessions
        self.time_scale = time_scale
        self._last_session_end: float | None = None
        self._first_session_start: float | None = None
        self._placed_session_ends: list[float] = []

    def apply(self, graph: ConversationGraphData) -> ConversationGraphData:
        """Rewrite ``relative_timestamp`` values on ``graph`` in place.

        :param graph: Conversation whose turn timestamps may be compressed
        :return: The same graph, after any timestamp rewrites
        """
        if self.max_wait is not None:
            self._compress_intra_session_gaps(graph)
        if self.max_session_wait is not None:
            self._compress_inter_session_gap(graph)
        if self.min_concurrent_sessions is not None:
            self._pack_min_concurrent_sessions(graph)
        if self.time_scale != 1.0:
            self._apply_time_scale(graph)
        return graph

    def _compress_intra_session_gaps(self, graph: ConversationGraphData) -> None:
        """Clamp serial gaps inside one session without affecting other sessions."""
        if self.max_wait is None:
            return

        timed: list[tuple[float, int]] = []
        for index, turn in enumerate(graph.turns):
            relative_timestamp = _turn_timestamp(turn)
            if relative_timestamp is None:
                continue
            timed.append((relative_timestamp, index))
        timed.sort()

        trim = 0.0
        last_ts: float | None = None
        for timestamp, index in timed:
            # Compare original gaps, not already-trimmed times, so close
            # requests stay close and only oversized gaps shrink.
            if last_ts is not None:
                gap = timestamp - last_ts
                if gap > self.max_wait:
                    trim += gap - self.max_wait
            last_ts = timestamp
            _set_turn_timestamp(graph.turns[index], timestamp - trim)

    def _compress_inter_session_gap(self, graph: ConversationGraphData) -> None:
        """Clamp idle time from the previous session's last request to this start.

        Overlapping sessions (for example WEKA conversations that each restart
        at time 0) are left in parallel. Only a positive gap on the shared
        timeline is shortened.
        """
        if self.max_session_wait is None:
            return

        bounds = self._session_bounds(graph)
        if bounds is None:
            return

        session_start, session_end = bounds
        if self._last_session_end is not None:
            gap = session_start - self._last_session_end
            if gap > self.max_session_wait:
                trim = gap - self.max_session_wait
                self._shift_session(graph, trim)
                session_end -= trim

        self._last_session_end = (
            session_end
            if self._last_session_end is None
            else max(self._last_session_end, session_end)
        )

    def _pack_min_concurrent_sessions(self, graph: ConversationGraphData) -> None:
        """Shift this session earlier so at least N sessions overlap.

        The first N sessions start together. Each later session starts when
        session ``i - N`` ends, which keeps N in flight during steady state.
        Sessions are never delayed past their current start.

        :param graph: Session whose timestamps may be shifted earlier
        """
        if self.min_concurrent_sessions is None:
            return

        bounds = self._session_bounds(graph)
        if bounds is None:
            return

        session_start, session_end = bounds
        placed = self._placed_session_ends
        target_count = self.min_concurrent_sessions
        if not placed:
            target_start = session_start
            self._first_session_start = session_start
        elif len(placed) < target_count:
            first_start = self._first_session_start
            target_start = first_start if first_start is not None else session_start
        else:
            # Start when the session from N slots ago ends, filling that lane.
            target_start = placed[len(placed) - target_count]

        new_start = min(session_start, target_start)
        self._shift_session(graph, session_start - new_start)
        placed.append(new_start + (session_end - session_start))

    def _apply_time_scale(self, graph: ConversationGraphData) -> None:
        """Multiply remaining timestamps after wait and pack caps."""
        for turn in graph.turns:
            relative_timestamp = _turn_timestamp(turn)
            if relative_timestamp is None:
                continue
            _set_turn_timestamp(turn, relative_timestamp * self.time_scale)

    def _session_bounds(
        self, graph: ConversationGraphData
    ) -> tuple[float, float] | None:
        times = [
            timestamp
            for timestamp in (_turn_timestamp(turn) for turn in graph.turns)
            if timestamp is not None
        ]
        if not times:
            return None
        return min(times), max(times)

    def _shift_session(self, graph: ConversationGraphData, trim: float) -> None:
        if trim <= 0:
            return
        for turn in graph.turns:
            relative_timestamp = _turn_timestamp(turn)
            if relative_timestamp is None:
                continue
            _set_turn_timestamp(turn, relative_timestamp - trim)


def _turn_timestamp(turn: ConversationTurnData) -> float | None:
    values = turn.columns.get("relative_timestamp_column")
    if values:
        return float(values[0])
    if turn.settings is not None:
        return turn.settings.relative_timestamp
    return None


def _set_turn_timestamp(turn: ConversationTurnData, timestamp: float) -> None:
    if turn.columns.get("relative_timestamp_column"):
        turn.columns["relative_timestamp_column"] = [timestamp]
    if turn.settings is not None and turn.settings.relative_timestamp is not None:
        turn.settings = turn.settings.model_copy(
            update={"relative_timestamp": timestamp}
        )
