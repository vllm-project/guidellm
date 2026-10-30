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

__all__ = [
    "TraceSessionTiming",
    "graph_max_timestamp",
    "graph_min_timestamp",
    "shift_graph_timestamps",
]


class TraceSessionTiming:
    """Compress intra-session gaps, inter-session idle, pack overlap, then scale.

    ``max_wait`` is applied independently inside each conversation.
    ``max_session_wait`` then clamps idle time from the previous session's
    last request to this session's first request. ``min_concurrent_sessions``
    shifts this session earlier if needed so at least that many sessions
    overlap; ``max_concurrent_sessions`` delays it so at most that many
    overlap. Sessions with no duration and no timestamp spread are left at their
    wait-capped start. ``time_scale`` multiplies the resulting timestamps.

    Caps are in unscaled trace seconds. Dataset copies apply wait caps per
    copy, place the copy, then pack with shared state, then scale. Callers
    should construct a new instance per dataset iteration so packing state
    does not leak across epochs.
    """

    def __init__(
        self,
        max_wait: float | None = None,
        max_session_wait: float | None = None,
        min_concurrent_sessions: int | None = None,
        max_concurrent_sessions: int | None = None,
        time_scale: float = 1.0,
    ) -> None:
        self.max_wait = max_wait
        self.max_session_wait = max_session_wait
        self.min_concurrent_sessions = min_concurrent_sessions
        self.max_concurrent_sessions = max_concurrent_sessions
        self.time_scale = time_scale
        self._last_session_end: float | None = None
        self._first_session_start: float | None = None
        self._placed_session_ends: list[float] = []

    def apply(self, graph: ConversationGraphData) -> ConversationGraphData:
        """Rewrite ``relative_timestamp`` values on ``graph`` in place.

        Wait caps, then packing, then ``time_scale``. Dataset copies split
        these steps so packing can run after copies are placed.

        :param graph: Conversation whose turn timestamps may be compressed
        :return: The same graph, after any timestamp rewrites
        """
        self.apply_wait_caps(graph)
        self.apply_pack(graph)
        self.apply_scale(graph)
        return graph

    def apply_wait_caps(self, graph: ConversationGraphData) -> ConversationGraphData:
        """Clamp intra-session and inter-session waits on ``graph``.

        :param graph: Conversation whose turn timestamps may be compressed
        :return: The same graph, after any wait-cap rewrites
        """
        if self.max_wait is not None:
            self._compress_intra_session_gaps(graph)
        if self.max_session_wait is not None:
            self._compress_inter_session_gap(graph)
        return graph

    def apply_pack(self, graph: ConversationGraphData) -> ConversationGraphData:
        """Pack ``graph`` to satisfy min/max concurrent session limits.

        :param graph: Conversation whose timestamps may be shifted
        :return: The same graph, after any packing shift
        """
        if (
            self.min_concurrent_sessions is not None
            or self.max_concurrent_sessions is not None
        ):
            self._pack_concurrent_sessions(graph)
        return graph

    def apply_scale(self, graph: ConversationGraphData) -> ConversationGraphData:
        """Multiply remaining timestamps after wait and pack caps.

        :param graph: Conversation whose timestamps may be scaled
        :return: The same graph, after any time-scale rewrite
        """
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

    def _pack_concurrent_sessions(self, graph: ConversationGraphData) -> None:
        """Place this session on shared lanes for min/max concurrent limits.

        The first ``min`` sessions start together when min is set; each later
        session may start when session ``i - min`` ends, but not later than
        its current start. When max is set, session ``i`` starts no earlier
        than session ``i - max`` ends. Min pull runs before max delay.
        Sessions with no recorded duration and no timestamp spread are not
        shifted.

        :param graph: Session whose timestamps may be shifted
        """
        bounds = _session_occupancy_bounds(graph)
        if bounds is None:
            return

        session_start, session_end = bounds
        placed = self._placed_session_ends
        if session_end == session_start:
            # Instantaneous (single-turn) session: packing cannot overlap
            # without collapsing distinct arrivals. Leave the start in place.
            if self._first_session_start is None:
                self._first_session_start = session_start
            placed.append(session_end)
            return

        new_start = self._packed_session_start(session_start, placed)
        if new_start < session_start:
            self._shift_session(graph, session_start - new_start)
        elif new_start > session_start:
            shift_graph_timestamps(graph, new_start - session_start)

        duration = session_end - session_start
        placed.append(new_start + duration)

    def _packed_session_start(self, session_start: float, placed: list[float]) -> float:
        """Compute the packed start time for a non-instantaneous session."""
        new_start = session_start
        min_count = self.min_concurrent_sessions
        if min_count is not None:
            if not placed:
                self._first_session_start = session_start
            elif placed:
                min_target = self._min_concurrent_target(
                    session_start, placed, min_count
                )
                new_start = min(new_start, min_target)

        max_count = self.max_concurrent_sessions
        if max_count is not None and len(placed) >= max_count:
            max_target = placed[len(placed) - max_count]
            new_start = max(new_start, max_target)
        return new_start

    def _min_concurrent_target(
        self, session_start: float, placed: list[float], min_count: int
    ) -> float:
        """Earliest start when pulling sessions earlier for min overlap."""
        if len(placed) < min_count:
            first_start = self._first_session_start
            return first_start if first_start is not None else session_start
        return placed[len(placed) - min_count]

    def _apply_time_scale(self, graph: ConversationGraphData) -> None:
        """Multiply timestamps and recorded durations after wait and pack caps."""
        for turn in graph.turns:
            relative_timestamp = _turn_timestamp(turn)
            if relative_timestamp is not None:
                _set_turn_timestamp(turn, relative_timestamp * self.time_scale)
            _scale_turn_duration(turn, self.time_scale)

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


def shift_graph_timestamps(graph: ConversationGraphData, offset: float) -> None:
    """Add ``offset`` to every turn that has a relative timestamp.

    Used to place dataset copies on the shared timeline.
    ``offset == 0`` is a no-op.

    :param graph: Conversation whose timestamps may be shifted later
    :param offset: Seconds to add to each present relative timestamp
    """
    if offset == 0:
        return
    for turn in graph.turns:
        relative_timestamp = _turn_timestamp(turn)
        if relative_timestamp is None:
            continue
        _set_turn_timestamp(turn, relative_timestamp + offset)


def graph_min_timestamp(graph: ConversationGraphData) -> float:
    """Return the earliest relative timestamp on ``graph``.

    :param graph: Conversation to inspect
    :return: Minimum relative timestamp among timed turns
    """
    return min(
        timestamp
        for timestamp in (_turn_timestamp(turn) for turn in graph.turns)
        if timestamp is not None
    )


def graph_max_timestamp(graph: ConversationGraphData) -> float:
    """Return the latest relative timestamp on ``graph``.

    :param graph: Conversation to inspect
    :return: Maximum relative timestamp among timed turns
    """
    return max(
        timestamp
        for timestamp in (_turn_timestamp(turn) for turn in graph.turns)
        if timestamp is not None
    )


def _idle_gap_turn_start(
    turn: ConversationTurnData,
    timestamp: float,
    turns_by_id: dict[str, ConversationTurnData],
    completions: dict[str, float],
) -> float:
    """Earliest start for ``turn`` given parent completions on the trace timeline."""
    if not turn.parents:
        return timestamp
    parent_starts: list[float] = []
    for parent in turn.parents:
        parent_turn = turns_by_id.get(parent.parent_node_id)
        if parent_turn is None:
            continue
        parent_timestamp = _turn_timestamp(parent_turn)
        parent_completion = completions.get(parent.parent_node_id)
        if parent_timestamp is None or parent_completion is None:
            continue
        parent_duration = _turn_duration(parent_turn)
        gap = timestamp - parent_timestamp - parent_duration
        parent_starts.append(parent_completion + max(gap, 0.0))
    return max(parent_starts) if parent_starts else timestamp


def _session_occupancy_bounds(
    graph: ConversationGraphData,
) -> tuple[float, float] | None:
    """Return session start and idle-gap critical-path completion time.

    :param graph: Conversation whose turns carry relative timestamps
    :return: ``(min_start, max_completion)`` or None when no timestamps exist
    """
    turns_by_id = {turn.node_id: turn for turn in graph.turns}
    completions: dict[str, float] = {}
    for _ in range(len(graph.turns) + 1):
        changed = False
        for turn in graph.turns:
            timestamp = _turn_timestamp(turn)
            if timestamp is None:
                continue
            effective_start = _idle_gap_turn_start(
                turn, timestamp, turns_by_id, completions
            )
            completion = effective_start + _turn_duration(turn)
            previous = completions.get(turn.node_id)
            if previous is None or completion > previous:
                completions[turn.node_id] = completion
                changed = True
        if not changed:
            break

    timestamps = [
        timestamp
        for timestamp in (_turn_timestamp(turn) for turn in graph.turns)
        if timestamp is not None
    ]
    if not timestamps:
        return None
    session_start = min(timestamps)
    if not completions:
        return session_start, session_start
    return session_start, max(completions.values())


def _turn_timestamp(turn: ConversationTurnData) -> float | None:
    values = turn.columns.get("relative_timestamp_column")
    if values:
        return float(values[0])
    if turn.settings is not None:
        return turn.settings.relative_timestamp
    return None


def _turn_duration(turn: ConversationTurnData) -> float:
    values = turn.columns.get("request_duration_column")
    if not values or values[0] is None:
        if turn.settings is not None and turn.settings.trace_duration is not None:
            return float(turn.settings.trace_duration)
        return 0.0
    return float(values[0])


def _scale_turn_duration(turn: ConversationTurnData, scale: float) -> None:
    """Multiply a recorded request duration so it stays in timestamp units.

    :param turn: Turn that may carry ``request_duration_column``
    :param scale: Factor already applied to relative timestamps
    """
    values = turn.columns.get("request_duration_column")
    if not values or values[0] is None:
        return
    turn.columns["request_duration_column"] = [float(values[0]) * scale]


def _set_turn_timestamp(turn: ConversationTurnData, timestamp: float) -> None:
    if turn.columns.get("relative_timestamp_column"):
        turn.columns["relative_timestamp_column"] = [timestamp]
    if turn.settings is not None and turn.settings.relative_timestamp is not None:
        turn.settings = turn.settings.model_copy(
            update={"relative_timestamp": timestamp}
        )
