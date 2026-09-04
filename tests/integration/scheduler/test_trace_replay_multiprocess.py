"""
Integration test: trace file through dataset pipeline and TraceReplayStrategy.

Validates multiprocess worker scheduling using real trace replay (not a test-only
strategy): trace_synthetic deserializer, generative mapper/finalizer, and
``TraceReplayStrategy.resolve_dequeued_target_start``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import TraceDatasetDeserializer
from guidellm.data.finalizers.generative import GenerativeRequestFinalizer
from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.scheduler import (
    BackendInterface,
    MaxDurationConstraint,
    MaxNumberConstraint,
    TraceReplayStrategy,
    WorkerProcessGroup,
)
from guidellm.scheduler.schemas import (
    ConversationGraph,
    ConversationNode,
    HistoryContext,
)
from guidellm.scheduler.schemas.conversation_graph import (
    GenerativeConversationGraph,
    GenerativeConversationNode,
)
from guidellm.schemas import GenerationRequest, RequestSettings
from guidellm.schemas.data import (
    GenerativeColumnMapperArgs,
    GenerativeRequestFinalizerArgs,
    MinimalTraceFormatArgs,
)
from guidellm.schemas.scheduler import (
    MaxDurationConstraintArgs,
    MaxRequestsConstraintArgs,
)
from tests.unit.testing_utils import async_timeout

TIME_SCALE = 2.0
RESOLVE_DELAY = 0.03
# Sorted trace: earliest ts=2 -> 0.0, ts=5 -> 3.0, ts=8 -> 6.0 (duplicates below)
EXPECTED_RELATIVE = [0.0, 0.0, 0.0, 0.1, 0.1, 1.5, 2.0, 2.0, 3.5, 7.0]
NUM_REQUESTS = len(EXPECTED_RELATIVE)


def _mock_processor() -> Mock:
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{i}" for i, _ in enumerate(tokens)
    )
    return proc


def _write_trace(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines))
    return path


def _requests_from_trace(
    trace_path: Path,
    *,
    time_scale: float = 1.0,
) -> tuple[list[ConversationGraph[GenerationRequest]], list[float]]:
    deserializer = TraceDatasetDeserializer()
    dataset = deserializer(
        config=MinimalTraceFormatArgs(path=trace_path, time_scale=time_scale),
        processor_factory=_mock_processor,
        random_seed=42,
    )

    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])
    finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    relative_timestamps: list[float] = []
    conv = next(iter(dataset))
    mapped = mapper([{"dataset": conv}])
    graph = finalizer(mapped)
    assert isinstance(graph, GenerativeConversationGraph)
    assert len(graph.nodes) == 10

    graphs: list[ConversationGraph[GenerationRequest]] = []
    for idx, node in enumerate(graph.nodes.values()):
        node.request.request_id = f"req_{idx}"
        offset = node.settings.relative_timestamp
        assert offset is not None
        relative_timestamps.append(offset)
        graphs.append(
            ConversationGraph(
                graph_id=f"graph_req_{idx}",
                nodes={
                    node.node_id: ConversationNode(
                        node_id=node.node_id,
                        agent_id=node.agent_id,
                        request=node.request,
                        settings=node.settings,
                    )
                },
                edges=[],
            )
        )

    return graphs, relative_timestamps


class FastMockBackend(BackendInterface):
    """Backend with short resolve delay to exercise multiprocess dequeue."""

    def __init__(self, resolve_delay: float = RESOLVE_DELAY):
        self._resolve_delay = resolve_delay

    @property
    def processes_limit(self) -> int | None:
        return None

    @property
    def requests_limit(self) -> int | None:
        return None

    def info(self) -> dict[str, Any]:
        return {"type": "fast_mock_trace_replay", "delay": self._resolve_delay}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request, request_info, history=None):
        request_info.timings.request_start = time.time()
        await asyncio.sleep(self._resolve_delay)
        request_info.timings.request_end = time.time()
        rid = (
            request.request_id
            if hasattr(request, "request_id")
            else request["request_id"]
        )
        yield f"ok_{rid}", request_info


def _request_index(request) -> int:
    rid = (
        request.request_id if hasattr(request, "request_id") else request["request_id"]
    )
    return int(rid.removeprefix("req_"))


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.asyncio
@async_timeout(60.0)
async def test_trace_replay_multiprocess_from_trace_file(tmp_path: Path):
    """Trace replay timing under multiprocessing with dataset-sourced settings.

    ### WRITTEN BY AI ###
    """
    # Unsorted rows; deserializer sorts by timestamp (t0=2.0 -> EXPECTED_RELATIVE).
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        [
            '{"timestamp": 9.0, "input_length": 10, "output_length": 5}',
            '{"timestamp": 2.0, "input_length": 10, "output_length": 5}',
            '{"timestamp": 5.5, "input_length": 10, "output_length": 5}',
            '{"timestamp": 2.0, "input_length": 10, "output_length": 5}',
            '{"timestamp": 4.0, "input_length": 10, "output_length": 5}',
            '{"timestamp": 2.1, "input_length": 10, "output_length": 5}',
            '{"timestamp": 2.0, "input_length": 10, "output_length": 5}',
            '{"timestamp": 3.5, "input_length": 10, "output_length": 5}',
            '{"timestamp": 2.1, "input_length": 10, "output_length": 5}',
            '{"timestamp": 4.0, "input_length": 10, "output_length": 5}',
        ],
    )

    requests, relative_timestamps = _requests_from_trace(trace, time_scale=TIME_SCALE)
    assert relative_timestamps == pytest.approx(
        [TIME_SCALE * timestamp for timestamp in EXPECTED_RELATIVE],
        abs=1e-9,
    )
    assert len(requests) == NUM_REQUESTS

    strategy = TraceReplayStrategy()
    group = WorkerProcessGroup(
        backend=FastMockBackend(resolve_delay=RESOLVE_DELAY),
        requests=requests,
        strategy=strategy,
        max_number=MaxNumberConstraint(
            args=MaxRequestsConstraintArgs(count=NUM_REQUESTS)
        ),
    )

    settings_by_index: dict[int, RequestSettings] = {}
    targeted_start_by_index: dict[int, float] = {}
    worker_nodes: set[int] = set()
    completed = 0

    try:
        await group.create_processes()
        assert group.processes is not None
        assert len(group.processes) >= 2

        start_time = time.time() + 0.05
        await group.start(start_time)

        async for (
            response,
            request,
            request_info,
            _state,
        ) in group.request_updates():
            index = _request_index(request)

            if request_info.settings.relative_timestamp is not None:
                settings_by_index[index] = request_info.settings

            if request_info.timings.targeted_start is not None:
                targeted_start_by_index[index] = request_info.timings.targeted_start

            if request_info.status == "completed":
                assert response == f"ok_req_{index}"
                worker_nodes.add(request_info.scheduler_node_id)
                completed += 1
                if completed == NUM_REQUESTS:
                    break
    finally:
        exceptions = await group.shutdown()
        assert exceptions == []

    assert len(settings_by_index) == NUM_REQUESTS
    assert len(targeted_start_by_index) == NUM_REQUESTS
    assert len(worker_nodes) >= 2

    for index, relative_timestamp in enumerate(EXPECTED_RELATIVE):
        scaled_timestamp = TIME_SCALE * relative_timestamp
        assert settings_by_index[index].relative_timestamp == pytest.approx(
            scaled_timestamp,
            abs=1e-9,
        )
        expected_target = start_time + scaled_timestamp
        assert targeted_start_by_index[index] == pytest.approx(
            expected_target,
            abs=0.05,
        )


def _linear_replay_graph(
    timestamps: list[float],
    *,
    graph_id: str = "linear_replay",
    request_prefix: str = "req",
) -> GenerativeConversationGraph:
    nodes: dict[str, GenerativeConversationNode] = {}
    parents_by_node: dict[str, list[tuple[str, HistoryContext]]] = {}
    prev: str | None = None
    for index, relative_timestamp in enumerate(timestamps):
        node_id = f"n{index}"
        nodes[node_id] = GenerativeConversationNode(
            node_id=node_id,
            agent_id="default",
            request=GenerationRequest(request_id=f"{request_prefix}_{index}"),
            settings=RequestSettings(relative_timestamp=relative_timestamp),
        )
        parents_by_node[node_id] = [(prev, "full")] if prev is not None else []
        prev = node_id
    return GenerativeConversationGraph.from_nodes_with_parents(
        nodes=nodes,
        parents_by_node=parents_by_node,
        graph_id=graph_id,
    )


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.asyncio
@async_timeout(60.0)
async def test_max_duration_cancels_long_replay_sleep():
    """max_duration stops workers sleeping on a future relative_timestamp.

    Use two independent graphs so the delayed request is sleeping from t=0.
    A linear parent-child graph only starts that sleep after the first request
    finishes, which on a loaded runner can let both complete before cancel.

    ## WRITTEN BY AI ##
    """
    immediate = _linear_replay_graph([0.0], graph_id="immediate", request_prefix="imm")
    delayed = _linear_replay_graph([5.0], graph_id="delayed", request_prefix="del")
    strategy = TraceReplayStrategy(time_scale=1.0)
    group = WorkerProcessGroup(
        backend=FastMockBackend(resolve_delay=RESOLVE_DELAY),
        requests=[immediate, delayed],
        strategy=strategy,
        max_duration=MaxDurationConstraint(args=MaxDurationConstraintArgs(seconds=0.4)),
    )
    statuses: set[str] = set()
    try:
        await group.create_processes()
        await group.start(time.time() + 0.05)
        run_started = time.time()
        async for _, _, request_info, _state in group.request_updates():
            statuses.add(request_info.status)
            if "cancelled" in statuses:
                break
        elapsed = time.time() - run_started
    finally:
        exceptions = await group.shutdown()
        assert exceptions == []

    assert "cancelled" in statuses
    assert elapsed < 2.5
