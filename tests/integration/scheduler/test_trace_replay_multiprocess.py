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

from guidellm.data.deserializers.trace_synthetic import (
    TraceSyntheticDataArgs,
    TraceSyntheticDatasetDeserializer,
)
from guidellm.data.finalizers.generative import (
    GenerativeRequestFinalizer,
    GenerativeRequestFinalizerArgs,
)
from guidellm.data.preprocessors.mappers import (
    GenerativeColumnMapper,
    GenerativeColumnMapperArgs,
)
from guidellm.scheduler import (
    BackendInterface,
    MaxNumberConstraint,
    MaxRequestsConstraintArgs,
    TraceReplayStrategy,
    WorkerProcessGroup,
)
from guidellm.schemas import GenerationRequest, RequestSettings
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
) -> tuple[list[list[GenerationRequest]], list[float]]:
    deserializer = TraceSyntheticDatasetDeserializer()
    dataset = deserializer(
        config=TraceSyntheticDataArgs(path=trace_path),
        processor_factory=_mock_processor,
        random_seed=42,
    )

    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])
    finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    conversations: list[list[GenerationRequest]] = []
    relative_timestamps: list[float] = []
    for index in range(len(dataset)):
        row = {column: dataset[column][index] for column in dataset.column_names}
        mapped = mapper([{"dataset": row}])
        requests = finalizer(mapped)
        assert len(requests) == 1
        request = requests[0]
        request.request_id = f"req_{index}"
        conversations.append([request])
        offset = request.settings.relative_timestamp
        assert offset is not None
        relative_timestamps.append(offset)

    return conversations, relative_timestamps


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

    async def resolve(self, request, request_info, request_history):
        _ = request_history
        await asyncio.sleep(self._resolve_delay)
        yield f"ok_{request.request_id}", request_info


def _request_index(request: GenerationRequest) -> int:
    return int(request.request_id.removeprefix("req_"))


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

    requests, relative_timestamps = _requests_from_trace(trace)
    assert relative_timestamps == pytest.approx(EXPECTED_RELATIVE, abs=1e-9)
    assert len(requests) == NUM_REQUESTS

    strategy = TraceReplayStrategy(time_scale=TIME_SCALE)
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
        assert settings_by_index[index].relative_timestamp == pytest.approx(
            relative_timestamp,
            abs=1e-9,
        )
        expected_target = start_time + TIME_SCALE * relative_timestamp
        assert targeted_start_by_index[index] == pytest.approx(
            expected_target,
            abs=0.05,
        )
