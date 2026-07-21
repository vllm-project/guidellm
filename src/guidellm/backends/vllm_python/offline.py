"""
VLLM Offline (batch) backend implementation for GuideLLM.

Provides batch-oriented inference using vLLM's synchronous LLM engine.
Requests are queued and processed in configurable batches, removing the
overhead of per-request engine interaction while still integrating with
the standard GuideLLM scheduler lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, PositiveInt

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.vllm_python.common import reset_cpu_affinity
from guidellm.backends.vllm_python.vllm import (
    _CHAT_TEMPLATE_UNSET,
    VLLMPythonBackend,
    VLLMPythonBackendArgs,
)
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.extras import vllm
from guidellm.logger import logger
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    StandardBaseModel,
)

__all__ = ["VLLMOfflineBackend", "VLLMOfflineBackendArgs"]


@BackendArgs.register("vllm_offline")
class VLLMOfflineBackendArgs(VLLMPythonBackendArgs):
    """Pydantic model for VLLM Offline backend creation arguments.

    Extends :class:`VLLMPythonBackendArgs` with batch-specific options
    and removes the ``stream`` field (offline generation is always
    non-streaming).
    """

    kind: Literal["vllm_offline"] = Field(  # type: ignore[assignment]
        default="vllm_offline",
        description="Backend type identifier for VLLM Offline backend.",
    )
    batch_size: PositiveInt = Field(
        default=32,
        description=(
            "Maximum number of requests to accumulate before "
            "dispatching a single vLLM generate() call.  Full "
            "batches flush immediately; partial batches wait up "
            "to ``batch_timeout`` seconds."
        ),
    )
    batch_timeout: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Seconds to wait for more requests before flushing a "
            "partial batch.  Full batches bypass this delay."
        ),
    )

    # Hide the inherited ``stream`` field -- offline is never streaming.
    stream: Literal[False] = Field(  # type: ignore[assignment]
        default=False,
        exclude=True,
        description="Offline backend does not support streaming.",
    )


class _OfflineResolvedRequest(StandardBaseModel):
    """Fully resolved request for the offline backend.

    Same as the online ``_ResolvedRequest`` but without a ``stream``
    field, since offline generation never streams.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(
        description="Fully resolved prompt string.",
    )
    multi_modal_data: dict[str, Any] | None = Field(
        default=None,
        description=("vLLM multi_modal_data from image/audio/video columns."),
    )


@dataclass
class _BatchedRequest:
    """Tracks a single request waiting for batch processing."""

    resolved_prompt: str
    multi_modal_data: dict[str, Any] | None
    max_tokens: int | None
    result: Any = None  # vllm.RequestOutput once available
    ready: asyncio.Event = field(default_factory=asyncio.Event)


@Backend.register("vllm_offline")
class VLLMOfflineBackend(VLLMPythonBackend):
    """
    Batch-oriented Python API backend for VLLM inference engine.

    Queues incoming requests and dispatches them in batches via
    ``vllm.LLM.generate()``, which runs the synchronous offline
    engine.  This avoids per-request scheduling overhead and is
    ideal for throughput benchmarking.

    Example:
    ::
        backend = VLLMOfflineBackend(
            VLLMOfflineBackendArgs(
                model="meta-llama/Llama-2-7b-chat-hf",
                batch_size=16,
            )
        )
        await backend.process_startup()
        async for response, request_info in backend.resolve(
            request, info
        ):
            process_response(response)
        await backend.process_shutdown()
    """

    _args: VLLMOfflineBackendArgs

    @classmethod
    def backend_args(cls) -> type[BackendArgs]:
        """Return the Pydantic model for this backend's creation
        arguments.
        """
        return VLLMOfflineBackendArgs

    def __init__(
        self,
        arguments: VLLMOfflineBackendArgs,
    ):
        """
        Initialize VLLM Offline backend.

        Sets up batch processing state in addition to the base
        backend initialisation.
        """
        super().__init__(arguments)

        # Batch processing state
        self._batch_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()
        self._pending_batch: list[_BatchedRequest] = []
        self._processing_task: asyncio.Task[None] | None = None
        self._shutting_down = False

        # The synchronous vLLM LLM engine (set during startup)
        self._llm: Any = None  # vllm.LLM
        self._engine_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def process_startup(self):
        """
        Mark the backend as active and reset process-local state.

        Engine construction is deferred to the first
        ``_ensure_engine()`` call so that the heavyweight vLLM
        multiprocess executor is only created inside the worker
        process that will actually run inference.  This avoids a
        double-startup that wastes resources reloading model
        weights and can cause CPU-affinity degradation.

        Asyncio primitives are recreated here because the benchmark
        framework forks worker processes after a parent
        validate/shutdown cycle; locks created on the parent's
        event loop are unusable in the child.

        :raises RuntimeError: If backend is already initialised
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        self._in_process = True
        self._shutting_down = False

        # Recreate asyncio primitives for the current event loop.
        self._batch_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()
        self._engine_lock = asyncio.Lock()
        self._pending_batch = []
        self._processing_task = None

    async def _ensure_engine(self) -> Any:
        """Create the vLLM ``LLM`` engine on first use."""
        if self._llm is not None:
            return self._llm

        async with self._engine_lock:
            if self._llm is not None:
                return self._llm

            loop = asyncio.get_running_loop()
            config = dict(self._args.vllm_config)
            engine_args = vllm.EngineArgs(  # type: ignore[attr-defined]
                **config,
            )

            def _create_engine() -> Any:
                reset_cpu_affinity()
                return vllm.LLM.from_engine_args(  # type: ignore[attr-defined]
                    engine_args,
                )

            self._llm = await loop.run_in_executor(
                None,
                _create_engine,
            )
            return self._llm

    async def process_shutdown(self):
        """
        Drain pending batch and tear down the vLLM LLM engine.

        :raises RuntimeError: If backend was not properly initialised
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        # Set flag under lock so no new requests can enqueue
        batch: list[_BatchedRequest] = []
        async with self._batch_lock:
            self._shutting_down = True
            if self._pending_batch:
                batch = self._take_pending_batch()
        if batch:
            await self._run_generate(batch)

        # Wait for any deferred flush to finish naturally.
        # _shutting_down prevents new enqueues so the loop will
        # see an empty _pending_batch and exit.
        if self._processing_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
            self._processing_task = None

        # Serialize with any in-flight _run_generate (e.g. from
        # _maybe_process_batch in a concurrent resolve()) before
        # tearing down the engine.
        async with self._generate_lock:
            if self._llm is not None:
                if hasattr(self._llm, "shutdown"):
                    self._llm.shutdown()
                del self._llm
                self._llm = None
                gc.collect()

        self._in_process = False

    async def validate(self):
        """
        Validate backend readiness.

        Only checks that ``process_startup()`` was called.  The
        engine itself is created lazily on the first request.

        :raises RuntimeError: If backend is not initialised
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

    def _validate_backend_initialized(self) -> Any:  # type: ignore[override]
        """
        Validate that the backend is initialised and return the LLM.

        :raises RuntimeError: If backend is not initialised
        :return: The initialised ``vllm.LLM`` instance
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")
        return self._llm

    # ------------------------------------------------------------------
    # Chat template / tokenizer (overrides for offline tokenizer path)
    # ------------------------------------------------------------------

    def _extract_prompt_chat_tokenizer(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Apply tokenizer chat template to formatted messages.

        Accesses the tokenizer through ``llm.get_tokenizer()``
        instead of ``engine.tokenizer`` used by the async parent.
        """
        llm = self._validate_backend_initialized()
        tokenizer = llm.get_tokenizer()
        if tokenizer is None:
            raise RuntimeError("Backend engine has no tokenizer.")

        if self._args.request_format in (
            "plain",
            "default-template",
        ):
            resolved: str | None = None
        else:
            if self._resolved_chat_template is _CHAT_TEMPLATE_UNSET:
                self._resolved_chat_template = self._resolve_chat_template()
            resolved = cast(
                "str | None",
                self._resolved_chat_template,
            )

        if resolved is not None:
            tokenizer.chat_template = resolved

        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt, str):
            return prompt
        raise RuntimeError("Backend received unexpected type from tokenizer.")

    # ------------------------------------------------------------------
    # Request resolution (no ``stream`` field)
    # ------------------------------------------------------------------

    def _resolve_request(  # type: ignore[override]
        self, request: GenerationRequest
    ) -> _OfflineResolvedRequest:
        """
        Build a fully resolved request for offline generation.

        Delegates to the parent's resolution logic but returns an
        ``_OfflineResolvedRequest`` (without a ``stream`` field).

        :param request: Column-based generation request
        :return: Resolved request with formatted prompt and
            multimodal data
        """
        parent_resolved = super()._resolve_request(request)
        return _OfflineResolvedRequest(
            prompt=parent_resolved.prompt,
            multi_modal_data=parent_resolved.multi_modal_data,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _take_pending_batch(self) -> list[_BatchedRequest]:
        """Snapshot and clear ``_pending_batch``.

        Must be called while holding ``_batch_lock``.
        """
        batch = list(self._pending_batch)
        self._pending_batch.clear()
        return batch

    async def _run_generate(self, batch: list[_BatchedRequest]) -> None:
        """Run ``LLM.generate()`` for *batch* and distribute results.

        Serialized by ``_generate_lock`` so only one generate() call
        runs at a time (vLLM's sync LLM is not safe for overlapping
        generates).  Does **not** hold ``_batch_lock`` so new requests
        can enqueue while generation is in progress.
        """
        if not batch:
            return

        # Build per-request generate inputs
        prompts: list[str | dict[str, Any]] = []
        sampling_params_list: list[Any] = []
        for batched_req in batch:
            if batched_req.multi_modal_data:
                prompts.append(
                    {
                        "prompt": batched_req.resolved_prompt,
                        "multi_modal_data": (batched_req.multi_modal_data),
                    }
                )
            else:
                prompts.append(batched_req.resolved_prompt)

            sampling_params_list.append(
                self._create_sampling_params(
                    batched_req.max_tokens,
                )
            )

        loop = asyncio.get_running_loop()

        async with self._generate_lock:
            try:
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self._llm.generate(  # type: ignore[union-attr]
                        prompts,
                        sampling_params=sampling_params_list,
                        use_tqdm=False,
                    ),
                )
            except (
                RuntimeError,
                ValueError,
                TypeError,
                OSError,
                KeyError,
            ) as exc:
                logger.error(
                    "vLLM offline generate() failed: {}: {}",
                    type(exc).__name__,
                    exc,
                )
                for batched_req in batch:
                    batched_req.result = exc
                    batched_req.ready.set()
                return

        # Distribute results back to callers
        for batched_req, output in zip(batch, outputs, strict=True):
            batched_req.result = output
            batched_req.ready.set()

    async def _maybe_process_batch(self) -> None:
        """Trigger batch processing if the batch is full."""
        batch: list[_BatchedRequest] = []
        async with self._batch_lock:
            if len(self._pending_batch) >= self._args.batch_size:
                batch = self._take_pending_batch()
        if batch:
            await self._run_generate(batch)

    async def resolve(  # type: ignore[override, misc]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: (list[tuple[GenerationRequest, GenerationResponse]] | None) = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Queue a request for batch processing and yield the response.

        Resolves the request (chat template, placeholders, multimodal
        data), adds it to the pending batch, and waits until the
        batch has been processed.  The caller receives exactly one
        ``(response, request_info)`` pair.

        :param request: Generation request with content and params
        :param request_info: Request tracking info updated with
            timing metadata
        :param history: Conversation history (not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialised or
            generation fails
        :yields: Single tuple of (response, updated_request_info)
        """
        self._validate_backend_initialized()
        await self._ensure_engine()
        self._validate_history(history)

        resolved = self._resolve_request(request)

        max_tokens = (
            request.output_metrics.text_tokens
            if request.output_metrics.text_tokens
            else None
        )

        batched_req = _BatchedRequest(
            resolved_prompt=resolved.prompt,
            multi_modal_data=resolved.multi_modal_data,
            max_tokens=max_tokens,
        )

        request_info.timings.request_start = time.time()

        # Enqueue atomically with shutdown check
        async with self._batch_lock:
            if self._shutting_down:
                raise RuntimeError("Backend is shutting down.")
            self._pending_batch.append(batched_req)

        # If the batch is full, process immediately
        await self._maybe_process_batch()

        # If not full yet, schedule a deferred flush so the last
        # partial batch does not sit forever.
        if not batched_req.ready.is_set():
            await self._schedule_deferred_flush()

        # Wait for this request's result
        await batched_req.ready.wait()

        result = batched_req.result

        # Propagate generation errors
        if isinstance(result, BaseException):
            self._raise_generation_error(result)

        request_output = result  # vllm.RequestOutput

        # Wire vLLM request metrics into timing info
        self._wire_vllm_metrics(request_info, request_output)

        request_info.timings.request_end = time.time()

        text = self._text_from_output(request_output)
        usage = self._usage_from_output(request_output)
        response_id = request_output.request_id if request_output.request_id else None

        response = VLLMResponseHandler.build_response(
            request, text, usage, response_id=response_id
        )
        yield response, request_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _schedule_deferred_flush(self) -> None:
        """Schedule a task that flushes the pending batch after
        ``batch_timeout`` seconds, giving concurrent requests a
        window to accumulate before dispatch.
        """

        async def _deferred_flush() -> None:
            while True:
                await asyncio.sleep(self._args.batch_timeout)
                async with self._batch_lock:
                    if not self._pending_batch:
                        return
                    batch = self._take_pending_batch()
                await self._run_generate(batch)

        # Only schedule one deferred flush at a time
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.ensure_future(_deferred_flush())

    @staticmethod
    def _wire_vllm_metrics(
        request_info: RequestInfo,
        request_output: Any,
    ) -> None:
        """Populate iteration counts and timing from vLLM metrics.

        Extracts token counts and, when available, maps vLLM's
        monotonic-clock ``RequestStateStats`` timestamps to
        wall-clock values anchored on the wall-clock
        ``arrival_time`` that vLLM also provides.
        """
        metrics = request_output.metrics if hasattr(request_output, "metrics") else None
        num_gen = (
            metrics.num_generation_tokens
            if metrics and hasattr(metrics, "num_generation_tokens")
            else 0
        )

        if num_gen > 0:
            request_info.timings.token_iterations = num_gen
        elif request_output.outputs and request_output.outputs[0].token_ids is not None:
            num_gen = len(request_output.outputs[0].token_ids)
            request_info.timings.token_iterations = num_gen

        if num_gen > 0:
            request_info.timings.request_iterations = 1

        if metrics is None:
            return

        arrival = metrics.arrival_time if hasattr(metrics, "arrival_time") else 0.0
        scheduled = metrics.scheduled_ts if hasattr(metrics, "scheduled_ts") else 0.0
        queued = metrics.queued_ts if hasattr(metrics, "queued_ts") else 0.0
        mono_base = scheduled or queued
        first_tok = (
            metrics.first_token_ts if hasattr(metrics, "first_token_ts") else 0.0
        )
        last_tok = metrics.last_token_ts if hasattr(metrics, "last_token_ts") else 0.0

        if not (arrival and mono_base and first_tok):
            return

        request_info.timings.first_request_iteration = arrival
        request_info.timings.first_token_iteration = arrival + (first_tok - mono_base)
        if last_tok:
            request_info.timings.last_token_iteration = arrival + (last_tok - mono_base)
            request_info.timings.last_request_iteration = arrival + (
                last_tok - mono_base
            )
