"""
vLLM Offline Backend for static/micro-batch inference.

Uses vLLM's LLM class for synchronous batch processing. Collects requests
into batches and processes them with LLM.generate() for maximum throughput.
Designed for offline benchmarking scenarios where batching efficiency is
more important than per-request latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast

from more_itertools import roundrobin
from pydantic import ConfigDict, Field, PositiveInt, model_validator

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.vllm_python import common
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.logger import logger

if TYPE_CHECKING:
    from guidellm.extras import vllm
else:
    from guidellm.extras import vllm
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    StandardBaseModel,
)

__all__ = ["VLLMOfflineBackend", "VLLMOfflineBackendArgs"]


@BackendArgs.register("vllm_offline")
class VLLMOfflineBackendArgs(BackendArgs):
    """Pydantic model for VLLM Offline backend creation arguments."""

    kind: Literal["vllm_offline"] = Field(
        default="vllm_offline",
        description="Backend type identifier for VLLM Offline backend.",
    )
    model: str = Field(
        description="Model identifier or path for VLLM to load",
    )
    vllm_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Configuration dictionary for vLLM EngineArgs parameters. Pass "
            "any valid EngineArgs parameters here (e.g. tensor_parallel_size, "
            "gpu_memory_utilization, max_model_len). The 'model' parameter is required "
            "and can be set here or via the top-level 'model' field; if set in both "
            "places, the top-level 'model' field takes precedence."
        ),
    )
    request_format: Literal["plain", "default-template"] | str = Field(
        default="default-template",
        description=(
            "Request format for VLLM Offline backend. "
            "Valid values: 'plain' (no chat template), "
            "'default-template' (use tokenizer default), or a path to "
            "/ inline Jinja2 chat template."
        ),
    )
    image_placeholder: str = Field(
        default="<image>",
        description="Placeholder for image items in multimodal prompts.",
    )
    audio_placeholder: str = Field(
        default="<|audio|>",
        description="Placeholder for audio items in multimodal prompts.",
    )
    batch_size: PositiveInt = Field(
        default=32,
        description=(
            "Number of requests to collect before processing as a batch. "
            "Larger batches improve throughput but increase latency."
        ),
    )

    @model_validator(mode="after")
    def validate_vllm_config(self):
        """Set defaults on vllm_config and ensure model is set."""
        if "model" in self.vllm_config:
            logger.warning(
                "The `model` input was passed to the vllm offline backend "
                "with the `vllm_config` input. Ignoring and overwriting "
                "with the value from the `model` input."
            )
        self.vllm_config["model"] = self.model
        return self


class _ResolvedRequest(StandardBaseModel):
    """Fully resolved request: prompt already formatted, ready for engine.generate."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(
        description="Fully resolved prompt string (templated, with placeholders)"
    )
    multi_modal_data: dict[str, Any] | None = Field(
        default=None,
        description="vLLM multi_modal_data from image/audio/video columns.",
    )


class _BatchedRequest:
    """Internal tracking for a request waiting in batch."""

    def __init__(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        resolved_prompt: str,
        multi_modal_data: dict[str, Any] | None,
        max_tokens: int | None,
    ):
        self.request = request
        self.request_info = request_info
        self.resolved_prompt = resolved_prompt
        self.multi_modal_data = multi_modal_data
        self.max_tokens = max_tokens
        self.request_id = str(uuid.uuid4())
        self.result: vllm.RequestOutput | None = None
        self.ready = asyncio.Event()


@Backend.register("vllm_offline")
class VLLMOfflineBackend(Backend):
    """
    Offline backend for vLLM using LLM class for batch processing.

    Collects requests into micro-batches and processes them together using
    vLLM's LLM.generate() for optimal throughput. Designed for offline
    benchmarking where batch efficiency is prioritized over streaming latency.

    Example:
    ::
        args = VLLMOfflineBackendArgs(
            model="meta-llama/Llama-2-7b-hf",
            batch_size=64,
            vllm_config={"tensor_parallel_size": 2}
        )
        backend = VLLMOfflineBackend(args)
        await backend.process_startup()
        async for response, info in backend.resolve(request, request_info):
            process_response(response)
        await backend.process_shutdown()
    """

    @classmethod
    def backend_args(cls) -> type[BackendArgs]:
        """Return the Pydantic model for this backend's creation arguments."""
        return VLLMOfflineBackendArgs

    def __init__(self, arguments: VLLMOfflineBackendArgs):
        """Initialize vLLM Offline backend with model and configuration."""
        super().__init__(arguments)
        self._args = arguments

        # Runtime state
        self._in_process = False
        self._shutting_down = False
        self._llm: Any = None  # vllm.LLM | None
        self._batch_lock = asyncio.Lock()
        self._pending_batch: list[_BatchedRequest] = []
        self._processing_task: asyncio.Task | None = None
        self._resolved_chat_template: str | None | object = common.CHAT_TEMPLATE_UNSET

    @property
    def processes_limit(self) -> int | None:
        """Limit to single process for batch coordination."""
        return 1

    @property
    def info(self) -> dict[str, Any]:
        """Get backend configuration details."""
        return self._args.model_dump()

    async def process_startup(self):
        """Initialize vLLM LLM instance with configured parameters."""
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        # Initialize LLM in thread pool to avoid blocking
        def _init_llm():
            engine_args = vllm.EngineArgs(**self._args.vllm_config)  # type: ignore[attr-defined]
            return vllm.LLM.from_engine_args(engine_args)  # type: ignore[attr-defined]

        self._llm = await asyncio.to_thread(_init_llm)
        self._in_process = True

    async def process_shutdown(self):
        """Clean up vLLM LLM instance and resources."""
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        # Set shutdown flag to reject new requests
        self._shutting_down = True

        # Cancel any pending processing
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        # Process any remaining requests in batch
        async with self._batch_lock:
            if self._pending_batch:
                await self._process_batch()

        if self._llm is not None:
            # LLM cleanup happens automatically via GC
            self._llm = None

        self._in_process = False
        self._shutting_down = False

    async def validate(self):
        """Validate backend readiness."""
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")
        # LLM is ready if it was constructed successfully

    async def available_models(self) -> list[str]:
        """Get available models from this backend."""
        return [self._args.model]

    async def default_model(self) -> str:
        """Get the default model for this backend."""
        return self._args.model

    def _validate_backend_initialized(self) -> Any:  # vllm.LLM
        """
        Validate that the backend is initialized and return the LLM.

        :raises RuntimeError: If backend is not initialized
        :return: The initialized LLM
        """
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")
        return self._llm

    def _build_multi_modal_data_from_columns(
        self, columns: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build vLLM multi_modal_data dict from image_column, audio_column."""
        return common.build_multi_modal_data_from_columns(columns)

    def _extract_text_from_content(
        self, content: str | list[dict[str, Any]] | Any
    ) -> str:
        """Extract text content from message content field."""
        return common.extract_text_from_content(content)

    def _build_placeholder_prefix(self, multi_modal_data: dict[str, Any]) -> str:
        """Build the placeholder prefix string for all modalities."""
        return common.build_placeholder_prefix(
            multi_modal_data,
            self._args.image_placeholder,
            self._args.audio_placeholder,
        )

    @staticmethod
    def _format_column_blocks(
        column_data: list[Any], column_type: str
    ) -> list[dict[str, Any]]:
        """Format data column items into vLLM-compatible content blocks."""
        return common.format_column_blocks(column_data, column_type)

    def _inject_placeholders_into_messages(
        self,
        formatted_messages: list[dict[str, Any]],
        multi_modal_data: dict[str, Any],
    ) -> None:
        """Inject multimodal placeholder tokens into the last user message."""
        common.inject_placeholders_into_messages(
            formatted_messages,
            multi_modal_data,
            self._args.image_placeholder,
            self._args.audio_placeholder,
        )

    def _extract_prompt_chat_plain(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Concatenate message content into a single raw prompt string."""
        return common.extract_prompt_chat_plain(formatted_messages)

    def _resolve_chat_template(self) -> str | None:
        """Resolve and validate request_format to a template string or None."""
        return common.resolve_chat_template(self._args.request_format)

    def _extract_prompt_chat_tokenizer(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Apply tokenizer chat template to formatted messages."""
        llm = self._validate_backend_initialized()
        # Lazy-resolve and cache the chat template
        if self._resolved_chat_template is common.CHAT_TEMPLATE_UNSET:
            self._resolved_chat_template = self._resolve_chat_template()
        resolved = cast("str | None", self._resolved_chat_template)
        return common.extract_prompt_chat_tokenizer(
            formatted_messages,
            llm.llm_engine.tokenizer.tokenizer,
            self._args.request_format,
            resolved,
        )

    def _resolve_request(self, request: GenerationRequest) -> _ResolvedRequest:
        """
        Build a fully resolved request from column-based GenerationRequest.

        Mirrors the HTTP backend's ``ChatCompletionsRequestHandler.format``:
        prefix items are space-joined into one system message and all data
        columns (text, image, audio) are formatted as typed content blocks
        then interleaved via ``roundrobin`` into a single user message.

        When a chat template is active and multimodal data is present, the
        list-of-blocks content is passed directly to the tokenizer so the
        template emits model-specific placeholder tokens.  For plain format
        or text-only requests the content is flattened to strings.

        :param request: Column-based generation request
        :return: Resolved request with formatted prompt and multimodal data
        :raises ValueError: If request has no text or multimodal columns
        """
        columns = request.columns

        messages: list[dict[str, Any]] = []

        prefix = " ".join(str(p) for p in columns.get("prefix_column", []) if p)
        if prefix:
            messages.append({"role": "system", "content": prefix})

        text_blocks = self._format_column_blocks(
            columns.get("text_column", []), "text_column"
        )

        multi_modal_data = self._build_multi_modal_data_from_columns(columns)

        # We use explicit content blocks (e.g. {"type": "image"}) when applying a
        # chat template so that the template itself can generate the correct,
        # model-specific tokens. Otherwise, we flatten to strings and fall back
        # to placeholder-string injection.
        use_content_blocks = (
            multi_modal_data
            and (text_blocks or prefix)
            and self._args.request_format != "plain"
        )

        if use_content_blocks:
            # Interleave text and media blocks into a single content list,
            # matching the HTTP backend's roundrobin approach.
            media_lists = [
                self._format_column_blocks(columns.get(col, []), col)
                for col in ("image_column", "audio_column")
            ]
            user_content: list[dict[str, Any]] = list(
                roundrobin(text_blocks, *media_lists)
            )
        else:
            # Text-only or plain mode: media is handled later via placeholder
            # injection, so only text blocks go into the user message here.
            user_content = list(text_blocks)

        if user_content:
            messages.append({"role": "user", "content": user_content})

        if messages:
            if use_content_blocks:
                prompt = self._extract_prompt_chat_tokenizer(messages)
            else:
                formatted_messages = [
                    {
                        "role": msg["role"],
                        "content": self._extract_text_from_content(
                            msg.get("content", "")
                        ),
                    }
                    for msg in messages
                ]

                if multi_modal_data:
                    # Placeholders must be injected into the message text
                    # *before* the chat template is applied so they end up
                    # inside the correct message turn.
                    self._inject_placeholders_into_messages(
                        formatted_messages, multi_modal_data
                    )

                if self._args.request_format == "plain":
                    prompt = self._extract_prompt_chat_plain(formatted_messages)
                else:
                    prompt = self._extract_prompt_chat_tokenizer(formatted_messages)
        elif multi_modal_data:
            # Multimodal-only (e.g. audio transcription with no text/prefix):
            # no messages to inject into, so use a raw placeholder prompt.
            prompt = self._build_placeholder_prefix(multi_modal_data)
        else:
            raise ValueError("Request must include text_column or multimodal columns.")

        return _ResolvedRequest(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )

    def _create_sampling_params(
        self,
        max_tokens_override: int | None = None,
    ) -> vllm.SamplingParams:
        """Create VLLM SamplingParams."""
        return common.create_sampling_params(vllm, max_tokens_override)

    async def _process_batch(self):
        """Process all pending requests as a batch using LLM.generate()."""
        if not self._pending_batch:
            return

        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")

        batch = self._pending_batch
        self._pending_batch = []

        logger.debug(f"Processing batch of {len(batch)} requests")

        # Build inputs for LLM.generate()
        prompts = []
        sampling_params_list = []

        for req in batch:
            prompt_input: dict[str, Any] | str
            if req.multi_modal_data:
                prompt_input = {
                    "prompt": req.resolved_prompt,
                    "multi_modal_data": req.multi_modal_data,
                }
            else:
                prompt_input = req.resolved_prompt

            prompts.append(prompt_input)
            sampling_params = self._create_sampling_params(req.max_tokens)
            sampling_params_list.append(sampling_params)

        # Process batch in thread pool
        def _generate_batch():
            return self._llm.generate(  # type: ignore[union-attr]
                prompts,
                sampling_params_list,
                use_tqdm=False,
            )

        try:
            outputs: list[vllm.RequestOutput] = await asyncio.to_thread(_generate_batch)

            # Match outputs to requests and mark ready
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Batch size mismatch: expected {len(batch)} outputs, "
                    f"got {len(outputs)}"
                )

            for req, output in zip(batch, outputs, strict=True):
                req.result = output
                req.ready.set()
        except Exception as exc:  # noqa: BLE001
            # Catch all exceptions to ensure requests don't hang forever.
            # This is safe here because we're marking requests as failed.
            logger.error(f"Batch processing failed: {exc}")
            # Mark all requests as failed but don't re-raise
            # (individual requests will see None result)
            for req in batch:
                req.ready.set()

    async def _maybe_process_batch(self):
        """Check if batch is full and process if so."""
        async with self._batch_lock:
            if len(self._pending_batch) >= self._args.batch_size:
                await self._process_batch()

    async def resolve(  # type: ignore[override]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process generation request by batching with others.

        Collects requests into micro-batches and processes them together
        using LLM.generate(). The caller waits for the batch to complete
        before receiving the response.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :yields: Single tuple of (response, updated_request_info)
        """
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")

        if self._shutting_down:
            raise RuntimeError("Backend is shutting down, cannot accept new requests.")

        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")

        # Resolve the request
        request_info.timings.request_start = time.time()
        resolved = self._resolve_request(request)

        # Create batched request tracker
        max_tokens = (
            request.output_metrics.text_tokens
            if request.output_metrics.text_tokens
            else None
        )

        batched_req = _BatchedRequest(
            request=request,
            request_info=request_info,
            resolved_prompt=resolved.prompt,
            multi_modal_data=resolved.multi_modal_data,
            max_tokens=max_tokens,
        )

        # Add to pending batch
        async with self._batch_lock:
            self._pending_batch.append(batched_req)

        # Trigger batch processing if full
        await self._maybe_process_batch()

        # Wait for result
        await batched_req.ready.wait()

        # Build response
        request_info.timings.request_end = time.time()

        if batched_req.result is not None:
            output = batched_req.result
            text = output.outputs[0].text if output.outputs else ""
            usage = {
                "prompt_tokens": len(output.prompt_token_ids or []),
                "completion_tokens": len(output.outputs[0].token_ids or [])
                if output.outputs
                else 0,
                "total_tokens": len(output.prompt_token_ids or [])
                + (len(output.outputs[0].token_ids or []) if output.outputs else 0),
            }

            response = VLLMResponseHandler.build_response(
                request, text, usage, response_id=output.request_id
            )
            yield response, request_info
        else:
            # Request failed during batch processing
            request_info.error = "Batch processing failed"
            yield None, request_info  # type: ignore[misc]
