"""
VLLM Python API backend implementation for GuideLLM.

Provides direct Python API integration with VLLM's AsyncLLMEngine, enabling local
inference without HTTP overhead. Supports both GPU-accelerated and CPU-only
inference, true async streaming with token-by-token generation, and flexible
configuration through dict-style parameters.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from more_itertools import roundrobin
from pydantic import ConfigDict, Field, model_validator

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.vllm_python import common
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.extras import vllm
from guidellm.logger import logger
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    StandardBaseModel,
)

__all__ = ["VLLMPythonBackend", "VLLMPythonBackendArgs"]


@BackendArgs.register("vllm_python")
class VLLMPythonBackendArgs(BackendArgs):
    """Pydantic model for VLLM Python backend creation arguments."""

    kind: Literal["vllm_python"] = Field(
        default="vllm_python",
        description="Backend type identifier for VLLM Python backend.",
    )
    model: str = Field(
        description="Huggingface model identifier or filesystem path for VLLM to load",
        examples=["meta-llama/Llama-2-7b-chat-hf"],
    )
    vllm_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Configuration dictionary for vLLM AsyncEngineArgs parameters. Pass "
            "any valid AsyncEngineArgs parameters here (e.g. tensor_parallel_size, "
            "gpu_memory_utilization, max_model_len). The 'model' parameter is required "
            "and can be set here or via the top-level 'model' field; if set in both "
            "places, the top-level 'model' field takes precedence."
        ),
        examples=[
            {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        ],
    )
    request_format: Literal["plain", "default-template"] | str = Field(
        default="default-template",
        description=(
            "Request format for VLLM Python backend. "
            "Valid values are 'plain' (no chat template), 'default-template' "
            "(use tokenizer default), or a path to / inline Jinja2 chat template."
        ),
        examples=[
            "/path/to/chat_template.jinja2",
        ],
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream responses from the backend.",
    )
    image_placeholder: str = Field(
        default="<image>",
        description=(
            "Placeholder string for image items in multimodal prompts. "
            "Used when injecting placeholders for multimodal data."
        ),
    )
    audio_placeholder: str = Field(
        default="<|audio|>",
        description=(
            "Placeholder string for audio items in multimodal prompts. "
            "Used when injecting placeholders for multimodal data."
        ),
    )

    @model_validator(mode="after")
    def validate_vllm_config(self):
        """Set defaults on vllm_config and ensure model is set."""

        if "model" in self.vllm_config:
            logger.warning(
                "The `model` input was passed to the vllm python backend "
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
    stream: bool = Field(description="Whether to stream the response")
    multi_modal_data: dict[str, Any] | None = Field(
        default=None,
        description="vLLM multi_modal_data from image/audio/video columns.",
    )


@Backend.register("vllm_python")
class VLLMPythonBackend(Backend):
    """
    Python API backend for VLLM inference engine.

    Engine parameters not set in vllm_config use vLLM's AsyncEngineArgs defaults.
    Example:
    ::
        backend = VLLMPythonBackend(model="meta-llama/Llama-2-7b-chat-hf")
        # Or: vllm_config={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            process_response(response)
        await backend.process_shutdown()
    """

    @classmethod
    def backend_args(cls) -> type[BackendArgs]:
        """Return the Pydantic model for this backend's creation arguments."""
        return VLLMPythonBackendArgs

    def __init__(
        self,
        arguments: VLLMPythonBackendArgs,
    ):
        """
        Initialize VLLM Python backend with model and configuration.
        """
        super().__init__(arguments)
        self._args = arguments

        # Runtime state
        self._in_process = False
        self._engine: vllm.AsyncLLMEngine | None = None
        self._resolved_chat_template: str | None | object = common.CHAT_TEMPLATE_UNSET

    @property
    def processes_limit(self) -> int | None:
        """
        Limits VLLM Python to a single process, since it starts up an entire
        VLLM engine.
        """
        return 1

    @property
    def info(self) -> dict[str, Any]:
        """
        Get backend configuration details.

        :return: Dictionary containing backend configuration details
        """
        return self._args.model_dump()

    async def process_startup(self):
        """
        Initialize VLLM AsyncLLMEngine instance with configured parameters.

        :raises RuntimeError: If backend is already initialized
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        engine_args = vllm.AsyncEngineArgs(**self._args.vllm_config)
        self._engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self._in_process = True

    async def process_shutdown(self):
        """
        Clean up VLLM AsyncLLMEngine instance and resources.

        :raises RuntimeError: If backend was not properly initialized
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend readiness by attempting a test generation.

        :raises RuntimeError: If backend cannot generate or is not initialized
        """
        engine = self._validate_backend_initialized()

        try:
            await engine.check_health()
        except BaseException as exc:
            raise RuntimeError(
                "Backend validation failed. Health check failed."
            ) from exc

    async def available_models(self) -> list[str]:
        """
        Get available models from this backend.

        :return: List containing the configured model identifier
        """
        # VLLM only supports one model per VLLM instance.
        return [self._args.model]

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: Model name or identifier
        """
        return self._args.model

    def _validate_backend_initialized(self) -> vllm.AsyncLLMEngine:
        """
        Validate that the backend is initialized and return the engine.

        :raises RuntimeError: If backend is not initialized
        :return: The initialized AsyncLLMEngine
        """
        if self._engine is None:
            raise RuntimeError("Backend not started up for process.")
        return self._engine

    def _validate_history(
        self, history: list[tuple[GenerationRequest, GenerationResponse]] | None
    ) -> None:
        """
        Validate that history is not provided (not yet supported).

        :param history: Conversation history
        :raises NotImplementedError: If history is provided
        """
        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")

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
        engine = self._validate_backend_initialized()
        # Lazy-resolve and cache the chat template
        if self._resolved_chat_template is common.CHAT_TEMPLATE_UNSET:
            self._resolved_chat_template = self._resolve_chat_template()
        resolved = cast("str | None", self._resolved_chat_template)
        return common.extract_prompt_chat_tokenizer(
            formatted_messages,
            engine.tokenizer,
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
            stream=self._args.stream,
            multi_modal_data=multi_modal_data,
        )

    def _update_request_timing(
        self, request_info: RequestInfo, iter_time: float
    ) -> None:
        """
        Update request iteration timing information.

        :param request_info: Request tracking info to update
        :param iter_time: Current iteration time
        """
        if request_info.timings.first_request_iteration is None:
            request_info.timings.first_request_iteration = iter_time
        request_info.timings.last_request_iteration = iter_time
        request_info.timings.request_iterations += 1

    def _update_token_timing(
        self, request_info: RequestInfo, iter_time: float, iterations: int = 1
    ) -> None:
        """
        Update token iteration timing information.

        :param request_info: Request tracking info to update
        :param iter_time: Current iteration time
        :param iterations: Number of token iterations (default: 1)
        """
        if request_info.timings.first_token_iteration is None:
            request_info.timings.first_token_iteration = iter_time
        request_info.timings.last_token_iteration = iter_time
        request_info.timings.token_iterations += iterations

    def _text_from_output(self, output: vllm.RequestOutput | None) -> str:
        """
        Extract generated text from VLLM RequestOutput.

        :param output: VLLM request output (may be None)
        :return: output.outputs[0].text if present, else ""
        """
        if output is None or not output.outputs:
            return ""
        return output.outputs[0].text or ""

    def _stream_usage_tokens(
        self,
        output: vllm.RequestOutput,
        request_info: RequestInfo,
    ) -> tuple[int, int]:
        """
        Compute token counts for the streaming path.

        Primary source is output.outputs[0].token_ids (always populated by vLLM).
        Falls back to token_iterations if token_ids is unexpectedly absent.

        :return: (input_tokens, output_tokens)
        """
        input_tokens = len(output.prompt_token_ids or [])
        output_tokens = 0

        if output.outputs:
            out = output.outputs[0]
            if out.token_ids is not None:
                output_tokens = len(out.token_ids)

        if output_tokens == 0 and request_info.timings.token_iterations:
            output_tokens = request_info.timings.token_iterations

        return input_tokens, output_tokens

    def _usage_from_output(
        self,
        output: vllm.RequestOutput | None,
        *,
        request_info: RequestInfo | None = None,
    ) -> dict[str, int] | None:
        """
        Build usage dict (prompt_tokens, completion_tokens, total_tokens).

        When request_info is None (non-stream), uses only output token counts.
        When request_info is set (stream), uses _stream_usage_tokens.
        """
        if output is None:
            return None
        input_tokens = len(output.prompt_token_ids or [])
        output_tokens = 0

        if request_info is None:
            if output.outputs:
                out = output.outputs[0]
                if out.token_ids is not None:
                    output_tokens = len(out.token_ids)
        else:
            input_tokens, output_tokens = self._stream_usage_tokens(
                output, request_info
            )

        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _build_final_response(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        final_output: vllm.RequestOutput | None,
        stream: bool,
        text: str = "",
    ) -> tuple[GenerationResponse, RequestInfo] | None:
        """Build and return the final (response, request_info) to yield, or None."""
        if final_output is None:
            return None
        # Use provided text (e.g. from cancel) or final vLLM output.
        final_text = text or self._text_from_output(final_output)
        # Text and usage from final output only; pass 0 for accumulated tokens.
        usage = self._usage_from_output(
            final_output,
            request_info=request_info if stream else None,
        )
        # We do not send streaming lines during the stream loop, so the handler
        # never receives an id from the stream. Set it from the final vLLM output
        # so the response has a usable id.
        response_id = final_output.request_id if final_output.request_id else None
        # Build response with final text only (no streamed chunks).
        response = VLLMResponseHandler.build_response(
            request, final_text, usage, response_id=response_id
        )
        return response, request_info

    def _create_sampling_params(
        self,
        max_tokens_override: int | None = None,
    ) -> vllm.SamplingParams:
        """Create VLLM SamplingParams."""
        return common.create_sampling_params(vllm, max_tokens_override)

    def _raise_generation_error(self, exc: BaseException) -> None:
        """Re-raise generation failure with context.

        Special-cases audio and engine-death errors.
        """
        error_msg = str(exc)
        # vLLM EngineDeadError: engine core subprocess died
        # (root cause is usually earlier in logs)
        if "EngineCore encountered an issue" in error_msg or (
            exc.__cause__ is not None
            and "EngineCore encountered an issue" in str(exc.__cause__)
        ):
            raise RuntimeError(
                "Generation failed: The vLLM engine core process "
                "has stopped. The failure usually happens during "
                "engine startup or the first request; the root "
                "cause is logged above by the engine core (look "
                "for tracebacks or errors before this message). "
                "Run with a single worker (e.g. --max-workers 1)"
                " and check stderr, or set "
                "GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL=DEBUG. "
                "Common causes: OOM during model load, "
                "unsupported ops (e.g. on CPU/MPS), model load "
                "failure, or model/vLLM compatibility (e.g. "
                "tensor shape errors in rotary_embedding or "
                "attention). For the latter, try the legacy "
                "engine: VLLM_USE_V1=0. Original error: "
                f"{exc}"
            ) from exc
        if "At most 0 audio" in error_msg or "audio(s) may be provided" in error_msg:
            raise RuntimeError(
                f"Generation failed: The model '{self._args.model}' does not "
                f"support audio inputs. Use an audio-capable model "
                f"(e.g. Whisper-based). Original error: {exc}"
            ) from exc
        raise RuntimeError(f"Generation failed: {exc}") from exc

    async def _run_generation(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        stream: bool,
        generate_input: str | dict[str, Any],
        sampling_params: vllm.SamplingParams,
        request_id: str,
        state: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """Run engine.generate loop and yield final (response, request_info)."""
        state["final_output"] = None
        engine = self._validate_backend_initialized()

        prompt_arg = cast("Any", generate_input)
        try:
            async for request_output in engine.generate(  # type: ignore[misc]
                prompt_arg, sampling_params, request_id
            ):
                iter_time = time.time()
                self._update_request_timing(request_info, iter_time)
                state["final_output"] = request_output

                if (
                    request_output.outputs
                    and request_output.outputs[0].finish_reason is not None
                ):
                    break

                if stream:
                    # Streaming gives more granular timing
                    # information that can be saved here.
                    self._update_token_timing(request_info, iter_time, 1)
        except Exception as e:
            logger.debug(
                "vLLM engine.generate() failed: {}: {}",
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"vLLM generation failed: {type(e).__name__}: {e}"
            ) from e

        request_info.timings.request_end = time.time()
        final_output = state["final_output"]
        result = self._build_final_response(
            request,
            request_info,
            final_output,
            stream,
            self._text_from_output(final_output) if final_output else "",
        )
        if result is not None:
            yield result

    async def resolve(  # type: ignore[override, misc]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process generation request and yield a single response.

        Resolves the request (chat template, placeholders, multimodal data),
        then runs async generation via AsyncLLMEngine. For streaming, records
        per-token timings. In both cases the caller receives exactly one
        (response, request_info) pair.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialized
        :yields: Single tuple of (response, updated_request_info)
        """
        self._validate_backend_initialized()
        self._validate_history(history)

        resolved = self._resolve_request(request)
        sampling_params = self._create_sampling_params(
            max_tokens_override=(
                request.output_metrics.text_tokens
                if request.output_metrics.text_tokens
                else None
            ),
        )

        generate_input: str | dict[str, Any]
        if resolved.multi_modal_data:
            generate_input = {
                "prompt": resolved.prompt,
                "multi_modal_data": resolved.multi_modal_data,
            }
        else:
            generate_input = resolved.prompt

        request_id = str(uuid.uuid4())
        request_info.timings.request_start = time.time()

        gen_state: dict[str, Any] = {}
        try:
            async for result in self._run_generation(
                request,
                request_info,
                resolved.stream,
                generate_input,
                sampling_params,
                request_id,
                gen_state,
            ):
                yield result
        except asyncio.CancelledError as err:
            final_output = gen_state.get("final_output")
            cancel_result = self._build_final_response(
                request,
                request_info,
                final_output,
                resolved.stream,
                self._text_from_output(final_output) if final_output else "",
            )
            if cancel_result is not None:
                yield cancel_result
            raise err
        except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
            self._raise_generation_error(exc)
