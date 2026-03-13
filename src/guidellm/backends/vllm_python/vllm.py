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
from pathlib import Path
from typing import Any, cast

import jinja2
from more_itertools import roundrobin
from pydantic import ConfigDict, Field, field_validator

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.extras.vllm import (
    HAS_VLLM,
    AsyncEngineArgs,
    AsyncLLMEngine,
    RequestOutput,
    SamplingParams,
)
from guidellm.logger import logger
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    StandardBaseModel,
)

try:
    from guidellm.extras.audio import _decode_audio

    HAS_AUDIO = True
except ImportError:
    _decode_audio = None  # type: ignore[assignment]
    HAS_AUDIO = False

try:
    from guidellm.extras.vision import image_dict_to_pil

    HAS_VISION = True
except ImportError:
    image_dict_to_pil = None  # type: ignore[assignment]
    HAS_VISION = False

# Sentinel for "chat template not yet resolved" cache.
_CHAT_TEMPLATE_UNSET: object = object()

__all__ = ["VLLMPythonBackend", "VLLMPythonBackendArgs"]


class VLLMPythonBackendArgs(BackendArgs):
    """Pydantic model for VLLM Python backend creation arguments."""

    model: str = Field(
        description="Model identifier or path for VLLM to load",
        json_schema_extra={
            "error_message": (
                "Backend '{backend_type}' requires a model parameter. "
                "Please provide --model with a valid model identifier."
            )
        },
    )
    target: str | None = Field(
        default=None,
        description="Target URL (ignored for VLLM Python backend, runs locally)",
        json_schema_extra={
            "error_message": (
                "Backend '{backend_type}' does not support a target parameter. "
                "Please remove --target as this backend runs locally."
            )
        },
    )
    request_format: str | None = Field(
        default=None,
        description=(
            "Request format for VLLM Python backend. "
            "Valid values: 'plain' (no chat template), 'default-template' "
            "(use tokenizer default), or a path to / inline Jinja2 chat template."
        ),
        json_schema_extra={
            "error_message": (
                "Backend '{backend_type}' received an invalid --request-format. "
                "Valid values: 'plain', 'default-template', a path to a Jinja2 "
                "template file, or an inline Jinja2 template string."
            )
        },
    )

    @field_validator("target")
    @classmethod
    def target_must_be_none(cls, v: str | None) -> str | None:
        """Reject target to prevent confusion.

        Validated by CLI before Backend.create.
        """
        if v is not None:
            raise ValueError("Target is not supported; this backend runs locally.")
        return v


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


def _check_vllm_available() -> None:
    """Check if vllm is available and raise helpful error if not."""
    if not HAS_VLLM:
        raise ImportError(
            "vllm is not installed. Install vllm to use the vllm python backend."
        )


def _has_jinja2_markers(s: str) -> bool:
    """Return True if the string contains Jinja2 template syntax ({{, {%, or {#)."""
    return "{{" in s or "{%" in s or "{#" in s


@Backend.register("vllm_python")
class VLLMPythonBackend(Backend):
    """
    Python API backend for VLLM inference engine.

    Directly uses VLLM's AsyncLLMEngine for local async inference. When CUDA is not
    available and ``device`` is not set in vllm_config, the backend sets
    ``device="cpu"`` so the engine runs on CPU; otherwise vLLM uses CUDA if
    available. You can pass ``device`` in vllm_config (e.g. ``"cpu"``, ``"cuda"``)
    and it is passed through to AsyncEngineArgs. Handles request/response conversion
    between GuideLLM schemas and VLLM's native API, with async support for finer
    token-by-token processing and timings.

    Engine parameters not set in vllm_config use vLLM's AsyncEngineArgs defaults.
    Example (optional overrides):
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
        model: str,
        vllm_config: dict[str, Any] | None = None,
        request_format: str | None = None,
        stream: bool = True,
        image_placeholder: str | None = None,
        audio_placeholder: str | None = None,
    ):
        """
        Initialize VLLM Python backend with model and configuration.

        :param model: Model identifier or path for VLLM to load
        :param vllm_config: Optional dict of VLLM AsyncEngineArgs parameters.
            Passed through with no GuideLLM defaults; only model (and optionally
            chat_template) are set by the backend. When CUDA is not available and
            ``device`` is not set here, the backend sets ``device="cpu"``. You can
            pass ``device`` (e.g. ``"cpu"``, ``"cuda"``) and it is passed through.
            Unset parameters use vLLM's defaults. Common options include
            tensor_parallel_size, gpu_memory_utilization, max_model_len, and any
            other parameter accepted by vllm.AsyncEngineArgs.
        :param request_format: "plain" (no chat template), "default-template"
            (use tokenizer default), or a chat template path / single-line string.
        :param stream: Whether to stream responses (default True).
        :param image_placeholder: Optional string to use as the image placeholder when
            injecting placeholders for multimodal prompts (e.g. Qwen3-VL may require
            a model-specific token). If not set, falls back to "<image>".
        :param audio_placeholder: Optional string to use as the audio placeholder when
            using audio_column; if unset, falls back to "<|audio|>".
        """
        _check_vllm_available()
        super().__init__(type_="vllm_python")

        self.model = model
        self.request_format = request_format
        self.stream = stream
        self._image_placeholder_override = image_placeholder
        self._audio_placeholder_override = audio_placeholder
        self.vllm_config = self._merge_config(vllm_config or {})

        # Runtime state
        self._in_process = False
        self._engine: AsyncLLMEngine | None = None
        self._resolved_chat_template: str | None | object = _CHAT_TEMPLATE_UNSET

    @property
    def processes_limit(self) -> int | None:
        """
        Limits VLLM Python to a single process, since it starts up an entire
        VLLM engine.
        """
        return 1

    def _merge_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """
        Build engine config from user config plus required model.

        No GuideLLM defaults are applied; any parameter not set here or in
        user_config is left to vLLM's AsyncEngineArgs defaults. Custom
        request_format (chat template) is not passed to the engine; it is
        applied at request time in _resolve_request to avoid AsyncEngineArgs
        compatibility issues across vLLM versions.

        :param user_config: User-provided configuration dictionary
        :return: Config dict for AsyncEngineArgs (model set)
        """
        config = dict(user_config)

        # Ensure model is set in config (required; overrides user if they passed it)
        if "model" in config:
            logger.warning(
                "The `model` input was passed to the vllm python backend "
                "with the `vllm_config` input. Ignoring and overwriting "
                "with the value from the `model` input."
            )
        config["model"] = self.model

        return config

    @property
    def info(self) -> dict[str, Any]:
        """
        Get backend configuration details.

        :return: Dictionary containing backend configuration details
        """
        return {
            "model": self.model,
            "vllm_config": self.vllm_config,
            "stream": self.stream,
            "in_process": self._in_process,
            "engine_initialized": self._engine is not None,
        }

    async def process_startup(self):
        """
        Initialize VLLM AsyncLLMEngine instance with configured parameters.

        :raises RuntimeError: If backend is already initialized
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        engine_args = AsyncEngineArgs(**self.vllm_config)  # type: ignore[misc]
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)  # type: ignore[misc]
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
        return [self.model]

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: Model name or identifier
        """
        return self.model

    def _validate_backend_initialized(self) -> AsyncLLMEngine:
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

    def _build_multi_modal_data_from_columns(  # noqa: C901, PLR0912
        self, columns: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Build vLLM multi_modal_data dict from image_column, audio_column.

        video_column is not yet supported (no frame extraction); it is skipped.
        """
        multi_modal_data: dict[str, Any] = {}
        # We look specifically for "image_column" and "audio_column" which contain lists
        # of dicts
        image_items = columns.get("image_column", [])
        audio_items = columns.get("audio_column", [])
        # video_column: not yet supported; would require frame extraction
        for item in image_items:
            if not item or not isinstance(item, dict):
                continue
            if not HAS_VISION or image_dict_to_pil is None:
                raise ImportError(
                    "Image column support requires guidellm[vision]. "
                    "Install with: pip install 'guidellm[vision]'"
                )
            # Convert raw image dicts into PIL Images as required by vLLM's vision
            # processor
            pil_image = image_dict_to_pil(item)
            if "image" not in multi_modal_data:
                multi_modal_data["image"] = pil_image
            else:
                # If multiple images exist, vLLM expects a list of PIL Images
                existing = multi_modal_data["image"]
                if isinstance(existing, list):
                    existing.append(pil_image)
                else:
                    multi_modal_data["image"] = [existing, pil_image]
        if audio_items:
            if len(audio_items) > 1:
                logger.warning(
                    "Only one audio item per request is supported; "
                    "ignoring {} extra audio item(s).",
                    len(audio_items) - 1,
                )
            first = audio_items[0]
            if not first or not isinstance(first, dict):
                logger.warning("audio_column item is empty or not a dict; skipping.")
            else:
                audio_bytes = first.get("audio")
                if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                    if not HAS_AUDIO or _decode_audio is None:
                        raise ImportError(
                            "Audio column support requires guidellm[audio]. "
                            "Install with: pip install 'guidellm[audio]'"
                        )
                    try:
                        # Decode raw audio bytes into an array since vLLM audio models
                        # expect either raw numpy arrays or specific tensor formats
                        audio_samples = _decode_audio(audio_bytes)
                        # torchcodec decodes audio on CPU, so .data is always
                        # a CPU torch.Tensor. .cpu() is a no-op on CPU tensors.
                        audio_array = audio_samples.data.cpu().numpy()
                        multi_modal_data["audio"] = audio_array
                    except (ValueError, TypeError, OSError, RuntimeError) as exc:
                        raise ValueError(
                            f"Failed to decode audio from audio_column for vLLM: {exc}"
                        ) from exc
        return multi_modal_data if multi_modal_data else None

    def _extract_text_from_content(
        self, content: str | list[dict[str, Any]] | Any
    ) -> str:
        """
        Extract text content from message content field.

        Handles both string content and list-based multimodal content blocks.
        For list-based content, extracts text from blocks with type "text" and
        concatenates them together.

        :param content: Content field which can be a string or list of content blocks
        :return: Extracted text string
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from content blocks with type "text"
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if text:
                            text_parts.append(text)
            return "".join(text_parts)
        # Fallback: convert to string
        return str(content) if content is not None else ""

    def _build_placeholder_prefix(self, multi_modal_data: dict[str, Any]) -> str:
        """
        Build the placeholder prefix string for all modalities in
        multi_modal_data.

        Returns a string like ``"<image>\\n<|audio|>\\n"`` with one
        placeholder per item, or ``""`` if no multimodal items are
        present.  Placeholder tokens default to ``<image>`` and
        ``<|audio|>`` but can be overridden via
        ``image_placeholder`` / ``audio_placeholder`` at construction.
        """
        parts: list[str] = []
        images = multi_modal_data.get("image")
        if images is not None:
            num = len(images) if isinstance(images, list | tuple) else 1
            if num > 0:
                ph = self._image_placeholder_override or "<image>"
                parts.extend([ph] * num)
        audio = multi_modal_data.get("audio")
        if audio is not None:
            # Single audio item (numpy array) — not a list of items.
            num = len(audio) if isinstance(audio, list | tuple) else 1
            if num > 0:
                ph = self._audio_placeholder_override or "<|audio|>"
                parts.extend([ph] * num)
        if not parts:
            return ""
        return "\n".join(parts) + "\n"

    @staticmethod
    def _format_column_blocks(
        column_data: list[Any], column_type: str
    ) -> list[dict[str, Any]]:
        """Format data column items into vLLM-compatible content blocks.

        Analogous to the HTTP backend's ``_format_prompts`` but emitting
        vLLM-specific block types that chat templates can render into the
        correct model-specific placeholder tokens.
        """
        blocks: list[dict[str, Any]] = []
        for item in column_data:
            if not item:
                continue
            if column_type == "text_column":
                blocks.append({"type": "text", "text": str(item)})
            elif column_type == "image_column":
                blocks.append({"type": "image"})
            elif column_type == "audio_column":
                blocks.append({"type": "audio"})
        return blocks

    def _inject_placeholders_into_messages(
        self,
        formatted_messages: list[dict[str, Any]],
        multi_modal_data: dict[str, Any],
    ) -> None:
        """
        Inject multimodal placeholder tokens into the last user message's content.

        vLLM requires one placeholder per multimodal item in the prompt text so its
        processor can apply prompt replacement. This must happen *before* the chat
        template is applied so that placeholders end up inside the correct message
        turn (not prepended to the entire formatted prompt).
        """
        prefix = self._build_placeholder_prefix(multi_modal_data)
        if not prefix:
            return
        for msg in reversed(formatted_messages):
            if msg.get("role") == "user":
                msg["content"] = prefix + (msg.get("content") or "")
                return
        if formatted_messages:
            formatted_messages[-1]["content"] = prefix + (
                formatted_messages[-1].get("content") or ""
            )

    def _extract_prompt_chat_plain(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Concatenate message content into a single raw prompt string.

        Equivalent to the HTTP /v1/completions behaviour: prefix + text
        with no role prefixes or trailing generation prompt.
        """
        return " ".join(
            msg["content"] for msg in formatted_messages if msg.get("content")
        )

    def _resolve_chat_template(self) -> str | None:
        """
        Resolve and validate request_format to a template string or None.

        Returns None for default tokenizer template; returns the template string
        when valid. Raises ValueError for invalid input (wrong format, bad path,
        or invalid Jinja2 syntax).
        """
        if self.request_format is None or self.request_format in (
            "plain",
            "default-template",
        ):
            # No custom template provided; 'plain' and 'default-template' are handled
            # internally
            return None
        value = self.request_format
        path = Path(value)
        # Treat the request_format string as a file path. If it exists and contains
        # Jinja2 syntax, read the content as the template.
        if path.exists() and path.is_file():
            content = path.read_text()
            if not _has_jinja2_markers(content):
                raise ValueError(
                    "Invalid chat template: path "
                    f"{path.as_posix()!r} exists but file content does not "
                    "contain Jinja2 template syntax ({{, {%}, or {#})."
                )
            try:
                jinja2.Template(content)
            except jinja2.TemplateSyntaxError as e:
                raise ValueError(
                    f"Invalid chat template in file {path.as_posix()!r}: {e}"
                ) from e
            return content
        if _has_jinja2_markers(value):
            try:
                jinja2.Template(value)
            except jinja2.TemplateSyntaxError as e:
                raise ValueError(f"Invalid chat template: {e}") from e
            return value
        raise ValueError(
            "request_format must be 'plain', 'default-template', a path to a "
            "Jinja2 template file, or a string containing Jinja2 template "
            "syntax ({{, {%}, or {#). Got: " + repr(value) + "."
        )

    def _extract_prompt_chat_tokenizer(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Apply tokenizer chat template to formatted messages."""
        engine = self._validate_backend_initialized()
        tokenizer = engine.tokenizer
        if tokenizer is None:
            raise RuntimeError("Backend engine has no tokenizer.")

        if self.request_format is None or self.request_format in (
            "plain",
            "default-template",
        ):
            resolved: str | None = None
        else:
            if self._resolved_chat_template is _CHAT_TEMPLATE_UNSET:
                self._resolved_chat_template = self._resolve_chat_template()
            resolved = cast("str | None", self._resolved_chat_template)
        if resolved is not None:
            # Safe to mutate: vLLM runs one model per engine and the resolved
            # template is constant across all requests for this backend instance.
            tokenizer.chat_template = resolved  # type: ignore[attr-defined]
        prompt = tokenizer.apply_chat_template(
            formatted_messages,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt, str):
            return prompt
        raise RuntimeError("Backend received unexpected type from tokenizer.")

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
            and self.request_format != "plain"
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

                if self.request_format == "plain":
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
            stream=self.stream,
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

    def _text_from_output(self, output: RequestOutput | None) -> str:
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
        output: RequestOutput,
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
        output: RequestOutput | None,
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
        final_output: RequestOutput | None,
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
    ) -> SamplingParams:
        """
        Create VLLM SamplingParams.

        When max_tokens_override is set (from benchmark output_metrics), it is used
        as max_tokens and EOS is ignored to force generation of exactly that many
        tokens, matching HTTP backend behavior. Otherwise vLLM defaults are used
        (generate until EOS or model max context).

        :param max_tokens_override: Optional max_tokens from request (e.g. benchmark)
        :return: Configured SamplingParams instance
        """
        params: dict[str, Any] = {}

        if max_tokens_override is not None and max_tokens_override > 0:
            params["max_tokens"] = max_tokens_override
            params["ignore_eos"] = True

        return SamplingParams(**params)  # type: ignore[misc]

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
                f"Generation failed: The model '{self.model}' does not "
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
        sampling_params: SamplingParams,
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
