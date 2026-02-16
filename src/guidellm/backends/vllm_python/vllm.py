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
from typing import Any, Literal, cast

import numpy as np

from guidellm.backends.backend import Backend
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.extras.audio import _decode_audio
from guidellm.logger import logger
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)
from guidellm.utils import json

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput

    HAS_VLLM = True
except ImportError:
    AsyncLLMEngine = None  # type: ignore[assignment, misc]
    AsyncEngineArgs = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]
    RequestOutput = None  # type: ignore[assignment, misc]
    HAS_VLLM = False

_Mode = Literal["audio", "chat", "text"]

# Audio file tuple: (path_or_name, data_bytes) or (path_or_name, data_bytes, mimetype)
_AUDIO_FILE_MIN_LEN = 2
_AUDIO_FILE_DATA_INDEX = 1
_AUDIO_FILE_MIMETYPE_INDEX = 2

__all__ = ["VLLMPythonBackend"]


class _RequestContext:
    """Resolved mode, body, stream, and files for a generation request."""

    __slots__ = ("mode", "body", "stream", "files")

    def __init__(
        self,
        mode: _Mode,
        body: dict[str, Any],
        stream: bool,
        files: dict[str, Any],
    ) -> None:
        self.mode = mode
        self.body = body
        self.stream = stream
        self.files = files


def _check_vllm_available() -> None:
    """Check if vllm is available and raise helpful error if not."""
    if not HAS_VLLM:
        raise ImportError(
            "vllm is not installed. Please install it using "
            "'pip install guidellm[vllm]' or 'pip install vllm>=0.6.0'"
        )


@Backend.register("vllm_python")
class VLLMPythonBackend(Backend):
    """
    Python API backend for VLLM inference engine.

    Directly uses VLLM's AsyncLLMEngine for local async inference. VLLM automatically
    uses CUDA if available, otherwise falls back to CPU. Handles request/response
    conversion between GuideLLM schemas and VLLM's native API, with async support for
    finer token-by-token processing and timings.

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
    def requires_target(cls) -> bool:
        """
        VLLM Python backend does not require a target URL.

        This backend runs locally and does not connect to a remote server.

        :return: False, as this backend does not require a target
        """
        return False

    @classmethod
    def requires_model(cls) -> bool:
        """
        VLLM Python backend requires a model parameter.

        The model must be specified to know which model to load for inference.

        :return: True, as this backend requires a model parameter
        """
        return True

    def __init__(
        self,
        model: str,
        vllm_config: dict[str, Any] | None = None,
        request_format: str | None = None,
        target: str | None = None,  # Backend.create API; unused (local)
        stream: bool = True,
    ):
        """
        Initialize VLLM Python backend with model and configuration.

        :param model: Model identifier or path for VLLM to load
        :param vllm_config: Optional dict of VLLM AsyncEngineArgs parameters.
            Passed through with no GuideLLM defaults; only model (and optionally
            chat_template) are set by the backend. Unset parameters use vLLM's
            defaults. Common options include tensor_parallel_size,
            gpu_memory_utilization, max_model_len, and any other parameter
            accepted by vllm.AsyncEngineArgs. VLLM uses CUDA if available, else CPU.
            The 'device' parameter is not supported and will be ignored.
        :param request_format: "plain" (no chat template), "default-template"
            (use tokenizer default), or a chat template path / single-line string.
        :param target: Target URL (ignored for VLLM Python backend, runs locally)
        :param stream: Whether to stream responses (default True). Can be overridden per
            request via request.arguments.stream.
        """
        _check_vllm_available()
        super().__init__(type_="vllm_python")
        _ = target  # Required by Backend.create(); unused for local backend

        self.model = model
        self.request_format = request_format
        self.stream = stream
        self.vllm_config = self._merge_config(vllm_config or {})

        # Runtime state
        self._in_process = False
        self._engine: AsyncLLMEngine | None = None

    @property
    def processes_limit(self) -> int | None:
        """
        Limits VLLM Python to a single process, since it starts up an entire
        VLLM engine.
        """
        return 1

    def _merge_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """
        Build engine config from user config plus required model and optional
        chat_template.

        No GuideLLM defaults are applied; any parameter not set here or in
        user_config is left to vLLM's AsyncEngineArgs defaults. When
        request_format is a custom template (not plain or default-template),
        adds chat_template to config.

        :param user_config: User-provided configuration dictionary
        :return: Config dict for AsyncEngineArgs (model set; chat_template if set)
        """
        config = dict(user_config)

        # Ensure model is set in config (required; overrides user if they passed it)
        config["model"] = self.model

        # Pass custom chat template to vLLM engine when applicable
        if self.request_format is not None and self.request_format not in (
            "plain",
            "default-template",
        ):
            config["chat_template"] = self.request_format

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
        :raises ImportError: If vllm is not available
        """
        _check_vllm_available()

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

        # Shutdown the async engine if it has a shutdown method
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend readiness by attempting a test generation.

        :raises RuntimeError: If backend cannot generate or is not initialized
        """
        if self._engine is None:
            raise RuntimeError("Backend not started up for process.")

        # Perform a minimal test generation to verify the model is working
        try:
            test_params = SamplingParams(temperature=0.0, max_tokens=1)  # type: ignore[misc]
            request_id = str(uuid.uuid4())
            # Use async generation for validation
            async for _ in self._engine.generate("test", test_params, request_id):  # type: ignore[misc]
                # Just consume the first output to verify it works
                break
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            raise RuntimeError(
                "Backend validation failed. VLLM model could not generate text."
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

    def _validate_backend_initialized(self) -> None:
        """
        Validate that the backend is initialized and ready.

        :raises RuntimeError: If backend is not initialized
        """
        if self._engine is None:
            raise RuntimeError("Backend not started up for process.")

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

    def _get_request_context(self, request: GenerationRequest) -> _RequestContext:
        """
        Resolve body, stream, and files from request; infer mode from that data.

        When the request has arguments, use body/stream/files from them. Otherwise
        build body from request.columns (always messages) so the standard benchmark
        pipeline works. Mode is inferred from body and files (audio/chat/text).
        """
        arguments = getattr(request, "arguments", None)

        if arguments is not None:
            body = getattr(arguments, "body", None) or {}
            stream_override = getattr(arguments, "stream", None)
            stream = self.stream if stream_override is None else bool(stream_override)
            files = getattr(arguments, "files", None) or {}
        else:
            columns = getattr(request, "columns", {}) or {}
            prefix_parts = list(columns.get("prefix_column", []) or [])
            text_parts = list(columns.get("text_column", []) or [])
            messages: list[dict[str, Any]] = []
            for p in prefix_parts:
                if p:
                    messages.append({"role": "system", "content": str(p)})
            for t in text_parts:
                if t:
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": str(t)}],
                        }
                    )
            body = {"messages": messages}
            stream = self.stream
            files = {}

        if files:
            mode: _Mode = "audio"
        elif body.get("messages"):
            mode = "chat"
        elif (
            body.get("prompt") is not None
            and body.get("prompt") != ""
            or "prompt" in body
        ):
            mode = "text"
        else:
            raise ValueError("Request must include prompt, messages, or audio files.")

        return _RequestContext(mode=mode, body=body, stream=stream, files=files)

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
            request_info.timings.token_iterations = 0

        request_info.timings.last_token_iteration = iter_time
        request_info.timings.token_iterations += iterations

    def _text_from_output(self, output: RequestOutput | None) -> str:
        """
        Extract generated text from VLLM RequestOutput.

        :param output: VLLM request output (may be None)
        :return: output.outputs[0].text if present, else ""
        """
        if output is None or not output.outputs or len(output.outputs) == 0:
            return ""
        return output.outputs[0].text or ""

    def _openai_payload_for_mode(
        self, mode: _Mode, text: str, *, is_delta: bool = False
    ) -> dict[str, Any]:
        """
        Build OpenAI-style payload (content/choices only, no usage or id) for a mode.

        :param mode: Inferred mode (audio, chat, text)
        :param text: Content text (full or delta)
        :param is_delta: If True, chat uses delta shape; else uses message shape
        :return: Dict with "text" (audio) or "choices" (text/chat)
        """
        if mode == "audio":
            return {"text": text}
        if mode == "text":
            return {"choices": [{"text": text}]}
        # chat
        if is_delta:
            return {"choices": [{"delta": {"content": text, "role": "assistant"}}]}
        return {"choices": [{"message": {"content": text, "role": "assistant"}}]}

    def _stream_usage_tokens(
        self,
        output: RequestOutput,
        accumulated_output_tokens: int | None,
        accumulated_text: str,
        request_info: RequestInfo,
    ) -> tuple[int, int]:
        """
        Stream path: compute output_tokens; return input and output token counts
        in the form of (input_tokens, output_tokens).
        """
        input_tokens = len(output.prompt_token_ids or [])
        output_tokens = 0
        source = "unknown"
        generated_text = accumulated_text

        if accumulated_output_tokens is not None and accumulated_output_tokens > 0:
            output_tokens = accumulated_output_tokens
            source = "token_ids"
        elif output.outputs and len(output.outputs) > 0:
            out = output.outputs[0]
            if not generated_text and out.text:
                generated_text = out.text
            if out.token_ids is not None:
                output_tokens = len(out.token_ids)
                source = "token_ids"
            else:
                logger.debug(
                    "[vllm_python streaming] final_output.outputs[0].token_ids "
                    "is None, len(generated_text)={}",
                    len(generated_text or ""),
                )
        if output_tokens == 0 and generated_text and self._engine is not None:
            try:
                tokenizer = getattr(self._engine, "tokenizer", None)
                if tokenizer is not None:
                    ids = tokenizer.encode(  # type: ignore[union-attr]
                        generated_text, add_special_tokens=False
                    )
                    output_tokens = len(ids)
                    source = "tokenizer"
                else:
                    logger.debug(
                        "[vllm_python streaming] engine has no tokenizer attribute"
                    )
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "[vllm_python streaming] tokenizer.encode failed: {}",
                    exc,
                )
        if output_tokens == 0 and request_info.timings.token_iterations:
            output_tokens = request_info.timings.token_iterations
            source = "token_iterations"
        logger.debug(
            "[vllm_python streaming usage] source={} completion_tokens={} "
            "prompt_tokens={} token_iterations={}",
            source,
            output_tokens,
            input_tokens,
            request_info.timings.token_iterations,
        )
        return input_tokens, output_tokens

    def _usage_from_output(
        self,
        output: RequestOutput | None,
        *,
        accumulated_output_tokens: int | None = None,
        accumulated_text: str = "",
        request_info: RequestInfo | None = None,
    ) -> dict[str, int] | None:
        """
        Build usage dict (prompt_tokens, completion_tokens, total_tokens).

        When request_info is None (non-stream), uses only output token counts.
        When request_info is set (stream), uses fallback chain via
        _stream_usage_tokens.
        """
        if output is None:
            return None
        input_tokens = len(output.prompt_token_ids or [])
        output_tokens = 0

        if request_info is None:
            if output.outputs and len(output.outputs) > 0:
                out = output.outputs[0]
                if out.token_ids is not None:
                    output_tokens = len(out.token_ids)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        input_tokens, output_tokens = self._stream_usage_tokens(
            output, accumulated_output_tokens, accumulated_text, request_info
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
        ctx: _RequestContext,
        response_handler: VLLMResponseHandler,
        stream: bool,
        accumulated_text: str = "",
        total_output_tokens: int = 0,
    ) -> tuple[GenerationResponse, RequestInfo] | None:
        """Build and return the final (response, request_info) to yield, or None."""
        if final_output is None:
            return None
        if stream:
            usage = self._usage_from_output(
                final_output,
                accumulated_output_tokens=total_output_tokens,
                accumulated_text=accumulated_text,
                request_info=request_info,
            )
            if usage is not None:
                response_handler.streaming_usage = usage
            return response_handler.compile_streaming(request), request_info
        openai_format = self._convert_vllm_output_to_openai_format(
            final_output, ctx.mode
        )
        return response_handler.compile_non_streaming(
            request, openai_format
        ), request_info

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
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text = block.get("text", "")
                        if text:
                            text_parts.append(text)
            return "".join(text_parts)
        # Fallback: convert to string
        return str(content) if content is not None else ""

    def _extract_audio_from_request(self, files: dict[str, Any]) -> tuple[bytes, str]:
        """
        Extract audio file from request files dict.

        Caller must only call when mode is audio. Audio files are expected in
        the format (filename, bytes, mimetype) or as raw bytes.

        :param files: Files dict from request context
        :return: Tuple of (audio_bytes, mimetype)
        :raises ValueError: If files is empty or no valid audio file found
        """
        if not files:
            raise ValueError("Audio request must include audio file in 'files'.")

        audio_file = None
        for value in files.values():
            if isinstance(value, list | tuple) and len(value) >= _AUDIO_FILE_MIN_LEN:
                audio_file = value
                break
            if isinstance(value, bytes):
                audio_file = ("audio.wav", value, "audio/wav")
                break

        if audio_file is None:
            raise ValueError("Audio request must include audio file in 'files'.")

        is_audio_tuple = (
            isinstance(audio_file, list | tuple)
            and len(audio_file) >= _AUDIO_FILE_MIN_LEN
        )
        if is_audio_tuple:
            audio_data = audio_file[_AUDIO_FILE_DATA_INDEX]
            if not isinstance(audio_data, bytes):
                raise ValueError(
                    f"Expected bytes for audio data, got {type(audio_data)}: "
                    f"{audio_data}"
                )
            mimetype = (
                audio_file[_AUDIO_FILE_MIMETYPE_INDEX]
                if len(audio_file) > _AUDIO_FILE_MIMETYPE_INDEX
                else "audio/wav"
            )
            return (audio_data, mimetype)

        raise ValueError("Invalid audio file format in 'files'.")

    def _extract_prompt_text(self, body: dict[str, Any]) -> str:
        """Extract prompt for text mode."""
        if (
            self._engine is not None
            and getattr(self._engine, "tokenizer", None) is not None
        ):
            logger.warning(
                "Tokenizer is set (from model) but not used: request has 'prompt' "
                "so mode was inferred as text; prompt is passed through without "
                "tokenizer formatting."
            )
        prompt = body.get("prompt", "")
        if not prompt:
            raise ValueError("Text request must include 'prompt' in body.")
        return prompt if isinstance(prompt, str) else str(prompt)

    def _extract_prompt_chat_plain(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Format messages as plain 'Role: content' lines."""
        parts = []
        for msg in formatted_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role and content:
                parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant: ")
        return "\n".join(parts)

    def _extract_prompt_chat_tokenizer(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Apply tokenizer chat template to formatted messages."""
        if self._engine is None:
            raise RuntimeError("Backend not started up while extracting prompt.")
        tokenizer = self._engine.tokenizer
        if tokenizer is None:
            raise RuntimeError("Backend engine has no tokenizer.")
        if self.request_format is not None and self.request_format not in (
            "plain",
            "default-template",
        ):
            template_value = self.request_format
            path = Path(template_value)
            if path.exists() and path.is_file():
                tokenizer.chat_template = path.read_text()  # type: ignore[attr-defined]
            else:
                tokenizer.chat_template = template_value  # type: ignore[assignment, attr-defined]
        prompt = tokenizer.apply_chat_template(
            formatted_messages,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt, str):
            return prompt
        raise RuntimeError("Backend received unexpected type from tokenizer.")

    def _extract_prompt_chat(self, body: dict[str, Any]) -> str:
        """Extract prompt for chat mode (plain or tokenizer)."""
        messages = body.get("messages", [])
        if not messages:
            raise ValueError("Chat request must include 'messages' in body.")
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                raw_content = msg.get("content", "")
                text_content = self._extract_text_from_content(raw_content)
                formatted_messages.append(
                    {"role": msg.get("role", ""), "content": text_content}
                )
            else:
                formatted_messages.append(msg)
        if self.request_format == "plain":
            return self._extract_prompt_chat_plain(formatted_messages)
        return self._extract_prompt_chat_tokenizer(formatted_messages)

    def _extract_prompt(
        self,
        body: dict[str, Any],
        mode: _Mode,
    ) -> str:
        """
        Extract prompt text from request body based on mode.

        For chat mode, uses tokenizer chat template or plain concat per
        request_format. For text mode uses body prompt. For audio, prompt
        is optional.

        :param body: Request body dict (prompt or messages)
        :param mode: Inferred mode (audio, chat, text)
        :return: Extracted prompt string
        :raises ValueError: If prompt cannot be extracted
        """
        if mode == "text":
            return self._extract_prompt_text(body)
        if mode == "chat":
            return self._extract_prompt_chat(body)
        if mode == "audio":
            prompt = body.get("prompt", "")
            return (
                prompt if isinstance(prompt, str) else str(prompt) if prompt else ""
            )
        raise ValueError(f"Unsupported mode: {mode!r}.")

    def _create_sampling_params(
        self,
        body: dict[str, Any],
        max_tokens_override: int | None = None,
    ) -> SamplingParams:
        """
        Create VLLM SamplingParams from request body.

        :param body: Request body dict with sampling parameters
        :param max_tokens_override: Optional max_tokens from request (e.g. benchmark)
        :return: Configured SamplingParams instance
        """
        max_tokens = (
            body.get("max_tokens")
            if body.get("max_tokens") is not None
            else (max_tokens_override if max_tokens_override is not None else 16)
        )
        if max_tokens == 0:
            max_tokens = 16

        using_benchmark_target = (
            body.get("max_tokens") is None and max_tokens_override is not None
        )
        if using_benchmark_target:
            ignore_eos = True
            stop: list[str] | None = []
        else:
            ignore_eos = body.get("ignore_eos", False)
            stop = body.get("stop")

        params = {
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": max_tokens,
            "min_tokens": body.get("min_tokens", 0),
            "stop": stop,
            "ignore_eos": ignore_eos,
            "frequency_penalty": body.get("frequency_penalty", 0.0),
            "presence_penalty": body.get("presence_penalty", 0.0),
        }

        params = {k: v for k, v in params.items() if v is not None}

        return SamplingParams(**params)  # type: ignore[misc]

    def _convert_vllm_output_to_openai_format(
        self,
        output: RequestOutput,
        mode: _Mode,
    ) -> dict[str, Any]:
        """
        Convert VLLM RequestOutput to OpenAI-style dict format.

        Converts VLLM's native output format to the format expected by
        VLLMResponseHandler, matching OpenAI API response structure.

        :param output: VLLM request output
        :param mode: Inferred mode (audio, chat, text) for response shape
        :return: OpenAI-style response dictionary
        """
        generated_text = self._text_from_output(output)
        usage = self._usage_from_output(output) or {}

        response_dict: dict[str, Any] = {
            **self._openai_payload_for_mode(mode, generated_text, is_delta=False),
            "usage": usage,
        }
        if hasattr(output, "request_id") and output.request_id:
            response_dict["id"] = output.request_id
        return response_dict

    def _build_audio_generate_input(
        self, prompt: str, ctx: _RequestContext
    ) -> dict[str, Any]:
        """Decode audio from request and build multimodal generate_input dict."""
        audio_bytes, _mimetype = self._extract_audio_from_request(ctx.files)
        if not isinstance(audio_bytes, bytes):
            raise TypeError(
                f"Expected bytes for audio data, got {type(audio_bytes)}: "
                f"{audio_bytes}"
            )
        if len(audio_bytes) == 0:
            raise ValueError("Audio data cannot be empty")
        try:
            audio_samples = _decode_audio(audio_bytes)
            if hasattr(audio_samples.data, "numpy"):
                audio_array = audio_samples.data.numpy()
            elif hasattr(audio_samples.data, "cpu"):
                audio_array = audio_samples.data.cpu().numpy()
            else:
                audio_array = np.asarray(audio_samples.data)
            return {
                "prompt": prompt,
                "multi_modal_data": {"audio": audio_array},
            }
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise ValueError(
                f"Failed to decode audio data for vLLM: {exc}"
            ) from exc

    def _process_streaming_delta(
        self,
        request_output: RequestOutput,
        ctx: _RequestContext,
        response_handler: VLLMResponseHandler,
        request_info: RequestInfo,
        accumulated_text: str,
        previous_token_count: int,
        iter_time: float,
    ) -> tuple[str, int, int]:
        """Emit one streaming delta; return (accum_text, token_delta, new_prev)."""
        generated_text = self._text_from_output(request_output)
        if generated_text == accumulated_text:
            return (accumulated_text, 0, previous_token_count)

        delta_text = generated_text[len(accumulated_text) :]
        delta_data = self._openai_payload_for_mode(
            ctx.mode, delta_text, is_delta=True
        )
        if hasattr(request_output, "request_id") and request_output.request_id:
            delta_data["id"] = request_output.request_id
        json_str = json.dumps(delta_data)
        json_str_decoded = (
            json_str.decode("utf-8") if isinstance(json_str, bytes) else json_str
        )
        sse_line = f"data: {json_str_decoded}"
        iterations = response_handler.add_streaming_line(sse_line)

        total_delta = 0
        new_prev = previous_token_count
        if iterations is not None and iterations > 0:
            out = (
                request_output.outputs[0]
                if request_output.outputs
                else None
            )
            if out is not None and out.token_ids is not None:
                current_count = len(out.token_ids)
                total_delta = max(0, current_count - previous_token_count)
                new_prev = current_count
            else:
                total_delta = 1
                new_prev = previous_token_count + 1
            self._update_token_timing(request_info, iter_time, iterations)
        return (generated_text, total_delta, new_prev)

    def _raise_generation_error(self, exc: BaseException) -> None:
        """Re-raise generation failure with context; special case for audio."""
        error_msg = str(exc)
        if (
            "At most 0 audio" in error_msg
            or "audio(s) may be provided" in error_msg
        ):
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
        ctx: _RequestContext,
        response_handler: VLLMResponseHandler,
        generate_input: str | dict[str, Any],
        sampling_params: SamplingParams,
        request_id: str,
        state: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """Run engine.generate loop and yield final (response, request_info)."""
        state["final_output"] = None
        state["accumulated_text"] = ""
        state["total_output_tokens"] = 0
        previous_token_count = 0
        stream = ctx.stream
        engine = self._engine
        if engine is None:
            raise RuntimeError("Backend not started up.")

        prompt_arg = cast("Any", generate_input)
        async for request_output in engine.generate(  # type: ignore[misc]
            prompt_arg, sampling_params, request_id
        ):
            iter_time = time.time()
            self._update_request_timing(request_info, iter_time)
            state["final_output"] = request_output

            if (
                request_output.outputs
                and len(request_output.outputs) > 0
                and request_output.outputs[0].finish_reason is not None
            ):
                break

            if stream:
                state["accumulated_text"], token_delta, previous_token_count = (
                    self._process_streaming_delta(
                        request_output,
                        ctx,
                        response_handler,
                        request_info,
                        state["accumulated_text"],
                        previous_token_count,
                        iter_time,
                    )
                )
                if token_delta > 0:
                    state["total_output_tokens"] += token_delta

        request_info.timings.request_end = time.time()
        result = self._build_final_response(
            request,
            request_info,
            state["final_output"],
            ctx,
            response_handler,
            stream,
            state["accumulated_text"],
            state["total_output_tokens"],
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

        Handles both streaming and non-streaming requests using AsyncLLMEngine's
        native async generation. For streaming, records per-token timings and
        yields once with a response compiled from the stream. For non-streaming,
        yields once with the complete response. In both cases the caller receives
        exactly one (response, request_info) pair.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialized
        :raises ValueError: If request body/files do not imply a valid mode
        :yields: Single tuple of (response, updated_request_info)
        """
        ctx = self._get_request_context(request)
        self._validate_backend_initialized()
        self._validate_history(history)
        response_handler = VLLMResponseHandler()
        prompt = self._extract_prompt(ctx.body, ctx.mode)
        sampling_params = self._create_sampling_params(
            ctx.body,
            max_tokens_override=(
                request.output_metrics.text_tokens
                if request.output_metrics.text_tokens
                else None
            ),
        )
        generate_input: str | dict[str, Any]
        if ctx.mode == "audio":
            prompt_str = prompt if isinstance(prompt, str) else ""
            generate_input = self._build_audio_generate_input(prompt_str, ctx)
        else:
            generate_input = prompt

        request_id = str(uuid.uuid4())
        request_info.timings.request_start = time.time()
        engine = self._engine
        if engine is None:
            raise RuntimeError("Backend not started up.")

        gen_state: dict[str, Any] = {}
        try:
            async for result in self._run_generation(
                request,
                request_info,
                ctx,
                response_handler,
                generate_input,
                sampling_params,
                request_id,
                gen_state,
            ):
                yield result
        except asyncio.CancelledError as err:
            cancel_result = self._build_final_response(
                request,
                request_info,
                gen_state.get("final_output"),
                ctx,
                response_handler,
                ctx.stream,
                gen_state.get("accumulated_text", ""),
                gen_state.get("total_output_tokens", 0),
            )
            if cancel_result is not None:
                yield cancel_result
            raise err
        except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
            self._raise_generation_error(exc)
