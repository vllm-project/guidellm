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
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Literal

_Mode = Literal["audio", "chat", "text"]

import numpy as np

try:
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.outputs import RequestOutput

    HAS_VLLM = True
except ImportError:
    AsyncLLMEngine = None  # type: ignore[assignment, misc]
    AsyncEngineArgs = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]
    RequestOutput = None  # type: ignore[assignment, misc]
    HAS_VLLM = False

from guidellm.backends.backend import Backend
from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.extras.audio import _decode_audio
from guidellm.schemas import GenerationRequest, GenerationResponse, RequestInfo, UsageMetrics
from guidellm.utils import json

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
            "vllm is not installed. Please install it using 'pip install guidellm[vllm]' "
            "or 'pip install vllm>=0.6.0'"
        )


@Backend.register("vllm_python")
class VLLMPythonBackend(Backend):
    """
    Python API backend for VLLM inference engine.

    Directly uses VLLM's AsyncLLMEngine for local async inference. VLLM automatically
    uses CUDA if available, otherwise falls back to CPU. Handles request/response
    conversion between GuideLLM schemas and VLLM's native API, with true async
    streaming support for token-by-token generation.

    Example:
    ::
        backend = VLLMPythonBackend(
            model="meta-llama/Llama-2-7b-chat-hf",
            vllm_config={
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        )

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

    # Default VLLM configuration
    DEFAULT_VLLM_CONFIG: dict[str, Any] = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
    }

    def __init__(
        self,
        model: str,
        vllm_config: dict[str, Any] | None = None,
        request_format: str | None = None,
        target: str | None = None,  # Ignored for VLLM Python backend
    ):
        """
        Initialize VLLM Python backend with model and configuration.

        :param model: Model identifier or path for VLLM to load
        :param vllm_config: Dictionary of VLLM AsyncEngineArgs initialization parameters.
            Merged with defaults. Common parameters include:
            - tensor_parallel_size: Number of tensor parallel replicas
            - gpu_memory_utilization: Fraction of GPU memory to use
            - max_model_len: Maximum sequence length
            - Any other parameter accepted by vllm.AsyncEngineArgs
            Note: VLLM automatically uses CUDA if available, CPU otherwise.
            The 'device' parameter is not supported and will be ignored.
        :param target: Target URL (ignored for VLLM Python backend, which runs locally)
        :param kwargs: Additional arguments from the benchmark. request_format can be
            "plain" (no chat template, text appending only), "default-template" (use
            tokenizer default), or a vLLM chat template file path / single-line string.
        """
        _check_vllm_available()
        super().__init__(type_="vllm_python")

        self.model = model
        self.request_format = request_format
        self.vllm_config = self._merge_config(vllm_config or {})

        # Runtime state
        self._in_process = False
        self._engine: AsyncLLMEngine | None = None

    @property
    def processes_limit(self) -> int | None:
        """
        Limits VLLM Python to a single process, since it starts up an entire VLLM engine.
        """
        return 1

    def _merge_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """
        Merge user configuration with defaults.

        When request_format is a custom template (not plain or default-template),
        adds chat_template to config for the vLLM engine if supported.

        :param user_config: User-provided configuration dictionary
        :return: Merged configuration with defaults applied
        """
        config = self.DEFAULT_VLLM_CONFIG.copy()

        # Merge user config, allowing overrides
        config.update(user_config)

        # Ensure model is set in config
        config["model"] = self.model

        # Pass custom chat template to vLLM engine when applicable
        if (
            self.request_format is not None
            and self.request_format not in ("plain", "default-template")
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

        # Initialize VLLM AsyncLLMEngine instance
        # AsyncLLMEngine initialization is async-friendly
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
        except Exception as exc:
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
            stream = bool(getattr(arguments, "stream", False))
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
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": str(t)}],
                    })
            body = {"messages": messages}
            stream = False
            files = {}

        if files:
            mode: _Mode = "audio"
        elif body.get("messages"):
            mode = "chat"
        elif body.get("prompt") is not None and body.get("prompt") != "":
            mode = "text"
        elif "prompt" in body:
            mode = "text"
        else:
            raise ValueError(
                "Request must include prompt, messages, or audio files."
            )

        return _RequestContext(mode=mode, body=body, stream=stream, files=files)

    def _update_request_timing(self, request_info: RequestInfo, iter_time: float) -> None:
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

    def _extract_text_from_content(self, content: str | list[dict[str, Any]] | Any) -> str:
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

    def _extract_audio_from_request(
        self, files: dict[str, Any]
    ) -> tuple[bytes, str]:
        """
        Extract audio file from request files dict.

        Caller must only call when mode is audio. Audio files are expected in
        the format (filename, bytes, mimetype) or as raw bytes.

        :param files: Files dict from request context
        :return: Tuple of (audio_bytes, mimetype)
        :raises ValueError: If files is empty or no valid audio file found
        """
        if not files:
            raise ValueError(
                "Audio request must include audio file in 'files'."
            )

        audio_file = None
        for value in files.values():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                audio_file = value
                break
            elif isinstance(value, bytes):
                audio_file = ("audio.wav", value, "audio/wav")
                break

        if audio_file is None:
            raise ValueError(
                "Audio request must include audio file in 'files'."
            )

        if isinstance(audio_file, (list, tuple)) and len(audio_file) >= 2:
            audio_data = audio_file[1]
            if not isinstance(audio_data, bytes):
                raise ValueError(
                    f"Expected bytes for audio data, got {type(audio_data)}: {audio_data}"
                )
            mimetype = audio_file[2] if len(audio_file) >= 3 else "audio/wav"
            return (audio_data, mimetype)

        raise ValueError("Invalid audio file format in 'files'.")

    def _extract_prompt(
        self,
        body: dict[str, Any],
        mode: _Mode,
    ) -> str:
        """
        Extract prompt text from request body based on mode.

        For chat mode, uses the tokenizer's chat template or plain concat per
        request_format. For text mode uses body prompt. For audio, prompt is optional.

        :param body: Request body dict (prompt or messages)
        :param mode: Inferred mode (audio, chat, text)
        :return: Extracted prompt string
        :raises ValueError: If prompt cannot be extracted
        """
        if mode == "text":
            prompt = body.get("prompt", "")
            if not prompt:
                raise ValueError("Text request must include 'prompt' in body.")
            return prompt if isinstance(prompt, str) else str(prompt)

        if mode == "chat":
            messages = body.get("messages", [])
            if not messages:
                raise ValueError(
                    "Chat request must include 'messages' in body."
                )

            # Convert messages to format expected by chat template or plain concat
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    raw_content = msg.get("content", "")
                    text_content = self._extract_text_from_content(raw_content)
                    formatted_messages.append({
                        "role": msg.get("role", ""),
                        "content": text_content,
                    })
                else:
                    formatted_messages.append(msg)

            if self.request_format == "plain":
                # No chat template: append message content only (e.g. "User: ...\nAssistant: ")
                parts = []
                for msg in formatted_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role and content:
                        parts.append(f"{role.capitalize()}: {content}")
                parts.append("Assistant: ")
                return "\n".join(parts)

            # default-template, None, or custom template: use tokenizer
            if self._engine is None:
                raise RuntimeError("Backend not started up while extracting prompt.")
            tokenizer = self._engine.tokenizer

            if (
                self.request_format is not None
                and self.request_format not in ("plain", "default-template")
            ):
                # Custom template: set on tokenizer if not already passed via engine config
                template_value = self.request_format
                path = Path(template_value)
                if path.exists() and path.is_file():
                    tokenizer.chat_template = path.read_text()
                else:
                    tokenizer.chat_template = template_value  # type: ignore[assignment]

            prompt = tokenizer.apply_chat_template(  # type: ignore[misc]
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(prompt, str):
                return prompt
            raise RuntimeError("Code not setup for state")

        if mode == "audio":
            prompt = body.get("prompt", "")
            return prompt if isinstance(prompt, str) else str(prompt) if prompt else ""
        raise ValueError(f"Unsupported mode: {mode!r}.")

    def _create_sampling_params(self, body: dict[str, Any]) -> SamplingParams:
        """
        Create VLLM SamplingParams from request body.

        :param body: Request body dict with sampling parameters
        :return: Configured SamplingParams instance
        """

        # Extract common parameters
        params = {
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": body.get("max_tokens", 16),
            "stop": body.get("stop", None),
            "frequency_penalty": body.get("frequency_penalty", 0.0),
            "presence_penalty": body.get("presence_penalty", 0.0),
        }

        # Remove None values
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
        # Extract generated text
        generated_text = ""
        if output.outputs and len(output.outputs) > 0:
            generated_text = output.outputs[0].text

        # Calculate token counts
        input_tokens = len(output.prompt_token_ids)
        output_tokens = 0
        if output.outputs and len(output.outputs) > 0:
            output_obj = output.outputs[0]
            if output_obj.token_ids is not None:
                output_tokens = len(output_obj.token_ids)

        # Build usage dict
        usage: dict[str, int] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        if mode == "audio":
            response_dict: dict[str, Any] = {
                "text": generated_text,
                "usage": usage,
            }
        elif mode == "text":
            response_dict = {
                "choices": [{"text": generated_text}],
                "usage": usage,
            }
        else:
            response_dict = {
                "choices": [
                    {"message": {"content": generated_text, "role": "assistant"}}
                ],
                "usage": usage,
            }

        # Add response ID if available
        if hasattr(output, "request_id") and output.request_id:
            response_dict["id"] = output.request_id

        return response_dict

    async def resolve(  # type: ignore[override]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process generation request and yield progressive responses.

        Handles both streaming and non-streaming requests using AsyncLLMEngine's
        native async streaming capabilities. For streaming requests, yields
        token-by-token as they are generated. For non-streaming, yields once
        with the complete response.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialized
        :raises ValueError: If request body/files do not imply a valid mode
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        # Resolve request context (supports standard pipeline with columns only)
        ctx = self._get_request_context(request)

        # Validate request
        self._validate_backend_initialized()
        self._validate_history(history)

        # Create response handler (no request type; handler infers from response shape)
        response_handler = VLLMResponseHandler()

        # Extract prompt and create sampling parameters
        prompt = self._extract_prompt(ctx.body, ctx.mode)
        sampling_params = self._create_sampling_params(ctx.body)
        stream = ctx.stream

        # For audio requests, construct multimodal input with audio data
        if ctx.mode == "audio":
            audio_bytes, mimetype = self._extract_audio_from_request(ctx.files)

            # Validate audio_bytes is actually bytes
            if not isinstance(audio_bytes, bytes):
                raise TypeError(
                    f"Expected bytes for audio data, got {type(audio_bytes)}: {audio_bytes}"
                )
            if len(audio_bytes) == 0:
                raise ValueError("Audio data cannot be empty")

            # Decode audio bytes to numpy array for vLLM
            # vLLM's _get_audio_with_sr expects decoded audio samples (numpy array, torch tensor, etc.)
            # not raw encoded bytes
            try:
                audio_samples = _decode_audio(audio_bytes)
                # Convert torch tensor to numpy array (vLLM accepts numpy arrays)
                if hasattr(audio_samples.data, 'numpy'):
                    audio_array = audio_samples.data.numpy()
                elif hasattr(audio_samples.data, 'cpu'):
                    audio_array = audio_samples.data.cpu().numpy()
                else:
                    # Already a numpy array or compatible
                    audio_array = np.asarray(audio_samples.data)
                
                # vLLM can accept either:
                # 1. Just the numpy array (orig_sr will be None, no resampling)
                # 2. Tuple (array, sample_rate) - but this triggers resampling which needs target_sr
                # Since the resampler may not have target_sr configured, pass just the array
                # The model should handle the sample rate appropriately
                audio_for_vllm = audio_array
            except Exception as exc:
                raise ValueError(
                    f"Failed to decode audio data for vLLM: {exc}"
                ) from exc

            # Construct multimodal prompt for vLLM v1 transcription models
            # vLLM v1's generate method accepts multimodal data in this format
            # The prompt should be a string, and multimodal data passed as a dict
            # Note: vLLM's _parse_audio_data automatically wraps single arrays in a list
            prompt_text = prompt if isinstance(prompt, str) else ""
            
            prompt = {
                "prompt": prompt_text,
                "multi_modal_data": {
                    "audio": audio_for_vllm,
                },
            }

        # Generate unique request ID for this generation
        request_id = str(uuid.uuid4())

        if not stream:
            # Non-streaming path
            request_info.timings.request_start = time.time()
            final_output: RequestOutput | None = None

            try:
                # Use AsyncLLMEngine's native async generation
                async for request_output in self._engine.generate(  # type: ignore[misc]
                    prompt, sampling_params, request_id
                ):
                    iter_time = time.time()
                    self._update_request_timing(request_info, iter_time)
                    final_output = request_output

                    # Check if generation is finished
                    if (
                        request_output.outputs
                        and len(request_output.outputs) > 0
                        and request_output.outputs[0].finish_reason is not None
                    ):
                        break

                request_info.timings.request_end = time.time()

                # Convert to OpenAI format and use handler
                if final_output is not None:
                    openai_format = self._convert_vllm_output_to_openai_format(
                        final_output, ctx.mode
                    )
                    yield response_handler.compile_non_streaming(request, openai_format), request_info

            except asyncio.CancelledError as err:
                # Yield current result to store iterative results before propagating
                if final_output is not None:
                    openai_format = self._convert_vllm_output_to_openai_format(
                        final_output, ctx.mode
                    )
                    yield response_handler.compile_non_streaming(request, openai_format), request_info
                raise err
            except Exception as exc:
                # Provide clearer error message for audio-related issues
                error_msg = str(exc)
                if "At most 0 audio" in error_msg or "audio(s) may be provided" in error_msg:
                    raise RuntimeError(
                        f"Generation failed: The model '{self.model}' does not support audio inputs. "
                        f"For audio transcriptions, please use an audio-capable model (e.g., Whisper-based models). "
                        f"Original error: {exc}"
                    ) from exc
                # Re-raise with context
                raise RuntimeError(f"Generation failed: {exc}") from exc

        else:
            # Streaming path
            try:
                request_info.timings.request_start = time.time()
                accumulated_text = ""
                final_output: RequestOutput | None = None

                # Use AsyncLLMEngine's native async generation
                async for request_output in self._engine.generate(  # type: ignore[misc]
                    prompt, sampling_params, request_id
                ):
                    iter_time = time.time()
                    self._update_request_timing(request_info, iter_time)

                    # Extract generated text from output
                    generated_text = ""
                    if request_output.outputs and len(request_output.outputs) > 0:
                        generated_text = request_output.outputs[0].text

                    # Check if this is a new token (text has changed)
                    if generated_text != accumulated_text:
                        # Calculate delta (new text)
                        delta_text = generated_text[len(accumulated_text) :]

                        # Build streaming delta format based on mode
                        if ctx.mode == "text":
                            delta_data = {"choices": [{"text": delta_text}]}
                        elif ctx.mode == "audio":
                            delta_data = {"text": delta_text}
                        else:
                            delta_data = {
                                "choices": [{"delta": {"content": delta_text, "role": "assistant"}}]
                            }

                        # Add response ID if available
                        if hasattr(request_output, "request_id") and request_output.request_id:
                            delta_data["id"] = request_output.request_id

                        # Format as SSE line and feed to handler
                        # json.dumps may return bytes (orjson) or str, ensure it's a string
                        json_str = json.dumps(delta_data)
                        if isinstance(json_str, bytes):
                            json_str = json_str.decode("utf-8")
                        sse_line = f"data: {json_str}"
                        iterations = response_handler.add_streaming_line(sse_line)

                        if iterations is not None and iterations > 0:
                            self._update_token_timing(request_info, iter_time, iterations)

                        accumulated_text = generated_text

                    final_output = request_output

                request_info.timings.request_end = time.time()

                # Yield final compiled response
                yield response_handler.compile_streaming(request), request_info

            except asyncio.CancelledError as err:
                # Yield current result to store iterative results before propagating
                yield response_handler.compile_streaming(request), request_info
                raise err
            except Exception as exc:
                # Provide clearer error message for audio-related issues
                error_msg = str(exc)
                if "At most 0 audio" in error_msg or "audio(s) may be provided" in error_msg:
                    raise RuntimeError(
                        f"Generation failed: The model '{self.model}' does not support audio inputs. "
                        f"For audio transcriptions, please use an audio-capable model (e.g., Whisper-based models). "
                        f"Original error: {exc}"
                    ) from exc
                # Re-raise with context
                raise RuntimeError(f"Generation failed: {exc}") from exc
