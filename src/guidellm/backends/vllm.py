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
from typing import Any

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
from guidellm.backends.response_handlers import GenerationResponseHandlerFactory
from guidellm.schemas import GenerationRequest, GenerationResponse, RequestInfo, UsageMetrics
from guidellm.utils import json

__all__ = ["VLLMPythonBackend"]


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
        target: str | None = None,  # Ignored for VLLM Python backend
        response_handlers: dict[str, Any] | None = None,
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
        """
        _check_vllm_available()
        super().__init__(type_="vllm_python")

        self.model = model
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

        :param user_config: User-provided configuration dictionary
        :return: Merged configuration with defaults applied
        """
        config = self.DEFAULT_VLLM_CONFIG.copy()

        # Merge user config, allowing overrides
        config.update(user_config)

        # Ensure model is set in config
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

    def _validate_request_type(self, request_type: str) -> None:
        """
        Validate that the request type is supported.

        :param request_type: The request type to validate
        :raises ValueError: If request type is not supported
        """
        if request_type not in ("text_completions", "chat_completions"):
            raise ValueError(
                f"Unsupported request type '{request_type}'. "
                "Only 'text_completions' and 'chat_completions' are supported."
            )

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

    def _extract_prompt(self, request: GenerationRequest) -> str:
        """
        Extract prompt text from generation request.

        For chat completions, uses the tokenizer's chat template if available,
        otherwise falls back to simple formatting.

        :param request: Generation request with body containing prompt data
        :return: Extracted prompt string
        :raises ValueError: If prompt cannot be extracted
        """
        body = request.arguments.body or {}

        if request.request_type == "text_completions":
            prompt = body.get("prompt", "")
            if not prompt:
                raise ValueError("text_completions request must include 'prompt' in body")
            return prompt if isinstance(prompt, str) else str(prompt)

        if request.request_type == "chat_completions":
            messages = body.get("messages", [])
            if not messages:
                raise ValueError(
                    "chat_completions request must include 'messages' in body"
                )

            # Try to use tokenizer's chat template if available
            if self._engine is None:
                raise RuntimeError("Backend not started up while extracting prompt.")
            tokenizer = self._engine.tokenizer

            # Convert messages to format expected by chat template
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    raw_content = msg.get("content", "")
                    # Extract text from content (handles both string and list formats)
                    text_content = self._extract_text_from_content(raw_content)
                    formatted_messages.append({
                        "role": msg.get("role", ""),
                        "content": text_content,
                    })
                else:
                    formatted_messages.append(msg)

            prompt = tokenizer.apply_chat_template(  # type: ignore[misc]
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(prompt, str):
                return prompt
            else:
                raise RuntimeError("Code not setup for state")

        raise ValueError(f"Unsupported request type: {request.request_type}")

    def _create_sampling_params(self, request: GenerationRequest) -> SamplingParams:
        """
        Create VLLM SamplingParams from generation request.

        :param request: Generation request with parameters
        :return: Configured SamplingParams instance
        """
        body = request.arguments.body or {}

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
        self, request: GenerationRequest, output: RequestOutput
    ) -> dict[str, Any]:
        """
        Convert VLLM RequestOutput to OpenAI-style dict format.

        Converts VLLM's native output format to the format expected by
        GenerationResponseHandler, matching OpenAI API response structure.

        :param request: Original generation request
        :param output: VLLM request output
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

        # Build choices array based on request type
        if request.request_type == "text_completions":
            choices = [{"text": generated_text}]
        elif request.request_type == "chat_completions":
            choices = [{"message": {"content": generated_text, "role": "assistant"}}]
        else:
            choices = [{"text": generated_text}]

        # Build response dict
        response_dict: dict[str, Any] = {
            "choices": choices,
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
        :raises ValueError: If request type is unsupported
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        # Validate request
        self._validate_backend_initialized()
        self._validate_history(history)
        self._validate_request_type(request.request_type)

        # Create response handler
        response_handler = GenerationResponseHandlerFactory.create(request.request_type)

        # Extract prompt and create sampling parameters
        prompt = self._extract_prompt(request)
        sampling_params = self._create_sampling_params(request)
        stream = request.arguments.stream or False

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
                        request, final_output
                    )
                    yield response_handler.compile_non_streaming(request, openai_format), request_info

            except asyncio.CancelledError as err:
                # Yield current result to store iterative results before propagating
                if final_output is not None:
                    openai_format = self._convert_vllm_output_to_openai_format(
                        request, final_output
                    )
                    yield response_handler.compile_non_streaming(request, openai_format), request_info
                raise err
            except Exception as exc:
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

                        # Build streaming delta format based on request type
                        if request.request_type == "text_completions":
                            delta_data = {"choices": [{"text": delta_text}]}
                        else:  # chat_completions
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
                # Re-raise with context
                raise RuntimeError(f"Generation failed: {exc}") from exc

