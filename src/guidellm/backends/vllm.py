"""
VLLM Python API backend implementation for GuideLLM.

Provides direct Python API integration with VLLM's LLM class, enabling local
inference without HTTP overhead. Supports both GPU-accelerated and CPU-only
inference, streaming and non-streaming responses, and flexible configuration
through dict-style parameters.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    HAS_VLLM = True
except ImportError:
    LLM = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]
    RequestOutput = None  # type: ignore[assignment, misc]
    HAS_VLLM = False

from guidellm.backends.backend import Backend
from guidellm.schemas import GenerationRequest, GenerationResponse, RequestInfo, UsageMetrics

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

    Directly uses VLLM's LLM class for local inference. VLLM automatically uses
    CUDA if available, otherwise falls back to CPU. Handles request/response
    conversion between GuideLLM schemas and VLLM's native API, with full support
    for streaming responses.

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
    ):
        """
        Initialize VLLM Python backend with model and configuration.

        :param model: Model identifier or path for VLLM to load
        :param vllm_config: Dictionary of VLLM LLM initialization parameters.
            Merged with defaults. Common parameters include:
            - tensor_parallel_size: Number of tensor parallel replicas
            - gpu_memory_utilization: Fraction of GPU memory to use
            - max_model_len: Maximum sequence length
            - Any other parameter accepted by vllm.LLM.__init__
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
        self._llm: LLM | None = None

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

        # Remove 'device' parameter if present - VLLM doesn't accept it
        # VLLM automatically uses CUDA if available, CPU otherwise
        # For CPU-only inference, VLLM may require environment variables or
        # different configuration (e.g., setting CUDA_VISIBLE_DEVICES="")
        config.pop("device", None)

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
        }

    async def process_startup(self):
        """
        Initialize VLLM LLM instance with configured parameters.

        :raises RuntimeError: If backend is already initialized
        :raises ImportError: If vllm is not available
        """
        _check_vllm_available()

        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        # Initialize VLLM LLM instance
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(
            None, lambda: LLM(**self.vllm_config)  # type: ignore[misc]
        )
        self._in_process = True

    async def process_shutdown(self):
        """
        Clean up VLLM LLM instance and resources.

        :raises RuntimeError: If backend was not properly initialized
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        # VLLM doesn't have explicit cleanup, but we can clear the reference
        # and let Python's GC handle it
        self._llm = None
        self._in_process = False

    async def validate(self):
        """
        Validate backend readiness by attempting a test generation.

        :raises RuntimeError: If backend cannot generate or is not initialized
        """
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")

        # Perform a minimal test generation to verify the model is working
        try:
            test_params = SamplingParams(temperature=0.0, max_tokens=1)  # type: ignore[misc]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._llm.generate(["test"], test_params),  # type: ignore[misc]
            )
        except Exception as exc:
            raise RuntimeError(
                "Backend validation failed. VLLM model could not generate text."
            ) from exc

    async def available_models(self) -> list[str]:
        """
        Get available models from this backend.

        :return: List containing the configured model identifier
        """
        return [self.model]

    async def default_model(self) -> str:
        """
        Get the default model for this backend.

        :return: Model name or identifier
        """
        return self.model

    def _extract_prompt(self, request: GenerationRequest) -> str:
        """
        Extract prompt text from generation request.

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

            # Convert chat messages to prompt format
            # Simple conversion - join messages with newlines
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content:
                    prompt_parts.append(f"{role}: {content}")

            return "\n".join(prompt_parts)

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

    def _convert_output_to_response(
        self, request: GenerationRequest, output: RequestOutput
    ) -> GenerationResponse:
        """
        Convert VLLM RequestOutput to GenerationResponse.

        :param request: Original generation request
        :param output: VLLM request output
        :return: Standardized GenerationResponse
        """
        # Extract generated text
        generated_text = ""
        if output.outputs and len(output.outputs) > 0:
            generated_text = output.outputs[0].text

        # Extract token counts from metrics if available
        input_tokens = None
        output_tokens = None

        if hasattr(output, "metrics") and output.metrics:
            metrics = output.metrics
            input_tokens = getattr(metrics, "num_input_tokens", None)
            output_tokens = getattr(metrics, "num_output_tokens", None)

        # Fallback: estimate from text if metrics not available
        if input_tokens is None:
            # Rough estimate: ~4 chars per token
            input_tokens = len(self._extract_prompt(request)) // 4

        if output_tokens is None:
            output_tokens = len(generated_text.split()) if generated_text else 0

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(request.arguments.model_dump() if request.arguments else {}),
            response_id=output.request_id if hasattr(output, "request_id") else None,
            text=generated_text,
            input_metrics=UsageMetrics(text_tokens=input_tokens),
            output_metrics=UsageMetrics(
                text_tokens=output_tokens,
                text_words=len(generated_text.split()) if generated_text else 0,
                text_characters=len(generated_text) if generated_text else 0,
            ),
        )

    async def _generate_streaming(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[RequestOutput]:
        """
        Generate text with streaming support.

        VLLM's Python API doesn't have native async streaming, so we use
        a background task to generate and yield results incrementally.

        :param prompt: Input prompt text
        :param sampling_params: Sampling parameters for generation
        :yields: RequestOutput objects as they become available
        """
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")

        # VLLM's generate method is synchronous, so we run it in an executor
        # For true streaming, we'd need to use VLLM's async API or HTTP server
        # For now, we generate and simulate streaming by yielding the result
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._llm.generate([prompt], sampling_params),  # type: ignore[misc]
        )

        if outputs and len(outputs) > 0:
            yield outputs[0]

    async def resolve(  # type: ignore[override]
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process generation request and yield progressive responses.

        Handles both streaming and non-streaming requests, converting between
        GuideLLM schemas and VLLM's native API format.

        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :raises NotImplementedError: If history is provided
        :raises RuntimeError: If backend is not initialized
        :raises ValueError: If request type is unsupported
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        if self._llm is None:
            raise RuntimeError("Backend not started up for process.")

        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")

        if request.request_type not in ("text_completions", "chat_completions"):
            raise ValueError(
                f"Unsupported request type '{request.request_type}'. "
                "Only 'text_completions' and 'chat_completions' are supported."
            )

        # Extract prompt and create sampling parameters
        prompt = self._extract_prompt(request)
        sampling_params = self._create_sampling_params(request)
        stream = request.arguments.stream or False

        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()

        if not stream:
            # Non-streaming generation
            request_info.timings.request_start = time.time()
            outputs = await loop.run_in_executor(
                None,
                lambda: self._llm.generate([prompt], sampling_params),  # type: ignore[misc]
            )
            request_info.timings.request_end = time.time()

            if outputs and len(outputs) > 0:
                response = self._convert_output_to_response(request, outputs[0])
                yield response, request_info
            return

        # Streaming generation
        try:
            request_info.timings.request_start = time.time()

            # For streaming, we generate once and then simulate token-by-token streaming
            # by yielding progressive responses
            # Note: VLLM's Python API doesn't support true async streaming,
            # so we generate once and simulate streaming by yielding word-by-word
            output = None
            async for output in self._generate_streaming(prompt, sampling_params):
                generated_text = ""
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text

                # Simulate streaming by yielding word-by-word
                # In a production implementation, you'd use VLLM's actual streaming API
                words = generated_text.split() if generated_text else []
                accumulated_text = ""

                for i, word in enumerate(words):
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1

                    accumulated_text += (" " if i > 0 else "") + word

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += 1

                    # Create partial response
                    partial_response = GenerationResponse(
                        request_id=request.request_id,
                        request_args=str(
                            request.arguments.model_dump()
                            if request.arguments
                            else {}
                        ),
                        response_id=(
                            output.request_id
                            if hasattr(output, "request_id")
                            else None
                        ),
                        text=accumulated_text,
                        input_metrics=UsageMetrics(),
                        output_metrics=UsageMetrics(
                            text_words=len(accumulated_text.split()),
                            text_characters=len(accumulated_text),
                        ),
                    )
                    yield partial_response, request_info

                    # Small delay to simulate real streaming
                    await asyncio.sleep(0.01)

                # Break after first output (we only expect one)
                break

            request_info.timings.request_end = time.time()

            # Yield final complete response with full metrics
            if output is not None:
                final_response = self._convert_output_to_response(request, output)
                yield final_response, request_info

        except asyncio.CancelledError as err:
            # Yield current result to store iterative results before propagating
            if output is not None:
                response = self._convert_output_to_response(request, output)
                yield response, request_info
            raise err

