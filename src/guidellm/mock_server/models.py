"""
Pydantic models for OpenAI API and vLLM API request/response validation.

This module defines comprehensive data models for validating and serializing API
requests and responses compatible with both OpenAI's API specification and vLLM's
extended parameters. It includes models for chat completions, legacy text completions,
tokenization operations, and error handling, supporting both streaming and non-streaming
responses with full type safety and validation.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionsRequest",
    "ChatCompletionsResponse",
    "ChatMessage",
    "CompletionChoice",
    "CompletionsRequest",
    "CompletionsResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    "ErrorDetail",
    "ErrorResponse",
    "StreamOptions",
    "TokenizeRequest",
    "TokenizeResponse",
    "Usage",
]


class Usage(BaseModel):
    """Token usage statistics for API requests and responses.

    Tracks the number of tokens consumed in prompts, completions, and total
    usage for billing and monitoring purposes.
    """

    prompt_tokens: int = Field(description="Number of tokens in the input prompt")
    completion_tokens: int = Field(
        description="Number of tokens in the generated completion"
    )
    total_tokens: int = Field(description="Total tokens used (prompt + completion)")

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, **kwargs):
        """Initialize usage statistics.

        :param prompt_tokens: Number of tokens in the input prompt
        :param completion_tokens: Number of tokens in the generated completion
        :param kwargs: Additional keyword arguments passed to BaseModel
        """
        super().__init__(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            **kwargs,
        )


class StreamOptions(BaseModel):
    """Configuration options for streaming API responses.

    Controls the behavior and content of streamed responses including
    whether to include usage statistics in the final chunk.
    """

    include_usage: bool | None = Field(
        default=None,
        description="Whether to include usage statistics in streaming responses",
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Represents one exchange in a conversational interface with role-based
    content and optional metadata for advanced features.
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Role of the message sender in the conversation"
    )
    content: str = Field(description="Text content of the message")
    name: str | None = Field(
        default=None, description="Optional name identifier for the message sender"
    )


class ChatCompletionsRequest(BaseModel):
    """Request parameters for chat completion API endpoints.

    Comprehensive model supporting both OpenAI standard parameters and vLLM
    extensions for advanced generation control, guided decoding, and performance
    optimization.
    """

    model: str = Field(description="Model identifier to use for generation")
    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    max_completion_tokens: int | None = Field(
        default=None, description="Maximum tokens in completion (OpenAI naming)"
    )
    temperature: float | None = Field(
        default=1.0, description="Sampling temperature for randomness control"
    )
    top_p: float | None = Field(default=1.0, description="Nucleus sampling parameter")
    n: int | None = Field(
        default=1, description="Number of completion choices to generate"
    )
    stream: bool | None = Field(
        default=False, description="Whether to stream response chunks"
    )
    stream_options: StreamOptions | None = Field(
        default=None, description="Configuration for streaming responses"
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences to end generation"
    )
    presence_penalty: float | None = Field(
        default=0.0, description="Penalty for token presence to encourage diversity"
    )
    frequency_penalty: float | None = Field(
        default=0.0, description="Penalty for token frequency to reduce repetition"
    )
    logit_bias: dict[str, float] | None = Field(
        default=None, description="Bias values for specific tokens"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducible outputs"
    )
    user: str | None = Field(
        default=None, description="User identifier for tracking and abuse monitoring"
    )

    # vLLM extensions
    use_beam_search: bool | None = Field(
        default=False, description="Enable beam search for better quality"
    )
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    min_p: float | None = Field(
        default=None, description="Minimum probability threshold for sampling"
    )
    repetition_penalty: float | None = Field(
        default=None, description="Penalty for repeated tokens"
    )
    length_penalty: float | None = Field(
        default=1.0, description="Length penalty for sequence scoring"
    )
    stop_token_ids: list[int] | None = Field(
        default=None, description="Token IDs that trigger generation stop"
    )
    include_stop_str_in_output: bool | None = Field(
        default=False, description="Include stop sequence in output"
    )
    ignore_eos: bool | None = Field(
        default=False, description="Ignore end-of-sequence tokens"
    )
    min_tokens: int | None = Field(
        default=0, description="Minimum number of tokens to generate"
    )
    skip_special_tokens: bool | None = Field(
        default=True, description="Skip special tokens in output"
    )
    spaces_between_special_tokens: bool | None = Field(
        default=True, description="Add spaces between special tokens"
    )
    truncate_prompt_tokens: int | None = Field(
        default=None, description="Maximum prompt tokens before truncation"
    )
    allowed_token_ids: list[int] | None = Field(
        default=None, description="Restrict generation to specific token IDs"
    )
    prompt_logprobs: int | None = Field(
        default=None, description="Number of logprobs to return for prompt tokens"
    )
    add_special_tokens: bool | None = Field(
        default=True, description="Add special tokens during processing"
    )
    guided_json: str | dict[str, Any] | None = Field(
        default=None, description="JSON schema for guided generation"
    )
    guided_regex: str | None = Field(
        default=None, description="Regex pattern for guided generation"
    )
    guided_choice: list[str] | None = Field(
        default=None, description="List of choices for guided generation"
    )
    guided_grammar: str | None = Field(
        default=None, description="Grammar specification for guided generation"
    )
    guided_decoding_backend: str | None = Field(
        default=None, description="Backend to use for guided decoding"
    )
    guided_whitespace_pattern: str | None = Field(
        default=None, description="Whitespace pattern for guided generation"
    )
    priority: int | None = Field(
        default=0, description="Request priority for scheduling"
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice from a chat completion response.

    Contains the generated message and metadata about why generation
    stopped and the choice's position in the response.
    """

    index: int = Field(description="Index of this choice in the response")
    message: ChatMessage = Field(description="Generated message content")
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None = (
        Field(description="Reason why generation finished")
    )


class ChatCompletionsResponse(BaseModel):
    """Response from chat completion API endpoints.

    Contains generated choices, usage statistics, and metadata for
    non-streaming chat completion requests.
    """

    id: str = Field(description="Unique identifier for this completion")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="Object type identifier"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for generation")
    choices: list[ChatCompletionChoice] = Field(
        description="Generated completion choices"
    )
    usage: Usage | None = Field(default=None, description="Token usage statistics")
    system_fingerprint: str | None = Field(
        default=None, description="System configuration fingerprint"
    )


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streamed chat completion response.

    Represents one piece of a streaming response with delta content
    and optional usage statistics in the final chunk.
    """

    id: str = Field(description="Unique identifier for this completion")
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk",
        description="Object type identifier for streaming chunks",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for generation")
    choices: list[dict[str, Any]] = Field(description="Delta choices for streaming")
    usage: Usage | None = Field(
        default=None, description="Token usage statistics (typically in final chunk)"
    )


class CompletionsRequest(BaseModel):
    """Request parameters for legacy text completion API endpoints.

    Supports the older text completion format with prompt-based input
    and the same extensive parameter set as chat completions for
    backward compatibility.
    """

    model: str = Field(description="Model identifier to use for generation")
    prompt: str | list[str] = Field(description="Input prompt(s) for completion")
    max_tokens: int | None = Field(
        default=16, description="Maximum number of tokens to generate"
    )
    temperature: float | None = Field(
        default=1.0, description="Sampling temperature for randomness control"
    )
    top_p: float | None = Field(default=1.0, description="Nucleus sampling parameter")
    n: int | None = Field(
        default=1, description="Number of completion choices to generate"
    )
    stream: bool | None = Field(
        default=False, description="Whether to stream response chunks"
    )
    stream_options: StreamOptions | None = Field(
        default=None, description="Configuration for streaming responses"
    )
    logprobs: int | None = Field(
        default=None, description="Number of logprobs to return"
    )
    echo: bool | None = Field(
        default=False, description="Whether to echo the prompt in output"
    )
    stop: str | list[str] | None = Field(
        default_factory=lambda: ["<|endoftext|>"],
        description="Stop sequences to end generation",
    )
    presence_penalty: float | None = Field(
        default=0.0, description="Penalty for token presence to encourage diversity"
    )
    frequency_penalty: float | None = Field(
        default=0.0, description="Penalty for token frequency to reduce repetition"
    )
    best_of: int | None = Field(
        default=1, description="Number of candidates to generate and return the best"
    )
    logit_bias: dict[str, float] | None = Field(
        default=None, description="Bias values for specific tokens"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducible outputs"
    )
    suffix: str | None = Field(
        default=None, description="Suffix to append after completion"
    )
    user: str | None = Field(
        default=None, description="User identifier for tracking and abuse monitoring"
    )

    # vLLM extensions (same as chat completions)
    use_beam_search: bool | None = Field(
        default=False, description="Enable beam search for better quality"
    )
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    min_p: float | None = Field(
        default=None, description="Minimum probability threshold for sampling"
    )
    repetition_penalty: float | None = Field(
        default=None, description="Penalty for repeated tokens"
    )
    length_penalty: float | None = Field(
        default=1.0, description="Length penalty for sequence scoring"
    )
    stop_token_ids: list[int] | None = Field(
        default=None, description="Token IDs that trigger generation stop"
    )
    include_stop_str_in_output: bool | None = Field(
        default=False, description="Include stop sequence in output"
    )
    ignore_eos: bool | None = Field(
        default=False, description="Ignore end-of-sequence tokens"
    )
    min_tokens: int | None = Field(
        default=0, description="Minimum number of tokens to generate"
    )
    skip_special_tokens: bool | None = Field(
        default=True, description="Skip special tokens in output"
    )
    spaces_between_special_tokens: bool | None = Field(
        default=True, description="Add spaces between special tokens"
    )
    truncate_prompt_tokens: int | None = Field(
        default=None, description="Maximum prompt tokens before truncation"
    )
    allowed_token_ids: list[int] | None = Field(
        default=None, description="Restrict generation to specific token IDs"
    )
    prompt_logprobs: int | None = Field(
        default=None, description="Number of logprobs to return for prompt tokens"
    )
    add_special_tokens: bool | None = Field(
        default=True, description="Add special tokens during processing"
    )
    guided_json: str | dict[str, Any] | None = Field(
        default=None, description="JSON schema for guided generation"
    )
    guided_regex: str | None = Field(
        default=None, description="Regex pattern for guided generation"
    )
    guided_choice: list[str] | None = Field(
        default=None, description="List of choices for guided generation"
    )
    guided_grammar: str | None = Field(
        default=None, description="Grammar specification for guided generation"
    )
    guided_decoding_backend: str | None = Field(
        default=None, description="Backend to use for guided decoding"
    )
    guided_whitespace_pattern: str | None = Field(
        default=None, description="Whitespace pattern for guided generation"
    )
    priority: int | None = Field(
        default=0, description="Request priority for scheduling"
    )


class CompletionChoice(BaseModel):
    """A single completion choice from a text completion response.

    Contains the generated text and metadata about completion
    quality and stopping conditions.
    """

    text: str = Field(description="Generated text content")
    index: int = Field(description="Index of this choice in the response")
    logprobs: dict[str, Any] | None = Field(
        default=None, description="Log probabilities for generated tokens"
    )
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        description="Reason why generation finished"
    )


class CompletionsResponse(BaseModel):
    """Response from legacy text completion API endpoints.

    Contains generated text choices, usage statistics, and metadata
    for non-streaming text completion requests.
    """

    id: str = Field(description="Unique identifier for this completion")
    object: Literal["text_completion"] = Field(
        default="text_completion", description="Object type identifier"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for generation")
    choices: list[CompletionChoice] = Field(description="Generated completion choices")
    usage: Usage | None = Field(default=None, description="Token usage statistics")
    system_fingerprint: str | None = Field(
        default=None, description="System configuration fingerprint"
    )


class TokenizeRequest(BaseModel):
    """Request for tokenizing text into token sequences.

    Converts input text into model-specific token representations
    with optional special token handling.
    """

    text: str = Field(description="Text to tokenize")
    add_special_tokens: bool | None = Field(
        default=True, description="Whether to add model-specific special tokens"
    )


class TokenizeResponse(BaseModel):
    """Response containing tokenized representation of input text.

    Provides both the token sequence and count for analysis
    and token budget planning.
    """

    tokens: list[int] = Field(description="List of token IDs")
    count: int = Field(description="Total number of tokens")


class DetokenizeRequest(BaseModel):
    """Request for converting token sequences back to text.

    Reconstructs human-readable text from model token representations
    with configurable special token handling.
    """

    tokens: list[int] = Field(description="List of token IDs to convert")
    skip_special_tokens: bool | None = Field(
        default=True, description="Whether to skip special tokens in output"
    )
    spaces_between_special_tokens: bool | None = Field(
        default=True, description="Whether to add spaces between special tokens"
    )


class DetokenizeResponse(BaseModel):
    """Response containing text reconstructed from tokens.

    Provides the human-readable text representation of the
    input token sequence.
    """

    text: str = Field(description="Reconstructed text from tokens")


class ErrorDetail(BaseModel):
    """Detailed error information for API failures.

    Provides structured error data including message, type classification,
    and optional error codes for debugging and error handling.
    """

    message: str = Field(description="Human-readable error description")
    type: str = Field(description="Error type classification")
    code: str | None = Field(
        default=None, description="Optional error code for programmatic handling"
    )


class ErrorResponse(BaseModel):
    """Standardized error response structure for API failures.

    Wraps error details in a consistent format compatible with
    OpenAI API error response conventions.
    """

    error: ErrorDetail = Field(description="Detailed error information")
