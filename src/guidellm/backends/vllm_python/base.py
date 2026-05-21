"""
Shared base functionality for vLLM backends.

Provides common code for vLLM-based backends including chat template
resolution, multimodal data handling, request formatting, and sampling
parameter creation. This base class is extended by both VLLMPythonBackend
(AsyncLLMEngine) and VLLMOfflineBackend (LLM class).
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import jinja2
from more_itertools import roundrobin
from pydantic import ConfigDict, Field

from guidellm.backends.backend import Backend
from guidellm.extras.vllm import HAS_VLLM, SamplingParams
from guidellm.logger import logger
from guidellm.schemas import GenerationRequest, StandardBaseModel

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

__all__ = ["VLLMBackendBase", "_ResolvedRequest"]


class _ResolvedRequest(StandardBaseModel):
    """
    Fully resolved request: prompt formatted, ready for engine.generate.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(
        description="Fully resolved prompt string (templated, placeholders)"
    )
    multi_modal_data: dict[str, Any] | None = Field(
        default=None,
        description="vLLM multi_modal_data from image/audio/video.",
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream (for VLLMPythonBackend)",
    )


def _check_vllm_available() -> None:
    """Check if vllm is available and raise helpful error if not."""
    if not HAS_VLLM:
        raise ImportError(
            "vllm is not installed. Install vllm to use vllm backends."
        )


def _has_jinja2_markers(s: str) -> bool:
    """Return True if the string contains Jinja2 template syntax ({{, {%, or {#)."""
    return "{{" in s or "{%" in s or "{#" in s


class VLLMBackendBase(Backend):
    """
    Base class for vLLM backends with shared functionality.

    Provides common utilities for chat template resolution, multimodal data
    handling, prompt formatting, and sampling parameter creation. Subclasses
    implement the specific engine initialization and request processing logic.
    """

    def __init__(self, model: str, request_format: str, image_placeholder: str,
                 audio_placeholder: str):
        """
        Initialize base vLLM backend.

        :param model: Model identifier or path
        :param request_format: Request formatting mode
        :param image_placeholder: Placeholder for images in prompts
        :param audio_placeholder: Placeholder for audio in prompts
        """
        _check_vllm_available()
        self._model = model
        self._request_format = request_format
        self._image_placeholder = image_placeholder
        self._audio_placeholder = audio_placeholder
        self._resolved_chat_template: str | None | object = _CHAT_TEMPLATE_UNSET

    @abstractmethod
    def _get_tokenizer(self):
        """
        Get the tokenizer from the backend's engine.

        Must be implemented by subclass.
        """
        ...

    @property
    def _stream_value(self) -> bool:
        """
        Get the streaming mode for this backend.

        Returns True by default (for offline backends).
        VLLMPythonBackend overrides this to return its _stream attribute.
        """
        return True

    def _build_multi_modal_data_from_columns(  # noqa: C901, PLR0912
        self, columns: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Build vLLM multi_modal_data dict from image_column, audio_column.

        video_column is not yet supported (no frame extraction); it is skipped.
        """
        multi_modal_data: dict[str, Any] = {}
        image_items = columns.get("image_column", [])
        audio_items = columns.get("audio_column", [])

        for item in image_items:
            if not item or not isinstance(item, dict):
                continue
            if not HAS_VISION or image_dict_to_pil is None:
                raise ImportError(
                    "Image column support requires guidellm[vision]. "
                    "Install with: pip install 'guidellm[vision]'"
                )
            pil_image = image_dict_to_pil(item)
            if "image" not in multi_modal_data:
                multi_modal_data["image"] = pil_image
            else:
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
                        audio_samples = _decode_audio(audio_bytes)
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
        """Extract text content from message content field."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if text:
                            text_parts.append(text)
            return "".join(text_parts)
        return str(content) if content is not None else ""

    def _build_placeholder_prefix(self, multi_modal_data: dict[str, Any]) -> str:
        """
        Build placeholder prefix string for all modalities.
        """
        parts: list[str] = []
        images = multi_modal_data.get("image")
        if images is not None:
            num = len(images) if isinstance(images, list | tuple) else 1
            if num > 0:
                parts.extend([self._image_placeholder] * num)
        audio = multi_modal_data.get("audio")
        if audio is not None:
            num = len(audio) if isinstance(audio, list | tuple) else 1
            if num > 0:
                parts.extend([self._audio_placeholder] * num)
        if not parts:
            return ""
        return "\n".join(parts) + "\n"

    @staticmethod
    def _format_column_blocks(
        column_data: list[Any], column_type: str
    ) -> list[dict[str, Any]]:
        """Format data column items into vLLM-compatible content blocks."""
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
        """Inject multimodal placeholder tokens into the last user message's content."""
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
        """Concatenate message content into a single raw prompt string."""
        return " ".join(
            msg["content"] for msg in formatted_messages if msg.get("content")
        )

    def _resolve_chat_template(self) -> str | None:
        """Resolve and validate request_format to a template string or None."""
        template = self._request_format
        if template in ("plain", "default-template"):
            return None
        path = Path(template)
        if path.exists() and path.is_file():
            content = path.read_text()
            if not _has_jinja2_markers(content):
                raise ValueError(
                    f"Invalid chat template: path {path.as_posix()!r} exists but "
                    "file content does not contain Jinja2 template syntax."
                )
            try:
                jinja2.Template(content)
            except jinja2.TemplateSyntaxError as e:
                raise ValueError(
                    f"Invalid chat template in file {path.as_posix()!r}: {e}"
                ) from e
            return content
        if _has_jinja2_markers(template):
            try:
                jinja2.Template(template)
            except jinja2.TemplateSyntaxError as e:
                raise ValueError(f"Invalid chat template: {e}") from e
            return template
        raise ValueError(
            "request_format must be 'plain', 'default-template', a path to a "
            "Jinja2 template file, or a string containing Jinja2 template "
            "syntax ({{, {%}, or {#). Got: " + repr(template) + "."
        )

    def _extract_prompt_chat_tokenizer(
        self, formatted_messages: list[dict[str, Any]]
    ) -> str:
        """Apply tokenizer chat template to formatted messages."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError("Backend engine has no tokenizer.")

        if self._request_format in ("plain", "default-template"):
            resolved: str | None = None
        else:
            if self._resolved_chat_template is _CHAT_TEMPLATE_UNSET:
                self._resolved_chat_template = self._resolve_chat_template()
            resolved = self._resolved_chat_template  # type: ignore[assignment]

        if resolved is not None:
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

        :param request: Column-based generation request
        :return: Resolved request with formatted prompt and multimodal data
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

        use_content_blocks = (
            multi_modal_data
            and (text_blocks or prefix)
            and self._request_format != "plain"
        )

        if use_content_blocks:
            media_lists = [
                self._format_column_blocks(columns.get(col, []), col)
                for col in ("image_column", "audio_column")
            ]
            user_content: list[dict[str, Any]] = list(
                roundrobin(text_blocks, *media_lists)
            )
        else:
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
                    self._inject_placeholders_into_messages(
                        formatted_messages, multi_modal_data
                    )

                if self._request_format == "plain":
                    prompt = self._extract_prompt_chat_plain(formatted_messages)
                else:
                    prompt = self._extract_prompt_chat_tokenizer(formatted_messages)
        elif multi_modal_data:
            prompt = self._build_placeholder_prefix(multi_modal_data)
        else:
            raise ValueError("Request must include text_column or multimodal columns.")

        return _ResolvedRequest(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
            stream=self._stream_value,
        )

    def _create_sampling_params(
        self,
        max_tokens_override: int | None = None,
    ) -> SamplingParams:
        """
        Create VLLM SamplingParams.

        :param max_tokens_override: Optional max_tokens from request
        :return: Configured SamplingParams instance
        """
        params: dict[str, Any] = {}

        if max_tokens_override is not None and max_tokens_override > 0:
            params["max_tokens"] = max_tokens_override
            params["ignore_eos"] = True

        return SamplingParams(**params)  # type: ignore[misc]
