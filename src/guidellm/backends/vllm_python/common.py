"""Shared helpers for vLLM Python backends (vllm_python and vllm_offline)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
from loguru import logger

from guidellm.utils import audio, vision

if TYPE_CHECKING:
    from guidellm.extras import vllm

__all__ = [
    "CHAT_TEMPLATE_UNSET",
    "build_multi_modal_data_from_columns",
    "build_placeholder_prefix",
    "create_sampling_params",
    "extract_prompt_chat_plain",
    "extract_prompt_chat_tokenizer",
    "extract_text_from_content",
    "format_column_blocks",
    "inject_placeholders_into_messages",
    "resolve_chat_template",
]

# Sentinel for "chat template not yet resolved" cache.
CHAT_TEMPLATE_UNSET = object()


def _has_jinja2_markers(s: str) -> bool:
    """Check if string contains Jinja2 template markers ({{, {%}, or {#})."""
    return "{{" in s or "{%" in s or "{#" in s


def extract_text_from_content(content: str | list[dict[str, Any]] | Any) -> str:
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


def build_placeholder_prefix(
    multi_modal_data: dict[str, Any],
    image_placeholder: str = "<image>",
    audio_placeholder: str = "<|audio|>",
) -> str:
    """
    Build the placeholder prefix string for all modalities in multi_modal_data.

    Returns a string like ``"<image>\\n<|audio|>\\n"`` with one placeholder per
    item, or ``""`` if no multimodal items are present.

    :param multi_modal_data: Multi-modal data dict with image/audio
    :param image_placeholder: Placeholder token for images
    :param audio_placeholder: Placeholder token for audio
    :return: Newline-joined placeholder string or empty string
    """
    parts: list[str] = []
    images = multi_modal_data.get("image")
    if images is not None:
        num = len(images) if isinstance(images, list | tuple) else 1
        if num > 0:
            parts.extend([image_placeholder] * num)
    audio = multi_modal_data.get("audio")
    if audio is not None:
        # Single audio item (numpy array) — not a list of items.
        num = len(audio) if isinstance(audio, list | tuple) else 1
        if num > 0:
            parts.extend([audio_placeholder] * num)
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def format_column_blocks(
    column_data: list[Any], column_type: str
) -> list[dict[str, Any]]:
    """
    Format data column items into vLLM-compatible content blocks.

    Analogous to the HTTP backend's ``_format_prompts`` but emitting
    vLLM-specific block types that chat templates can render into the
    correct model-specific placeholder tokens.

    :param column_data: List of items from a data column
    :param column_type: Column type (text_column, image_column, audio_column)
    :return: List of typed content block dicts
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


def inject_placeholders_into_messages(
    formatted_messages: list[dict[str, Any]],
    multi_modal_data: dict[str, Any],
    image_placeholder: str = "<image>",
    audio_placeholder: str = "<|audio|>",
) -> None:
    """
    Inject multimodal placeholder tokens into the last user message's content.

    vLLM requires one placeholder per multimodal item in the prompt text so its
    processor can apply prompt replacement. This must happen *before* the chat
    template is applied so that placeholders end up inside the correct message
    turn (not prepended to the entire formatted prompt).

    :param formatted_messages: List of message dicts (modified in-place)
    :param multi_modal_data: Multi-modal data dict
    :param image_placeholder: Placeholder token for images
    :param audio_placeholder: Placeholder token for audio
    """
    prefix = build_placeholder_prefix(
        multi_modal_data, image_placeholder, audio_placeholder
    )
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


def extract_prompt_chat_plain(
    formatted_messages: list[dict[str, Any]],
) -> str:
    """
    Concatenate message content into a single raw prompt string.

    Equivalent to the HTTP /v1/completions behaviour: prefix + text
    with no role prefixes or trailing generation prompt.

    :param formatted_messages: List of message dicts with role/content
    :return: Space-joined content string
    """
    return " ".join(msg["content"] for msg in formatted_messages if msg.get("content"))


def build_multi_modal_data_from_columns(
    columns: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Build vLLM multi_modal_data dict from image_column, audio_column.

    video_column is not yet supported (no frame extraction); it is skipped.

    :param columns: Request columns containing image_column and/or audio_column
    :return: Multi-modal data dict for vLLM, or None if no multi-modal data
    :raises ValueError: If audio decoding fails
    """
    multi_modal_data: dict[str, Any] = {}
    # We look specifically for "image_column" and "audio_column"
    # which contain lists of dicts
    image_items = columns.get("image_column", [])
    audio_items = columns.get("audio_column", [])
    # video_column: not yet supported; would require frame extraction
    for item in image_items:
        if not item or not isinstance(item, dict):
            continue
        # Convert raw image dicts into PIL Images as required by vLLM's vision
        # processor
        pil_image = vision.image_dict_to_pil(item)
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
                try:
                    # Decode raw audio bytes into an array since vLLM audio models
                    # expect either raw numpy arrays or specific tensor formats
                    audio_data = audio.decode_audio(audio_bytes)
                    multi_modal_data["audio"] = audio_data
                except (ValueError, TypeError, OSError, RuntimeError) as exc:
                    raise ValueError(
                        f"Failed to decode audio from audio_column for vLLM: {exc}"
                    ) from exc
    return multi_modal_data if multi_modal_data else None


def resolve_chat_template(request_format: str) -> str | None:
    """
    Resolve and validate request_format to a template string or None.

    Returns None for default tokenizer template; returns the template string
    when valid. Raises ValueError for invalid input (wrong format, bad path,
    or invalid Jinja2 syntax).

    :param request_format: Template format string
        (plain, default-template, path, or Jinja2)
    :return: Template string or None for default
    :raises ValueError: If request_format is invalid
    """
    template = request_format
    if template in (
        "plain",
        "default-template",
    ):
        # No custom template provided; 'plain' and 'default-template' are handled
        # internally
        return None
    path = Path(template)
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


def extract_prompt_chat_tokenizer(
    formatted_messages: list[dict[str, Any]],
    tokenizer: Any,
    request_format: str,
    resolved_chat_template: str | None,
) -> str:
    """
    Apply tokenizer chat template to formatted messages.

    :param formatted_messages: List of message dicts with role/content
    :param tokenizer: Tokenizer instance from vLLM engine
    :param request_format: Request format ('plain', 'default-template', or custom)
    :param resolved_chat_template: Pre-resolved custom template or None for default
    :return: Formatted prompt string
    :raises RuntimeError: If tokenizer is missing or returns unexpected type
    """
    if tokenizer is None:
        raise RuntimeError("Backend engine has no tokenizer.")

    if request_format in (
        "plain",
        "default-template",
    ):
        resolved: str | None = None
    else:
        resolved = resolved_chat_template
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


def create_sampling_params(
    vllm_module: Any,
    max_tokens_override: int | None = None,
) -> vllm.SamplingParams:
    """
    Create VLLM SamplingParams.

    When max_tokens_override is set (from benchmark output_metrics), it is used
    as max_tokens and EOS is ignored to force generation of exactly that many
    tokens, matching HTTP backend behavior. Otherwise vLLM defaults are used
    (generate until EOS or model max context).

    :param vllm_module: vLLM module (from guidellm.extras)
    :param max_tokens_override: Optional max_tokens from request (e.g. benchmark)
    :return: Configured SamplingParams instance
    """
    params: dict[str, Any] = {}

    if max_tokens_override is not None and max_tokens_override > 0:
        params["max_tokens"] = max_tokens_override
        params["ignore_eos"] = True

    return vllm_module.SamplingParams(**params)
