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
    "create_sampling_params",
    "extract_prompt_chat_tokenizer",
    "resolve_chat_template",
]

# Sentinel for "chat template not yet resolved" cache.
CHAT_TEMPLATE_UNSET = object()


def _has_jinja2_markers(s: str) -> bool:
    """Check if string contains Jinja2 template markers ({{, {%}, or {#})."""
    return "{{" in s or "{%" in s or "{#" in s


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
