"""
Unit tests for VLLM Python backend.

Tests _resolve_request (prompt from columns, multimodal data, roundrobin
content blocks, placeholders), request_format (plain, default-template,
custom template), sampling params, usage extraction, lifecycle, and
resolve() integration.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from guidellm.backends.vllm_python.vllm import (
    VLLMPythonBackend,
    _has_jinja2_markers,
    _ResolvedRequest,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    UsageMetrics,
)


def _fake_sampling_params(**kwargs):
    """
    Fake SamplingParams for tests when vLLM is not installed.
    Returns a SimpleNamespace so only explicitly passed kwargs become attributes.
    """
    return SimpleNamespace(**kwargs)


def _mock_audio_decode_result(audio_array: np.ndarray) -> Mock:
    """
    Build a mock torchcodec AudioSamples whose .data behaves like a CPU
    torch.Tensor: .data.cpu() returns self, .data.cpu().numpy() returns
    the given numpy array.
    """
    mock_data = Mock()
    mock_data.cpu.return_value = mock_data
    mock_data.numpy.return_value = audio_array
    result = Mock()
    result.data = mock_data
    return result


@pytest.fixture
def backend():
    """VLLMPythonBackend instance without requiring vllm to be installed."""
    with (
        patch("guidellm.backends.vllm_python.vllm._check_vllm_available"),
        patch(
            "guidellm.backends.vllm_python.vllm.SamplingParams",
            _fake_sampling_params,
        ),
    ):
        yield VLLMPythonBackend(model="test-model")


class TestResolveRequest:
    """
    Test _resolve_request: prompt resolution from columns.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_text_column_resolves_to_prompt(self, backend):
        """
        Request with text_column resolves to a prompt string via plain format.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
        request = GenerationRequest(columns={"text_column": ["hello"]})
        resolved = backend_plain._resolve_request(request)
        assert isinstance(resolved, _ResolvedRequest)
        assert resolved.prompt == "hello"
        assert resolved.stream is True
        assert resolved.multi_modal_data is None

    @pytest.mark.sanity
    def test_stream_false_propagated(self):
        """
        When backend.stream=False, resolved.stream is False.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(
                model="test-model", stream=False, request_format="plain"
            )
        request = GenerationRequest(columns={"text_column": ["hello"]})
        resolved = backend._resolve_request(request)
        assert resolved.stream is False

    @pytest.mark.sanity
    def test_prefix_and_text_columns_build_messages(self):
        """
        Columns with prefix_column and text_column are formatted into prompt.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model", request_format="plain")
        request = GenerationRequest(
            columns={
                "prefix_column": ["System prompt"],
                "text_column": ["User question"],
            }
        )
        resolved = backend._resolve_request(request)
        assert resolved.prompt == "System prompt User question"

    @pytest.mark.sanity
    def test_text_only_no_media_multi_modal_data_none(self, backend):
        """
        Request with only text columns leaves multi_modal_data None.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
        request = GenerationRequest(columns={"text_column": ["hello"]})
        resolved = backend_plain._resolve_request(request)
        assert resolved.multi_modal_data is None

    @pytest.mark.smoke
    def test_audio_column_only_resolves_with_placeholder_prompt(self, backend):
        """
        Request with only audio_column resolves prompt to audio placeholder.
        ## WRITTEN BY AI ##
        """
        mock_audio_array = np.array([0.0, 0.1], dtype=np.float32)
        mock_decode_result = _mock_audio_decode_result(mock_audio_array)

        request = GenerationRequest(
            columns={
                "audio_column": [{"audio": b"fake-wav-bytes", "format": "wav"}],
            }
        )
        with patch(
            "guidellm.backends.vllm_python.vllm._decode_audio",
            return_value=mock_decode_result,
        ):
            resolved = backend._resolve_request(request)
        assert resolved.multi_modal_data is not None
        assert "audio" in resolved.multi_modal_data
        np.testing.assert_array_equal(
            resolved.multi_modal_data["audio"], mock_audio_array
        )
        assert "<|audio|>" in resolved.prompt

    @pytest.mark.smoke
    def test_image_column_resolves_with_multi_modal_data(self):
        """
        Request with image_column sets multi_modal_data and injects placeholder.
        ## WRITTEN BY AI ##
        """
        mock_pil = Mock()
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model", request_format="plain")
        request = GenerationRequest(
            columns={
                "text_column": ["Describe this"],
                "image_column": [
                    {"image": "data:image/jpeg;base64,/9j/4AAQ="},
                ],
            }
        )
        with patch(
            "guidellm.backends.vllm_python.vllm.image_dict_to_pil",
            return_value=mock_pil,
        ):
            resolved = backend._resolve_request(request)
        assert resolved.multi_modal_data is not None
        assert "image" in resolved.multi_modal_data
        assert resolved.multi_modal_data["image"] is mock_pil
        assert "<image>" in resolved.prompt
        assert "Describe this" in resolved.prompt

    @pytest.mark.sanity
    def test_empty_columns_raises(self, backend):
        """
        Request with no text or multimodal columns raises ValueError.
        ## WRITTEN BY AI ##
        """
        request = GenerationRequest(columns={})
        with pytest.raises(ValueError, match="text_column or multimodal"):
            backend._resolve_request(request)

    @pytest.mark.sanity
    def test_audio_and_text_with_chat_template_uses_content_blocks(self):
        """
        With text + audio + chat template, apply_chat_template receives messages
        with {"type": "audio"} content blocks (not a hardcoded placeholder string).
        ## WRITTEN BY AI ##
        """
        mock_audio_array = np.array([0.0, 0.1], dtype=np.float32)
        mock_decode_result = _mock_audio_decode_result(mock_audio_array)

        captured_messages = []

        def fake_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ):
            captured_messages.append(messages)
            return "<|user|>\n<|begin_of_audio|><|end_of_audio|>\nHello\n"

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = fake_apply_chat_template

        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(
                model="test-model", request_format="default-template"
            )
        backend._engine = Mock()
        backend._engine.tokenizer = mock_tokenizer

        request = GenerationRequest(
            columns={
                "text_column": ["Hello"],
                "audio_column": [{"audio": b"fake-wav-bytes", "format": "wav"}],
            }
        )
        with patch(
            "guidellm.backends.vllm_python.vllm._decode_audio",
            return_value=mock_decode_result,
        ):
            resolved = backend._resolve_request(request)

        assert len(captured_messages) == 1
        msgs = captured_messages[0]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        types = [b["type"] for b in user_msg["content"]]
        assert "audio" in types
        assert "text" in types
        assert resolved.multi_modal_data is not None
        assert "audio" in resolved.multi_modal_data

    @pytest.mark.sanity
    def test_audio_and_text_plain_format_uses_placeholder_string(self):
        """
        With text + audio + plain format, placeholder strings are used (not
        content blocks), preserving the existing plain-mode behavior.
        ## WRITTEN BY AI ##
        """
        mock_audio_array = np.array([0.0, 0.1], dtype=np.float32)
        mock_decode_result = _mock_audio_decode_result(mock_audio_array)

        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model", request_format="plain")

        request = GenerationRequest(
            columns={
                "text_column": ["Hello"],
                "audio_column": [{"audio": b"fake-wav-bytes", "format": "wav"}],
            }
        )
        with patch(
            "guidellm.backends.vllm_python.vllm._decode_audio",
            return_value=mock_decode_result,
        ):
            resolved = backend._resolve_request(request)

        assert "<|audio|>" in resolved.prompt
        assert "Hello" in resolved.prompt
        assert resolved.multi_modal_data is not None


class TestImagePlaceholderInjection:
    """
    Test image placeholder defaults, overrides, and message-level injection.
    """

    @pytest.mark.smoke
    def test_build_placeholder_prefix_default_image(self, backend):
        """
        _build_placeholder_prefix uses default '<image>' when no override.
        ## WRITTEN BY AI ##
        """
        result = backend._build_placeholder_prefix({"image": Mock()})
        assert result == "<image>\n"

    @pytest.mark.sanity
    def test_build_placeholder_prefix_image_override(self):
        """
        _build_placeholder_prefix uses image_placeholder override.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="Qwen/Qwen3-VL-2B-Instruct",
                image_placeholder=("<|vision_start|><|image_pad|><|vision_end|>"),
            )
        result = backend_custom._build_placeholder_prefix({"image": Mock()})
        assert result == ("<|vision_start|><|image_pad|><|vision_end|>\n")

    @pytest.mark.sanity
    def test_inject_placeholders_into_messages_no_media_unchanged(self, backend):
        """
        _inject_placeholders_into_messages leaves messages unchanged when no
        recognized multimodal keys (image/audio) are present.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Hello"}]
        backend._inject_placeholders_into_messages(msgs, {})
        assert msgs[0]["content"] == "Hello"
        backend._inject_placeholders_into_messages(msgs, {"video": "data"})
        assert msgs[0]["content"] == "Hello"

    @pytest.mark.smoke
    def test_inject_placeholders_into_messages_single_image(self, backend):
        """
        _inject_placeholders_into_messages prepends one image placeholder into
        the last user message content.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "What is this?"}]
        backend._inject_placeholders_into_messages(msgs, {"image": Mock()})
        assert msgs[0]["content"] == "<image>\nWhat is this?"

    @pytest.mark.sanity
    def test_inject_placeholders_into_messages_multiple_images(self, backend):
        """
        _inject_placeholders_into_messages prepends N image placeholders for N images.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Describe both."}]
        backend._inject_placeholders_into_messages(msgs, {"image": [Mock(), Mock()]})
        assert msgs[0]["content"] == "<image>\n<image>\nDescribe both."

    @pytest.mark.sanity
    def test_inject_placeholders_targets_last_user_message(self, backend):
        """
        _inject_placeholders_into_messages injects into the last user message,
        not the system message.
        ## WRITTEN BY AI ##
        """
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Describe this image"},
        ]
        backend._inject_placeholders_into_messages(msgs, {"image": Mock()})
        assert msgs[0]["content"] == "You are a helper."
        assert msgs[1]["content"] == "First question"
        assert msgs[3]["content"] == "<image>\nDescribe this image"


class TestAudioPlaceholderInjection:
    """
    Test audio placeholder defaults, overrides, and message-level injection.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_build_placeholder_prefix_default_audio(self, backend):
        """
        _build_placeholder_prefix uses default '<|audio|>' when no override.
        ## WRITTEN BY AI ##
        """
        result = backend._build_placeholder_prefix(
            {"audio": np.array([0.0], dtype=np.float32)}
        )
        assert result == "<|audio|>\n"

    @pytest.mark.sanity
    def test_build_placeholder_prefix_audio_override(self):
        """
        _build_placeholder_prefix uses audio_placeholder override.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="zai-org/GLM-ASR-Nano-2512",
                audio_placeholder=("<|begin_of_audio|><|pad|><|end_of_audio|>"),
            )
        result = backend_custom._build_placeholder_prefix(
            {"audio": np.array([0.0], dtype=np.float32)}
        )
        assert result == ("<|begin_of_audio|><|pad|><|end_of_audio|>\n")

    @pytest.mark.sanity
    def test_inject_placeholders_into_messages_no_audio_unchanged(self, backend):
        """
        _inject_placeholders_into_messages leaves messages unchanged when no audio.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Hello"}]
        backend._inject_placeholders_into_messages(msgs, {"image": Mock()})
        assert "<|audio|>" not in msgs[0]["content"]

    @pytest.mark.smoke
    def test_inject_placeholders_into_messages_single_audio(self, backend):
        """
        _inject_placeholders_into_messages prepends one audio placeholder into
        the last user message content.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Transcribe."}]
        multi_modal_data = {"audio": np.array([0.0], dtype=np.float32)}
        backend._inject_placeholders_into_messages(msgs, multi_modal_data)
        assert msgs[0]["content"] == "<|audio|>\nTranscribe."

    @pytest.mark.sanity
    def test_inject_placeholders_into_messages_multiple_audio(self, backend):
        """
        _inject_placeholders_into_messages prepends N audio placeholders for N audio.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Compare the two clips."}]
        multi_modal_data = {
            "audio": [
                np.array([0.0], dtype=np.float32),
                np.array([0.1], dtype=np.float32),
            ]
        }
        backend._inject_placeholders_into_messages(msgs, multi_modal_data)
        assert msgs[0]["content"] == ("<|audio|>\n<|audio|>\nCompare the two clips.")

    @pytest.mark.sanity
    def test_inject_placeholders_image_and_audio_combined(self, backend):
        """
        _inject_placeholders_into_messages handles both image and audio together.
        ## WRITTEN BY AI ##
        """
        msgs = [{"role": "user", "content": "Describe what you see and hear."}]
        multi_modal_data = {
            "image": Mock(),
            "audio": np.array([0.0], dtype=np.float32),
        }
        backend._inject_placeholders_into_messages(msgs, multi_modal_data)
        assert msgs[0]["content"] == (
            "<image>\n<|audio|>\nDescribe what you see and hear."
        )


class TestBuildPlaceholderPrefix:
    """
    Test _build_placeholder_prefix for prompt prepending.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_no_multimodal_data_returns_empty(self, backend):
        """
        _build_placeholder_prefix returns '' when no image or audio.
        ## WRITTEN BY AI ##
        """
        assert backend._build_placeholder_prefix({}) == ""

    @pytest.mark.smoke
    def test_single_image_returns_prefix(self, backend):
        """
        _build_placeholder_prefix returns '<image>\\n' for a single image.
        ## WRITTEN BY AI ##
        """
        assert backend._build_placeholder_prefix({"image": Mock()}) == "<image>\n"

    @pytest.mark.sanity
    def test_multiple_images_returns_prefixes(self, backend):
        """
        _build_placeholder_prefix returns N image placeholders for N images.
        ## WRITTEN BY AI ##
        """
        result = backend._build_placeholder_prefix({"image": [Mock(), Mock()]})
        assert result == "<image>\n<image>\n"

    @pytest.mark.smoke
    def test_single_audio_returns_prefix(self, backend):
        """
        _build_placeholder_prefix returns '<|audio|>\\n' for a single audio.
        ## WRITTEN BY AI ##
        """
        result = backend._build_placeholder_prefix(
            {"audio": np.array([0.0], dtype=np.float32)}
        )
        assert result == "<|audio|>\n"

    @pytest.mark.sanity
    def test_image_and_audio_combined(self, backend):
        """
        _build_placeholder_prefix returns both image and audio placeholders.
        ## WRITTEN BY AI ##
        """
        result = backend._build_placeholder_prefix(
            {"image": Mock(), "audio": np.array([0.0], dtype=np.float32)}
        )
        assert result == "<image>\n<|audio|>\n"


class TestHasJinja2Markers:
    """
    Test _has_jinja2_markers helper for template format detection.
    """

    @pytest.mark.smoke
    def test_has_jinja2_markers_true_for_expressions(self):
        """
        _has_jinja2_markers returns True for strings containing {{.
        ## WRITTEN BY AI ##
        """
        assert _has_jinja2_markers("{{ message.content }}") is True
        assert _has_jinja2_markers("prefix {{ x }}") is True

    @pytest.mark.sanity
    def test_has_jinja2_markers_true_for_control(self):
        """
        _has_jinja2_markers returns True for {% and {#.
        ## WRITTEN BY AI ##
        """
        assert _has_jinja2_markers("{% for m in messages %}") is True
        assert _has_jinja2_markers("{# comment #}") is True

    @pytest.mark.sanity
    def test_has_jinja2_markers_false_for_plain_strings(self):
        """
        _has_jinja2_markers returns False for strings with no template syntax.
        ## WRITTEN BY AI ##
        """
        assert _has_jinja2_markers("chat_completions") is False
        assert _has_jinja2_markers("plain text") is False
        assert _has_jinja2_markers("") is False


class TestVLLMRequestFormat:
    """
    Test request_format: plain, default-template, and custom template.
    """

    @pytest.mark.smoke
    def test_request_format_plain_produces_concatenated_prompt(self):
        """
        With request_format=plain, _resolve_request produces plain concatenation.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
        request = GenerationRequest(
            columns={
                "text_column": ["Hello"],
            }
        )
        resolved = backend_plain._resolve_request(request)
        assert resolved.prompt == "Hello"
        assert "User:" not in resolved.prompt
        assert "Assistant:" not in resolved.prompt

    @pytest.mark.sanity
    def test_request_format_chat_completions_raises_not_a_template(self):
        """
        request_format with no Jinja2 markers (e.g. 'chat_completions') raises
        ValueError with message that includes received value and allowed options.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_api = VLLMPythonBackend(
                model="test-model", request_format="chat_completions"
            )
            backend_api._engine = Mock()
            backend_api._engine.tokenizer = Mock()
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        with pytest.raises(ValueError) as exc_info:
            backend_api._resolve_request(request)
        msg = str(exc_info.value)
        assert "chat_completions" in msg
        assert "plain" in msg or "default-template" in msg
        assert "Jinja2" in msg or "template" in msg.lower()

    @pytest.mark.smoke
    def test_request_format_default_template_uses_apply_chat_template(self):
        """
        With request_format=default-template, apply_chat_template is used.
        ## WRITTEN BY AI ##
        """
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_default = VLLMPythonBackend(
                model="test-model", request_format="default-template"
            )
            backend_default._engine = Mock()
            backend_default._engine.tokenizer = mock_tokenizer
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        resolved = backend_default._resolve_request(request)
        assert resolved.prompt == "formatted_prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_kw = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("tokenize") is False
        assert call_kw.get("add_generation_prompt") is True

    @pytest.mark.smoke
    def test_request_format_none_uses_apply_chat_template(self):
        """
        With request_format=None, tokenizer.apply_chat_template is used.
        ## WRITTEN BY AI ##
        """
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "default_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_none = VLLMPythonBackend(model="test-model")
            backend_none._engine = Mock()
            backend_none._engine.tokenizer = mock_tokenizer
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        resolved = backend_none._resolve_request(request)
        assert resolved.prompt == "default_prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.sanity
    def test_request_format_custom_template_string_sets_tokenizer_and_applies(self):
        """
        With request_format=custom template, chat_template is set then applied.
        ## WRITTEN BY AI ##
        """
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "custom_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="test-model",
                request_format="{{ messages[0]['content'] }}",
            )
            backend_custom._engine = Mock()
            backend_custom._engine.tokenizer = mock_tokenizer
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        resolved = backend_custom._resolve_request(request)
        assert resolved.prompt == "custom_prompt"
        assert mock_tokenizer.chat_template == "{{ messages[0]['content'] }}"
        mock_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.sanity
    def test_request_format_custom_template_from_file(self, tmp_path):
        """
        With request_format=file path, chat_template is set from file then applied.
        ## WRITTEN BY AI ##
        """
        template_file = tmp_path / "template.jinja"
        template_file.write_text("Custom: {{ messages[0]['content'] }}")
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "Custom: Hi"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_file = VLLMPythonBackend(
                model="test-model", request_format=str(template_file)
            )
            backend_file._engine = Mock()
            backend_file._engine.tokenizer = mock_tokenizer
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        resolved = backend_file._resolve_request(request)
        assert resolved.prompt == "Custom: Hi"
        assert mock_tokenizer.chat_template == "Custom: {{ messages[0]['content'] }}"

    @pytest.mark.sanity
    def test_request_format_file_template_cached_on_second_request(self, tmp_path):
        """
        With request_format=file path, the second request uses cached content.
        ## WRITTEN BY AI ##
        """
        template_file = tmp_path / "template.jinja"
        template_file.write_text(
            "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "Hi"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_file = VLLMPythonBackend(
                model="test-model", request_format=str(template_file)
            )
            backend_file._engine = Mock()
            backend_file._engine.tokenizer = mock_tokenizer
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        backend_file._resolve_request(request)
        first_template = mock_tokenizer.chat_template
        backend_file._resolve_request(request)
        assert mock_tokenizer.chat_template == first_template
        assert mock_tokenizer.apply_chat_template.call_count == 2

    @pytest.mark.sanity
    def test_request_format_file_with_no_markers_raises(self, tmp_path):
        """
        request_format=path to file with no Jinja2 markers raises ValueError.
        ## WRITTEN BY AI ##
        """
        no_markers_file = tmp_path / "plain.txt"
        no_markers_file.write_text("just plain text")
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_file = VLLMPythonBackend(
                model="test-model", request_format=str(no_markers_file)
            )
            backend_file._engine = Mock()
            backend_file._engine.tokenizer = Mock()
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        with pytest.raises(ValueError) as exc_info:
            backend_file._resolve_request(request)
        msg = str(exc_info.value)
        assert "Jinja2" in msg or "template" in msg.lower()

    @pytest.mark.sanity
    def test_request_format_invalid_jinja2_string_raises(self):
        """
        request_format with invalid Jinja2 syntax raises ValueError.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_bad = VLLMPythonBackend(
                model="test-model", request_format="{{ unclosed"
            )
            backend_bad._engine = Mock()
            backend_bad._engine.tokenizer = Mock()
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        with pytest.raises(ValueError) as exc_info:
            backend_bad._resolve_request(request)
        msg = str(exc_info.value)
        assert "Invalid chat template" in msg or "template" in msg.lower()

    @pytest.mark.sanity
    def test_request_format_stored_on_backend(self):
        """
        Custom request_format is stored on the backend, not in vllm_config.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="test-model",
                request_format="/path/to/template.jinja",
            )
        assert backend_custom.request_format == "/path/to/template.jinja"
        assert "chat_template" not in backend_custom.vllm_config

    @pytest.mark.sanity
    def test_request_format_plain_not_in_vllm_config(self):
        """
        request_format=plain does not add chat_template to vllm_config.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
        assert backend_plain.request_format == "plain"
        assert "chat_template" not in backend_plain.vllm_config

    @pytest.mark.sanity
    def test_request_format_default_template_not_in_vllm_config(self):
        """
        request_format=default-template does not add chat_template to vllm_config.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_def = VLLMPythonBackend(
                model="test-model", request_format="default-template"
            )
        assert backend_def.request_format == "default-template"
        assert "chat_template" not in backend_def.vllm_config

    @pytest.mark.sanity
    def test_vllm_config_empty_uses_vllm_defaults(self):
        """
        With vllm_config empty or None, backend only sets model; no extra keys.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_empty = VLLMPythonBackend(model="test-model", vllm_config={})
            backend_none = VLLMPythonBackend(model="test-model", vllm_config=None)
        for b in (backend_empty, backend_none):
            assert b.vllm_config.get("model") == "test-model"
            assert "tensor_parallel_size" not in b.vllm_config
            assert "gpu_memory_utilization" not in b.vllm_config


class TestVLLMStreamingUsageFromOutput:
    """
    Test _usage_from_output: token_ids vs token_iterations fallback.
    """

    @pytest.mark.smoke
    def test_streaming_usage_uses_token_ids(self, backend):
        """
        When token_ids is set, it is used for completion_tokens.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.token_ids = [100, 101, 102, 103]
        mock_out.text = "Hi"
        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2]
        mock_final.outputs = [mock_out]
        mock_info = Mock()
        mock_info.timings.token_iterations = 1
        usage = backend._usage_from_output(
            mock_final,
            request_info=mock_info,
        )
        assert usage is not None
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 6

    @pytest.mark.sanity
    def test_streaming_usage_fallback_to_token_iterations(self, backend):
        """
        When token_ids is None, falls back to token_iterations.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.token_ids = None
        mock_out.text = "Hi"
        mock_final = Mock()
        mock_final.prompt_token_ids = [1]
        mock_final.outputs = [mock_out]
        mock_info = Mock()
        mock_info.timings.token_iterations = 7
        usage = backend._usage_from_output(
            mock_final,
            request_info=mock_info,
        )
        assert usage is not None
        assert usage["completion_tokens"] == 7
        assert usage["total_tokens"] == 8


class TestVLLMStreamingFinalTokenCount:
    """
    Test that streaming uses the final output's token_ids for the response.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_final_output_token_ids_used_for_count(self, backend):
        """
        vLLM yields cumulative token_ids; final output has 3 token_ids -> 3 tokens.
        ## WRITTEN BY AI ##
        """
        out1 = Mock()
        out1.text = "A"
        out1.token_ids = [100]
        out1.finish_reason = None
        out2 = Mock()
        out2.text = "AB"
        out2.token_ids = [100, 101]
        out2.finish_reason = None
        out3 = Mock()
        out3.text = "ABC"
        out3.token_ids = [100, 101, 102]
        out3.finish_reason = None
        req1, req2, req3 = Mock(), Mock(), Mock()
        for req in (req1, req2, req3):
            req.request_id = None
            req.prompt_token_ids = [1, 2]
        req1.outputs, req2.outputs, req3.outputs = [out1], [out2], [out3]

        async def mock_generate(_prompt, _params, _req_id):
            for r in (req1, req2, req3):
                yield r

        backend._engine = Mock()
        backend._engine.generate = mock_generate
        request = GenerationRequest(
            columns={"text_column": ["Hi"]},
            output_metrics=UsageMetrics(text_tokens=10),
        )
        request_info = RequestInfo()

        with patch.object(
            backend,
            "_resolve_request",
            return_value=_ResolvedRequest(
                prompt="Hi", stream=True, multi_modal_data=None
            ),
        ):
            final_response = None
            async for response, _ in backend.resolve(request, request_info):
                final_response = response

        assert final_response is not None
        assert final_response.output_metrics.text_tokens == 3


class TestVLLMCreateSamplingParams:
    """
    Test _create_sampling_params: max_tokens_override and defaults.
    """

    @pytest.mark.smoke
    def test_override_used(self, backend):
        """
        max_tokens_override is used as max_tokens.
        ## WRITTEN BY AI ##
        """
        params = backend._create_sampling_params(max_tokens_override=2000)
        assert params.max_tokens == 2000

    @pytest.mark.smoke
    def test_default_uses_vllm_defaults(self, backend):
        """
        Without override, no params are set so vLLM defaults are used.
        ## WRITTEN BY AI ##
        """
        params = backend._create_sampling_params()
        assert not hasattr(params, "max_tokens")
        assert not hasattr(params, "ignore_eos")

    @pytest.mark.sanity
    def test_override_sets_ignore_eos(self, backend):
        """
        When override is used, ignore_eos=True to match HTTP backend behavior.
        ## WRITTEN BY AI ##
        """
        params = backend._create_sampling_params(max_tokens_override=2000)
        assert params.ignore_eos is True

    @pytest.mark.sanity
    def test_zero_override_uses_vllm_defaults(self, backend):
        """
        max_tokens_override=0 is treated as no override (vLLM defaults).
        ## WRITTEN BY AI ##
        """
        params = backend._create_sampling_params(max_tokens_override=0)
        assert not hasattr(params, "max_tokens")
        assert not hasattr(params, "ignore_eos")


class TestVLLMLifecycle:
    """
    process_startup/shutdown/validate with mocked AsyncEngineArgs and AsyncLLMEngine.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_process_startup_success(self):
        """
        Success: mock AsyncEngineArgs and from_engine_args; set _engine, _in_process.
        ## WRITTEN BY AI ##
        """
        mock_engine = Mock()
        with (
            patch("guidellm.backends.vllm_python.vllm._check_vllm_available"),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncEngineArgs",
                return_value=Mock(),
            ),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncLLMEngine"
            ) as mock_engine_cls,
        ):
            mock_engine_cls.from_engine_args = Mock(return_value=mock_engine)
            backend = VLLMPythonBackend(model="test-model")
            await backend.process_startup()
        assert backend._engine is mock_engine
        assert backend._in_process is True

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_process_startup_idempotency_raises(self):
        """
        Idempotency: raise RuntimeError when _in_process already True.
        ## WRITTEN BY AI ##
        """
        mock_engine = Mock()
        with (
            patch("guidellm.backends.vllm_python.vllm._check_vllm_available"),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncEngineArgs",
                return_value=Mock(),
            ),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncLLMEngine"
            ) as mock_engine_cls,
        ):
            mock_engine_cls.from_engine_args = Mock(return_value=mock_engine)
            backend = VLLMPythonBackend(model="test-model")
            await backend.process_startup()
            with pytest.raises(RuntimeError, match="Backend already started up"):
                await backend.process_startup()

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_process_shutdown_success(self):
        """
        Success: _engine.shutdown() called; _engine None, _in_process False.
        ## WRITTEN BY AI ##
        """
        mock_engine = Mock()
        with (
            patch("guidellm.backends.vllm_python.vllm._check_vllm_available"),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncEngineArgs",
                return_value=Mock(),
            ),
            patch(
                "guidellm.backends.vllm_python.vllm.AsyncLLMEngine"
            ) as mock_engine_cls,
        ):
            mock_engine_cls.from_engine_args = Mock(return_value=mock_engine)
            backend = VLLMPythonBackend(model="test-model")
            await backend.process_startup()
            await backend.process_shutdown()
        mock_engine.shutdown.assert_called_once()
        assert backend._engine is None
        assert backend._in_process is False

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_process_shutdown_not_started_raises(self):
        """
        Raise RuntimeError when not started.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model")
            backend._in_process = False
            backend._engine = None
        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.process_shutdown()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_validate_engine_none_raises(self, backend):
        """
        Raise RuntimeError when _engine is None.
        ## WRITTEN BY AI ##
        """
        backend._engine = None
        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.validate()

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_validate_success(self, backend):
        """
        Success: mock _engine.check_health and _engine.generate; no raise.
        ## WRITTEN BY AI ##
        """

        async def check_health_ok():
            pass

        async def one_yield(*args, **kwargs):
            yield Mock(outputs=[Mock()], prompt_token_ids=[])

        backend._engine = Mock()
        backend._engine.check_health = check_health_ok
        backend._engine.generate = one_yield
        await backend.validate()


class TestVLLMModels:
    """
    available_models and default_model.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_available_models_returns_model_list(self, backend):
        """
        Returns [self.model].
        ## WRITTEN BY AI ##
        """
        models = await backend.available_models()
        assert models == ["test-model"]

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_default_model_returns_model(self, backend):
        """
        Returns self.model.
        ## WRITTEN BY AI ##
        """
        model = await backend.default_model()
        assert model == "test-model"


class TestVLLMValidateHelpers:
    """
    _validate_backend_initialized and _validate_history.
    """

    @pytest.mark.sanity
    def test_validate_backend_initialized_raises_when_engine_none(self, backend):
        """
        Raises RuntimeError when _engine is None.
        ## WRITTEN BY AI ##
        """
        backend._engine = None
        with pytest.raises(RuntimeError, match="Backend not started up"):
            backend._validate_backend_initialized()

    @pytest.mark.sanity
    def test_validate_backend_initialized_passes_when_engine_set(self, backend):
        """
        Does not raise when _engine is set.
        ## WRITTEN BY AI ##
        """
        backend._engine = Mock()
        backend._validate_backend_initialized()

    @pytest.mark.sanity
    def test_validate_history_raises_when_history_not_none(self, backend):
        """
        Raises NotImplementedError when history is not None.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(
            NotImplementedError, match="Multi-turn requests not yet supported"
        ):
            backend._validate_history([(Mock(), Mock())])

    @pytest.mark.sanity
    def test_validate_history_passes_when_history_none(self, backend):
        """
        Does not raise when history is None.
        ## WRITTEN BY AI ##
        """
        backend._validate_history(None)


class TestVLLMTextFromOutput:
    """
    _text_from_output: None, empty outputs, normal.
    """

    @pytest.mark.sanity
    def test_text_from_output_none_returns_empty(self, backend):
        """
        output is None -> "".
        ## WRITTEN BY AI ##
        """
        assert backend._text_from_output(None) == ""

    @pytest.mark.sanity
    def test_text_from_output_empty_outputs_returns_empty(self, backend):
        """
        output.outputs empty -> "".
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.outputs = []
        assert backend._text_from_output(mock_out) == ""

    @pytest.mark.smoke
    def test_text_from_output_normal_returns_first_text(self, backend):
        """
        Normal -> output.outputs[0].text.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.outputs = [Mock(text="Hello world")]
        assert backend._text_from_output(mock_out) == "Hello world"

    @pytest.mark.sanity
    def test_text_from_output_none_text_returns_empty(self, backend):
        """
        output.outputs[0].text is None -> "".
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.outputs = [Mock(text=None)]
        assert backend._text_from_output(mock_out) == ""


class TestVLLMUsageFromOutputNonStream:
    """
    _usage_from_output with request_info=None (non-stream path).
    """

    @pytest.mark.smoke
    def test_usage_from_output_non_stream_uses_output_token_counts(self, backend):
        """
        Non-stream: use only output token counts.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.prompt_token_ids = [1, 2, 3]
        mock_out.outputs = [Mock(token_ids=[10, 20])]
        usage = backend._usage_from_output(mock_out, request_info=None)
        assert usage == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    @pytest.mark.sanity
    def test_usage_from_output_non_stream_outputs_empty(self, backend):
        """
        Non-stream with no outputs -> completion_tokens 0.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.prompt_token_ids = [1]
        mock_out.outputs = []
        usage = backend._usage_from_output(mock_out, request_info=None)
        assert usage["completion_tokens"] == 0
        assert usage["prompt_tokens"] == 1


class TestVLLMBuildFinalResponse:
    """
    _build_final_response: final_output None; stream vs non-stream.
    """

    @pytest.mark.sanity
    def test_build_final_response_none_returns_none(self, backend):
        """
        final_output is None -> return None.
        ## WRITTEN BY AI ##
        """
        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        result = backend._build_final_response(request, request_info, None, stream=True)
        assert result is None

    @pytest.mark.smoke
    def test_build_final_response_stream_returns_response(self, backend):
        """
        Stream path builds GenerationResponse with text and usage.
        ## WRITTEN BY AI ##
        """
        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2]
        mock_final.outputs = [Mock(token_ids=[3, 4], text="ab")]
        mock_final.request_id = "req-1"
        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        result = backend._build_final_response(
            request, request_info, mock_final, stream=True, text="ab"
        )
        assert result is not None
        resp, info = result
        assert isinstance(resp, GenerationResponse)
        assert resp.text == "ab"
        assert info is request_info

    @pytest.mark.sanity
    def test_build_final_response_non_stream_returns_response(self, backend):
        """
        Non-stream path builds GenerationResponse with text and usage.
        ## WRITTEN BY AI ##
        """
        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2]
        mock_final.outputs = [Mock(token_ids=[3, 4], text="hello")]
        mock_final.request_id = "req-1"
        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        result = backend._build_final_response(
            request, request_info, mock_final, stream=False
        )
        assert result is not None
        resp, info = result
        assert isinstance(resp, GenerationResponse)
        assert resp.text == "hello"
        assert info is request_info


class TestVLLMExtractTextFromContent:
    """
    _extract_text_from_content: string, list of text blocks, fallback.
    """

    @pytest.mark.smoke
    def test_extract_text_from_content_string_returns_same(self, backend):
        """
        String -> same string.
        ## WRITTEN BY AI ##
        """
        assert backend._extract_text_from_content("hello") == "hello"

    @pytest.mark.sanity
    def test_extract_text_from_content_list_text_blocks_concatenated(self, backend):
        """
        List of {type:text, text:x} -> concatenated.
        ## WRITTEN BY AI ##
        """
        content = [
            {"type": "text", "text": "a"},
            {"type": "image_url", "url": "x"},
            {"type": "text", "text": "b"},
        ]
        assert backend._extract_text_from_content(content) == "ab"

    @pytest.mark.sanity
    def test_extract_text_from_content_list_no_text_blocks_returns_empty(self, backend):
        """
        List without text blocks -> "".
        ## WRITTEN BY AI ##
        """
        content = [{"type": "image_url", "url": "x"}]
        assert backend._extract_text_from_content(content) == ""

    @pytest.mark.sanity
    def test_extract_text_from_content_fallback_str(self, backend):
        """
        Non-str non-list -> str(content).
        ## WRITTEN BY AI ##
        """
        assert backend._extract_text_from_content(123) == "123"

    @pytest.mark.sanity
    def test_extract_text_from_content_none_returns_empty(self, backend):
        """
        content None -> "".
        ## WRITTEN BY AI ##
        """
        assert backend._extract_text_from_content(None) == ""


class TestVLLMResolveValidation:
    """
    resolve(): history -> NotImplementedError; _engine None -> RuntimeError.
    """

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_resolve_with_history_raises_not_implemented(self, backend):
        """
        history=[(...)] -> NotImplementedError.
        ## WRITTEN BY AI ##
        """
        backend._engine = Mock()
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        request_info = RequestInfo()
        history = [(Mock(), Mock())]
        with pytest.raises(
            NotImplementedError, match="Multi-turn requests not yet supported"
        ):
            async for _ in backend.resolve(request, request_info, history=history):
                pass

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_resolve_engine_none_raises_runtime_error(self, backend):
        """
        _engine is None -> RuntimeError.
        ## WRITTEN BY AI ##
        """
        backend._engine = None
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        request_info = RequestInfo()
        with pytest.raises(RuntimeError, match="Backend not started up"):
            async for _ in backend.resolve(request, request_info):
                pass


class TestVLLMResolveCancelledError:
    """
    resolve(): CancelledError after yield; response still yielded, then re-raise.
    """

    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_resolve_cancelled_error_yields_then_reraises(self, backend):
        """
        During generate raise CancelledError; assert response yielded then re-raise.
        ## WRITTEN BY AI ##
        """
        out = Mock()
        out.text = "partial"
        out.token_ids = [1]
        out.finish_reason = None
        req_out = Mock()
        req_out.prompt_token_ids = [1]
        req_out.outputs = [out]
        req_out.request_id = "r1"

        async def yield_then_cancel(prompt, sampling_params, request_id):
            yield req_out
            raise asyncio.CancelledError

        backend._engine = Mock()
        backend._engine.generate = yield_then_cancel
        request = GenerationRequest(columns={"text_column": ["Hi"]})
        request_info = RequestInfo()
        results = []

        async def collect():
            async for response, info in backend.resolve(request, request_info):
                results.append((response, info))

        with (
            patch.object(
                backend,
                "_resolve_request",
                return_value=_ResolvedRequest(
                    prompt="Hi", stream=True, multi_modal_data=None
                ),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await collect()
        assert len(results) == 1
        assert results[0][0].text == "partial"


class TestVLLMResolveAudioFromColumns:
    """
    resolve() with audio_column: multimodal data passed to engine.generate.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_resolve_audio_from_columns_uses_multi_modal_data(self, backend):
        """
        Audio from audio_column: generate_input is dict with multi_modal_data.
        ## WRITTEN BY AI ##
        """
        mock_audio_array = np.array([0.0, 0.1], dtype=np.float32)
        mock_decode_result = _mock_audio_decode_result(mock_audio_array)

        out = Mock()
        out.text = "transcribed"
        out.token_ids = [1, 2]
        out.finish_reason = "stop"
        req_out = Mock()
        req_out.prompt_token_ids = [1]
        req_out.outputs = [out]
        req_out.request_id = "r1"

        seen_prompt_arg = []

        async def mock_generate(prompt, sampling_params, request_id):
            seen_prompt_arg.append(prompt)
            assert isinstance(prompt, dict)
            assert "prompt" in prompt
            assert "multi_modal_data" in prompt
            assert "audio" in prompt["multi_modal_data"]
            yield req_out

        request = GenerationRequest(
            columns={
                "audio_column": [{"audio": b"fake-wav-bytes", "format": "wav"}],
            }
        )
        request.output_metrics = UsageMetrics()

        with patch(
            "guidellm.backends.vllm_python.vllm._decode_audio",
            return_value=mock_decode_result,
        ):
            backend._engine = Mock()
            backend._engine.generate = mock_generate
            request_info = RequestInfo()
            results = []
            async for response, info in backend.resolve(request, request_info):
                results.append((response, info))
            assert len(results) == 1
            assert results[0][0].text == "transcribed"
            assert len(seen_prompt_arg) == 1
            assert seen_prompt_arg[0]["multi_modal_data"]["audio"] is mock_audio_array
            prompt_str = seen_prompt_arg[0]["prompt"]
            assert "<|audio|>" in prompt_str
