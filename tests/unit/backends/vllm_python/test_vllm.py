"""
Unit tests for VLLM Python backend.

Tests _get_request_context (mode from body/files; body from columns or arguments)
and request_format (plain, default-template, custom template) for chat prompts.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import numpy as np
import pytest

from guidellm.backends.vllm_python.vllm import VLLMPythonBackend, _RequestContext
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    UsageMetrics,
)


@pytest.fixture
def backend():
    """VLLMPythonBackend instance without requiring vllm to be installed."""
    with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
        return VLLMPythonBackend(model="test-model")


class TestVLLMGetRequestContext:
    """
    Test _get_request_context: mode inferred from body and files.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_columns_only_infers_chat(self, backend):
        """
        Request with only columns (no arguments) builds messages and infers mode chat.
        ## WRITTEN BY AI ##
        """
        request = GenerationRequest(columns={"text_column": ["hello"]})
        ctx = backend._get_request_context(request)
        assert ctx.mode == "chat"
        assert "messages" in ctx.body
        assert len(ctx.body["messages"]) == 1
        assert ctx.body["messages"][0]["role"] == "user"
        assert ctx.stream is True  # backend default stream=True (match HTTP behavior)
        assert ctx.files == {}

    @pytest.mark.smoke
    def test_columns_only_uses_backend_stream_false(self):
        """
        When backend.stream=False and no arguments, ctx.stream is False.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model", stream=False)
        request = GenerationRequest(columns={"text_column": ["hello"]})
        ctx = backend._get_request_context(request)
        assert ctx.stream is False

    @pytest.mark.smoke
    def test_arguments_stream_overrides_backend_stream(self):
        """
        When request.arguments.stream is set, it overrides backend.stream.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend = VLLMPythonBackend(model="test-model", stream=False)
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {"messages": [{"role": "user", "content": "Hi"}]}
        request.arguments.stream = True  # override backend's stream=False
        request.arguments.files = {}
        ctx = backend._get_request_context(request)
        assert ctx.stream is True

    @pytest.mark.smoke
    def test_arguments_body_prompt_infers_text(self, backend):
        """
        Request with arguments.body containing only prompt infers mode text.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {"prompt": "Complete this"}
        request.arguments.stream = False
        request.arguments.files = {}
        ctx = backend._get_request_context(request)
        assert ctx.mode == "text"
        assert ctx.body == {"prompt": "Complete this"}
        assert ctx.files == {}

    @pytest.mark.smoke
    def test_arguments_body_messages_infers_chat(self, backend):
        """
        Request with arguments.body containing messages infers mode chat.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {
            "messages": [{"role": "user", "content": "Hi"}],
        }
        request.arguments.stream = True
        request.arguments.files = {}
        ctx = backend._get_request_context(request)
        assert ctx.mode == "chat"
        assert ctx.body["messages"] == [{"role": "user", "content": "Hi"}]
        assert ctx.stream is True
        assert ctx.files == {}

    @pytest.mark.smoke
    def test_arguments_non_empty_files_infers_audio(self, backend):
        """
        Request with non-empty files infers mode audio.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {}
        request.arguments.stream = False
        request.arguments.files = {"file": (b"audio-data", "audio/wav")}
        ctx = backend._get_request_context(request)
        assert ctx.mode == "audio"
        assert ctx.files == {"file": (b"audio-data", "audio/wav")}

    @pytest.mark.smoke
    def test_arguments_files_take_precedence_over_body(self, backend):
        """
        When files is non-empty, mode is audio even if body has messages.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {"messages": [{"role": "user", "content": "Hi"}]}
        request.arguments.stream = False
        request.arguments.files = {"file": (b"x", "audio/wav")}
        ctx = backend._get_request_context(request)
        assert ctx.mode == "audio"

    @pytest.mark.smoke
    def test_arguments_empty_body_no_files_raises(self, backend):
        """
        Request with no prompt, no messages, and no files raises ValueError.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {}
        request.arguments.stream = False
        request.arguments.files = {}
        with pytest.raises(
            ValueError, match="Request must include prompt, messages, or audio files"
        ):
            backend._get_request_context(request)

    @pytest.mark.smoke
    def test_arguments_body_prompt_key_empty_string_infers_text(self, backend):
        """
        Body with key 'prompt' present (even empty string) infers mode text.
        ## WRITTEN BY AI ##
        """
        request = Mock(spec=GenerationRequest)
        request.arguments = Mock()
        request.arguments.body = {"prompt": ""}
        request.arguments.stream = False
        request.arguments.files = {}
        ctx = backend._get_request_context(request)
        assert ctx.mode == "text"
        assert ctx.body["prompt"] == ""

    @pytest.mark.smoke
    def test_columns_prefix_and_text_build_messages(self, backend):
        """
        Columns with prefix_column and text_column build multiple messages, mode chat.
        ## WRITTEN BY AI ##
        """
        request = GenerationRequest(
            columns={
                "prefix_column": ["System prompt"],
                "text_column": ["User question"],
            }
        )
        ctx = backend._get_request_context(request)
        assert ctx.mode == "chat"
        assert len(ctx.body["messages"]) == 2
        assert ctx.body["messages"][0]["role"] == "system"
        assert ctx.body["messages"][0]["content"] == "System prompt"
        assert ctx.body["messages"][1]["role"] == "user"
        assert ctx.body["messages"][1]["content"][0]["text"] == "User question"


class TestVLLMRequestFormat:
    """
    Test request_format: plain, default-template, and custom template.
    """

    @pytest.mark.smoke
    def test_request_format_plain_produces_concatenated_prompt(self, backend):
        """
        With request_format=plain, chat prompt is plain concatenation (no template).
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Bye"},
            ]
        }
        # _extract_prompt for chat uses self._engine.tokenizer only when not plain
        prompt = backend_plain._extract_prompt(body, "chat")
        assert "User: Hello" in prompt
        assert "Assistant: Hi" in prompt
        assert "User: Bye" in prompt
        assert "Assistant: " in prompt
        # No template markers like [INST] or <|im_start|>
        assert "[INST]" not in prompt
        assert "<|im_start|>" not in prompt

    @pytest.mark.smoke
    def test_request_format_default_template_uses_apply_chat_template(self, backend):
        """
        With request_format=default-template or None, apply_chat_template is used.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_default = VLLMPythonBackend(
                model="test-model", request_format="default-template"
            )
            backend_default._engine = Mock()
            backend_default._engine.tokenizer = mock_tokenizer
        prompt = backend_default._extract_prompt(body, "chat")
        assert prompt == "formatted_prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_kw = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("tokenize") is False
        assert call_kw.get("add_generation_prompt") is True

    @pytest.mark.smoke
    def test_request_format_none_uses_apply_chat_template(self, backend):
        """
        With request_format=None, tokenizer.apply_chat_template is used.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "default_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_none = VLLMPythonBackend(model="test-model")
            backend_none._engine = Mock()
            backend_none._engine.tokenizer = mock_tokenizer
        prompt = backend_none._extract_prompt(body, "chat")
        assert prompt == "default_prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.smoke
    def test_request_format_custom_template_string_sets_tokenizer_and_applies(
        self, backend
    ):
        """
        With request_format=custom, chat_template is set then apply_chat_template.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "custom_prompt"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="test-model",
                request_format="{{ messages[0]['content'] }}",
            )
            backend_custom._engine = Mock()
            backend_custom._engine.tokenizer = mock_tokenizer
        prompt = backend_custom._extract_prompt(body, "chat")
        assert prompt == "custom_prompt"
        assert mock_tokenizer.chat_template == "{{ messages[0]['content'] }}"
        mock_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.smoke
    def test_request_format_custom_template_from_file_sets_tokenizer_and_applies(
        self, backend, tmp_path
    ):
        """
        With request_format=file path, chat_template is set from file then applied.
        ## WRITTEN BY AI ##
        """
        template_file = tmp_path / "template.jinja"
        template_file.write_text("Custom: {{ messages[0]['content'] }}")
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "Custom: Hi"
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_file = VLLMPythonBackend(
                model="test-model", request_format=str(template_file)
            )
            backend_file._engine = Mock()
            backend_file._engine.tokenizer = mock_tokenizer
        prompt = backend_file._extract_prompt(body, "chat")
        assert prompt == "Custom: Hi"
        assert mock_tokenizer.chat_template == "Custom: {{ messages[0]['content'] }}"
        mock_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.smoke
    def test_request_format_stored_and_passed_to_vllm_config(self, backend):
        """
        Custom request_format is added to vllm_config as chat_template.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_custom = VLLMPythonBackend(
                model="test-model",
                request_format="/path/to/template.jinja",
            )
        assert backend_custom.request_format == "/path/to/template.jinja"
        assert (
            backend_custom.vllm_config.get("chat_template") == "/path/to/template.jinja"
        )

    @pytest.mark.smoke
    def test_request_format_plain_not_in_vllm_config(self, backend):
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

    @pytest.mark.smoke
    def test_request_format_default_template_not_in_vllm_config(self, backend):
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

    @pytest.mark.smoke
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


class TestVLLMExtractPromptTokenizerWarnings:
    """
    Test warnings when tokenizer is set but not used for prompt extraction.
    """

    @pytest.mark.smoke
    def test_text_mode_warns_when_engine_has_tokenizer(self, backend):
        """
        With mode=text and engine.tokenizer set, a warning is logged.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend._engine = Mock()
            backend._engine.tokenizer = Mock()
        with patch("guidellm.backends.vllm_python.vllm.logger") as mock_logger:
            result = backend._extract_prompt({"prompt": "Complete this"}, "text")
        assert result == "Complete this"
        mock_logger.warning.assert_called_once()
        call_msg = mock_logger.warning.call_args[0][0]
        assert "Tokenizer is set" in call_msg
        assert "mode was inferred as text" in call_msg

    @pytest.mark.smoke
    def test_text_mode_no_warning_when_engine_has_no_tokenizer(self, backend):
        """
        With mode=text and no engine tokenizer, no warning is logged.
        ## WRITTEN BY AI ##
        """
        backend._engine = None
        with patch("guidellm.backends.vllm_python.vllm.logger") as mock_logger:
            result = backend._extract_prompt({"prompt": "Complete this"}, "text")
        assert result == "Complete this"
        mock_logger.warning.assert_not_called()

    @pytest.mark.smoke
    def test_chat_plain_no_warning_when_engine_has_no_tokenizer(self, backend):
        """
        With request_format=plain and no engine tokenizer, no warning is logged.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
            backend_plain._engine = None
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        with patch("guidellm.backends.vllm_python.vllm.logger") as mock_logger:
            backend_plain._extract_prompt(body, "chat")
        mock_logger.warning.assert_not_called()


class TestVLLMStreamingUsageFromOutput:
    """
    Test _usage_from_output (stream path): token_ids vs tokenizer fallback.
    """

    @pytest.mark.smoke
    def test_streaming_usage_from_output_token_ids_none_uses_tokenizer(self, backend):
        """
        When token_ids is None, output_tokens are derived from tokenizer.encode.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.token_ids = None
        mock_out.text = "Hello world"
        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2, 3]
        mock_final.outputs = [mock_out]
        mock_info = Mock()
        mock_info.timings.token_iterations = 10
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [10, 20, 30, 40, 50]  # 5 tokens
        backend._engine = Mock()
        backend._engine.tokenizer = mock_tokenizer
        usage = backend._usage_from_output(
            mock_final,
            accumulated_text="Hello world",
            request_info=mock_info,
        )
        assert usage is not None
        assert usage["prompt_tokens"] == 3
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 8
        mock_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=False
        )

    @pytest.mark.smoke
    def test_streaming_usage_from_output_uses_token_ids_when_available(self, backend):
        """
        When token_ids is set, it is used and tokenizer is not called.
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
        backend._engine = Mock()
        backend._engine.tokenizer = Mock()
        usage = backend._usage_from_output(
            mock_final,
            accumulated_text="Hi",
            request_info=mock_info,
        )
        assert usage is not None
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 6
        backend._engine.tokenizer.encode.assert_not_called()

    @pytest.mark.smoke
    def test_streaming_usage_from_output_fallback_to_token_iterations(self, backend):
        """
        When token_ids is None and tokenizer fails, use token_iterations.
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
        backend._engine = Mock()
        backend._engine.tokenizer = Mock(side_effect=RuntimeError("no tokenizer"))
        usage = backend._usage_from_output(
            mock_final,
            accumulated_text="Hi",
            request_info=mock_info,
        )
        assert usage is not None
        assert usage["completion_tokens"] == 7
        assert usage["total_tokens"] == 8


class TestVLLMStreamingCumulativeTokenCount:
    """
    Test that streaming correctly counts tokens when vLLM yields CUMULATIVE token_ids.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_cumulative_token_ids_counted_as_delta(self, backend):
        """
        vLLM yields cumulative token_ids; we add delta only.
        ## WRITTEN BY AI ##
        """
        # Simulate 3 chunks, cumulative token_ids [1],[1,2],[1,2,3] -> 3 tokens
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
            req.prompt_token_ids = [1, 2]  # for _usage_from_output
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

        with patch.object(backend, "_extract_prompt", return_value="Hi"):
            final_response = None
            async for response, _ in backend.resolve(request, request_info):
                final_response = response

        assert final_response is not None
        assert final_response.output_metrics.text_tokens == 3


class TestVLLMCreateSamplingParams:
    """
    Test _create_sampling_params: max_tokens from body vs request override.
    """

    @pytest.mark.smoke
    def test_max_tokens_override_used_when_body_has_no_max_tokens(self, backend):
        """
        When body has no max_tokens, max_tokens_override is used.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        params = backend._create_sampling_params(body, max_tokens_override=2000)
        assert params.max_tokens == 2000

    @pytest.mark.smoke
    def test_body_max_tokens_takes_precedence_over_override(self, backend):
        """
        When body has max_tokens, override is ignored.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64}
        params = backend._create_sampling_params(body, max_tokens_override=2000)
        assert params.max_tokens == 64

    @pytest.mark.smoke
    def test_default_max_tokens_when_no_body_no_override(self, backend):
        """
        When body has no max_tokens and no override, default is 16.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        params = backend._create_sampling_params(body)
        assert params.max_tokens == 16

    @pytest.mark.smoke
    def test_override_used_sets_ignore_eos_and_stop_like_http(self, backend):
        """
        When max_tokens_override is used, ignore_eos=True and stop=[] to match HTTP.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        params = backend._create_sampling_params(body, max_tokens_override=2000)
        assert params.max_tokens == 2000
        assert params.ignore_eos is True
        assert params.stop == []

    @pytest.mark.smoke
    def test_override_used_with_600_also_sets_ignore_eos_and_stop(self, backend):
        """
        When max_tokens_override is used (e.g. 600), same: ignore_eos=True, stop=[].
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        params = backend._create_sampling_params(body, max_tokens_override=600)
        assert params.max_tokens == 600
        assert params.ignore_eos is True
        assert params.stop == []

    @pytest.mark.smoke
    def test_body_max_tokens_uses_body_ignore_eos_and_stop(self, backend):
        """
        When body has max_tokens, ignore_eos and stop come from body.
        ## WRITTEN BY AI ##
        """
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 64,
            "ignore_eos": True,
            "stop": ["</s>"],
        }
        params = backend._create_sampling_params(body, max_tokens_override=2000)
        assert params.max_tokens == 64
        assert params.ignore_eos is True
        assert params.stop == ["</s>"]

    @pytest.mark.smoke
    def test_max_tokens_zero_safeguard(self, backend):
        """
        When body has max_tokens=0, we use 16 so vLLM never receives 0.
        ## WRITTEN BY AI ##
        """
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0}
        params = backend._create_sampling_params(body)
        assert params.max_tokens == 16


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
    @pytest.mark.smoke
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
    @pytest.mark.smoke
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
    @pytest.mark.smoke
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
        Success: mock _engine.generate async iterator yielding one item; no raise.
        ## WRITTEN BY AI ##
        """

        async def one_yield(*args, **kwargs):
            yield Mock(outputs=[Mock()], prompt_token_ids=[])

        backend._engine = Mock()
        backend._engine.generate = one_yield
        with patch(
            "guidellm.backends.vllm_python.vllm.SamplingParams", return_value=Mock()
        ):
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

    @pytest.mark.smoke
    def test_validate_backend_initialized_raises_when_engine_none(self, backend):
        """
        Raises RuntimeError when _engine is None.
        ## WRITTEN BY AI ##
        """
        backend._engine = None
        with pytest.raises(RuntimeError, match="Backend not started up"):
            backend._validate_backend_initialized()

    @pytest.mark.smoke
    def test_validate_backend_initialized_passes_when_engine_set(self, backend):
        """
        Does not raise when _engine is set.
        ## WRITTEN BY AI ##
        """
        backend._engine = Mock()
        backend._validate_backend_initialized()

    @pytest.mark.smoke
    def test_validate_history_raises_when_history_not_none(self, backend):
        """
        Raises NotImplementedError when history is not None.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(
            NotImplementedError, match="Multi-turn requests not yet supported"
        ):
            backend._validate_history([(Mock(), Mock())])

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_text_from_output_none_returns_empty(self, backend):
        """
        output is None -> "".
        ## WRITTEN BY AI ##
        """
        assert backend._text_from_output(None) == ""

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_text_from_output_none_text_returns_empty(self, backend):
        """
        output.outputs[0].text is None -> "".
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.outputs = [Mock(text=None)]
        assert backend._text_from_output(mock_out) == ""


class TestVLLMOpenAIPayloadForMode:
    """
    _openai_payload_for_mode: audio, text, chat delta/message.
    """

    @pytest.mark.smoke
    def test_openai_payload_audio_returns_text_key(self, backend):
        """
        mode audio -> {"text": ...}.
        ## WRITTEN BY AI ##
        """
        out = backend._openai_payload_for_mode("audio", "transcribed", is_delta=False)
        assert out == {"text": "transcribed"}

    @pytest.mark.smoke
    def test_openai_payload_text_returns_choices_text(self, backend):
        """
        mode text -> choices[].text.
        ## WRITTEN BY AI ##
        """
        out = backend._openai_payload_for_mode("text", "completion", is_delta=False)
        assert out == {"choices": [{"text": "completion"}]}

    @pytest.mark.smoke
    def test_openai_payload_chat_delta_returns_choices_delta(self, backend):
        """
        mode chat is_delta=True -> choices[].delta.
        ## WRITTEN BY AI ##
        """
        out = backend._openai_payload_for_mode("chat", "hi", is_delta=True)
        assert out == {"choices": [{"delta": {"content": "hi", "role": "assistant"}}]}

    @pytest.mark.smoke
    def test_openai_payload_chat_message_returns_choices_message(self, backend):
        """
        mode chat is_delta=False -> choices[].message.
        ## WRITTEN BY AI ##
        """
        out = backend._openai_payload_for_mode("chat", "hello", is_delta=False)
        assert out == {
            "choices": [{"message": {"content": "hello", "role": "assistant"}}]
        }


class TestVLLMUsageFromOutputNonStream:
    """
    _usage_from_output with request_info=None (non-stream path).
    """

    @pytest.mark.smoke
    def test_usage_from_output_non_stream_uses_output_token_counts(self, backend):
        """
        Non-stream: use only output token counts (no tokenizer/request_info).
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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_build_final_response_none_returns_none(self, backend):
        """
        final_output is None -> return None.
        ## WRITTEN BY AI ##
        """
        from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler

        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        ctx = _RequestContext(mode="chat", body={}, stream=True, files={})
        handler = VLLMResponseHandler()
        result = backend._build_final_response(
            request, request_info, None, ctx, handler, stream=True
        )
        assert result is None

    @pytest.mark.smoke
    def test_build_final_response_stream_returns_compile_streaming(self, backend):
        """
        Stream path uses compile_streaming.
        ## WRITTEN BY AI ##
        """
        from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler

        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2]
        mock_final.outputs = [Mock(token_ids=[3, 4], text="ab")]
        mock_final.request_id = "req-1"
        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        ctx = _RequestContext(mode="chat", body={}, stream=True, files={})
        handler = VLLMResponseHandler()
        handler.add_streaming_line('data: {"choices": [{"delta": {"content": "ab"}}]}')
        result = backend._build_final_response(
            request,
            request_info,
            mock_final,
            ctx,
            handler,
            stream=True,
            accumulated_text="ab",
            total_output_tokens=2,
        )
        assert result is not None
        resp, info = result
        assert isinstance(resp, GenerationResponse)
        assert resp.text == "ab"
        assert info is request_info

    @pytest.mark.smoke
    def test_build_final_response_non_stream_returns_compile_non_streaming(
        self, backend
    ):
        """
        Non-stream: convert to OpenAI format and compile_non_streaming.
        ## WRITTEN BY AI ##
        """
        from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler

        mock_final = Mock()
        mock_final.prompt_token_ids = [1, 2]
        mock_final.outputs = [Mock(token_ids=[3, 4], text="hello")]
        mock_final.request_id = "req-1"
        request = GenerationRequest(columns={"text_column": ["x"]})
        request_info = RequestInfo()
        ctx = _RequestContext(mode="text", body={}, stream=False, files={})
        handler = VLLMResponseHandler()
        result = backend._build_final_response(
            request, request_info, mock_final, ctx, handler, stream=False
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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_extract_text_from_content_list_no_text_blocks_returns_empty(self, backend):
        """
        List without text blocks -> "".
        ## WRITTEN BY AI ##
        """
        content = [{"type": "image_url", "url": "x"}]
        assert backend._extract_text_from_content(content) == ""

    @pytest.mark.smoke
    def test_extract_text_from_content_fallback_str(self, backend):
        """
        Non-str non-list -> str(content).
        ## WRITTEN BY AI ##
        """
        assert backend._extract_text_from_content(123) == "123"

    @pytest.mark.smoke
    def test_extract_text_from_content_none_returns_empty(self, backend):
        """
        content None -> "".
        ## WRITTEN BY AI ##
        """
        assert backend._extract_text_from_content(None) == ""


class TestVLLMExtractAudioFromRequest:
    """
    _extract_audio_from_request: empty, no file, tuple, raw bytes, non-bytes, invalid.
    """

    @pytest.mark.smoke
    def test_extract_audio_empty_files_raises(self, backend):
        """
        Empty files -> ValueError.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="Audio request must include audio file"):
            backend._extract_audio_from_request({})

    @pytest.mark.smoke
    def test_extract_audio_no_valid_file_raises(self, backend):
        """
        No tuple/bytes in values -> ValueError.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="Audio request must include audio file"):
            backend._extract_audio_from_request({"k": "not-bytes"})

    @pytest.mark.smoke
    def test_extract_audio_tuple_returns_bytes_and_mimetype(self, backend):
        """
        Tuple (name, bytes, mimetype) -> return bytes and mimetype.
        ## WRITTEN BY AI ##
        """
        data = b"audio-bytes"
        out = backend._extract_audio_from_request({"f": ("x.wav", data, "audio/wav")})
        assert out == (data, "audio/wav")

    @pytest.mark.smoke
    def test_extract_audio_tuple_two_elements_default_mimetype(self, backend):
        """
        Tuple (name, bytes) -> mimetype default audio/wav.
        ## WRITTEN BY AI ##
        """
        data = b"audio"
        out = backend._extract_audio_from_request({"f": ("x.wav", data)})
        assert out == (data, "audio/wav")

    @pytest.mark.smoke
    def test_extract_audio_raw_bytes_returns_wav_tuple(self, backend):
        """
        Raw bytes value -> ("audio.wav", bytes, "audio/wav").
        ## WRITTEN BY AI ##
        """
        data = b"raw"
        out = backend._extract_audio_from_request({"f": data})
        assert out == (data, "audio/wav")

    @pytest.mark.smoke
    def test_extract_audio_tuple_non_bytes_at_index_1_raises(self, backend):
        """
        Tuple with non-bytes at index 1 -> ValueError.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="Expected bytes for audio data"):
            backend._extract_audio_from_request(
                {"f": ("x.wav", "not-bytes", "audio/wav")}
            )


class TestVLLMExtractPromptEdgeCases:
    """
    _extract_prompt: text/chat empty -> ValueError; audio optional prompt.
    """

    @pytest.mark.smoke
    def test_extract_prompt_text_empty_prompt_raises(self, backend):
        """
        Text mode: body prompt missing or empty -> ValueError.
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="Text request must include 'prompt'"):
            backend._extract_prompt({"prompt": ""}, "text")
        with pytest.raises(ValueError, match="Text request must include 'prompt'"):
            backend._extract_prompt({}, "text")

    @pytest.mark.smoke
    def test_extract_prompt_chat_empty_messages_raises(self, backend):
        """
        Chat mode: messages empty -> ValueError.
        ## WRITTEN BY AI ##
        """
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            backend_plain = VLLMPythonBackend(
                model="test-model", request_format="plain"
            )
            backend_plain._engine = None
        with pytest.raises(ValueError, match="Chat request must include 'messages'"):
            backend_plain._extract_prompt({"messages": []}, "chat")

    @pytest.mark.smoke
    def test_extract_prompt_audio_returns_prompt_or_empty(self, backend):
        """
        Audio mode: return body.get("prompt", "") (optional prompt).
        ## WRITTEN BY AI ##
        """
        assert backend._extract_prompt({"prompt": "caption"}, "audio") == "caption"
        assert backend._extract_prompt({}, "audio") == ""
        assert backend._extract_prompt({"prompt": ""}, "audio") == ""


class TestVLLMConvertVllmOutputToOpenAIFormat:
    """
    _convert_vllm_output_to_openai_format: OpenAI-shaped dict with usage and id.
    """

    @pytest.mark.smoke
    def test_convert_vllm_output_to_openai_format_includes_usage_and_id(self, backend):
        """
        Builds dict with mode-specific payload, usage, and id from output.request_id.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.prompt_token_ids = [1, 2]
        mock_out.outputs = [Mock(token_ids=[3], text="hi")]
        mock_out.request_id = "vllm-req-123"
        result = backend._convert_vllm_output_to_openai_format(mock_out, "chat")
        assert result["choices"] == [
            {"message": {"content": "hi", "role": "assistant"}}
        ]
        assert result["usage"] == {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        }
        assert result["id"] == "vllm-req-123"

    @pytest.mark.smoke
    def test_convert_vllm_output_to_openai_format_no_request_id_omits_id(self, backend):
        """
        When output has no request_id or falsy, id is omitted.
        ## WRITTEN BY AI ##
        """
        mock_out = Mock()
        mock_out.prompt_token_ids = []
        mock_out.outputs = [Mock(token_ids=[], text="")]
        mock_out.request_id = None
        result = backend._convert_vllm_output_to_openai_format(mock_out, "text")
        assert "choices" in result
        assert "usage" in result
        assert "id" not in result


class TestVLLMResolveNonStream:
    """
    resolve() with stream=False: one generate yield, compile_non_streaming.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_resolve_non_stream_yields_one_response(self, backend):
        """
        stream=False; mock generate yields once; one response via compile_non_streaming.
        ## WRITTEN BY AI ##
        """
        out = Mock()
        out.text = "done"
        out.token_ids = [1, 2, 3]
        out.finish_reason = "stop"
        req_out = Mock()
        req_out.prompt_token_ids = [1, 2]
        req_out.outputs = [out]
        req_out.request_id = "r1"

        async def one_yield(prompt, sampling_params, request_id):
            yield req_out

        backend._engine = Mock()
        backend._engine.generate = one_yield
        request = Mock(spec=GenerationRequest)
        request.columns = None
        request.arguments = Mock()
        request.arguments.body = {"messages": [{"role": "user", "content": "Hi"}]}
        request.arguments.stream = False
        request.arguments.files = {}
        request.output_metrics = UsageMetrics()
        request.request_id = "req-1"
        request.arguments.model_dump_json = Mock(return_value="{}")
        request_info = RequestInfo()

        results = []
        with patch.object(
            backend, "_extract_prompt", return_value="User: Hi\nAssistant: "
        ):
            async for response, info in backend.resolve(request, request_info):
                results.append((response, info))

        assert len(results) == 1
        resp, info = results[0]
        assert isinstance(resp, GenerationResponse)
        assert resp.text == "done"
        assert info is request_info


class TestVLLMResolveValidation:
    """
    resolve(): history -> NotImplementedError; _engine None -> RuntimeError.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
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
    @pytest.mark.smoke
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
    @pytest.mark.smoke
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

        with patch.object(
            backend, "_extract_prompt", return_value="User: Hi\nAssistant: "
        ), pytest.raises(asyncio.CancelledError):
            await collect()
        assert len(results) == 1
        assert results[0][0].text == "partial"


class TestVLLMResolveAudioError:
    """
    resolve(): At most 0 audio / audio(s) may be provided -> RuntimeError with guidance.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_resolve_audio_error_wraps_with_guidance(self, backend):
        """
        'At most 0 audio' -> RuntimeError with audio-capable model guidance.
        ## WRITTEN BY AI ##
        """

        async def mock_generate_raise(*args, **kwargs):
            raise ValueError("At most 0 audio(s) may be provided.")
            yield  # makes this an async generator so async for raises

        with patch("guidellm.extras.audio._decode_audio") as mock_decode:
            mock_decode.return_value = Mock(data=np.array([0.0]))
            backend._engine = Mock()
            backend._engine.generate = mock_generate_raise
            request = Mock(spec=GenerationRequest)
            request.columns = None
            request.arguments = Mock()
            request.arguments.body = {}
            request.arguments.stream = False
            request.arguments.files = {"f": ("audio.wav", b"audio", "audio/wav")}
            request.output_metrics = UsageMetrics()
            request.request_id = "r1"
            request_info = RequestInfo()
            with pytest.raises(RuntimeError) as exc_info:
                async for _ in backend.resolve(request, request_info):
                    pass
            assert "does not support audio" in str(exc_info.value)
            assert "audio-capable model" in str(exc_info.value)


class TestVLLMResolveAudioMode:
    """
    resolve() audio: mock _decode_audio and generate; one response.
    """

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_resolve_audio_mode_yields_one_response(self, backend):
        """
        Mode audio; mock _decode_audio and engine.generate; assert one response.
        ## WRITTEN BY AI ##
        """
        mock_audio_array = np.array([0.0, 0.1])
        mock_decode_result = Mock()
        mock_decode_result.data = mock_audio_array

        out = Mock()
        out.text = "transcribed"
        out.token_ids = [1, 2]
        out.finish_reason = "stop"
        req_out = Mock()
        req_out.prompt_token_ids = [1]
        req_out.outputs = [out]
        req_out.request_id = "r1"

        async def mock_generate(prompt, sampling_params, request_id):
            assert isinstance(prompt, dict)
            assert "prompt" in prompt
            assert "multi_modal_data" in prompt
            assert "audio" in prompt["multi_modal_data"]
            yield req_out

        with patch(
            "guidellm.extras.audio._decode_audio",
            return_value=mock_decode_result,
        ):
            backend._engine = Mock()
            backend._engine.generate = mock_generate
            request = Mock(spec=GenerationRequest)
            request.columns = None
            request.arguments = Mock()
            request.arguments.body = {"prompt": ""}
            request.arguments.stream = False
            request.arguments.files = {"f": ("audio.wav", b"audio-bytes", "audio/wav")}
            request.output_metrics = UsageMetrics()
            request.request_id = "r1"
            request.arguments.model_dump_json = Mock(return_value="{}")
            request_info = RequestInfo()
            results = []
            async for response, info in backend.resolve(request, request_info):
                results.append((response, info))
            assert len(results) == 1
            assert results[0][0].text == "transcribed"
