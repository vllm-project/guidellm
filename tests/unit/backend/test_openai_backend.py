import time

import pytest

from guidellm.backend import OpenAIHTTPBackend, ResponseSummary, StreamingTextResponse
from guidellm.settings import settings


@pytest.mark.smoke
def test_openai_http_backend_default_initialization():
    backend = OpenAIHTTPBackend()
    assert backend.target == settings.openai.base_url
    assert backend.model is None
    assert backend.headers.get("Authorization") == settings.openai.bearer_token
    assert backend.organization == settings.openai.organization
    assert backend.project == settings.openai.project
    assert backend.timeout == settings.request_timeout
    assert backend.http2 is True
    assert backend.follow_redirects is True
    assert backend.max_output_tokens == settings.openai.max_output_tokens
    assert backend.extra_query is None


@pytest.mark.smoke
def test_openai_http_backend_intialization():
    backend = OpenAIHTTPBackend(
        target="http://test-target",
        model="test-model",
        api_key="test-key",
        organization="test-org",
        project="test-proj",
        timeout=10,
        http2=False,
        follow_redirects=False,
        max_output_tokens=100,
        extra_query={"foo": "bar"},
    )
    assert backend.target == "http://test-target"
    assert backend.model == "test-model"
    assert backend.headers.get("Authorization") == "Bearer test-key"
    assert backend.organization == "test-org"
    assert backend.project == "test-proj"
    assert backend.timeout == 10
    assert backend.http2 is False
    assert backend.follow_redirects is False
    assert backend.max_output_tokens == 100
    assert backend.extra_query == {"foo": "bar"}


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_available_models(httpx_openai_mock):
    backend = OpenAIHTTPBackend(target="http://target.mock")
    models = await backend.available_models()
    assert models == ["mock-model"]


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_validate(httpx_openai_mock):
    backend = OpenAIHTTPBackend(target="http://target.mock", model="mock-model")
    await backend.validate()

    backend = OpenAIHTTPBackend(target="http://target.mock")
    await backend.validate()
    assert backend.model == "mock-model"

    backend = OpenAIHTTPBackend(target="http://target.mock", model="invalid-model")
    with pytest.raises(ValueError):
        await backend.validate()


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_text_completions(httpx_openai_mock):
    backend = OpenAIHTTPBackend(target="http://target.mock", model="mock-model")

    index = 0
    final_resp = None
    async for response in backend.text_completions("Test Prompt", request_id="test-id"):
        assert isinstance(response, (StreamingTextResponse, ResponseSummary))

        if index == 0:
            assert isinstance(response, StreamingTextResponse)
            assert response.type_ == "start"
            assert response.iter_count == 0
            assert response.delta == ""
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == "test-id"
        elif not isinstance(response, ResponseSummary):
            assert response.type_ == "iter"
            assert response.iter_count == index
            assert len(response.delta) > 0
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == "test-id"
        else:
            assert not final_resp
            final_resp = response
            assert isinstance(response, ResponseSummary)
            assert len(response.value) > 0
            assert response.request_args is not None
            assert response.iterations > 0
            assert response.start_time > 0
            assert response.end_time == pytest.approx(time.time(), abs=0.01)
            assert response.request_prompt_tokens is None
            assert response.request_output_tokens is None
            assert response.response_prompt_tokens == 3
            assert response.response_output_tokens > 0  # type: ignore
            assert response.request_id == "test-id"

        index += 1
    assert final_resp


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_text_completions_counts(httpx_openai_mock):
    backend = OpenAIHTTPBackend(
        target="http://target.mock",
        model="mock-model",
        max_output_tokens=100,
    )
    final_resp = None

    async for response in backend.text_completions(
        "Test Prompt", request_id="test-id", prompt_token_count=3, output_token_count=10
    ):
        final_resp = response

    assert final_resp
    assert isinstance(final_resp, ResponseSummary)
    assert len(final_resp.value) > 0
    assert final_resp.request_args is not None
    assert final_resp.request_prompt_tokens == 3
    assert final_resp.request_output_tokens == 10
    assert final_resp.response_prompt_tokens == 3
    assert final_resp.response_output_tokens == 10
    assert final_resp.request_id == "test-id"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_chat_completions(httpx_openai_mock):
    backend = OpenAIHTTPBackend(target="http://target.mock", model="mock-model")

    index = 0
    final_resp = None
    async for response in backend.chat_completions("Test Prompt", request_id="test-id"):
        assert isinstance(response, (StreamingTextResponse, ResponseSummary))

        if index == 0:
            assert isinstance(response, StreamingTextResponse)
            assert response.type_ == "start"
            assert response.iter_count == 0
            assert response.delta == ""
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == "test-id"
        elif not isinstance(response, ResponseSummary):
            assert response.type_ == "iter"
            assert response.iter_count == index
            assert len(response.delta) > 0
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == "test-id"
        else:
            assert not final_resp
            final_resp = response
            assert isinstance(response, ResponseSummary)
            assert len(response.value) > 0
            assert response.request_args is not None
            assert response.iterations > 0
            assert response.start_time > 0
            assert response.end_time == pytest.approx(time.time(), abs=0.01)
            assert response.request_prompt_tokens is None
            assert response.request_output_tokens is None
            assert response.response_prompt_tokens == 3
            assert response.response_output_tokens > 0  # type: ignore
            assert response.request_id == "test-id"

        index += 1

    assert final_resp


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openai_http_backend_chat_completions_counts(httpx_openai_mock):
    backend = OpenAIHTTPBackend(
        target="http://target.mock",
        model="mock-model",
        max_output_tokens=100,
    )
    final_resp = None

    async for response in backend.chat_completions(
        "Test Prompt", request_id="test-id", prompt_token_count=3, output_token_count=10
    ):
        final_resp = response

    assert final_resp
    assert isinstance(final_resp, ResponseSummary)
    assert len(final_resp.value) > 0
    assert final_resp.request_args is not None
    assert final_resp.request_prompt_tokens == 3
    assert final_resp.request_output_tokens == 10
    assert final_resp.response_prompt_tokens == 3
    assert final_resp.response_output_tokens == 10
    assert final_resp.request_id == "test-id"
