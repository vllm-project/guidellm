# import json
# from collections.abc import AsyncIterable
# from typing import Any, Literal, Optional
# from unittest.mock import MagicMock, patch

# import httpx
# import pytest
# import respx

# from guidellm.backend import ResponseSummary, StreamingTextResponse

# from .mock_backend import MockBackend


# @pytest.fixture
# def mock_auto_tokenizer():
#     with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:

#         def _fake_tokenize(text: str) -> list[int]:
#             tokens = text.split()
#             return [0] * len(tokens)

#         mock_tokenizer = MagicMock()
#         mock_tokenizer.tokenize = MagicMock(side_effect=_fake_tokenize)
#         mock_from_pretrained.return_value = mock_tokenizer
#         yield mock_tokenizer


# @pytest.fixture
# def mock_backend(request):
#     params = request.param if hasattr(request, "param") else {}
#     kwargs = {}

#     for key in ("model", "target", "iter_delay"):
#         if key in params:
#             kwargs[key] = params[key]

#     return MockBackend(**kwargs)


# class MockCompletionsIter(AsyncIterable):
#     def __init__(
#         self,
#         type_: Literal["text", "chat"],
#         prompt: str,
#         output_token_count: Optional[int],
#         target: Optional[str] = None,
#         model: Optional[str] = None,
#         iter_delay: Optional[float] = None,
#     ):
#         self._type = type_
#         self._backend = MockBackend(
#             model=model,
#             target=target,
#             iter_delay=iter_delay,
#         )
#         self._prompt = prompt
#         self._output_token_count = output_token_count

#     async def __aiter__(self):
#         async for token_iter in (
#             self._backend.text_completions(
#                 prompt=self._prompt, output_token_count=self._output_token_count
#             )
#             if self._type == "text"
#             else self._backend.chat_completions(
#                 content=self._prompt, output_token_count=self._output_token_count
#             )
#         ):
#             if (
#                 isinstance(token_iter, StreamingTextResponse)
#                 and token_iter.type_ == "start"
#             ):
#                 continue

#             data: dict[str, Any]

#             if isinstance(token_iter, StreamingTextResponse):
#                 if self._type == "text":
#                     data = {
#                         "choices": [
#                             {
#                                 "index": token_iter.iter_count,
#                                 "text": token_iter.delta,
#                             }
#                         ]
#                     }
#                 elif self._type == "chat":
#                     data = {
#                         "choices": [
#                             {
#                                 "index": token_iter.iter_count,
#                                 "delta": {"content": token_iter.delta},
#                             }
#                         ]
#                     }
#                 else:
#                     raise ValueError("Invalid type for mock completions")
#             elif isinstance(token_iter, ResponseSummary):
#                 data = {
#                     "usage": {
#                         "prompt_tokens": (
#                             len(self._prompt.split()) + self._prompt.count(" ")
#                         ),
#                         "completion_tokens": token_iter.response_output_tokens,
#                     }
#                 }
#             else:
#                 raise ValueError("Invalid token_iter type")

#             yield f"data: {json.dumps(data)}\n".encode()

#         yield b"data: [DONE]\n"


# @pytest.fixture
# def httpx_openai_mock(request):
#     params = request.param if hasattr(request, "param") else {}
#     model = params.get("model", "mock-model")
#     target = params.get("target", "http://target.mock")
#     iter_delay = params.get("iter_delay", None)

#     with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock_router:

#         async def _mock_completions_response(request) -> AsyncIterable[str]:
#             headers = request.headers
#             payload = json.loads(request.content)

#             assert headers["Content-Type"] == "application/json"
#             assert payload["model"] == model
#             assert payload["stream"] is True
#             assert payload["stream_options"] == {"include_usage": True}
#             assert payload["prompt"] is not None
#             assert len(payload["prompt"]) > 0
#             assert payload["max_completion_tokens"] > 0
#             assert payload["max_tokens"] > 0

#             return httpx.Response(  # type: ignore
#                 200,
#                 stream=MockCompletionsIter(  # type: ignore
#                     type_="text",
#                     prompt=payload["prompt"],
#                     output_token_count=(
#                         payload["max_completion_tokens"]
#                         if payload.get("ignore_eos", False)
#                         else None
#                     ),
#                     target=target,
#                     model=model,
#                     iter_delay=iter_delay,
#                 ),
#             )

#         async def _mock_chat_completions_response(request):
#             headers = request.headers
#             payload = json.loads(request.content)

#             assert headers["Content-Type"] == "application/json"
#             assert payload["model"] == model
#             assert payload["stream"] is True
#             assert payload["stream_options"] == {"include_usage": True}
#             assert payload["messages"] is not None
#             assert len(payload["messages"]) > 0
#             assert payload["max_completion_tokens"] > 0
#             assert payload["max_tokens"] > 0

#             return httpx.Response(  # type: ignore
#                 200,
#                 stream=MockCompletionsIter(  # type: ignore
#                     type_="chat",
#                     prompt=payload["messages"][0]["content"],
#                     output_token_count=(
#                         payload["max_completion_tokens"]
#                         if payload.get("ignore_eos", False)
#                         else None
#                     ),
#                     target=target,
#                     model=model,
#                     iter_delay=iter_delay,
#                 ),
#             )

#         mock_router.route(method="GET", path="/v1/models").mock(
#             return_value=httpx.Response(
#                 200, json={"data": [{"id": model} if model else {"id": "mock-model"}]}
#             )
#         )
#         mock_router.route(method="POST", path="/v1/completions").mock(
#             side_effect=_mock_completions_response  # type: ignore
#         )
#         mock_router.route(method="POST", path="/v1/chat/completions").mock(
#             side_effect=_mock_chat_completions_response
#         )

#         yield mock_router
