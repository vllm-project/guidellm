#!/usr/bin/env python3
"""
sanic_server.py

Simple Sanic-based mock server that implements common OpenAI / vLLM-compatible routes:
- GET  /                    : health
- GET  /v1/models           : list models
- POST /v1/chat/completions : chat completions (supports streaming via ?stream=true)
- POST /v1/completions      : classic completions
- POST /v1/embeddings       : fake embeddings
- POST /v1/moderations      : fake moderation

Usage:
    pip install sanic==25.3.0 or latest
Command:
    python sanic_server.py or \
    python sanic_server.py --host=0.0.0.0 --port=8000 --workers=1 --debug
"""

import argparse
import asyncio
import json
import random

from sanic import Sanic
from sanic.request import Request
from sanic.response import ResponseStream
from sanic.response import json as sjson

app = Sanic("sanic_server")


# ---------- utils ----------


def fake_tokenize(text: str) -> list[str]:
    # crude whitespace tokenizer for token counting
    return text.strip().split()


def make_choice_text(prompt: str) -> str:
    # Very simple deterministic reply generator
    # Echo some truncated summary for testing
    tail = prompt.strip()[:120]
    return f"Mock reply summarizing: {tail}"


def now_ms() -> int:
    return int(asyncio.get_event_loop().time() * 1000)


# ---------- routes ----------


@app.get("/")
async def health(request: Request):
    return sjson({"ok": True, "msg": "mock openai/vllm server"})


@app.get("/v1/models")
async def list_models(request: Request):
    # minimal model list
    models = [
        {"id": "mock-qwen-2.5", "object": "model", "owned_by": "mock"},
        {"id": "facebook/opt-125m", "object": "model", "owned_by": "mock"},
    ]
    return sjson({"object": "list", "data": models})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Supports:
      - JSON body with 'messages' (OpenAI format)
      - query param stream=true or JSON {'stream': true}
          => responds with text/event-stream chunks containing 'data: {json}\n\n'
    """
    body = request.json or {}
    stream_mode = False
    if request.args.get("stream", "false").lower() == "true":
        stream_mode = True

    messages = body.get("messages", [])
    prompt_text = ""
    if isinstance(messages, list) and messages:
        # approximate prompt as concatenation of last user message(s)
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                prompt_text += content + " "

    # build a deterministic reply
    reply = make_choice_text(prompt_text or "hello")
    prompt_tokens = len(fake_tokenize(prompt_text))
    completion_tokens = len(fake_tokenize(reply))

    # create response object (non-streaming)
    def make_response_obj():
        return {
            "id": f"cmpl-mock-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": now_ms(),
            "model": body.get("model", "mock-qwen-2.5"),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }

    if not stream_mode:
        return sjson(make_response_obj())

    # streaming mode: SSE-style chunks with 'data: <json>\n\n'
    async def streaming_fn(resp):
        # send an initial "response.start" like chunk
        await resp.write(
            f"data: \
                    {json.dumps({'type': 'response.start', 'created': now_ms()})}\n\n"
        )

        # simulate token-by-token streaming
        tokens = fake_tokenize(reply)
        chunk_text = ""
        for i, tk in enumerate(tokens):
            chunk_text += tk + (" " if i < len(tokens) - 1 else "")
            chunk_payload = {
                "id": f"cmpl-mock-{random.randint(1000, 9999)}",
                "object": "chat.completion.chunk",
                "created": now_ms(),
                "model": body.get("model", "mock-qwen-2.5"),
                "choices": [
                    {
                        "delta": {"content": tk + (" " if i < len(tokens) - 1 else "")},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
            # write chunk
            await resp.write(f"data: {json.dumps(chunk_payload)}\n\n")
            # small jitter between tokens
            await asyncio.sleep(0.03)
        # final done event
        done_payload = {"type": "response.done", "created": now_ms()}
        await resp.write(f"data: {json.dumps(done_payload)}\n\n")

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    return ResponseStream(streaming_fn, headers=headers)


@app.post("/v1/completions")
async def completions(request: Request):
    body = request.json or {}
    prompt = body.get("prompt") or (
        body.get("messages")
        and " ".join([m.get("content", "") for m in body.get("messages", [])])
    )
    if not prompt:
        prompt = "hello"
    # optional max_tokens
    max_tokens = int(body.get("max_tokens", 64))
    reply = make_choice_text(prompt)
    tokenized = fake_tokenize(reply)[:max_tokens]
    text_out = " ".join(tokenized)

    prompt_tokens = len(fake_tokenize(prompt))
    completion_tokens = len(tokenized)

    resp = {
        "id": f"cmpl-mock-{random.randint(1000, 9999)}",
        "object": "text_completion",
        "created": now_ms(),
        "model": body.get("model", "mock-qwen-2.5"),
        "choices": [{"text": text_out, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    # simulate a small server-side latency
    await asyncio.sleep(0.01)
    return sjson(resp)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = request.json or {}
    inputs = body.get("input") or body.get("inputs") or []
    if isinstance(inputs, str):
        inputs = [inputs]
    # produce deterministic embedding length 16
    dim = int(request.args.get("dim", body.get("dim", 16)))
    out = []
    for i, txt in enumerate(inputs):
        # make pseudo-random but deterministic numbers based on hash
        seed = abs(hash(txt)) % (10**8)
        random.seed(seed)
        vec = [round((random.random() - 0.5), 6) for _ in range(dim)]
        out.append({"object": "embedding", "embedding": vec, "index": i})
    return sjson({"data": out, "model": body.get("model", "mock-embed-1")})


@app.post("/v1/moderations")
async def moderations(request: Request):
    body = request.json or {}
    input_text = body.get("input") or ""
    # super naive: classify as 'flagged' if contains "bad"
    flagged = "bad" in input_text.lower()
    return sjson(
        {
            "id": "mod-mock-1",
            "model": body.get("model", "mock-moderation"),
            "results": [{"flagged": flagged}],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sanic_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        workers=args.workers,
        access_log=False,
    )
