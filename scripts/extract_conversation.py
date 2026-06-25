#!/usr/bin/env python3
"""Extract and pretty-print turn-by-turn conversation data from benchmarks.json.

Reads a benchmark report and prints each request's conversation history and
response in a human-readable format, with tool calls rendered as YAML.

Usage::

    python scripts/extract_conversation.py [path/to/benchmarks.json]
    python scripts/extract_conversation.py benchmarks.json --limit 5
    python scripts/extract_conversation.py benchmarks.json -n 30
    python scripts/extract_conversation.py benchmarks.json --turn 3

If no path is given, defaults to ``benchmarks.json`` in the current directory.
The ``--limit`` / ``-n`` flag caps the number of requests printed (default: 15).
The ``--turn`` / ``-t`` flag prints only the request at the given 1-based index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def extract_conversations(
    path: Path,
    *,
    limit: int = 15,
    turn: int | None = None,
) -> None:
    """Extract and print conversations from a benchmark JSON file.

    :param path: Path to the benchmark JSON file.
    :param limit: Maximum number of requests to print.
    :param turn: If set, print only the request at this 1-based index.
    """
    data = json.loads(path.read_text())

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return

    benchmark = benchmarks[0]
    reqs = benchmark["requests"]["successful"]

    # Sort by message count so turns appear in conversation order
    reqs = sorted(
        reqs,
        key=lambda r: len(
            json.loads(r["request_args"]).get("body", {}).get("messages", [])
            if r.get("request_args")
            else []
        ),
    )

    total = len(reqs)

    # Select a single turn or apply the limit
    truncated = False
    if turn is not None:
        if turn < 1 or turn > total:
            print(f"  Error: turn {turn} out of range (1-{total})")
            return
        reqs = [reqs[turn - 1]]
        start_index = turn - 1
    else:
        truncated = total > limit
        reqs = reqs[:limit]
        start_index = 0

    for ri, req in enumerate(reqs):
        _print_request(req, turn_number=start_index + ri + 1, total=total)

    if truncated:
        print(f"\n  ... ({total - limit} more requests not shown, use -n to adjust)")


def _print_request(req: dict[str, Any], *, turn_number: int, total: int) -> None:
    """Print a single request's conversation and response.

    :param req: Request dict from the benchmark data.
    :param turn_number: 1-based turn index for display.
    :param total: Total number of requests in the benchmark.
    """
    args = json.loads(req["request_args"]) if req.get("request_args") else {}
    body = args.get("body", {})
    msgs = body.get("messages", [])

    print(
        f" Turn {turn_number} of {total}"
        f"    (prompt_tokens: {req.get('prompt_tokens')}"
        f" | output_tokens: {req.get('output_tokens')})"
    )
    if body.get("tool_choice"):
        print(f"  tool_choice: {body['tool_choice']}")

    for m in msgs:
        _print_message(m)

    # Response produced by this turn
    tc = req.get("tool_calls")
    output = req.get("output")

    if tc:
        print("  [RESPONSE - TOOL CALLS]")
        for t in tc:
            _print_tool_call_yaml(t)
    elif output:
        print("  [RESPONSE]")
        for line in output.split("\n"):
            print(f"    {line}")
    else:
        print("  [NO RESPONSE]")


def _print_message(m: dict[str, Any]) -> None:
    """Print a single conversation message to stdout.

    :param m: Message dict with role, content, and optional tool_calls.
    """
    role = m["role"].upper()
    content = m.get("content")
    tool_calls = m.get("tool_calls")
    tool_call_id = m.get("tool_call_id")

    print(f"  [{role}]")

    if tool_call_id:
        print(f"    tool_call_id: {tool_call_id}")

    if content is not None:
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    print(f"    {part['text']}")
        else:
            print(f"    {content}")

    if tool_calls:
        for tc in tool_calls:
            _print_tool_call_yaml(tc)


def _print_tool_call_yaml(tc: dict[str, Any]) -> None:
    """Print a tool call as indented YAML.

    :param tc: Tool call dict with id, function name, and arguments.
    """
    fn = tc["function"]
    try:
        tc_args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
    except (json.JSONDecodeError, TypeError):
        tc_args = fn.get("arguments", "")

    tc_yaml = {
        "id": tc["id"],
        "function": fn["name"],
        "arguments": tc_args,
    }
    dumped = yaml.dump(tc_yaml, default_flow_style=False, width=100).rstrip()
    for line in dumped.split("\n"):
        print(f"    {line}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :returns: Parsed namespace with path and limit attributes.
    """
    parser = argparse.ArgumentParser(
        description="Extract and pretty-print conversations from a benchmark JSON.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="benchmarks.json",
        help="Path to the benchmark JSON file (default: benchmarks.json)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=15,
        help="Maximum number of requests to print (default: 15)",
    )
    parser.add_argument(
        "-t",
        "--turn",
        type=int,
        default=None,
        help="Print only the request at this 1-based turn index",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ns = _parse_args()
    extract_conversations(Path(ns.path), limit=ns.limit, turn=ns.turn)
