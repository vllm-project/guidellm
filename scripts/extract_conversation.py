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
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# ANSI color helpers — disabled when stdout is not a terminal
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence when color is enabled."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _style(code: str) -> Callable[[str], str]:
    """Return a function that wraps text in the given ANSI code."""
    return lambda t: _c(code, t)


# Semantic color shortcuts
_bold = _style("1")
_dim = _style("2")
_red = _style("31")
_green = _style("32")
_yellow = _style("33")
_blue = _style("34")
_magenta = _style("35")
_cyan = _style("36")
_bright_green = _style("1;32")
_bright_yellow = _style("1;33")
_bright_cyan = _style("1;36")

_ROLE_COLORS: dict[str, Callable[[str], str]] = {
    "SYSTEM": _magenta,
    "USER": _green,
    "ASSISTANT": _cyan,
    "TOOL": _yellow,
}


def _role_color(role: str, text: str) -> str:
    """Apply the role-specific color, falling back to bold."""
    fn = _ROLE_COLORS.get(role, _bold)
    return fn(text)


def _is_multi_agent(reqs: list[dict[str, Any]]) -> bool:
    """Check whether the request list contains multiple distinct agents.

    :param reqs: List of request stat dicts from the benchmark.
    :return: True if more than one non-default agent_id is present.
    """
    agents: set[str] = set()
    for req in reqs:
        info = req.get("info", {})
        agent_id = info.get("agent_id")
        if agent_id:
            agents.add(agent_id)
    return len(agents) > 1


def _print_multi_agent_summary(reqs: list[dict[str, Any]]) -> None:
    """Print a summary header listing distinct agents and graph count.

    :param reqs: List of request stat dicts from the benchmark.
    """
    agents: set[str] = set()
    graphs: set[str] = set()
    for req in reqs:
        info = req.get("info", {})
        agent_id = info.get("agent_id")
        graph_id = info.get("graph_id") or info.get("conversation_id")
        if agent_id:
            agents.add(agent_id)
        if graph_id:
            graphs.add(graph_id)

    print(f"  {_bright_yellow('[MULTI-AGENT BENCHMARK]')}")
    print(f"    {_dim('agents:')} {_yellow(str(sorted(agents)))}")
    print(f"    {_dim('conversations:')} {_yellow(str(len(graphs)))}")
    print()


def _build_node_index(reqs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lookup from node_id to request info for DAG traversal.

    Indexes all requests by their node_id (scoped per graph_id) so that
    ancestor chains can be walked back through parent_node_ids.

    :param reqs: List of request stat dicts from the benchmark.
    :return: Mapping of ``(graph_id, node_id)`` composite key to request info.
    """
    index: dict[str, dict[str, Any]] = {}
    for req in reqs:
        info = req.get("info", {})
        node_id = info.get("node_id")
        graph_id = info.get("graph_id") or info.get("conversation_id") or ""
        if node_id:
            index[f"{graph_id}:{node_id}"] = info
    return index


def _walk_history_path(
    info: dict[str, Any],
    node_index: dict[str, dict[str, Any]],
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Walk back through parent_node_ids to build the full ancestor chain.

    Returns the path from root to the current node. Each entry is
    ``(node_id, agent_id, merge_parents)`` where ``merge_parents`` lists
    any additional parents beyond the primary one (these contribute their
    last value to history assembly).

    :param info: RequestInfo dict for the current node.
    :param node_index: Node index built by :func:`_build_node_index`.
    :return: Ordered path from root ancestor to current node (inclusive).
    """
    graph_id = info.get("graph_id") or info.get("conversation_id") or ""
    node_id = info.get("node_id")

    if not node_id:
        return []

    # Walk backwards collecting ancestors via primary parent
    path: list[tuple[str, str, list[tuple[str, str]]]] = []
    visited: set[str] = set()
    current = info

    while current:
        cur_node = current.get("node_id", "?")
        cur_agent = current.get("agent_id") or "?"
        key = f"{graph_id}:{cur_node}"

        if key in visited:
            break
        visited.add(key)

        # Identify merge parents (non-primary parents that feed last value)
        parents = current.get("parent_node_ids", [])
        merge_parents: list[tuple[str, str]] = []
        for extra_pid in parents[1:]:
            extra_key = f"{graph_id}:{extra_pid}"
            extra_info = node_index.get(extra_key)
            extra_agent = extra_info.get("agent_id", "?") if extra_info else "?"
            merge_parents.append((extra_pid, extra_agent))

        path.append((cur_node, cur_agent, merge_parents))

        # Walk to first parent (primary history source)
        if not parents:
            break
        parent_key = f"{graph_id}:{parents[0]}"
        current = node_index.get(parent_key)

    path.reverse()
    return path


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
    multi_agent = _is_multi_agent(reqs)
    node_index = _build_node_index(reqs) if multi_agent else {}

    if multi_agent:
        _print_multi_agent_summary(reqs)

    # Select a single turn or apply the limit
    truncated = False
    if turn is not None:
        if turn < 1 or turn > total:
            print(f"  {_red(f'Error: turn {turn} out of range (1-{total})')}")
            return
        reqs = [reqs[turn - 1]]
        start_index = turn - 1
    else:
        truncated = total > limit
        reqs = reqs[:limit]
        start_index = 0

    for ri, req in enumerate(reqs):
        _print_request(
            req,
            turn_number=start_index + ri + 1,
            total=total,
            show_agent_context=multi_agent,
            node_index=node_index,
        )

    if truncated:
        print(
            f"\n  {_dim(f'... ({total - limit} more requests not shown, use -n to adjust)')}"
        )


def _print_agent_context(
    req: dict[str, Any],
    node_index: dict[str, dict[str, Any]],
) -> None:
    """Print subagent/graph context and full history path for a request.

    Shows the node identity line, parents, and the complete ancestor chain
    so that the assembled conversation history can be verified.

    :param req: Request dict from the benchmark data.
    :param node_index: Node index for DAG traversal.
    """
    info = req.get("info", {})
    agent_id = info.get("agent_id")
    node_id = info.get("node_id")
    parent_node_ids = info.get("parent_node_ids", [])
    graph_id = info.get("graph_id") or info.get("conversation_id")

    parts: list[str] = []
    if agent_id:
        parts.append(f"{_dim('agent:')} {_blue(agent_id)}")
    if node_id:
        parts.append(f"{_dim('node:')} {_blue(node_id)}")
    if parent_node_ids:
        parts.append(f"{_dim('parents:')} {_blue(str(parent_node_ids))}")
    if graph_id:
        parts.append(f"{_dim('graph:')} {_blue(graph_id)}")

    if parts:
        print(f"    {_dim(' | ').join(parts)}")

    # Show the full history path (ancestor chain leading to this node)
    history_path = _walk_history_path(info, node_index)
    if len(history_path) > 1:
        steps: list[str] = []
        for nid, agent, merge_parents in history_path:
            step = f"{_blue(nid)} {_dim(f'({agent})')}"
            for mid, ma in merge_parents:
                step += f" {_dim('+')} {_blue(mid)} {_dim(f'({ma})')}"
            steps.append(step)
        print(f"    {_dim('history:')} {_dim(' -> ').join(steps)}")


def _print_request(
    req: dict[str, Any],
    *,
    turn_number: int,
    total: int,
    show_agent_context: bool = False,
    node_index: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print a single request's conversation and response.

    :param req: Request dict from the benchmark data.
    :param turn_number: 1-based turn index for display.
    :param total: Total number of requests in the benchmark.
    :param show_agent_context: Whether to print subagent/graph metadata.
    :param node_index: Node index for DAG traversal (used for history path).
    """
    args = json.loads(req["request_args"]) if req.get("request_args") else {}
    body = args.get("body", {})
    msgs = body.get("messages", [])

    separator = _dim("─" * 72)
    print(f"\n{separator}")
    print(
        f" {_bold(f'Turn {turn_number} of {total}')}"
        f"    {_dim('(')}prompt_tokens: {_bright_cyan(str(req.get('prompt_tokens')))}"
        f" {_dim('|')} output_tokens: {_bright_green(str(req.get('output_tokens')))}{_dim(')')}"
    )

    if show_agent_context:
        _print_agent_context(req, node_index or {})

    if body.get("tool_choice"):
        print(f"  {_dim('tool_choice:')} {_yellow(str(body['tool_choice']))}")

    for m in msgs:
        _print_message(m)

    # Response produced by this turn
    tc = req.get("tool_calls")
    output = req.get("output")

    if tc:
        print(f"  {_bright_cyan('[RESPONSE - TOOL CALLS]')}")
        for t in tc:
            _print_tool_call_yaml(t)
    elif output:
        print(f"  {_bright_green('[RESPONSE]')}")
        for line in output.split("\n"):
            print(f"    {_green(line)}")
    else:
        print(f"  {_dim('[NO RESPONSE]')}")


def _print_message(m: dict[str, Any]) -> None:
    """Print a single conversation message to stdout.

    :param m: Message dict with role, content, and optional tool_calls.
    """
    role = m["role"].upper()
    content = m.get("content")
    tool_calls = m.get("tool_calls")
    tool_call_id = m.get("tool_call_id")

    print(f"  {_role_color(role, f'[{role}]')}")

    if tool_call_id:
        print(f"    {_dim('tool_call_id:')} {_dim(tool_call_id)}")

    color_fn = _ROLE_COLORS.get(role, _bold)
    if content is not None:
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    print(f"    {color_fn(part['text'])}")
        else:
            for line in str(content).split("\n"):
                print(f"    {color_fn(line)}")

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
        # Color YAML keys vs values for readability
        if ": " in line:
            key, _, value = line.partition(": ")
            print(f"    {_yellow(key)}: {_dim(value)}")
        else:
            print(f"    {_yellow(line)}")


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
