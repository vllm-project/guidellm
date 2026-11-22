#!/usr/bin/env python3
"""Read the machine-readable contract from the pull request template.

The template is the single source of truth for the certification statement,
the use-of-AI checkbox labels, and the marker that separates user-authored
content from generated content. Required sections are delimited by HTML
comments::

    <!-- begin:certification -->
    ...
    <!-- end:certification -->

    <!-- begin:ai-disclosure -->
    - [ ] ...
    <!-- end:ai-disclosure -->

Exactly one ``begin:`` sentinel must be left unpaired; that sentinel is the
generated-content marker.

Usage::

    python3 scripts/pr_template.py [--template PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md"
)

CERTIFICATION_SENTINEL = "certification"
AI_DISCLOSURE_SENTINEL = "ai-disclosure"

_BEGIN_RE = re.compile(
    r"^[ \t]*<!--[ \t]*begin:(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)[ \t]*-->[ \t]*$",
    re.MULTILINE,
)
_END_RE = re.compile(
    r"^[ \t]*<!--[ \t]*end:(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)[ \t]*-->[ \t]*$",
    re.MULTILINE,
)
_LABEL_RE = re.compile(
    r"^[ \t]*[-*+][ \t]+\[[ xX]\][ \t]+(?P<label>.+?)[ \t]*$",
    re.MULTILINE,
)


class TemplateContractError(ValueError):
    """Raised when the PR template does not satisfy the machine contract."""


@dataclass(frozen=True)
class PRTemplateContract:
    """The values the PR gate reads from the pull request template."""

    certification: str
    ai_labels: tuple[str, ...]
    squash_marker: str


def _collect_blocks(template: str, source: str) -> tuple[dict[str, str], list[str]]:
    """Collect paired sentinel blocks and unpaired begin sentinels.

    :param template: The normalized template text.
    :param source: Template name used in error messages.
    :return: A mapping of block name to inner text, and the unpaired markers.
    :raises TemplateContractError: If sentinels are malformed.
    """
    events = [(match.start(), "begin", match) for match in _BEGIN_RE.finditer(template)]
    events += [(match.start(), "end", match) for match in _END_RE.finditer(template)]
    events.sort(key=lambda event: event[0])

    open_blocks: dict[str, re.Match[str]] = {}
    blocks: dict[str, str] = {}
    for _, kind, match in events:
        name = match.group("name")
        if kind == "begin":
            if name in open_blocks or name in blocks:
                raise TemplateContractError(
                    f"{source}: duplicate or nested begin sentinel {name!r}"
                )
            open_blocks[name] = match
            continue
        if name not in open_blocks:
            raise TemplateContractError(
                f"{source}: end sentinel {name!r} has no matching begin"
            )
        begin = open_blocks.pop(name)
        blocks[name] = template[begin.end() : match.start()]

    unpaired = [match.group(0).strip() for match in open_blocks.values()]
    return blocks, unpaired


def parse_contract(template: str, *, source: str = "<template>") -> PRTemplateContract:
    """Extract the PR gate contract from a template.

    :param template: The template text.
    :param source: Template name used in error messages.
    :return: The parsed contract.
    :raises TemplateContractError: If required content is missing or ambiguous.
    """
    template = template.replace("\r\n", "\n").replace("\r", "\n")
    blocks, unpaired = _collect_blocks(template, source)

    if CERTIFICATION_SENTINEL not in blocks:
        raise TemplateContractError(
            f"{source}: missing {CERTIFICATION_SENTINEL!r} sentinel block"
        )
    certification = blocks[CERTIFICATION_SENTINEL].strip()
    if not certification:
        raise TemplateContractError(
            f"{source}: {CERTIFICATION_SENTINEL!r} block is empty"
        )

    if AI_DISCLOSURE_SENTINEL not in blocks:
        raise TemplateContractError(
            f"{source}: missing {AI_DISCLOSURE_SENTINEL!r} sentinel block"
        )
    labels = tuple(
        label.strip() for label in _LABEL_RE.findall(blocks[AI_DISCLOSURE_SENTINEL])
    )
    if not labels:
        raise TemplateContractError(
            f"{source}: {AI_DISCLOSURE_SENTINEL!r} block has no checkbox labels"
        )
    if len(set(labels)) != len(labels):
        raise TemplateContractError(f"{source}: duplicate use-of-AI labels")

    if len(unpaired) != 1:
        raise TemplateContractError(
            f"{source}: expected exactly one unpaired begin sentinel, "
            f"found {len(unpaired)}: {unpaired}"
        )

    return PRTemplateContract(certification, labels, unpaired[0])


def load_contract(path: Path | str | None = None) -> PRTemplateContract:
    """Read and parse the PR template contract.

    :param path: Template path, or ``None`` for the repository default.
    :return: The parsed contract.
    :raises TemplateContractError: If the template cannot be read or parsed.
    """
    template_path = DEFAULT_TEMPLATE_PATH if path is None else Path(path)
    try:
        template = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TemplateContractError(
            f"could not read PR template {template_path}: {exc}"
        ) from exc
    return parse_contract(template, source=str(template_path))


def main() -> None:
    """Print the generated-content marker from the template contract."""
    parser = argparse.ArgumentParser(
        description="Print the PR template's generated-content marker.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to the PR template.",
    )
    args = parser.parse_args()
    try:
        contract = load_contract(args.template)
    except TemplateContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(contract.squash_marker)


if __name__ == "__main__":
    main()
