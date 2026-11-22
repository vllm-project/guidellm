#!/usr/bin/env python3
"""Verify a PR description satisfies the project's contribution requirements.

The current PR body is read from stdin. The contract (certification statement,
use-of-AI labels, and generated-content marker) comes from the pull request
template, which is the single source of truth. The process exits non-zero and
prints GitHub Actions ``::error`` annotations when the certification statement
or the use-of-AI disclosure is missing. Content generated below the marker is
ignored, matching ``scripts/format_pr.py``.

Usage::

    printf '%s' "$PR_BODY" | python3 scripts/check_pr_description.py
"""

from __future__ import annotations

import re
import sys

if __package__:
    from scripts import pr_template
else:
    import pr_template

CERTIFICATION_ERROR = (
    "::error title=Certification required::Keep the certification statement "
    "from the PR template in the description."
)
AI_DISCLOSURE_ERROR = (
    "::error title=AI disclosure required::Check at least one box in the "
    "'Use of AI' section of the PR description."
)
TEMPLATE_ERROR = "::error title=PR template contract invalid::"
SUCCESS_MESSAGE = (
    "Certification statement is present and at least one use-of-AI box is checked."
)


def validate(body: str, contract: pr_template.PRTemplateContract) -> list[str]:
    """Validate a PR body and return GitHub annotation strings.

    The certification match is a case-insensitive substring, and the AI label
    match is a case-insensitive prefix, preserving the prior ``grep`` behavior.

    :param body: The raw PR body.
    :param contract: The template contract to validate against.
    :return: Zero or more ``::error`` annotations; empty when valid.
    """
    content = body.replace("\r", "").partition(contract.squash_marker)[0]
    alternation = "|".join(re.escape(label) for label in contract.ai_labels)
    ai_box = re.compile(
        rf"^[ \t]*[-*+][ \t]+\[x\][ \t]+(?:{alternation})",
        re.IGNORECASE | re.MULTILINE,
    )

    errors: list[str] = []
    if contract.certification.lower() not in content.lower():
        errors.append(CERTIFICATION_ERROR)
    if ai_box.search(content) is None:
        errors.append(AI_DISCLOSURE_ERROR)
    return errors


def main() -> None:
    """Read the PR body from stdin and exit non-zero when invalid."""
    try:
        contract = pr_template.load_contract()
    except pr_template.TemplateContractError as exc:
        print(f"{TEMPLATE_ERROR}{exc}")
        sys.exit(2)

    errors = validate(sys.stdin.read(), contract)
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    print(SUCCESS_MESSAGE)


if __name__ == "__main__":
    main()
