"""Tests for the PR description validator.

The validator lives in ``scripts/check_pr_description.py`` and is invoked by the
``PR Styling`` workflow. Its contract comes from the PR template, so these
tests exercise both the shipped template and synthetic templates.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_pr_description, pr_template
from scripts.check_pr_description import (
    AI_DISCLOSURE_ERROR,
    CERTIFICATION_ERROR,
    SUCCESS_MESSAGE,
    validate,
)
from scripts.pr_template import parse_contract

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_pr_description.py"

CONTRACT = pr_template.load_contract()
SYNTHETIC = parse_contract(
    "\n".join(
        [
            "<!-- begin:certification -->",
            "New certification.",
            "<!-- end:certification -->",
            "",
            "<!-- begin:ai-disclosure -->",
            "- [ ] New label",
            "<!-- end:ai-disclosure -->",
            "",
            "<!-- begin:generated -->",
            "",
        ]
    )
)

pytestmark = pytest.mark.sanity


def _template_with_box(checked: str | None = None) -> str:
    """Read the shipped template, optionally checking one AI-use box.

    :param checked: The AI-use label to check, or ``None`` to leave all
        boxes unchecked.
    :return: A PR body based on the template.
    """
    body = TEMPLATE_PATH.read_text(encoding="utf-8")
    if checked is not None:
        body = body.replace(f"- [ ] {checked}", f"- [x] {checked}", 1)
    return body


def _body(
    *,
    certification: str | None = CONTRACT.certification,
    checked: str | None = None,
    labels: tuple[str, ...] = CONTRACT.ai_labels,
) -> str:
    """Build a compact PR body for focused checks.

    :param certification: The certification line, or ``None`` to omit it.
    :param checked: The AI-use label to check, or ``None`` for none.
    :param labels: The AI-use labels to include.
    :return: A PR body.
    """
    lines = ["## Summary", "", "Do the thing.", ""]
    if certification is not None:
        lines.extend([certification, ""])
    lines.extend(["## Use of AI", ""])
    lines.extend(f"- [{'x' if label == checked else ' '}] {label}" for label in labels)
    return "\n".join(lines)


def _run_cli(body: str) -> subprocess.CompletedProcess[str]:
    """Run the validator script with a body on stdin.

    :param body: The PR body to validate.
    :return: The completed process.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT_PATH)],
        input=body,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("label", CONTRACT.ai_labels)
def test_each_template_label_satisfies_disclosure(label):
    """
    Every template checkbox satisfies the disclosure when checked.

    ## WRITTEN BY AI ##
    """
    assert validate(_template_with_box(label), CONTRACT) == []


def test_unmodified_template_fails_ai_only():
    """
    The untouched template passes certification but not AI disclosure.

    ## WRITTEN BY AI ##
    """
    assert validate(_template_with_box(), CONTRACT) == [AI_DISCLOSURE_ERROR]


def test_missing_certification_fails():
    """
    Omit the certification statement and only that check fails.

    ## WRITTEN BY AI ##
    """
    body = _body(certification=None, checked=CONTRACT.ai_labels[0])

    assert validate(body, CONTRACT) == [CERTIFICATION_ERROR]


def test_missing_ai_disclosure_fails():
    """
    Leave every box unchecked and only the AI disclosure fails.

    ## WRITTEN BY AI ##
    """
    assert validate(_body(checked=None), CONTRACT) == [AI_DISCLOSURE_ERROR]


def test_empty_body_fails_both_checks():
    """
    An empty body fails both checks.

    ## WRITTEN BY AI ##
    """
    assert validate("", CONTRACT) == [CERTIFICATION_ERROR, AI_DISCLOSURE_ERROR]


def test_reworded_certification_is_honored():
    """
    A synthetic contract's certification wording is accepted without code changes.

    ## WRITTEN BY AI ##
    """
    body = _body(
        certification=SYNTHETIC.certification,
        checked=SYNTHETIC.ai_labels[0],
        labels=SYNTHETIC.ai_labels,
    )

    assert validate(body, SYNTHETIC) == []
    assert CERTIFICATION_ERROR in validate(body, CONTRACT)


def test_added_label_is_honored():
    """
    A label added only to the synthetic contract is accepted.

    ## WRITTEN BY AI ##
    """
    body = _body(
        certification=SYNTHETIC.certification,
        checked="New label",
        labels=("New label",),
    )

    assert validate(body, SYNTHETIC) == []
    assert AI_DISCLOSURE_ERROR in validate(body, CONTRACT)


def test_removed_label_is_rejected():
    """
    A label absent from the synthetic contract is rejected.

    ## WRITTEN BY AI ##
    """
    body = _body(
        certification=SYNTHETIC.certification,
        checked=CONTRACT.ai_labels[0],
    )

    assert validate(body, SYNTHETIC) == [AI_DISCLOSURE_ERROR]


def test_certification_below_marker_is_ignored():
    """
    Generated content below the contract marker cannot satisfy certification.

    ## WRITTEN BY AI ##
    """
    body = "\n".join(
        [
            "## Summary",
            "",
            f"- [x] {SYNTHETIC.ai_labels[0]}",
            "",
            SYNTHETIC.squash_marker,
            "",
            SYNTHETIC.certification,
            "",
        ]
    )

    assert validate(body, SYNTHETIC) == [CERTIFICATION_ERROR]


def test_ai_box_below_marker_is_ignored():
    """
    Generated content below the contract marker cannot satisfy AI disclosure.

    ## WRITTEN BY AI ##
    """
    body = "\n".join(
        [
            "## Summary",
            "",
            SYNTHETIC.certification,
            "",
            SYNTHETIC.squash_marker,
            "",
            f"- [x] {SYNTHETIC.ai_labels[0]}",
            "",
        ]
    )

    assert validate(body, SYNTHETIC) == [AI_DISCLOSURE_ERROR]


def test_unrelated_checked_box_is_rejected():
    """
    A checked box that is not an AI-use option is rejected.

    ## WRITTEN BY AI ##
    """
    body = f"## Summary\n\n{CONTRACT.certification}\n\n- [x] Something else\n"

    assert validate(body, CONTRACT) == [AI_DISCLOSURE_ERROR]


def test_truncated_ai_label_is_rejected():
    """
    A label missing part of its text is rejected.

    ## WRITTEN BY AI ##
    """
    truncated = CONTRACT.ai_labels[0][:-1]
    body = f"## Summary\n\n{CONTRACT.certification}\n\n- [x] {truncated}\n"

    assert validate(body, CONTRACT) == [AI_DISCLOSURE_ERROR]


def test_label_with_trailing_text_is_accepted():
    """
    A recognized label followed by extra text still matches.

    ## WRITTEN BY AI ##
    """
    body = (
        f"## Summary\n\n{CONTRACT.certification}\n\n"
        f"- [x] {CONTRACT.ai_labels[0]} and more\n"
    )

    assert validate(body, CONTRACT) == []


@pytest.mark.parametrize("prefix", ["-", "*", "+", "   -"])
def test_list_marker_variants_and_indentation_pass(prefix):
    """
    Any supported list marker and indentation is accepted.

    ## WRITTEN BY AI ##
    """
    body = (
        f"## Summary\n\n{CONTRACT.certification}\n\n"
        f"{prefix} [x] {CONTRACT.ai_labels[0]}\n"
    )

    assert validate(body, CONTRACT) == []


def test_uppercase_checkbox_and_case_insensitive_cert():
    """
    Uppercase checkboxes and certification text are accepted.

    ## WRITTEN BY AI ##
    """
    body = (
        f"## Summary\n\n{CONTRACT.certification.upper()}\n\n"
        f"- [X] {CONTRACT.ai_labels[0].upper()}\n"
    )

    assert validate(body, CONTRACT) == []


def test_crlf_body_passes():
    """
    CRLF line endings do not break line-anchored matching.

    ## WRITTEN BY AI ##
    """
    body = _template_with_box(CONTRACT.ai_labels[0]).replace("\n", "\r\n")

    assert validate(body, CONTRACT) == []


def test_cli_returns_zero_for_valid_body():
    """
    The CLI exits zero and prints the success message for a valid body.

    ## WRITTEN BY AI ##
    """
    result = _run_cli(_template_with_box(CONTRACT.ai_labels[0]))

    assert result.returncode == 0
    assert "::error" not in result.stdout
    assert SUCCESS_MESSAGE in result.stdout


def test_cli_emits_annotations_and_exits_one():
    """
    The CLI exits one with annotations and never echoes the body.

    ## WRITTEN BY AI ##
    """
    result = _run_cli("SECRET BODY TEXT")

    assert result.returncode == 1
    assert CERTIFICATION_ERROR in result.stdout
    assert AI_DISCLOSURE_ERROR in result.stdout
    assert "SECRET BODY TEXT" not in result.stdout


def test_main_fails_closed_when_template_contract_invalid(monkeypatch, capsys):
    """
    An invalid template makes the CLI fail closed without echoing the body.

    ## WRITTEN BY AI ##
    """

    def fail_contract():
        raise pr_template.TemplateContractError("broken template")

    monkeypatch.setattr(
        check_pr_description.pr_template, "load_contract", fail_contract
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("SECRET BODY TEXT"))

    with pytest.raises(SystemExit) as exc_info:
        check_pr_description.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "::error title=PR template contract invalid::" in captured.out
    assert "SECRET BODY TEXT" not in captured.out
