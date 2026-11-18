"""Tests for the PR styling certification check.

The check lives inline in the ``PR Styling`` workflow. These tests execute the
workflow's actual shell block against synthetic PR bodies so the gate and the
PR template cannot drift apart.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr-styling.yml"
TEMPLATE_PATH = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

CERT_LINE = (
    "- By submitting this PR I certify that I have reviewed every change and "
    "addition to this pull request and take responsibility for its "
    "correctness, quality, and impact."
)
AI_LABELS = (
    "No AI was used in the creation of this PR",
    "Includes code generated or substantially modified by an AI agent",
    "Includes tests generated or substantially modified by an AI agent",
    "AI was used to create this description or communicate with maintainers",
)

BASH = shutil.which("bash")
assert BASH is not None


def _run_block() -> str:
    """Extract the certification-check shell block from the workflow.

    :return: The ``run`` script for the certification-check step.
    """
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["certification-check"]["steps"]
    run_blocks = [step["run"] for step in steps if "run" in step]
    assert len(run_blocks) == 1
    return run_blocks[0]


RUN_BLOCK = _run_block()


def _run_check(body: str) -> subprocess.CompletedProcess[str]:
    """Run the workflow's certification check against a PR body.

    :param body: The PR body passed through the ``PR_BODY`` environment variable.
    :return: The completed process for the check.
    """
    return subprocess.run(  # noqa: S603
        [BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c", RUN_BLOCK],
        env={"PATH": os.environ["PATH"], "PR_BODY": body},
        capture_output=True,
        text=True,
        check=False,
    )


def _body(checked: str | None, cert: str | None = CERT_LINE) -> str:
    """Build a PR body resembling the current template.

    :param checked: The single AI-use label to check, or ``None`` for none.
    :param cert: The certification line, or ``None`` to omit it.
    :return: A PR body.
    """
    lines = ["## Summary", "", "Do the thing.", "", "---", ""]
    if cert is not None:
        lines.extend([cert, ""])
    lines.extend(["## Use of AI", ""])
    lines.extend(
        f"- [{'x' if label == checked else ' '}] {label}" for label in AI_LABELS
    )
    lines.extend(
        ["", "<!-- Do not edit below this line -->", "", "<!-- begin:squash-data -->"]
    )
    return "\n".join(lines)


def _template_body(checked: str | None = None) -> str:
    """Load the real PR template, optionally checking one AI-use box.

    :param checked: The AI-use label to check, or ``None`` to leave all
        boxes unchecked.
    :return: A PR body based on the shipped template.
    """
    body = TEMPLATE_PATH.read_text(encoding="utf-8")
    if checked is not None:
        body = body.replace(f"- [ ] {checked}", f"- [x] {checked}", 1)
    return body


@pytest.mark.sanity
@pytest.mark.parametrize("label", AI_LABELS)
def test_pr_template_passes_with_any_box_checked(label):
    """
    The shipped template passes the gate once any one box is checked.

    ## WRITTEN BY AI ##
    """
    result = _run_check(_template_body(label))

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.sanity
def test_pr_template_fails_until_a_box_is_checked():
    """
    The unmodified template fails only the AI disclosure check.

    ## WRITTEN BY AI ##
    """
    result = _run_check(_template_body())

    assert result.returncode == 1
    assert "AI disclosure required" in result.stdout
    assert "Certification required" not in result.stdout


@pytest.mark.sanity
@pytest.mark.parametrize("label", AI_LABELS)
def test_any_single_ai_box_passes(label):
    """
    Any one checked use-of-AI box satisfies the AI disclosure.

    ## WRITTEN BY AI ##
    """
    result = _run_check(_body(checked=label))

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.sanity
def test_missing_ai_box_fails():
    """
    The certification alone must not satisfy the AI disclosure.

    ## WRITTEN BY AI ##
    """
    result = _run_check(_body(checked=None))

    assert result.returncode == 1
    assert "AI disclosure required" in result.stdout


@pytest.mark.sanity
def test_missing_certification_fails():
    """
    A checked AI box alone must not replace the certification statement.

    ## WRITTEN BY AI ##
    """
    result = _run_check(_body(checked=AI_LABELS[0], cert=None))

    assert result.returncode == 1
    assert "Certification required" in result.stdout


@pytest.mark.sanity
def test_certification_below_squash_marker_fails():
    """
    Generated content below the squash marker must not satisfy the check.

    ## WRITTEN BY AI ##
    """
    body = "\n".join(
        [
            "## Summary",
            "",
            f"- [x] {AI_LABELS[0]}",
            "",
            "<!-- begin:squash-data -->",
            "",
            CERT_LINE,
            "",
        ]
    )

    result = _run_check(body)

    assert result.returncode == 1
    assert "Certification required" in result.stdout


@pytest.mark.sanity
def test_ai_box_below_squash_marker_fails():
    """
    A checked box below the squash marker must not satisfy the disclosure.

    ## WRITTEN BY AI ##
    """
    body = "\n".join(
        [
            "## Summary",
            "",
            CERT_LINE,
            "",
            "<!-- begin:squash-data -->",
            "",
            f"- [x] {AI_LABELS[0]}",
            "",
        ]
    )

    result = _run_check(body)

    assert result.returncode == 1
    assert "AI disclosure required" in result.stdout


@pytest.mark.sanity
def test_empty_body_fails():
    """
    An empty body satisfies neither check.

    ## WRITTEN BY AI ##
    """
    result = _run_check("")

    assert result.returncode == 1
    assert "Certification required" in result.stdout


@pytest.mark.sanity
def test_crlf_body_passes():
    """
    CRLF line endings do not break the line-anchored checkbox check.

    ## WRITTEN BY AI ##
    """
    body = _body(checked=AI_LABELS[0]).replace("\n", "\r\n")

    result = _run_check(body)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.sanity
def test_template_certification_matches_workflow():
    """
    The template's certification line must match the workflow's expectation.

    ## WRITTEN BY AI ##
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert CERT_LINE in template
    assert CERT_LINE in RUN_BLOCK


@pytest.mark.sanity
def test_template_ai_labels_match_workflow():
    """
    Every AI-use box in the template must be recognized by the workflow.

    ## WRITTEN BY AI ##
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    section = template.split("## Use of AI", 1)[1]
    labels = re.findall(r"^- \[ \] (.+)$", section, re.MULTILINE)

    assert tuple(labels) == AI_LABELS
    for label in labels:
        result = _run_check(_body(checked=label))
        assert result.returncode == 0, result.stdout + result.stderr
