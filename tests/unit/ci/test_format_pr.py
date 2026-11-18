"""Tests for the PR description formatter's contract-driven marker."""

from __future__ import annotations

import io
import sys

import pytest

from scripts import format_pr, pr_template

pytestmark = pytest.mark.sanity


def test_extract_user_content_stops_at_marker():
    """
    Extraction stops at the supplied contract marker.

    ## WRITTEN BY AI ##
    """
    body = "above\n\n<!-- begin:custom -->\n\nbelow"

    assert format_pr.extract_user_content(body, "<!-- begin:custom -->") == "above"


def test_extract_user_content_without_marker_returns_body():
    """
    A body without the marker is entirely user content.

    ## WRITTEN BY AI ##
    """
    body = "above\n\nbelow\n"

    assert (
        format_pr.extract_user_content(body, "<!-- begin:custom -->")
        == "above\n\nbelow"
    )


def test_format_pr_description_uses_contract_marker(monkeypatch):
    """
    The formatter writes the contract marker, not the legacy literal.

    ## WRITTEN BY AI ##
    """

    def fake_run_git(*args, stdin=None):
        if args[0] == "merge-base":
            return "base"
        if args[0] == "log" and "--format=%H" in args:
            return "sha1"
        if args[0] == "log" and "--format=%B" in args:
            return "Subject\n\nSigned-off-by: A <a@example.com>"
        if args[0] == "interpret-trailers":
            return "Signed-off-by: A <a@example.com>"
        return ""

    monkeypatch.setattr(format_pr, "run_git", fake_run_git)

    result = format_pr.format_pr_description(
        "Body", "origin/main", "sha1", "<!-- begin:custom -->"
    )

    assert result.startswith("Body\n\n<!-- begin:custom -->")
    assert pr_template.load_contract().squash_marker not in result
    assert "Signed-off-by: A <a@example.com>" in result


def test_main_fails_closed_when_contract_invalid(monkeypatch, capsys):
    """
    An invalid template exits two with an error on stderr only.

    ## WRITTEN BY AI ##
    """

    def fail_contract():
        raise pr_template.TemplateContractError("broken template")

    monkeypatch.setattr(format_pr.pr_template, "load_contract", fail_contract)
    monkeypatch.setattr(sys, "argv", ["format_pr.py", "origin/main", "abc"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("Body"))

    with pytest.raises(SystemExit) as exc_info:
        format_pr.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid PR template contract" in captured.err
