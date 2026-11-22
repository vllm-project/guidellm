"""Tests for the PR template contract parser."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import pr_template
from scripts.pr_template import (
    DEFAULT_TEMPLATE_PATH,
    PRTemplateContract,
    TemplateContractError,
    load_contract,
    parse_contract,
)

SCRIPT_PATH = Path(pr_template.__file__)

pytestmark = pytest.mark.sanity


def _template(
    *,
    certification: str = "New certification text.",
    labels: tuple[str, ...] = ("Alpha", "Beta"),
    marker: str = "<!-- begin:generated -->",
) -> str:
    """Build a synthetic template.

    :param certification: Certification text.
    :param labels: AI-use checkbox labels.
    :param marker: The unpaired generated-content sentinel.
    :return: A synthetic template.
    """
    checkboxes = "\n".join(f"- [ ] {label}" for label in labels)
    return "\n".join(
        [
            "<!-- begin:certification -->",
            certification,
            "<!-- end:certification -->",
            "",
            "<!-- begin:ai-disclosure -->",
            checkboxes,
            "<!-- end:ai-disclosure -->",
            "",
            marker,
            "",
        ]
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the template CLI.

    :param args: Extra CLI arguments.
    :return: The completed process.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_shipped_template_parses():
    """
    The shipped template yields a complete contract.

    ## WRITTEN BY AI ##
    """
    contract = load_contract()

    assert contract.certification
    assert contract.ai_labels
    assert contract.squash_marker.startswith("<!-- begin:")
    assert contract.squash_marker.endswith("-->")


def test_shipped_template_has_one_unpaired_sentinel():
    """
    The shipped template leaves exactly one begin sentinel unpaired.

    ## WRITTEN BY AI ##
    """
    template = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    begins = re.findall(r"<!-- begin:([\w-]+) -->", template)
    ends = re.findall(r"<!-- end:([\w-]+) -->", template)

    assert len(begins) == len(ends) + 1
    assert set(ends).issubset(begins)


def test_shipped_contract_blocks_precede_squash_marker():
    """
    Required blocks appear before the generated-content marker.

    ## WRITTEN BY AI ##
    """
    template = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    marker_index = template.index(load_contract().squash_marker)

    assert template.index("<!-- begin:certification -->") < marker_index
    assert template.index("<!-- begin:ai-disclosure -->") < marker_index


def test_synthetic_template_is_fully_contract_driven():
    """
    Wording, labels, and marker rename all come from the template.

    ## WRITTEN BY AI ##
    """
    contract = parse_contract(
        _template(
            certification="Totally different statement.",
            labels=("Gamma", "Delta", "Epsilon"),
            marker="<!-- begin:renamed -->",
        )
    )

    assert contract == PRTemplateContract(
        "Totally different statement.",
        ("Gamma", "Delta", "Epsilon"),
        "<!-- begin:renamed -->",
    )


def test_parse_contract_normalizes_crlf():
    """
    parse_contract accepts CRLF text directly.

    ## WRITTEN BY AI ##
    """
    contract = parse_contract(_template().replace("\n", "\r\n"))

    assert contract.certification == "New certification text."
    assert contract.ai_labels == ("Alpha", "Beta")


def test_crlf_template_parses(tmp_path):
    """
    CRLF line endings are normalized before parsing.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "template.md"
    path.write_bytes(_template().replace("\n", "\r\n").encode())

    contract = load_contract(path)

    assert contract.certification == "New certification text."
    assert contract.ai_labels == ("Alpha", "Beta")


def test_missing_certification_sentinel_raises():
    """
    A template without a complete certification block is rejected.

    ## WRITTEN BY AI ##
    """
    template = _template().replace("<!-- end:certification -->", "")

    with pytest.raises(TemplateContractError):
        parse_contract(template)


def test_missing_ai_disclosure_sentinel_raises():
    """
    A template without a complete AI block is rejected.

    ## WRITTEN BY AI ##
    """
    template = _template().replace("<!-- end:ai-disclosure -->", "")

    with pytest.raises(TemplateContractError):
        parse_contract(template)


def test_empty_ai_disclosure_block_raises():
    """
    An AI block with no checkboxes is rejected.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        parse_contract(_template(labels=()))


def test_duplicate_labels_raise():
    """
    Duplicate AI labels are rejected.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        parse_contract(_template(labels=("Alpha", "Alpha")))


def test_duplicate_certification_block_raises():
    """
    A second certification block is rejected.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        parse_contract(_template() + _template())


def test_stray_end_sentinel_raises():
    """
    An end sentinel without a begin is rejected.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        parse_contract(_template() + "<!-- end:orphan -->\n")


def test_multiple_unpaired_sentinels_raise():
    """
    More than one unpaired begin sentinel is ambiguous.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        parse_contract(_template() + "<!-- begin:extra -->\n")


def test_missing_template_file_raises(tmp_path):
    """
    A missing template path is reported as a contract error.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(TemplateContractError):
        load_contract(tmp_path / "missing.md")


def test_cli_prints_shipped_marker():
    """
    The CLI prints the shipped template's marker.

    ## WRITTEN BY AI ##
    """
    result = _run_cli()

    assert result.returncode == 0
    assert result.stdout.strip() == load_contract().squash_marker


def test_cli_honors_template_override(tmp_path):
    """
    The CLI reads an alternate template.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "template.md"
    path.write_text(_template(marker="<!-- begin:custom -->"), encoding="utf-8")

    result = _run_cli("--template", str(path))

    assert result.returncode == 0
    assert result.stdout.strip() == "<!-- begin:custom -->"


def test_cli_fails_on_invalid_template(tmp_path):
    """
    The CLI reports an invalid template on stderr and no marker on stdout.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "template.md"
    path.write_text("no sentinels here\n", encoding="utf-8")

    result = _run_cli("--template", str(path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
