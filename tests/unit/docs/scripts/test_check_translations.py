"""Tests for translated documentation metadata checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docs.scripts.check_translations import (
    file_sha256,
    translation_routes,
    update_source_hashes,
    validate_translations,
)

MANIFEST_PATH = Path("docs/zh/.translation-sources.json")


def _create_translation_project(
    project_root: Path,
    source_content: str,
    translation_content: str,
    source_sha256: str | None = None,
) -> Path:
    source = project_root / "docs/en/index.md"
    translation = project_root / "docs/zh/index.md"
    manifest = project_root / MANIFEST_PATH
    translation.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text(source_content, encoding="utf-8")
    translation.write_text(translation_content, encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "translations": [
                    {
                        "source": "docs/en/index.md",
                        "translation": "docs/zh/index.md",
                        "source_sha256": source_sha256 or file_sha256(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source


@pytest.mark.smoke
def test_valid_translation_metadata(tmp_path: Path):
    """
    Accept a current translation with unchanged code examples.

    ## WRITTEN BY AI ##
    """
    _create_translation_project(
        tmp_path,
        "# Home\n\n```bash\nguidellm --help\n```\n",
        "# 首页\n\n```bash\nguidellm --help\n```\n",
    )

    status = validate_translations(tmp_path, MANIFEST_PATH)

    assert status.errors == ()
    assert status.stale == ()


@pytest.mark.regression
def test_source_change_is_non_blocking_stale_warning(tmp_path: Path):
    """
    Report source drift separately from structural validation errors.

    ## WRITTEN BY AI ##
    """
    _create_translation_project(
        tmp_path,
        "# Home\n",
        "# 首页\n",
        source_sha256="0" * 64,
    )

    status = validate_translations(tmp_path, MANIFEST_PATH)

    assert status.errors == ()
    assert len(status.stale) == 1
    assert "docs/zh/index.md is behind docs/en/index.md" in status.stale[0]


@pytest.mark.regression
def test_changed_code_example_is_an_error(tmp_path: Path):
    """
    Reject a translation that changes executable examples.

    ## WRITTEN BY AI ##
    """
    _create_translation_project(
        tmp_path,
        "# Home\n\n```bash\nguidellm --help\n```\n",
        "# 首页\n\n```bash\nguidellm run\n```\n",
    )

    status = validate_translations(tmp_path, MANIFEST_PATH)

    assert status.stale == ()
    assert status.errors == (
        "Fenced code blocks differ between docs/en/index.md and docs/zh/index.md",
    )


@pytest.mark.sanity
def test_update_hashes_and_build_routes(tmp_path: Path):
    """
    Refresh source metadata and expose version-relative language routes.

    ## WRITTEN BY AI ##
    """
    source = _create_translation_project(
        tmp_path,
        "# Home\n",
        "# 首页\n",
        source_sha256="0" * 64,
    )

    update_source_hashes(tmp_path, MANIFEST_PATH)

    status = validate_translations(tmp_path, MANIFEST_PATH)
    assert status.errors == ()
    assert status.stale == ()
    assert translation_routes(tmp_path, MANIFEST_PATH) == {
        "": {
            "translation": "zh/",
            "current": True,
        }
    }
    manifest = json.loads((tmp_path / MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["translations"][0]["source_sha256"] == file_sha256(source)
