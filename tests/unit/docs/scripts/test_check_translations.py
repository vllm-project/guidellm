"""Tests for translated documentation metadata checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from docs.scripts import check_translations
from docs.scripts.check_translations import (
    TRANSLATION_ROOTS,
    file_sha256,
    load_translations,
    translation_routes,
    update_source_hashes,
    validate_translations,
)


def _create_translation(
    project_root: Path,
    relative_path: str,
    source_content: str,
    translation_content: str,
    *,
    language: str = "zh",
    source_sha256: str | None = None,
    extra_front_matter: str = "",
) -> tuple[Path, Path]:
    source = project_root / "docs/en" / relative_path
    translation = project_root / "docs" / language / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    translation.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(source_content, encoding="utf-8")
    digest = source_sha256 if source_sha256 is not None else file_sha256(source)
    front_matter = f'source_sha256: !!str "{digest}"\n{extra_front_matter}'
    translation.write_text(
        f"---\n{front_matter}---\n\n{translation_content}", encoding="utf-8"
    )
    return source, translation


@pytest.mark.smoke
def test_valid_translation_metadata(tmp_path: Path):
    """
    Accept a current translation with unchanged code examples.

    ## WRITTEN BY AI ##
    """
    _create_translation(
        tmp_path,
        "index.md",
        "# Home\n\n```bash\nguidellm --help\n```\n",
        "# 首页\n\n```bash\nguidellm --help\n```\n",
    )

    status = validate_translations(tmp_path)

    assert status.errors == ()
    assert status.stale == ()


@pytest.mark.regression
def test_source_change_is_non_blocking_stale_warning(tmp_path: Path):
    """
    Report source drift separately from structural validation errors.

    ## WRITTEN BY AI ##
    """
    _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
        source_sha256="0" * 64,
    )

    status = validate_translations(tmp_path)

    assert status.errors == ()
    assert len(status.stale) == 1
    assert "docs/zh/index.md is behind docs/en/index.md" in status.stale[0]


@pytest.mark.parametrize("source_sha256", ["0" * 64, "1" * 60 + "e123"])
@pytest.mark.regression
def test_quoted_numeric_and_scientific_notation_hashes_remain_strings(
    tmp_path: Path, source_sha256: str
):
    """
    Accept quoted hashes that YAML could otherwise interpret as numbers.

    ## WRITTEN BY AI ##
    """
    _, translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
        source_sha256=source_sha256,
    )
    translation.write_text(
        translation.read_text(encoding="utf-8").replace(
            f'source_sha256: !!str "{source_sha256}"',
            f'source_sha256: "{source_sha256}"',
        ),
        encoding="utf-8",
    )

    translations = load_translations(tmp_path)

    assert translations[0].source_sha256 == source_sha256
    assert validate_translations(tmp_path).errors == ()


@pytest.mark.regression
def test_unquoted_numeric_source_hash_is_rejected(tmp_path: Path):
    """
    Reject numeric YAML values instead of recovering a hash from its text.

    ## WRITTEN BY AI ##
    """
    digest = "0" * 64
    _, translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
        source_sha256=digest,
    )
    translation.write_text(
        translation.read_text(encoding="utf-8").replace(
            f'source_sha256: !!str "{digest}"', f"source_sha256: {digest}"
        ),
        encoding="utf-8",
    )

    status = validate_translations(tmp_path)

    assert any("source_sha256" in error for error in status.errors)


@pytest.mark.regression
def test_missing_source_hash_is_an_error(tmp_path: Path):
    """
    Reject translated pages whose front matter omits the source hash.

    ## WRITTEN BY AI ##
    """
    _, translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
    )
    translation.write_text("---\ntitle: 首页\n---\n\n# 首页\n", encoding="utf-8")

    status = validate_translations(tmp_path)

    assert any("source_sha256" in error for error in status.errors)
    assert any("docs/zh/index.md" in error for error in status.errors)

    update_source_hashes([Path("docs/zh/index.md")], project_root=tmp_path)

    assert validate_translations(tmp_path).errors == ()


@pytest.mark.regression
def test_missing_english_source_is_an_error(tmp_path: Path):
    """
    Report when a translated page has no mirrored English source.

    ## WRITTEN BY AI ##
    """
    source, _ = _create_translation(
        tmp_path,
        "guides/missing.md",
        "# Guide\n",
        "# 指南\n",
    )
    source.unlink()

    status = validate_translations(tmp_path)

    assert any("docs/en/guides/missing.md" in error for error in status.errors)


@pytest.mark.regression
def test_changed_code_example_is_an_error(tmp_path: Path):
    """
    Reject a translation that changes executable examples.

    ## WRITTEN BY AI ##
    """
    _create_translation(
        tmp_path,
        "index.md",
        "# Home\n\n```bash\nguidellm --help\n```\n",
        "# 首页\n\n```bash\nguidellm run\n```\n",
    )

    status = validate_translations(tmp_path)

    assert status.stale == ()
    assert status.errors == (
        "Fenced code blocks differ between docs/en/index.md and docs/zh/index.md",
    )


@pytest.mark.parametrize(("language", "other_language"), [("zh", "ko"), ("ko", "zh")])
@pytest.mark.sanity
def test_update_only_selected_translation_and_build_current_routes(
    tmp_path: Path, language: str, other_language: str
):
    """
    Update only selected page metadata and expose current language routes.

    ## WRITTEN BY AI ##
    """
    source, translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
        language=language,
        source_sha256="0" * 64,
        extra_front_matter="title: 首页\nsidebar_position: 1\n",
    )
    _, other_translation = _create_translation(
        tmp_path,
        "guides/setup.md",
        "# Setup\n",
        "# 安装\n",
        language=language,
        source_sha256="1" * 64,
        extra_front_matter="title: 安装指南\ncustom_field: retained\n",
    )
    other_before = other_translation.read_text(encoding="utf-8")
    _, other_language_translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 首页\n",
        language=other_language,
        source_sha256="2" * 64,
    )
    _, other_language_setup = _create_translation(
        tmp_path,
        "guides/setup.md",
        "# Setup\n",
        "# 安装\n",
        language=other_language,
        source_sha256="3" * 64,
    )
    other_language_before = {
        other_language_translation: other_language_translation.read_text(
            encoding="utf-8"
        ),
        other_language_setup: other_language_setup.read_text(encoding="utf-8"),
    }

    update_source_hashes([Path("docs") / language / "index.md"], project_root=tmp_path)

    status = validate_translations(tmp_path, TRANSLATION_ROOTS[language])
    updated_text = translation.read_text(encoding="utf-8")
    assert status.errors == ()
    assert len(status.stale) == 1
    assert (
        f"docs/{language}/guides/setup.md is behind docs/en/guides/setup.md"
        in status.stale[0]
    )
    assert f'source_sha256: !!str "{file_sha256(source)}"' in updated_text
    assert "title: 首页" in updated_text
    assert "sidebar_position: 1" in updated_text
    assert other_translation.read_text(encoding="utf-8") == other_before
    assert all(
        path.read_text(encoding="utf-8") == before
        for path, before in other_language_before.items()
    )
    assert translation_routes(tmp_path, TRANSLATION_ROOTS[language]) == {
        "": {
            "translation": f"{language}/",
            "current": True,
        },
        "guides/setup/": {
            "translation": f"{language}/guides/setup/",
            "current": False,
        },
    }
    assert (
        translation_routes(tmp_path, TRANSLATION_ROOTS[other_language])[""][
            "translation"
        ]
        == f"{other_language}/"
    )


@pytest.mark.regression
def test_cli_checks_all_languages_by_default_and_can_filter_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """
    Check every configured language by default and honor language filtering.

    ## WRITTEN BY AI ##
    """
    _create_translation(tmp_path, "index.md", "# Home\n", "# 首页\n")
    _, ko_translation = _create_translation(
        tmp_path,
        "index.md",
        "# Home\n",
        "# 홈\n",
        language="ko",
    )
    ko_translation.write_text("---\ntitle: 홈\n---\n\n# 홈\n", encoding="utf-8")
    monkeypatch.setattr(check_translations, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(sys, "argv", ["check_translations"])
    assert check_translations.main() == 1
    assert "ERROR [ko]:" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["check_translations", "--language", "zh"])
    assert check_translations.main() == 0
    assert capsys.readouterr().out == "Translation metadata is valid.\n"
