"""Validate and update metadata for translated documentation."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = Path("docs")
ENGLISH_DOCS_ROOT = DOCS_ROOT / "en"
DEVELOPER_SOURCES = {
    Path("developer/code-of-conduct.md"): Path("CODE_OF_CONDUCT.md"),
    Path("developer/contributing.md"): Path("CONTRIBUTING.md"),
    Path("developer/developing.md"): Path("DEVELOPING.md"),
}
TRANSLATION_ROOTS = {
    "zh": DOCS_ROOT / "zh",
    "ko": DOCS_ROOT / "ko",
}
DEFAULT_TRANSLATIONS_ROOT = TRANSLATION_ROOTS["zh"]
FRONT_MATTER = re.compile(
    r"\A---[ \t]*\r?\n(?P<metadata>.*?)^---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
SOURCE_HASH = re.compile(r"(?m)^source_sha256:[ \t]*[^\r\n]*(?:\r?\n|\Z)")


@dataclass(frozen=True)
class TranslationSource:
    """Describe a translated page and its canonical English source."""

    source: Path
    translation: Path
    source_sha256: str


@dataclass(frozen=True)
class TranslationStatus:
    """Report validation errors and translations whose source has changed."""

    errors: tuple[str, ...]
    stale: tuple[str, ...]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_path(project_root: Path, value: str) -> Path:
    path = (project_root / value).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError(f"Path escapes the repository: {value}")

    return path


def _translated_pages(project_root: Path, translations_root: Path) -> list[Path]:
    return sorted((project_root / translations_root).rglob("*.md"))


def _front_matter(content: str, translation: Path) -> tuple[re.Match[str], dict]:
    match = FRONT_MATTER.match(content)
    if match is None:
        raise ValueError(f"Missing YAML front matter: {translation}")
    try:
        metadata = yaml.safe_load(match.group("metadata"))
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML front matter in {translation}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(f"YAML front matter must be a mapping: {translation}")
    return match, metadata


def _translation_source(
    project_root: Path, translation: Path, translations_root: Path
) -> TranslationSource:
    relative_path = translation.relative_to(project_root / translations_root)
    source = DEVELOPER_SOURCES.get(relative_path, ENGLISH_DOCS_ROOT / relative_path)
    translation_path = translations_root / relative_path
    _, metadata = _front_matter(
        translation.read_text(encoding="utf-8"), translation_path
    )
    source_hash = metadata.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise ValueError(f"Missing or invalid source_sha256 in {translation_path}")
    return TranslationSource(
        source=source, translation=translation_path, source_sha256=source_hash
    )


def load_translations(
    project_root: Path = PROJECT_ROOT,
    translations_root: Path = DEFAULT_TRANSLATIONS_ROOT,
) -> list[TranslationSource]:
    """Load source revisions from each translated page's front matter."""
    return [
        _translation_source(project_root, path, translations_root)
        for path in _translated_pages(project_root, translations_root)
    ]


def _fenced_code_blocks(content: str) -> tuple[str, ...]:
    blocks: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    block_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.lstrip()
        if fence_character is None:
            if stripped.startswith(("```", "~~~")):
                fence_character = stripped[0]
                fence_length = len(stripped) - len(stripped.lstrip(fence_character))
                block_lines = []
            continue

        if stripped.startswith(fence_character * fence_length):
            blocks.append("\n".join(block_lines))
            fence_character = None
            fence_length = 0
            block_lines = []
            continue

        block_lines.append(line)

    return tuple(blocks)


def validate_translations(
    project_root: Path = PROJECT_ROOT,
    translations_root: Path = DEFAULT_TRANSLATIONS_ROOT,
) -> TranslationStatus:
    """Validate translation mappings, source revisions, and code examples."""
    errors: list[str] = []
    stale: list[str] = []
    for path in _translated_pages(project_root, translations_root):
        try:
            entry = _translation_source(project_root, path, translations_root)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue

        entry_status = _validate_entry(project_root, entry, translations_root)
        errors.extend(entry_status.errors)
        stale.extend(entry_status.stale)

    return TranslationStatus(errors=tuple(errors), stale=tuple(stale))


def _validate_entry(
    project_root: Path, entry: TranslationSource, translations_docs_root: Path
) -> TranslationStatus:
    errors: list[str] = []
    stale: list[str] = []
    english_root = (project_root / ENGLISH_DOCS_ROOT).resolve()
    translations_root = (project_root / translations_docs_root).resolve()

    try:
        source = _repository_path(project_root, entry.source.as_posix())
        translation = _repository_path(project_root, entry.translation.as_posix())
    except ValueError as error:
        return TranslationStatus(errors=(str(error),), stale=())

    if not source.is_relative_to(english_root) and entry.source not in (
        DEVELOPER_SOURCES.values()
    ):
        errors.append(f"English source must be under docs/en/: {entry.source}")
    if not translation.is_relative_to(translations_root):
        errors.append(
            f"Translation must be under "
            f"{translations_docs_root.as_posix()}/: {entry.translation}"
        )
    if not source.is_file():
        errors.append(f"English source does not exist: {entry.source}")
    if not translation.is_file():
        errors.append(f"Translation does not exist: {entry.translation}")
    if errors:
        return TranslationStatus(errors=tuple(errors), stale=())

    if file_sha256(source) != entry.source_sha256:
        stale.append(
            f"{entry.translation} is behind {entry.source}; "
            "update the translation and refresh its source hash"
        )

    source_blocks = _fenced_code_blocks(source.read_text(encoding="utf-8"))
    translation_blocks = _fenced_code_blocks(translation.read_text(encoding="utf-8"))
    if source_blocks != translation_blocks:
        errors.append(
            f"Fenced code blocks differ between {entry.source} and {entry.translation}"
        )

    return TranslationStatus(errors=tuple(errors), stale=tuple(stale))


def update_source_hashes(
    translation_paths: list[Path], project_root: Path = PROJECT_ROOT
) -> None:
    """Refresh source hashes for only the reviewed translated pages."""
    for translation_path in translation_paths:
        translation = _repository_path(project_root, translation_path.as_posix())
        translations_root = next(
            (
                (project_root / root).resolve()
                for root in TRANSLATION_ROOTS.values()
                if translation.is_relative_to((project_root / root).resolve())
            ),
            None,
        )
        if translations_root is None or translation.suffix != ".md":
            raise ValueError(
                f"Translation must be Markdown under a supported language "
                f"directory: {translation_path}"
            )
        if not translation.is_file():
            raise FileNotFoundError(f"Translation does not exist: {translation_path}")

        relative_path = translation.relative_to(translations_root)
        source_path = DEVELOPER_SOURCES.get(
            relative_path, ENGLISH_DOCS_ROOT / relative_path
        )
        source = project_root / source_path
        if not source.is_file():
            raise FileNotFoundError(f"English source does not exist: {source_path}")

        content = translation.read_text(encoding="utf-8")
        match, _ = _front_matter(content, translation_path)
        metadata = match.group("metadata")
        source_hash = file_sha256(source)
        # The explicit tag keeps mdformat-frontmatter from dropping the quotes.
        hash_line = f'source_sha256: !!str "{source_hash}"\n'
        if SOURCE_HASH.search(metadata):
            updated_metadata = SOURCE_HASH.sub(hash_line, metadata, count=1)
        else:
            updated_metadata = f"{metadata}{hash_line}"
        translation.write_text(
            content[: match.start("metadata")]
            + updated_metadata
            + content[match.end("metadata") :],
            encoding="utf-8",
        )


def translation_routes(
    project_root: Path = PROJECT_ROOT,
    translations_root: Path = DEFAULT_TRANSLATIONS_ROOT,
) -> dict[str, dict[str, str | bool]]:
    """Build browser routes and currency state for every translated page."""
    routes: dict[str, dict[str, str | bool]] = {}
    for entry in load_translations(project_root, translations_root):
        source = _repository_path(project_root, entry.source.as_posix())
        translation = _repository_path(project_root, entry.translation.as_posix())
        if not source.is_file() or not translation.is_file():
            continue

        relative_path = entry.translation.relative_to(translations_root)
        source_route = _documentation_route(
            ENGLISH_DOCS_ROOT / relative_path, DOCS_ROOT
        )
        translation_route = _documentation_route(entry.translation, DOCS_ROOT)
        routes[source_route] = {
            "translation": translation_route,
            "current": file_sha256(source) == entry.source_sha256,
        }

    return routes


def _documentation_route(path: Path, docs_root: Path) -> str:
    relative_path = path.relative_to(docs_root)
    if relative_path.is_relative_to(Path("en")):
        relative_path = relative_path.relative_to("en")

    if relative_path.name == "index.md":
        route = relative_path.parent
    else:
        route = relative_path.with_suffix("")

    route_text = route.as_posix()
    return "" if route_text == "." else f"{route_text}/"


def main() -> int:
    """Run the translation metadata validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat stale translations as errors.",
    )
    parser.add_argument(
        "--update",
        nargs="+",
        type=Path,
        metavar="TRANSLATION",
        help="Refresh source hashes for the specified reviewed pages.",
    )
    parser.add_argument(
        "--language",
        choices=tuple(TRANSLATION_ROOTS),
        help="Check one language instead of all translations.",
    )
    args = parser.parse_args()

    if args.update:
        update_source_hashes(args.update)

    languages = (args.language,) if args.language else tuple(TRANSLATION_ROOTS)
    failed = False
    for language in languages:
        status = validate_translations(PROJECT_ROOT, TRANSLATION_ROOTS[language])
        for message in status.errors:
            sys.stdout.write(f"ERROR [{language}]: {message}\n")
        for message in status.stale:
            sys.stdout.write(f"WARNING [{language}]: {message}\n")
        if status.errors or (args.strict and status.stale):
            failed = True
    if failed:
        return 1

    sys.stdout.write("Translation metadata is valid.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
