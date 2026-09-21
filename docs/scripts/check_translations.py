"""Validate and update metadata for translated documentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = Path("docs/zh/.translation-sources.json")


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


def load_manifest(
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[TranslationSource]:
    """Load translation source mappings from the repository manifest."""
    absolute_manifest = _repository_path(project_root, manifest_path.as_posix())
    data = json.loads(absolute_manifest.read_text(encoding="utf-8"))

    if data.get("version") != 1:
        raise ValueError("Translation manifest version must be 1")

    raw_translations = data.get("translations")
    if not isinstance(raw_translations, list):
        raise ValueError("Translation manifest must contain a translations list")

    translations: list[TranslationSource] = []
    for index, raw_translation in enumerate(raw_translations):
        if not isinstance(raw_translation, dict):
            raise ValueError(f"Translation entry {index} must be an object")

        required_keys = {"source", "translation", "source_sha256"}
        if set(raw_translation) != required_keys:
            raise ValueError(
                f"Translation entry {index} must contain only {sorted(required_keys)}"
            )

        values = [raw_translation[key] for key in sorted(required_keys)]
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"Translation entry {index} values must be strings")

        translations.append(
            TranslationSource(
                source=Path(raw_translation["source"]),
                translation=Path(raw_translation["translation"]),
                source_sha256=raw_translation["source_sha256"],
            )
        )

    return translations


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
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> TranslationStatus:
    """Validate translation mappings, source revisions, and code examples."""
    try:
        translations = load_manifest(project_root, manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return TranslationStatus(errors=(str(error),), stale=())

    errors: list[str] = []
    stale: list[str] = []
    seen_sources: set[Path] = set()
    seen_translations: set[Path] = set()
    for entry in translations:
        if entry.source in seen_sources:
            errors.append(f"Duplicate English source: {entry.source}")
        if entry.translation in seen_translations:
            errors.append(f"Duplicate translation: {entry.translation}")
        seen_sources.add(entry.source)
        seen_translations.add(entry.translation)

        entry_status = _validate_entry(project_root, entry)
        errors.extend(entry_status.errors)
        stale.extend(entry_status.stale)

    errors.extend(_find_unregistered_pages(project_root, seen_translations))

    return TranslationStatus(errors=tuple(errors), stale=tuple(stale))


def _validate_entry(project_root: Path, entry: TranslationSource) -> TranslationStatus:
    errors: list[str] = []
    stale: list[str] = []
    docs_root = (project_root / "docs").resolve()
    translations_root = (docs_root / "zh").resolve()

    try:
        source = _repository_path(project_root, entry.source.as_posix())
        translation = _repository_path(project_root, entry.translation.as_posix())
    except ValueError as error:
        return TranslationStatus(errors=(str(error),), stale=())

    if not source.is_relative_to(docs_root) or source.is_relative_to(translations_root):
        errors.append(f"English source must be under docs/: {entry.source}")
    if not translation.is_relative_to(translations_root):
        errors.append(f"Translation must be under docs/zh/: {entry.translation}")
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


def _find_unregistered_pages(
    project_root: Path, registered_paths: set[Path]
) -> tuple[str, ...]:
    translations_root = (project_root / "docs/zh").resolve()
    registered = {path.as_posix() for path in registered_paths}
    errors: list[str] = []

    if not translations_root.is_dir():
        return ()

    for path in sorted(translations_root.rglob("*.md")):
        relative_path = path.relative_to(project_root).as_posix()
        if relative_path not in registered:
            errors.append(f"Unregistered translated page: {relative_path}")

    return tuple(errors)


def update_source_hashes(
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    """Refresh recorded source hashes after translations have been reviewed."""
    absolute_manifest = _repository_path(project_root, manifest_path.as_posix())
    data = json.loads(absolute_manifest.read_text(encoding="utf-8"))
    raw_translations = data.get("translations")
    if not isinstance(raw_translations, list):
        raise ValueError("Translation manifest must contain a translations list")

    for raw_translation in raw_translations:
        if not isinstance(raw_translation, dict):
            raise ValueError("Translation entries must be objects")
        source_value = raw_translation.get("source")
        if not isinstance(source_value, str):
            raise ValueError("Translation source must be a string")
        source = _repository_path(project_root, source_value)
        if not source.is_file():
            raise FileNotFoundError(f"English source does not exist: {source_value}")
        raw_translation["source_sha256"] = file_sha256(source)

    absolute_manifest.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def translation_routes(
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, dict[str, str | bool]]:
    """Build browser routes and currency state for every translated page."""
    routes: dict[str, dict[str, str | bool]] = {}
    for entry in load_manifest(project_root, manifest_path):
        source = _repository_path(project_root, entry.source.as_posix())
        translation = _repository_path(project_root, entry.translation.as_posix())
        if not source.is_file() or not translation.is_file():
            continue

        source_route = _documentation_route(entry.source, Path("docs"))
        translation_route = _documentation_route(entry.translation, Path("docs"))
        routes[source_route] = {
            "translation": translation_route,
            "current": file_sha256(source) == entry.source_sha256,
        }

    return routes


def _documentation_route(path: Path, docs_root: Path) -> str:
    relative_path = path.relative_to(docs_root)
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
        action="store_true",
        help="Refresh source hashes after reviewing translated pages.",
    )
    args = parser.parse_args()

    if args.update:
        update_source_hashes()

    status = validate_translations()
    for message in status.errors:
        sys.stdout.write(f"ERROR: {message}\n")
    for message in status.stale:
        sys.stdout.write(f"WARNING: {message}\n")

    if status.errors or (args.strict and status.stale):
        return 1

    sys.stdout.write("Translation metadata is valid.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
