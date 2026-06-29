"""
Copy required files from outside of the docs directory into the docs directory
for the documentation build and site.
Uses mkdocs-gen-files to handle the file generation and compatibility with MkDocs.
"""

from dataclasses import dataclass
from pathlib import Path

import mkdocs_gen_files


@dataclass
class ProcessFile:
    root_path: Path
    docs_path: Path
    title: str
    weight: float


def find_project_root() -> Path:
    start_path = Path(__file__).absolute()
    current_path = start_path.parent

    while current_path:
        if (current_path / "mkdocs.yml").exists():
            return current_path
        current_path = current_path.parent

    raise FileNotFoundError(
        f"Could not find mkdocs.yml in the directory tree starting from {start_path}"
    )


def process_files(files: list[ProcessFile], project_root: Path):
    for file in files:
        source_path = project_root / file.root_path
        target_path = file.docs_path

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file {source_path} does not exist for copying into docs "
                f"directory at {target_path}"
            )

        frontmatter = f"---\ntitle: {file.title}\nweight: {file.weight}\n---\n\n"
        content = source_path.read_text(encoding="utf-8")

        with mkdocs_gen_files.open(target_path, "w") as file_handle:
            file_handle.write(frontmatter)
            file_handle.write(content)

        mkdocs_gen_files.set_edit_path(target_path, source_path)


def migrate_developer_docs():
    project_root = find_project_root()
    files = [
        ProcessFile(
            root_path=Path("CODE_OF_CONDUCT.md"),
            docs_path=Path("developer/code-of-conduct.md"),
            title="Code of Conduct",
            weight=-10,
        ),
        ProcessFile(
            root_path=Path("CONTRIBUTING.md"),
            docs_path=Path("developer/contributing.md"),
            title="Contributing Guide",
            weight=-8,
        ),
        ProcessFile(
            root_path=Path("DEVELOPING.md"),
            docs_path=Path("developer/developing.md"),
            title="Development Guide",
            weight=-6,
        ),
    ]
    process_files(files, project_root)


migrate_developer_docs()
