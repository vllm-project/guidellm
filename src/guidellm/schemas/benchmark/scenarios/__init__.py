"""
Builtin benchmark scenario definitions and discovery utilities.

This module provides access to predefined benchmark scenarios stored as JSON files
within the scenarios directory. It enables discovery and retrieval of builtin
scenarios by name or filename, supporting both stem names (without extension) and
full filenames for flexible scenario loading.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Annotated

__all__ = ["SCENARIO_DIR", "get_builtin_scenarios"]

SCENARIO_DIR: Annotated[
    Path,
    "Directory path containing builtin scenario JSON files",
] = Path(__file__).parent


@cache
def get_builtin_scenarios() -> dict[str, Path]:
    """
    Retrieve all builtin scenario definitions from the scenarios directory.

    Scans the scenarios directory for JSON files and returns a mapping of scenario
    names to their file paths. Each scenario is indexed by both its stem name
    (filename without extension) and full filename for convenient lookup.

    :return: Dictionary mapping scenario names and filenames to their Path objects
    """
    builtin = {}
    for path in SCENARIO_DIR.glob("*.json"):
        builtin[path.stem] = path
        builtin[path.name] = path

    return builtin
