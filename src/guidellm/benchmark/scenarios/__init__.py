"""
Builtin benchmark scenario definitions and discovery utilities.

This module provides access to predefined benchmark scenarios stored as JSON files
within the scenarios directory. It enables discovery and retrieval of builtin
scenarios by name or filename, supporting both stem names (without extension) and
full filenames for flexible scenario loading.
"""

from __future__ import annotations

from guidellm.schemas.benchmark.scenarios import SCENARIO_DIR, get_builtin_scenarios

__all__ = ["SCENARIO_DIR", "get_builtin_scenarios"]
