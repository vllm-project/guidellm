"""
Unit tests for the registry module.
"""

from __future__ import annotations

from typing import TypeVar

import pytest

from guidellm.utils.registry import RegisterT, RegistryObjT


@pytest.mark.smoke
def test_registry_obj_type():
    """Test that RegistryObjT is configured correctly as a TypeVar."""
    assert isinstance(RegistryObjT, type(TypeVar("test")))
    assert RegistryObjT.__name__ == "RegistryObjT"
    assert RegistryObjT.__bound__ is None
    assert RegistryObjT.__constraints__ == ()


@pytest.mark.smoke
def test_registered_type():
    """Test that RegisterT is configured correctly as a TypeVar."""
    assert isinstance(RegisterT, type(TypeVar("test")))
    assert RegisterT.__name__ == "RegisterT"
    assert RegisterT.__bound__ is type
    assert RegisterT.__constraints__ == ()
