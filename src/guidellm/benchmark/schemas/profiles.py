"""
Profile argument schemas for multi-strategy benchmark execution.

Defines the base argument model for profile configuration, including warmup
and cooldown phase settings. Uses Pydantic class registry for polymorphic
deserialization of profile-specific argument types.
"""

from __future__ import annotations

from guidellm.schemas.benchmark.profiles import ProfileArgs

__all__ = ["ProfileArgs"]
