"""Shared prefetch argument type for profiles and scheduling strategies."""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt

__all__ = ["PrefetchType"]


PrefetchType = NonNegativeInt | Literal["start", "all"]
