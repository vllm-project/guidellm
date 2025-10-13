from __future__ import annotations

try:
    import orjson as json
except ImportError:
    import json


__all__ = ["json"]
