from __future__ import annotations

try:
    import orjson as json
except ImportError:
    import json  # type: ignore[no-redef] # Done only after a failure.


__all__ = ["json"]
