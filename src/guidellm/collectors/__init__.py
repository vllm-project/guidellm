"""
Collectors gather data from outside GuideLLM while each benchmark runs.

Configure them with ``--collector kind=...``. Each collector reports one result
per benchmark, stored in that benchmark's ``collectors`` list in the order the
collectors were configured.
"""

from guidellm.collectors.collector import (
    Collector,
    OneshotCollector,
    PeriodicCollector,
)
from guidellm.collectors.process import CollectorProcess
from guidellm.collectors.prometheus import (
    PrometheusCollector,
    PrometheusSnapshot,
    histogram_quantile,
    parse_prometheus_text,
    summarize_snapshots,
)

__all__ = [
    "Collector",
    "CollectorProcess",
    "OneshotCollector",
    "PeriodicCollector",
    "PrometheusCollector",
    "PrometheusSnapshot",
    "histogram_quantile",
    "parse_prometheus_text",
    "summarize_snapshots",
]

# Import plugin modules that mirror this package, e.g. `<plugin>.collectors`.
# Kept last so plugins can import anything this package defines.
from guidellm.utils.plugins import load_plugin_submodules

load_plugin_submodules(__name__)
