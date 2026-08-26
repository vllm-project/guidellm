"""
Contains entrypoints for GuideLLM submodules.

Each entrypoint is lazy loaded to avoid unnecessary imports and dependencies.
This is important to ensure a resposive CLI and lightweight worker spawning.
"""

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    "guidellm",
    submod_attrs={
        "benchmark": [
            "GenerativeConsoleBenchmarkerProgress",
            "benchmark_generative_text",
            "reimport_benchmarks_report",
        ],
        "mock_server": ["MockServer"],
        "data": ["process_dataset"],
    },
)
