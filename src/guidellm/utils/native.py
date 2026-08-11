"""
Native Rust-accelerated utilities for GuideLLM.

This module re-exports functions from the compiled Rust extension
(guidellm.utils._rust). The Rust extension is a mandatory dependency.
"""

from guidellm.utils._rust import hello_world

__all__ = ["hello_world"]
