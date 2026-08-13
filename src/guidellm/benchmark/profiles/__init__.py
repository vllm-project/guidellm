"""
Orchestrate multi-strategy benchmark execution through configurable profiles.

Provides abstractions for coordinating sequential execution of scheduling strategies
during benchmarking workflows. Profiles automatically generate strategies based on
configuration parameters, manage runtime constraints, and track completion state
across execution sequences. Each profile type implements a specific execution pattern
(synchronous, concurrent, throughput-focused, rate-based async, or adaptive sweep)
that determines how benchmark requests are scheduled and executed.
"""

from __future__ import annotations

from .asynchronous import AsyncProfile, AsyncProfileArgs
from .concurrent import ConcurrentProfile, ConcurrentProfileArgs
from .profile import Profile, ProfileFactory
from .replay import ReplayProfile, ReplayProfileArgs
from .sweep import SweepProfile, SweepProfileArgs
from .synchronous import SynchronousProfile, SynchronousProfileArgs
from .throughput import ThroughputProfile, ThroughputProfileArgs

__all__ = [
    "AsyncProfile",
    "AsyncProfileArgs",
    "ConcurrentProfile",
    "ConcurrentProfileArgs",
    "Profile",
    "ProfileFactory",
    "ReplayProfile",
    "ReplayProfileArgs",
    "SweepProfile",
    "SweepProfileArgs",
    "SynchronousProfile",
    "SynchronousProfileArgs",
    "ThroughputProfile",
    "ThroughputProfileArgs",
]
