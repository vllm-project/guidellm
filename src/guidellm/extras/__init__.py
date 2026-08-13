"""
Code that depends on optional dependencies.

All dependent code should import in one of two ways:

1. import guidellm.extras
2. from guidellm.extras import submodule

As most of the codebase eager imports, importing specific functions or classes may cause
ImportErrors if the optional dependencies are missing. Importing from the module or
submodule level ensures errors are deferred to calling point.

CRITICAL: Import Pattern for Lazy-Loaded Dependencies
======================================================

When importing from extras modules, use module imports to preserve lazy loading:

CORRECT:
    import guidellm.extras.audio as libs
    decoder = libs.AudioDecoder(...)

WRONG:
    from guidellm.extras.audio import AudioDecoder
    decoder = AudioDecoder(...)

The from-import bypasses lazy loading and fails immediately if dependencies are missing.
Module imports defer errors until attribute access, allowing graceful error messages.

Architecture: utils.audio/vision contain implementations; extras.audio/vision export
only external library classes (torchcodec, PIL). Implementations use module imports.
"""

import guidellm.utils.lazy_loader as lazy

submodules = ["vllm", "vision", "audio", "plot"]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    lazy_submodules=True,  # Only import submodules when accessed
)
