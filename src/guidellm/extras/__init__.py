"""
Code that depends on optional dependencies.

All dependent code should import in one of two ways:

1. import guidellm.extras
2. from guidellm.extras import submodule

As most of the codebase eager imports, importing specific functions or classes may cause
ImportErrors if the optional dependencies are missing. Importing from the module or
submodule level ensures errors are deferred to calling point.
"""

import lazy_loader as lazy

submodules = ["vllm", "vision", "audio"]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    lazy_submodules=True,  # Only import submodules when accessed
)
