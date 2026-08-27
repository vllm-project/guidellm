"""
Wrapper allow for lazy loading of the torch package.

Use this to ensure torch is only imported when necessary
which can save some time in startup and spawning workers.
"""

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    package="torch",
)
