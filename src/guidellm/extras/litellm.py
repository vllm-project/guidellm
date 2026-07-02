"""
LiteLLM wrapper with same interface as LiteLLM.
"""

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    package="litellm",
    error_message=(
        "Please install litellm to use LiteLLM features: "
        "pip install 'guidellm[litellm]'"
    ),
)
