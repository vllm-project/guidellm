"""
vLLM wrapper with same interface as vLLM.
"""

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    package="vllm",
    error_message="Please install vllm to use vLLM features",
)
