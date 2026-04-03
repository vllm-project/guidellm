"""
vLLM wrapper with same interface as vLLM.
"""

try:
    import vllm
except ImportError as e:
    raise AttributeError("Please install vllm to use vLLM features") from e


def __getattr__(name: str):
    return getattr(vllm, name)


__all__ = vllm.__all__
