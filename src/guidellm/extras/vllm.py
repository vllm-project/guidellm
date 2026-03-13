from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput

    HAS_VLLM = True
else:
    from guidellm.utils import ExtrasImporter

    _vllm_importer = ExtrasImporter(
        {
            "SamplingParams": "vllm.SamplingParams",
            "AsyncEngineArgs": "vllm.engine.arg_utils.AsyncEngineArgs",
            "AsyncLLMEngine": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "RequestOutput": "vllm.outputs.RequestOutput",
        },
        extras_group="vllm",
        eager=False,
    )

    # Make imports available at module level for runtime use
    SamplingParams = _vllm_importer.SamplingParams
    AsyncEngineArgs = _vllm_importer.AsyncEngineArgs
    AsyncLLMEngine = _vllm_importer.AsyncLLMEngine
    RequestOutput = _vllm_importer.RequestOutput
    HAS_VLLM = _vllm_importer.is_available

__all__ = [
    "HAS_VLLM",
    "AsyncEngineArgs",
    "AsyncLLMEngine",
    "RequestOutput",
    "SamplingParams",
]
