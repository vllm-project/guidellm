try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput

    HAS_VLLM = True
except ImportError:
    AsyncLLMEngine = None  # type: ignore[assignment, misc]
    AsyncEngineArgs = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]
    RequestOutput = None  # type: ignore[assignment, misc]
    HAS_VLLM = False
