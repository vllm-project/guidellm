from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import ScheduledRequestInfo, SchedulerMessagingPydanticRegistry

__all__ = ["scheduler_register_benchmark_objects"]


def scheduler_register_benchmark_objects():
    SchedulerMessagingPydanticRegistry.register("GenerationRequest")(GenerationRequest)
    SchedulerMessagingPydanticRegistry.register("GenerationResponse")(
        GenerationResponse
    )
    SchedulerMessagingPydanticRegistry.register("GenerationRequestTimings")(
        GenerationRequestTimings
    )
    SchedulerMessagingPydanticRegistry.register(
        "ScheduledRequestInfo[GenerationRequestTimings]"
    )(ScheduledRequestInfo[GenerationRequestTimings])
