import uuid
from typing import Any, Literal, Optional

from pydantic import Field

from guidellm.utils import StandardBaseModel

__all__ = ["GenerationRequest"]


class GenerationRequest(StandardBaseModel):
    """
    A class representing a request for generation.
    This class is used to encapsulate the details of a generation request,
    including the request ID, type, content, parameters, statistics, and constraints.
    It is designed to be used with the BackendRequestsWorker class to handle
    the generation process.

    :param request_id: The unique identifier for the request.
    :param request_type: The type of request (e.g., text, chat).
    :param content: The content for the request to send to the backend.
        If request_type is 'text', this should be a string or list of strings
        which will be resolved by backend.text_completions.
        If request_type is 'chat', this should be a string,
        a list of (str, Dict[str, Union[str, Dict[str, str]], Path, Image]),
        or Any raw content which will be resolved by backend.chat_completions.
        If raw content, raw_content=True must be passed in the params.
    :param params: Additional parameters for the request passed in as kwargs.
        For an http backend, these are passed into the body of the request.
    :param stats: Statistics for the request, such as the number of prompt tokens.
        Used for tracking and reporting purposes.
    :param constraints: Constraints for the request, such as the maximum number
        of output tokens. Used for controlling the behavior of the backend.
    """

    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        default="text_completions",
        description=(
            "The type of request (e.g., text, chat). "
            "If request_type='text_completions', resolved by backend.text_completions. "
            "If request_typ='chat_completions', resolved by backend.chat_completions."
        ),
    )
    content: Any = Field(
        description=(
            "The content for the request to send to the backend. "
            "If request_type is 'text', this should be a string or list of strings "
            "which will be resolved by backend.text_completions. "
            "If request_type is 'chat', this should be a string, "
            "a list of (str, Dict[str, Union[str, Dict[str, str]], Path, Image]), "
            "or Any raw content which will be resolved by backend.chat_completions. "
            "If raw content, raw_content=True must be passed in the params."
        )
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters for the request that will be passed in as kwargs. "
            "For an http backend, these are passed into the body of the request. "
        ),
    )
    stats: dict[Literal["prompt_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Statistics for the request, such as the number of prompt tokens. "
            "Used for tracking and reporting purposes."
        ),
    )
    constraints: dict[Literal["output_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Constraints for the request, such as the maximum number of output tokens. "
            "Used for controlling the behavior of the backend."
        ),
    )
